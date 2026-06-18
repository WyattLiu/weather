"""Execution advisor — turns the intraday microstructure stats + the live spread into
a CONCRETE execution plan for each order the engine wants to place:

  • WHICH MINUTE to work it  — the tightest day×hour cell + tightest sub-minute window,
                               with the Thursday EIA-print (10:30 ET) caveat.
  • HOW TO LADDER it         — a limit-price ladder from mid → touch, timed across the
                               working window, with rung count/aggressiveness keyed to
                               the live spread width and DTE (the engine's P(mid) model).

All RTH-only (09:30–16:00). Surfaced in the dashboard so the operator sees, per order,
"post $X at 15:05, step to $Y by 15:15, cross to $Z (bid) by 15:25".
"""
import os
import json
import psycopg2

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
GRID_CACHE = os.path.join(CACHE, 'microstructure_grid.json')
# postgres extract(dow): 0=Sun..6=Sat → trading days Mon(1)..Fri(5)
DOW = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri'}
RTH_OPEN, RTH_CLOSE = '09:30:00', '16:00:00'
_SPLITS = [('2018-01-05', 4.0), ('2024-01-24', 4.0)]


def _conn():
    return psycopg2.connect(**DB)


def _p_mid_fill(rel_spread, dte):
    """Fallback formula (used only when the empirical bucket is unseen)."""
    rel = rel_spread / 100.0 if rel_spread > 1 else rel_spread
    return max(0.0, min(1.0, (1 - min(1.0, rel / 0.20)) *
                        (0.6 + 0.4 * min(1.0, (dte or 30) / 45.0))))


_CALIB = None


def _load_calib():
    """Empirical mid-fill calibration (spread-pennies × OI), fit from the real minute
    path — see fill_quality.py / cache/fill_quality_calibration.json."""
    global _CALIB
    if _CALIB is None:
        try:
            _CALIB = json.load(open(os.path.join(CACHE, 'fill_quality_calibration.json')))
        except Exception:
            _CALIB = {}
    return _CALIB


def _bucket(buckets, v):
    for lo, hi in buckets:
        if lo <= v < hi:
            return f"{lo}-{'+' if hi > 1e8 else int(hi)}"
    return None


def p_mid_empirical(spread_cents, oi):
    """Data-calibrated P(mid-fill) from spread-in-pennies × OI. Returns (p, source, n);
    p is None if the bucket was never observed (caller falls back to the formula)."""
    cal = _load_calib()
    if cal and cal.get('grid'):
        sb = _bucket([tuple(x) for x in cal['spread_buckets_cents']], spread_cents)
        ob = _bucket([tuple(x) for x in cal['oi_buckets']], oi or 0)
        g = cal['grid'].get(f"{sb}|{ob}")
        if g and g.get('p_mid') is not None and g.get('n', 0) >= 30:
            return float(g['p_mid']), 'empirical', int(g['n'])
    return None, None, None


def load_grid(refresh=False):
    """day×hour + minute-of-day median spread grid (near-money, two-sided, RTH).
    Cached to JSON; refresh re-mines from PG (cheap server-side aggregation)."""
    if not refresh and os.path.exists(GRID_CACHE):
        try:
            return json.load(open(GRID_CACHE))
        except Exception:
            pass
    cur = _conn().cursor()
    base = """
      WITH q AS (
        SELECT bar_time, strike, option_right, underlying_price,
               MAX(CASE WHEN data_type='BID' THEN close END) bid,
               MAX(CASE WHEN data_type='ASK' THEN close END) ask
        FROM ung_options_history
        WHERE bar_time::time >= '09:30' AND bar_time::time <= '16:00'
        GROUP BY bar_time, strike, option_right, underlying_price)
      SELECT {dims},
             percentile_cont(0.5) WITHIN GROUP (ORDER BY (ask-bid)/((ask+bid)/2)*100) med,
             count(*) n
      FROM q
      WHERE bid > 0 AND ask > bid
        AND abs(strike-underlying_price)/NULLIF(underlying_price,0) <= 0.10
      GROUP BY {grp} ORDER BY {grp}"""
    # day × hour
    cur.execute(base.format(dims="extract(dow from bar_time)::int, extract(hour from bar_time)::int",
                            grp="1,2"))
    dh = {f"{int(d)}_{int(h)}": {'spread': round(float(m), 1), 'n': int(n)}
          for d, h, m, n in cur.fetchall() if m is not None}
    # minute-of-day (HH:MM) profile — for "which minute"
    cur.execute(base.format(dims="to_char(bar_time,'HH24:MI')", grp="1"))
    mod = {hhmm: {'spread': round(float(m), 1), 'n': int(n)}
           for hhmm, m, n in cur.fetchall() if m is not None}
    # thursday print effect
    cur.execute(base.format(
        dims="CASE WHEN extract(hour from bar_time)<11 THEN 'pre' ELSE 'post' END",
        grp="1").replace("WHERE bid > 0", "WHERE extract(dow from bar_time)=4 AND bid > 0"))
    thu = {k: round(float(m), 1) for k, m, n in cur.fetchall() if m is not None}
    grid = {'day_hour': dh, 'minute': mod, 'thursday': thu}
    try:
        json.dump(grid, open(GRID_CACHE, 'w'), indent=0)
    except Exception:
        pass
    return grid


def best_window(grid, dow=None):
    """Tightest hours overall, and the tightest sub-minute window within the best hour."""
    dh = grid.get('day_hour', {})
    # rank (dow,hour) cells by spread (need a real sample)
    cells = [(k, v['spread']) for k, v in dh.items() if v['n'] > 50]
    cells.sort(key=lambda x: x[1])
    best_cells = [{'when': f"{DOW.get(int(k.split('_')[0]), k.split('_')[0])} "
                           f"{int(k.split('_')[1]):02d}:00", 'spread': s}
                  for k, s in cells[:3]]
    # best hour overall (by median across days) → tightest 10-min sub-window
    by_hour = {}
    for k, v in dh.items():
        h = int(k.split('_')[1])
        by_hour.setdefault(h, []).append(v['spread'])
    hour_med = {h: sum(xs) / len(xs) for h, xs in by_hour.items()}
    best_hour = min(hour_med, key=hour_med.get) if hour_med else 15
    mod = grid.get('minute', {})
    sub = sorted(((hhmm, v['spread']) for hhmm, v in mod.items()
                  if hhmm.startswith(f"{best_hour:02d}:") and v['n'] > 20),
                 key=lambda x: x[1])
    tight_minute = sub[0][0] if sub else f"{best_hour:02d}:00"
    return {'tightest_cells': best_cells, 'best_hour': best_hour,
            'tightest_minute': tight_minute,
            'tightest_minute_spread': sub[0][1] if sub else None}


def latest_spread(K_adj, dte, right, date=None, expiry=None):
    """Most recent two-sided RTH quote for the contract → live bid/ask/mid/spread.
    Uses the order's exact `expiry` when available (falls back to nearest-DTE).
    Returns None if no quote."""
    import pandas as pd
    cur = _conn().cursor()
    # map adjusted strike → raw, find nearest expiry to target DTE on the latest day
    cur.execute("SELECT max(trade_date) FROM ung_options_history")
    last = cur.fetchone()[0]
    if last is None:
        return None
    d = pd.Timestamp(date or last)
    K_raw = float(K_adj)
    for sd, f in _SPLITS:
        if d < pd.Timestamp(sd):
            K_raw /= f
    cur.execute("""SELECT DISTINCT expiration FROM ung_options_history
                   WHERE trade_date=%s AND option_right=%s""", (last, right))
    exps = sorted(r[0] for r in cur.fetchall())
    if not exps:
        return None
    if expiry:   # prefer the order's exact expiry if we have quotes for it
        try:
            want = pd.Timestamp(expiry).date()
            exp = min(exps, key=lambda e: abs((e - want).days))
            if abs((exp - want).days) > 7:           # too far → fall back to DTE
                raise ValueError
        except Exception:
            expiry = None
    if not expiry:
        target = (pd.Timestamp(last) + pd.Timedelta(days=int(dte or 30))).date()
        exp = min(exps, key=lambda e: abs((e - target).days))
    cur.execute("""
        SELECT MAX(CASE WHEN data_type='BID' THEN close END) bid,
               MAX(CASE WHEN data_type='ASK' THEN close END) ask
        FROM ung_options_history
        WHERE trade_date=%s AND expiration=%s AND ABS(strike-%s)<0.001 AND option_right=%s
          AND bar_time::time>=%s AND bar_time::time<=%s
        GROUP BY bar_time ORDER BY bar_time DESC LIMIT 1""",
        (last, exp.isoformat(), round(K_raw, 1), right, RTH_OPEN, RTH_CLOSE))
    r = cur.fetchone()
    if not r or r[0] is None or r[1] is None:
        return None
    bid, ask = float(r[0]), float(r[1])
    if bid <= 0 or ask <= bid:
        return None
    sf = 1.0
    for sd, f in _SPLITS:
        if d < pd.Timestamp(sd):
            sf = f
    bid, ask = bid * sf, ask * sf
    mid = (bid + ask) / 2
    # daily OI for this exact contract (raw strike) — drives the empirical fill model
    cur.execute("""SELECT open_interest FROM ung_options_oi
                   WHERE trade_date=%s AND expiration=%s AND ABS(strike-%s)<0.001
                     AND option_right=%s ORDER BY trade_date DESC LIMIT 1""",
                (last, exp.isoformat(), round(K_raw, 1), right))
    _oi = cur.fetchone()
    # STALENESS: the quote is from `last` (max trade_date in the minute table). If the
    # options feed has lagged the real today, this 'live' quote is actually N days old —
    # the caller MUST flag it so a stale price is never presented as the current market.
    _qdate = pd.Timestamp(last).normalize()
    _stale = max(0, (pd.Timestamp.today().normalize() - _qdate).days)
    return {'bid': round(bid, 3), 'ask': round(ask, 3), 'mid': round(mid, 3),
            'spread_pct': round((ask - bid) / mid * 100, 1),
            'spread_cents': round((ask - bid) * 100, 1),
            'oi': int(_oi[0]) if _oi and _oi[0] is not None else None,
            'expiry': exp.isoformat(),
            'asof': str(_qdate.date()), 'stale_days': int(_stale)}


def build_ladder(mid, bid, ask, side, dte, spread_pct, oi=None, window=30):
    """Limit-price ladder mid → touch, timed across `window` minutes. Rung count and
    aggressiveness scale with spread width; dwell-at-mid scales with the EMPIRICAL
    P(mid-fill) (spread-pennies × OI, fit from the real minute path)."""
    half = (ask - bid) / 2.0
    cents = (ask - bid) * 100.0
    p, p_src, p_n = p_mid_empirical(cents, oi)
    if p is None:                              # unseen bucket → fallback formula
        p, p_src, p_n = _p_mid_fill(spread_pct, dte), 'formula', None
    if spread_pct < 10:
        fracs = [0.0, 0.34, 0.67]            # tight: insist near mid
    elif spread_pct < 20:
        fracs = [0.0, 0.25, 0.5, 0.75]
    else:
        fracs = [0.0, 0.2, 0.4, 0.6, 0.8]    # wide: plan to give more
    n = len(fracs)
    first = round(window * (0.4 + 0.4 * p))  # longer at mid when fill-at-mid likely
    step = (window - first) / (n - 1) if n > 1 else 0
    mins = [0] + [round(first + step * (i - 1)) for i in range(1, n)]
    sgn = -1 if side == 'sell' else 1        # sell: descend to bid; buy: ascend to ask
    rungs = []
    for i, (f, m) in enumerate(zip(fracs, mins)):
        px = round(mid + sgn * f * half, 2)
        lbl = 'mid' if f == 0 else ('bid (touch)' if (side == 'sell' and i == n - 1)
                                    else 'ask (touch)' if i == n - 1 else f'{int(f*100)}% to touch')
        rungs.append({'t_plus_min': m, 'limit': px, 'rung': lbl})
    # expected fill ≈ engine give: mid minus (1-p)*half on a sell (plus on a buy)
    exp_fill = round(mid + sgn * (1 - p) * 0.5 * half, 3)
    return {'rungs': rungs, 'p_mid': round(p, 2), 'p_mid_source': p_src, 'p_mid_n': p_n,
            'spread_cents': round(cents, 1), 'oi': oi,
            'expected_fill': exp_fill, 'work_window_min': window}


def execution_plan(K, dte, right, side, spot=None, date=None, grid=None, expiry=None):
    """Full per-order plan: timing window + live spread + limit ladder + caveats.
    right in {'P','C'}; side in {'sell','buy'}."""
    grid = grid or load_grid()
    win = best_window(grid)
    q = latest_spread(K, dte, right, date=date, expiry=expiry)
    import datetime as _dt
    today_dow = (_dt.date.fromisoformat(date) if date else _dt.date.today()).weekday()  # Mon=0
    is_thu = today_dow == 3
    caveats = []
    if is_thu:
        thu = grid.get('thursday', {})
        caveats.append(f"Thursday EIA storage print 10:30 ET — spreads {thu.get('pre','~34')}% "
                       f"pre vs {thu.get('post','~10')}% post. Do NOT work before 11:00.")
    plan = {'timing': {
                'recommended': f"work near {win['tightest_minute']} ET "
                               f"({win['tightest_minute_spread']}% median spread)",
                'tightest_windows': win['tightest_cells'],
                'best_hour': f"{win['best_hour']:02d}:00",
                'avoid': "09:30 open (wide/auction) & Thursday pre-11:00 (EIA print)"},
            'caveats': caveats}
    if q:
        plan['live_quote'] = q
        lad = build_ladder(q['mid'], q['bid'], q['ask'], side, dte, q['spread_pct'],
                           oi=q.get('oi'))
        # anchor ladder to real clock times starting at the tightest minute,
        # clamped so the full work window completes by the 16:00 close
        start = win['tightest_minute']
        sh, sm = int(start[:2]), int(start[3:5])
        latest = 16 * 60 - lad['work_window_min']
        if sh * 60 + sm > latest:
            sh, sm = latest // 60, latest % 60
            start = f"{sh:02d}:{sm:02d}"
        for rg in lad['rungs']:
            tot = sh * 60 + sm + rg['t_plus_min']
            tot = min(tot, 16 * 60)          # never past 16:00 close
            rg['clock'] = f"{tot//60:02d}:{tot%60:02d}"
        plan['ladder'] = lad
        # ---- plain-English lines the human reads & types into IBKR/WS ----
        act = 'SELL-TO-OPEN' if side == 'sell' else 'BUY-TO-CLOSE'
        rname = 'PUT' if right == 'P' else 'CALL'
        lines = [
            f"{act}  UNG ${K:.2f} {rname}  exp {q['expiry']} ({dte}d)",
            f"Live: bid ${q['bid']:.2f} / ask ${q['ask']:.2f} / mid ${q['mid']:.2f}"
            f"  (spread {q['spread_pct']}% / {lad['spread_cents']:.0f}¢, "
            f"OI {q.get('oi') if q.get('oi') is not None else '?'}, "
            f"P(mid)={lad['p_mid']} {lad['p_mid_source']})",
            f"Window: work near {start} ET (~{win['tightest_minute_spread']}% spread). "
            f"Avoid 09:30 open & Thu pre-11:00.",
            "Ladder (patient → cross):"]
        for rg in lad['rungs']:
            lines.append(f"  {rg['clock']}  limit ${rg['limit']:.2f}   [{rg['rung']}]")
        lines.append(f"Expected fill ≈ ${lad['expected_fill']:.2f}")
        if caveats:
            lines.append("⚠ " + caveats[0])
        plan['human_lines'] = lines
    else:
        plan['live_quote'] = None
        plan['note'] = 'no recent two-sided quote for this contract; price off the chain mid'
        plan['human_lines'] = [
            f"{'SELL-TO-OPEN' if side=='sell' else 'BUY-TO-CLOSE'} UNG ${K:.2f} "
            f"{'PUT' if right=='P' else 'CALL'} ({dte}d) — no live two-sided quote; "
            f"work near {win['tightest_minute']} ET, post at chain mid and step toward touch."]
    return plan


_RIGHT = {'PUT': 'P', 'CALL': 'C', 'P': 'P', 'C': 'C'}


def plan_for_recs(recs, spot=None):
    """Annotate live engine recommendations with execution plans (options only)."""
    grid = load_grid()
    out = []
    for r in recs or []:
        rt = _RIGHT.get((r.get('right') or '').upper())
        K, dte = r.get('strike'), r.get('dte')
        if not rt or not K:
            out.append({**r, 'exec_plan': None})
            continue
        side = 'sell' if (r.get('side') or '').upper().startswith('SELL') else 'buy'
        try:
            plan = execution_plan(K, dte, rt, side, spot, grid=grid, expiry=r.get('expiry'))
            rr = {**r, 'exec_plan': plan}
            _reconcile_economics(rr, plan)   # model price vs REAL quote — accuracy guard
            out.append(rr)
        except Exception as e:
            out.append({**r, 'exec_plan': {'error': repr(e)[:120]}})
    return out


def _reconcile_economics(rec, plan):
    """ACCURACY GUARD: the engine prices orders off its model/cached chain; the operator
    fills at the REAL market. When a live two-sided quote exists, recompute the order's
    economics at the real executable price and FLAG material divergence so a stale-model
    'take-profit' is never executed blind. Mutates rec in place (adds rec['reconcile'])."""
    q = (plan or {}).get('live_quote')
    lad = (plan or {}).get('ladder') or {}
    if not q or 'mid' not in q:
        return
    qty = int(abs(rec.get('qty') or 0)) or 1
    ty = rec.get('type') or ''
    real_fill = lad.get('expected_fill', q['mid'])
    qstale = int(q.get('stale_days') or 0)
    rc = {'real_bid': q['bid'], 'real_ask': q['ask'], 'real_mid': q['mid'],
          'real_fill': round(real_fill, 2), 'quote_asof': q.get('asof'),
          'quote_stale_days': qstale}
    if qstale >= 1:
        # The quote is NOT today's market — the options feed lags. Say so loudly; do not
        # present a stale price as a live fill (root-caused 2026-06-18: feed frozen 6d).
        rc['stale_warning'] = (f"⚠ quote is {qstale}d STALE (as of {q.get('asof')}) — the UNG "
                               f"options feed (ThetaData→PG) has lagged; this is NOT today's "
                               f"market. Refresh the feed before trusting this price.")
    # Take-profit buy-to-close: profit = (entry premium − buyback)·100·qty. Recompute the
    # buyback at the real price; back out the entry premium from the model figures.
    if ty in ('PUT_TP', 'CALL_TP'):
        mbb = rec.get('model_buyback')
        mpnl = rec.get('pnl')
        if mbb is not None and mpnl is not None:
            entry_prem = mbb + (mpnl / (100.0 * qty))         # implied by the engine
            real_pnl = (entry_prem - real_fill) * 100.0 * qty
            rc.update(model_buyback=mbb, model_pnl=round(mpnl, 0),
                      real_buyback=round(real_fill, 2), real_pnl=round(real_pnl, 0),
                      entry_prem=round(entry_prem, 2))
            # ACCURACY ONLY — never advise deviating from the engine's decision. The engine
            # emitted this order, so the backtest executed it; the operator must too (live =
            # backtest). When the real fill differs materially from the model, report the REAL
            # price so the order is FILLED accurately — not to suggest skipping or waiting.
            if abs(real_fill - mbb) >= 0.03:
                rc['flag'] = (f"EXECUTE at the real market: fill ≈${real_fill:.2f} "
                              f"(bid ${q['bid']:.2f}/ask ${q['ask']:.2f}); engine modeled ${mbb:.2f}. "
                              f"Realistic P&L +${real_pnl:,.0f}. This is a validated decision — fill it "
                              f"at the real price via the ladder; only the price is updated, not the call.")
    rec['reconcile'] = rc
    # propagate a one-line caveat into the exec plan so the UI shows it inline
    if rc.get('flag'):
        plan.setdefault('caveats', []).insert(0, rc['flag'])


if __name__ == '__main__':
    g = load_grid(refresh=True)
    print("Best windows:", json.dumps(best_window(g), indent=2))
    print("\nThursday print:", g.get('thursday'))
    for side in ('sell', 'buy'):
        print(f"\n=== plan: {side} UNG $11 PUT 30DTE ===")
        print(json.dumps(execution_plan(11.0, 30, 'P', side, grid=g), indent=2))
