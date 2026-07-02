"""Intraday fill model — answers "what price would this actually fill at, and could
it fill within ~30 min?" using the real intraday bid/ask PATH from PG
(market_scanner.ung_options_history), not a single EOD snapshot.

A patient hand-trader works a limit near mid and fills at the GOOD moment — when the
spread is tightest during the window — rather than crossing the wide EOD touch. This
module models that: the achievable fill is between mid and the touch, scaled by the
*intraday* spread, and we take the best executable price across the working window.

Used to (a) re-price the daily strategy's trades at realistic execution and (b) measure
execution stability (fill rate, slippage-vs-mid, how often you can't fill near mid).
"""
import psycopg2

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
_CONN = None


def _conn():
    global _CONN
    if _CONN is None or _CONN.closed:
        _CONN = psycopg2.connect(**DB)
    return _CONN


# Regular trading hours (US equity options): 09:30–16:00 ET. NO fill may be modeled
# outside this window — enforced in SQL (defence-in-depth with the backfill filter).
RTH_OPEN = '09:30:00'
RTH_CLOSE = '16:00:00'
TIGHT_SPREAD = 0.06     # arrival-spread threshold: ≤6% → take the tight window; > → patient mid


def _bars(trade_date, expiration, strike_raw, right):
    """Intraday (bar_time, bid, ask) for a contract on a day, BID/ASK pivoted.
    RTH-ONLY: bars outside 09:30–16:00 are excluded at the source."""
    cur = _conn().cursor()
    cur.execute("""
        SELECT bar_time,
               MAX(CASE WHEN data_type='BID' THEN close END) AS bid,
               MAX(CASE WHEN data_type='ASK' THEN close END) AS ask
        FROM ung_options_history
        WHERE trade_date=%s AND expiration=%s
          AND ABS(strike-%s) < 0.001 AND option_right=%s
          AND bar_time::time >= %s AND bar_time::time <= %s
        GROUP BY bar_time ORDER BY bar_time""",
                (trade_date, expiration, float(strike_raw), right, RTH_OPEN, RTH_CLOSE))
    return [(t, float(b) if b is not None else None,
             float(a) if a is not None else None) for t, b, a in cur.fetchall()]


def fill(trade_date, expiration, strike_raw, right, side,
         ref_time=None, window_min=30, patience=0.5):
    """Model a realistic fill. side='sell' (open short / sell) or 'buy' (buy to close).
    ref_time: HH:MM:SS to start the working window (None = whole RTH session).
    patience ∈ [0,1]: 0 = cross to touch immediately, 1 = insist on mid.
    Returns dict with filled, price, mid, vs_mid (slippage), spread_pct, fill_time."""
    rows = _bars(trade_date, expiration, strike_raw, right)
    quotes = [(t, b, a) for (t, b, a) in rows if b and a and a > b and b > 0]
    if not quotes:
        return {'filled': False, 'reason': 'no two-sided quote', 'n_quotes': 0}
    if ref_time:
        import datetime as _dt
        lo = _dt.datetime.strptime(ref_time, '%H:%M:%S').time()
        win = [q for q in quotes if q[0].time() >= lo][:max(1, window_min // 60 + 1)]
        quotes = win or quotes[-1:]
    best = None
    for (t, bid, ask) in quotes:
        mid = (bid + ask) / 2.0
        half = (ask - bid) / 2.0
        rel = (ask - bid) / mid if mid > 0 else 1.0
        # concession from mid toward the touch: tighter spread + more patience → smaller.
        cross = (1.0 - patience) * min(1.0, rel / 0.30)        # 0 (tight/patient) .. 1 (wide/eager)
        px = (mid - cross * half) if side == 'sell' else (mid + cross * half)
        better = best is None or (px > best['price'] if side == 'sell' else px < best['price'])
        if better:
            best = {'price': round(px, 4), 'mid': round(mid, 4),
                    'spread_pct': round(rel * 100, 1), 'fill_time': str(t.time())}
    best['vs_mid'] = round(best['price'] - best['mid'], 4)
    best['filled'] = True
    best['n_quotes'] = len(quotes)
    return best


_SPLITS = [('2018-01-05', 4.0), ('2024-01-24', 4.0)]  # match real_chain
_EXP_CACHE = {}


def _expiries_on(trade_date):
    if trade_date in _EXP_CACHE:
        return _EXP_CACHE[trade_date]
    cur = _conn().cursor()
    cur.execute("SELECT DISTINCT expiration FROM ung_options_history WHERE trade_date=%s",
                (trade_date,))
    out = sorted(r[0] for r in cur.fetchall())
    _EXP_CACHE[trade_date] = out
    return out


def intraday_fill_price(date, K_adj, dte, right, spot, side, patience=0.6):
    """Engine adapter: realistic intraday fill price for an (adjusted-strike, DTE)
    order, from the PG intraday path. Returns price or None (caller falls back).
    right in {'P','C'}; side 'sell'/'buy'."""
    import pandas as pd
    d = pd.Timestamp(date).normalize()
    ds = d.date().isoformat()
    exps = _expiries_on(ds)
    if not exps:
        return None
    target = (d + pd.Timedelta(days=int(dte))).date()
    exp = min(exps, key=lambda e: abs((e - target).days))
    if abs((exp - target).days) > 12:           # no expiry near the target DTE
        return None
    # adjusted strike → raw (ThetaData scale)
    K_raw = float(K_adj)
    for sd, f in _SPLITS:
        if d < pd.Timestamp(sd):
            K_raw /= f
    r = fill(ds, exp.isoformat(), round(K_raw, 1), right, side, patience=patience)
    if not r.get('filled'):
        return None
    # scale price back to adjusted dollars if pre-split
    px = r['price']
    for sd, f in _SPLITS:
        if d < pd.Timestamp(sd):
            px *= f
    return px, r.get('spread_pct'), r.get('vs_mid')


def execute_audit(date, K_adj, dte, right, side, exec_window=15,
                  avoid_print=True, patience=None):
    """VERIFIABLE execution — replays the ACTUAL minute bid/ask path; no P(mid) model.

    Policy (a patient hand-trader working a limit, then crossing):
      1. Post a passive limit at the MID of the first bar in the working window.
      2. Walk the real minute path: if the market actually trades THROUGH that mid
         (best bid ≥ mid for a sell / best ask ≤ mid for a buy), it FILLS AT MID at
         that real minute — proven by the data, how='passive_mid'.
      3. If it never trades through, you CROSS at the tightest-spread minute in the
         window and fill at the REAL TOUCH (bid for a sell / ask for a buy) —
         how='crossed_touch'. Still a real observed quote, not a modeled price.

    Every returned price is an actual quote that existed in ung_options_history;
    nothing is interpolated. Working window = bars at/after `exec_window` hour,
    Thursday pre-11:00 (EIA print) excluded."""
    import pandas as pd
    d = pd.Timestamp(date).normalize()
    ds = d.date().isoformat()
    exps = _expiries_on(ds)
    if not exps:
        return None
    target = (d + pd.Timedelta(days=int(dte))).date()
    exp = min(exps, key=lambda e: abs((e - target).days))
    if abs((exp - target).days) > 12:
        return None
    K_raw = float(K_adj)
    for sd, f in _SPLITS:
        if d < pd.Timestamp(sd):
            K_raw /= f
    bars = [(t, b, a) for (t, b, a) in _bars(ds, exp.isoformat(), round(K_raw, 1), right)
            if b and a and a > b and b > 0]
    if not bars:
        return None
    is_thu = d.dayofweek == 3
    allowed = [(t, b, a) for (t, b, a) in bars
               if not (avoid_print and is_thu and t.hour < 11)] or bars
    # working window = bars at/after the preferred exec hour (else the whole session)
    win = [x for x in allowed if x[0].hour >= exec_window] or allowed
    post_mid = (win[0][1] + win[0][2]) / 2.0       # resting limit = first-bar mid
    arr_spread = (win[0][2] - win[0][1]) / post_mid if post_mid > 0 else 1.0
    fill = None
    # CAUSAL execution — NO look-ahead. A real operator cannot pick the tightest FUTURE minute; the old
    # `min(win, key=spread)` selection did exactly that, fabricating ~half a spread of unachievable
    # "improvement" (found 2026-07-02 via execution_policy_compare: it showed a fake SELL +6.95% vs
    # cross-now that vanishes once the choice is causal). Honest policy:
    #   - tight arrival spread -> cross IMMEDIATELY at the arrival bar (win[0]) — 'tight_take'
    #   - wide arrival spread   -> rest a limit at the arrival mid; fill at mid if the tape trades THROUGH
    #     it ('passive_mid', causal — the market came to your resting order); if patience expires unfilled,
    #     cross at the LAST observed minute (win[-1]) — 'crossed_touch'. Every branch uses only past/present.
    if arr_spread <= TIGHT_SPREAD:
        t, b, a = win[0]
        fill = (b if side == 'sell' else a, t, b, a, 'tight_take')
    for (t, b, a) in (win if fill is None else []):  # WIDE: did the path trade through the resting mid?
        if side == 'sell' and b >= post_mid:
            fill = (post_mid, t, b, a, 'passive_mid'); break
        if side == 'buy' and a <= post_mid:
            fill = (post_mid, t, b, a, 'passive_mid'); break
    if fill is None:                                 # patience expired -> cross at the LAST observed minute
        t, b, a = win[-1]
        fill = (b if side == 'sell' else a, t, b, a, 'crossed_touch')
    px, t, bid, ask, how = fill
    mid = (bid + ask) / 2
    rel = (ask - bid) / mid if mid > 0 else 1.0
    sf = 1.0
    for sd, f in _SPLITS:
        if d < pd.Timestamp(sd):
            sf = f
    return {'price': round(px * sf, 4), 'exec_time': str(t),
            'bid': round(bid * sf, 4), 'ask': round(ask * sf, 4),
            'spread_pct': round(rel * 100, 1), 'mid': round(mid * sf, 4),
            'vs_mid': round((px - mid) * sf, 4), 'source': 'intraday', 'how': how}


if __name__ == '__main__':
    # demo: the $11 put on 2026-06-12 — sell, patient vs eager, vs the EOD touch
    print("Intraday fill demo — UNG $11 PUT, 2026-06-12 (exp 2026-07-17):")
    for pat in (0.9, 0.5, 0.1):
        r = fill('2026-06-12', '2026-07-17', 11.0, 'P', 'sell', patience=pat)
        if r.get('filled'):
            print(f"  sell, patience {pat}: fill ${r['price']} (mid ${r['mid']}, "
                  f"slip {r['vs_mid']:+.3f}, tightest spread {r['spread_pct']}%, "
                  f"{r['n_quotes']} quotes, @ {r['fill_time']})")
        else:
            print(f"  patience {pat}: {r.get('reason')}")
