"""MINUTE-FILL validation of the vega-scrape + intraday combo-limit scaling.

Re-prices the CONSOLIDATED-LOW straddle entries with REAL minute fills instead of EOD mid:
  • CROSS        — buy at net ask / sell at net bid (worst case; the EOD-study baseline)
  • COMBO-MID    — post a combo LIMIT at the net mid; if the minute path trades through within
                   WORK_MIN, fill at mid; else cross at the end (patient combo, what you'd type)
  • SCALE        — split into TRANCHES across the session, each worked as combo-mid (scale in/out)
Also probes an INTRADAY-TIMING edge: net spread (cost to cross) by time-of-day bucket.

Combo net for a long straddle = call + put: net_ask = Cask+Pask, net_bid = Cbid+Pbid,
net_mid = (net_ask+net_bid)/2. A combo limit fills on the NET, so we model the net path.

  venv/bin/python research/spy_vol/spy_minute_combo_study.py
"""
import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')
LOW, STABLE = 15.0, 1.2
WORK_MIN = 15            # minutes to work a combo-mid limit before crossing
TRANCHES = [('10:00', 1/3), ('12:30', 1/3), ('15:30', 1/3)]   # scale-in/out clock × weight


def minute_net(cur, exp, K, d):
    """Per-minute combo net (bid,ask) for the straddle (call+put) on date d.
    Returns {hhmm: (net_bid, net_ask)} where both legs are two-sided that minute."""
    cur.execute("""
        SELECT bar_time, option_right, data_type, close
        FROM spy_options_history
        WHERE trade_date=%s AND expiration=%s AND strike=%s
        ORDER BY bar_time
    """, (d, exp, K))
    leg = defaultdict(dict)     # hhmm -> {('C','BID'):x,...}
    for bt, right, dt, close in cur.fetchall():
        leg[bt.strftime('%H:%M')][(right, dt)] = float(close)
    out = {}
    for hhmm, m in leg.items():
        if all((r, s) in m for r in 'CP' for s in ('BID', 'ASK')):
            nb = m[('C', 'BID')] + m[('P', 'BID')]
            na = m[('C', 'ASK')] + m[('P', 'ASK')]
            if na > nb > 0:
                out[hhmm] = (nb, na)
    return out


def fill_cross(net, side):
    """Marketable cross at the last available minute: buy→ask, sell→bid."""
    if not net:
        return None
    last = sorted(net)[-1]
    nb, na = net[last]
    return na if side == 'buy' else nb


def fill_combo_mid(net, side, start='10:00', work=WORK_MIN):
    """Post a combo limit at the net mid at `start`; fill at mid if the path trades through
    within `work` minutes, else cross at end of the window. Returns fill price or None."""
    mins = sorted(net)
    mins = [m for m in mins if m >= start] or mins
    if not mins:
        return None
    s = mins[0]
    nb, na = net[s]
    limit = (nb + na) / 2
    window = [m for m in mins if m <= _addmin(s, work)]
    for m in window:
        b, a = net[m]
        if side == 'buy' and a <= limit:     # ask came down to our bid → filled at mid
            return limit
        if side == 'sell' and b >= limit:    # bid came up to our ask → filled at mid
            return limit
    b, a = net[window[-1]]                    # not filled → cross
    return a if side == 'buy' else b


def fill_scale(net, side):
    """Scale across TRANCHES, each worked as a combo-mid from its clock. Weighted-avg fill."""
    px, w = 0.0, 0.0
    for clock, wt in TRANCHES:
        f = fill_combo_mid(net, side, start=clock, work=WORK_MIN)
        if f is not None:
            px += f * wt; w += wt
    return px / w if w else None


def _addmin(hhmm, m):
    h, mi = int(hhmm[:2]), int(hhmm[3:])
    t = h * 60 + mi + m
    return f"{t//60:02d}:{t%60:02d}"


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    vix = spv['VIX']; v10 = vix.rolling(10).std()
    entries = spv[(spv.index.weekday == 0) & (vix < LOW) & (v10 < STABLE)]   # CONSOLIDATED-LOW
    conn = _conn(); cur = conn.cursor()
    rows = []
    tod_spread = defaultdict(list)
    for d_ts, row in entries.iterrows():
        d = d_ts.date(); spot = float(row['SPY'])
        pe = pick_entry(cur, d, spot, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe
        net_in = minute_net(cur, exp, K, d)
        if not net_in:
            continue
        # exit ~21 DTE later (≈ entry + (dte-21) days), find a trading date with quotes
        exit_d = d + pd.Timedelta(days=max(1, dte - 21))
        # search forward up to 5 days for a date that has minute quotes
        net_out = {}
        for off in range(0, 6):
            cand = (pd.Timestamp(exit_d) + pd.Timedelta(days=off)).date()
            net_out = minute_net(cur, exp, K, cand)
            if net_out:
                break
        if not net_out:
            continue
        # time-of-day spread (cost to cross = (ask-bid)/mid) on entry day
        for hhmm, (nb, na) in net_in.items():
            tod_spread[hhmm[:2]].append((na - nb) / ((na + nb) / 2))
        # three fill policies, entry then exit
        def rt(fin, fout):
            ein = fin(net_in, 'buy'); eout = fout(net_out, 'sell')
            return (eout - ein) / ein if (ein and eout and ein > 0) else None
        rows.append({
            'date': d, 'K': K, 'dte': dte,
            'cross': rt(fill_cross, fill_cross),
            'combo_mid': rt(lambda n, s: fill_combo_mid(n, s), lambda n, s: fill_combo_mid(n, s)),
            'scale': rt(fill_scale, fill_scale),
        })
    conn.close()

    df = pd.DataFrame(rows).dropna()
    print(f"=== MINUTE-FILL validation — CONSOLIDATED-LOW straddle entries (n={len(df)}) ===")
    print(f"{'fill policy':<14}{'win%':>7}{'avg ret':>9}{'median':>9}{'total':>9}")
    print('-' * 48)
    for col, lbl in [('cross', 'CROSS (worst)'), ('combo_mid', 'COMBO-MID'), ('scale', 'SCALE in/out')]:
        r = df[col].values
        print(f"{lbl:<14}{(r>0).mean()*100:>6.0f}%{r.mean():>+9.1%}{np.median(r):>+9.1%}{r.sum():>+9.1%}")
    print(f"\n  combo-mid vs cross edge (spread saved): {(df['combo_mid']-df['cross']).mean():+.2%}/trade avg")

    print("\n=== INTRADAY: net combo spread (cost to cross) by time-of-day ===")
    for hh in sorted(tod_spread):
        v = np.array(tod_spread[hh])
        if len(v) > 20:
            print(f"  {hh}:00h  median {np.median(v):.2%}  (n={len(v)})")
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
