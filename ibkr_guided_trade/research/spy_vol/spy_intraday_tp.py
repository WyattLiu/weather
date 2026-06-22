"""SPY INTRADAY TP-on-pop scalp: can a same-day take-profit beat the -0.75% hold-to-close bleed?

Buy the ~45 DTE ATM straddle at 14:00 (combo-mid), then walk the minute path: exit at the BID the
first minute the combo MID is ≥ entry·(1+TP); else exit at the close bid. Realistic: patient-in
(mid), cross-out (bid). Sweep TP ∈ {0.5,1,2,3,5%}. Reports vs hold-to-close (exit bid) baseline.

  venv/bin/python research/spy_vol/spy_intraday_tp.py
"""
import os
import sys
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry
from spy_minute_combo_study import minute_net, fill_combo_mid

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')
ENTRY_T = '14:00'
TPS = [0.005, 0.01, 0.02, 0.03, 0.05]


def mins_between(a, b):
    return (int(b[:2]) * 60 + int(b[3:5])) - (int(a[:2]) * 60 + int(a[3:5]))


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)
    spv.index = spv.index.normalize()
    spv = spv[spv.index >= '2018-01-01']
    spy = spv['SPY']
    entries = [d for d in spv.index if d.weekday() in (0, 2, 4)]
    cur = _conn().cursor()

    res = {tp: [] for tp in TPS}; hit = {tp: 0 for tp in TPS}; hm = {tp: [] for tp in TPS}
    close_real = []
    n = 0
    for d_ts in entries:
        d = d_ts.date(); S = float(spy.loc[d_ts])
        pe = pick_entry(cur, d, S, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe
        net = minute_net(cur, exp, K, d)
        if not net or len(net) < 60:
            continue
        entry = fill_combo_mid(net, 'buy', start=ENTRY_T, work=15)
        if not entry or entry <= 0:
            continue
        mins = sorted(t for t in net if t >= ENTRY_T)
        if len(mins) < 5:
            continue
        n += 1
        close_bid = net[mins[-1]][0]
        close_real.append(close_bid / entry - 1)
        for tp in TPS:
            ex = None; held = None
            for t in mins:
                b, a = net[t]; mid = (b + a) / 2
                if mid >= entry * (1 + tp):
                    ex = b; held = mins_between(ENTRY_T, t); hit[tp] += 1; break   # cross out at bid
            if ex is None:
                ex = net[mins[-1]][0]; held = mins_between(ENTRY_T, mins[-1])      # close bid
            res[tp].append(ex / entry - 1); hm[tp].append(held)
    cur.connection.close()

    print(f"=== SPY INTRADAY TP-ON-POP scalp (n={n}, enter 14:00 mid, exit bid) ===\n")
    cr = np.array(close_real)
    print(f"  hold-to-close (no TP):  avg {cr.mean():+.2%}  win {(cr>0).mean():.0%}\n")
    print(f"{'TP':>6}{'n':>6}{'avg ret':>9}{'win%':>7}{'hitTP%':>8}{'avg hold(min)':>15}")
    print('-' * 51)
    for tp in TPS:
        r = np.array(res[tp])
        print(f"{tp:>6.1%}{len(r):>6}{r.mean():>+9.2%}{(r>0).mean():>6.0%}{hit[tp]/n*100:>7.0f}%{np.mean(hm[tp]):>15.0f}")
    print("\n  (win% > hitTP% because a near-miss can still close green; TP exits at bid so realized < TP)")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
