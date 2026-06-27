"""'Hedged short' structures: how much MORE of the free-fall do we capture by tilting toward puts?
Compare on the same low-VIX (<=16) entries, same exit (+30/-40/30d, enter ask/exit bid):
  straddle 1C+1P · put-tilt 1C+2P · put-tilt 1C+3P · long ATM put 1P · put-spread (buy ATM P / sell 7% P)
Report avg ret / win% / cost-as-%-of-spot, split by whether SPY ROSE or FELL over the hold —
so we see downside capture vs the upside give-up (the cost of leaning bearish).

  venv/bin/python research/spy_vol/spy_puttilt.py
"""
import os
import sys
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry, eod_mid

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')
PT, STOP, MAXHOLD = 0.30, 0.40, 30


def legval(legs, d):
    """legs = list of (path, qty, side); side +1 long (ask in / bid out), -1 short (bid in / ask out).
    returns (entry_cost, common_dates) using ask/bid for fills."""
    common = None
    for pth, _, _ in legs:
        s = set(pth)
        common = s if common is None else (common & s)
    common = sorted(t for t in common if t >= d)
    if not common or d not in common:
        return None
    entry = 0.0
    for pth, q, side in legs:
        entry += q * (pth[d][2] if side > 0 else -pth[d][1])
    return entry, common


def sim(which, d):
    r = legval(which, d)
    if not r:
        return None
    entry, common = r
    if entry <= 0:
        return None
    fut = [t for t in common if t > d]
    mx = 0.0
    for i, t in enumerate(fut):
        mid = sum(q * side * (pth[t][0]) for pth, q, side in which)
        # exit: long sells bid, short buys ask
        ex = sum(q * (pth[t][1] if side > 0 else -pth[t][2]) for pth, q, side in which)
        ret = mid / entry - 1; held = (t - d).days
        mx = max(mx, ret)
        if ret >= PT or ret <= -STOP or held >= MAXHOLD or i == len(fut) - 1:
            return {'entry': entry, 'ret': ex / entry - 1, 'xdate': t}
    return None


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    spv = spv[spv.index >= '2018-01-01']
    vix = spv['VIX']; spy = spv['SPY']
    rv20 = np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252)
    cur = _conn().cursor(); pc = {}

    def pth(exp, K, r, d):
        k = (exp, K, r)
        if k not in pc:
            pc[k] = eod_mid(cur, exp, K, r, d, exp)
        return pc[k]

    structs = {'straddle 1C1P': [], 'put-tilt 1C2P': [], 'put-tilt 1C3P': [],
               'long put 1P': [], 'put-spread': []}
    for d_ts in spv.index:
        d = d_ts.date(); S = float(spy.loc[d_ts]); vx = float(vix.loc[d_ts])
        if vx > 16 or math.isnan(rv20.loc[d_ts]):
            continue
        pe = pick_entry(cur, d, S, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe; T = dte / 365
        ca = pth(exp, K, 'C', d); pa = pth(exp, K, 'P', d)
        if d not in ca or d not in pa:
            continue
        emid = ca[d][0] + pa[d][0]
        iv = emid / (0.7979 * S * math.sqrt(T)) if T > 0 else 0
        if iv < float(rv20.loc[d_ts]):
            continue
        cur.execute("SELECT DISTINCT strike FROM spy_options_history WHERE trade_date=%s AND expiration=%s", (d, exp))
        ks = sorted(float(r[0]) for r in cur.fetchall())
        Kp7 = min(ks, key=lambda k: abs(k - S * 0.93)) if ks else None
        p7 = pth(exp, Kp7, 'P', d) if Kp7 else None
        float(spy.loc[pd.Timestamp(d)])
        defs = {
            'straddle 1C1P': [(ca, 1, 1), (pa, 1, 1)],
            'put-tilt 1C2P': [(ca, 1, 1), (pa, 2, 1)],
            'put-tilt 1C3P': [(ca, 1, 1), (pa, 3, 1)],
            'long put 1P': [(pa, 1, 1)],
            'put-spread': ([(pa, 1, 1), (p7, 1, -1)] if p7 else None),
        }
        for name, legs in defs.items():
            if not legs:
                continue
            r = sim(legs, d)
            if r:
                spy_move = float(spy.loc[pd.Timestamp(r['xdate'])]) / S - 1 if pd.Timestamp(r['xdate']) in spy.index else np.nan
                structs[name].append((r['ret'], r['entry'] / S, spy_move))

    print("=== 'HEDGED SHORT' STRUCTURES (low-VIX entries, +30/-40/30d, enter ask/exit bid) ===\n")
    print(f"{'structure':<16}{'n':>5}{'avg':>8}{'win%':>6}{'cost/S':>8}{'  ret|SPY↓':>11}{'ret|SPY↑':>10}")
    print('-' * 64)
    for name, rows in structs.items():
        if not rows:
            continue
        a = np.array([r[0] for r in rows]); c = np.array([r[1] for r in rows]); mv = np.array([r[2] for r in rows])
        dn = a[mv < 0]; up = a[mv > 0]
        print(f"{name:<16}{len(a):>5}{a.mean():>+8.1%}{(a>0).mean()*100:>5.0f}%{c.mean()*100:>7.1f}%"
              f"{(dn.mean() if len(dn) else 0):>+11.1%}{(up.mean() if len(up) else 0):>+10.1%}")
    print("\n  cost/S = entry debit % of spot; ret|SPY↓ = avg return on holds where SPY fell (free-fall capture)")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
