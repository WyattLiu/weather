"""AUDIT the SPY vega kernel: dump every trade, find left-on-table profit, missed green days,
signal false-negatives, and compare alternative exit strategies. Mirrors the conc=1 kernel
(VIX<=16 & IV>=RV20, ~45DTE ATM, enter ask / exit bid).

  venv/bin/python research/spy_vol/spy_vega_audit.py
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
VIX_THR, PT, STOP, VOLPOP, MAXHOLD = 16.0, 0.30, 0.40, 3.0, 15


def get_paths(cur, exp, K, d, cache):
    ck = (exp, K)
    if ck not in cache:
        cache[ck] = (eod_mid(cur, exp, K, 'C', d, exp), eod_mid(cur, exp, K, 'P', d, exp))
    return cache[ck]


def sim_trade(c, p, d, vix, exits):
    """Walk the straddle path from entry day d. Return dict with actual exit + diagnostics +
    a set of alternative-exit returns. exits = dict of name->(pt,stop,volpop,maxhold,to_expiry)."""
    if d not in c or d not in p:
        return None
    entry = c[d][2] + p[d][2]
    if entry <= 0:
        return None
    vix0 = float(vix.loc[pd.Timestamp(d)])
    fut = [t for t in sorted(set(c) & set(p)) if t > d]
    if not fut:
        return None
    maxmid = 0.0
    base = None              # the kernel's actual exit (PT/STOP/VOLPOP/MAXHOLD)
    alt = {k: None for k in exits}
    for i, t in enumerate(fut):
        mid = c[t][0] + p[t][0]; bid = c[t][1] + p[t][1]
        ret_mid = mid / entry - 1; ret_bid = bid / entry - 1
        held = (t - d).days
        vix_t = float(vix.loc[pd.Timestamp(t)]) if pd.Timestamp(t) in vix.index else vix0
        maxmid = max(maxmid, ret_mid)
        # base kernel exit
        if base is None and (ret_mid >= PT or ret_mid <= -STOP or vix_t >= vix0 + VOLPOP
                             or held >= MAXHOLD or i == len(fut) - 1):
            reason = ('profit' if ret_mid >= PT else 'stop' if ret_mid <= -STOP
                      else 'volpop' if vix_t >= vix0 + VOLPOP else 'time' if held >= MAXHOLD else 'expiry')
            base = {'edate': d, 'xdate': t, 'held': held, 'reason': reason, 'entry': entry,
                    'ret': ret_bid, 'vix0': vix0, 'vix_x': vix_t}
        # alternative exits
        for name, (pt, stop, vp, mh, toexp) in exits.items():
            if alt[name] is None:
                hit = (ret_mid >= pt or (stop and ret_mid <= -stop)
                       or (vp and vix_t >= vix0 + vp) or (not toexp and held >= mh)
                       or i == len(fut) - 1)
                if hit:
                    alt[name] = ret_bid
    if base:
        base['maxmid'] = maxmid
        base['alt'] = alt
    return base


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    spv = spv[spv.index >= '2018-01-01']
    vix = spv['VIX']; spy = spv['SPY']
    rv20 = np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252)
    days = list(spv.index)
    cur = _conn().cursor()
    cache = {}

    ALT = {  # name: (pt, stop, volpop, maxhold, to_expiry)
        'PT30_hold30d': (0.30, 0.40, 3.0, 30, False),
        'PT30_hold30d_noVP': (0.30, 0.40, 0.0, 30, False),
        'PT30_hold45d_noVP': (0.30, 0.40, 0.0, 45, False),
        'PT50_hold30d_noVP': (0.50, 0.40, 0.0, 30, False),
        'PT50_hold45d_noVP': (0.50, 0.40, 0.0, 45, False),
        'PT75_hold45d_noVP': (0.75, 0.0, 0.0, 45, False),
        'noPT_hold30d_noVP': (9.99, 0.0, 0.0, 30, False),
    }

    trades = []
    skipped_green = []     # green days we couldn't take because already in a trade (conc=1)
    open_until = None
    allgreen = []
    for d_ts in days:
        d = d_ts.date(); S = float(spy.loc[d_ts])
        if math.isnan(rv20.loc[d_ts]):
            continue
        vx = float(vix.loc[d_ts])
        if vx > VIX_THR:
            continue
        pe = pick_entry(cur, d, S, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe
        c, p = get_paths(cur, exp, K, d, cache)
        if d not in c or d not in p:
            continue
        entry_mid = c[d][0] + p[d][0]
        iv = entry_mid / (0.7979 * S * math.sqrt(dte / 365)) if dte > 0 else 0
        if iv < float(rv20.loc[d_ts]):           # "cheap" → skip (the trap)
            continue
        allgreen.append(d)
        if open_until is not None and d_ts <= open_until:
            skipped_green.append(d)               # would-be entry, but conc=1 blocks
            continue
        tr = sim_trade(c, p, d, vix, ALT)
        if not tr:
            continue
        trades.append(tr)
        open_until = pd.Timestamp(tr['xdate'])

    # also: what did the SKIPPED green days return (missed by conc=1)?
    skip_rets = []
    for d in skipped_green:
        d_ts = pd.Timestamp(d); S = float(spy.loc[d_ts])
        pe = pick_entry(cur, d, S, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe
        c, p = get_paths(cur, exp, K, d, cache)
        tr = sim_trade(c, p, d, vix, {})
        if tr:
            skip_rets.append(tr['ret'])

    tdf = pd.DataFrame([{k: v for k, v in t.items() if k != 'alt'} for t in trades])
    print(f"=== ALL KERNEL TRADES (VIX<=16 & IV>=RV, conc=1, 2018-2026) — n={len(tdf)} ===")
    pd.set_option('display.width', 200)
    show = tdf[['edate', 'xdate', 'held', 'reason', 'entry', 'ret', 'maxmid', 'vix0', 'vix_x']].copy()
    show['ret'] = (show['ret'] * 100).round(1); show['maxmid'] = (show['maxmid'] * 100).round(1)
    show['entry'] = show['entry'].round(2)
    print(show.to_string(index=False))

    print("\n=== by exit reason ===")
    for r, g in tdf.groupby('reason'):
        print(f"  {r:<8} n={len(g):>2}  avg ret {g['ret'].mean():+.1%}  win {(g['ret']>0).mean():.0%}  "
              f"avg held {g['held'].mean():.0f}d")
    print(f"  TOTAL avg ret {tdf['ret'].mean():+.2%}  win {(tdf['ret']>0).mean():.0%}  median {tdf['ret'].median():+.1%}")

    print("\n=== LEFT ON THE TABLE (max-mid reached vs actual exit) ===")
    tdf['left'] = tdf['maxmid'] - tdf['ret']
    big = tdf[tdf['left'] > 0.15].sort_values('left', ascending=False)
    print(f"  {len(tdf[tdf['maxmid']>=PT])} trades touched +{PT:.0%} mid; {len(big)} left >15% on the table")
    print(f"  avg max-mid {tdf['maxmid'].mean():+.1%} vs avg exit {tdf['ret'].mean():+.1%}  "
          f"(gap {tdf['left'].mean():+.1%})")
    stops = tdf[tdf['reason'] == 'stop']
    print(f"  stops that later reached +{PT:.0%} mid: {len(stops[stops['maxmid']>=PT])}/{len(stops)}")

    print("\n=== MISSED green days (conc=1 blocked) ===")
    print(f"  {len(skipped_green)} green days fell inside an open trade; "
          f"their standalone avg ret {np.mean(skip_rets):+.1%} (n={len(skip_rets)}) "
          f"vs taken {tdf['ret'].mean():+.1%}")

    print("\n=== ALTERNATIVE EXIT STRATEGIES (same entries, avg / win / total-compounded) ===")
    base_series = tdf['ret'].values
    def comp(x): return np.prod([1 + r for r in x]) - 1
    print(f"  {'BASE +30/-40/VIX+3/15d':<24} avg {np.mean(base_series):+.2%}  win {(base_series>0).mean():.0%}  compounded {comp(base_series):+.1%}")
    for name in ALT:
        rs = [t['alt'][name] for t in trades if t['alt'][name] is not None]
        if rs:
            print(f"  {name:<24} avg {np.mean(rs):+.2%}  win {(np.array(rs)>0).mean():.0%}  compounded {comp(rs):+.1%}")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
