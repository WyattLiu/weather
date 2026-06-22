"""How sensitive is the edge to the VIX cutoff? Per-trade bands (from spy_allday.csv, not-cheap
filter) + kernel NAV sweep (alloc20/conc2) across thresholds 14..18. Informs caution tiers.

  venv/bin/python research/spy_vol/spy_vix_threshold.py
"""
import os
import sys
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

df = pd.read_csv(os.path.join(THIS, 'spy_allday.csv'))
nc = df[df['iv_rv'] >= 0]          # kernel's not-cheap filter


def line(sub):
    r = sub['ret']
    if not len(r):
        return "n=0"
    return f"n={len(r):>4}  avg {r.mean():>+7.1%}  win {(r>0).mean()*100:>3.0f}%  big>20% {(r>0.20).mean()*100:>3.0f}%  >+50% {(r>0.50).sum():>2}"


print("=== PER-TRADE by VIX CUTOFF (not-cheap entries; cumulative ≤thr) ===")
for thr in (14, 15, 16, 16.5, 17, 18, 99):
    lab = f"VIX≤{thr}" if thr < 99 else "all"
    print(f"  {lab:<8}: {line(nc[nc['vix'] <= thr])}")
print("\n=== MARGINAL BANDS (the cost of each extra point of VIX) ===")
bands = [(0, 14), (14, 15), (15, 16), (16, 16.5), (16.5, 17), (17, 18), (18, 99)]
for lo, hi in bands:
    sub = nc[(nc['vix'] > lo) & (nc['vix'] <= hi)]
    print(f"  VIX {lo:>4}-{hi:<4}: {line(sub)}")

# ---- kernel NAV sweep by threshold ----
try:
    from spy_vega_kernel import backtest
    from spy_vega_study import _conn
    spv = pd.read_csv(os.path.join(THIS, 'cache', 'spy_vix_daily.csv'), index_col=0, parse_dates=True)
    spv.index = spv.index.normalize(); spv = spv[spv.index >= '2018-01-01']
    vix = spv['VIX']; spy = spv['SPY']
    rv20 = np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252)
    days = list(spv.index); cur = _conn().cursor(); cache = {}
    print("\n=== KERNEL NAV by VIX threshold (alloc20/conc2, +30/30d/no-volpop) ===")
    print(f"  {'thr':<7}{'CAGR':>7}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}{'deploy':>8}")
    for thr in (14, 15, 16, 16.5, 17, 18):
        ser = pd.Series(float(thr), index=vix.index)
        r = backtest(cur, days, vix, spy, rv20, ser, 0.20, False, 2, cache=cache)
        print(f"  ≤{thr:<6}{r['cagr']:>6.1%}{r['sharpe']:>8.2f}{r['mdd']:>8.1%}{r['trades']:>8}{r['deploy%']:>7.0%}")
except Exception as e:
    print("\n  (kernel sweep skipped:", repr(e)[:80], ")")
print("DONE", flush=True)
