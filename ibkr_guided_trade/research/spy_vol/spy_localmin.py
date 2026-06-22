"""Is a LOCAL MIN of VIX a good enough entry (vs absolute VIX<=16)? Compute causal local-min signals
from the VIX series, merge with the all-day straddle outcomes (spy_allday.csv, +30/-40/30d, not-cheap),
and compare. Tests: does local-min work at ANY VIX level? does it ADD to VIX<=16? does it rescue
high-VIX entries?

  venv/bin/python research/spy_vol/spy_localmin.py
"""
import os
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
vix = pd.read_csv(os.path.join(THIS, 'cache', 'spy_vix_daily.csv'), index_col=0, parse_dates=True)['VIX'].dropna()
vix.index = vix.index.normalize()

# causal signals (use only info up to & including day t)
f = pd.DataFrame(index=vix.index)
f['vix'] = vix
f['lmin10'] = vix <= vix.rolling(10).min()           # at/within a 10-day low
f['lmin20'] = vix <= vix.rolling(20).min()
f['lmin42'] = vix <= vix.rolling(42).min()
f['near_lmin20'] = vix <= vix.rolling(20).min() * 1.03   # within 3% of 20d low
# confirmed trough: yesterday was a local min (fell into it, ticked up today) — causal, 1-day lag
f['trough'] = (vix.shift(1) < vix.shift(2)) & (vix.shift(1) < vix) & (vix.shift(1) <= vix.rolling(20).min().shift(1) * 1.05)
# rising off a recent low
f['rising_off_low'] = (vix > vix.shift(1)) & (vix.shift(1) <= vix.rolling(20).min().shift(1) * 1.05)

f.index.name = 'date'
ad = pd.read_csv(os.path.join(THIS, 'spy_allday.csv'), parse_dates=['date'])
ad['date'] = ad['date'].dt.normalize()
m = ad.merge(f.reset_index(), on='date', how='inner', suffixes=('', '_f'))
m = m[m['iv_rv'] >= 0]          # not-cheap (kernel filter)


def line(sub):
    r = sub['ret']
    if not len(r):
        return "n=0"
    return f"n={len(r):>4}  avg {r.mean():>+7.1%}  win {(r>0).mean()*100:>3.0f}%  big>20% {(r>0.20).mean()*100:>3.0f}%"


print(f"=== LOCAL-MIN of VIX as entry (not-cheap entries, +30/-40/30d) — pool n={len(m)} ===")
print(f"  baseline ALL            : {line(m)}")
print(f"  VIX<=16 (current)       : {line(m[m['vix']<=16])}")
print(f"  VIX<=15                 : {line(m[m['vix']<=15])}")
print("\n  --- local-min ALONE (any VIX level) ---")
print(f"  lmin10                  : {line(m[m['lmin10']])}")
print(f"  lmin20                  : {line(m[m['lmin20']])}")
print(f"  lmin42                  : {line(m[m['lmin42']])}")
print(f"  near_lmin20 (within 3%) : {line(m[m['near_lmin20']])}")
print(f"  confirmed trough        : {line(m[m['trough']])}")
print(f"  rising off low          : {line(m[m['rising_off_low']])}")
print("\n  --- does local-min ADD to / replace VIX<=16? ---")
print(f"  lmin20 & VIX<=16        : {line(m[m['lmin20'] & (m['vix']<=16)])}")
print(f"  lmin20 & VIX>16 (rescue?): {line(m[m['lmin20'] & (m['vix']>16)])}")
print(f"  trough & VIX<=16        : {line(m[m['trough'] & (m['vix']<=16)])}")
print(f"  trough & VIX>16 (rescue?): {line(m[m['trough'] & (m['vix']>16)])}")
print(f"  VIX<=16 & NOT lmin20     : {line(m[(m['vix']<=16) & ~m['lmin20']])}")
print("\n  --- coverage: how many entries each fires on ---")
for c in ['lmin20', 'near_lmin20', 'trough']:
    both = (m[c] & (m['vix'] <= 16)).sum()
    print(f"    {c}: {m[c].sum()} fires ({m[c].sum()/len(m)*100:.0f}% of pool); {both} also VIX<=16")
print("DONE")
