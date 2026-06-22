"""#2 IV TERM-STRUCTURE early-warning: does front-month vs ~45D ATM IV slope predict vol spikes /
straddle payoffs? Contango (far>near, slope>0) = calm/complacent; backwardation (near>far, slope<0)
= stress. Computes slope per day, merges with spy_allday.csv, buckets the 45D straddle return and
forward 10d VIX change by slope — overall and within VIX<=16.

  venv/bin/python research/spy_vol/spy_termstructure.py
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


def atm_iv_at(cur, S, d, lo, hi):
    pe = pick_entry(cur, d, S, lo, hi)
    if not pe:
        return None
    exp, K, dte = pe; T = dte / 365
    c = eod_mid(cur, exp, K, 'C', d, d); p = eod_mid(cur, exp, K, 'P', d, d)
    if d not in c or d not in p:
        return None
    mid = c[d][0] + p[d][0]
    return mid / (0.7979 * S * math.sqrt(T)) if T > 0 else None


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    spv = spv[spv.index >= '2018-01-01']
    vix = spv['VIX']; spy = spv['SPY']
    vix_fwd10 = vix.shift(-10) - vix            # forward 10d VIX change
    ad = pd.read_csv(os.path.join(THIS, 'spy_allday.csv'), parse_dates=['date'])
    ad['date'] = ad['date'].dt.normalize()
    adi = ad.set_index('date')

    cur = _conn().cursor()
    rows = []
    for d_ts in spv.index:
        d = d_ts.date(); S = float(spy.loc[d_ts])
        ivf = atm_iv_at(cur, S, d, 5, 28)        # front month (0-28 DTE)
        ivb = atm_iv_at(cur, S, d, 38, 52)       # ~45 DTE back
        if ivf is None or ivb is None:
            continue
        rows.append({'date': d_ts, 'iv_front': ivf, 'iv_back': ivb, 'slope': ivb - ivf,
                     'vix': float(vix.loc[d_ts]), 'fwd_vix10': float(vix_fwd10.loc[d_ts]) if not math.isnan(vix_fwd10.loc[d_ts]) else np.nan,
                     'ret': float(adi.loc[d_ts, 'ret']) if d_ts in adi.index else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(THIS, 'spy_termstructure.csv'), index=False)
    n = len(df); inv = (df['slope'] < 0).mean()
    print(f"=== #2 IV TERM-STRUCTURE (front 0-28D vs back ~45D ATM IV) — n={n} ===")
    print(f"  slope<0 (backwardation/stress) on {inv:.0%} of days; mean slope {df['slope'].mean():+.3f}\n")

    def buck(sub, col, label):
        s = sub[['ret', 'fwd_vix10', 'slope']].dropna().reset_index(drop=True)
        if len(s) < 20:
            print(f"  {label}: n<20"); return
        qs = s['slope'].quantile([0, .25, .5, .75, 1.0]).values
        print(f"  {label} — by term-slope quartile:  straddle ret | win% | fwd-10d ΔVIX")
        for i in range(4):
            lo, hi = qs[i], qs[i + 1]
            x = s[(s['slope'] >= lo) & (s['slope'] <= hi)] if i == 3 else s[(s['slope'] >= lo) & (s['slope'] < hi)]
            if len(x):
                print(f"    slope[{lo:>+6.3f},{hi:>+6.3f}] n={len(x):>4}  {x['ret'].mean():>+7.1%} | "
                      f"{(x['ret']>0).mean()*100:>3.0f}% | {x['fwd_vix10'].mean():>+5.1f}")

    buck(df, 'slope', 'ALL days')
    print()
    buck(df[df['vix'] <= 16], 'slope', 'VIX<=16 entries only')
    print(f"\n  corr(slope, fwd-10d ΔVIX) = {df['slope'].corr(df['fwd_vix10']):+.2f}   "
          f"corr(slope, straddle ret) = {df['slope'].corr(df['ret']):+.2f}")
    # does inversion at low VIX flag imminent spikes?
    lv = df[df['vix'] <= 16]
    flat_or_inv = lv[lv['slope'] < lv['slope'].median()]
    steep = lv[lv['slope'] >= lv['slope'].median()]
    print(f"\n  VIX<=16 & FLATTER/inverted term: straddle {flat_or_inv['ret'].mean():+.1%} win {(flat_or_inv['ret']>0).mean():.0%} "
          f"fwdΔVIX {flat_or_inv['fwd_vix10'].mean():+.1f}")
    print(f"  VIX<=16 & STEEP contango:        straddle {steep['ret'].mean():+.1%} win {(steep['ret']>0).mean():.0%} "
          f"fwdΔVIX {steep['fwd_vix10'].mean():+.1f}")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
