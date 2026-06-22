"""BRUTE-FORCE: simulate a ~45 DTE ATM straddle on EVERY trading day 2018-2026 (no kernel filter),
tag all winners, and dump rich features so we can see what CONNECTS the winners — including winners
the kernel's VIX<=16 filter never saw. Exit +30%/-40%/30d (v2, no volpop), enter ask / exit bid.

Outputs spy_allday.csv and prints winner-rate / avg-ret by feature bucket, plus 'kernel-missed'
winners (VIX>16) and the biggest blind winners with their setup.

  venv/bin/python research/spy_vol/spy_allday_scan.py
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


def sim(c, p, d):
    if d not in c or d not in p:
        return None
    entry = c[d][2] + p[d][2]
    if entry <= 0:
        return None
    fut = [t for t in sorted(set(c) & set(p)) if t > d]
    mx = 0.0
    for i, t in enumerate(fut):
        mid = c[t][0] + p[t][0]; bid = c[t][1] + p[t][1]
        ret = mid / entry - 1; held = (t - d).days
        mx = max(mx, ret)
        if ret >= PT or ret <= -STOP or held >= MAXHOLD or i == len(fut) - 1:
            return {'entry': entry, 'ret': bid / entry - 1, 'maxmid': mx, 'held': held}
    return None


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    spv = spv[spv.index >= '2018-01-01']
    vix = spv['VIX']; spy = spv['SPY']
    lr = np.log(spy / spy.shift(1))
    feat = pd.DataFrame(index=spv.index)
    feat['vix'] = vix
    feat['vix_std10'] = vix.rolling(10).std()
    feat['vix_pct'] = vix.rolling(252, min_periods=60).apply(lambda w: (w.iloc[-1] >= w).mean())
    feat['vix_chg5'] = vix - vix.shift(5)
    feat['rv20'] = lr.rolling(20).std() * math.sqrt(252)
    feat['rv5'] = lr.rolling(5).std() * math.sqrt(252)
    feat['mom20'] = spy / spy.shift(20) - 1
    feat['mom5'] = spy / spy.shift(5) - 1
    feat['dist_high'] = spy / spy.rolling(252, min_periods=60).max() - 1
    feat['dow'] = spv.index.weekday

    cur = _conn().cursor()
    rows = []
    for d_ts in spv.index:
        d = d_ts.date(); S = float(spy.loc[d_ts])
        if math.isnan(feat.loc[d_ts, 'rv20']):
            continue
        pe = pick_entry(cur, d, S, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe; T = dte / 365
        c = eod_mid(cur, exp, K, 'C', d, exp); p = eod_mid(cur, exp, K, 'P', d, exp)
        if d not in c or d not in p:
            continue
        emid = c[d][0] + p[d][0]
        iv = emid / (0.7979 * S * math.sqrt(T)) if T > 0 else np.nan
        # 5% wing skew (approx)
        cur.execute("SELECT DISTINCT strike FROM spy_options_history WHERE trade_date=%s AND expiration=%s", (d, exp))
        ks = [float(r[0]) for r in cur.fetchall()]
        skew = np.nan
        if ks:
            Kp = min(ks, key=lambda k: abs(k - S * 0.95)); Kc = min(ks, key=lambda k: abs(k - S * 1.05))
            cp = eod_mid(cur, exp, Kp, 'P', d, d); cc = eod_mid(cur, exp, Kc, 'C', d, d)
            if d in cp and d in cc:
                skew = (cp[d][0] - cc[d][0]) / (0.40 * S * math.sqrt(T))
        r = sim(c, p, d)
        if not r:
            continue
        row = {'date': d, 'ret': r['ret'], 'maxmid': r['maxmid'], 'held': r['held'],
               'iv': iv, 'iv_rv': iv - feat.loc[d_ts, 'rv20'], 'skew': skew}
        for col in feat.columns:
            row[col] = float(feat.loc[d_ts, col])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(THIS, 'spy_allday.csv'), index=False)

    N = len(df); W = df[df['ret'] > 0]; BW = df[df['ret'] > 0.20]
    print(f"=== BRUTE-FORCE ALL-DAY STRADDLE SCAN (every trading day, +30/-40/30d) — n={N} ===")
    print(f"  baseline: avg {df['ret'].mean():+.2%}  win {(df['ret']>0).mean():.0%}  "
          f">+20% days: {len(BW)} ({len(BW)/N:.0%})  >+50%: {(df['ret']>0.50).sum()}\n")

    def bucket(col, qs=(0, .25, .5, .75, 1.0)):
        s = df[[col, 'ret']].dropna()
        edges = s[col].quantile(list(qs)).values
        print(f"  {col} quartiles → avg ret | win% | %that are >+20% winners:")
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sub = s[(s[col] >= lo) & (s[col] <= hi)] if i == len(edges) - 2 else s[(s[col] >= lo) & (s[col] < hi)]
            if len(sub):
                print(f"    [{lo:>7.2f},{hi:>7.2f}] n={len(sub):>4}  {sub['ret'].mean():>+7.1%} | "
                      f"{(sub['ret']>0).mean()*100:>3.0f}% | {(sub['ret']>0.20).mean()*100:>3.0f}%")

    for col in ['vix', 'vix_std10', 'vix_pct', 'iv_rv', 'skew', 'mom20', 'dist_high', 'rv20', 'vix_chg5']:
        bucket(col)
    print()
    # winners the kernel never saw (VIX>16)
    miss = df[(df['vix'] > 16) & (df['ret'] > 0.20)]
    print(f"=== KERNEL-MISSED big winners (VIX>16, ret>+20%): {len(miss)} of {len(BW)} total big winners ===")
    print(f"  of all >+20% winners, {(BW['vix']>16).mean()*100:.0f}% had VIX>16 at entry "
          f"(median VIX of big winners {BW['vix'].median():.1f}; median mom20 {BW['mom20'].median():+.1%}; "
          f"median dist_high {BW['dist_high'].median():+.1%})")
    print("\n  biggest 10 blind winners (ret / VIX / mom20 / dist_high / iv_rv / skew):")
    for _, r in df.nlargest(10, 'ret').iterrows():
        print(f"    {r['date']}  {r['ret']:+.0%}  VIX{r['vix']:>5.1f}  mom20{r['mom20']:>+6.1%}  "
              f"dist{r['dist_high']:>+6.1%}  ivrv{r['iv_rv']:>+5.2f}  skew{r['skew']:>+5.2f}")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
