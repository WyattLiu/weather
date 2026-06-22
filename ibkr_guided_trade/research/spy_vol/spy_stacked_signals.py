"""SPY STACKED-SIGNAL multi-day backtest with realistic fills (ask-in / bid-out).

Re-runs the real long-straddle trade (spy_vega_study.run_trade: enter ask, exit bid, first-of
+30%/−40%/VIX+3/max-hold) over dense Mon/Wed/Fri entries 2018-2026, then scores each entry by how
many frontier signals it passes and buckets net return by stack-score:
  LOW VIX (bottom tercile) · IV<RV (cheap) · QUIET intraday-RV (bottom tercile) · FLAT skew (<median)
Shows whether stacking lifts return/Sharpe/win and the deployment frontier (return vs % weeks traded).

  venv/bin/python research/spy_vol/spy_stacked_signals.py
"""
import os
import sys
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry, run_trade

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')
FEAT = os.path.join(THIS, 'spy_intraday_vs_multiday.csv')   # date + signals from prior study
MAXHOLD, PT, STOP, VOLPOP = 15, 0.30, 0.40, 3.0


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)
    spv.index = spv.index.normalize()
    vix = spv['VIX']; spy = spv['SPY']
    vix_path = {d.date(): float(v) for d, v in vix.items()}
    feat = pd.read_csv(FEAT, parse_dates=['date'])
    feat['date'] = feat['date'].dt.date
    f = feat.set_index('date')

    # signal thresholds from the feature distribution
    vix_lo = f['vix'].quantile(1/3)
    rv_lo = f['intra_rv'].quantile(1/3)
    skew_md = f['skew'].median()

    cur = _conn().cursor()
    recs = []
    for d, row in f.iterrows():
        d_ts = pd.Timestamp(d)
        if d_ts not in spy.index:
            continue
        S = float(spy.loc[d_ts])
        pe = pick_entry(cur, d, S, 38, 52)
        if not pe:
            continue
        exp, K, dte = pe
        tr = run_trade(cur, d, S, exp, K, MAXHOLD, PT, STOP, VOLPOP, float(vix.loc[d_ts]), vix_path)
        if not tr:
            continue
        sig_lowvix = int(row['vix'] <= vix_lo)
        sig_cheap = int(row['iv_rv'] < 0) if pd.notna(row['iv_rv']) else 0
        sig_quiet = int(row['intra_rv'] <= rv_lo) if pd.notna(row['intra_rv']) else 0
        sig_flat = int(row['skew'] < skew_md) if pd.notna(row['skew']) else 0
        score = sig_lowvix + sig_cheap + sig_quiet + sig_flat
        recs.append({'date': d, 'ret': tr['ret'], 'held': tr['held'], 'reason': tr['reason'],
                     'score': score, 'lowvix': sig_lowvix, 'cheap': sig_cheap,
                     'quiet': sig_quiet, 'flat': sig_flat})
    cur.connection.close()

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(THIS, 'spy_stacked_signals.csv'), index=False)
    N = len(df)
    print(f"=== SPY STACKED SIGNALS — realistic fills (ask-in/bid-out), exit +{PT:.0%}/-{STOP:.0%}/VIX+{VOLPOP:.0f}/{MAXHOLD}d ===")
    print(f"    n={N}  thresholds: VIX≤{vix_lo:.1f} | IV<RV | intra_rv≤{rv_lo:.1%} | skew<{skew_md:.3f}\n")

    def line(x):
        x = np.asarray(x, float)
        sh = x.mean() / x.std() * math.sqrt(len(x)) if (len(x) and x.std() > 0) else 0
        return len(x), x.mean(), (x > 0).mean(), sh

    b = line(df['ret'])
    print(f"  baseline (all entries): n={b[0]} avg {b[1]:+.2%} win {b[2]:.0%} t={b[3]:.2f}\n")
    print(f"{'stack score':<12}{'n':>5}{'avg ret':>9}{'win%':>7}{'t-stat':>8}{'% weeks':>9}")
    print('-' * 50)
    for s in range(5):
        sub = df[df['score'] == s]['ret']
        if len(sub):
            L = line(sub)
            print(f"{s:<12}{L[0]:>5}{L[1]:>+9.2%}{L[2]:>6.0%}{L[3]:>8.2f}{L[0]/N*100:>8.0f}%")
    # cumulative: score >= k (the deployment frontier)
    print(f"\n{'score >= k':<12}{'n':>5}{'avg ret':>9}{'win%':>7}{'t-stat':>8}{'% weeks':>9}")
    print('-' * 50)
    for k in range(5):
        sub = df[df['score'] >= k]['ret']
        if len(sub):
            L = line(sub)
            print(f"{'>= '+str(k):<12}{L[0]:>5}{L[1]:>+9.2%}{L[2]:>6.0%}{L[3]:>8.2f}{L[0]/N*100:>8.0f}%")
    # single-signal marginal (each on vs off)
    print("\n=== single-signal marginal (signal ON vs OFF, avg net ret) ===")
    for s in ('lowvix', 'cheap', 'quiet', 'flat'):
        on = df[df[s] == 1]['ret']; off = df[df[s] == 0]['ret']
        print(f"  {s:<8} ON {on.mean():+.2%} (n={len(on)})  vs OFF {off.mean():+.2%}  Δ{on.mean()-off.mean():+.2%}")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
