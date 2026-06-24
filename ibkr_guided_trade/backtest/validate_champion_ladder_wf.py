"""Confirm the PROMOTED champion (gamma-weighted [14,30] ladder, now the default) holds across rolling
12-month windows vs the same champion WITHOUT the ladder. Parallel (120 cores). Win = ladder Sharpe >=
no-ladder Sharpe (and not materially worse MaxDD)."""
import sys
import os
import math
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple


def metrics(strat, df, nav0=100000):
    hist, trades = run_strategy_simple(df, strat, nav0, 0)
    hist = hist.set_index(pd.to_datetime(hist['date']))
    nav = hist['nav']
    rets = nav.pct_change().dropna()
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    ann = ((nav.iloc[-1] / nav0) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    sh = rets.mean() / (rets.std() + 1e-9) * math.sqrt(252)
    mdd = ((nav - nav.cummax()) / nav.cummax() * 100).min()
    vol = rets.std() * math.sqrt(252) * 100
    worst = rets.min() * 100
    return ann, sh, mdd, vol, worst


df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'master_dataset.csv'),
                 index_col=0, parse_dates=True)
df = precompute_factor_z(df).dropna(subset=['UNG'])
champ = STRATEGIES['regime_wheel_boxx_greeks']                       # NOW has the [14,30] ladder
noladder = {**champ, 'gamma_weighted_ladder': False, 'dte_ladder': [30]}

# rolling 12mo windows, step 4mo
starts = pd.date_range('2021-07-01', '2025-06-01', freq='4MS')
windows = [(s, s + pd.DateOffset(months=12)) for s in starts]
windows = [(s, e) for s, e in windows if e <= df.index[-1]]


def _job(args):
    wi, which = args
    s, e = windows[wi]
    d = df.loc[s:e]
    if len(d) < 50:
        return None
    st = champ if which == 'LAD' else noladder
    try:
        return (wi, which) + metrics(st, d)
    except Exception:
        return None


if __name__ == '__main__':
    jobs = [(wi, which) for wi in range(len(windows)) for which in ('LAD', 'NOLAD')]
    with mp.Pool(min(len(jobs), 24)) as pool:
        res = [r for r in pool.map(_job, jobs) if r]
    by = {(r[0], r[1]): r[2:] for r in res}
    print(f"=== PROMOTED champion: gamma-wt [14,30] ladder vs no-ladder, {len(windows)} rolling 12mo windows ===")
    print(f"  {'window':<22}{'LAD Sh':>8}{'NOLAD Sh':>10}{'ΔSh':>7}{'LAD MDD':>9}{'NOLAD MDD':>11}")
    sh_win = mdd_win = n = 0
    for wi in range(len(windows)):
        L = by.get((wi, 'LAD')); N = by.get((wi, 'NOLAD'))
        if not L or not N:
            continue
        s, e = windows[wi]
        dsh = L[1] - N[1]
        sh_win += dsh >= -0.02
        mdd_win += L[2] >= N[2] - 0.5
        n += 1
        print(f"  {str(s.date())+'..'+str(e.date()):<22}{L[1]:>8.2f}{N[1]:>10.2f}{dsh:>+7.2f}{L[2]:>8.1f}%{N[2]:>10.1f}%")
    print(f"\n  ladder Sharpe >= no-ladder: {sh_win}/{n} windows | MDD not worse: {mdd_win}/{n}")
    print("DONE", flush=True)
