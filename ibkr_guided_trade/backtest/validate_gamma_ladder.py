"""GAMMA-WEIGHTED ladder (contracts proportional to 1/gamma so each expiry carries EQUAL gamma),
skipping the high-gamma 7-DTE bucket. Isolates: (a) DROP 7-DTE, (b) GAMMA-WEIGHT on top.
Scores FRONTIER (ann/Sharpe/MaxDD) + SMOOTHNESS (daily vol + worst single-day). TRAIN/TEST.
PARALLEL: each (variant, window) run is independent → one process each (120 cores available)."""
import sys
import os
import math
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple


def metrics(strat, df, nav0=100000):
    hist, trades = run_strategy_simple(df, strat, nav0, 0)
    hist = hist.set_index(pd.to_datetime(hist['date']))
    nav = hist['nav']
    rets = nav.pct_change().dropna()
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    ann = ((nav.iloc[-1] / nav0) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    vol = rets.std() * math.sqrt(252) * 100
    sh = rets.mean() / (rets.std() + 1e-9) * math.sqrt(252)
    mdd = ((nav - nav.cummax()) / nav.cummax() * 100).min()
    worst = rets.min() * 100
    return ann, sh, mdd, vol, worst, len(trades)


df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'master_dataset.csv'),
                 index_col=0, parse_dates=True)
df = precompute_factor_z(df).dropna(subset=['UNG'])
tr, te = df.loc[TRAIN_START:TRAIN_END], df.loc[TEST_START:TEST_END]
base = STRATEGIES['regime_wheel_boxx_greeks']
variants = {
    'champion [30] single':       dict(base),
    'even [14,30] (drop 7)':      {**base, 'dte_ladder': [14, 30]},
    'GAMMA-wt [14,30]':           {**base, 'dte_ladder': [14, 30], 'gamma_weighted_ladder': True},
    'GAMMA-wt [14,30,45]':        {**base, 'dte_ladder': [14, 30, 45], 'gamma_weighted_ladder': True},
    'GAMMA-wt [21,45]':           {**base, 'dte_ladder': [21, 45], 'gamma_weighted_ladder': True},
}


def _job(args):
    name, st, win = args
    d = tr if win == 'TRAIN' else te
    try:
        return (name, win) + metrics(st, d)
    except Exception as e:
        return (name, win, 'ERR', str(e)[:40])


if __name__ == '__main__':
    jobs = [(name, st, win) for name, st in variants.items() for win in ('TRAIN', 'TEST')]
    with mp.Pool(min(len(jobs), 24)) as pool:
        res = pool.map(_job, jobs)
    order = list(variants.keys())
    res.sort(key=lambda r: (order.index(r[0]), 0 if r[1] == 'TRAIN' else 1))
    print("=== GAMMA-WEIGHTED ladder (no 7-DTE): frontier + smoothness [PARALLEL] ===")
    print("  (smoothness = lower daily-vol + smaller worst-day = fewer gamma cliffs)\n")
    print(f"  {'variant':<28}{'win':<7}{'ann':>7}{'Sh':>6}{'MaxDD':>8}{'vol':>7}{'worstD':>8}{'trd':>6}")
    print('  ' + '-' * 77)
    for r in res:
        if len(r) == 8:
            name, win, a, s, m, v, w, n = r
            print(f"  {name:<28}{win:<7}{a:>6.1f}%{s:>6.2f}{m:>7.1f}%{v:>6.1f}%{w:>7.1f}%{n:>6}")
        else:
            print(f"  {r[0]:<28}{r[1]:<7} {r[2]} {r[3]}")
    print("DONE", flush=True)
