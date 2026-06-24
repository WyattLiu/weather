"""Which (share-target mode × DTE-ladder) best improves the FRONTIER while ENSURING SMOOTHNESS?
Smoothness = annualized daily NAV vol + worst single-day drop (a gamma cliff at a single concentrated
expiry shows up as a big one-day move). Isolates the two levers: forward-share early-reaccum, and the
DTE-ladder (deterministic per_dte_qty = put_qty // n_dtes — SAME rule in backtest and live)."""
import sys
import os
import math
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
L3 = [7, 14, 30]
L4 = [7, 14, 30, 45]
variants = {
    'champion: reactive | [30] single':        dict(base),
    'reactive | ladder [7,14,30]':             {**base, 'dte_ladder': L3},
    'FWD early-reaccum | [30] single':         {**base, 'stat_share_target': True},
    'FWD early-reaccum | ladder [7,14,30]':    {**base, 'stat_share_target': True, 'dte_ladder': L3},
    'FWD early-reaccum | ladder [7,14,30,45]': {**base, 'stat_share_target': True, 'dte_ladder': L4},
}
print("=== FRONTIER + SMOOTHNESS: share-target mode x DTE-ladder ===")
print("  (smoothness = lower daily-vol + smaller worst-day; both = fewer gamma cliffs)\n")
print(f"  {'variant':<40}{'win':<7}{'ann':>7}{'Sh':>6}{'MaxDD':>8}{'vol':>7}{'worstD':>8}{'trd':>6}")
print('  ' + '-' * 89)
for name, st in variants.items():
    for lbl, d in (('TRAIN', tr), ('TEST', te)):
        try:
            a, s, m, v, w, n = metrics(st, d)
            print(f"  {name:<40}{lbl:<7}{a:>6.1f}%{s:>6.2f}{m:>7.1f}%{v:>6.1f}%{w:>7.1f}%{n:>6}", flush=True)
        except Exception as e:
            print(f"  {name:<40}{lbl:<7} ERR {str(e)[:30]}", flush=True)
print("DONE", flush=True)
