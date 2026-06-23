"""Validate the proportional-cap change: champion with NEW max_short_pct_nav=0.085 vs OLD fixed
max_short_per_strike=10, on TRAIN and TEST windows. Confirms the relative cap doesn't degrade."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import measure_period, TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'master_dataset.csv'),
                index_col=0, parse_dates=True)
df = precompute_factor_z(df).dropna(subset=['UNG'])
tr, te = df.loc[TRAIN_START:TRAIN_END], df.loc[TEST_START:TEST_END]

base = STRATEGIES['regime_wheel_boxx_greeks']
variants = {'PROPORTIONAL 0.085·NAV (new)': dict(base),
            'FIXED per_strike=10 (old)': {**base, 'max_short_pct_nav': None}}
print(f"=== CAP VALIDATION — champion walk-forward (cash $100k) ===")
print(f"  TRAIN {TRAIN_START}→{TRAIN_END} | TEST {TEST_START}→{TEST_END} (SEALED)\n")
print(f"  {'variant':<30}{'window':<7}{'ann':>8}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}")
print('  ' + '-'*69)
for name, strat in variants.items():
    for lbl, d in (('TRAIN', tr), ('TEST', te)):
        s = measure_period(strat, d, 100000)
        if s and 'error' not in s:
            print(f"  {name:<30}{lbl:<7}{s['ann']:>7.1f}%{s['sharpe']:>8.2f}{s['mdd']:>7.1f}%{s['n_trades']:>8}", flush=True)
        else:
            print(f"  {name:<30}{lbl:<7} ERROR {s.get('error') if s else 'none'}", flush=True)
print("DONE", flush=True)
