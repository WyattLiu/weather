"""GAMMA→DELTA: hedge the statistical FORWARD delta (real drift) vs the BS instantaneous delta.
Does controlling where the book actually drifts (puts assign into decline) improve risk control?"""
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
variants = {'BS instantaneous delta (current)': dict(base),
            'STATISTICAL forward delta': {**base, 'stat_delta_hedge': True}}
print("=== GAMMA→DELTA: statistical forward-delta hedge vs BS instantaneous ===")
print(f"  {'variant':<34}{'win':<7}{'ann':>8}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}")
print('  ' + '-'*73)
for name, st in variants.items():
    for lbl, d in (('TRAIN', tr), ('TEST', te)):
        s = measure_period(st, d, 100000)
        if s and 'error' not in s:
            print(f"  {name:<34}{lbl:<7}{s['ann']:>7.1f}%{s['sharpe']:>8.2f}{s['mdd']:>7.1f}%{s['n_trades']:>8}", flush=True)
        else:
            print(f"  {name:<34}{lbl:<7} ERR", flush=True)
print("DONE", flush=True)
