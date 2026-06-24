"""STAGE 4 frontier: champion with the NOTIONAL per-strike cap vs the gamma-aware SCENARIO-Δ cap
(probability-weighted, DTE-aware) at several targets. TRAIN/TEST walk-forward. Does pricing the
concentration by EXPECTED assignment-delta beat the flat notional cap on CAGR/Sharpe/MaxDD?"""
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
variants = {
    'NOTIONAL cap 0.085·NAV (current)': dict(base),
    'SCENARIO-Δ target 0.15·NAV/spot': {**base, 'scenario_delta_target': 0.15},
    'SCENARIO-Δ target 0.25·NAV/spot': {**base, 'scenario_delta_target': 0.25},
    'SCENARIO-Δ target 0.40·NAV/spot': {**base, 'scenario_delta_target': 0.40},
}
print("=== STAGE 4: gamma-aware SCENARIO-Δ cap vs NOTIONAL cap (champion walk-forward) ===")
print(f"  TRAIN {TRAIN_START}→{TRAIN_END} | TEST {TEST_START}→{TEST_END} (sealed)\n")
print(f"  {'variant':<34}{'win':<7}{'ann':>8}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}")
print('  ' + '-'*73)
for name, st in variants.items():
    for lbl, d in (('TRAIN', tr), ('TEST', te)):
        s = measure_period(st, d, 100000)
        if s and 'error' not in s:
            print(f"  {name:<34}{lbl:<7}{s['ann']:>7.1f}%{s['sharpe']:>8.2f}{s['mdd']:>7.1f}%{s['n_trades']:>8}", flush=True)
        else:
            print(f"  {name:<34}{lbl:<7} ERR {s.get('error') if s else '?'}", flush=True)
print("DONE", flush=True)
