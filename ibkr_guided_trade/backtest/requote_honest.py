"""Re-quote the champion under the honest WS cost model: $0 commission + spread-in-engine + MODELED
early assignment (deep-ITM |delta|>0.99 + extrinsic<$0.02). Isolate the early-assign impact and count
how often it fires for UNG."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import measure_period, TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache','master_dataset.csv'),
                 index_col=0, parse_dates=True)
df = precompute_factor_z(df).dropna(subset=['UNG'])
champ = STRATEGIES['regime_wheel_boxx_greeks']
# count early-assign events over the full sample
_, tr = run_strategy_simple(df, champ, 100000, 0)
t = pd.DataFrame(tr)
ea_p = (t['type']=='PUT_EARLY_ASSIGN').sum() if 'type' in t else 0
ea_c = (t['type']=='CALL_EARLY_ASSIGN').sum() if 'type' in t else 0
ap = (t['type']=='PUT_ASSIGN').sum(); ac = (t['type']=='CALL_ASSIGN').sum()
print(f"=== Early-assignment events (full sample, champion) ===")
print(f"  PUT_EARLY_ASSIGN {ea_p} (vs {ap} at-expiry PUT_ASSIGN) | CALL_EARLY_ASSIGN {ea_c} (vs {ac} CALL_ASSIGN)")
print()
trd, ted = df.loc[TRAIN_START:TRAIN_END], df.loc[TEST_START:TEST_END]
print("=== Honest re-quote: champion (WS $0 comm) early-assign ON vs OFF ===")
print(f"  {'variant':<32}{'win':<7}{'ann':>8}{'Sharpe':>8}{'MaxDD':>8}")
for name, st in [('early-assign ON (honest)', champ),
                 ('early-assign OFF', {**champ, 'model_early_assign': False})]:
    for lbl, d in (('TRAIN', trd), ('TEST', ted)):
        s = measure_period(st, d, 100000)
        print(f"  {name:<32}{lbl:<7}{s['ann']:>7.1f}%{s['sharpe']:>8.2f}{s['mdd']:>7.1f}%", flush=True)
print("DONE", flush=True)
