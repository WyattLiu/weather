"""How much LIVE leakage does Friday re-accumulation close? For every put the champion writes on a
called-away bar (same-bar re-accum), compute the PURE theta it earns over the gap to the next bar
(hold spot constant → isolate time-decay, the systematic edge; the spot move is a wash). That decay
is exactly what live FORGOES if it waits to the next session. Sum, annualize, as % of NAV."""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple, bs_put

IV = 0.55
df = precompute_factor_z(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache',
                         'master_dataset.csv'), index_col=0, parse_dates=True)).dropna(subset=['UNG'])
ung = df['UNG']
dates = list(df.index)
hist, trades = run_strategy_simple(df, STRATEGIES['regime_wheel_boxx_greeks'], 100000, 0)
t = pd.DataFrame(trades)
t['date'] = pd.to_datetime(t['date'])

# called-away bars
ca_dates = set(t[t['type'].astype(str).str.upper().str.contains('CALL') &
                 t['type'].astype(str).str.upper().str.contains('ASSIGN|CALLED|AWAY')]['date'])
opens = t[t['type'].astype(str).str.upper().str.startswith('OPEN_PUT')]
leak = 0.0
n_legs = 0
for _, o in opens.iterrows():
    if o['date'] not in ca_dates:
        continue
    K = float(o.get('K') or 0); dte = int(o.get('dte') or 30); q = int(o.get('qty') or 0)
    if K <= 0 or q <= 0:
        continue
    try:
        i = dates.index(o['date'])
    except ValueError:
        continue
    if i + 1 >= len(dates):
        continue
    gap_days = (dates[i + 1] - dates[i]).days            # 1 for weekday, 3 over a weekend
    S = float(ung.iloc[i])
    # pure theta = same spot, less time (isolate decay; spot move is a wash)
    decay = bs_put(S, K, dte / 365, IV) - bs_put(S, K, max(dte - gap_days, 0.5) / 365, IV)
    leak += max(0.0, decay) * 100 * q
    n_legs += 1

yrs = (df.index[-1] - df.index[0]).days / 365.25
print("=== Weekend-theta leakage closed by Friday re-accumulation ===")
print(f"  same-bar re-accum put legs: {n_legs} over {yrs:.1f}y")
print(f"  total pure theta forgone if sold next-session: ${leak:,.0f}")
print(f"  = ${leak/yrs:,.0f}/yr = {leak/yrs/100000*100:.2f}% of NAV/yr (the live leakage now closed)")
print(f"  NOTE: closes a live<->backtest gap; does NOT change the backtest frontier (already captured).")
print("DONE", flush=True)
