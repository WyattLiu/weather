"""Measure the champion's PROVEN monthly premium-harvest (theta) rate from the backtest, as % of NAV
— the realistic, sustained rate (book continuously re-sold), to anchor the live dashboard's monthly
theta instead of the naive daily-snapshot×30."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'master_dataset.csv'),
                index_col=0, parse_dates=True)
df = precompute_factor_z(df).dropna(subset=['UNG'])
hist, trades = run_strategy_simple(df, STRATEGIES['regime_wheel_boxx_greeks'], 100000, 0)
t = pd.DataFrame(trades)
n_months = (df.index[-1] - df.index[0]).days / 30.44
nav0 = 100000
# gross premium collected (credits on opens) and net option premium P&L (credits - buybacks)
opens = t[t['type'].isin(['OPEN_PUT', 'OPEN_CALL'])] if 'type' in t else t.iloc[0:0]
gross_credit = opens['credit'].sum() if 'credit' in opens else 0
# net realized option premium = sum of all credits + pnl on option-type trades (excl share trades)
opt_types = ['OPEN_PUT','OPEN_CALL','CLOSE_PUT','CLOSE_CALL','TP_PUT','TP_CALL','ROLL','BTC_PUT','BTC_CALL','EXPIRE_PUT','EXPIRE_CALL']
opt = t[t['type'].astype(str).str.contains('PUT|CALL|ROLL', regex=True)] if 'type' in t else t.iloc[0:0]
net_opt_pnl = opt['pnl'].sum() if 'pnl' in opt else 0
net_credit = (opt['credit'].sum() if 'credit' in opt else 0)
print(f"=== champion premium-harvest rate (backtest {df.index[0].date()}..{df.index[-1].date()}, {n_months:.0f} mo) ===")
print(f"  gross premium COLLECTED: ${gross_credit:,.0f}  = ${gross_credit/n_months:,.0f}/mo  = {gross_credit/n_months/nav0*100:.2f}% of NAV/mo")
print(f"  net option P&L (credits−buybacks, gross of assignment): ${net_opt_pnl:,.0f} = ${net_opt_pnl/n_months:,.0f}/mo = {net_opt_pnl/n_months/nav0*100:.2f}%/mo")
final_nav = hist['nav'].iloc[-1] if hasattr(hist,'columns') and 'nav' in hist else (hist[-1] if hasattr(hist,'__len__') else nav0)
print(f"  total return: ${final_nav-nav0:,.0f} over {n_months:.0f}mo = {(final_nav/nav0-1)/n_months*100:.2f}%/mo total")
print(f"\n  => REALISTIC monthly theta ≈ {net_opt_pnl/n_months/nav0*100:.2f}% of NAV (net option premium, backtest-proven)")
print("DONE", flush=True)
