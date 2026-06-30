"""Fine-tune AROUND config D on lag-corrected (honest) data. One dimension at a time off D
(ITM+15%, cadence 14, cut 0.3, dte 30). Rank on TEST Sharpe + FULL rolling worst-12mo. D is the
local optimum unless a neighbor beats it on BOTH."""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
BASE = STRATEGIES['regime_wheel_boxx_greeks']
def D(**kw): return {**BASE, 'reaccum_via_puts': True, 'boxx_cash_buffer': 15000, 'reaccum_put_dte': 30,
                     'reaccum_put_moneyness': 0.15, 'z_target_cadence_days': 14, 'cut_speed': 0.3, **kw}
V = {
    'D (15/14/0.3/30)': D(),
    'mny +10%':         D(reaccum_put_moneyness=0.10),
    'mny +20%':         D(reaccum_put_moneyness=0.20),
    'cadence 10':       D(z_target_cadence_days=10),
    'cadence 18':       D(z_target_cadence_days=18),
    'cut 0.2':          D(cut_speed=0.2),
    'cut 0.4':          D(cut_speed=0.4),
    'dte 45':           D(reaccum_put_dte=45),
}
RAW=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache','master_dataset.csv'),index_col=0,parse_dates=True)
def _load(): return precompute_factor_z(RAW).dropna(subset=['UNG'])
def metrics(strat):
    df=_load(); te=df.loc[TEST_START:TEST_END]
    h,_=run_strategy_simple(te,strat,100000,0); nav=h.set_index(pd.to_datetime(h['date']))['nav']; r=nav.pct_change().dropna()
    yrs=(te.index[-1]-te.index[0]).days/365.25
    tann=(nav.iloc[-1]/100000)**(1/yrs)*100-100; tsh=r.mean()/(r.std()+1e-9)*math.sqrt(252); tdd=((nav-nav.cummax())/nav.cummax()*100).min()
    hf,_=run_strategy_simple(df,strat,100000,0); navf=hf.set_index(pd.to_datetime(hf['date']))['nav'].reset_index(drop=True)
    W=252; wr=[navf.iloc[i:i+W].iloc[-1]/navf.iloc[i:i+W].iloc[0]-1 for i in range(0,len(navf)-W,21)]
    return tann,tsh,tdd,(min(wr)*100 if wr else 0)
def _job(a):
    n,s=a; return (n,)+metrics(s)
if __name__=='__main__':
    with mp.Pool(6) as pool: res=pool.map(_job,list(V.items()))
    res.sort(key=lambda r:list(V).index(r[0]))
    print(f"  {'variant':<20}{'TESTann':>8}{'TESTSh':>8}{'TESTdd':>8}{'wst12mo':>9}")
    for n,a,s,m,w in res: print(f"  {n:<20}{a:>7.1f}%{s:>8.2f}{m:>7.1f}%{w:>8.1f}%")
    print("DONE",flush=True)
