"""Explore: sell ITM put ≡ buy shares + covered call (parity). DEEPER ITM ⇒ more share-like (delta→1,
less short-vol) ⇒ higher Sharpe, while margin-financed (BOXX kept). Sweep moneyness; does deep-ITM
maintain Sharpe toward shares' 2.01 — and does it survive buffer:0 (all-BOXX) or need a buffer?"""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
BASE = STRATEGIES['regime_wheel_boxx_greeks']
def pv(m, buf): return {**BASE, 'boxx_cash_buffer': buf, 'reaccum_via_puts': True, 'reaccum_put_dte': 30, 'reaccum_put_moneyness': m}
V = {
    'shares (Sharpe ref)':   dict(BASE),
    'puts buf0  +5%':        pv(0.05, 0),
    'puts buf0  +15%':       pv(0.15, 0),
    'puts buf0  +25%':       pv(0.25, 0),
    'puts buf15k +15%':      pv(0.15, 15000),
    'puts buf15k +25%':      pv(0.25, 15000),
}
def _load():
    return precompute_factor_z(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache','master_dataset.csv'),index_col=0,parse_dates=True)).dropna(subset=['UNG'])
def metrics(strat, df, nav0=100000):
    hist,_=run_strategy_simple(df, strat, nav0, 0)
    hist=hist.set_index(pd.to_datetime(hist['date'])); nav=hist['nav']; r=nav.pct_change().dropna()
    yrs=(df.index[-1]-df.index[0]).days/365.25
    return (((nav.iloc[-1]/nav0)**(1/yrs)-1)*100, r.mean()/(r.std()+1e-9)*math.sqrt(252), ((nav-nav.cummax())/nav.cummax()*100).min())
def _job(a):
    n,s=a; return (n,)+metrics(s,_load().loc[TEST_START:TEST_END])
if __name__=='__main__':
    with mp.Pool(6) as pool: res=pool.map(_job, list(V.items()))
    res.sort(key=lambda r:list(V).index(r[0]))
    print(f"  {'variant':<22}{'TEST ann':>9}{'Sharpe':>8}{'MaxDD':>8}")
    for n,a,s,m in res: print(f"  {n:<22}{a:>8.1f}%{s:>8.2f}{m:>7.1f}%")
    print("DONE",flush=True)
