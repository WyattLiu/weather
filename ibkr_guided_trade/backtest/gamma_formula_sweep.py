"""What's a GOOD target-gamma formula? Sweep: constant (×NAV/spot) vs DYNAMIC curve-slope
(γ = −target/spot, so book Δ self-tracks the target curve) vs scaled curve. All Δ+γ mode + buffer 15k
(needs the share leg). Report return / Sharpe / MaxDD / realized avg gamma — find the best frontier point."""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
BASE = STRATEGIES['regime_wheel_boxx_greeks']
def dg(**kw): return {**BASE, 'reaccum_delta_gamma': True, 'boxx_cash_buffer': 15000, 'reaccum_put_dte': 30, 'reaccum_put_moneyness': 0.05, **kw}
V = {
    'shares (γ=0)':         dict(BASE),
    'const γ=-0.02':        dg(target_gamma_per_nav=-0.02),
    'const γ=-0.04':        dg(target_gamma_per_nav=-0.04),
    'CURVE (−tgt/spot)':    dg(gamma_target_mode='curve'),
    'CURVE ×0.5':           dg(gamma_target_mode='curve_k', gamma_curve_k=0.5),
    'CURVE ×1.5':           dg(gamma_target_mode='curve_k', gamma_curve_k=1.5),
}
def _load():
    return precompute_factor_z(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache','master_dataset.csv'),index_col=0,parse_dates=True)).dropna(subset=['UNG'])
def metrics(strat, df, nav0=100000):
    hist,_=run_strategy_simple(df, strat, nav0, 0)
    hist=hist.set_index(pd.to_datetime(hist['date'])); nav=hist['nav']; r=nav.pct_change().dropna()
    yrs=(df.index[-1]-df.index[0]).days/365.25
    return (((nav.iloc[-1]/nav0)**(1/yrs)-1)*100, r.mean()/(r.std()+1e-9)*math.sqrt(252), ((nav-nav.cummax())/nav.cummax()*100).min(), hist['net_gamma'].mean())
def _job(a):
    n,s,w=a; d=_load(); d=d.loc[TRAIN_START:TRAIN_END] if w=='TRAIN' else d.loc[TEST_START:TEST_END]
    return (n,w)+metrics(s,d)
if __name__=='__main__':
    jobs=[(n,s,w) for n,s in V.items() for w in ('TRAIN','TEST')]
    with mp.Pool(6) as pool: res=pool.map(_job,jobs)
    res.sort(key=lambda r:(list(V).index(r[0]),0 if r[1]=='TRAIN' else 1))
    print(f"  {'gamma formula':<20}{'win':<7}{'ann':>7}{'Sh':>6}{'MaxDD':>7}{'avgγ':>9}")
    for n,w,a,s,m,g in res: print(f"  {n:<20}{w:<7}{a:>6.1f}%{s:>6.2f}{m:>6.1f}%{g:>9.0f}")
    print("DONE",flush=True)
