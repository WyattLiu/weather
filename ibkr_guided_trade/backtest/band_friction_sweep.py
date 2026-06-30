"""Reduce churn/friction by tolerating drift: sweep the hysteresis band width (delta_band_k). Wider band
→ act only on bigger drift → fewer trades → less spread friction, but looser curve tracking ('misses').
Find the k that maximizes NET return (friction already in NAV via SPREAD_*). Continuous + discrete+band."""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
BASE = STRATEGIES['regime_wheel_boxx_greeks']
def cont(k): return {**BASE, 'z_target_cadence_days': 1, 'delta_band_sizing': True, 'delta_band_k': k, 'cut_speed': 0.3}
def disc_band(k): return {**BASE, 'delta_band_sizing': True, 'delta_band_k': k}   # discrete cadence + tolerance band
V = {
    'discrete (no band)':  dict(BASE),
    'cont k=1.0':          cont(1.0),
    'cont k=2.0':          cont(2.0),
    'cont k=3.0':          cont(3.0),
    'discrete+band k=2.0': disc_band(2.0),
    'discrete+band k=3.0': disc_band(3.0),
}
def _load():
    return precompute_factor_z(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache','master_dataset.csv'),index_col=0,parse_dates=True)).dropna(subset=['UNG'])
def metrics(strat, df, nav0=100000):
    hist,trades=run_strategy_simple(df, strat, nav0, 0)
    hist=hist.set_index(pd.to_datetime(hist['date'])); nav=hist['nav']; r=nav.pct_change().dropna()
    yrs=(df.index[-1]-df.index[0]).days/365.25
    return (((nav.iloc[-1]/nav0)**(1/yrs)-1)*100, r.mean()/(r.std()+1e-9)*math.sqrt(252), ((nav-nav.cummax())/nav.cummax()*100).min(), len(trades))
def _job(a):
    n,s,w=a; d=_load(); d=d.loc[TRAIN_START:TRAIN_END] if w=='TRAIN' else d.loc[TEST_START:TEST_END]
    return (n,w)+metrics(s,d)
if __name__=='__main__':
    jobs=[(n,s,w) for n,s in V.items() for w in ('TRAIN','TEST')]
    with mp.Pool(6) as pool: res=pool.map(_job,jobs)
    res.sort(key=lambda r:(list(V).index(r[0]),0 if r[1]=='TRAIN' else 1))
    print(f"  {'tolerance':<22}{'win':<7}{'ann':>7}{'Sh':>6}{'MaxDD':>7}{'trades':>8}")
    for n,w,a,s,m,t in res: print(f"  {n:<22}{w:<7}{a:>6.1f}%{s:>6.2f}{m:>6.1f}%{t:>8d}")
    print("DONE",flush=True)
