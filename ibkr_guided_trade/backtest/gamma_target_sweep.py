"""DELTA+GAMMA targeted accumulation — sweep the gamma target. γ=0 ⇒ pure shares (flat delta);
more-negative ⇒ more puts carry the negative-gamma budget. Tests whether ANY non-zero gamma target
(e.g. the −0.03 that matches the target-delta curve slope) beats pure shares, verified on the sealed TEST."""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
BASE = STRATEGIES['regime_wheel_boxx_greeks']
V = {
    'shares (γ=0 base)':       dict(BASE),
    'Δ+γ  γ=0':                {**BASE, 'reaccum_delta_gamma': True, 'target_gamma_per_nav': 0.0},
    'Δ+γ  γ=-0.03 (curve)':    {**BASE, 'reaccum_delta_gamma': True, 'target_gamma_per_nav': -0.03},
    'Δ+γ  γ=-0.06':            {**BASE, 'reaccum_delta_gamma': True, 'target_gamma_per_nav': -0.06},
    'Δ+γ  γ=-0.12':            {**BASE, 'reaccum_delta_gamma': True, 'target_gamma_per_nav': -0.12},
}
def _load():
    return precompute_factor_z(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache','master_dataset.csv'),index_col=0,parse_dates=True)).dropna(subset=['UNG'])
def metrics(strat, df, nav0=100000):
    hist, trades = run_strategy_simple(df, strat, nav0, 0)
    hist = hist.set_index(pd.to_datetime(hist['date'])); nav=hist['nav']; r=nav.pct_change().dropna()
    yrs=(df.index[-1]-df.index[0]).days/365.25
    t=trades['type'].astype(str)
    return (((nav.iloc[-1]/nav0)**(1/yrs)-1)*100, r.mean()/(r.std()+1e-9)*math.sqrt(252),
            ((nav-nav.cummax())/nav.cummax()*100).min(), hist['net_gamma'].mean(),
            int((t=='Z_TARGET_ADD_PUTS').sum()), int((t=='Z_TARGET_ADD').sum()))
def _job(a):
    n,s,w=a; d=_load(); d=d.loc[TRAIN_START:TRAIN_END] if w=='TRAIN' else d.loc[TEST_START:TEST_END]
    return (n,w)+metrics(s,d)
if __name__=='__main__':
    jobs=[(n,s,w) for n,s in V.items() for w in ('TRAIN','TEST')]
    with mp.Pool(6) as pool: res=pool.map(_job,jobs)
    res.sort(key=lambda r:(list(V).index(r[0]),0 if r[1]=='TRAIN' else 1))
    print(f"  {'variant':<22}{'win':<7}{'ann':>7}{'Sh':>6}{'MaxDD':>7}{'avgγ':>8}{'puAdd':>7}{'shAdd':>7}")
    for n,w,a,s,m,gm,np_,ns in res: print(f"  {n:<22}{w:<7}{a:>6.1f}%{s:>6.2f}{m:>6.1f}%{gm:>8.0f}{np_:>7d}{ns:>7d}")
    print("DONE",flush=True)
