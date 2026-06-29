"""Sweep z_target_cadence_days: faster cadence = quicker convergence to the share target (captures
cheap-level accumulation sooner) vs more turnover/whipsaw. Champion baseline is 21d."""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
def metrics(strat, df, nav0=100000):
    hist, trades = run_strategy_simple(df, strat, nav0, 0)
    hist = hist.set_index(pd.to_datetime(hist['date'])); nav = hist['nav']; r = nav.pct_change().dropna()
    yrs = (df.index[-1]-df.index[0]).days/365.25
    ann = ((nav.iloc[-1]/nav0)**(1/yrs)-1)*100
    sh = r.mean()/(r.std()+1e-9)*math.sqrt(252)
    mdd = ((nav-nav.cummax())/nav.cummax()*100).min()
    return ann, sh, mdd, len(trades)
df = precompute_factor_z(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache','master_dataset.csv'),
     index_col=0, parse_dates=True)).dropna(subset=['UNG'])
tr, te = df.loc[TRAIN_START:TRAIN_END], df.loc[TEST_START:TEST_END]
base = STRATEGIES['regime_wheel_boxx_greeks']
V = {f'cadence_{c}': {**base, 'z_target_cadence_days': c} for c in (3,5,10,14,21)}
def _job(a):
    name, st, win = a; d = tr if win=='TRAIN' else te
    return (name, win)+metrics(st, d)
if __name__=='__main__':
    jobs=[(n,s,w) for n,s in V.items() for w in ('TRAIN','TEST')]
    with mp.Pool(5) as pool: res=pool.map(_job, jobs)
    res.sort(key=lambda r:(list(V).index(r[0]), 0 if r[1]=='TRAIN' else 1))
    print(f"  {'cadence':<14}{'win':<7}{'ann':>8}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}")
    for n,w,a,s,m,t in res:
        star=' <- champion' if n=='cadence_21' and w=='TEST' else ''
        print(f"  {n:<14}{w:<7}{a:>7.1f}%{s:>8.2f}{m:>7.1f}%{t:>8d}{star}")
    print("DONE", flush=True)
