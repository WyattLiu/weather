"""FULL walk-forward on the candidate: CURVE×1.5 gamma + cut_speed 0.3 (Δ+γ blend, CAD-funded buffer).
TRAIN/TEST/FULL + rolling worst-12mo. Promote only if TEST Sharpe holds ~2.0 AND worst-window is sane."""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
BASE = STRATEGIES['regime_wheel_boxx_greeks']
CAND = {**BASE, 'reaccum_delta_gamma': True, 'boxx_cash_buffer': 15000, 'reaccum_put_dte': 30,
        'reaccum_put_moneyness': 0.05, 'gamma_target_mode': 'curve_k', 'gamma_curve_k': 1.5, 'cut_speed': 0.3}
V = {'shares champion': dict(BASE), 'CURVE1.5 + cut0.3': CAND}
def _load():
    return precompute_factor_z(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache','master_dataset.csv'),index_col=0,parse_dates=True)).dropna(subset=['UNG'])
def _stats(nav, nav0, yrs):
    r=nav.pct_change().dropna()
    return (((nav.iloc[-1]/nav0)**(1/yrs)-1)*100, r.mean()/(r.std()+1e-9)*math.sqrt(252), ((nav-nav.cummax())/nav.cummax()*100).min())
def metrics(strat, df):
    hist,_=run_strategy_simple(df, strat, 100000, 0)
    hist=hist.set_index(pd.to_datetime(hist['date'])); nav=hist['nav'].reset_index(drop=True)
    yrs=(df.index[-1]-df.index[0]).days/365.25
    full=_stats(nav,100000,yrs); W=252; wr=[]; wd=[]
    for i in range(0,len(nav)-W,21):
        seg=nav.iloc[i:i+W]; wr.append(seg.iloc[-1]/seg.iloc[0]-1); wd.append(((seg-seg.cummax())/seg.cummax()).min())
    return full+((min(wr)*100 if wr else 0),(min(wd)*100 if wd else 0))
def _job(a):
    n,s,w=a; df=_load(); d=df.loc[TRAIN_START:TRAIN_END] if w=='TRAIN' else (df.loc[TEST_START:TEST_END] if w=='TEST' else df)
    return (n,w)+metrics(s,d)
if __name__=='__main__':
    jobs=[(n,s,w) for n,s in V.items() for w in ('TRAIN','TEST','FULL')]
    with mp.Pool(6) as pool: res=pool.map(_job,jobs)
    res.sort(key=lambda r:(list(V).index(r[0]),{'TRAIN':0,'TEST':1,'FULL':2}[r[1]]))
    print(f"  {'config':<20}{'win':<6}{'ann':>7}{'Sh':>6}{'MaxDD':>7}{'wst12mo-ret':>12}{'wst12mo-dd':>11}")
    for n,w,a,s,m,wr,wd in res: print(f"  {n:<20}{w:<6}{a:>6.1f}%{s:>6.2f}{m:>6.1f}%{wr:>11.1f}%{wd:>10.1f}%")
    print("DONE",flush=True)
