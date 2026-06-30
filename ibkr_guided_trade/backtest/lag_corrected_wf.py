"""Honest re-validation: LAG the weekly storage to its RELEASE date (Friday report-week → Thursday
release ≈ 5 trading days later) to remove the ~6-day look-ahead, then re-run champion + config D.
storage_surprise_z is recomputed on the lagged storage. Shows how much the look-ahead inflated results."""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple
RAW = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache','master_dataset.csv'), index_col=0, parse_dates=True)
def prep(lag):
    d = RAW.copy()
    if lag:
        for c in ('eia_storage_weekly','days_supply'):
            if c in d.columns: d[c] = d[c].shift(5)        # weekly storage → release (~5 trading days)
        for c in ('eia_production','eia_consumption'):
            if c in d.columns: d[c] = d[c].shift(21)        # monthly → ~1mo release lag
    return precompute_factor_z(d).dropna(subset=['UNG'])
CHAMP = STRATEGIES['regime_wheel_boxx_greeks']
D = {**CHAMP, 'reaccum_via_puts': True, 'boxx_cash_buffer': 15000, 'reaccum_put_dte': 30,
     'reaccum_put_moneyness': 0.15, 'z_target_cadence_days': 14, 'cut_speed': 0.3}
V = {'champion (shares)': CHAMP, 'config D (puts)': D}
def metrics(strat, df, nav0=100000):
    hist,_=run_strategy_simple(df, strat, nav0, 0)
    hist=hist.set_index(pd.to_datetime(hist['date'])); nav=hist['nav']; r=nav.pct_change().dropna()
    yrs=(df.index[-1]-df.index[0]).days/365.25
    return (((nav.iloc[-1]/nav0)**(1/yrs)-1)*100, r.mean()/(r.std()+1e-9)*math.sqrt(252), ((nav-nav.cummax())/nav.cummax()*100).min())
def _job(a):
    n,s,lagname,win=a; lag=(lagname=='LAG-FIXED'); d=prep(lag)
    d=d.loc[TRAIN_START:TRAIN_END] if win=='TRAIN' else d.loc[TEST_START:TEST_END]
    return (n,lagname,win)+metrics(s,d)
if __name__=='__main__':
    jobs=[(n,s,lg,w) for n,s in V.items() for lg in ('original(look-ahead)','LAG-FIXED') for w in ('TRAIN','TEST')]
    with mp.Pool(6) as pool: res=pool.map(_job,jobs)
    order={('champion (shares)','original(look-ahead)'):0,('champion (shares)','LAG-FIXED'):1,('config D (puts)','original(look-ahead)'):2,('config D (puts)','LAG-FIXED'):3}
    res.sort(key=lambda r:(order[(r[0],r[1])],0 if r[2]=='TRAIN' else 1))
    print(f"  {'config':<18}{'data':<22}{'win':<7}{'ann':>7}{'Sh':>6}{'MaxDD':>7}")
    for n,lg,w,a,s,m in res: print(f"  {n:<18}{lg:<22}{w:<7}{a:>6.1f}%{s:>6.2f}{m:>6.1f}%")
    print("DONE",flush=True)
