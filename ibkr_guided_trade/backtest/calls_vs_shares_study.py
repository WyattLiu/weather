"""3-WAY: reach the share-target's delta via SHARES (sells BOXX) vs LONG CALLS (keeps BOXX, long cheap
vol). Long-call accumulation is NEW (never trialled). NAV now marks long calls to market (long_calls_mtm)
so the calls arm is valued fairly. Reports walk-forward + the add-method mix + capital deployed."""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple

BASE = STRATEGIES['regime_wheel_boxx_greeks']
VARIANTS = {
    'shares (sells BOXX)':      dict(BASE),
    'calls when IV<0.30':       {**BASE, 'reaccum_via_calls': True, 'reaccum_calls_iv_max': 0.30, 'reaccum_call_dte': 90},
    'sell ATM puts (45d)':      {**BASE, 'reaccum_via_puts': True, 'reaccum_put_dte': 45},
    'sell ATM puts (30d)':      {**BASE, 'reaccum_via_puts': True, 'reaccum_put_dte': 30},
}


def _load():
    return precompute_factor_z(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'cache', 'master_dataset.csv'), index_col=0, parse_dates=True)).dropna(subset=['UNG'])


def metrics(strat, df, nav0=100000):
    hist, trades = run_strategy_simple(df, strat, nav0, 0)
    hist = hist.set_index(pd.to_datetime(hist['date'])); nav = hist['nav']; r = nav.pct_change().dropna()
    yrs = (df.index[-1]-df.index[0]).days/365.25
    ann = ((nav.iloc[-1]/nav0)**(1/yrs)-1)*100
    sh = r.mean()/(r.std()+1e-9)*math.sqrt(252)
    mdd = ((nav-nav.cummax())/nav.cummax()*100).min()
    t = trades['type'].astype(str)
    n_sh = int((t == 'Z_TARGET_ADD').sum())
    n_ca = int((t == 'Z_TARGET_ADD_CALLS').sum())
    n_pa = int((t == 'Z_TARGET_ADD_PUTS').sum())
    return ann, sh, mdd, n_sh, n_ca, n_pa


def _job(a):
    kind, name, st = a; df = _load()
    d = df.loc[TRAIN_START:TRAIN_END] if kind == 'TRAIN' else df.loc[TEST_START:TEST_END]
    return (name, kind) + metrics(st, d)


if __name__ == '__main__':
    jobs = [(w, n, s) for n, s in VARIANTS.items() for w in ('TRAIN', 'TEST')]
    with mp.Pool(6) as pool: res = pool.map(_job, jobs)
    res.sort(key=lambda r: (list(VARIANTS).index(r[0]), 0 if r[1] == 'TRAIN' else 1))
    print(f"  {'method':<22}{'win':<7}{'ann':>7}{'Sh':>6}{'MaxDD':>7}{'shAdd':>7}{'caAdd':>7}{'puAdd':>7}")
    for n, w, a, s, m, nsh, nca, npa in res:
        print(f"  {n:<22}{w:<7}{a:>6.1f}%{s:>6.2f}{m:>6.1f}%{nsh:>7d}{nca:>7d}{npa:>7d}")
    print("DONE", flush=True)
