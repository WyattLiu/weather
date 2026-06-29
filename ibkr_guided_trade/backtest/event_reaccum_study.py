"""STUDY: best confident behaviour AFTER a large called-away, and how to glide back to target delta.

Compares the champion (wait-for-21d-cadence) against EVENT-DRIVEN re-accumulation: when a block of
>= reaccum_lots_threshold call lots is called away, re-evaluate the share target off-cadence for a
short window so the book re-accumulates toward the desired delta immediately.

Two evidence layers:
  1. Walk-forward (TRAIN + sealed TEST): ann / Sharpe / MaxDD / turnover — does the policy pay net?
  2. Post-called-away conditional: across EVERY large called-away in the sample, the average forward
     21-day NAV return — does re-accumulating sooner actually capture the recovery, or just churn?
"""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from honest_walkforward import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from replay_engine import STRATEGIES, precompute_factor_z, run_strategy_simple

BASE = STRATEGIES['regime_wheel_boxx_greeks']
VARIANTS = {
    'baseline (21d cadence)':  dict(BASE),
    'event reaccum w=3':       {**BASE, 'reaccum_on_called_away': True, 'reaccum_window': 3,  'reaccum_lots_threshold': 5},
    'event reaccum w=5':       {**BASE, 'reaccum_on_called_away': True, 'reaccum_window': 5,  'reaccum_lots_threshold': 5},
    'event reaccum w=10':      {**BASE, 'reaccum_on_called_away': True, 'reaccum_window': 10, 'reaccum_lots_threshold': 5},
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
    return ann, sh, mdd, len(trades)


def post_called_away(strat, df, nav0=100000, fwd=21, lots=5):
    """Avg forward-`fwd`-bar NAV return measured from each large (>=`lots`) called-away event."""
    hist, trades = run_strategy_simple(df, strat, nav0, 0)
    hist = hist.set_index(pd.to_datetime(hist['date'])); nav = hist['nav'].reset_index(drop=True)
    if 'type' not in trades.columns:
        return 0.0, 0
    ev = trades[(trades['type'].astype(str).str.contains('CALL_ASSIGN')) & (trades.get('qty', 0).abs() >= lots)]
    dates = pd.to_datetime(hist.index)
    rets = []
    for d in pd.to_datetime(ev['date']).unique():
        locs = dates.get_indexer([d])
        j = locs[0] if len(locs) else -1
        if j >= 0 and j + fwd < len(nav):
            rets.append(nav.iloc[j+fwd] / nav.iloc[j] - 1)
    avg = (sum(rets)/len(rets)*100) if rets else 0.0
    return avg, len(rets)


def _job(a):
    kind, name, st = a
    df = _load()
    if kind == 'TRAIN': return ('WF', name, 'TRAIN') + metrics(st, df.loc[TRAIN_START:TRAIN_END])
    if kind == 'TEST':  return ('WF', name, 'TEST')  + metrics(st, df.loc[TEST_START:TEST_END])
    if kind == 'PCA':   return ('PCA', name, 'full') + post_called_away(st, df)


if __name__ == '__main__':
    jobs = ([('TRAIN', n, s) for n, s in VARIANTS.items()]
            + [('TEST', n, s) for n, s in VARIANTS.items()]
            + [('PCA', n, s) for n, s in VARIANTS.items()])
    with mp.Pool(6) as pool: res = pool.map(_job, jobs)
    wf = {(n, w): r for tag, n, w, *r in res if tag == 'WF'}
    pca = {n: r for tag, n, w, *r in res if tag == 'PCA'}
    print(f"  {'variant':<24}{'win':<7}{'ann':>8}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}")
    for n in VARIANTS:
        for w in ('TRAIN', 'TEST'):
            a, s, m, t = wf[(n, w)]
            print(f"  {n:<24}{w:<7}{a:>7.1f}%{s:>8.2f}{m:>7.1f}%{t:>8d}")
    print(f"\n  POST-CALLED-AWAY (avg fwd-21d NAV return after large called-aways, full sample):")
    print(f"  {'variant':<24}{'avg fwd-21d':>14}{'n events':>10}")
    for n in VARIANTS:
        avg, ne = pca[n]
        print(f"  {n:<24}{avg:>13.2f}%{ne:>10d}")
    print("DONE", flush=True)
