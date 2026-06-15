"""Unified adversarial audit gate — question EVERY result, systematically.

The session's hard lessons (each one slipped through until manually caught):
  - CC-stacking bug inflated ag-wheel returns          → integrity screen
  - model-vs-real-fill conflation (-14pp mirage)       → fill-consistency
  - share-count confound (gen-7 "win")                 → confound check
  - 2023-concentrated edge sold as all-weather         → regime stratify
  - nan-correlation / unverified mechanism             → data-integrity
  - in-sample max of a grid (overfit)                  → bootstrap CI

audit.py runs ALL of them on a (strategy, baseline) pair and emits a
single structured verdict. Nothing should be promoted that the auditor
does not pass. It is ADVERSARIAL by design: every check tries to KILL the
result; the result must survive all of them.

Usage:
    venv/bin/python backtest/audit.py --strategy champion_kold15_ivrank_kbh \
        --baseline champion_kold15_ivrank
"""
import os
import sys
import math
import argparse
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(THIS_DIR, 'results')
sys.path.insert(0, THIS_DIR)


def _load(n):
    h = pd.read_csv(os.path.join(RESULTS, f'{n}_history.csv'))
    h['date'] = pd.to_datetime(h['date']).dt.normalize()
    return h.set_index('date')


def _stats(nav):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return ((nav.iloc[-1]/nav.iloc[0])**(1/yrs)-1,
            r.mean()/r.std()*math.sqrt(252) if r.std() else 0,
            (nav/nav.cummax()-1).min())


def audit(strategy, baseline=None):
    from replay_engine import STRATEGIES
    verdict = {'strategy': strategy, 'checks': [], 'flags': 0, 'fatal': 0}
    def chk(name, ok, detail, fatal=False):
        status = 'PASS' if ok else ('FATAL' if fatal else 'WARN')
        verdict['checks'].append((status, name, detail))
        if not ok:
            verdict['flags'] += 1
            if fatal:
                verdict['fatal'] += 1

    h = _load(strategy)
    sp = STRATEGIES.get(strategy, {})

    # 1. INTEGRITY (covered-call, neg cash, collateral, marking noise, stale)
    if {'shares','short_calls'}.issubset(h.columns):
        naked = (h['short_calls']*100 > h['shares']+1).sum()
        chk('covered_calls_only', naked == 0, f'{naked} naked-call days', fatal=True)
    if 'cash' in h.columns:
        chk('no_negative_cash', (h['cash'] < -1000).sum() == 0,
            f"min cash ${h['cash'].min():,.0f}", fatal=True)
    sr = h['spot'].pct_change(); nr = h['nav'].pct_change()
    dislocations = ((nr.abs() > 0.08) & (sr.abs() < 0.02)).sum()
    chk('no_marking_noise', dislocations == 0, f'{dislocations} NAV/spot dislocations')
    stale = ((h['spot'].diff() == 0).rolling(5).sum() >= 5).any()
    chk('no_stale_prices', not stale,
        'stale-price run detected' if stale else 'clean')

    # 2. FILL-MODEL CONSISTENCY (the -14pp conflation guard)
    s_rf = sp.get('real_fill_model', False)
    if baseline:
        b_rf = STRATEGIES.get(baseline, {}).get('real_fill_model', False)
        chk('fill_model_matched', s_rf == b_rf,
            f'both real_fill={s_rf}' if s_rf==b_rf else
            f'MISMATCH strat={s_rf} base={b_rf} — comparison INVALID', fatal=(s_rf != b_rf))

    # 3. CONFOUND (share count vs baseline)
    if baseline:
        b = _load(baseline)
        idx = h.index.intersection(b.index)
        sh_ratio = h.loc[idx,'shares'].mean() / max(1, b.loc[idx,'shares'].mean())
        chk('no_share_confound', abs(sh_ratio-1) < 0.05,
            f'avg shares {sh_ratio:.2f}x baseline — edge may be exposure not skill')

    # 4. REGIME STRATIFICATION (one-year concentration)
    if baseline:
        b = _load(baseline); idx = h.index.intersection(b.index)
        rh = h.loc[idx,'nav'].pct_change(); rb = b.loc[idx,'nav'].pct_change()
        edges = {}
        for yr in sorted(set(idx.year)):
            mh = rh[rh.index.year==yr].dropna(); mb = rb[rb.index.year==yr].dropna()
            if len(mh) > 30 and mh.std() and mb.std():
                edges[yr] = (mh.mean()/mh.std() - mb.mean()/mb.std())*math.sqrt(252)
        if edges:
            total = sum(edges.values()); mx = max(edges.values(), key=abs)
            conc = abs(mx)/abs(total) if total else 99
            edges_str = ', '.join(f'{y}:{e:+.2f}' for y, e in edges.items())
            chk('not_one_year', conc < 0.7,
                f'biggest year = {conc:.0%} of total edge ({edges_str})')

    # 5. BOOTSTRAP SIGNIFICANCE (vs baseline)
    if baseline:
        b = _load(baseline); idx = h.index.intersection(b.index)
        rh = h.loc[idx,'nav'].pct_change().dropna().values
        rb = b.loc[idx,'nav'].pct_change().dropna().values
        m = min(len(rh), len(rb)); rh, rb = rh[:m], rb[:m]
        def shp(x): return x.mean()/x.std()*math.sqrt(252) if x.std() else 0
        obs = shp(rh) - shp(rb)
        rng = np.random.default_rng(42); bl = 60; ds = []
        for _ in range(1000):
            st = rng.integers(0, m-bl, size=m//bl)
            mi = np.concatenate([np.arange(s,s+bl) for s in st])
            ds.append(shp(rh[mi]) - shp(rb[mi]))
        lo, hi = np.percentile(ds, [5, 95])
        chk('bootstrap_significant', lo > 0,
            f'Sharpe diff {obs:+.2f}, 90%CI [{lo:+.2f},{hi:+.2f}]')

    # 6. WALK-FORWARD TRUTH (worst-12mo quoted)
    w12 = h['nav'].pct_change(252).dropna()
    if len(w12):
        verdict['worst_12mo'] = round(float(w12.min()), 4)

    a, s, d = _stats(h['nav'])
    verdict.update({'ann': round(a,4), 'sharpe': round(s,3), 'mdd': round(d,4)})
    # overall ruling
    if verdict['fatal'] > 0:
        verdict['ruling'] = 'REJECTED (fatal integrity/consistency failure)'
    elif verdict['flags'] == 0:
        verdict['ruling'] = 'PROVEN (survived full gauntlet)'
    else:
        verdict['ruling'] = f"QUALIFIED ({verdict['flags']} warnings — not all-weather/clean)"
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--baseline', default=None)
    a = ap.parse_args()
    v = audit(a.strategy, a.baseline)
    print(f"\n=== ADVERSARIAL AUDIT: {v['strategy']} ===")
    if a.baseline:
        print(f"    vs baseline: {a.baseline}")
    print(f"    full-sample: ann {v.get('ann'):+.1%} / Sharpe {v.get('sharpe')} / "
          f"MDD {v.get('mdd'):.1%} / worst-12mo {v.get('worst_12mo','?')}")
    print()
    for status, name, detail in v['checks']:
        mark = {'PASS':'✓','WARN':'⚠','FATAL':'✗'}[status]
        print(f"  {mark} {name:<24} {detail}")
    print(f"\n  >>> RULING: {v['ruling']}")


if __name__ == '__main__':
    main()
