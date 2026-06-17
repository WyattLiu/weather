"""Full day-by-day OPERATOR BLOTTER — track EVERYTHING so the human can interactively
step through each day: regime + signals, book state (cash/shares/BOXX/KOLD), the RUNNING
short-put/call position BY STRIKE (reconstructed from the trade flow → real concentration),
coverage, daily P&L, and the day's ACTIONS (strike×expiry) vs SIGNALS (stand-downs).

Writes results/daily_journal.csv (sortable for interactive review) + prints a blotter.
"""
import os, sys, argparse
from collections import defaultdict
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_engine as R

ACTIONS = {'OPEN_PUT', 'OPEN_CC', 'OPEN_ITM_CC', 'PUT_TP', 'CALL_TP', 'PUT_ROLL_DOWN',
           'PUT_ASSIGN', 'CALL_ASSIGN', 'OPEN_LONG_PUT_FLOOR', 'KOLD_REGIME_BUY',
           'KOLD_SHOULDER_ENTRY', 'KOLD_DD_HEDGE_BUY', 'Z_TARGET_ADD', 'DD_TRIM_SHARES',
           'OPEN_REBUILD_PUT', 'ELEVATOR_CLOSE', 'CALL_ROLL_UP', 'CALL_GAMMA_CLOSE'}
LBL = {'OPEN_PUT': 'SELL put', 'OPEN_CC': 'SELL call', 'OPEN_ITM_CC': 'SELL ITM-call',
       'PUT_TP': 'BTC put(TP)', 'CALL_TP': 'BTC call(TP)', 'PUT_ROLL_DOWN': 'ROLL put↓',
       'PUT_ASSIGN': 'put ASSIGNED', 'CALL_ASSIGN': 'call ASSIGNED',
       'KOLD_SHOULDER_ENTRY': 'KOLD hedge', 'Z_TARGET_ADD': 'BUY shares',
       'DD_TRIM_SHARES': 'TRIM shares', 'ELEVATOR_CLOSE': 'elevator close'}


def _apply(sp, sc, r):
    """Update running short-put/call dicts (strike→contracts) from one trade."""
    ty = r['type']; K = r.get('K'); q = abs(int(r['qty'])) if ('qty' in r and r['qty'] == r['qty']) else 0
    K = round(float(K), 1) if (K == K and K) else None
    if ty == 'OPEN_PUT' and K: sp[K] += q
    elif ty in ('OPEN_CC', 'OPEN_ITM_CC') and K: sc[K] += q
    elif ty in ('PUT_TP', 'PUT_ASSIGN', 'PUT_EXPIRE_OTM') and K: sp[K] = max(0, sp[K] - q)
    elif ty in ('CALL_TP', 'CALL_ASSIGN', 'CALL_EXPIRE_OTM') and K: sc[K] = max(0, sc[K] - q)
    elif ty == 'PUT_ROLL_DOWN':
        fk = r.get('from_K'); tk = r.get('to_K')
        if fk == fk and fk: sp[round(float(fk), 1)] = max(0, sp[round(float(fk), 1)] - q)
        if tk == tk and tk: sp[round(float(tk), 1)] += q


def main(kernel, start, end):
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), parse_dates=[0], index_col=0)
    df = R.precompute_factor_z(df).dropna(subset=['UNG']).loc[start:end]
    p = {**R.STRATEGIES[kernel], 'intraday_exec': False, 'real_chain_pricing': False}
    h, t = R.run_strategy_simple(df, p, 100000, 0)
    h = h.set_index(pd.to_datetime(h['date']))
    t['d'] = pd.to_datetime(t['date'])
    bx = df['BOXX'].reindex(h.index, method='ffill').fillna(117)
    ssz = df['storage_surprise_z'].reindex(h.index, method='ffill')
    sp, sc = defaultdict(int), defaultdict(int)      # running position by strike
    rows, prev_nav = [], None
    for dt, hr in h.iterrows():
        td = t[t['d'] == dt]
        acts, sigs = [], []
        for _, r in td.iterrows():
            ty = r['type']
            if ty in ACTIONS:
                _apply(sp, sc, r)
            K = r.get('K'); q = r.get('qty')
            ks = f" {int(abs(q)) if q == q else ''}×${K:.1f}" if (K == K and K) else ''
            (acts if ty in ACTIONS else sigs).append(LBL.get(ty, ty) + ks)
        nput = sum(sp.values()); ncall = sum(sc.values())
        topK = max(sp.items(), key=lambda x: x[1]) if any(sp.values()) else (None, 0)
        sh = int(hr['shares'])
        nav = hr['nav']; dnav = (nav - prev_nav) if prev_nav else 0; prev_nav = nav
        rows.append({'date': dt.date(),
                     'regime': ('ACC' if ssz.get(dt, 0) < -0.5 else 'DIST' if ssz.get(dt, 0) > 0.5 else 'NEU'),
                     'spot': round(hr['spot'], 2), 'ssz': round(float(ssz.get(dt, 0) or 0), 2),
                     'nav': round(nav, 0), 'dNAV': round(dnav, 0),
                     'cash': round(hr['cash'], 0), 'shares': sh,
                     'boxx$': round(hr['boxx'] * bx.get(dt, 117), 0),
                     'nSP': nput, 'nSC': ncall, 'cov': f"{ncall}/{sh // 100}",
                     'covered': ncall <= sh // 100,
                     'topPutK': topK[0], 'topPutQty': topK[1],
                     'ACTIONS': ' · '.join(acts), 'SIGNALS': ' · '.join(sigs)})
    j = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), 'results', 'daily_journal.csv')
    j.to_csv(out, index=False)
    print(f"=== OPERATOR BLOTTER: {kernel} ({start}→{end}) — cash-start $100k/0sh ===")
    print(f"{len(j)} days · {(j['ACTIONS'] != '').sum()} trade-days · uncovered-days {(~j['covered']).sum()} "
          f"· PEAK put-concentration {j['topPutQty'].max()} contracts at one strike "
          f"(${j.loc[j['topPutQty'].idxmax(), 'topPutK']})\n")
    for _, r in j[j['ACTIONS'] != ''].tail(22).iterrows():
        print(f"{r['date']} {r['regime']:4} ${r['spot']:5.2f} ssz{r['ssz']:+.1f} "
              f"NAV${r['nav']:,.0f}({r['dNAV']:+,.0f}) sh{r['shares']:>5} BOXX${r['boxx$']:,.0f} "
              f"| SP{r['nSP']:>3} SC{r['nSC']:>3} cov{r['cov']:>6} maxK{r['topPutQty']:>3}@${r['topPutK']}")
        if r['ACTIONS']: print(f"      ▶ {r['ACTIONS']}")
        if r['SIGNALS']: print(f"      · {r['SIGNALS']}")
    print(f"\n(full sortable journal → results/daily_journal.csv, {len(j)} days)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernel', default='regime_wheel_boxx')
    ap.add_argument('--start', default='2026-03-15')
    ap.add_argument('--end', default='2026-06-16')
    a = ap.parse_args()
    main(a.kernel, a.start, a.end)
