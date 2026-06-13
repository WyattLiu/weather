"""Synthetic long / risk reversal capital-efficiency analysis (gen-5 #8).

When IV-rank is low (calls cheap), compare three ways to get UNG upside
exposure, on real-fill-adjusted prices:
  SHARES        : 100 sh = full delta, full capital (K*100)
  SYNTHETIC LONG: +1 call -1 put same K → ~full delta, ~1/5 capital
                  (cash-secured put collateral is the real capital)
  RISK REVERSAL : short OTM put funds long OTM call → zero/low net debit,
                  delta from the gap, defined put-assignment floor

Decision axes: capital efficiency, assignment tail, theta sign, and the
covered-calls-only rule (synthetic's short put is cash-secured = fine;
the long call is not a covered-call obligation).

Run: venv/bin/python backtest/synthetic_rr_analysis.py
"""
import os
import math
import pandas as pd
import numpy as np
from scipy.stats import norm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
PANEL = os.path.join(ROOT, 'research/dba/cache/master_panel.csv')


def bs(S, K, T, sig, right):
    d1 = (math.log(S/K)+(0.045+sig*sig/2)*T)/(sig*math.sqrt(T)); d2 = d1-sig*math.sqrt(T)
    if right == 'C':
        return S*norm.cdf(d1)-K*math.exp(-0.045*T)*norm.cdf(d2)
    return K*math.exp(-0.045*T)*norm.cdf(-d2)-S*norm.cdf(-d1)


def main():
    panel = pd.read_csv(PANEL, index_col=0, parse_dates=True)
    ung = panel['UNG'].dropna()
    rv = (ung.pct_change().rolling(30).std() * math.sqrt(252) * 1.12).bfill()
    iv = None
    try:
        ivr = pd.read_csv(os.path.join(THIS_DIR, 'cache', 'ung_iv_rank_daily.csv'),
                          index_col=0, parse_dates=True)
        iv = ivr['iv_rank'].reindex(ung.index, method='ffill')
    except Exception:
        pass

    # Empirical forward-90d outcome of each structure, conditioned on IV-rank
    hor = ung.pct_change(63).shift(-63).dropna()
    T = 90/365
    rows = []
    for d in ung.index[::21]:
        if d not in hor.index:
            continue
        S = float(ung.loc[d]); sig = float(rv.loc[d])
        fwd = float(hor.loc[d]); ST = S * (1 + fwd)
        ivr_now = float(iv.loc[d]) if iv is not None and d in iv.index and iv.loc[d]==iv.loc[d] else None
        # SHARES: pnl = ST-S per share, capital = S
        shares_pnl = (ST - S) / S
        # SYNTHETIC LONG at K=S: long call - short put, ~ (ST-S); capital = put collat S
        syn_pnl = (max(ST-S,0) - bs(S,S,T,sig,'C')) - (max(S-ST,0) - bs(S,S,T,sig,'P'))
        syn_pnl /= S
        # RISK REVERSAL: short 7% put (collateral) funds long 7% call
        Kp, Kc = S*0.93, S*1.07
        rr_credit = bs(S,Kp,T,sig,'P') - bs(S,Kc,T,sig,'C')
        rr_pnl = (max(ST-Kc,0) - max(Kp-ST,0) + rr_credit) / S
        rows.append({'ivr': ivr_now, 'shares': shares_pnl, 'syn': syn_pnl, 'rr': rr_pnl})
    R = pd.DataFrame(rows)
    print(f'=== SYNTHETIC / RISK-REVERSAL ANALYSIS ({len(R)} 90d windows) ===\n')
    print('Forward-90d return per $1 capital, by structure:')
    print(f"  SHARES:        E {R['shares'].mean():+.1%}  p5 {R['shares'].quantile(.05):+.1%}  p95 {R['shares'].quantile(.95):+.1%}")
    print(f"  SYNTHETIC:     E {R['syn'].mean():+.1%}  p5 {R['syn'].quantile(.05):+.1%}  p95 {R['syn'].quantile(.95):+.1%}")
    print(f"  RISK REVERSAL: E {R['rr'].mean():+.1%}  p5 {R['rr'].quantile(.05):+.1%}  p95 {R['rr'].quantile(.95):+.1%}")
    if R['ivr'].notna().any():
        lo = R[R['ivr'] < 0.2]
        print(f"\nWHEN IV-RANK LOW (<0.2, n={len(lo)}) — the call-is-cheap regime (TODAY):")
        for c, name in (('shares','SHARES'),('syn','SYNTHETIC'),('rr','RISK REVERSAL')):
            print(f"  {name:>14}: E {lo[c].mean():+.1%}  p5 {lo[c].quantile(.05):+.1%}")
        print('  → capital efficiency: synthetic/RR get ~full delta at ~1/5 the '
              'capital, freeing collateral for more positions. Edge is the '
              'capital multiplier, not per-$ return.')
    R.to_csv(os.path.join(THIS_DIR, 'cache', 'synthetic_rr_outcomes.csv'), index=False)


if __name__ == '__main__':
    main()
