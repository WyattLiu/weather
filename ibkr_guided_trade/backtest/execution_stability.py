"""Execution-stability analyzer — takes the strategy's actual trades and, using the
REAL intraday bid/ask path (PG ung_options_history), reports for each order:
can it fill near mid in the window? what slippage vs mid? how wide is the spread?

Answers "test stability and execution": fill rate, slippage distribution, and how the
intraday-realistic fill compares to the pessimistic EOD-touch assumption.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import replay_engine as R
from intraday_fill import intraday_fill_price


def main(kernel, start, end, patience):
    df = pd.read_csv(os.path.join(THIS, 'cache', 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    df = R.precompute_factor_z(df).dropna(subset=['UNG']).loc[start:end]
    _, t = R.run_strategy_simple(df, R.STRATEGIES[kernel], 48000, 6200)
    opens = t[t['type'].isin(['OPEN_PUT', 'OPEN_CC', 'OPEN_ITM_CC'])].copy()
    print(f"{kernel}: {len(opens)} open orders in {start}→{end}; checking intraday fillability "
          f"(patience={patience})\n")

    rows = []
    for _, o in opens.iterrows():
        K = o.get('K'); dte = o.get('dte')
        if K != K or K is None:
            continue
        right = 'P' if o['type'] == 'OPEN_PUT' else 'C'
        dte = int(dte) if (dte == dte and dte) else 30
        d = pd.Timestamp(o['date']).date().isoformat()
        spot = float(o.get('spot') or 0) or None
        res = intraday_fill_price(d, float(K), dte, right, spot, 'sell', patience=patience)
        if res is None:
            rows.append({'filled': False})
            continue
        px, spread_pct, vs_mid = res
        rows.append({'filled': True, 'fill': px, 'spread_pct': spread_pct, 'vs_mid': vs_mid,
                     'mid': px - vs_mid})
    rdf = pd.DataFrame(rows)
    n = len(rdf); nf = int(rdf['filled'].sum()) if n else 0
    print(f"=== EXECUTION STABILITY ({nf}/{n} orders had intraday coverage) ===")
    if nf:
        f = rdf[rdf['filled']]
        # slippage vs mid as % of mid (negative = you gave up vs mid on a sell)
        slip_pct = (f['vs_mid'] / f['mid'].replace(0, np.nan) * 100).dropna()
        print(f"  intraday coverage:      {nf}/{n} ({100*nf/n:.0f}%)")
        print(f"  median spread (intraday tightest): {f['spread_pct'].median():.1f}%  "
              f"(p90 {f['spread_pct'].quantile(0.9):.1f}%)")
        print(f"  slippage vs MID (sell):  median {slip_pct.median():+.1f}%  "
              f"mean {slip_pct.mean():+.1f}%  (0 = filled at mid)")
        print(f"  filled within 1¢ of mid: {100*(f['vs_mid'].abs() < 0.01).mean():.0f}% of orders")
        print(f"  filled within 3% of mid: {100*(slip_pct.abs() < 3).mean():.0f}% of orders")
        # compare to EOD-touch pessimism: touch = full half-spread below mid
        eod_slip = -(f['spread_pct'] / 2)  # always-touch sell loses half the spread
        print(f"\n  vs EOD-touch assumption (always cross): EOD slip median {eod_slip.median():.1f}% "
              f"vs intraday {slip_pct.median():+.1f}%")
        print(f"  → patient intraday execution recovers ~{eod_slip.median() - slip_pct.median():.1f}% "
              f"of premium per order vs the EOD-touch model")
    print("\n(Stability: low/tight slippage + high near-mid fill rate = the strategy is)")
    print(" executable as a patient hand-trade; wide tails = orders that won't fill near mid.)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kernel', default='champion_kold15_ivrank_kbh')
    p.add_argument('--start', default='2026-03-15')
    p.add_argument('--end', default='2026-06-12')
    p.add_argument('--patience', type=float, default=0.6)
    a = p.parse_args()
    main(a.kernel, a.start, a.end, a.patience)
