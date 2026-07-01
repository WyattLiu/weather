"""Backfill BOXX for its pre-inception gap (2021-07 → 2022-12-27) in master_dataset.csv.

BOXX (Alpha Architect box-spread ETF) did not exist before 2022-12-28 ($99.76 inception). Before this
the master dataset had BOXX=NaN, and the engine defaulted the price to a flat 117 — 16% too high AND
constant, which (a) understated the cash-sweep yield and (b) produced a FICTIONAL ~-11% one-day 'crash'
at the seam when 117 dropped to the real 99.76 (BOXX is ~79% of the book). That fake crash depressed
TRAIN return and inflated max-drawdown across every study.

Fix: BOXX ≈ a risk-free box-spread, so synthesize the pre-inception series from BIL (SPDR 1-3mo T-bill
ETF, total-return / auto-adjusted — exists since 2007) as the accrual SHAPE, scaled so it meets the real
BOXX at the inception seam. 2021 rates ~0, rising through 2022 → a smooth ~1.3% drift to $99.76, no seam
jump. Idempotent: a no-op once BOXX has no pre-seam gap. Run after any full master_dataset rebuild.

  venv/bin/python backtest/backfill_boxx.py
"""
import os

import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(THIS, 'cache', 'master_dataset.csv')


def backfill(verbose=True):
    m = pd.read_csv(MASTER, index_col=0, parse_dates=True)
    if 'BOXX' not in m.columns:
        print('[boxx] no BOXX column — skip'); return
    seam = m['BOXX'].first_valid_index()
    gap = m.index < seam
    if not gap.any():
        if verbose:
            print(f'[boxx] already complete (BOXX from {seam.date()}); no pre-inception gap'); return
    anchor = float(m.loc[seam, 'BOXX'])
    import yfinance as yf
    bil = yf.download('BIL', start=str((m.index[0] - pd.Timedelta(days=7)).date()),
                      end=str(seam.date()), progress=False, auto_adjust=True)['Close'].squeeze()
    bil = bil.reindex(m.loc[:seam].index).ffill().bfill()
    synth = bil * (anchor / float(bil.loc[seam]))          # T-bill accrual shape, anchored to inception
    m.loc[gap, 'BOXX'] = synth.reindex(m.index)[gap].values
    m.to_csv(MASTER)
    if verbose:
        drift = (anchor / float(m['BOXX'].iloc[0]) - 1) * 100
        print(f'[boxx] backfilled {m.index[0].date()}→{seam.date()}: '
              f'{m["BOXX"].iloc[0]:.2f} → {anchor:.2f} (+{drift:.2f}% T-bill accrual, no seam jump)')


if __name__ == '__main__':
    backfill()
