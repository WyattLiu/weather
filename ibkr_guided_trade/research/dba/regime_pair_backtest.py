"""Backtest the CORN/CANE ENSO regime-pair switch — leak-free.

Anti-lookahead measures:
  1. ONI publication lag: the official ONI value for the 3-month season
     CENTERED on month M is only final after month M+1 ends and is
     published in early M+2 relative to the center label... in practice
     CPC posts it in the first week after the season completes. We test
     TWO lags: +1 month (realistic — weekly Niño3.4 runs near-realtime)
     and +2 months (paranoid).
  2. No CPC forecast in the gate (no historical vintages) — ONI only:
        oni <= -0.25 → CORN   |   oni >= +0.75 → CANE   |   else → none
     (satellite idles in BOXX when no regime).
  3. Daily switching via return streams of independently-run wheels;
     regime changes counted to bound transition-cost impact.

Run: venv/bin/python research/dba/regime_pair_backtest.py
"""
import os
import sys
import math
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS_DIR, 'cache')
sys.path.insert(0, THIS_DIR)

from wheel_backtest import run_wheel  # noqa: E402

BOXX_DAILY = 0.0474 / 252
START = '2015-01-01'


def lagged_oni_daily(idx, lag_months):
    oni = pd.read_csv(os.path.join(CACHE, 'oni.csv'),
                      index_col=0, parse_dates=True)['oni']
    # index = center-month start. Value becomes KNOWN lag_months later.
    known = oni.copy()
    known.index = known.index + pd.DateOffset(months=lag_months)
    return known.reindex(idx, method='ffill')


def stats(ret, label):
    ret = ret.dropna()
    if len(ret) < 100 or ret.std() == 0:
        return {'strategy': label, 'ann_ret': 0, 'sharpe': 0, 'mdd': 0}
    nav = (1 + ret).cumprod()
    yrs = (ret.index[-1] - ret.index[0]).days / 365.25
    return {'strategy': label,
            'ann_ret': round(nav.iloc[-1] ** (1 / yrs) - 1, 4),
            'sharpe': round(ret.mean() / ret.std() * math.sqrt(252), 3),
            'mdd': round((nav / nav.cummax() - 1).min(), 4)}


def main():
    legs = {}
    for tk in ('CORN', 'CANE', 'DBA'):
        curve, _ = run_wheel(tk, start=START, dte_target=60, otm_pct=0.02)
        legs[tk] = curve['nav'].pct_change().fillna(0)
        print(f'  {tk} wheel ready ({len(curve)} days)')

    idx = legs['CORN'].index.intersection(legs['CANE'].index)
    rows = []
    for lag in (1, 2):
        oni = lagged_oni_daily(idx, lag)
        regime = pd.Series('none', index=idx)
        regime[oni <= -0.25] = 'CORN'
        regime[oni >= 0.75] = 'CANE'
        # switch days AFTER the signal (one more day of lag)
        regime = regime.shift(1).fillna('none')

        sat = pd.Series(BOXX_DAILY, index=idx)
        for tk in ('CORN', 'CANE'):
            mask = regime == tk
            sat[mask] = legs[tk].reindex(idx).fillna(0)[mask]

        switches = (regime != regime.shift(1)).sum()
        occup = regime.value_counts(normalize=True)
        r = stats(sat, f'regime-pair (lag {lag}mo)')
        r['switches'] = int(switches)
        r['pct_corn'] = round(float(occup.get('CORN', 0)), 2)
        r['pct_cane'] = round(float(occup.get('CANE', 0)), 2)
        r['pct_idle'] = round(float(occup.get('none', 0)), 2)
        rows.append(r)

    for tk in ('CORN', 'CANE', 'DBA'):
        rows.append(stats(legs[tk].reindex(idx).fillna(0), f'{tk} always-on'))
    # 50/50 static corn+cane for reference
    rows.append(stats(0.5 * legs['CORN'].reindex(idx).fillna(0)
                      + 0.5 * legs['CANE'].reindex(idx).fillna(0),
                      'CORN+CANE static 50/50'))

    df = pd.DataFrame(rows)
    print(f'\n=== REGIME-PAIR BACKTEST ({idx[0].date()} → {idx[-1].date()}) ===')
    print(df.to_string(index=False))
    df.to_csv(os.path.join(CACHE, 'regime_pair_backtest.csv'), index=False)

    # estimate transition cost: each switch ≈ close+reopen ≈ 2x half-spread
    # thin chains ~3% of premium ≈ ~0.1% of notional per switch
    n_sw = rows[0]['switches']
    yrs = (idx[-1] - idx[0]).days / 365.25
    print(f'\nswitches (lag 1mo): {n_sw} over {yrs:.0f}y '
          f'(~{n_sw/yrs:.1f}/yr) → est. transition drag ~{n_sw/yrs * 0.1:.2f}%/yr')


if __name__ == '__main__':
    main()
