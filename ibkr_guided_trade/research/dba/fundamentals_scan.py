"""Scan non-weather fundamentals vs DBA forward returns.

Factors:
  cot_z       — ag-basket avg managed-money net %OI, 156wk z-score
                (positioning extreme; expect MEAN REVERSION → negative spread)
  cot_chg_13w — 13-week change in positioning (flow momentum)
  fpi_mom_3m  — FAO Food Price Index 3m momentum (published ~1mo lag,
                shifted 2 months to be conservative)
  fpi_yoy     — FAO FPI year-over-year
  stu_z       — world grain stocks-to-use (corn+wheat+soy avg), z vs
                5yr history, mapped to daily by marketing year.
                CAVEAT: current-vintage PSD values (revision lookahead).
  crude_3m    — CL 3m trend (biofuel/input channel)

Same quintile machinery as factor_scan.py.
"""
import os
import sys
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS_DIR, 'cache')
FUND = os.path.join(CACHE, 'fundamentals')
sys.path.insert(0, THIS_DIR)

from factor_scan import quintile_table  # noqa: E402


def build():
    panel = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    dba = panel['DBA'].dropna()
    f = pd.DataFrame(index=dba.index)
    f['fwd_21d'] = dba.pct_change(21).shift(-21)
    f['fwd_63d'] = dba.pct_change(63).shift(-63)

    # COT: basket-average managed-money net %OI
    cot = pd.read_csv(os.path.join(FUND, 'cot_ag.csv'), parse_dates=['date'])
    basket = cot.groupby('date')['mm_net_pct_oi'].mean()
    basket = basket.reindex(f.index, method='ffill')
    f['cot_z'] = ((basket - basket.rolling(756, min_periods=252).mean())
                  / basket.rolling(756, min_periods=252).std())
    f['cot_chg_13w'] = basket - basket.shift(65)

    # FAO FPI: published with ~1 month lag → shift 2 months of trading days
    fao = pd.read_csv(os.path.join(FUND, 'fao_fpi.csv'), parse_dates=['date'])
    fpi_col = [c for c in fao.columns if 'food price index' in str(c).lower()]
    fpi = fao.set_index('date')[fpi_col[0]].astype(float)
    fpi_d = fpi.reindex(f.index, method='ffill').shift(42)
    f['fpi_mom_3m'] = fpi_d.pct_change(63)
    f['fpi_yoy'] = fpi_d.pct_change(252)

    # USDA stocks-to-use: avg of corn/wheat/soy z-scores, by marketing year.
    # Marketing year Y data becomes "known" ~Oct of calendar year Y.
    stu = pd.read_csv(os.path.join(FUND, 'usda_stocks_to_use.csv'))
    piv = stu.pivot_table(index='market_year', columns='commodity',
                          values='stocks_to_use')
    z = (piv - piv.rolling(5, min_periods=3).mean()) / piv.rolling(5, min_periods=3).std()
    stu_z_yr = z.mean(axis=1)
    f['stu_z'] = pd.Series(
        {d: stu_z_yr.get(d.year if d.month >= 10 else d.year - 1)
         for d in f.index})

    # Crude trend from master dataset
    md = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(THIS_DIR)),
                                  'backtest', 'cache', 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    _idx = pd.to_datetime(md.index, utc=True).tz_localize(None)
    md.index = _idx.normalize()
    md = md.groupby(md.index).first()
    if 'CL' in md.columns:
        f['crude_3m'] = md['CL'].pct_change(63).reindex(f.index).ffill()

    return f


def main():
    f = build()
    rows = []
    for fac in ['cot_z', 'cot_chg_13w', 'fpi_mom_3m', 'fpi_yoy',
                'stu_z', 'crude_3m']:
        if fac not in f.columns or f[fac].dropna().empty:
            print(f'  {fac}: NO DATA')
            continue
        for fwd in ('fwd_21d', 'fwd_63d'):
            r = quintile_table(f, fac, fwd)
            if r:
                rows.append(r)
    res = pd.DataFrame(rows).sort_values('p')
    print('\n=== FUNDAMENTALS SCAN — quintile top-minus-bottom fwd returns ===')
    print(res.to_string(index=False))
    res.to_csv(os.path.join(CACHE, 'dba_fundamentals_scan.csv'), index=False)
    f.to_csv(os.path.join(CACHE, 'dba_fundamentals_panel.csv'))
    print(f'\n→ {CACHE}/dba_fundamentals_scan.csv')


if __name__ == '__main__':
    main()
