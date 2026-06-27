"""Test the two Arnold-pillar leading signals on UNG: WEATHER (degree-day anomaly,
from STEO ZWHDPUS+ZWCDPUS) and CURVE (contango = NG futures - HH spot).

Verdict (2026-06): neither is a strong leading signal on UNG's 5yr history.
  - Contango: no monthly timing power (corr +0.02); weak directional tail at 63d
    (steep -10.9% vs backwardation -5.6%). A structural DRAG signal, not timing —
    already addressed by holding less UNG (regime distribute + BOXX sweep).
  - Weather (degree-day anomaly): weak +0.12 corr, hot/cold>1σ next-mo +3.8% vs
    mild -2.1%; BUT partly redundant with storage_surprise_z (corr -0.30) and only
    51 realized months (low power). Directionally right, not robust.
  - Echoes steo_scan.csv: the STEO HH price-forecast premium also failed (p 0.36-0.99).

Storage_surprise_z already captures the fundamental edge. Arnold's weather edge needs
LEADING forecasts (NOAA 6-10 / 8-14 day GWDD), not lagging monthly STEO — that's the
real data gap to close if we want this signal. NOT wired (overfit risk on weak signal).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ibkr_guided_trade', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'backtest'))
import pandas as pd
import replay_engine as R

STEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'steo', 'apr26_base.xlsx')


def degree_day_anomaly():
    t1 = pd.ExcelFile(STEO).parse('1tab', header=None)
    hdd = pd.to_numeric(t1[t1[0].astype(str) == 'ZWHDPUS'].iloc[0].iloc[2:], errors='coerce').dropna()
    cdd = pd.to_numeric(t1[t1[0].astype(str) == 'ZWCDPUS'].iloc[0].iloc[2:], errors='coerce').dropna()
    n = min(len(hdd), len(cdd))
    dd = pd.Series(hdd.values[:n] + cdd.values[:n], index=pd.date_range('2022-01-01', periods=n, freq='MS'))
    dd = dd[dd.index <= '2026-03-01']      # realized only (apr26 vintage)
    return dd - dd.groupby(dd.index.month).transform('mean')


def main():
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), parse_dates=[0], index_col=0)
    df = df[~df.index.duplicated()]
    u = df['UNG'].dropna()
    nxt = (u.resample('MS').last().pct_change().shift(-1)) * 100
    anom = degree_day_anomaly()
    al = pd.concat([anom.rename('dd'), nxt.rename('f')], axis=1).dropna()
    print(f"WEATHER (degree-day anomaly), n={len(al)}: corr {al['dd'].corr(al['f']):+.3f}")
    cont = (df['NG'].ffill() - df['eia_hh_spot_daily'].ffill()).resample('MS').last()
    cm = pd.concat([cont.rename('c'), nxt.rename('f')], axis=1).dropna()
    print(f"CONTANGO (NG-HHspot): corr {cm['c'].corr(cm['f']):+.3f}")


if __name__ == '__main__':
    main()
