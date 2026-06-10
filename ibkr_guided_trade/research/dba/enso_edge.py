"""DBA × ENSO edge measurement.

DBA (Invesco DB Agriculture Fund) — diversified ag basket.
  ~12.5% Sugar  ~12% Cocoa  ~12% Coffee
  ~13% Corn   ~12% Soybeans  ~6% Wheat
  ~12% Live Cattle  ~6% Feeder Cattle  ~7% Lean Hogs
  (rest cash/T-bills)

ENSO (El Niño/Southern Oscillation) — NOAA ONI (Oceanic Niño Index).
  Source: https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
  ONI = 3-month running mean of ERSSTv5 SST anomalies in Niño 3.4.
  Convention:
    ONI ≥ +0.5°C for 5 consecutive overlapping seasons = El Niño
    Strong El Niño: ONI ≥ +1.5°C
    ONI ≤ -0.5°C = La Niña

Hypothesis: Strong El Niño → coffee/cocoa/sugar supply shock
  (drought in West Africa for cocoa, dry Brazil for coffee/sugar) →
  DBA spike. We measure: DBA 3/6/12-month forward return conditional
  on current ENSO phase.
"""
import os
import sys
import urllib.request
from io import StringIO
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, 'cache')
os.makedirs(CACHE, exist_ok=True)

ONI_URL = 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
SEAS_TO_MO = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
              'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}


def fetch_oni():
    """NOAA ONI history. Cols: SEAS YR TOTAL ANOM (ONI = 3-mo SST anom)."""
    import subprocess
    cache = os.path.join(CACHE, 'oni.csv')
    age_h = (pd.Timestamp.now().timestamp() - os.path.getmtime(cache))/3600 if os.path.exists(cache) else 1e9
    if age_h > 24*7:
        print('[oni] downloading NOAA ONI...')
        raw = os.path.join(CACHE, 'oni_raw.txt')
        subprocess.run(['curl', '-s', '--max-time', '30', '-L', '-o', raw, ONI_URL],
                       check=True)
        rows = []
        with open(raw) as f:
            for line in f.readlines()[1:]:
                parts = line.split()
                if len(parts) != 4:
                    continue
                try:
                    seas, yr, anom = parts[0], int(parts[1]), float(parts[3])
                    mo = SEAS_TO_MO.get(seas)
                    if mo is None:
                        continue
                    rows.append((pd.Timestamp(yr, mo, 1), anom))
                except ValueError:
                    continue
        df = pd.DataFrame(rows, columns=['date', 'anom']).set_index('date').sort_index()
        df.to_csv(cache)
        print(f'[oni] {len(df)} months {df.index.min().date()} → {df.index.max().date()}')
    return pd.read_csv(cache, index_col=0, parse_dates=True)


def fetch_dba():
    """DBA inception 2007-01-05."""
    import yfinance as yf
    t = yf.Ticker('DBA')
    h = t.history(period='max')['Close']
    h.index = h.index.tz_localize(None) if h.index.tz else h.index
    return h


def classify_enso(anom):
    """ONI buckets."""
    if anom >= 1.5:
        return 'strong_nino'
    elif anom >= 0.5:
        return 'weak_nino'
    elif anom <= -1.5:
        return 'strong_nina'
    elif anom <= -0.5:
        return 'weak_nina'
    return 'neutral'


def main():
    oni = fetch_oni()
    oni['phase'] = oni['anom'].apply(classify_enso)
    print('\n=== ONI phase distribution (1950-now) ===')
    print(oni['phase'].value_counts())

    dba = fetch_dba()
    print(f'\nDBA: {len(dba)} bars, {dba.index[0].date()} → {dba.index[-1].date()}, '
          f'last ${dba.iloc[-1]:.2f}')

    # Resample DBA to monthly (last close)
    dba_m = dba.resample('ME').last()
    dba_m.index = dba_m.index.to_period('M').to_timestamp()  # align to month-start

    # Align ONI monthly index to month-start
    oni.index = oni.index.to_period('M').to_timestamp()

    # Forward returns at 3, 6, 12 months
    rets = pd.DataFrame({'dba': dba_m})
    for n in [3, 6, 12]:
        rets[f'fwd_{n}m'] = dba_m.shift(-n) / dba_m - 1

    df = rets.join(oni[['anom', 'phase']], how='inner').dropna(subset=['phase'])

    print('\n=== DBA forward returns by ENSO phase (since 2007) ===')
    summary = df.groupby('phase')[['fwd_3m', 'fwd_6m', 'fwd_12m']].agg(['mean', 'std', 'count'])
    print(summary.round(4))

    # Targeted: strong El Niño → big move?
    print('\n=== Strong El Niño episodes (since DBA inception) ===')
    se = df[df['phase'] == 'strong_nino']
    if len(se):
        print(se[['anom', 'fwd_3m', 'fwd_6m', 'fwd_12m']].to_string())

    # Current state
    latest = oni.iloc[-1]
    print(f'\n=== Current ENSO (as of {oni.index[-1].strftime("%Y-%m")}) ===')
    print(f'ONI = {latest["anom"]:+.2f}°C → {latest["phase"]}')

    # Trajectory: last 6 months
    print('\nLast 6 months ONI:')
    print(oni.tail(6))

    # Save data for later
    df.to_csv(os.path.join(CACHE, 'dba_enso_panel.csv'))
    print(f'\nSaved panel to {CACHE}/dba_enso_panel.csv')

    return df, oni


if __name__ == '__main__':
    main()
