"""Unified daily-refresh data layer for DBA × weather research.

Sources:
  1. NOAA ONI — Niño 3.4 monthly SST anomaly (oni.ascii.txt) [updated monthly]
  2. US Drought Monitor — CONUS DSCI weekly index [updated Thursdays]
     DSCI = Drought Severity & Coverage Index (0-500). 500 = all CONUS in
     D4 exceptional drought. Historical baseline ~80-150.
  3. DBA daily prices (yfinance) — Invesco DB Agriculture
  4. Component ETF prices: JO (coffee), NIB (cocoa), CANE (sugar),
     CORN, SOYB, WEAT, COW (cattle) — for component decomposition

Cron: daily 18:00 ET via refresh_dba_data.sh
"""
import os
import sys
import json
import subprocess
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, 'cache')
os.makedirs(CACHE, exist_ok=True)

ONI_URL = 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
USDM_URL = ('https://usdmdataservices.unl.edu/api/USStatistics/GetDSCI'
            '?aoi=conus&startdate={start}&enddate={end}&statisticsType=1')
SEAS_TO_MO = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
              'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}


def _age_hours(path):
    if not os.path.exists(path):
        return float('inf')
    return (pd.Timestamp.now().timestamp() - os.path.getmtime(path)) / 3600


def fetch_oni(max_age_h=24*3):
    """NOAA ONI seasonal anomaly. Updated monthly ~10th."""
    cache = os.path.join(CACHE, 'oni.csv')
    if _age_hours(cache) > max_age_h:
        print('[oni] downloading...')
        raw = os.path.join(CACHE, 'oni_raw.txt')
        subprocess.run(['curl', '-s', '--max-time', '30', '-L', '-o', raw, ONI_URL], check=True)
        rows = []
        with open(raw) as f:
            for line in f.readlines()[1:]:
                p = line.split()
                if len(p) != 4:
                    continue
                try:
                    mo = SEAS_TO_MO.get(p[0])
                    if mo is None:
                        continue
                    rows.append((pd.Timestamp(int(p[1]), mo, 1), float(p[3])))
                except ValueError:
                    continue
        df = pd.DataFrame(rows, columns=['date', 'oni']).set_index('date').sort_index()
        df.to_csv(cache)
        print(f'[oni] {len(df)} months → {df.index.max().date()}, last={df["oni"].iloc[-1]:+.2f}')
    return pd.read_csv(cache, index_col=0, parse_dates=True)


def fetch_drought(max_age_h=24*3):
    """US Drought Monitor CONUS DSCI weekly. Updated Thursdays."""
    cache = os.path.join(CACHE, 'usdm_dsci.csv')
    if _age_hours(cache) > max_age_h:
        print('[usdm] downloading...')
        # Cover from 2000 (USDM started 2000-01-04)
        url = USDM_URL.format(start='1/1/2000', end='12/31/2026')
        raw = subprocess.check_output(
            ['curl', '-s', '--max-time', '30', '-L',
             '-H', 'Accept: application/json', url]
        ).decode()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: CSV format (default response)
            from io import StringIO
            df = pd.read_csv(StringIO(raw), header=None, names=['name', 'mapDate', 'dsci'])
            df['mapDate'] = pd.to_datetime(df['mapDate'], format='%Y%m%d')
            df = df.set_index('mapDate')[['dsci']].sort_index()
            df.to_csv(cache)
            print(f'[usdm] (csv) {len(df)} weeks → {df.index.max().date()}, last DSCI={df["dsci"].iloc[-1]}')
            return df
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['mapDate'])
        df = df.set_index('date')[['dsci']].sort_index()
        df.to_csv(cache)
        print(f'[usdm] {len(df)} weeks → {df.index.max().date()}, last DSCI={df["dsci"].iloc[-1]}')
    return pd.read_csv(cache, index_col=0, parse_dates=True)


def fetch_etfs(max_age_h=24, tickers=None):
    """DBA + softs/grains/livestock component ETFs (yfinance)."""
    import yfinance as yf
    cache = os.path.join(CACHE, 'etf_prices.csv')
    if tickers is None:
        # JO/NIB/COW were delisted 2023-07; use teucrium ETFs + DBA itself
        tickers = ['DBA', 'CANE', 'CORN', 'SOYB', 'WEAT', 'UNG']
    if _age_hours(cache) > max_age_h:
        print(f'[etf] downloading {tickers}...')
        rows = {}
        for t in tickers:
            try:
                h = yf.Ticker(t).history(period='max')['Close']
                if not h.empty:
                    h.index = h.index.tz_localize(None) if h.index.tz else h.index
                    rows[t] = h
                    print(f'  {t}: {len(h)} bars, {h.index.min().date()}→{h.index.max().date()}, ${h.iloc[-1]:.2f}')
            except Exception as e:
                print(f'  {t}: failed ({e})')
        df = pd.DataFrame(rows)
        df.to_csv(cache)
    return pd.read_csv(cache, index_col=0, parse_dates=True)


def build_master_panel():
    """Daily-indexed panel with all signals forward-filled."""
    oni = fetch_oni()
    dsci = fetch_drought()
    etfs = fetch_etfs()

    # Reindex to daily, forward fill the slow signals
    df = etfs.copy()
    df.index = df.index.tz_localize(None) if df.index.tz else df.index

    if oni is not None:
        df['oni'] = oni['oni'].reindex(df.index, method='ffill')
    if dsci is not None:
        df['dsci'] = dsci['dsci'].reindex(df.index, method='ffill')

    # Derived: ONI 3-month change (regime transition speed)
    df['oni_delta_3m'] = df['oni'] - df['oni'].shift(63)
    # ENSO state buckets
    def _phase(x):
        if pd.isna(x):
            return 'unknown'
        if x >= 1.5:
            return 'strong_nino'
        if x >= 0.5:
            return 'weak_nino'
        if x <= -1.5:
            return 'strong_nina'
        if x <= -0.5:
            return 'weak_nina'
        return 'neutral'
    df['enso_phase'] = df['oni'].apply(_phase)

    # DSCI z-score (rolling 5y baseline)
    if 'dsci' in df.columns:
        base = df['dsci'].rolling(252*5, min_periods=252).mean()
        sd = df['dsci'].rolling(252*5, min_periods=252).std()
        df['dsci_z'] = (df['dsci'] - base) / sd

    out = os.path.join(CACHE, 'master_panel.csv')
    df.to_csv(out)
    print(f'\nMaster panel: {df.shape} → {out}')
    print(f'Coverage: {df.index.min().date()} → {df.index.max().date()}')
    print(f'Latest signals:')
    latest = df.dropna(subset=['oni', 'dsci']).iloc[-1]
    print(f'  ONI={latest["oni"]:+.2f}  phase={latest["enso_phase"]}  '
          f'DSCI={latest["dsci"]:.0f}  dsci_z={latest.get("dsci_z", float("nan")):+.2f}')
    return df


if __name__ == '__main__':
    build_master_panel()
