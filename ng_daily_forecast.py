#!/usr/bin/env python3
"""
NG Daily Forecast & Forward Curve Fair Value
Runs the 10-factor IC-weighted composite model daily, then iterates the AR(2)
partial-adjustment model forward along the futures curve to identify cheap/rich
contracts and estimate UNG contango drag.

Output: ng_daily_forecast.png + console table
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import subprocess
import tempfile
import os
import zipfile
from io import BytesIO
from datetime import datetime
import yfinance as yf
from scipy.stats import spearmanr, percentileofscore, norm as sp_norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from numpy.polynomial.polynomial import polyfit, polyval
import re
import warnings
warnings.filterwarnings('ignore')

print("NG Daily Fair Value Forecast")
print("=" * 65)

# ============================================
# Data Fetching (copied from ng_fair_value_composite.py)
# ============================================
EIA_BASE = 'https://www.eia.gov/dnav/ng/hist_xls/'


def curl_fetch(url, timeout=30):
    """Download URL content using curl (bypasses EIA's Python-UA block)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xls') as f:
        tmp = f.name
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '-o', tmp, '-w', '%{http_code}',
             '-H', 'User-Agent: Mozilla/5.0', url],
            capture_output=True, text=True, timeout=timeout)
        code = result.stdout.strip()
        if code != '200':
            raise RuntimeError(f'HTTP {code}')
        with open(tmp, 'rb') as f:
            return f.read()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fetch_monthly(url, name):
    """Fetch and parse EIA monthly XLS."""
    try:
        print(f"  Fetching {name}...")
        content = curl_fetch(url)
        df = pd.read_excel(BytesIO(content), sheet_name='Data 1', skiprows=2)
        df['date'] = pd.to_datetime(df.iloc[:, 0])
        df['value'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        df = df[df['value'].notna()][['date', 'value']].copy().sort_values('date')
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
        print(f"    {len(df)} records ({df['date'].min():%Y-%m} to {df['date'].max():%Y-%m})")
        return df
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame(columns=['date', 'value'])


def _parse_bh_xlsx(path_or_content):
    """Parse Baker Hughes XLSX for US gas rig count."""
    if isinstance(path_or_content, (str, os.PathLike)):
        df = pd.read_excel(path_or_content, sheet_name='NAM Weekly', skiprows=9)
    else:
        df = pd.read_excel(BytesIO(path_or_content), sheet_name='NAM Weekly', skiprows=9)
    df.columns = df.iloc[0].values
    df = df.iloc[1:].copy()
    us_gas = df[(df['Country'] == 'UNITED STATES') & (df['DrillFor'] == 'Gas')].copy()
    us_gas['date'] = pd.to_datetime(us_gas['US_PublishDate'])
    us_gas['value'] = pd.to_numeric(us_gas['Rig Count Value'], errors='coerce')
    weekly = us_gas.groupby('date')['value'].sum().reset_index().sort_values('date')
    return weekly[['date', 'value']]


def _shell_download(url, dest, label):
    """Download a file using shell curl (avoids HTTP/2 stream errors)."""
    print(f"  Downloading {label}...")
    ret = os.system(f'curl -s -L -o "{dest}" "{url}" 2>/dev/null')
    if ret == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"    OK: {os.path.getsize(dest)} bytes")
        return True
    print(f"    FAILED (exit {ret})")
    return False


def fetch_rig_count():
    """Fetch US gas-directed rig count from Baker Hughes."""
    BH_BASE = 'https://rigcount.bakerhughes.com/static-files/'
    files = [
        (BH_BASE + 'e98bcf83-c458-4a88-8f35-4ac4d77628bb', '/tmp/bh_historical.xlsx', 'BH Historical (2013-2025)'),
        (BH_BASE + '3885a62b-d9b1-4fbf-ad12-5838912a05dd', '/tmp/bh_current.xlsx', 'BH Current Report'),
    ]
    frames = []
    for url, path, label in files:
        need_download = True
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            age_hours = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
            if age_hours < 24:
                print(f"  Using cached {label} ({age_hours:.0f}h old)")
                need_download = False
        if need_download:
            _shell_download(url, path, label)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            try:
                df = _parse_bh_xlsx(path)
                print(f"    {len(df)} weeks ({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d})")
                frames.append(df)
            except Exception as e:
                print(f"    Parse FAILED: {e}")
    if frames:
        combined = pd.concat(frames).drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
        return combined
    return pd.DataFrame(columns=['date', 'value'])


def fetch_storage():
    """Fetch weekly storage from EIA."""
    try:
        print("  Fetching Storage...")
        content = curl_fetch('https://ir.eia.gov/ngs/ngshistory.xls', timeout=45)
        df = pd.read_excel(BytesIO(content), sheet_name='html_report_history', skiprows=5)
        total_col = None
        for col in df.columns:
            cs = str(col).lower()
            if 'total' in cs and ('lower' in cs or '48' in cs):
                total_col = col
                break
        if total_col is None:
            total_col = df.columns[-1]
        result = pd.DataFrame()
        result['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        result['storage_bcf'] = pd.to_numeric(df[total_col], errors='coerce')
        result = result[result['date'].notna() & result['storage_bcf'].notna()].copy()
        result = result.sort_values('date')
        print(f"    {len(result)} records")
        return result
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame(columns=['date', 'storage_bcf'])


def mmcf_to_bcfd(df):
    """Convert MMcf/month to Bcf/d."""
    df = df.copy()
    df['bcfd'] = df['value'] / df['date'].dt.days_in_month / 1000
    return df


def fetch_cot():
    """Fetch CFTC COT disaggregated data for NG managed money positioning."""
    print("  Fetching CFTC COT data...")
    frames = []
    current_year = datetime.now().year
    for year in range(2015, current_year + 1):
        url = f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip'
        dest = f'/tmp/cftc_{year}.zip'
        need_download = True
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            age_hours = (datetime.now().timestamp() - os.path.getmtime(dest)) / 3600
            max_age = 24 if year == current_year else 720
            if age_hours < max_age:
                need_download = False
        if need_download:
            ret = os.system(f'curl -s -L -o "{dest}" "{url}" 2>/dev/null')
            if ret != 0 or not os.path.exists(dest) or os.path.getsize(dest) < 1000:
                continue
        try:
            with zipfile.ZipFile(dest) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
                    ng = df[df['CFTC_Contract_Market_Code'] == '023651'].copy()
                    if len(ng) > 0:
                        frames.append(ng)
        except Exception:
            continue

    if frames:
        cot = pd.concat(frames, ignore_index=True)
        cot['date'] = pd.to_datetime(cot['Report_Date_as_YYYY-MM-DD'])
        cot['mm_net'] = (cot['M_Money_Positions_Long_All'].astype(float) -
                         cot['M_Money_Positions_Short_All'].astype(float))
        # Swap dealer net positioning
        try:
            cot['swap_net'] = (pd.to_numeric(cot['Swap_Positions_Long_All'], errors='coerce') -
                               pd.to_numeric(cot['Swap__Positions_Short_All'], errors='coerce'))
        except KeyError:
            cot['swap_net'] = np.nan
        cot = cot[['date', 'mm_net', 'swap_net']].sort_values('date').drop_duplicates(subset='date')
        print(f"    {len(cot)} weekly reports ({cot['date'].min():%Y-%m-%d} to {cot['date'].max():%Y-%m-%d})")
        return cot
    print("    FAILED: no data")
    return pd.DataFrame(columns=['date', 'mm_net', 'swap_net'])


def fetch_fred(series_id, name):
    """Fetch a FRED series via CSV download (no API key needed)."""
    try:
        print(f"  Fetching FRED {name} ({series_id})...")
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        df = pd.read_csv(url)
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna().sort_values('date')
        print(f"    {len(df)} records ({df['date'].min():%Y-%m} to {df['date'].max():%Y-%m})")
        return df
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame(columns=['date', 'value'])


def fetch_cpc_hdd():
    """Fetch NOAA CPC population-weighted weekly HDD data for current heating season."""
    print("  Fetching CPC Population-Weighted HDD...")
    url = 'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days/wsahddy.txt'
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', url], capture_output=True, text=True, timeout=15)
        if result.returncode != 0 or len(result.stdout) < 500:
            print("    FAILED: curl error")
            return pd.DataFrame(columns=['date', 'hdd_week', 'hdd_dev_norm'])

        lines = result.stdout.split('\n')

        # Extract the last date from header
        last_date = None
        for line in lines[:10]:
            m = re.search(r'LAST DATE.*?IS\s+(\w+\s+\d+,\s+\d+)', line)
            if m:
                last_date = pd.to_datetime(m.group(1))
                break
        if last_date is None:
            print("    FAILED: could not parse date")
            return pd.DataFrame(columns=['date', 'hdd_week', 'hdd_dev_norm'])

        # Parse: find population-weighted section, extract state rows + US total
        # The first section is population-weighted (before "GAS HOME HEATING")
        us_row = None
        state_rows = []
        for line in lines:
            if 'GAS HOME HEATING' in line:
                break  # end of population-weighted section
            stripped = line.strip()
            if stripped.startswith('UNITED STATES'):
                us_row = stripped
            elif stripped and stripped[0].isalpha() and 'REGION' not in stripped and \
                 'STATE' not in stripped and 'HEATING' not in stripped and \
                 'POPULATION' not in stripped and 'CLIMATE' not in stripped and \
                 'LAST' not in stripped and 'ACCUMULATION' not in stripped and \
                 'WEEK' not in stripped and 'TOTAL' not in stripped:
                state_rows.append(stripped)

        if us_row is None:
            print("    FAILED: could not find US row")
            return pd.DataFrame(columns=['date', 'hdd_week', 'hdd_dev_norm'])

        # Parse US row: "UNITED STATES     141  -37  -91    3014  -208    -1    -6     0"
        parts = us_row.split()
        # Find numeric values after "STATES"
        nums = []
        found_states = False
        for p in parts:
            if p == 'STATES':
                found_states = True
                continue
            if found_states:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass

        if len(nums) >= 3:
            hdd_week = nums[0]     # WEEK TOTAL
            hdd_dev = nums[1]      # WEEK DEV FROM NORM
            hdd_cum_dev = nums[4] if len(nums) > 4 else 0  # CUM DEV FROM NORM
            hdd_cum_dev_pct = nums[6] if len(nums) > 6 else 0  # CUM DEV PRCT FROM NORM

            result_df = pd.DataFrame([{
                'date': last_date,
                'hdd_week': hdd_week,
                'hdd_dev_norm': hdd_dev,
                'hdd_cum_dev': hdd_cum_dev,
                'hdd_cum_dev_pct': hdd_cum_dev_pct,
            }])
            print(f"    Week ending {last_date:%Y-%m-%d}: HDD={hdd_week:.0f}, "
                  f"dev={hdd_dev:+.0f}, cum_dev={hdd_cum_dev:+.0f} ({hdd_cum_dev_pct:+.0f}%)")
            return result_df
        else:
            print(f"    FAILED: could not parse US row (got {len(nums)} numbers)")
            return pd.DataFrame(columns=['date', 'hdd_week', 'hdd_dev_norm'])

    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame(columns=['date', 'hdd_week', 'hdd_dev_norm'])


def _parse_cpc_daily_dd(text, dd_type='Cooling'):
    """Parse CPC daily degree day text file (pipe-delimited).
    Format: 3 header lines, then:
      Region|YYYYMMDD|YYYYMMDD|...  (date header, one column per day of year)
      AL|0|0|1|2|...               (state code + daily values)
    Returns DataFrame with columns: date, dd_value (national daily total, summed across states).
    """
    lines = text.strip().split('\n')

    # Find the header row starting with "Region|"
    date_cols = None
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('Region|'):
            parts = stripped.split('|')
            # Parse date columns from header: Region|20250101|20250102|...
            date_cols = []
            for p in parts[1:]:
                p = p.strip()
                if len(p) == 8 and p.isdigit():
                    try:
                        date_cols.append(pd.Timestamp(year=int(p[:4]), month=int(p[4:6]), day=int(p[6:8])))
                    except ValueError:
                        date_cols.append(None)
                else:
                    date_cols.append(None)
            data_start = i + 1
            break

    if date_cols is None or data_start is None:
        return pd.DataFrame(columns=['date', 'dd_value'])

    # Parse state rows: STATE_CODE|val1|val2|...
    # Sum values across all states for each day
    n_days = len(date_cols)
    daily_totals = np.zeros(n_days)
    n_states = 0

    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or '|' not in stripped:
            continue
        parts = stripped.split('|')
        state = parts[0].strip()
        if not state or not state[0].isalpha():
            continue
        # Skip non-state rows
        if state.upper() in ('REGION', 'TOTAL', 'CONUS'):
            continue

        values = parts[1:]
        for j, val_str in enumerate(values):
            if j >= n_days:
                break
            val_str = val_str.strip()
            if val_str and val_str not in ('M', '-', ''):
                try:
                    daily_totals[j] += float(val_str)
                except ValueError:
                    pass
        n_states += 1

    if n_states == 0:
        return pd.DataFrame(columns=['date', 'dd_value'])

    # Build result, filtering out None dates and future/zero-data dates
    records = []
    today = pd.Timestamp.now().normalize()
    for j, dt in enumerate(date_cols):
        if dt is not None and dt <= today:
            records.append({'date': dt, 'dd_value': daily_totals[j]})

    if not records:
        return pd.DataFrame(columns=['date', 'dd_value'])

    df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    return df


def fetch_cpc_daily_dd(dd_type='Cooling'):
    """Fetch CPC population-weighted daily degree day data for years 2005-2026.
    dd_type: 'Cooling' or 'Heating'
    Returns (monthly_df, daily_df) where:
      monthly_df: columns date, dd_monthly, dd_dev, dd_zscore
      daily_df: columns date, dd_value (raw daily national totals)
    """
    label = 'CDD' if dd_type == 'Cooling' else 'HDD'
    print(f"  Fetching CPC Daily {label} (2005-2026)...")

    all_daily = []
    current_year = datetime.now().year

    for year in range(2005, min(current_year + 1, 2027)):
        url = f'https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/{year}/StatesCONUS.{dd_type}.txt'
        dest = f'/tmp/cpc_{dd_type.lower()}_{year}.txt'

        # Cache: reuse if <24h for current year, <30 days for past years
        need_download = True
        if os.path.exists(dest) and os.path.getsize(dest) > 500:
            age_hours = (datetime.now().timestamp() - os.path.getmtime(dest)) / 3600
            max_age = 24 if year == current_year else 720
            if age_hours < max_age:
                need_download = False

        if need_download:
            ret = os.system(f'curl -s -L -o "{dest}" "{url}" 2>/dev/null')
            if ret != 0 or not os.path.exists(dest) or os.path.getsize(dest) < 500:
                continue

        try:
            with open(dest, 'r', errors='replace') as f:
                text = f.read()
            daily = _parse_cpc_daily_dd(text, dd_type)
            if len(daily) > 0:
                all_daily.append(daily)
        except Exception:
            continue

    if not all_daily:
        print(f"    FAILED: no {label} data parsed")
        return pd.DataFrame(columns=['date', 'dd_monthly', 'dd_dev', 'dd_zscore']), pd.DataFrame(columns=['date', 'dd_value'])

    combined = pd.concat(all_daily, ignore_index=True)
    combined = combined.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
    print(f"    {len(combined)} daily {label} records ({combined['date'].min():%Y-%m-%d} to {combined['date'].max():%Y-%m-%d})")

    # Aggregate to monthly totals
    combined['month'] = combined['date'].dt.to_period('M').dt.to_timestamp()
    monthly_dd = combined.groupby('month')['dd_value'].sum().reset_index()
    monthly_dd.columns = ['date', 'dd_monthly']

    # Compute long-term monthly normals (2005-2024 baseline)
    baseline = monthly_dd[monthly_dd['date'] < '2025-01-01'].copy()
    baseline['cal_month'] = baseline['date'].dt.month
    normals = baseline.groupby('cal_month')['dd_monthly'].agg(['mean', 'std']).reset_index()
    normals.columns = ['cal_month', 'dd_normal', 'dd_std']

    # Calculate deviation and z-score
    monthly_dd['cal_month'] = monthly_dd['date'].dt.month
    monthly_dd = monthly_dd.merge(normals, on='cal_month', how='left')
    monthly_dd['dd_dev'] = monthly_dd['dd_monthly'] - monthly_dd['dd_normal']
    monthly_dd['dd_zscore'] = monthly_dd['dd_dev'] / monthly_dd['dd_std'].clip(lower=1.0)

    result = monthly_dd[['date', 'dd_monthly', 'dd_dev', 'dd_zscore']].dropna()
    print(f"    {len(result)} monthly {label} records")
    return result, combined


# ============================================
# Futures Curve Fetch
# ============================================
MONTH_CODES = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
               7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}


def fetch_futures_curve():
    """Fetch NG futures curve from yfinance."""
    print("  Fetching NG futures curve...")
    today = datetime.now()
    contracts = []
    for months_ahead in range(0, 24):
        dt = today + pd.DateOffset(months=months_ahead)
        code = MONTH_CODES[dt.month]
        yr = dt.year % 100
        ticker = f"NG{code}{yr:02d}.NYM"
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if len(hist) > 0:
                price = hist['Close'].iloc[-1]
                if price > 0:
                    contracts.append({
                        'month': dt.to_period('M').to_timestamp(),
                        'ticker': ticker,
                        'price': price,
                        'months_ahead': months_ahead,
                    })
        except Exception:
            continue
    curve = pd.DataFrame(contracts)
    if len(curve) > 0:
        print(f"    {len(curve)} contracts fetched ({curve['ticker'].iloc[0]} to {curve['ticker'].iloc[-1]})")
    else:
        print("    WARNING: No futures contracts fetched")
    return curve


# ============================================
# Fetch all data
# ============================================
print("\n--- EIA Monthly Data ---")
production = fetch_monthly(EIA_BASE + 'N9070US2m.xls', 'Dry Gas Production')
consumption = fetch_monthly(EIA_BASE + 'N9140US2m.xls', 'Total Consumption')
lng_exports = fetch_monthly(EIA_BASE + 'N9133US2m.xls', 'LNG Exports')
pipeline_exports = fetch_monthly(EIA_BASE + 'N9132US2m.xls', 'Pipeline Exports')
lng_imports = fetch_monthly(EIA_BASE + 'N9103US2m.xls', 'LNG Imports')
pipeline_imports = fetch_monthly(EIA_BASE + 'N9102US2m.xls', 'Pipeline Imports')
power_burn = fetch_monthly(EIA_BASE + 'N3045US2m.xls', 'Power Burn')

print("\n--- Weekly Data ---")
rig_count = fetch_rig_count()
storage = fetch_storage()

print("\n--- CFTC COT ---")
cot = fetch_cot()

print("\n--- Weather (CPC) ---")
cpc_hdd = fetch_cpc_hdd()
cpc_cdd_monthly, cpc_cdd_daily_raw = fetch_cpc_daily_dd('Cooling')
cpc_hdd_monthly, cpc_hdd_daily_raw = fetch_cpc_daily_dd('Heating')

print("\n--- FRED Macro ---")
indpro = fetch_fred('INDPRO', 'Industrial Production')

print("\n--- Market Data (yfinance) ---")
ng = yf.Ticker("NG=F")
ng_daily = ng.history(period="max", interval="1d")
if len(ng_daily) > 0 and hasattr(ng_daily.index, 'tz') and ng_daily.index.tz is not None:
    ng_daily.index = ng_daily.index.tz_localize(None)
ng_daily = ng_daily[['Close']].rename(columns={'Close': 'ng_price'})
ng_daily.index.name = 'date'
ng_daily = ng_daily.reset_index()
print(f"  NG=F: {len(ng_daily)} daily bars")

oil = yf.Ticker("CL=F")
oil_daily = oil.history(period="max", interval="1d")
if len(oil_daily) > 0 and hasattr(oil_daily.index, 'tz') and oil_daily.index.tz is not None:
    oil_daily.index = oil_daily.index.tz_localize(None)
oil_daily = oil_daily[['Close']].rename(columns={'Close': 'oil_price'})
oil_daily.index.name = 'date'
oil_daily = oil_daily.reset_index()
print(f"  CL=F: {len(oil_daily)} daily bars")

dxy = yf.Ticker("DX=F")
dxy_daily = dxy.history(period="max", interval="1d")
if len(dxy_daily) > 0 and hasattr(dxy_daily.index, 'tz') and dxy_daily.index.tz is not None:
    dxy_daily.index = dxy_daily.index.tz_localize(None)
if len(dxy_daily) > 0:
    dxy_daily = dxy_daily[['Close']].rename(columns={'Close': 'dxy'})
    dxy_daily.index.name = 'date'
    dxy_daily = dxy_daily.reset_index()
else:
    dxy_daily = pd.DataFrame(columns=['date', 'dxy'])
print(f"  DX=F: {len(dxy_daily)} daily bars")

ttf = yf.Ticker("TTF=F")
ttf_daily = ttf.history(period="max", interval="1d")
if len(ttf_daily) > 0 and hasattr(ttf_daily.index, 'tz') and ttf_daily.index.tz is not None:
    ttf_daily.index = ttf_daily.index.tz_localize(None)
if len(ttf_daily) > 0:
    ttf_daily = ttf_daily[ttf_daily['Close'] > 0][['Close']].rename(columns={'Close': 'ttf_price'})
    ttf_daily.index.name = 'date'
    ttf_daily = ttf_daily.reset_index()
else:
    ttf_daily = pd.DataFrame(columns=['date', 'ttf_price'])
print(f"  TTF=F: {len(ttf_daily)} daily bars")

print("\n--- Term Structure (EIA Futures Contracts) ---")
c1_content = curl_fetch(EIA_BASE + 'RNGC1d.xls')
c4_content = curl_fetch(EIA_BASE + 'RNGC4d.xls')

ts_slope_daily = pd.DataFrame()
if c1_content and c4_content:
    c1 = pd.read_excel(BytesIO(c1_content), sheet_name='Data 1', skiprows=2)
    c4 = pd.read_excel(BytesIO(c4_content), sheet_name='Data 1', skiprows=2)
    for df, name in [(c1, 'c1'), (c4, 'c4')]:
        df.columns = ['date', name]
        df['date'] = pd.to_datetime(df['date'])
        df[name] = pd.to_numeric(df[name], errors='coerce')
    ts = c1.merge(c4, on='date', how='inner').dropna()
    ts['ts_slope'] = (ts['c1'] - ts['c4']) / ts['c4'] * 100
    ts_slope_daily = ts[['date', 'ts_slope', 'c1', 'c4']].sort_values('date')
    print(f"  C1-C4 slope: {len(ts_slope_daily)} days ({ts_slope_daily['date'].min():%Y-%m-%d} to {ts_slope_daily['date'].max():%Y-%m-%d})")

    # Extend beyond EIA data end using NG=F proxy
    eia_end = ts_slope_daily['date'].max()
    recent_ng = ng_daily[ng_daily['date'] > eia_end].copy()
    if len(recent_ng) > 0:
        ng_daily_temp = ng_daily.copy()
        ng_daily_temp['cal_month'] = ng_daily_temp['date'].dt.month
        seasonal_med = ng_daily_temp.groupby('cal_month')['ng_price'].median()
        overall_med = ng_daily_temp['ng_price'].median()
        seasonal_idx_daily = seasonal_med / overall_med

        recent_ng['cal_month'] = recent_ng['date'].dt.month
        recent_ng['seas_idx'] = recent_ng['cal_month'].map(seasonal_idx_daily)
        recent_ng['seas_price'] = overall_med * recent_ng['seas_idx']
        recent_ng['ts_slope'] = (recent_ng['ng_price'] - recent_ng['seas_price']) / recent_ng['seas_price'] * 100
        eia_std = ts_slope_daily['ts_slope'].std()
        proxy_std = recent_ng['ts_slope'].std()
        if proxy_std > 0:
            recent_ng['ts_slope'] = recent_ng['ts_slope'] * (eia_std / proxy_std)
        extended = recent_ng[['date', 'ts_slope']].copy()
        extended['c1'] = np.nan
        extended['c4'] = np.nan
        ts_slope_daily = pd.concat([ts_slope_daily, extended], ignore_index=True).sort_values('date')
        print(f"  Extended with proxy to {ts_slope_daily['date'].max():%Y-%m-%d}")
else:
    print("  FAILED to fetch term structure data")

print("\n--- Futures Curve ---")
futures_curve = fetch_futures_curve()

# ============================================
# Build monthly datasets (same as composite model)
# ============================================
print("\n--- Building Monthly Factors ---")

datasets = {
    'production': production, 'consumption': consumption,
    'lng_exports': lng_exports, 'pipeline_exports': pipeline_exports,
    'lng_imports': lng_imports, 'pipeline_imports': pipeline_imports,
}
for name in datasets:
    if not datasets[name].empty:
        datasets[name] = mmcf_to_bcfd(datasets[name])

prod = datasets['production']
cons = datasets['consumption']
lng_exp = datasets['lng_exports']
pipe_exp = datasets['pipeline_exports']
lng_imp = datasets['lng_imports']
pipe_imp = datasets['pipeline_imports']

monthly = prod[['date', 'bcfd']].rename(columns={'bcfd': 'prod'}).copy()
for col_name, df in [
    ('cons', cons), ('lng_exp', lng_exp), ('pipe_exp', pipe_exp),
    ('lng_imp', lng_imp), ('pipe_imp', pipe_imp),
]:
    if not df.empty:
        monthly = monthly.merge(df[['date', 'bcfd']].rename(columns={'bcfd': col_name}),
                                on='date', how='outer')
    else:
        monthly[col_name] = np.nan

monthly = monthly.sort_values('date').reset_index(drop=True)
for col in ['lng_imp', 'pipe_imp', 'lng_exp', 'pipe_exp']:
    monthly[col] = monthly[col].fillna(0)

monthly['total_supply'] = monthly['prod'] + monthly['lng_imp'] + monthly['pipe_imp']
monthly['total_demand'] = monthly['cons'] + monthly['lng_exp'] + monthly['pipe_exp']
monthly['balance'] = monthly['total_supply'] - monthly['total_demand']

# NG price — monthly
ng_daily['month'] = ng_daily['date'].dt.to_period('M')
ng_monthly = ng_daily.groupby('month')['ng_price'].last().reset_index()
ng_monthly['date'] = ng_monthly['month'].dt.to_timestamp()
ng_monthly = ng_monthly[['date', 'ng_price']].sort_values('date')

# Seasonal index
ng_monthly['cal_month'] = ng_monthly['date'].dt.month
ng_monthly['seasonal_idx'] = np.nan
for m in range(1, 13):
    mask = ng_monthly['cal_month'] == m
    ng_monthly.loc[mask, 'seasonal_idx'] = (
        ng_monthly.loc[mask, 'ng_price']
        .expanding(min_periods=3)
        .median()
        .shift(1)
    )
ng_monthly['ng_deseas'] = ng_monthly['ng_price'] / ng_monthly['seasonal_idx']

# ============================================
# Factor construction (same 10 factors)
# ============================================

# Factor 1: Rig Count Momentum
if not rig_count.empty:
    rig_m = rig_count.copy()
    rig_m['month'] = rig_m['date'].dt.to_period('M').dt.to_timestamp()
    rig_m = rig_m.groupby('month')['value'].mean().reset_index()
    rig_m.columns = ['date', 'rig_count']
    rig_m['rig_ma3'] = rig_m['rig_count'].rolling(3).mean()
    rig_m['rig_ma6'] = rig_m['rig_count'].rolling(6).mean()
    rig_m['rig_momentum'] = rig_m['rig_ma3'] / rig_m['rig_ma6']
    rig_factor = rig_m[['date', 'rig_momentum']].dropna()
    print(f"  Rig Momentum: {len(rig_factor)} months")
else:
    rig_factor = pd.DataFrame(columns=['date', 'rig_momentum'])

# Factor 2: Export Tightening
if not lng_exp.empty and not prod.empty:
    exp_tight = monthly[['date', 'prod']].copy()
    exp_tight = exp_tight.merge(
        lng_exp[['date', 'bcfd']].rename(columns={'bcfd': 'lng_exp_bcfd'}),
        on='date', how='inner')
    exp_tight['prod_yoy'] = exp_tight['prod'].pct_change(12) * 100
    exp_tight['lng_yoy'] = exp_tight['lng_exp_bcfd'].pct_change(12) * 100
    exp_tight['export_tightening'] = exp_tight['lng_yoy'] - exp_tight['prod_yoy']
    exp_tight = exp_tight[['date', 'export_tightening']].dropna()
    print(f"  Export Tightening: {len(exp_tight)} months")
else:
    exp_tight = pd.DataFrame(columns=['date', 'export_tightening'])

# Factor 3: Storage Deviation
WORKING_GAS_CAPACITY = 4500  # Bcf, US working gas capacity
if not storage.empty:
    stor = storage.copy()
    stor['year'] = stor['date'].dt.year
    stor['week'] = stor['date'].dt.isocalendar().week.astype(int)
    stor_list = []
    for _, row in stor.iterrows():
        yr, wk = row['year'], row['week']
        hist = stor[(stor['year'] >= yr - 5) & (stor['year'] < yr) & (stor['week'] == wk)]
        if len(hist) >= 3:
            avg_5yr = hist['storage_bcf'].mean()
            dev = (row['storage_bcf'] - avg_5yr) / avg_5yr * 100
            stor_list.append({'date': row['date'], 'storage_dev': dev})
    stor_dev_weekly = pd.DataFrame(stor_list)  # keep weekly version for daily model
    if not stor_dev_weekly.empty:
        stor_dev = stor_dev_weekly.copy()
        stor_dev['month'] = stor_dev['date'].dt.to_period('M').dt.to_timestamp()
        stor_dev = stor_dev.groupby('month')['storage_dev'].last().reset_index()
        stor_dev.columns = ['date', 'storage_dev']
        print(f"  Storage Deviation: {len(stor_dev)} months ({len(stor_dev_weekly)} weekly)")
    else:
        stor_dev = pd.DataFrame(columns=['date', 'storage_dev'])
        stor_dev_weekly = pd.DataFrame(columns=['date', 'storage_dev'])
else:
    stor_dev = pd.DataFrame(columns=['date', 'storage_dev'])
    stor_dev_weekly = pd.DataFrame(columns=['date', 'storage_dev'])

# ----- Enhanced Storage Factors -----
# Pre-compute weekly storage helpers used by the new factors.
# Bullish convention for all factors here: positive value = bullish for NG.
inj_pace_weekly = pd.DataFrame(columns=['date', 'weekly_change', 'avg_5yr_change',
                                        'injection_surprise', 'injection_pace'])
pct_full_weekly = pd.DataFrame(columns=['date', 'storage_bcf', 'pct_full'])
days_supply_weekly = pd.DataFrame(columns=['date', 'storage_bcf', 'days_supply'])
trajectory_weekly = pd.DataFrame(columns=['date', 'projected_peak_dev'])

inj_pace = pd.DataFrame(columns=['date', 'injection_pace'])
pct_full_factor = pd.DataFrame(columns=['date', 'pct_full'])
days_supply_factor = pd.DataFrame(columns=['date', 'days_supply'])
trajectory = pd.DataFrame(columns=['date', 'projected_peak_dev'])

if not storage.empty:
    storage_sorted = storage.sort_values('date').reset_index(drop=True).copy()
    storage_sorted['weekly_change'] = storage_sorted['storage_bcf'].diff()
    storage_sorted['year'] = storage_sorted['date'].dt.year
    storage_sorted['week'] = storage_sorted['date'].dt.isocalendar().week.astype(int)

    # --- Factor: Weekly Injection Pace (vs 5-year avg) ---
    # Smaller-than-normal injection (or larger withdrawal) = bullish (positive).
    inj_pace_list = []
    for _, row in storage_sorted.iterrows():
        yr, wk = row['year'], row['week']
        if pd.isna(row['weekly_change']):
            continue
        hist = storage_sorted[
            (storage_sorted['year'] >= yr - 5) &
            (storage_sorted['year'] < yr) &
            (storage_sorted['week'] == wk) &
            storage_sorted['weekly_change'].notna()
        ]
        if len(hist) >= 3:
            avg_5yr_change = hist['weekly_change'].mean()
            surprise = row['weekly_change'] - avg_5yr_change  # Bcf
            # Bullish sign convention: smaller injection / bigger draw = bullish
            injection_pace = -surprise / 100.0  # scale to roughly unit range
            inj_pace_list.append({
                'date': row['date'],
                'weekly_change': row['weekly_change'],
                'avg_5yr_change': avg_5yr_change,
                'injection_surprise': surprise,
                'injection_pace': injection_pace,
            })
    if inj_pace_list:
        inj_pace_weekly = pd.DataFrame(inj_pace_list)
        inj_pace_m = inj_pace_weekly.copy()
        inj_pace_m['month'] = inj_pace_m['date'].dt.to_period('M').dt.to_timestamp()
        inj_pace = inj_pace_m.groupby('month')['injection_pace'].last().reset_index()
        inj_pace.columns = ['date', 'injection_pace']
        print(f"  Injection Pace: {len(inj_pace)} months ({len(inj_pace_weekly)} weekly)")
    else:
        print("  Injection Pace: insufficient history")

    # --- Factor: Working Gas % Full ---
    # Higher = more oversupply = bearish, so we sign as -1 in factor_defs.
    pct_full_weekly = storage_sorted[['date', 'storage_bcf']].copy()
    pct_full_weekly['pct_full'] = pct_full_weekly['storage_bcf'] / WORKING_GAS_CAPACITY * 100
    pct_full_m = pct_full_weekly.copy()
    pct_full_m['month'] = pct_full_m['date'].dt.to_period('M').dt.to_timestamp()
    pct_full_factor = pct_full_m.groupby('month')['pct_full'].last().reset_index()
    pct_full_factor.columns = ['date', 'pct_full']
    print(f"  Working Gas % Full: {len(pct_full_factor)} months")

    # --- Factor: Storage Days-of-Supply ---
    # Days = storage_bcf / daily consumption (bcf/day). Higher = more buffer = bearish.
    if 'monthly' in dir() and not monthly.empty and 'cons' in monthly.columns:
        cons_monthly = monthly[['date', 'cons']].dropna().copy()
        cons_monthly['month'] = cons_monthly['date'].dt.to_period('M').dt.to_timestamp()
        cons_monthly = cons_monthly[['month', 'cons']].rename(columns={'cons': 'cons_bcfd'})
        ds = storage_sorted[['date', 'storage_bcf']].copy()
        ds['month'] = ds['date'].dt.to_period('M').dt.to_timestamp()
        ds = ds.merge(cons_monthly, on='month', how='left')
        ds['cons_bcfd'] = ds['cons_bcfd'].ffill()
        ds['days_supply'] = ds['storage_bcf'] / ds['cons_bcfd']
        days_supply_weekly = ds[['date', 'storage_bcf', 'days_supply']].dropna().copy()
        if not days_supply_weekly.empty:
            ds_m = days_supply_weekly.copy()
            ds_m['month'] = ds_m['date'].dt.to_period('M').dt.to_timestamp()
            days_supply_factor = ds_m.groupby('month')['days_supply'].last().reset_index()
            days_supply_factor.columns = ['date', 'days_supply']
            print(f"  Storage Days of Supply: {len(days_supply_factor)} months")
        else:
            print("  Storage Days of Supply: no overlap with consumption data")
    else:
        print("  Storage Days of Supply: monthly consumption unavailable")

    # --- Factor: Storage Trajectory to End-of-October ---
    # Project current level vs historical end-of-October peak.
    # current_dev_pct < 0 (current below avg peak) = bullish (positive sign).
    # We invert so positive = bullish: projected_peak_dev = -(current - avg_peak)/avg_peak * 100
    trajectory_list = []
    for _, row in storage_sorted.iterrows():
        current_date = row['date']
        if not (4 <= current_date.month <= 10):
            continue
        yr = current_date.year
        hist_end_oct = storage_sorted[
            (storage_sorted['date'].dt.month == 10) &
            (storage_sorted['date'].dt.day >= 25) &
            (storage_sorted['year'] >= yr - 5) &
            (storage_sorted['year'] < yr)
        ]
        if len(hist_end_oct) > 0:
            avg_peak = hist_end_oct['storage_bcf'].mean()
            # Negative dev (currently below historical peak) is bullish, so flip sign.
            current_dev_pct = -(row['storage_bcf'] - avg_peak) / avg_peak * 100
            trajectory_list.append({
                'date': current_date,
                'projected_peak_dev': current_dev_pct,
            })
    if trajectory_list:
        trajectory_weekly = pd.DataFrame(trajectory_list)
        traj_m = trajectory_weekly.copy()
        traj_m['month'] = traj_m['date'].dt.to_period('M').dt.to_timestamp()
        trajectory = traj_m.groupby('month')['projected_peak_dev'].last().reset_index()
        trajectory.columns = ['date', 'projected_peak_dev']
        print(f"  Storage Trajectory: {len(trajectory)} months (injection-season only)")
    else:
        print("  Storage Trajectory: no injection-season data")

# Factor 4: S/D Balance
sd_balance = monthly[['date', 'balance']].dropna().copy()
print(f"  S/D Balance: {len(sd_balance)} months")

# Factor 5: Oil/NG Ratio
oil_ng = ng_daily[['date', 'ng_price']].merge(oil_daily[['date', 'oil_price']], on='date', how='inner')
oil_ng['oil_ng_ratio'] = oil_ng['oil_price'] / oil_ng['ng_price']
oil_ng['month'] = oil_ng['date'].dt.to_period('M').dt.to_timestamp()
oil_ng_m = oil_ng.groupby('month')['oil_ng_ratio'].last().reset_index()
oil_ng_m.columns = ['date', 'oil_ng_ratio']
print(f"  Oil/NG Ratio: {len(oil_ng_m)} months")

# Factor 6: DXY Momentum
if len(dxy_daily) == 0:
    dxy_m = pd.DataFrame(columns=['date', 'dxy_roc'])
    print("  DXY 3m ROC: 0 months (DXY unavailable)")
else:
    dxy_daily['month'] = dxy_daily['date'].dt.to_period('M').dt.to_timestamp()
    dxy_m = dxy_daily.groupby('month')['dxy'].last().reset_index()
    dxy_m.columns = ['date', 'dxy']
    dxy_m['dxy_roc'] = dxy_m['dxy'].pct_change(3) * 100
    dxy_m = dxy_m[['date', 'dxy_roc']].dropna()
    print(f"  DXY 3m ROC: {len(dxy_m)} months")

# Factor 7: Realized Vol Percentile
ng_daily['ret'] = ng_daily['ng_price'].pct_change()
ng_daily['rvol_30d'] = ng_daily['ret'].rolling(21).std() * np.sqrt(252) * 100
ng_daily['month'] = ng_daily['date'].dt.to_period('M').dt.to_timestamp()
rvol_m = ng_daily.groupby('month')['rvol_30d'].last().reset_index()
rvol_m.columns = ['date', 'rvol']
rvol_m['rvol_pctile'] = rvol_m['rvol'].rolling(36, min_periods=12).apply(
    lambda x: percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50)
rvol_m = rvol_m[['date', 'rvol_pctile']].dropna()
print(f"  Realized Vol Pctile: {len(rvol_m)} months")

# Factor 8: COT Positioning
if not cot.empty:
    cot['month'] = cot['date'].dt.to_period('M').dt.to_timestamp()
    cot_m = cot.groupby('month')['mm_net'].last().reset_index()
    cot_m.columns = ['date', 'cot_mm_net']
    cot_m['cot_pctile'] = cot_m['cot_mm_net'].rolling(156, min_periods=52).apply(
        lambda x: percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50)
    cot_m = cot_m[['date', 'cot_pctile']].dropna()
    print(f"  COT MM Percentile: {len(cot_m)} months")
else:
    cot_m = pd.DataFrame(columns=['date', 'cot_pctile'])

# Factor 9: Industrial Production YoY
if not indpro.empty:
    indpro['date'] = indpro['date'].dt.to_period('M').dt.to_timestamp()
    indpro_m = indpro.groupby('date')['value'].last().reset_index()
    indpro_m['indpro_yoy'] = indpro_m['value'].pct_change(12) * 100
    indpro_m = indpro_m[['date', 'indpro_yoy']].dropna()
    print(f"  IndPro YoY: {len(indpro_m)} months")
else:
    indpro_m = pd.DataFrame(columns=['date', 'indpro_yoy'])

# Factor: Power Burn YoY (Natural Gas Deliveries to Electric Power Consumers)
# Higher gas-for-power burn vs same month a year ago = bullish demand.
# Uses Bcf/d normalization to handle different days-in-month, then 3mo trailing
# average to smooth weather/outage noise before the YoY comparison.
if not power_burn.empty:
    pb = power_burn.copy()
    pb['date'] = pb['date'].dt.to_period('M').dt.to_timestamp()
    pb = pb.groupby('date')['value'].last().reset_index()
    pb['bcfd'] = pb['value'] / pb['date'].dt.days_in_month / 1000
    pb['bcfd_smooth'] = pb['bcfd'].rolling(3, min_periods=2).mean()
    pb['power_burn_yoy'] = (pb['bcfd_smooth'] / pb['bcfd_smooth'].shift(12) - 1) * 100
    power_burn_m = pb[['date', 'power_burn_yoy']].dropna()
    print(f"  Power Burn YoY: {len(power_burn_m)} months "
          f"(range {power_burn_m['power_burn_yoy'].min():.1f} to {power_burn_m['power_burn_yoy'].max():.1f}%)")

    # Factor: Power Burn Anomaly (deseasonalized, summer-relevant)
    # For each month, compare burn to the same calendar-month rolling 5-year mean.
    # Captures heat-wave-driven demand spikes that pure YoY misses (e.g. when both
    # this year and last year were hot, the YoY is flat but the anomaly stays high).
    pb['cal_month'] = pb['date'].dt.month
    # Rolling 5-year same-month stats: shift-then-rolling within each calendar-month group
    def _same_month_anomaly(group):
        # group is sorted by date; for each row, compare to mean of prior 5 same-month values
        shifted = group['bcfd'].shift(1)  # exclude current month from baseline
        roll_mean = shifted.rolling(5, min_periods=3).mean()
        roll_std = shifted.rolling(5, min_periods=3).std()
        return (group['bcfd'] - roll_mean) / roll_std
    pb_sorted = pb.sort_values('date').reset_index(drop=True)
    pb_sorted['power_burn_anomaly'] = pb_sorted.groupby('cal_month', group_keys=False).apply(_same_month_anomaly)
    power_burn_anom_m = pb_sorted[['date', 'power_burn_anomaly']].dropna()
    print(f"  Power Burn Anomaly: {len(power_burn_anom_m)} months "
          f"(range {power_burn_anom_m['power_burn_anomaly'].min():.2f} to {power_burn_anom_m['power_burn_anomaly'].max():.2f} z)")
else:
    power_burn_m = pd.DataFrame(columns=['date', 'power_burn_yoy'])
    power_burn_anom_m = pd.DataFrame(columns=['date', 'power_burn_anomaly'])

# Factor 10: Term Structure Slope
if not ts_slope_daily.empty:
    ts_slope_daily['month'] = ts_slope_daily['date'].dt.to_period('M').dt.to_timestamp()
    ts_m = ts_slope_daily.groupby('month')['ts_slope'].last().reset_index()
    ts_m.columns = ['date', 'ts_slope']
    print(f"  Term Structure Slope: {len(ts_m)} months")
else:
    ts_m = pd.DataFrame(columns=['date', 'ts_slope'])

# Factor 11: Non-Linear (Convex) Storage Deviation
if not stor_dev.empty:
    stor_convex = stor_dev.copy()
    DEFICIT_THRESHOLD = 5.0  # % deviation threshold
    stor_convex['storage_convex'] = stor_convex['storage_dev'].apply(
        lambda x: -(max(0, -x - DEFICIT_THRESHOLD))**2 / 100.0 if x < -DEFICIT_THRESHOLD else
                   (max(0, x - DEFICIT_THRESHOLD))**2 / 100.0 if x > DEFICIT_THRESHOLD else 0)
    stor_convex = stor_convex[['date', 'storage_convex']].copy()
    print(f"  Storage Convex: {(stor_convex['storage_convex'] != 0).sum()} months with non-zero signal")
else:
    stor_convex = pd.DataFrame(columns=['date', 'storage_convex'])

# --- Diagnostics for the new storage factors (latest weekly print) ---
if not inj_pace_weekly.empty:
    last = inj_pace_weekly.dropna(subset=['injection_pace']).iloc[-1]
    wk = int(last.get('date', pd.Timestamp.today()).isocalendar().week) if pd.notna(last['date']) else 0
    print(f"  Injection Pace (week {wk}): actual={last['weekly_change']:+.0f} Bcf, "
          f"5yr avg={last['avg_5yr_change']:+.0f} Bcf, surprise={last['injection_surprise']:+.0f} Bcf "
          f"(bullish factor={last['injection_pace']:+.2f})")
if not pct_full_weekly.empty:
    last_pf = pct_full_weekly.iloc[-1]
    yr = last_pf['date'].year
    wk = int(last_pf['date'].isocalendar().week)
    hist_pf = pct_full_weekly[
        (pct_full_weekly['date'].dt.year >= yr - 5) &
        (pct_full_weekly['date'].dt.year < yr) &
        (pct_full_weekly['date'].dt.isocalendar().week == wk)
    ]
    avg_pf = hist_pf['pct_full'].mean() if len(hist_pf) >= 3 else float('nan')
    print(f"  Working Gas % Full: {last_pf['pct_full']:.1f}% (vs 5yr avg "
          f"{avg_pf:.1f}%)" if pd.notna(avg_pf) else
          f"  Working Gas % Full: {last_pf['pct_full']:.1f}%")
if not days_supply_weekly.empty:
    last_ds = days_supply_weekly.iloc[-1]
    yr = last_ds['date'].year
    wk = int(last_ds['date'].isocalendar().week)
    hist_ds = days_supply_weekly[
        (days_supply_weekly['date'].dt.year >= yr - 5) &
        (days_supply_weekly['date'].dt.year < yr) &
        (days_supply_weekly['date'].dt.isocalendar().week == wk)
    ]
    avg_ds = hist_ds['days_supply'].mean() if len(hist_ds) >= 3 else float('nan')
    print(f"  Days of Supply: {last_ds['days_supply']:.1f} days (vs 5yr avg "
          f"{avg_ds:.1f})" if pd.notna(avg_ds) else
          f"  Days of Supply: {last_ds['days_supply']:.1f} days")
if not trajectory_weekly.empty:
    last_tr = trajectory_weekly.iloc[-1]
    print(f"  Storage Trajectory: projected_peak_dev={last_tr['projected_peak_dev']:+.1f}% "
          f"(positive=below 5yr peak=bullish, as of {last_tr['date']:%Y-%m-%d})")

# Factor 12: Coal Switching Proximity (derived from NG price — no new data)
coal_switch = ng_monthly[['date', 'ng_price']].copy()
COAL_SWITCH_THRESHOLD = 2.42  # $/MMBtu PRB coal parity
coal_switch['coal_switch'] = 1 / (1 + np.exp(2 * (coal_switch['ng_price'] - COAL_SWITCH_THRESHOLD)))
coal_switch = coal_switch[['date', 'coal_switch']].dropna()
print(f"  Coal Switch Proximity: {len(coal_switch)} months")

# Factor 13: Combined Weather Demand (HDD winter + CDD summer)
# Uses CPC daily data aggregated to monthly. HDD deviation for Oct-Mar, CDD deviation for Apr-Sep.
# Positive deviation = more heating/cooling demand = bullish for NG.
weather_demand = pd.DataFrame(columns=['date', 'weather_demand'])
if not cpc_hdd_monthly.empty or not cpc_cdd_monthly.empty:
    wd_frames = []
    # HDD for winter months (Oct-Mar): positive dev = colder = bullish
    if not cpc_hdd_monthly.empty:
        hdd_m = cpc_hdd_monthly.copy()
        hdd_winter = hdd_m[hdd_m['date'].dt.month.isin([10, 11, 12, 1, 2, 3])].copy()
        hdd_winter = hdd_winter[['date', 'dd_zscore']].rename(columns={'dd_zscore': 'weather_demand'})
        wd_frames.append(hdd_winter)
    # CDD for summer months (Apr-Sep): positive dev = hotter = bullish
    if not cpc_cdd_monthly.empty:
        cdd_m = cpc_cdd_monthly.copy()
        cdd_summer = cdd_m[cdd_m['date'].dt.month.isin([4, 5, 6, 7, 8, 9])].copy()
        cdd_summer = cdd_summer[['date', 'dd_zscore']].rename(columns={'dd_zscore': 'weather_demand'})
        wd_frames.append(cdd_summer)
    if wd_frames:
        weather_demand = pd.concat(wd_frames, ignore_index=True).sort_values('date').drop_duplicates(subset='date')
        print(f"  Weather Demand (HDD+CDD): {len(weather_demand)} months")

        # Print diagnostics for current/recent months
        now = pd.Timestamp(datetime.now().replace(day=1))
        recent_wd = weather_demand[weather_demand['date'] >= now - pd.DateOffset(months=2)]
        for _, row in recent_wd.iterrows():
            m = row['date'].month
            dd_type = 'HDD' if m in [10, 11, 12, 1, 2, 3] else 'CDD'
            # Look up the raw values
            if dd_type == 'CDD' and not cpc_cdd_monthly.empty:
                raw = cpc_cdd_monthly[cpc_cdd_monthly['date'] == row['date']]
                if len(raw) > 0:
                    print(f"    {row['date']:%Y-%m} CDD: {raw.iloc[0]['dd_monthly']:.0f} "
                          f"(dev={raw.iloc[0]['dd_dev']:+.0f}, z={row['weather_demand']:+.2f})")
            elif dd_type == 'HDD' and not cpc_hdd_monthly.empty:
                raw = cpc_hdd_monthly[cpc_hdd_monthly['date'] == row['date']]
                if len(raw) > 0:
                    print(f"    {row['date']:%Y-%m} HDD: {raw.iloc[0]['dd_monthly']:.0f} "
                          f"(dev={raw.iloc[0]['dd_dev']:+.0f}, z={row['weather_demand']:+.2f})")
    else:
        print("  Weather Demand: no data available")
else:
    print("  Weather Demand: no CPC daily data available")

# Factor 14: Swap Dealer Positioning
if not cot.empty and 'swap_net' in cot.columns:
    cot_swap_m = cot.copy()
    cot_swap_m['month'] = cot_swap_m['date'].dt.to_period('M').dt.to_timestamp()
    cot_swap_m = cot_swap_m.groupby('month')['swap_net'].last().reset_index()
    cot_swap_m.columns = ['date', 'swap_net']
    cot_swap_m['swap_pctile'] = cot_swap_m['swap_net'].rolling(156, min_periods=52).apply(
        lambda x: percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50)
    cot_swap_m = cot_swap_m[['date', 'swap_pctile']].dropna()
    print(f"  Swap Dealer Pctile: {len(cot_swap_m)} months")
else:
    cot_swap_m = pd.DataFrame(columns=['date', 'swap_pctile'])

# Factor 15: TTF/HH Ratio (LNG Export Incentive)
# Higher TTF/HH ratio = more incentive to export US LNG = bullish for HH
if len(ttf_daily) > 0 and len(ng_daily) > 0:
    ttf_hh = ng_daily[['date', 'ng_price']].merge(ttf_daily[['date', 'ttf_price']], on='date', how='inner')
    ttf_hh['ttf_hh_ratio'] = ttf_hh['ttf_price'] / ttf_hh['ng_price']
    ttf_hh['month'] = ttf_hh['date'].dt.to_period('M').dt.to_timestamp()
    ttf_hh_m = ttf_hh.groupby('month')['ttf_hh_ratio'].last().reset_index()
    ttf_hh_m.columns = ['date', 'ttf_hh_ratio']
    print(f"  TTF/HH Ratio: {len(ttf_hh_m)} months")
else:
    ttf_hh_m = pd.DataFrame(columns=['date', 'ttf_hh_ratio'])
    print("  TTF/HH Ratio: 0 months (TTF data unavailable)")

# ============================================
# Merge all factors
# ============================================
print("\n--- Merging All Factors ---")
cutoff = pd.Timestamp('2015-01-01')

master = ng_monthly[['date', 'ng_price', 'ng_deseas', 'seasonal_idx']].copy()
master = master[master['date'] >= cutoff]

factor_defs = [
    (rig_factor, 'rig_momentum', 'Rig Momentum', -1, 0),
    (exp_tight, 'export_tightening', 'Export Tightening', 1, 2),
    (stor_dev, 'storage_dev', 'Storage Deviation', 1, 0),
    (sd_balance, 'balance', 'S/D Balance', -1, 2),
    (oil_ng_m, 'oil_ng_ratio', 'Oil/NG Ratio', 1, 0),
    (dxy_m, 'dxy_roc', 'DXY Momentum', -1, 0),
    (rvol_m, 'rvol_pctile', 'Realized Vol', -1, 0),
    (cot_m, 'cot_pctile', 'COT Positioning', -1, 0),
    (indpro_m, 'indpro_yoy', 'Industrial Prod', 1, 2),
    (ts_m, 'ts_slope', 'Term Structure', -1, 0),
    (stor_convex, 'storage_convex', 'Storage Convex', 1, 0),
    (coal_switch, 'coal_switch', 'Coal Switch', 1, 0),
    (weather_demand, 'weather_demand', 'Weather Demand', 1, 0),
    (cot_swap_m, 'swap_pctile', 'Swap Dealer', 1, 0),
    (ttf_hh_m, 'ttf_hh_ratio', 'TTF/HH Ratio', 1, 0),
    # New storage-based factors. Sign convention follows existing factors:
    # - injection_pace already constructed bullish-positive  -> sign +1
    # - pct_full higher = oversupply, bearish                 -> sign -1
    # - days_supply higher = more buffer, bearish             -> sign -1
    # - projected_peak_dev already constructed bullish-positive -> sign +1
    (inj_pace, 'injection_pace', 'Injection Pace', 1, 0),
    (pct_full_factor, 'pct_full', 'Working Gas Full %', -1, 0),
    (days_supply_factor, 'days_supply', 'Storage Days Supply', -1, 0),
    (trajectory, 'projected_peak_dev', 'Storage Trajectory', 1, 0),
    # Power Burn YoY: structural electricity-generation demand for NG.
    # Higher YoY burn = bullish (more demand). EIA monthly data has ~2mo pub lag.
    (power_burn_m, 'power_burn_yoy', 'Power Burn YoY', 1, 2),
    # Power Burn Anomaly: deseasonalized burn vs same-calendar-month 5yr mean.
    # Captures heat-driven demand surges that YoY misses. Bullish when high.
    (power_burn_anom_m, 'power_burn_anomaly', 'Power Burn Anomaly', 1, 2),
]

factor_cols = []
factor_labels = []
factor_signs = []
factor_lags = []

current_month = pd.Timestamp(datetime.now().replace(day=1).strftime('%Y-%m-01'))

for fdf, col, label, sign, pub_lag in factor_defs:
    factor_cols.append(col)
    factor_labels.append(label)
    factor_signs.append(sign)
    factor_lags.append(pub_lag)
    if not fdf.empty:
        fdf_merge = fdf[['date', col]].copy()
        if pub_lag > 0:
            fdf_merge['date'] = fdf_merge['date'] + pd.DateOffset(months=pub_lag)
            print(f"    {label}: shifted +{pub_lag}mo for publication lag")
        # Forward-fill: if current month missing, carry forward latest value
        if fdf_merge['date'].max() < current_month and pub_lag == 0:
            last_row = fdf_merge.dropna(subset=[col]).iloc[-1].copy()
            last_row['date'] = current_month
            fdf_merge = pd.concat([fdf_merge, pd.DataFrame([last_row])], ignore_index=True)
            print(f"    {label}: forward-filled latest value into {current_month:%Y-%m}")
        master = master.merge(fdf_merge, on='date', how='left')
    else:
        master[col] = np.nan

master = master.sort_values('date').reset_index(drop=True)

# Forward returns for IC measurement
for n in [1, 3, 6]:
    master[f'fwd_{n}m'] = master['ng_deseas'].shift(-n) / master['ng_deseas'] - 1

# Log returns for GBM target
master['log_ret_1m'] = np.log(master['ng_price'].shift(-1) / master['ng_price'])

mask = master[factor_cols].notna().sum(axis=1) >= 4
analysis = master[mask].copy()
print(f"  Analysis: {len(analysis)} months ({analysis['date'].min():%Y-%m} to {analysis['date'].max():%Y-%m})")

# ============================================
# Z-score normalize + sign-align → composite
# ============================================
print("\n--- Building Composite ---")

z_cols = []
for col, sign in zip(factor_cols, factor_signs):
    zcol = f'{col}_z'
    z_cols.append(zcol)
    rolling_mean = analysis[col].rolling(60, min_periods=24).mean()
    rolling_std = analysis[col].rolling(60, min_periods=24).std()
    z = (analysis[col] - rolling_mean) / rolling_std
    analysis[zcol] = z * sign

# IC weights (full-sample Spearman with 3m fwd return)
ic_for_weights = {}
for col, label, sign in zip(factor_cols, factor_labels, factor_signs):
    valid = analysis.dropna(subset=[col, 'fwd_3m'])
    if len(valid) >= 30:
        corr, _ = spearmanr(valid[col] * sign, valid['fwd_3m'])
        ic_for_weights[col] = max(abs(corr), 0.02) if not np.isnan(corr) else 0.02
    else:
        ic_for_weights[col] = 0.02

ic_w = pd.Series({zcol: ic_for_weights[col] for zcol, col in zip(z_cols, factor_cols)})
z_df = analysis[z_cols]
analysis['composite'] = z_df.multiply(ic_w).sum(axis=1) / z_df.notna().multiply(ic_w).sum(axis=1)
analysis['n_factors'] = z_df.notna().sum(axis=1)
print(f"  IC weights: {', '.join(f'{lab}={ic_for_weights[c]:.3f}' for c, lab in zip(factor_cols, factor_labels))}")

# ============================================
# Full-sample AR(2) model on log prices (for forward curve prediction)
# ============================================
print("\n--- Full-Sample AR(2) Model ---")

fv_data = analysis.dropna(subset=['composite', 'ng_price']).copy()
fv_data['log_price'] = np.log(fv_data['ng_price'])
fv_data['log_price_lag1'] = fv_data['log_price'].shift(1)
fv_data['log_price_lag2'] = fv_data['log_price'].shift(2)
fv_data['composite_lag1'] = fv_data['composite'].shift(1)
fv_data['composite_lag2'] = fv_data['composite'].shift(2)

# Seasonal Fourier terms
fv_data['month_num'] = fv_data['date'].dt.month
fv_data['sin_m'] = np.sin(2 * np.pi * fv_data['month_num'] / 12)
fv_data['cos_m'] = np.cos(2 * np.pi * fv_data['month_num'] / 12)

# Full-sample OLS with LAGGED features (no lookahead):
# log(P_t) = a + b1*log(P_{t-1}) + b2*log(P_{t-2}) +
#             b3*composite_{t-1} + b4*composite_{t-2} + b5*sin(t) + b6*cos(t)
# All regressors are known at end of month t-1 (except seasonal, which is deterministic).
# This is the same specification as the walk-forward model in ng_fair_value_composite.py.
train_cols_ar = ['log_price_lag1', 'log_price_lag2', 'composite_lag1', 'composite_lag2', 'sin_m', 'cos_m']
valid_train = fv_data.dropna(subset=train_cols_ar + ['log_price']).copy()
print(f"  Training samples: {len(valid_train)}")

X_full = np.column_stack([np.ones(len(valid_train))] + [valid_train[c].values for c in train_cols_ar])
y_full = valid_train['log_price'].values
beta_full = np.linalg.lstsq(X_full, y_full, rcond=None)[0]

residuals = y_full - X_full @ beta_full
res_std = residuals.std()
r2_full = 1 - (residuals.var() / y_full.var())

coef_names = ['const', 'log(P_t-1)', 'log(P_t-2)', 'composite_lag1', 'composite_lag2', 'sin(month)', 'cos(month)']
print(f"  Full-sample R²: {r2_full:.4f}, residual std: {res_std:.4f}")
print("  Coefficients:")
for cn, bv in zip(coef_names, beta_full):
    print(f"    {cn:18s}: {bv:+.4f}")

# Current month nowcast fair value (predict current month from lagged data)
latest = fv_data.iloc[-1]
x_now = np.array([1.0] + [latest[c] for c in train_cols_ar])
log_fv_now = x_now @ beta_full
fv_now = np.exp(log_fv_now)
current_price = latest['ng_price']
current_composite = latest['composite']

# Also compute expanding-window FV for the latest month (like composite model)
# for a more conservative current-month estimate
fv_data['fair_value'] = np.nan
fv_data['fv_std'] = np.nan
fv_data['log_fv'] = np.nan
nowcast_cols = ['log_price_lag1', 'log_price_lag2', 'composite_lag1', 'composite_lag2']
for i in range(36, len(fv_data)):
    train = fv_data.iloc[1:i]
    cur = fv_data.iloc[i]
    valid_mask = train[nowcast_cols + ['log_price']].notna().all(axis=1)
    t = train[valid_mask]
    if len(t) < 20:
        continue
    X = np.column_stack([np.ones(len(t))] + [t[c].values for c in nowcast_cols])
    y = t['log_price'].values
    try:
        beta_ew = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        continue
    resid_ew = y - X @ beta_ew
    x_cur = [cur.get(c, np.nan) for c in nowcast_cols]
    if any(np.isnan(v) for v in x_cur):
        continue
    x_cur = np.array([1.0] + x_cur)
    log_fv = x_cur @ beta_ew
    fv_data.iloc[i, fv_data.columns.get_loc('log_fv')] = log_fv
    fv_data.iloc[i, fv_data.columns.get_loc('fv_std')] = resid_ew.std()

fv_data['fair_value'] = np.exp(fv_data['log_fv'])
fv_data['fv_upper'] = np.exp(fv_data['log_fv'] + fv_data['fv_std'])
fv_data['fv_lower'] = np.exp(fv_data['log_fv'] - fv_data['fv_std'])

print(f"\n  Current NG:     ${current_price:.2f}")
print(f"  Full-sample FV: ${fv_now:.2f} ({(fv_now/current_price - 1)*100:+.1f}%)")

# Machine-readable lines for downstream parsers (UNG visualizer, etc.)
# Bull/bear bands use the expanding-window residual std (1σ).
_latest_with_fv = fv_data.dropna(subset=['fair_value']).iloc[-1] if fv_data['fair_value'].notna().any() else None
if _latest_with_fv is not None:
    _fv_base = float(_latest_with_fv['fair_value'])
    _fv_upper = float(_latest_with_fv['fv_upper'])
    _fv_lower = float(_latest_with_fv['fv_lower'])
else:
    _fv_base = float(fv_now)
    # Fallback: use full-sample residual std
    _resid = (fv_data['log_price'] - np.log(fv_data['fair_value'])).dropna() if fv_data['fair_value'].notna().any() else None
    _sigma = float(_resid.std()) if _resid is not None and len(_resid) > 5 else 0.10
    _fv_upper = float(np.exp(np.log(_fv_base) + _sigma))
    _fv_lower = float(np.exp(np.log(_fv_base) - _sigma))

print(f"PREDICTION_NG_CURRENT: ${current_price:.2f}")
print(f"PREDICTION_NG_FV_BASE: ${_fv_base:.2f}")
print(f"PREDICTION_NG_FV_BULL: ${_fv_upper:.2f}")
print(f"PREDICTION_NG_FV_BEAR: ${_fv_lower:.2f}")

# ============================================
# Supply/Demand Regime Classifier
# (cyclical axis B per CENTRAL_PHILOSOPHY.md)
# ============================================
# Uses storage deviation z-score + days_supply vs 5yr norm + production/consumption
# balance to classify SURPLUS / BALANCED / SHORTAGE. Conservative thresholds:
#   SURPLUS  : storage_dev_z > +1 AND (days_supply > 5yr median OR balance>0)
#   SHORTAGE : storage_dev_z < -1 AND (days_supply < 5yr median OR balance<0)
#   BALANCED : everything else.
_supply_regime = 'BALANCED'
_supply_regime_z = 0.0
try:
    # Use z-score of storage deviation from analysis frame (last available)
    _stor_z_series = analysis['storage_dev_z'].dropna() if 'storage_dev_z' in analysis.columns else None
    _days_supply_series = None
    if not days_supply_weekly.empty:
        _days_supply_series = days_supply_weekly['days_supply']
    _stor_z = float(_stor_z_series.iloc[-1]) if _stor_z_series is not None and len(_stor_z_series) else 0.0
    _supply_regime_z = round(_stor_z, 2)

    _ds_above_med = False
    _ds_below_med = False
    if _days_supply_series is not None and len(_days_supply_series) > 60:
        _ds_med = _days_supply_series.tail(260).median()  # ~5yr median (weekly)
        _ds_now = _days_supply_series.iloc[-1]
        _ds_above_med = _ds_now > _ds_med * 1.05
        _ds_below_med = _ds_now < _ds_med * 0.95

    if _stor_z > 1.0 and _ds_above_med:
        _supply_regime = 'SURPLUS'
    elif _stor_z < -1.0 and _ds_below_med:
        _supply_regime = 'SHORTAGE'
    else:
        _supply_regime = 'BALANCED'
except Exception as _e:
    print(f"Supply regime classify failed: {_e}")
    _supply_regime = 'BALANCED'

print(f"PREDICTION_SUPPLY_REGIME: {_supply_regime}")
print(f"PREDICTION_STORAGE_Z: {_supply_regime_z:+.2f}")

# ============================================
# Volatility Regime Detection
# ============================================
print("\n--- Volatility Regime ---")
ng_daily_copy = ng_daily.copy()
ng_daily_copy['rvol_60d'] = ng_daily_copy['ret'].rolling(42).std() * np.sqrt(252) * 100
ng_daily_copy['rvol_60d_pctile'] = ng_daily_copy['rvol_60d'].rolling(504, min_periods=252).apply(
    lambda x: percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50)

current_vol_pctile = ng_daily_copy['rvol_60d_pctile'].dropna().iloc[-1] if len(ng_daily_copy['rvol_60d_pctile'].dropna()) > 0 else 50
is_high_vol = current_vol_pctile > 75
is_low_vol = current_vol_pctile < 25

# Regime-adjusted residual std from partitioning historical residuals
# rvol_pctile already in fv_data from factor merge
high_vol_idx = fv_data['rvol_pctile'] > 75
low_vol_idx = fv_data['rvol_pctile'] < 25

# Align with residuals length
res_len = len(residuals)
high_mask = high_vol_idx.values[-res_len:] if len(high_vol_idx) >= res_len else np.zeros(res_len, dtype=bool)
low_mask = low_vol_idx.values[-res_len:] if len(low_vol_idx) >= res_len else np.zeros(res_len, dtype=bool)

res_std_high = residuals[high_mask].std() if high_mask.sum() > 5 else res_std * 1.5
res_std_low = residuals[low_mask].std() if low_mask.sum() > 5 else res_std * 0.7

if is_high_vol:
    res_std_regime = res_std_high
    regime_label = f'HIGH VOL ({current_vol_pctile:.0f}th pctile)'
elif is_low_vol:
    res_std_regime = res_std_low
    regime_label = f'LOW VOL ({current_vol_pctile:.0f}th pctile)'
else:
    res_std_regime = res_std
    regime_label = f'NORMAL VOL ({current_vol_pctile:.0f}th pctile)'

print(f"  Regime: {regime_label}")
print(f"  Residual std — full: {res_std*100:.1f}%, regime-adj: {res_std_regime*100:.1f}%")
print(f"  (high-vol σ={res_std_high*100:.1f}%, low-vol σ={res_std_low*100:.1f}%)")

# ============================================
# Gradient Boosting on AR(2) Residuals
# ============================================
latest_row = analysis.dropna(subset=['composite']).iloc[-1]
print("\n--- Gradient Boosting Residual Model ---")
gb_features = z_cols
gb_train_data = valid_train.copy()
gb_train_data['ar_residual'] = residuals

# Only rows with enough z-scored factors
gb_valid = gb_train_data.dropna(subset=gb_features, thresh=len(gb_features) - 4)
use_gb = False
gb_residual_pred = 0.0
fv_now_gb = fv_now
gb_importances = {}
gb_cv_r2 = 0.0

if len(gb_valid) >= 50:
    X_gb = gb_valid[gb_features].fillna(0).values
    y_gb = gb_valid['ar_residual'].values

    gb_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42)
    gb_model.fit(X_gb, y_gb)

    gb_importances = dict(zip(gb_features, gb_model.feature_importances_))
    x_gb_now = np.array([latest_row.get(zcol, 0) for zcol in gb_features]).reshape(1, -1)
    np.nan_to_num(x_gb_now, copy=False)
    gb_residual_pred = gb_model.predict(x_gb_now)[0]
    fv_now_gb = fv_now * np.exp(gb_residual_pred)

    cv_scores = cross_val_score(gb_model, X_gb, y_gb, cv=5, scoring='r2')
    gb_cv_r2 = cv_scores.mean()
    use_gb = gb_cv_r2 > 0.02

    print(f"  GBM CV R²: {gb_cv_r2:.4f} ({'USING' if use_gb else 'SKIPPED — too low'})")
    print(f"  GBM residual correction: {gb_residual_pred*100:+.2f}%")
    if use_gb:
        print(f"  GBM-adjusted FV: ${fv_now_gb:.2f} (vs linear ${fv_now:.2f})")
    sorted_imp = sorted(gb_importances.items(), key=lambda x: x[1], reverse=True)[:5]
    feat_strs = [f"{k.replace('_z', '')}={v:.3f}" for k, v in sorted_imp]
    print(f"  Top features: {', '.join(feat_strs)}")
else:
    print(f"  SKIPPED: only {len(gb_valid)} valid rows (need 50)")

# ============================================
# Multi-Frequency Daily Prediction Model (GBM)
# ============================================
print("\n--- Multi-Frequency Daily Model ---")

# Initialize downstream-compatible variables (used by confidence/conviction section)
gbm_fv = None
gbm_pred = None
gbm_cv_scores = []
gbm_feature_importance = []
quantile_preds = {}
gbm_linear_r2 = 0.0
gbm_nonlinear_r2 = 0.0
gbm_agree = True
gbm_divergence = 0.0

# --- Step 1: Build daily DataFrame from NG prices ---
daily = ng_daily[['date', 'ng_price']].copy()
daily = daily[daily['date'] >= '2015-01-01'].copy()
daily = daily.sort_values('date').reset_index(drop=True)

# DAILY FEATURES: returns at multiple horizons
daily['ret_1d'] = daily['ng_price'].pct_change(1)
daily['ret_5d'] = daily['ng_price'].pct_change(5)
daily['ret_21d'] = daily['ng_price'].pct_change(21)

# Realized volatility at multiple windows
daily['rvol_21d'] = daily['ret_1d'].rolling(21).std() * np.sqrt(252)
daily['rvol_63d'] = daily['ret_1d'].rolling(63).std() * np.sqrt(252)

# Price momentum signals
daily['ma_5'] = daily['ng_price'].rolling(5).mean()
daily['ma_21'] = daily['ng_price'].rolling(21).mean()
daily['ma_63'] = daily['ng_price'].rolling(63).mean()
daily['momentum_21_63'] = daily['ma_21'] / daily['ma_63'] - 1

# --- Step 2: Merge daily cross-asset data ---
# Oil/NG ratio
daily = daily.merge(oil_daily[['date', 'oil_price']], on='date', how='left')
daily['oil_price'] = daily['oil_price'].ffill()
daily['oil_ng_ratio'] = daily['oil_price'] / daily['ng_price']

# TTF/NG ratio
if len(ttf_daily) > 0:
    daily = daily.merge(ttf_daily[['date', 'ttf_price']], on='date', how='left')
    daily['ttf_price'] = daily['ttf_price'].ffill()
    daily['ttf_ng_ratio'] = daily['ttf_price'] / daily['ng_price']
else:
    daily['ttf_price'] = np.nan
    daily['ttf_ng_ratio'] = np.nan

# DXY (kept for data availability but not used as feature)
if len(dxy_daily) > 0:
    daily = daily.merge(dxy_daily[['date', 'dxy']], on='date', how='left')
    daily['dxy'] = daily['dxy'].ffill()
else:
    daily['dxy'] = np.nan

# Term structure slope (daily C1-C4)
if not ts_slope_daily.empty:
    daily = daily.merge(ts_slope_daily[['date', 'ts_slope']], on='date', how='left')
    daily['ts_slope'] = daily['ts_slope'].ffill()
else:
    daily['ts_slope'] = np.nan

# --- Step 3: Forward-fill weekly data to daily ---

# EIA Storage: weekly storage_bcf + storage deviation
if not storage.empty:
    stor_weekly = storage[['date', 'storage_bcf']].copy()
    daily = daily.merge(stor_weekly, on='date', how='left')
    daily['storage_bcf'] = daily['storage_bcf'].ffill()

if not stor_dev_weekly.empty:
    daily = daily.merge(stor_dev_weekly[['date', 'storage_dev']], on='date', how='left')
    daily['storage_dev'] = daily['storage_dev'].ffill()
else:
    daily['storage_dev'] = np.nan

# New storage factors (weekly -> forward-filled to daily)
if not inj_pace_weekly.empty:
    daily = daily.merge(
        inj_pace_weekly[['date', 'injection_pace']], on='date', how='left')
    daily['injection_pace'] = daily['injection_pace'].ffill()
else:
    daily['injection_pace'] = np.nan

if not pct_full_weekly.empty:
    daily = daily.merge(
        pct_full_weekly[['date', 'pct_full']], on='date', how='left')
    daily['pct_full'] = daily['pct_full'].ffill()
else:
    daily['pct_full'] = np.nan

if not days_supply_weekly.empty:
    daily = daily.merge(
        days_supply_weekly[['date', 'days_supply']], on='date', how='left')
    daily['days_supply'] = daily['days_supply'].ffill()
else:
    daily['days_supply'] = np.nan

# Acceleration of storage_dev: 1-week (~5 trading days) change in deviation.
# Positive acceleration = deviation getting larger (could be bullish or bearish
# depending on sign of underlying dev). We expose the raw signed change here so
# downstream models can interact it with storage_dev itself.
daily['storage_dev_accel'] = daily['storage_dev'].diff(5)

# CDD/HDD daily degree day data from CPC (kept for data availability, not used as features)
if len(cpc_hdd_daily_raw) > 0:
    hdd_raw = cpc_hdd_daily_raw[['date', 'dd_value']].rename(columns={'dd_value': 'hdd_daily'})
    daily = daily.merge(hdd_raw, on='date', how='left')
    daily['hdd_daily'] = daily['hdd_daily'].ffill()
else:
    daily['hdd_daily'] = np.nan

if len(cpc_cdd_daily_raw) > 0:
    cdd_raw = cpc_cdd_daily_raw[['date', 'dd_value']].rename(columns={'dd_value': 'cdd_daily'})
    daily = daily.merge(cdd_raw, on='date', how='left')
    daily['cdd_daily'] = daily['cdd_daily'].ffill()
else:
    daily['cdd_daily'] = np.nan

# Baker Hughes rig count (weekly)
if not rig_count.empty:
    rig_w = rig_count[['date', 'value']].rename(columns={'value': 'rig_count'}).copy()
    daily = daily.merge(rig_w, on='date', how='left')
    daily['rig_count'] = daily['rig_count'].ffill()
else:
    daily['rig_count'] = np.nan

# CFTC COT managed money net positioning (weekly)
if not cot.empty:
    cot_w = cot[['date', 'mm_net']].copy()
    if 'swap_net' in cot.columns:
        cot_w = cot[['date', 'mm_net', 'swap_net']].copy()
    daily = daily.merge(cot_w, on='date', how='left')
    daily['mm_net'] = daily['mm_net'].ffill()
    # Percentile of mm_net over trailing 3 years (~156 weekly obs = ~780 daily)
    daily['cot_net_pct'] = daily['mm_net'].rolling(780, min_periods=260).apply(
        lambda x: percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50)
    if 'swap_net' in daily.columns:
        daily['swap_net'] = daily['swap_net'].ffill()
    else:
        daily['swap_net'] = np.nan
else:
    daily['mm_net'] = np.nan
    daily['cot_net_pct'] = np.nan
    daily['swap_net'] = np.nan

# --- Step 4: Forward-fill monthly data to daily ---

# Monthly production, consumption, LNG exports
monthly_for_merge = monthly[['date', 'prod', 'cons', 'lng_exp', 'balance']].copy()
monthly_for_merge = monthly_for_merge.rename(columns={'date': 'month_date'})
daily['_month'] = daily['date'].dt.to_period('M').dt.to_timestamp()
daily = daily.merge(monthly_for_merge, left_on='_month', right_on='month_date', how='left')
daily.drop(columns=['month_date'], inplace=True, errors='ignore')
for col in ['prod', 'cons', 'lng_exp', 'balance']:
    daily[col] = daily[col].ffill()
# YoY changes (12 months = ~252 trading days)
daily['prod_yoy'] = daily['prod'].pct_change(252) if 'prod' in daily.columns else np.nan

# Industrial production from FRED
if not indpro.empty:
    indpro_m = indpro.copy()
    indpro_m['date'] = indpro_m['date'].dt.to_period('M').dt.to_timestamp()
    indpro_m = indpro_m.groupby('date')['value'].last().reset_index()
    indpro_m = indpro_m.rename(columns={'date': 'indpro_month', 'value': 'indpro'})
    daily = daily.merge(indpro_m, left_on='_month', right_on='indpro_month', how='left')
    daily.drop(columns=['indpro_month'], inplace=True, errors='ignore')
    daily['indpro'] = daily['indpro'].ffill()
else:
    daily['indpro'] = np.nan

daily.drop(columns=['_month'], inplace=True, errors='ignore')

# --- Step 5: NEW economically-motivated features ---

# Seasonal features (Fourier)
daily['day_of_year'] = daily['date'].dt.dayofyear
daily['sin_doy'] = np.sin(2 * np.pi * daily['day_of_year'] / 365.25)
daily['cos_doy'] = np.cos(2 * np.pi * daily['day_of_year'] / 365.25)

# Injection season flag (April-October)
daily['injection_season'] = ((daily['date'].dt.month >= 4) & (daily['date'].dt.month <= 10)).astype(float)

# UNG Roll Features: UNG rolls futures ~10-14 trading days before front month expiry
# Front month NG expiry is ~3 business days before calendar month end
# So roll window is approximately day 10-18 of each month
daily['day_of_month'] = daily['date'].dt.day
daily['in_roll_window'] = ((daily['day_of_month'] >= 10) & (daily['day_of_month'] <= 18)).astype(float)
# Roll pressure: during roll window, contango (positive ts_slope) means selling pressure
daily['roll_pressure'] = daily['in_roll_window'] * daily['ts_slope'].fillna(0)

# Storage deviation change (weekly pace of surplus/deficit change)
daily['storage_dev_change'] = daily['storage_dev'].diff(5)

# Volatility regime: is short-term vol above long-term vol?
daily['high_vol_regime'] = (daily['rvol_21d'] > daily['rvol_63d']).astype(float)

# Price distance from 52-week high and low (mean reversion signals)
daily['dist_from_52w_high'] = daily['ng_price'] / daily['ng_price'].rolling(252, min_periods=63).max() - 1
daily['dist_from_52w_low'] = daily['ng_price'] / daily['ng_price'].rolling(252, min_periods=63).min() - 1

# Cross-asset momentum agreement: do oil and NG trend in same direction?
daily['oil_ret_21d'] = daily['oil_price'].pct_change(21)
daily['cross_momentum_agree'] = (np.sign(daily['ret_21d']) == np.sign(daily['oil_ret_21d'])).astype(float)

# --- Step 6: Multi-horizon targets ---
daily['fwd_ret_5d'] = daily['ng_price'].shift(-5) / daily['ng_price'] - 1
daily['fwd_ret_21d'] = daily['ng_price'].shift(-21) / daily['ng_price'] - 1
daily['fwd_ret_63d'] = daily['ng_price'].shift(-63) / daily['ng_price'] - 1

# --- Step 7: Define curated feature set (~20 features with economic rationale) ---
# Every feature has a clear reason to predict NG returns:
#   Momentum: ret_5d, ret_21d, momentum_21_63
#   Volatility: rvol_21d, high_vol_regime
#   Cross-asset: oil_ng_ratio, ttf_ng_ratio, cross_momentum_agree
#   Term structure: ts_slope
#   Fundamentals: storage_dev, storage_dev_change, rig_count, prod_yoy
#   Positioning: cot_net_pct
#   Seasonal: sin_doy, cos_doy
#   UNG flow: in_roll_window, roll_pressure
#   Mean reversion: dist_from_52w_high, dist_from_52w_low

curated_features = [
    # Momentum (3)
    'ret_5d', 'ret_21d', 'momentum_21_63',
    # Volatility (2)
    'rvol_21d', 'high_vol_regime',
    # Cross-asset (3)
    'oil_ng_ratio', 'ttf_ng_ratio', 'cross_momentum_agree',
    # Term structure (1)
    'ts_slope',
    # Fundamentals (4)
    'storage_dev', 'storage_dev_change', 'rig_count', 'prod_yoy',
    # Positioning (1)
    'cot_net_pct',
    # Seasonal (2)
    'sin_doy', 'cos_doy',
    # UNG roll (2)
    'in_roll_window', 'roll_pressure',
    # Mean reversion (2)
    'dist_from_52w_high', 'dist_from_52w_low',
]

# Remove features that are entirely NaN or have too few valid values
feature_cols_daily = [c for c in curated_features
                      if c in daily.columns and daily[c].notna().sum() > 100]

# Categorize features for importance reporting
_feature_categories = {
    'ret_5d': 'Momentum', 'ret_21d': 'Momentum', 'momentum_21_63': 'Momentum',
    'rvol_21d': 'Volatility', 'high_vol_regime': 'Volatility',
    'oil_ng_ratio': 'Cross-asset', 'ttf_ng_ratio': 'Cross-asset',
    'cross_momentum_agree': 'Cross-asset',
    'ts_slope': 'Term structure',
    'storage_dev': 'Fundamentals', 'storage_dev_change': 'Fundamentals',
    'rig_count': 'Fundamentals', 'prod_yoy': 'Fundamentals',
    'cot_net_pct': 'Positioning',
    'sin_doy': 'Seasonal', 'cos_doy': 'Seasonal',
    'in_roll_window': 'UNG roll', 'roll_pressure': 'UNG roll',
    'dist_from_52w_high': 'Mean reversion', 'dist_from_52w_low': 'Mean reversion',
}

def _categorize_feature(f):
    return _feature_categories.get(f, 'Other')

# Count features by category for reporting
_cat_counts = {}
for f in feature_cols_daily:
    cat = _categorize_feature(f)
    _cat_counts[cat] = _cat_counts.get(cat, 0) + 1

print(f"Features: {len(feature_cols_daily)} total "
      f"({', '.join(f'{cat}: {n}' for cat, n in sorted(_cat_counts.items()))})")

# --- Step 8: Train multi-horizon GBM models ---
# Key anti-overfitting settings vs old model:
#   n_estimators: 100 (was 300) -- fewer trees
#   max_depth: 2 (was 4) -- much shallower trees
#   learning_rate: 0.05 (was 0.03) -- larger steps, fewer trees needed
#   subsample: 0.7 (was 0.8) -- more randomness per tree
#   min_samples_leaf: 30 (was 10) -- much larger minimum leaf size
#   max_features: 0.5 (NEW) -- only use half the features per split
#   Features: ~20 curated (was ~33) -- fewer, more meaningful features

daily_model_results = {}
gbm_use_linear_fallback = False

for horizon_name, target_col in [('5d', 'fwd_ret_5d'), ('21d', 'fwd_ret_21d'), ('63d', 'fwd_ret_63d')]:
    ml_df = daily[feature_cols_daily + [target_col]].dropna()

    if len(ml_df) < 200:
        print(f"  {horizon_name}: SKIPPED -- only {len(ml_df)} samples (need 200)")
        continue

    # Pre-filter: only include features with abs correlation > 0.02 with target
    # This removes pure noise features before training
    correlations = ml_df[feature_cols_daily].corrwith(ml_df[target_col]).abs()
    useful_features = correlations[correlations > 0.02].index.tolist()
    if len(useful_features) < 5:
        useful_features = correlations.nlargest(10).index.tolist()

    X = ml_df[useful_features].values
    y = ml_df[target_col].values

    # Walk-forward CV
    tscv = TimeSeriesSplit(n_splits=5)

    gbm_params = dict(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.7, min_samples_leaf=30, max_features=0.5,
        random_state=42
    )

    # Mean model
    gbm = GradientBoostingRegressor(**gbm_params)

    # Walk-forward CV scores
    cv_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        gbm_cv_fold = GradientBoostingRegressor(**gbm_params)
        gbm_cv_fold.fit(X_tr, y_tr)
        score = gbm_cv_fold.score(X_te, y_te)
        cv_scores.append(score)

    # Linear baseline: simple OLS with momentum_21_63 as single predictor
    momentum_col = 'momentum_21_63' if 'momentum_21_63' in useful_features else useful_features[0]
    momentum_idx = useful_features.index(momentum_col)
    x_lin = X[:, momentum_idx]
    valid_lin = ~np.isnan(x_lin)
    if valid_lin.sum() > 10:
        lin_coeffs = polyfit(x_lin[valid_lin], y[valid_lin], 1)
        lin_pred = polyval(x_lin[valid_lin], lin_coeffs)
        ss_res = np.sum((y[valid_lin] - lin_pred) ** 2)
        ss_tot = np.sum((y[valid_lin] - y[valid_lin].mean()) ** 2)
        linear_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        linear_r2 = 0.0

    cv_r2_mean = np.mean(cv_scores)

    # If R² is negative, the GBM is worse than predicting the mean -- flag for linear fallback
    if cv_r2_mean < 0:
        print(f"  WARNING: {horizon_name} GBM CV R²={cv_r2_mean:.3f} is negative -- "
              f"model overfitting, will use linear baseline for predictions")
        if horizon_name == '21d':
            gbm_use_linear_fallback = True

    # Fit final model on all data
    gbm.fit(X, y)

    # Feature importance
    importances = gbm.feature_importances_
    feat_importance = sorted(zip(useful_features, importances), key=lambda x: -x[1])

    # Quantile models for prediction intervals (same conservative params)
    quantile_models_hz = {}
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        qgbm = GradientBoostingRegressor(
            loss='quantile', alpha=q, **gbm_params
        )
        qgbm.fit(X, y)
        quantile_models_hz[q] = qgbm

    daily_model_results[horizon_name] = {
        'model': gbm,
        'cv_scores': cv_scores,
        'cv_r2': cv_r2_mean,
        'linear_r2': linear_r2,
        'n_samples': len(ml_df),
        'feat_importance': feat_importance,
        'quantile_models': quantile_models_hz,
        'feature_cols': useful_features,
        'linear_coeffs': lin_coeffs if valid_lin.sum() > 10 else None,
        'linear_feature': momentum_col,
    }

    dropped_features = [f for f in feature_cols_daily if f not in useful_features]
    print(f"  {horizon_name} forecast: {len(ml_df)} samples, {len(useful_features)} features "
          f"(dropped {len(dropped_features)} low-corr), "
          f"CV R²={cv_r2_mean:.3f} (linear: {linear_r2:.3f}), "
          f"folds: [{', '.join(f'{s:.3f}' for s in cv_scores)}]")

# --- Step 9: Current predictions ---
# Build current feature vector from the last row of daily
latest_daily = daily.dropna(subset=feature_cols_daily[:5]).iloc[-1]

# ============================================
# GBM Feature Importance Summary (keep from training above)
# ============================================
if '21d' in daily_model_results:
    primary_result = daily_model_results['21d']
elif '5d' in daily_model_results:
    primary_result = daily_model_results['5d']
else:
    primary_result = None

# Set downstream-compatible variables for confidence/conviction section
if '21d' in daily_model_results:
    _primary = daily_model_results['21d']
    _used_features = _primary['feature_cols']
    X_now_primary = np.array([latest_daily.get(c, 0) for c in _used_features]).reshape(1, -1)
    np.nan_to_num(X_now_primary, copy=False)
    if _primary['cv_r2'] < 0 and _primary['linear_coeffs'] is not None:
        lin_feat_val = latest_daily.get(_primary['linear_feature'], 0)
        if np.isnan(lin_feat_val):
            lin_feat_val = 0.0
        gbm_pred = polyval(lin_feat_val, _primary['linear_coeffs'])
    else:
        gbm_pred = _primary['model'].predict(X_now_primary)[0]
    gbm_fv = current_price * (1 + gbm_pred)
    gbm_nonlinear_r2 = _primary['cv_r2']
    gbm_linear_r2 = _primary['linear_r2']
    gbm_cv_scores = _primary['cv_scores']
    gbm_feature_importance = _primary['feat_importance']
    for q, qmodel in _primary['quantile_models'].items():
        quantile_preds[q] = current_price * (1 + qmodel.predict(X_now_primary)[0])
elif '5d' in daily_model_results:
    _primary = daily_model_results['5d']
    _used_features = _primary['feature_cols']
    X_now_primary = np.array([latest_daily.get(c, 0) for c in _used_features]).reshape(1, -1)
    np.nan_to_num(X_now_primary, copy=False)
    if _primary['cv_r2'] < 0 and _primary['linear_coeffs'] is not None:
        lin_feat_val = latest_daily.get(_primary['linear_feature'], 0)
        if np.isnan(lin_feat_val):
            lin_feat_val = 0.0
        gbm_pred = polyval(lin_feat_val, _primary['linear_coeffs'])
    else:
        gbm_pred = _primary['model'].predict(X_now_primary)[0]
    gbm_fv = current_price * (1 + gbm_pred)
    gbm_nonlinear_r2 = _primary['cv_r2']
    gbm_linear_r2 = _primary['linear_r2']
    gbm_cv_scores = _primary['cv_scores']
    gbm_feature_importance = _primary['feat_importance']
    for q, qmodel in _primary['quantile_models'].items():
        quantile_preds[q] = current_price * (1 + qmodel.predict(X_now_primary)[0])

if gbm_fv is not None:
    linear_fv = fv_now
    nonlinear_fv = gbm_fv
    gbm_divergence = (nonlinear_fv - linear_fv) / linear_fv * 100
    linear_dir = np.sign(linear_fv - current_price)
    nonlinear_dir = np.sign(nonlinear_fv - current_price)
    gbm_agree = (linear_dir == nonlinear_dir) or linear_dir == 0 or nonlinear_dir == 0

# ============================================
# Contrarian Divergence Model
# ============================================
print("\n--- Contrarian Divergence Model ---")

# Use the 'analysis' DataFrame as 'merged' — it has composite z-scores and prices
merged = analysis.copy()
merged = merged.sort_values('date').reset_index(drop=True)

# Compute the composite z-score (already exists as 'composite')
composite_z = current_composite

# --- 1. Forward returns at multiple horizons ---
# log_ret_1m already exists; create log_ret_2m and log_ret_3m
merged['log_ret_1m'] = np.log(merged['ng_price'].shift(-1) / merged['ng_price'])
merged['log_ret_2m'] = np.log(merged['ng_price'].shift(-2) / merged['ng_price'])
merged['log_ret_3m'] = np.log(merged['ng_price'].shift(-3) / merged['ng_price'])

# Current z percentile in history
all_historical_z = merged['composite'].dropna().values
z_percentile = percentileofscore(all_historical_z, composite_z) if len(all_historical_z) > 0 else 50.0
divergence_from_fv = (current_price / fv_now - 1) * 100  # negative = cheap

print(f"Current composite z: {composite_z:+.2f} ({z_percentile:.0f}th percentile of history)")
if composite_z > 1.0:
    signal_label = "STRONG BUY (extreme cheap)"
elif composite_z > 0.5:
    signal_label = "BUY (cheap)"
elif composite_z > 0.25:
    signal_label = "LEAN LONG (slightly cheap)"
elif composite_z > -0.25:
    signal_label = "NEUTRAL"
elif composite_z > -0.5:
    signal_label = "LEAN SHORT (slightly rich)"
elif composite_z > -1.0:
    signal_label = "SELL (rich)"
else:
    signal_label = "STRONG SELL (extreme rich)"
print(f"Current signal: {signal_label}")
print(f"Model says NG is {divergence_from_fv:+.1f}% vs fair value (${current_price:.2f} vs FV ${fv_now:.2f})")

# --- 2. Historical Divergence Analysis: Buckets and Hit Rates ---
print("\nHistorical Divergence Analysis:")

buckets = [
    ('Extreme Cheap (z > 1.0)',     lambda z: z > 1.0),
    ('Very Cheap (0.5 < z <= 1.0)', lambda z: 0.5 < z <= 1.0),
    ('Cheap (0.25 < z <= 0.5)',     lambda z: 0.25 < z <= 0.5),
    ('Neutral (-0.25 to 0.25)',     lambda z: -0.25 <= z <= 0.25),
    ('Rich (-0.5 to -0.25)',        lambda z: -0.5 <= z < -0.25),
    ('Very Rich (-1.0 to -0.5)',    lambda z: -1.0 <= z < -0.5),
    ('Extreme Rich (z < -1.0)',     lambda z: z < -1.0),
]

# Determine which bucket the current z falls into
current_bucket_label = 'Neutral (-0.25 to 0.25)'
for bname, bfunc in buckets:
    if bfunc(composite_z):
        current_bucket_label = bname
        break

header = (f"  {'Bucket':<30s} | {'N':>4s} | {'1m Med':>7s} | {'1m Up%':>6s} | "
          f"{'2m Med':>7s} | {'2m Up%':>6s} | {'3m Med':>7s} | {'Sharpe':>6s}")
print(header)
print(f"  {'-'*30}-+-{'-'*4}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")

bucket_stats = []
for bname, bfunc in buckets:
    mask = merged['composite'].apply(lambda z: bfunc(z) if not np.isnan(z) else False)
    bdata = merged[mask].copy()
    n_obs = len(bdata)

    if n_obs < 2:
        marker = '  <-- YOU ARE HERE' if bname == current_bucket_label else ''
        print(f"  {bname:<30s} | {n_obs:>4d} |     n/a |    n/a |     n/a |    n/a |     n/a |    n/a{marker}")
        bucket_stats.append({'name': bname, 'n': n_obs, 'med_1m': np.nan, 'up_1m': np.nan,
                             'med_2m': np.nan, 'up_2m': np.nan, 'med_3m': np.nan, 'sharpe': np.nan})
        continue

    r1m = bdata['log_ret_1m'].dropna()
    r2m = bdata['log_ret_2m'].dropna()
    r3m = bdata['log_ret_3m'].dropna()

    med_1m = r1m.median() * 100 if len(r1m) > 0 else np.nan
    up_1m = (r1m > 0).mean() * 100 if len(r1m) > 0 else np.nan
    med_2m = r2m.median() * 100 if len(r2m) > 0 else np.nan
    up_2m = (r2m > 0).mean() * 100 if len(r2m) > 0 else np.nan
    med_3m = r3m.median() * 100 if len(r3m) > 0 else np.nan
    sharpe_1m = (r1m.mean() / r1m.std() * np.sqrt(12)) if len(r1m) > 2 and r1m.std() > 0 else np.nan

    bucket_stats.append({'name': bname, 'n': n_obs, 'med_1m': med_1m, 'up_1m': up_1m,
                         'med_2m': med_2m, 'up_2m': up_2m, 'med_3m': med_3m, 'sharpe': sharpe_1m})

    marker = '  <-- YOU ARE HERE' if bname == current_bucket_label else ''

    def _fmt(v, fmt_str):
        return f'{v:{fmt_str}}' if not np.isnan(v) else '    n/a'

    print(f"  {bname:<30s} | {n_obs:>4d} | {_fmt(med_1m, '+6.1f')}% | {_fmt(up_1m, '5.0f')}% | "
          f"{_fmt(med_2m, '+6.1f')}% | {_fmt(up_2m, '5.0f')}% | {_fmt(med_3m, '+6.1f')}% | "
          f"{_fmt(sharpe_1m, '5.2f')}{marker}")

# --- 3. Analog Analysis: Find historical months with similar composite z ---
print(f"\nAnalog Analysis (z ~ {composite_z:+.2f}, +/-0.15):")

analog_radius = 0.15
analog_mask = (merged['composite'] >= composite_z - analog_radius) & \
              (merged['composite'] <= composite_z + analog_radius) & \
              merged['composite'].notna()
analogs = merged[analog_mask].copy()
n_analogs = len(analogs)
print(f"  N analogs: {n_analogs}")

if n_analogs >= 3:
    for horizon_name, ret_col in [('1-month', 'log_ret_1m'), ('2-month', 'log_ret_2m'), ('3-month', 'log_ret_3m')]:
        a_ret = analogs[ret_col].dropna()
        if len(a_ret) >= 3:
            a_med = a_ret.median() * 100
            a_up = (a_ret > 0).mean() * 100
            a_min = a_ret.min() * 100
            a_max = a_ret.max() * 100
            print(f"  {horizon_name}: median {a_med:+.1f}%, up {a_up:.0f}%, "
                  f"range [{a_min:+.1f}% to {a_max:+.1f}%]")
        else:
            print(f"  {horizon_name}: insufficient data (n={len(a_ret)})")
else:
    # Try wider radius
    analog_radius_wide = 0.25
    analog_mask_wide = (merged['composite'] >= composite_z - analog_radius_wide) & \
                       (merged['composite'] <= composite_z + analog_radius_wide) & \
                       merged['composite'].notna()
    analogs_wide = merged[analog_mask_wide].copy()
    n_analogs_wide = len(analogs_wide)
    print(f"  Too few analogs at +/-0.15, widening to +/-0.25: {n_analogs_wide} analogs")
    analogs = analogs_wide
    n_analogs = n_analogs_wide
    for horizon_name, ret_col in [('1-month', 'log_ret_1m'), ('2-month', 'log_ret_2m'), ('3-month', 'log_ret_3m')]:
        a_ret = analogs[ret_col].dropna()
        if len(a_ret) >= 3:
            a_med = a_ret.median() * 100
            a_up = (a_ret > 0).mean() * 100
            a_min = a_ret.min() * 100
            a_max = a_ret.max() * 100
            print(f"  {horizon_name}: median {a_med:+.1f}%, up {a_up:.0f}%, "
                  f"range [{a_min:+.1f}% to {a_max:+.1f}%]")
        else:
            print(f"  {horizon_name}: insufficient data (n={len(a_ret)})")

# --- 4. Contrarian Strategy Backtest ---
print("\nContrarian Strategy Backtest:")
print(f"  {'Threshold':>10s} | {'Trades':>6s} | {'Win Rate':>8s} | {'Avg Ret':>8s} | "
      f"{'Sharpe':>7s} | {'Max Loss':>8s} | {'Max Gain':>8s} | {'Profit Factor':>13s}")
print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*13}")

backtest_results = []
for threshold in [0.25, 0.50, 0.75, 1.00]:
    trades = []
    for i in range(len(merged) - 1):
        row = merged.iloc[i]
        z_val = row.get('composite', np.nan)
        if np.isnan(z_val):
            continue
        fwd = merged.iloc[i]['log_ret_1m']
        if np.isnan(fwd):
            continue

        if z_val > threshold:
            # Buy signal: expect price to go up (cheap)
            trades.append(fwd)
        elif z_val < -threshold:
            # Short signal: expect price to go down (rich)
            trades.append(-fwd)

    if len(trades) >= 3:
        trades_arr = np.array(trades)
        win_rate = (trades_arr > 0).mean() * 100
        avg_ret = trades_arr.mean() * 100
        sharpe = trades_arr.mean() / trades_arr.std() * np.sqrt(12) if trades_arr.std() > 0 else 0
        max_loss = trades_arr.min() * 100
        max_gain = trades_arr.max() * 100
        pos_sum = trades_arr[trades_arr > 0].sum()
        neg_sum = abs(trades_arr[trades_arr < 0].sum())
        profit_factor = pos_sum / neg_sum if neg_sum > 0 else np.inf

        backtest_results.append({
            'threshold': threshold, 'n_trades': len(trades_arr),
            'win_rate': win_rate, 'avg_return': avg_ret, 'sharpe': sharpe,
            'max_loss': max_loss, 'max_gain': max_gain, 'profit_factor': profit_factor,
        })

        pf_str = f'{profit_factor:.2f}' if profit_factor != np.inf else 'inf'
        print(f"  z > {threshold:.2f}    | {len(trades_arr):>6d} | {win_rate:>6.0f}%  | {avg_ret:>+7.1f}% | "
              f"{sharpe:>6.2f}  | {max_loss:>+7.1f}% | {max_gain:>+7.1f}% | {pf_str:>13s}")
    else:
        print(f"  z > {threshold:.2f}    | {len(trades):>6d} |      n/a |      n/a |     n/a |      n/a |      n/a |           n/a")

# --- 5. Current Signal Strength Summary ---
print("\n--- Current Signal Strength ---")

# Expected return from analogs at current z level
expected_1m_ret = np.nan
analog_1m = analogs['log_ret_1m'].dropna() if n_analogs >= 3 else pd.Series(dtype=float)
if len(analog_1m) >= 3:
    expected_1m_ret = analog_1m.median() * 100

# Find the best matching backtest threshold
best_bt = None
for bt in backtest_results:
    if abs(composite_z) >= bt['threshold']:
        best_bt = bt

# Kelly optimal from analog returns
if len(analog_1m) >= 5:
    p_win = (analog_1m > 0).mean()
    avg_win = analog_1m[analog_1m > 0].mean() if (analog_1m > 0).any() else 0
    avg_loss = abs(analog_1m[analog_1m <= 0].mean()) if (analog_1m <= 0).any() else 1
    b_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    kelly_pct = max(0, (p_win * b_ratio - (1 - p_win)) / b_ratio) * 100 if b_ratio > 0 else 0
else:
    kelly_pct = 0.0
    p_win = np.nan

# Historical hit rate at this z-level
hit_rate_str = f"{(analog_1m > 0).mean()*100:.0f}%" if len(analog_1m) >= 3 else "n/a"

print(f"  Current z:            {composite_z:+.2f} ({z_percentile:.0f}th percentile)")
print(f"  Signal:               {signal_label}")
print(f"  FV divergence:        {divergence_from_fv:+.1f}%")
if not np.isnan(expected_1m_ret):
    print(f"  Expected 1m return:   {expected_1m_ret:+.1f}% (from {n_analogs} analogs)")
print(f"  Historical hit rate:  {hit_rate_str}")
if best_bt is not None:
    print(f"  Best backtest match:  z>{best_bt['threshold']:.2f} threshold "
          f"(win {best_bt['win_rate']:.0f}%, sharpe {best_bt['sharpe']:.2f}, "
          f"PF {best_bt['profit_factor']:.2f})")
print(f"  Kelly optimal:        {kelly_pct:.0f}% of portfolio")

# Verdict
if best_bt is not None and best_bt['win_rate'] > 55 and best_bt['sharpe'] > 0.3:
    verdict = "ACTIONABLE"
    verdict_detail = f"When model was this {'cheap' if composite_z > 0 else 'rich'} historically, it was right {best_bt['win_rate']:.0f}% of the time"
elif best_bt is not None and best_bt['win_rate'] > 50:
    verdict = "MARGINAL"
    verdict_detail = f"Slight edge ({best_bt['win_rate']:.0f}% win rate) but not statistically robust"
else:
    verdict = "NO EDGE"
    if abs(composite_z) < 0.25:
        verdict_detail = "Composite z near zero -- no divergence to exploit"
    else:
        verdict_detail = "Insufficient historical edge at this z-level"

print(f"\n  VERDICT: {verdict}")
print(f"  {verdict_detail}")

# --- 6. Daily Divergence Tracking ---
print("\n--- Daily Divergence Index ---")

# Build a daily cheapness score from available daily features
daily_div_components = {}

# Distance from 52-week low (near low = cheap)
if 'dist_from_52w_low' in daily.columns and daily['dist_from_52w_low'].notna().sum() > 100:
    daily_div_components['dist_52w_low'] = daily['dist_from_52w_low'].rank(pct=True) * 0.3

# Term structure slope (steep contango = cheap front month)
if 'ts_slope' in daily.columns and daily['ts_slope'].notna().sum() > 100:
    daily_div_components['ts_slope_inv'] = (-daily['ts_slope']).rank(pct=True) * 0.2

# Oil/NG ratio (high ratio = NG cheap vs oil)
if 'oil_ng_ratio' in daily.columns and daily['oil_ng_ratio'].notna().sum() > 100:
    daily_div_components['oil_ng_cheap'] = daily['oil_ng_ratio'].rank(pct=True) * 0.25

# TTF/NG ratio (high global premium = NG cheap)
if 'ttf_ng_ratio' in daily.columns and daily['ttf_ng_ratio'].notna().sum() > 100:
    daily_div_components['ttf_ng_cheap'] = daily['ttf_ng_ratio'].rank(pct=True) * 0.25

if daily_div_components:
    div_df = pd.DataFrame(daily_div_components, index=daily.index)
    total_weight = sum(float(c.iloc[-1]) if not np.isnan(c.iloc[-1]) else 0 for c in [])  # just for summing
    daily['daily_divergence'] = div_df.sum(axis=1) / div_df.notna().sum(axis=1) * len(daily_div_components)

    # Normalize so it ranges 0-1 (percentile rank of the sum)
    daily['daily_divergence_pct'] = daily['daily_divergence'].rank(pct=True)

    # Current daily divergence
    latest_daily_div = daily['daily_divergence_pct'].dropna().iloc[-1] if daily['daily_divergence_pct'].notna().any() else np.nan
    if not np.isnan(latest_daily_div):
        if latest_daily_div > 0.7:
            daily_signal = "CHEAP (agrees with monthly)"
        elif latest_daily_div < 0.3:
            daily_signal = "NOT CHEAP (caution)"
        else:
            daily_signal = "NEUTRAL"

        print(f"  Daily divergence percentile: {latest_daily_div*100:.0f}%")
        print(f"  Daily signal: {daily_signal}")
        print(f"  Components ({len(daily_div_components)}):")
        for comp_name in daily_div_components:
            comp_val = daily_div_components[comp_name].dropna().iloc[-1] if daily_div_components[comp_name].notna().any() else np.nan
            if not np.isnan(comp_val):
                print(f"    {comp_name:<20s}: {comp_val:.3f}")

        # Confidence adjustment: if daily and monthly agree, higher confidence
        if composite_z > 0.25 and latest_daily_div > 0.6:
            print("  --> Daily CONFIRMS monthly cheap signal (higher confidence)")
        elif composite_z > 0.25 and latest_daily_div < 0.4:
            print("  --> Daily CONTRADICTS monthly cheap signal (lower confidence)")
        elif composite_z < -0.25 and latest_daily_div < 0.4:
            print("  --> Daily CONFIRMS monthly rich signal (higher confidence)")
        elif composite_z < -0.25 and latest_daily_div > 0.6:
            print("  --> Daily CONTRADICTS monthly rich signal (lower confidence)")
        else:
            print("  --> No strong agreement/disagreement between daily and monthly")
    else:
        print("  Daily divergence: insufficient data")
else:
    print("  Daily divergence: no components available")

# GBM feature importance summary (brief, from training above)
if primary_result is not None:
    print("\n--- GBM Feature Importance (for reference) ---")
    cat_importance = {}
    for feat, imp in primary_result['feat_importance']:
        cat = _categorize_feature(feat)
        cat_importance[cat] = cat_importance.get(cat, 0) + imp
    total_imp = sum(cat_importance.values())
    if total_imp > 0:
        for cat in sorted(cat_importance.keys(), key=lambda c: -cat_importance[c]):
            pct = cat_importance[cat] / total_imp * 100
            print(f"  {cat:<24s} {pct:5.1f}%")
    print(f"  Top 5: {', '.join(f'{f}={v:.3f}' for f, v in primary_result['feat_importance'][:5])}")

# ============================================
# Forward Curve Fair Value: Seasonal Relative Value
# ============================================
# Instead of iterating the AR(2) forward (which compounds error and gives huge
# variance beyond 2-3 months), we compute fair value for each contract using:
#   1. Historical seasonal shape of the NG futures curve (median ratio of each
#      calendar month's price to the front month, computed over the full history)
#   2. Composite z-score adjustment: bullish composite → all contracts should
#      trade above their seasonal norm, and vice versa.
# This gives ONE-STEP uncertainty (res_std) for each contract, not compounded.
print("\n--- Forward Curve Relative Value ---")

current_month = latest['date'].month

# 1-month ahead prediction from the AR(2) model (single step, tightest)
lp1 = np.log(current_price)
lp2 = fv_data['log_price'].iloc[-2] if len(fv_data) >= 2 else lp1
c1 = current_composite
c2 = fv_data['composite'].iloc[-2] if len(fv_data) >= 2 else c1
next_month = (current_month % 12) + 1
x_1m = np.array([1.0, lp1, lp2, c1, c2,
                  np.sin(2 * np.pi * next_month / 12),
                  np.cos(2 * np.pi * next_month / 12)])
fv_1m = np.exp(x_1m @ beta_full)
print(f"  1-month ahead FV: ${fv_1m:.2f} (±1σ: ${fv_1m * np.exp(-res_std):.2f}-${fv_1m * np.exp(res_std):.2f})")

# Build seasonal curve shape from NG history
# For each calendar month, compute the median ratio to the current month's price
# This tells us: "if NG is $X in February, historically July is 1.15x that"
ng_monthly_hist = ng_monthly.dropna(subset=['ng_price']).copy()
ng_monthly_hist['cal_month'] = ng_monthly_hist['date'].dt.month

# Compute median price per calendar month over trailing 10 years
cutoff_seas = ng_monthly_hist['date'].max() - pd.DateOffset(years=10)
recent_hist = ng_monthly_hist[ng_monthly_hist['date'] >= cutoff_seas]
seasonal_median = recent_hist.groupby('cal_month')['ng_price'].median()
current_month_median = seasonal_median.get(current_month, current_price)

# Seasonal ratio: how each month compares to the current calendar month historically
seasonal_ratio = seasonal_median / current_month_median

# Composite adjustment: the model's composite_lag1 coefficient tells us how much
# a +1 z-score composite shifts log prices. Apply this uniformly to all contracts.
composite_shift = beta_full[3] * current_composite  # composite_lag1 coeff * current z
print(f"  Composite price adjustment: {composite_shift:+.4f} log ({np.exp(composite_shift)*100 - 100:+.1f}%)")

# Build FV for each futures contract
if not futures_curve.empty:
    comparison = futures_curve.copy()
    fv_prices = []
    fv_neutral = []  # seasonal only, no composite adjustment
    for _, row in comparison.iterrows():
        target_cal_month = row['month'].month
        seas_r = seasonal_ratio.get(target_cal_month, 1.0)
        # Neutral FV: pure seasonal shape from current price
        fv_neutral.append(current_price * seas_r)
        # Full FV: seasonal + composite signal adjustment
        fv = current_price * seas_r * np.exp(composite_shift)
        fv_prices.append(fv)
    comparison['fv_price'] = fv_prices
    comparison['fv_neutral'] = fv_neutral

    comparison['spread'] = comparison['fv_price'] - comparison['price']
    comparison['spread_pct'] = (comparison['fv_price'] / comparison['price'] - 1) * 100
    # Edge must exceed 1σ model uncertainty to be meaningful
    comparison['edge_vs_sigma'] = comparison['spread_pct'].abs() / (res_std * 100)
    comparison['signal'] = comparison.apply(
        lambda r: 'CHEAP' if r['edge_vs_sigma'] > 0.5 and r['spread'] > 0
        else 'RICH' if r['edge_vs_sigma'] > 0.5 and r['spread'] < 0
        else 'FAIR', axis=1)

    # Trade analysis: stop at 1σ adverse, target at FV
    comparison['stop'] = np.where(
        comparison['spread'] > 0,
        comparison['price'] * np.exp(-res_std),
        comparison['price'] * np.exp(+res_std))
    comparison['risk'] = np.abs(comparison['price'] - comparison['stop'])
    comparison['reward'] = np.abs(comparison['fv_price'] - comparison['price'])
    comparison['rr_ratio'] = comparison['reward'] / comparison['risk'].clip(lower=0.01)
    comparison['direction'] = comparison['spread'].apply(lambda x: 'LONG' if x > 0 else 'SHORT')
    # Edge score: penalize edges that don't exceed model uncertainty
    comparison['edge_score'] = comparison['edge_vs_sigma'] * comparison['spread_pct'].abs()

    print(f"  Seasonal FV curve computed for {len(comparison)} contracts")
    print(f"  Model 1σ uncertainty: ±{res_std*100:.1f}% (edges must exceed this to be actionable)")
else:
    comparison = pd.DataFrame()

# ============================================
# UNG Contango Calculator
# ============================================
print("\n--- UNG Contango Impact ---")

if not futures_curve.empty and len(futures_curve) >= 2:
    ung = futures_curve.sort_values('months_ahead').copy()
    ung['roll_cost'] = ung['price'].pct_change()  # C2/C1 - 1 for each month
    ung['cum_drag'] = (1 + ung['roll_cost'].fillna(0)).cumprod() - 1

    # Model-implied return from current price
    if not comparison.empty:
        ung_analysis = comparison.merge(ung[['months_ahead', 'roll_cost', 'cum_drag']],
                                        on='months_ahead', how='left')
        ung_analysis['ng_return'] = (ung_analysis['fv_price'] / current_price - 1) * 100
        ung_analysis['ung_equiv'] = ung_analysis['ng_return'] - ung_analysis['cum_drag'] * 100
    else:
        ung_analysis = pd.DataFrame()
else:
    ung = pd.DataFrame()
    ung_analysis = pd.DataFrame()

# ============================================
# Console Output
# ============================================
print("\n" + "=" * 65)
print("=== NG DAILY FAIR VALUE FORECAST ===")
print("=" * 65)
print(f"Date: {datetime.now():%Y-%m-%d}")
print(f"Current NG (front month): ${current_price:.2f}")

# Count bullish/bearish factors
latest_row = analysis.dropna(subset=['composite']).iloc[-1]
n_bull = sum(1 for zcol in z_cols if not np.isnan(latest_row.get(zcol, np.nan)) and latest_row[zcol] > 0.3)
n_bear = sum(1 for zcol in z_cols if not np.isnan(latest_row.get(zcol, np.nan)) and latest_row[zcol] < -0.3)
n_neutral = int(latest_row['n_factors']) - n_bull - n_bear
bias = 'BULLISH' if current_composite > 0.5 else 'BEARISH' if current_composite < -0.5 else 'NEUTRAL'
print(f"Composite z-score: {current_composite:+.2f} ({int(latest_row['n_factors'])} factors, {n_bull} BULL/{n_bear} BEAR/{n_neutral} NEUTRAL)")
spread_now = (fv_now / current_price - 1) * 100
print(f"Current month FV: ${fv_now:.2f} ({spread_now:+.1f}% {'cheap' if spread_now > 0 else 'rich'})")

if not comparison.empty:
    print("\n--- Forward Curve Relative Value ---")
    print(f"{'Contract':<12s} | {'Ticker':<14s} | {'Market':>7s} | {'Seas FV':>8s} | {'Spread':>7s} | {'vs 1σ':>5s} | {'Signal':<6s}")
    print(f"{'-'*12}-+-{'-'*14}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}")
    for _, row in comparison.iterrows():
        mstr = row['month'].strftime('%b %Y') if hasattr(row['month'], 'strftime') else str(row['month'])
        tkr = row.get('ticker', '')
        sigma_str = f'{row["edge_vs_sigma"]:.2f}σ'
        print(f"{mstr:<12s} | {tkr:<14s} | ${row['price']:>5.2f} | ${row['fv_price']:>6.2f} | {row['spread']:>+6.2f} | {sigma_str:>5s} | {row['signal']:<6s}")

if not comparison.empty:
    print("\n--- ACTIONABLE TRADE RECOMMENDATIONS ---")
    print(f"  (only contracts where edge > 0.5σ model uncertainty, i.e. ±{res_std*100:.0f}%)")
    # Filter to contracts with edge exceeding model uncertainty
    trades = comparison[comparison['edge_vs_sigma'] > 0.5].sort_values('edge_score', ascending=False)
    if len(trades) > 0:
        # Top trades header
        print(f"{'Rank':<5s} {'Ticker':<14s} {'Dir':<6s} {'Entry':>7s} {'Target':>8s} "
              f"{'Stop':>7s} {'Edge':>6s} {'R:R':>5s} {'vs 1σ':>5s}")
        print(f"{'-'*5} {'-'*14} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*6} {'-'*5} {'-'*5}")
        for rank, (_, row) in enumerate(trades.iterrows(), 1):
            mstr = row['month'].strftime('%b %Y') if hasattr(row['month'], 'strftime') else ''
            tkr = row.get('ticker', mstr)
            print(f"{rank:<5d} {tkr:<14s} {row['direction']:<6s} ${row['price']:>5.2f} "
                  f"${row['fv_price']:>6.2f} ${row['stop']:>5.2f} "
                  f"{row['spread_pct']:>+5.1f}% {row['rr_ratio']:>4.1f}x "
                  f"{row['edge_vs_sigma']:>4.1f}σ")
            if rank >= 10:
                break
        # Summary
        n_long = (trades['direction'] == 'LONG').sum()
        n_short = (trades['direction'] == 'SHORT').sum()
        best = trades.iloc[0]
        best_mstr = best['month'].strftime('%b %Y') if hasattr(best['month'], 'strftime') else ''
        print(f"\n  Total: {n_long} LONG / {n_short} SHORT signals")
        print(f"  Best trade: {best['direction']} {best.get('ticker', best_mstr)} "
              f"@ ${best['price']:.2f} → ${best['fv_price']:.2f} "
              f"(edge {best['spread_pct']:+.1f}%, R:R {best['rr_ratio']:.1f}x, "
              f"{best['edge_vs_sigma']:.1f}σ)")
    else:
        print("  No contracts with >1.5% edge found — market roughly at fair value")

    # Calendar spread opportunities
    print("\n--- CALENDAR SPREAD IDEAS ---")
    sig_contracts = comparison[comparison['edge_vs_sigma'] > 0.5]
    cheap = sig_contracts[sig_contracts['spread'] > 0].sort_values('months_ahead')
    rich = sig_contracts[sig_contracts['spread'] < 0].sort_values('months_ahead')
    if len(cheap) > 0 and len(rich) > 0:
        # Buy cheap, sell rich
        for _, c_row in cheap.head(3).iterrows():
            for _, r_row in rich.head(3).iterrows():
                if c_row['months_ahead'] != r_row['months_ahead']:
                    c_tkr = c_row.get('ticker', c_row['month'].strftime('%b%y'))
                    r_tkr = r_row.get('ticker', r_row['month'].strftime('%b%y'))
                    net_edge = c_row['spread_pct'] - r_row['spread_pct']
                    print(f"  BUY {c_tkr} (FV ${c_row['fv_price']:.2f}, mkt ${c_row['price']:.2f}, "
                          f"edge {c_row['spread_pct']:+.1f}%)")
                    print(f"  SELL {r_tkr} (FV ${r_row['fv_price']:.2f}, mkt ${r_row['price']:.2f}, "
                          f"edge {r_row['spread_pct']:+.1f}%)")
                    print(f"  → Net edge: {net_edge:+.1f}%\n")
                    break
            break
    elif len(cheap) > 0:
        print("  No rich contracts for spread — outright longs preferred")
    elif len(rich) > 0:
        print("  No cheap contracts for spread — outright shorts preferred")
    else:
        print("  No strong enough signals for calendar spreads")

if not ung_analysis.empty and len(ung_analysis) > 0:
    print("\n--- UNG CONTANGO IMPACT ---")
    key_months = ung_analysis[ung_analysis['months_ahead'].isin([3, 6, 9, 12])]
    if len(key_months) == 0:
        key_months = ung_analysis.iloc[::3]
    for _, row in key_months.iterrows():
        mstr = row['month'].strftime('%b %Y') if hasattr(row['month'], 'strftime') else str(row['month'])
        cd = row.get('cum_drag', 0) * 100
        print(f"To {mstr}: NG model return {row['ng_return']:+.1f}%, "
              f"contango drag {cd:+.1f}%, UNG equiv {row['ung_equiv']:+.1f}%")

# Factor Dashboard
print("\n--- Factor Dashboard ---")
print(f"{'Factor':<22s} | {'Raw Value':>10s} | {'Z-Score':>8s} | {'Weight':>7s} | {'Signal':<8s}")
print(f"{'-'*22}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
for col, zcol, label, sign in zip(factor_cols, z_cols, factor_labels, factor_signs):
    raw_val = latest_row.get(col, np.nan)
    z_val = latest_row.get(zcol, np.nan)
    wt = ic_for_weights.get(col, 0)
    if np.isnan(z_val):
        sig = 'N/A'
    elif z_val > 0.3:
        sig = 'BULLISH'
    elif z_val < -0.3:
        sig = 'BEARISH'
    else:
        sig = 'NEUTRAL'
    raw_str = f'{raw_val:>10.3f}' if not np.isnan(raw_val) else f'{"N/A":>10s}'
    z_str = f'{z_val:>+8.2f}' if not np.isnan(z_val) else f'{"N/A":>8s}'
    print(f"{label:<22s} | {raw_str} | {z_str} | {wt:>7.3f} | {sig:<8s}")

# ============================================
# Model Confidence & Conviction
# ============================================
print("\n--- Model Confidence & Conviction ---")

# --- 1. Factor Agreement Metrics ---
# Collect active factors' z-scores and weights
active_z_scores = []
active_weights = []
for col, zcol in zip(factor_cols, z_cols):
    z_val = latest_row.get(zcol, np.nan)
    wt = ic_for_weights.get(col, 0)
    if not np.isnan(z_val):
        active_z_scores.append(z_val)
        active_weights.append(wt)

active_signs = [np.sign(z) for z in active_z_scores]

# Consensus: what fraction of weighted factors agree on direction
bullish_weight = sum(w for w, s in zip(active_weights, active_signs) if s > 0)
bearish_weight = sum(w for w, s in zip(active_weights, active_signs) if s < 0)
total_weight = bullish_weight + bearish_weight
consensus = abs(bullish_weight - bearish_weight) / total_weight if total_weight > 0 else 0

# Dispersion: standard deviation of z-scores (high = disagreement)
factor_dispersion = np.std(active_z_scores) if len(active_z_scores) > 1 else 0.0

# Signal strength: absolute composite z-score
signal_strength = abs(current_composite)

# --- 2. Historical Prediction Error Analysis ---
# For each historical month with a composite z-score, compute what FV adjustment
# the model would have predicted, compare to actual 1-month forward return
hist_for_error = fv_data.dropna(subset=['composite', 'ng_price', 'log_fv']).copy()
hist_for_error = hist_for_error.sort_values('date').reset_index(drop=True)
hist_for_error['actual_log_ret'] = np.log(hist_for_error['ng_price']).shift(-1) - np.log(hist_for_error['ng_price'])
hist_for_error['predicted_log_ret'] = hist_for_error['log_fv'] - np.log(hist_for_error['ng_price'])
hist_for_error['prediction_error'] = hist_for_error['actual_log_ret'] - hist_for_error['predicted_log_ret']
hist_for_error = hist_for_error.dropna(subset=['prediction_error'])

# Also compute per-row signal strength, consensus, dispersion for stratification
hist_err_signal_strength = []
hist_err_consensus = []
hist_err_dispersion = []
for idx, row in hist_for_error.iterrows():
    row_z = [row.get(zcol, np.nan) for zcol in z_cols]
    row_w = [ic_for_weights.get(col, 0) for col in factor_cols]
    valid_z = [(z, w) for z, w in zip(row_z, row_w) if not np.isnan(z)]
    if len(valid_z) > 0:
        zs = [z for z, w in valid_z]
        ws = [w for z, w in valid_z]
        bw = sum(w for z, w in valid_z if np.sign(z) > 0)
        brw = sum(w for z, w in valid_z if np.sign(z) < 0)
        tw = bw + brw
        hist_err_consensus.append(abs(bw - brw) / tw if tw > 0 else 0)
        hist_err_signal_strength.append(abs(row.get('composite', 0)))
        hist_err_dispersion.append(np.std(zs) if len(zs) > 1 else 0.0)
    else:
        hist_err_consensus.append(0)
        hist_err_signal_strength.append(0)
        hist_err_dispersion.append(0)

hist_for_error['hist_signal'] = hist_err_signal_strength
hist_for_error['hist_consensus'] = hist_err_consensus
hist_for_error['hist_dispersion'] = hist_err_dispersion

# Stratify by signal strength
def _signal_bucket(s):
    if s < 0.3:
        return 'weak'
    elif s < 0.6:
        return 'moderate'
    elif s < 1.0:
        return 'strong'
    else:
        return 'very_strong'

def _consensus_bucket(c):
    if c < 0.3:
        return 'low'
    elif c < 0.6:
        return 'medium'
    else:
        return 'high'

hist_for_error['sig_bucket'] = hist_for_error['hist_signal'].apply(_signal_bucket)
hist_for_error['cons_bucket'] = hist_for_error['hist_consensus'].apply(_consensus_bucket)

# Current stratum
current_sig_bucket = _signal_bucket(signal_strength)
current_cons_bucket = _consensus_bucket(consensus)

# Get prediction error std for current stratum (signal bucket)
stratum_mask = hist_for_error['sig_bucket'] == current_sig_bucket
stratum_data = hist_for_error[stratum_mask]
if len(stratum_data) >= 10:
    historical_error_std = stratum_data['prediction_error'].std()
else:
    # Fall back to overall
    historical_error_std = hist_for_error['prediction_error'].std() if len(hist_for_error) > 0 else res_std

# Also compute error std stratified by consensus
cons_stratum_mask = hist_for_error['cons_bucket'] == current_cons_bucket
cons_stratum_data = hist_for_error[cons_stratum_mask]
if len(cons_stratum_data) >= 10:
    historical_error_std_cons = cons_stratum_data['prediction_error'].std()
else:
    historical_error_std_cons = historical_error_std

# Use average of signal and consensus stratified estimates
base_uncertainty = (historical_error_std + historical_error_std_cons) / 2.0

# --- 3. Adaptive Confidence Bands ---
# Adjust uncertainty based on current factor metrics
# Less uncertainty when: high consensus, strong signal, low dispersion
# Use sqrt-dampened adjustment to keep multiplier in reasonable range (0.7-1.5)
uncertainty_multiplier = np.sqrt((1 + factor_dispersion) / (1 + consensus * signal_strength))
adaptive_uncertainty = base_uncertainty * uncertainty_multiplier

# Confidence bands in log price space, then convert
fv_log_bands = np.log(fv_now)
bands = {
    '50%': (np.exp(fv_log_bands - 0.674 * adaptive_uncertainty),
            np.exp(fv_log_bands + 0.674 * adaptive_uncertainty)),
    '80%': (np.exp(fv_log_bands - 1.282 * adaptive_uncertainty),
            np.exp(fv_log_bands + 1.282 * adaptive_uncertainty)),
    '95%': (np.exp(fv_log_bands - 1.960 * adaptive_uncertainty),
            np.exp(fv_log_bands + 1.960 * adaptive_uncertainty)),
}

# --- 4. Conviction Score ---
consensus_score = consensus
strength_score = min(1.0, signal_strength / 1.0)
precision_score = 1.0 / (1.0 + factor_dispersion)
n_factors_score = len(active_z_scores) / 15.0

conviction = (
    consensus_score * 0.35 +
    strength_score * 0.25 +
    precision_score * 0.25 +
    n_factors_score * 0.15
) * 100

# Non-linear model agreement adjustment
# If linear and non-linear agree on direction: boost conviction by +10
# If they diverge: reduce conviction by -10
gbm_conviction_adj = 0
if gbm_fv is not None:
    if gbm_agree:
        gbm_conviction_adj = +10
    else:
        gbm_conviction_adj = -10
    # Also factor in relative model quality
    if gbm_nonlinear_r2 > gbm_linear_r2 and gbm_nonlinear_r2 > 0:
        gbm_conviction_adj += 5  # non-linear is better, extra boost
    conviction += gbm_conviction_adj

conviction = max(0, min(100, conviction))

if conviction > 70:
    conviction_label = "HIGH"
elif conviction > 45:
    conviction_label = "MEDIUM"
else:
    conviction_label = "LOW"

# --- 5. Cost-Adjusted Edge ---
raw_edge = (fv_now - current_price) / current_price

information_ratio = raw_edge / adaptive_uncertainty if adaptive_uncertainty > 0 else 0

kelly_conv_pct = raw_edge * (conviction / 100) / (adaptive_uncertainty ** 2) if adaptive_uncertainty > 0 else 0
kelly_conv_pct = max(0, min(1, kelly_conv_pct))

actionable_80 = abs(raw_edge) > 1.282 * adaptive_uncertainty
actionable_95 = abs(raw_edge) > 1.960 * adaptive_uncertainty

# --- 6. Print Output ---
consensus_label = "HIGH" if consensus > 0.6 else "MEDIUM" if consensus > 0.3 else "LOW"

print("Factor Agreement:")
print(f"  Bullish weight: {bullish_weight:.2f} | Bearish weight: {bearish_weight:.2f}")
print(f"  Consensus: {consensus*100:.0f}% ({consensus_label})")
print(f"  Factor dispersion: {factor_dispersion:.2f}")
print(f"  Active factors: {len(active_z_scores)}/{len(factor_cols)}")

print(f"\nConviction Score: {conviction:.0f}/100 ({conviction_label})")
print(f"  Consensus:  {consensus_score * 35:.0f}/35")
print(f"  Strength:   {strength_score * 25:.0f}/25")
print(f"  Precision:  {precision_score * 25:.0f}/25")
print(f"  Breadth:    {n_factors_score * 15:.0f}/15")
if gbm_fv is not None:
    agree_str = "AGREE" if gbm_agree else "DIVERGE"
    print(f"  GBM adj:    {gbm_conviction_adj:+d} (linear/non-linear {agree_str})")

print("\nFair Value Bounds:")
print(f"  Current: ${current_price:.2f}")
print(f"  FV:      ${fv_now:.2f} (raw edge: {raw_edge*100:+.1f}%)")
print(f"            50% band: [${bands['50%'][0]:.2f} - ${bands['50%'][1]:.2f}]")
print(f"            80% band: [${bands['80%'][0]:.2f} - ${bands['80%'][1]:.2f}]")
print(f"            95% band: [${bands['95%'][0]:.2f} - ${bands['95%'][1]:.2f}]")

print("\nEdge Quality:")
print(f"  Information ratio: {information_ratio:.2f}")
print(f"  Cost-adjusted edge: {raw_edge*100:+.1f}%")
print(f"  Actionable at 80%: {'YES' if actionable_80 else 'NO'}")
print(f"  Actionable at 95%: {'YES' if actionable_95 else 'NO'}")
print(f"  Kelly fraction: {kelly_conv_pct*100:.0f}%")

# --- 8. Backtest the Confidence Bands ---
print("\n--- Band Calibration (backtest) ---")
# For each historical month, compute what bands would have been, check coverage
bt_data = hist_for_error.copy()
bt_coverage = {'50%': 0, '80%': 0, '95%': 0}
bt_total = 0

for idx, row in bt_data.iterrows():
    if np.isnan(row.get('log_fv', np.nan)):
        continue
    # Compute this row's adaptive uncertainty
    bt_sig = row['hist_signal']
    bt_cons = row['hist_consensus']
    bt_disp = row['hist_dispersion']

    # Use same stratum logic for base uncertainty (but from data available up to that point)
    bt_sig_bucket = _signal_bucket(bt_sig)
    bt_prior = bt_data.loc[:idx]
    bt_prior_stratum = bt_prior[bt_prior['sig_bucket'] == bt_sig_bucket]
    if len(bt_prior_stratum) >= 5:
        bt_base_unc = bt_prior_stratum['prediction_error'].std()
    else:
        bt_base_unc = bt_prior['prediction_error'].std() if len(bt_prior) > 3 else res_std

    bt_unc_mult = np.sqrt((1 + bt_disp) / (1 + bt_cons * bt_sig))
    bt_adaptive_unc = bt_base_unc * bt_unc_mult

    if bt_adaptive_unc <= 0 or np.isnan(bt_adaptive_unc):
        continue

    # Actual next-month log price
    actual_log_ret = row['actual_log_ret']
    if np.isnan(actual_log_ret):
        continue

    predicted_log_ret = row['predicted_log_ret']
    error = actual_log_ret - predicted_log_ret

    bt_total += 1
    if abs(error) <= 0.674 * bt_adaptive_unc:
        bt_coverage['50%'] += 1
    if abs(error) <= 1.282 * bt_adaptive_unc:
        bt_coverage['80%'] += 1
    if abs(error) <= 1.960 * bt_adaptive_unc:
        bt_coverage['95%'] += 1

if bt_total > 0:
    print(f"  50% band: actual coverage {bt_coverage['50%']/bt_total*100:.0f}% (target: 50%) [n={bt_total}]")
    print(f"  80% band: actual coverage {bt_coverage['80%']/bt_total*100:.0f}% (target: 80%)")
    print(f"  95% band: actual coverage {bt_coverage['95%']/bt_total*100:.0f}% (target: 95%)")
else:
    print("  Insufficient data for backtest")

# ============================================
# Probabilistic Forecast
# ============================================
print("\n--- Probabilistic Forecast ---")

# --- Historical Return Distribution by Composite Signal ---
# Build monthly returns aligned with composite z-scores
hist_comp = analysis.dropna(subset=['composite', 'ng_price']).copy()
hist_comp = hist_comp.sort_values('date').reset_index(drop=True)
hist_comp['ret_1m'] = hist_comp['ng_price'].pct_change(1).shift(-1)
hist_comp['ret_2m'] = (hist_comp['ng_price'].shift(-2) / hist_comp['ng_price'] - 1)

# Bucket composite z-scores
def _comp_bucket(z):
    if z < -0.5:
        return '< -0.5'
    elif z < 0.0:
        return '-0.5 to 0'
    elif z < 0.5:
        return '0 to 0.5'
    elif z < 1.0:
        return '0.5 to 1.0'
    else:
        return '> 1.0'

hist_comp['bucket'] = hist_comp['composite'].apply(_comp_bucket)
current_bucket = _comp_bucket(current_composite)

# Current bucket statistics
bucket_data = hist_comp[hist_comp['bucket'] == current_bucket].dropna(subset=['ret_1m'])
bucket_data_2m = hist_comp[hist_comp['bucket'] == current_bucket].dropna(subset=['ret_2m'])

print(f"Monte Carlo: 10,000 paths | Current: ${current_price:.2f} | FV: ${fv_now:.2f} | "
      f"RVol: {ng_daily['rvol_30d'].dropna().iloc[-1]:.0f}%")

# --- Monte Carlo Price Simulation ---
np.random.seed(42)
n_paths = 10000
horizons = [30, 45, 60]
max_horizon = max(horizons)

# Get current realized vol (annualized, already in %)
rvol_annual = ng_daily['rvol_30d'].dropna().iloc[-1]  # in % terms
daily_vol = rvol_annual / 100.0 / np.sqrt(252)

# Scale daily vol by the uncertainty multiplier from conviction analysis
# uncertainty_multiplier > 1 when factors disagree (wider MC cone)
# uncertainty_multiplier < 1 when factors agree (tighter MC cone)
daily_vol = daily_vol * uncertainty_multiplier

# Mean reversion parameters
# Half-life of NG price deviations from FV is typically 30-60 days
# speed = ln(2) / half_life
# Adjust by conviction: high conviction -> faster mean reversion (shorter half-life)
base_half_life = 45  # days
half_life = max(15, base_half_life * (1 - 0.5 * conviction / 100))  # conviction scales 0-50% reduction
mean_reversion_speed = np.log(2) / half_life  # conviction-adjusted

# Daily drift: mean revert toward FV
log_fv = np.log(fv_now)
log_start = np.log(current_price)

# Simulate paths (vectorized)
# Shape: (n_paths, max_horizon)
random_shocks = np.random.standard_normal((n_paths, max_horizon))

log_prices = np.zeros((n_paths, max_horizon + 1))
log_prices[:, 0] = log_start

for t in range(max_horizon):
    drift = mean_reversion_speed * (log_fv - log_prices[:, t])
    log_prices[:, t + 1] = log_prices[:, t] + drift + daily_vol * random_shocks[:, t]

prices = np.exp(log_prices)

# Compute distributions at each horizon
print("\nPrice Distribution:")
mc_results = {}
for h in horizons:
    p = prices[:, h]
    pcts = np.percentile(p, [5, 25, 50, 75, 95])
    mc_results[h] = {
        'mean': np.mean(p),
        'median': pcts[2],
        'p5': pcts[0],
        'p25': pcts[1],
        'p75': pcts[3],
        'p95': pcts[4],
        'prices': p,
    }
    print(f"  {h}-day horizon:")
    print(f"    5th percentile:  ${pcts[0]:.2f}")
    print(f"    25th percentile: ${pcts[1]:.2f}")
    print(f"    Median:          ${pcts[2]:.2f}")
    print(f"    75th percentile: ${pcts[3]:.2f}")
    print(f"    95th percentile: ${pcts[4]:.2f}")

# Probability of reaching key price levels at 45 days
print("\nProbability of reaching (at 45 days):")
price_levels = [2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.10, 3.20, 3.30, 3.50, 4.00]
p45 = mc_results[45]['prices']
for level in price_levels:
    prob = np.mean(p45 >= level) * 100
    if prob > 0.5 and prob < 99.5:
        print(f"  P(NG > ${level:.2f}): {prob:.0f}%")

# --- Option Payoff Analysis ---
# Jun $2.80 Call, ~45 DTE
strike = 2.80
dte = 45
p_expiry = mc_results[dte]['prices']

itm_mask = p_expiry > strike
p_itm = np.mean(itm_mask) * 100
payoffs = np.maximum(p_expiry - strike, 0)
expected_payoff = np.mean(payoffs)
expected_payoff_contract = expected_payoff * 10000  # NG contract = 10,000 MMBtu

# Estimate premium from Black-76 approximation
# Using current vol and time to expiry
T = dte / 365.0
d1 = (np.log(current_price / strike) + 0.5 * (rvol_annual / 100.0)**2 * T) / ((rvol_annual / 100.0) * np.sqrt(T))
d2 = d1 - (rvol_annual / 100.0) * np.sqrt(T)
bs_premium = current_price * sp_norm.cdf(d1) - strike * sp_norm.cdf(d2)
bs_premium_contract = bs_premium * 10000

ev_ratio = expected_payoff / bs_premium if bs_premium > 0 else float('inf')

# Kelly criterion: f* = (p*b - q) / b where b = avg win / avg loss, p = win prob
if itm_mask.sum() > 0:
    avg_win = payoffs[itm_mask].mean()
    b = avg_win / bs_premium if bs_premium > 0 else 0
    p = p_itm / 100.0
    q = 1 - p
    kelly = (p * b - q) / b if b > 0 else 0
    kelly = max(kelly, 0)  # no negative Kelly
else:
    kelly = 0

print(f"\nJun ${strike:.2f}C Option Analysis ({dte} DTE):")
print(f"  P(ITM at expiry):     {p_itm:.0f}%")
print(f"  Expected payoff:      ${expected_payoff:.4f} per MMBtu (${expected_payoff_contract:.0f})")
print(f"  Estimated premium:    ${bs_premium:.4f} (${bs_premium_contract:.0f})")
print(f"  Expected value ratio: {ev_ratio:.2f}x {'(>1 = positive EV)' if ev_ratio > 1 else '(<1 = negative EV)'}")
print(f"  Kelly fraction:       {kelly*100:.1f}%")

# --- Historical Analog ---
print(f"\nHistorical Analog (composite z ~ {current_composite:+.1f}, bucket '{current_bucket}'):")
if len(bucket_data) >= 5:
    med_1m = bucket_data['ret_1m'].median() * 100
    up_1m = (bucket_data['ret_1m'] > 0).mean() * 100
    pct_1m = np.percentile(bucket_data['ret_1m'].dropna() * 100, [10, 25, 75, 90])
    print(f"  1-month forward: median {med_1m:+.1f}%, up {up_1m:.0f}% of time (n={len(bucket_data)})")
    print(f"    10th/25th/75th/90th: {pct_1m[0]:+.1f}% / {pct_1m[1]:+.1f}% / {pct_1m[2]:+.1f}% / {pct_1m[3]:+.1f}%")
else:
    print(f"  1-month forward: insufficient data (n={len(bucket_data)})")

if len(bucket_data_2m) >= 5:
    med_2m = bucket_data_2m['ret_2m'].median() * 100
    up_2m = (bucket_data_2m['ret_2m'] > 0).mean() * 100
    pct_2m = np.percentile(bucket_data_2m['ret_2m'].dropna() * 100, [10, 25, 75, 90])
    print(f"  2-month forward: median {med_2m:+.1f}%, up {up_2m:.0f}% of time (n={len(bucket_data_2m)})")
    print(f"    10th/25th/75th/90th: {pct_2m[0]:+.1f}% / {pct_2m[1]:+.1f}% / {pct_2m[2]:+.1f}% / {pct_2m[3]:+.1f}%")
else:
    print(f"  2-month forward: insufficient data (n={len(bucket_data_2m)})")

# ============================================
# Charts: 4x2 GridSpec
# ============================================
print("\n--- Creating Charts ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 28))
gs = gridspec.GridSpec(4, 2, height_ratios=[1.2, 0.8, 0.8, 0.8],
                       hspace=0.30, wspace=0.25)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(4)])

C_FAIR = '#1565C0'
C_PRICE = '#333333'
C_BAND = '#90CAF9'
C_BULL = '#2E7D32'
C_BEAR = '#C62828'

# ============================================
# Top-Left: Model FV Curve vs Actual Futures Curve
# ============================================
ax = axes[0, 0]
if not comparison.empty:
    months = comparison['month']
    market = comparison['price']
    model = comparison['fv_price']
    neutral = comparison['fv_neutral']

    # ±1σ band around the adjusted FV (regime-adjusted)
    fv_upper = model * np.exp(+res_std_regime)
    fv_lower = model * np.exp(-res_std_regime)
    ax.fill_between(months, fv_lower, fv_upper, alpha=0.08, color=C_BAND,
                    label=f'±1σ ({res_std_regime*100:.0f}%, {regime_label.split()[0]})')

    # Shade the composite effect: area between neutral and adjusted FV
    composite_dir = 'Bullish' if composite_shift > 0 else 'Bearish'
    comp_color = C_BULL if composite_shift > 0 else C_BEAR
    ax.fill_between(months, neutral, model, alpha=0.30, color=comp_color,
                    label=f'Composite Effect ({composite_dir} {np.exp(composite_shift)*100 - 100:+.1f}%)')

    # Neutral seasonal baseline (dashed)
    ax.plot(months, neutral, color='#999999', linewidth=1.5, linestyle='--', marker='.',
            markersize=4, label='Seasonal FV (neutral)', zorder=2)

    # Adjusted FV (solid, main)
    ax.plot(months, model, color=C_FAIR, linewidth=2.5, marker='o', markersize=6,
            label=f'Seasonal FV + Composite (z={current_composite:+.2f})', zorder=3)

    # Market futures curve
    ax.plot(months, market, color=C_PRICE, linewidth=2.5, marker='s', markersize=5,
            label='Market (Futures)', zorder=3)

    # Fill between adjusted FV and market: green=cheap, red=rich
    ax.fill_between(months, model, market,
                    where=(model >= market), alpha=0.12, color=C_BULL,
                    interpolate=True, label='Cheap (FV > Mkt)')
    ax.fill_between(months, model, market,
                    where=(model < market), alpha=0.12, color=C_BEAR,
                    interpolate=True, label='Rich (FV < Mkt)')

    # Annotate — bold if edge > 0.5σ
    for _, row in comparison.iterrows():
        sp = row['spread']
        sig = row['edge_vs_sigma'] > 0.5
        color = C_BULL if sp > 0 else C_BEAR
        fw = 'bold' if sig else 'normal'
        alpha = 1.0 if sig else 0.5
        ax.annotate(f'{sp:+.2f}', xy=(row['month'], row['fv_price']),
                    textcoords='offset points', xytext=(0, 12),
                    fontsize=7, fontweight=fw, color=color, ha='center', alpha=alpha)

    ax.axhline(y=current_price, color='orange', linewidth=1, linestyle='--', alpha=0.6,
               label=f'Current NG ${current_price:.2f}')

    ax.legend(fontsize=7.5, loc='upper left')
else:
    ax.text(0.5, 0.5, 'No futures curve data', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)

ax.set_ylabel('$/MMBtu', fontsize=11, fontweight='bold')
ax.set_title(f'NG Forward Curve Fair Value — {datetime.now():%Y-%m-%d}', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.grid(True, alpha=0.3)

# ============================================
# [1,1] Cheap/Rich Signal by Contract Month
# ============================================
ax = axes[1, 1]
if not comparison.empty:
    labels_bar = [r['month'].strftime('%b %y') if hasattr(r['month'], 'strftime') else str(r['month'])
                  for _, r in comparison.iterrows()]
    values_bar = comparison['spread_pct'].values
    for i, (_, row) in enumerate(comparison.iterrows()):
        color = C_BULL if row['spread_pct'] > 0 else C_BEAR
        # Solid if edge > 0.5σ (actionable), faded if not
        alpha = 0.8 if row['edge_vs_sigma'] > 0.5 else 0.25
        ax.barh(i, row['spread_pct'], color=color, alpha=alpha, edgecolor='white')

    ax.set_yticks(range(len(labels_bar)))
    ax.set_yticklabels(labels_bar, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()

    # Draw ±1σ threshold lines
    ax.axvline(x=res_std * 100 * 0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axvline(x=-res_std * 100 * 0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.text(res_std * 100 * 0.5, -0.5, '0.5σ', fontsize=7, color='gray', ha='left')

    # Annotate with dollar spread and sigma
    for i, (_, row) in enumerate(comparison.iterrows()):
        sp = row['spread']
        pct = row['spread_pct']
        sig = row['edge_vs_sigma']
        color = C_BULL if pct > 0 else C_BEAR
        alpha = 1.0 if sig > 0.5 else 0.4
        fw = 'bold' if sig > 0.5 else 'normal'
        offset = 0.3 if pct >= 0 else -0.3
        ha = 'left' if pct >= 0 else 'right'
        label = f'${sp:+.2f} ({sig:.1f}σ)' if sig > 0.5 else f'${sp:+.2f}'
        ax.text(pct + offset, i, label, va='center', ha=ha,
                fontsize=7.5, fontweight=fw, color=color, alpha=alpha)

    ax.set_xlabel('Model - Market (%)', fontsize=10)
else:
    ax.text(0.5, 0.5, 'No futures data available', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)

ax.set_title('Cheap/Rich Signal by Contract', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# ============================================
# [1,0] Factor Z-Score Bar Chart (Visual Dashboard)
# ============================================
ax = axes[1, 0]

factor_data = []
for col, zcol, label, sign in zip(factor_cols, z_cols, factor_labels, factor_signs):
    z_val = latest_row.get(zcol, np.nan)
    wt = ic_for_weights.get(col, 0)
    if not np.isnan(z_val):
        factor_data.append({'label': label, 'z': z_val, 'weight': wt})

if factor_data:
    factor_df = pd.DataFrame(factor_data).sort_values('z')
    y_pos = range(len(factor_df))
    max_wt = factor_df['weight'].max() if factor_df['weight'].max() > 0 else 1
    for i, (_, row) in enumerate(factor_df.iterrows()):
        color = C_BULL if row['z'] > 0.3 else C_BEAR if row['z'] < -0.3 else '#999999'
        alpha = 0.4 + 0.6 * (row['weight'] / max_wt)
        ax.barh(i, row['z'], color=color, alpha=alpha, edgecolor='white', height=0.7)
        ha = 'left' if row['z'] >= 0 else 'right'
        offset = 0.05 if row['z'] >= 0 else -0.05
        ax.text(row['z'] + offset, i, f'{row["z"]:+.2f}', va='center', ha=ha,
                fontsize=8, fontweight='bold' if abs(row['z']) > 1 else 'normal')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(factor_df['label'], fontsize=8)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=0.3, color=C_BULL, linewidth=0.5, linestyle=':', alpha=0.4)
    ax.axvline(x=-0.3, color=C_BEAR, linewidth=0.5, linestyle=':', alpha=0.4)
    ax.set_xlabel('Z-Score (sign-adjusted, opacity ∝ IC weight)', fontsize=9)
    # Composite callout
    comp_color = C_BULL if current_composite > 0.3 else C_BEAR if current_composite < -0.3 else '#333'
    ax.text(0.98, 0.02, f'COMPOSITE: {current_composite:+.2f}\n{bias}',
            transform=ax.transAxes, fontsize=11, fontweight='bold', ha='right', va='bottom',
            color=comp_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))

ax.set_title(f'Factor Z-Scores ({int(latest_row["n_factors"])} active, IC-weighted)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# ============================================
# [1,1] Cheap/Rich — add curve avg annotation at end
# (keep existing panel, just add annotation)
# ============================================
# Add curve average annotation to cheap/rich panel
if not comparison.empty:
    curve_avg = comparison['spread_pct'].mean()
    axes[1, 1].text(0.98, 0.02, f'Curve avg: {curve_avg:+.1f}%',
                    transform=axes[1, 1].transAxes, fontsize=8, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ============================================
# [2,0] Weather Demand Panel (HDD + CDD combined)
# ============================================
ax = axes[2, 0]
has_weather_data = not cpc_hdd_monthly.empty or not cpc_cdd_monthly.empty
if has_weather_data:
    # Plot monthly weather demand z-scores over last 24 months
    wd_plot = weather_demand[weather_demand['date'] >= weather_demand['date'].max() - pd.DateOffset(months=24)].copy()
    if len(wd_plot) > 0:
        colors = [C_BULL if v > 0 else C_BEAR for v in wd_plot['weather_demand']]
        ax.bar(wd_plot['date'], wd_plot['weather_demand'], width=25, color=colors, alpha=0.7, edgecolor='white')
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.axhline(y=1, color=C_BULL, linewidth=0.5, linestyle=':', alpha=0.4)
        ax.axhline(y=-1, color=C_BEAR, linewidth=0.5, linestyle=':', alpha=0.4)

        # Label each bar with HDD/CDD
        for _, row in wd_plot.iterrows():
            m = row['date'].month
            dd_type = 'H' if m in [10, 11, 12, 1, 2, 3] else 'C'
            ax.text(row['date'], row['weather_demand'] + (0.1 if row['weather_demand'] >= 0 else -0.15),
                    dd_type, ha='center', va='bottom' if row['weather_demand'] >= 0 else 'top',
                    fontsize=6, color='#666')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.set_ylabel('Z-Score (deviation from normal)', fontsize=9)

        # Current month annotation
        latest_wd = wd_plot.iloc[-1]
        wd_z = latest_wd['weather_demand']
        wd_month = latest_wd['date'].month
        wd_type = 'HDD' if wd_month in [10, 11, 12, 1, 2, 3] else 'CDD'
        wd_signal = 'Bullish' if wd_z > 0.3 else 'Bearish' if wd_z < -0.3 else 'Neutral'
        wd_color = C_BULL if wd_z > 0.3 else C_BEAR if wd_z < -0.3 else '#333'
        ax.text(0.02, 0.98, f'{latest_wd["date"]:%b %Y}: {wd_type} z={wd_z:+.2f} ({wd_signal})',
                transform=ax.transAxes, fontsize=9, va='top', fontweight='bold', color=wd_color,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # Also show weekly HDD from CPC if available
        if not cpc_hdd.empty:
            latest_hdd = cpc_hdd.iloc[-1]
            hdd_week_val = latest_hdd['hdd_week']
            hdd_dev_val = latest_hdd['hdd_dev_norm']
            ax.text(0.98, 0.98, f'This week: {hdd_week_val:.0f} HDD ({hdd_dev_val:+.0f} dev)',
                    transform=ax.transAxes, fontsize=8, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No recent weather demand data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
    ax.grid(True, alpha=0.3)
else:
    ax.axis('off')
    ax.text(0.5, 0.5, 'CPC degree day data unavailable', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)

ax.set_title('Weather Demand: HDD (Oct-Mar) + CDD (Apr-Sep) Z-Scores', fontsize=13, fontweight='bold')

# ============================================
# [2,1] UNG Contango Impact (KEEP)
# ============================================
ax = axes[2, 1]
if not ung.empty and not comparison.empty and len(ung_analysis) > 0:
    months_plot = ung_analysis['month']
    cum_drag_pct = ung_analysis['cum_drag'] * 100
    ng_ret = ung_analysis['ng_return']
    ung_eq = ung_analysis['ung_equiv']

    ax.plot(months_plot, ng_ret, color=C_FAIR, linewidth=2.2, marker='o', markersize=5,
            label='NG Model Return (%)')
    ax.plot(months_plot, ung_eq, color='#FF6F00', linewidth=2.2, marker='s', markersize=5,
            label='UNG Equiv Return (%)')
    ax.fill_between(months_plot, ng_ret, ung_eq, alpha=0.15, color=C_BEAR,
                    label='Contango Drag')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Annotate key months
    for _, row in ung_analysis.iterrows():
        if row['months_ahead'] in [3, 6, 12] or row['months_ahead'] == ung_analysis['months_ahead'].max():
            mstr = row['month'].strftime('%b %y') if hasattr(row['month'], 'strftime') else ''
            ax.annotate(f'{row["ung_equiv"]:+.1f}%',
                        xy=(row['month'], row['ung_equiv']),
                        textcoords='offset points', xytext=(0, -15),
                        fontsize=8, fontweight='bold', color='#FF6F00', ha='center')

    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylabel('Return (%)', fontsize=10, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
elif not ung.empty and len(ung) >= 2:
    # Just show contango structure even without model comparison
    months_plot = ung['month'] if 'month' in ung.columns else ung.index
    ax.bar(range(len(ung)), ung['roll_cost'].fillna(0) * 100, color=C_BEAR, alpha=0.6)
    ax.set_ylabel('Monthly Roll Cost (%)', fontsize=10)
    ax.set_xlabel('Contract #', fontsize=10)
else:
    ax.text(0.5, 0.5, 'Insufficient futures data\nfor UNG analysis',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)

ax.set_title('UNG Contango Impact on Returns', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# ============================================
# [0,1] Historical Model Track Record (NEW — 24 months)
# ============================================
ax = axes[0, 1]
plot_data = fv_data.dropna(subset=['fair_value']).tail(24)
if len(plot_data) > 0:
    ax.fill_between(plot_data['date'], plot_data['fv_lower'], plot_data['fv_upper'],
                    alpha=0.15, color=C_BAND, label='±1σ band')
    ax.plot(plot_data['date'], plot_data['fair_value'], color=C_FAIR, linewidth=2,
            label='Model FV')
    ax.plot(plot_data['date'], plot_data['ng_price'], color=C_PRICE, linewidth=2,
            marker='.', markersize=5, label='Actual NG')

    # Current month highlighted
    ax.plot(plot_data['date'].iloc[-1], plot_data['ng_price'].iloc[-1],
            'o', color='orange', markersize=10, zorder=5, label='Current')

    # Hit rate: % of actuals within ±1σ
    valid_band = plot_data.dropna(subset=['fv_lower', 'fv_upper'])
    if len(valid_band) > 0:
        in_band = ((valid_band['ng_price'] >= valid_band['fv_lower']) &
                   (valid_band['ng_price'] <= valid_band['fv_upper'])).mean() * 100
    else:
        in_band = 0
    ax.text(0.02, 0.98, f'In-band: {in_band:.0f}% | R²: {r2_full:.2f} | σ: {res_std*100:.1f}%',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.legend(fontsize=8)
    ax.set_ylabel('$/MMBtu', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Insufficient model history', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)

ax.set_title('Model Track Record (24mo expanding-window)', fontsize=13, fontweight='bold')

# ============================================
# [3,0] Model Diagnostics & Regime (NEW)
# ============================================
ax = axes[3, 0]
ax.axis('off')

diag_lines = []
diag_lines.append('MODEL DIAGNOSTICS')
diag_lines.append('')
diag_lines.append(f'AR(2) Full-Sample R²:       {r2_full:.3f}')
diag_lines.append(f'Residual Std (full):        {res_std*100:.1f}%')
diag_lines.append(f'Residual Std (regime-adj):  {res_std_regime*100:.1f}%')
diag_lines.append(f'Vol Regime:                 {regime_label}')
diag_lines.append(f'Factors Active:             {int(latest_row["n_factors"])} / {len(factor_cols)}')
diag_lines.append('')

if use_gb:
    diag_lines.append(f'GBM Residual Correction:    {gb_residual_pred*100:+.2f}%')
    diag_lines.append(f'GBM CV R²:                  {gb_cv_r2:.3f}')
    diag_lines.append('')
    diag_lines.append('Top GBM Feature Importances:')
    sorted_imp = sorted(gb_importances.items(), key=lambda x: x[1], reverse=True)[:5]
    for feat, imp in sorted_imp:
        label_clean = feat.replace('_z', '')
        diag_lines.append(f'  {label_clean:<22s}  {imp:.3f}')
else:
    diag_lines.append(f'GBM: skipped (CV R²={gb_cv_r2:.3f} < 0.02)')

diag_lines.append('')
diag_lines.append('PRICE SUMMARY')
diag_lines.append(f'Current NG:     ${current_price:.2f}')
diag_lines.append(f'Linear FV:      ${fv_now:.2f} ({(fv_now/current_price-1)*100:+.1f}%)')
if use_gb:
    diag_lines.append(f'GBM-adj FV:     ${fv_now_gb:.2f} ({(fv_now_gb/current_price-1)*100:+.1f}%)')
diag_lines.append(f'1mo-ahead FV:   ${fv_1m:.2f}')

ax.text(0.05, 0.95, '\n'.join(diag_lines), transform=ax.transAxes,
        fontsize=9, fontfamily='monospace', va='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5', edgecolor='#CCCCCC'))

ax.set_title('Model Diagnostics & Regime', fontsize=13, fontweight='bold')

# ============================================
# [3,1] Condensed Trade Recommendations (Top 5)
# ============================================
ax = axes[3, 1]
ax.axis('off')

if not comparison.empty:
    trades_top = comparison[comparison['edge_vs_sigma'] > 0.5].sort_values(
        'edge_score', ascending=False).head(5)
    if len(trades_top) > 0:
        t_rows = []
        t_labels = []
        for _, row in trades_top.iterrows():
            tkr = row.get('ticker', row['month'].strftime('%b%y'))
            t_labels.append(f'{row["direction"][:1]} {tkr}')
            t_rows.append([
                f"${row['price']:.2f}",
                f"${row['fv_price']:.2f}",
                f"${row['stop']:.2f}",
                f"{row['spread_pct']:+.1f}%",
                f"{row['rr_ratio']:.1f}x",
                f"{row['edge_vs_sigma']:.1f}σ",
            ])

        t_cols = ['Market', 'FV', 'Stop', 'Edge', 'R:R', 'Sigma']
        tbl = ax.table(cellText=t_rows, rowLabels=t_labels, colLabels=t_cols,
                       cellLoc='center', rowLoc='right', loc='upper center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)

        for j in range(len(t_cols)):
            tbl[0, j].set_facecolor('#1565C0')
            tbl[0, j].set_text_props(color='white', fontweight='bold')
        for i, (_, row) in enumerate(trades_top.iterrows()):
            bg = '#E8F5E9' if row['direction'] == 'LONG' else '#FFEBEE'
            for j in range(-1, len(t_cols)):
                tbl[i + 1, j].set_facecolor(bg)
            if row['edge_vs_sigma'] >= 1.0:
                for j in range(-1, len(t_cols)):
                    tbl[i + 1, j].set_text_props(fontweight='bold')

        # Best trade callout
        best = trades_top.iloc[0]
        best_tkr = best.get('ticker', '')
        n_long = (trades_top['direction'] == 'LONG').sum()
        n_short = (trades_top['direction'] == 'SHORT').sum()
        ax.text(0.5, 0.08,
                f'Best: {best["direction"]} {best_tkr} ({best["spread_pct"]:+.1f}%, '
                f'{best["rr_ratio"]:.1f}x R:R) | {n_long}L/{n_short}S',
                transform=ax.transAxes, fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))
    else:
        ax.text(0.5, 0.5, 'No contracts exceed 0.5σ edge threshold',
                transform=ax.transAxes, ha='center', va='center', fontsize=11)
else:
    ax.text(0.5, 0.5, 'No futures data', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)

ax.set_title('Top Trade Recommendations', fontsize=13, fontweight='bold')

# ============================================
# Final layout & suptitle
# ============================================
fig.suptitle(
    f'NG Daily Fair Value Forecast — {datetime.now():%Y-%m-%d}\n'
    f'Composite: {current_composite:+.2f} ({bias}) | FV: ${fv_now:.2f} ({spread_now:+.1f}%) | '
    f'Regime: {regime_label} | Factors: {int(latest_row["n_factors"])}/{len(factor_cols)}',
    fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/home/wyatt/weather/ng_daily_forecast.png', dpi=150, bbox_inches='tight')
print("\nChart saved to ng_daily_forecast.png")

# ============================================
# Probability Cone Chart (separate file)
# ============================================
print("--- Creating Probability Cone Chart ---")

C_BG = '#1a1a2e'
C_TEXT = '#e0e0e0'
C_GRID = '#333355'

fig2, ax2 = plt.subplots(figsize=(14, 8), facecolor=C_BG)
ax2.set_facecolor(C_BG)

days_forward = np.arange(0, max_horizon + 1)

# Compute percentile bands at every day
pct_bands = {}
for pct in [10, 25, 50, 75, 90]:
    pct_bands[pct] = np.percentile(prices, pct, axis=0)

# Fan chart: 10-90, 25-75 bands
ax2.fill_between(days_forward, pct_bands[10], pct_bands[90],
                 alpha=0.15, color='#64B5F6', label='10th-90th percentile')
ax2.fill_between(days_forward, pct_bands[25], pct_bands[75],
                 alpha=0.25, color='#42A5F5', label='25th-75th percentile')
ax2.plot(days_forward, pct_bands[50], color='#FFFFFF', linewidth=2.5,
         label='Median path', zorder=4)

# Individual sample paths (faded)
n_show = 50
for i in range(n_show):
    ax2.plot(days_forward, prices[i, :], color='#64B5F6', alpha=0.04, linewidth=0.5)

# Mark FV line
ax2.axhline(y=fv_now, color='#66BB6A', linewidth=1.5, linestyle='--', alpha=0.8,
            label=f'Fair Value ${fv_now:.2f}')

# Mark current price
ax2.axhline(y=current_price, color='#FFB74D', linewidth=1.5, linestyle='-', alpha=0.8,
            label=f'Current ${current_price:.2f}')

# Mark option strike
ax2.axhline(y=strike, color='#EF5350', linewidth=1.5, linestyle=':', alpha=0.8,
            label=f'${strike:.2f} Strike')

# Mark horizon lines
for h in horizons:
    ax2.axvline(x=h, color=C_GRID, linewidth=0.8, linestyle=':', alpha=0.5)
    # Annotate median and range at this horizon
    med = mc_results[h]['median']
    p5 = mc_results[h]['p5']
    p95 = mc_results[h]['p95']
    ax2.plot(h, med, 'o', color='#FFFFFF', markersize=8, zorder=5)
    ax2.annotate(f'{h}d\n${med:.2f}\n[${p5:.2f}-${p95:.2f}]',
                 xy=(h, p95), xytext=(h + 1.5, p95 + 0.02),
                 fontsize=7.5, color=C_TEXT, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#333355', edgecolor='#555577', alpha=0.9))

# Probability annotations at 45 days
p45_data = mc_results[45]['prices']
anno_levels = [lv for lv in price_levels if np.mean(p45_data >= lv) * 100 > 2 and np.mean(p45_data >= lv) * 100 < 98]
for level in anno_levels[:6]:  # limit to avoid clutter
    prob = np.mean(p45_data >= level) * 100
    ax2.annotate(f'P>${level:.2f}: {prob:.0f}%',
                 xy=(45, level), xytext=(max_horizon + 2, level),
                 fontsize=7, color='#B0BEC5', ha='left', va='center',
                 arrowprops=dict(arrowstyle='-', color='#555577', lw=0.5))

# Option payoff info box
info_text = (
    f'Jun ${strike:.2f}C ({dte} DTE)\n'
    f'P(ITM): {p_itm:.0f}%\n'
    f'E[payoff]: ${expected_payoff:.4f}/MMBtu\n'
    f'Est premium: ${bs_premium:.4f}\n'
    f'EV ratio: {ev_ratio:.2f}x\n'
    f'Kelly: {kelly*100:.1f}%'
)
ax2.text(0.02, 0.02, info_text, transform=ax2.transAxes, fontsize=9,
         fontfamily='monospace', color=C_TEXT, va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#333355', edgecolor='#555577', alpha=0.95))

# Historical analog info box
if len(bucket_data) >= 5:
    analog_text = (
        f'Historical Analog (z ~ {current_composite:+.1f})\n'
        f'1mo: med {bucket_data["ret_1m"].median()*100:+.1f}%, up {(bucket_data["ret_1m"]>0).mean()*100:.0f}%\n'
    )
    if len(bucket_data_2m) >= 5:
        analog_text += f'2mo: med {bucket_data_2m["ret_2m"].median()*100:+.1f}%, up {(bucket_data_2m["ret_2m"]>0).mean()*100:.0f}%'
    ax2.text(0.98, 0.02, analog_text, transform=ax2.transAxes, fontsize=9,
             fontfamily='monospace', color=C_TEXT, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#333355', edgecolor='#555577', alpha=0.95))

# Monte Carlo info box
mc_info = (
    f'Monte Carlo: {n_paths:,} paths\n'
    f'RVol: {rvol_annual:.0f}% | Adaptive σ: {daily_vol*100:.2f}%/d\n'
    f'Mean Rev: t½={half_life:.0f}d (λ={mean_reversion_speed:.4f}/d)\n'
    f'Conviction: {conviction:.0f}/100 ({conviction_label})\n'
    f'FV: ${fv_now:.2f} | Composite: {current_composite:+.2f}'
)
ax2.text(0.02, 0.98, mc_info, transform=ax2.transAxes, fontsize=8,
         fontfamily='monospace', color=C_TEXT, va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#333355', edgecolor='#555577', alpha=0.95))

ax2.set_xlabel('Days Forward', fontsize=12, color=C_TEXT, fontweight='bold')
ax2.set_ylabel('NG Price ($/MMBtu)', fontsize=12, color=C_TEXT, fontweight='bold')
ax2.set_title(
    f'NG Probabilistic Price Cone — {datetime.now():%Y-%m-%d}\n'
    f'10,000 Monte Carlo paths | Mean-reversion to FV ${fv_now:.2f}',
    fontsize=14, fontweight='bold', color=C_TEXT)

ax2.legend(fontsize=9, loc='upper right', facecolor='#333355', edgecolor='#555577',
           labelcolor=C_TEXT)
ax2.tick_params(colors=C_TEXT)
ax2.spines['bottom'].set_color(C_GRID)
ax2.spines['top'].set_color(C_GRID)
ax2.spines['left'].set_color(C_GRID)
ax2.spines['right'].set_color(C_GRID)
ax2.grid(True, alpha=0.2, color=C_GRID)
ax2.set_xlim(0, max_horizon + 8)

plt.tight_layout()
plt.savefig('/home/wyatt/weather/ng_probability_cone.png', dpi=150, bbox_inches='tight',
            facecolor=C_BG)
print("Probability cone chart saved to ng_probability_cone.png")

print("Done.")
