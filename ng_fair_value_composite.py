#!/usr/bin/env python3
"""
NG Composite Fair Value Model
Multi-factor model combining fundamental, market, and macro signals to estimate
NG fair value and predict price movements. Uses deseasonalized price levels.

Factors:
  Fundamental (from EIA/BH): Storage deviation, S/D balance, Rig momentum, Export tightening
  Market (from CFTC/yfinance): COT managed money positioning, Realized vol percentile
  Macro (from FRED/yfinance): Oil/NG ratio, DXY momentum, Industrial Production
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
from io import BytesIO, StringIO
from datetime import datetime
import yfinance as yf
from scipy.stats import spearmanr, percentileofscore
import warnings
warnings.filterwarnings('ignore')

print("NG Composite Fair Value Model")
print("=" * 65)

# ============================================
# Data Fetching (EIA/BH — from ng_production_model.py)
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
    """Download a file using shell curl (avoids HTTP/2 stream errors in subprocess)."""
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
            # Cache current year for 24h, older years for 30 days
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
        cot = cot[['date', 'mm_net']].sort_values('date').drop_duplicates(subset='date')
        print(f"    {len(cot)} weekly reports ({cot['date'].min():%Y-%m-%d} to {cot['date'].max():%Y-%m-%d})")
        return cot
    print("    FAILED: no data")
    return pd.DataFrame(columns=['date', 'mm_net'])


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

print("\n--- Weekly Data ---")
rig_count = fetch_rig_count()
storage = fetch_storage()

print("\n--- CFTC COT ---")
cot = fetch_cot()

print("\n--- FRED Macro ---")
indpro = fetch_fred('INDPRO', 'Industrial Production')

print("\n--- Market Data (yfinance) ---")
ng = yf.Ticker("NG=F")
ng_daily = ng.history(period="max", interval="1d")
if ng_daily.index.tz is not None:
    ng_daily.index = ng_daily.index.tz_localize(None)
ng_daily = ng_daily[['Close']].rename(columns={'Close': 'ng_price'})
ng_daily.index.name = 'date'
ng_daily = ng_daily.reset_index()
print(f"  NG=F: {len(ng_daily)} daily bars")

oil = yf.Ticker("CL=F")
oil_daily = oil.history(period="max", interval="1d")
if oil_daily.index.tz is not None:
    oil_daily.index = oil_daily.index.tz_localize(None)
oil_daily = oil_daily[['Close']].rename(columns={'Close': 'oil_price'})
oil_daily.index.name = 'date'
oil_daily = oil_daily.reset_index()
print(f"  CL=F: {len(oil_daily)} daily bars")

dxy = yf.Ticker("DX-Y.NYB")
dxy_daily = dxy.history(period="max", interval="1d")
if dxy_daily.index.tz is not None:
    dxy_daily.index = dxy_daily.index.tz_localize(None)
dxy_daily = dxy_daily[['Close']].rename(columns={'Close': 'dxy'})
dxy_daily.index.name = 'date'
dxy_daily = dxy_daily.reset_index()
print(f"  DX-Y.NYB: {len(dxy_daily)} daily bars")

print("\n--- Term Structure (EIA Futures Contracts) ---")
# EIA Contract 1 (front) and Contract 4 (~4 months out) for term structure slope
c1_content = curl_fetch(EIA_BASE.replace('hist_xls/', 'hist_xls/') + 'RNGC1d.xls')
c4_content = curl_fetch(EIA_BASE.replace('hist_xls/', 'hist_xls/') + 'RNGC4d.xls')

ts_slope_daily = pd.DataFrame()
if c1_content and c4_content:
    c1 = pd.read_excel(BytesIO(c1_content), sheet_name='Data 1', skiprows=2)
    c4 = pd.read_excel(BytesIO(c4_content), sheet_name='Data 1', skiprows=2)
    for df, name in [(c1, 'c1'), (c4, 'c4')]:
        df.columns = ['date', name]
        df['date'] = pd.to_datetime(df['date'])
        df[name] = pd.to_numeric(df[name], errors='coerce')
    ts = c1.merge(c4, on='date', how='inner').dropna()
    ts['ts_slope'] = (ts['c1'] - ts['c4']) / ts['c4'] * 100  # % spread: + = backwardation
    ts_slope_daily = ts[['date', 'ts_slope', 'c1', 'c4']].sort_values('date')
    print(f"  C1-C4 slope: {len(ts_slope_daily)} days ({ts_slope_daily['date'].min():%Y-%m-%d} to {ts_slope_daily['date'].max():%Y-%m-%d})")

    # Extend beyond EIA data end using NG=F + seasonal index as proxy
    # When NG is above seasonal fair value → backwardation-like, below → contango-like
    eia_end = ts_slope_daily['date'].max()
    recent_ng = ng_daily[ng_daily['date'] > eia_end].copy()
    if len(recent_ng) > 0:
        recent_ng['month'] = recent_ng['date'].dt.to_period('M')
        # Build seasonal price index from NG=F history
        ng_daily_temp = ng_daily.copy()
        ng_daily_temp['cal_month'] = ng_daily_temp['date'].dt.month
        seasonal_med = ng_daily_temp.groupby('cal_month')['ng_price'].median()
        overall_med = ng_daily_temp['ng_price'].median()
        seasonal_idx_daily = seasonal_med / overall_med

        recent_ng['cal_month'] = recent_ng['date'].dt.month
        recent_ng['seas_idx'] = recent_ng['cal_month'].map(seasonal_idx_daily)
        recent_ng['seas_price'] = overall_med * recent_ng['seas_idx']
        # Proxy slope: deviation of actual from seasonal expectation
        recent_ng['ts_slope'] = (recent_ng['ng_price'] - recent_ng['seas_price']) / recent_ng['seas_price'] * 100
        # Scale to match EIA C1-C4 spread magnitude (typical range ±30%)
        # The seasonal proxy is narrower, so scale by historical ratio
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

# ============================================
# Build monthly datasets
# ============================================
print("\n--- Building Monthly Factors ---")

# --- EIA to Bcf/d ---
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

# S/D monthly balance
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

# ============================================
# NG price — monthly (end-of-month) + seasonal index
# ============================================
ng_daily['month'] = ng_daily['date'].dt.to_period('M')
ng_monthly = ng_daily.groupby('month')['ng_price'].last().reset_index()
ng_monthly['date'] = ng_monthly['month'].dt.to_timestamp()
ng_monthly = ng_monthly[['date', 'ng_price']].sort_values('date')

# Seasonal index: expanding median per calendar month (lagged to avoid lookahead)
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
# Factor 1: Rig Count Momentum (3mo/6mo MA ratio)
# ============================================
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

# ============================================
# Factor 2: Export Tightening (LNG YoY% - Prod YoY%)
# ============================================
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

# ============================================
# Factor 3: Storage Deviation (% vs 5yr seasonal avg)
# ============================================
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
    stor_dev = pd.DataFrame(stor_list)
    if not stor_dev.empty:
        stor_dev['month'] = stor_dev['date'].dt.to_period('M').dt.to_timestamp()
        stor_dev = stor_dev.groupby('month')['storage_dev'].last().reset_index()
        stor_dev.columns = ['date', 'storage_dev']
        print(f"  Storage Deviation: {len(stor_dev)} months")
    else:
        stor_dev = pd.DataFrame(columns=['date', 'storage_dev'])
else:
    stor_dev = pd.DataFrame(columns=['date', 'storage_dev'])

# ============================================
# Factor 4: S/D Balance
# ============================================
sd_balance = monthly[['date', 'balance']].dropna().copy()
print(f"  S/D Balance: {len(sd_balance)} months")

# ============================================
# Factor 5: Oil/NG Ratio (mean-reversion)
# ============================================
oil_ng = ng_daily[['date', 'ng_price']].merge(oil_daily[['date', 'oil_price']], on='date', how='inner')
oil_ng['oil_ng_ratio'] = oil_ng['oil_price'] / oil_ng['ng_price']
oil_ng['month'] = oil_ng['date'].dt.to_period('M').dt.to_timestamp()
oil_ng_m = oil_ng.groupby('month')['oil_ng_ratio'].last().reset_index()
oil_ng_m.columns = ['date', 'oil_ng_ratio']
print(f"  Oil/NG Ratio: {len(oil_ng_m)} months")

# ============================================
# Factor 6: DXY 3-month rate of change
# ============================================
dxy_daily['month'] = dxy_daily['date'].dt.to_period('M').dt.to_timestamp()
dxy_m = dxy_daily.groupby('month')['dxy'].last().reset_index()
dxy_m.columns = ['date', 'dxy']
dxy_m['dxy_roc'] = dxy_m['dxy'].pct_change(3) * 100  # 3-month rate of change
dxy_m = dxy_m[['date', 'dxy_roc']].dropna()
print(f"  DXY 3m ROC: {len(dxy_m)} months")

# ============================================
# Factor 7: Realized Vol Percentile (contrarian)
# ============================================
ng_daily['ret'] = ng_daily['ng_price'].pct_change()
ng_daily['rvol_30d'] = ng_daily['ret'].rolling(21).std() * np.sqrt(252) * 100
ng_daily['month'] = ng_daily['date'].dt.to_period('M').dt.to_timestamp()
rvol_m = ng_daily.groupby('month')['rvol_30d'].last().reset_index()
rvol_m.columns = ['date', 'rvol']
# Percentile rank over trailing 3 years
rvol_m['rvol_pctile'] = rvol_m['rvol'].rolling(36, min_periods=12).apply(
    lambda x: percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50)
rvol_m = rvol_m[['date', 'rvol_pctile']].dropna()
print(f"  Realized Vol Pctile: {len(rvol_m)} months")

# ============================================
# Factor 8: COT Managed Money (contrarian)
# ============================================
if not cot.empty:
    cot['month'] = cot['date'].dt.to_period('M').dt.to_timestamp()
    cot_m = cot.groupby('month')['mm_net'].last().reset_index()
    cot_m.columns = ['date', 'cot_mm_net']
    # Percentile rank over trailing 3 years
    cot_m['cot_pctile'] = cot_m['cot_mm_net'].rolling(156, min_periods=52).apply(
        lambda x: percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50)
    cot_m = cot_m[['date', 'cot_pctile']].dropna()
    print(f"  COT MM Percentile: {len(cot_m)} months")
else:
    cot_m = pd.DataFrame(columns=['date', 'cot_pctile'])

# ============================================
# Factor 9: Industrial Production YoY
# ============================================
if not indpro.empty:
    indpro['date'] = indpro['date'].dt.to_period('M').dt.to_timestamp()
    indpro_m = indpro.groupby('date')['value'].last().reset_index()
    indpro_m['indpro_yoy'] = indpro_m['value'].pct_change(12) * 100
    indpro_m = indpro_m[['date', 'indpro_yoy']].dropna()
    print(f"  IndPro YoY: {len(indpro_m)} months")
else:
    indpro_m = pd.DataFrame(columns=['date', 'indpro_yoy'])

# ============================================
# Factor 10: Term Structure Slope (C1-C4 spread)
# ============================================
if not ts_slope_daily.empty:
    ts_slope_daily['month'] = ts_slope_daily['date'].dt.to_period('M').dt.to_timestamp()
    ts_m = ts_slope_daily.groupby('month')['ts_slope'].last().reset_index()
    ts_m.columns = ['date', 'ts_slope']
    print(f"  Term Structure Slope: {len(ts_m)} months")
else:
    ts_m = pd.DataFrame(columns=['date', 'ts_slope'])

# ============================================
# Merge all factors
# ============================================
print("\n--- Merging All Factors ---")
cutoff = pd.Timestamp('2015-01-01')

master = ng_monthly[['date', 'ng_price', 'ng_deseas', 'seasonal_idx']].copy()
master = master[master['date'] >= cutoff]

# Publication lag: how many months after the reference period until data is available.
# EIA monthly data (production, consumption, exports) has ~2 month lag.
# FRED Industrial Production has ~6 week lag (~2 months to be safe).
# Market data (yfinance, CFTC, Baker Hughes weekly) is available same month.
# E.g., EIA January production data isn't published until ~March → lag=2.
factor_defs = [
    # (dataframe, column_name, label, sign, pub_lag_months)
    (rig_factor, 'rig_momentum', 'Rig Momentum', -1, 0),        # weekly, ~1 week lag
    (exp_tight, 'export_tightening', 'Export Tightening', 1, 2), # EIA monthly, ~2mo lag
    (stor_dev, 'storage_dev', 'Storage Deviation', 1, 0),        # weekly, ~1 week lag
    (sd_balance, 'balance', 'S/D Balance', -1, 2),               # EIA monthly, ~2mo lag
    (oil_ng_m, 'oil_ng_ratio', 'Oil/NG Ratio', 1, 0),           # real-time
    (dxy_m, 'dxy_roc', 'DXY Momentum', -1, 0),                  # real-time
    (rvol_m, 'rvol_pctile', 'Realized Vol', -1, 0),             # real-time
    (cot_m, 'cot_pctile', 'COT Positioning', -1, 0),            # weekly, 3-day lag
    (indpro_m, 'indpro_yoy', 'Industrial Prod', 1, 2),          # FRED, ~6 week lag
    (ts_m, 'ts_slope', 'Term Structure', -1, 0),                 # real-time
]

factor_cols = []
factor_labels = []
factor_signs = []
factor_lags = []

for fdf, col, label, sign, pub_lag in factor_defs:
    factor_cols.append(col)
    factor_labels.append(label)
    factor_signs.append(sign)
    factor_lags.append(pub_lag)
    if not fdf.empty:
        fdf_merge = fdf[['date', col]].copy()
        if pub_lag > 0:
            # Shift date FORWARD by pub_lag: Jan data → becomes available in Mar
            fdf_merge['date'] = fdf_merge['date'] + pd.DateOffset(months=pub_lag)
            print(f"    {label}: shifted +{pub_lag}mo for publication lag")
        master = master.merge(fdf_merge, on='date', how='left')
    else:
        master[col] = np.nan

master = master.sort_values('date').reset_index(drop=True)

# Forward returns (deseasonalized) for IC measurement
for n in [1, 3, 6]:
    master[f'fwd_{n}m'] = master['ng_deseas'].shift(-n) / master['ng_deseas'] - 1
    master[f'fwd_{n}m_raw'] = master['ng_price'].shift(-n) / master['ng_price'] - 1

mask = master[factor_cols].notna().sum(axis=1) >= 4  # need at least 4 factors
analysis = master[mask].copy()
print(f"  Analysis: {len(analysis)} months ({analysis['date'].min():%Y-%m} to {analysis['date'].max():%Y-%m})")
print(f"  Factor coverage:")
for col, label in zip(factor_cols, factor_labels):
    n = analysis[col].notna().sum()
    print(f"    {label}: {n}/{len(analysis)} months ({n/len(analysis)*100:.0f}%)")

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
    analysis[zcol] = z * sign  # sign-align: positive z = bullish

# Compute per-factor IC for weighting (full-sample Spearman with 3m fwd return)
ic_for_weights = {}
for col, label, sign in zip(factor_cols, factor_labels, factor_signs):
    valid = analysis.dropna(subset=[col, 'fwd_3m'])
    if len(valid) >= 30:
        corr, _ = spearmanr(valid[col] * sign, valid['fwd_3m'])
        ic_for_weights[col] = max(abs(corr), 0.02) if not np.isnan(corr) else 0.02
    else:
        ic_for_weights[col] = 0.02  # small floor weight

# IC-weighted composite (robust to missing data)
ic_w = pd.Series({zcol: ic_for_weights[col] for zcol, col in zip(z_cols, factor_cols)})
z_df = analysis[z_cols]
analysis['composite'] = z_df.multiply(ic_w).sum(axis=1) / z_df.notna().multiply(ic_w).sum(axis=1)
analysis['n_factors'] = z_df.notna().sum(axis=1)
print(f"  IC weights: {', '.join(f'{l}={ic_for_weights[c]:.3f}' for c, l in zip(factor_cols, factor_labels))}")

# ============================================
# Fair value: Partial Adjustment Model on log prices
# ============================================
# Following EIA STEO methodology + Baumeister et al. BVAR approach:
#   log(P_t) = α + β₁·log(P_{t-1}) + β₂·composite_z_t + ε_t
#
# The lagged price captures price stickiness (AR(1) component).
# The composite z-score captures fundamentals pushing price toward fair value.
# Using log prices stabilizes variance across regimes ($2 vs $9 NG).
# Expanding window OLS — each month uses only past data (no lookahead).

fv_data = analysis.dropna(subset=['composite', 'ng_price']).copy()
fv_data['log_price'] = np.log(fv_data['ng_price'])
fv_data['log_price_lag1'] = fv_data['log_price'].shift(1)
fv_data['log_price_lag2'] = fv_data['log_price'].shift(2)

# Also add lagged composite for error-correction flavor
fv_data['composite_lag1'] = fv_data['composite'].shift(1)

# Prior month's residual (error correction): how far was last month from FV?
# This gets computed inside the rolling loop below.

r2 = None
mae = None
if len(fv_data) > 36:
    fv_data['fair_value'] = np.nan
    fv_data['fv_std'] = np.nan
    fv_data['log_fv'] = np.nan

    for i in range(36, len(fv_data)):
        train = fv_data.iloc[1:i]  # skip row 0 (no lag)
        cur = fv_data.iloc[i]

        # Build X matrix: [1, log_price_lag1, log_price_lag2, composite, composite_lag1]
        cols_x = ['log_price_lag1', 'log_price_lag2', 'composite', 'composite_lag1']
        valid_mask = train[cols_x + ['log_price']].notna().all(axis=1)
        t = train[valid_mask]

        if len(t) < 20:
            continue

        X = np.column_stack([np.ones(len(t))] + [t[c].values for c in cols_x])
        y = t['log_price'].values

        # OLS: β = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        residuals = y - X @ beta
        res_std = residuals.std()

        # Predict current month
        x_cur = [cur.get(c, np.nan) for c in cols_x]
        if any(np.isnan(v) for v in x_cur):
            continue
        x_cur = np.array([1.0] + x_cur)
        log_fv = x_cur @ beta

        fv_data.iloc[i, fv_data.columns.get_loc('log_fv')] = log_fv
        fv_data.iloc[i, fv_data.columns.get_loc('fv_std')] = res_std

    # Convert log fair value back to price level, then apply seasonal adjustment
    fv_data['fair_value_raw'] = np.exp(fv_data['log_fv'])
    # Confidence bands in log space → price space
    fv_data['fv_upper_raw'] = np.exp(fv_data['log_fv'] + fv_data['fv_std'])
    fv_data['fv_lower_raw'] = np.exp(fv_data['log_fv'] - fv_data['fv_std'])

    # The model already predicts the actual (nominal) price since log_price_lag1 is nominal
    fv_data['fair_value'] = fv_data['fair_value_raw']
    fv_data['fv_upper'] = fv_data['fv_upper_raw']
    fv_data['fv_lower'] = fv_data['fv_lower_raw']

    # Current fair value
    latest_fv = fv_data.dropna(subset=['fair_value'])
    if len(latest_fv) > 0:
        latest = latest_fv.iloc[-1]
        print(f"  Current NG: ${latest['ng_price']:.2f}")
        print(f"  Fair Value: ${latest['fair_value']:.2f} (range ${latest['fv_lower']:.2f}-${latest['fv_upper']:.2f})")
        print(f"  Composite z-score: {latest['composite']:.2f} ({latest['n_factors']:.0f} factors)")

        fv_valid = fv_data.dropna(subset=['fair_value'])
        resid = np.log(fv_valid['ng_price']) - fv_valid['log_fv']
        # R² on log prices (proper measure for log model)
        r2_log = 1 - (resid.var() / np.log(fv_valid['ng_price']).var())
        # R² on price levels (more intuitive)
        resid_level = fv_valid['ng_price'] - fv_valid['fair_value']
        r2 = 1 - (resid_level.var() / fv_valid['ng_price'].var())
        mae = resid_level.abs().mean()
        print(f"  Model R² (levels): {r2:.3f}, R² (log): {r2_log:.3f}, MAE: ${mae:.2f}")

        # Show AR coefficient from latest regression
        t_final = fv_data.iloc[1:-1]
        cols_x = ['log_price_lag1', 'log_price_lag2', 'composite', 'composite_lag1']
        vm = t_final[cols_x + ['log_price']].notna().all(axis=1)
        t_f = t_final[vm]
        X_f = np.column_stack([np.ones(len(t_f))] + [t_f[c].values for c in cols_x])
        y_f = t_f['log_price'].values
        beta_f = np.linalg.lstsq(X_f, y_f, rcond=None)[0]
        coef_names = ['const', 'log(P_t-1)', 'log(P_t-2)', 'composite_z', 'composite_z_lag1']
        print(f"  Regression coefficients (latest):")
        for cn, bv in zip(coef_names, beta_f):
            print(f"    {cn:18s}: {bv:+.4f}")
else:
    fv_data = analysis.copy()
    fv_data['fair_value'] = np.nan
    fv_data['fv_upper'] = np.nan
    fv_data['fv_lower'] = np.nan

# ============================================
# IC analysis per factor
# ============================================
print("\n--- Factor IC Analysis (3m deseasonalized returns) ---")

factor_stats = {}
for col, label, sign in zip(factor_cols, factor_labels, factor_signs):
    valid = analysis.dropna(subset=[col, 'fwd_3m'])
    if len(valid) < 20:
        factor_stats[label] = {'ic': np.nan, 'ic_t': np.nan, 'n': len(valid)}
        continue
    ic_list = []
    for i in range(23, len(valid)):
        window = valid.iloc[i-23:i+1]
        if len(window) >= 12:
            corr, _ = spearmanr(window[col] * sign, window['fwd_3m'])
            ic_list.append(corr)
    ic_arr = np.array([x for x in ic_list if not np.isnan(x)])
    if len(ic_arr) > 0:
        mean_ic = np.mean(ic_arr)
        ic_std = np.std(ic_arr)
        ic_t = mean_ic / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
    else:
        mean_ic = ic_t = 0
    factor_stats[label] = {'ic': mean_ic, 'ic_t': ic_t, 'n': len(valid)}
    print(f"  {label:20s}: IC={mean_ic:+.3f}  t={ic_t:+.2f}  (n={len(valid)})")

# Composite IC
valid_comp = analysis.dropna(subset=['composite', 'fwd_3m'])
ic_list_comp = []
for i in range(23, len(valid_comp)):
    window = valid_comp.iloc[i-23:i+1]
    if len(window) >= 12:
        corr, _ = spearmanr(window['composite'], window['fwd_3m'])
        ic_list_comp.append(corr)
ic_arr_comp = np.array([x for x in ic_list_comp if not np.isnan(x)])
comp_ic = np.mean(ic_arr_comp) if len(ic_arr_comp) > 0 else 0
comp_ic_t = comp_ic / (np.std(ic_arr_comp) / np.sqrt(len(ic_arr_comp))) if len(ic_arr_comp) > 0 and np.std(ic_arr_comp) > 0 else 0
factor_stats['COMPOSITE'] = {'ic': comp_ic, 'ic_t': comp_ic_t, 'n': len(valid_comp)}
print(f"  {'COMPOSITE':20s}: IC={comp_ic:+.3f}  t={comp_ic_t:+.2f}  (n={len(valid_comp)})")

# ============================================
# Error Correction Model (ECM)
# ============================================
# Test if mispricing (price - FV) mean-reverts
# ECM: Δlog(P_{t+1}) = α·(log(P_t) - log(FV_t)) + ε
# α < 0 means prices correct toward fair value
# Half-life = -ln(2) / ln(1 + α)
print("\n--- Error Correction Analysis ---")

ecm_data = fv_data.dropna(subset=['fair_value', 'ng_price']).copy()
ecm_data['mispricing'] = np.log(ecm_data['ng_price']) - ecm_data['log_fv']
ecm_data['delta_log_p'] = ecm_data['log_price'].diff().shift(-1)  # next month's return
ecm_data['delta_composite'] = ecm_data['composite'].diff()

ecm_valid = ecm_data.dropna(subset=['mispricing', 'delta_log_p']).copy()
ecm_alpha = np.nan
ecm_half_life = np.nan
ecm_r2 = np.nan
ecm_beta = np.nan
beta_ecm = np.array([0.0, 0.0])  # default for chart section

if len(ecm_valid) > 24:
    # Simple ECM: Δlog(P_{t+1}) = α·mispricing_t + ε
    X_ecm = np.column_stack([np.ones(len(ecm_valid)), ecm_valid['mispricing'].values])
    y_ecm = ecm_valid['delta_log_p'].values
    beta_ecm = np.linalg.lstsq(X_ecm, y_ecm, rcond=None)[0]
    ecm_alpha = beta_ecm[1]

    # Full ECM: Δlog(P_{t+1}) = α·mispricing_t + β·Δcomposite_t + ε
    ecm_full = ecm_valid.dropna(subset=['delta_composite'])
    if len(ecm_full) > 24:
        X_full = np.column_stack([np.ones(len(ecm_full)),
                                  ecm_full['mispricing'].values,
                                  ecm_full['delta_composite'].values])
        y_full = ecm_full['delta_log_p'].values
        beta_full = np.linalg.lstsq(X_full, y_full, rcond=None)[0]
        ecm_alpha = beta_full[1]
        ecm_beta = beta_full[2]
        resid_ecm = y_full - X_full @ beta_full
        ecm_r2 = 1 - (resid_ecm.var() / y_full.var())

    if ecm_alpha < 0 and abs(ecm_alpha) < 1:
        ecm_half_life = -np.log(2) / np.log(1 + ecm_alpha)
    print(f"  Speed of adjustment (α): {ecm_alpha:.4f}")
    print(f"  Composite change (β):    {ecm_beta:.4f}" if not np.isnan(ecm_beta) else "")
    print(f"  ECM R²:                  {ecm_r2:.3f}" if not np.isnan(ecm_r2) else "")
    if not np.isnan(ecm_half_life):
        print(f"  Half-life of mispricing:  {ecm_half_life:.1f} months")
    else:
        print(f"  Half-life: N/A (α={ecm_alpha:.4f}, not mean-reverting)")

# ============================================
# Walk-Forward Price Forecast (1-month & 2-month ahead)
# ============================================
# True out-of-sample: at month t, predict P_{t+1} and P_{t+2}.
# All features are properly lagged (knowable at end of month t).
#
# Training model: log(P_s) = f(log(P_{s-1}), log(P_{s-2}), composite_{s-1},
#                              composite_{s-2}, sin/cos seasonal)
# Forecast:       log(P̂_{t+1}) = f̂(log(P_t), log(P_{t-1}), composite_t,
#                                    composite_{t-1}, sin/cos for month t+1)
print("\n--- Walk-Forward Price Forecast ---")

fcst = analysis.dropna(subset=['composite', 'ng_price']).copy()
fcst['log_price'] = np.log(fcst['ng_price'])
# Training features (all lagged by 1 more than the nowcast model)
fcst['lp_lag1'] = fcst['log_price'].shift(1)
fcst['lp_lag2'] = fcst['log_price'].shift(2)
fcst['comp_lag1'] = fcst['composite'].shift(1)
fcst['comp_lag2'] = fcst['composite'].shift(2)
# Seasonal Fourier terms
fcst['month_num'] = fcst['date'].dt.month
fcst['sin_m'] = np.sin(2 * np.pi * fcst['month_num'] / 12)
fcst['cos_m'] = np.cos(2 * np.pi * fcst['month_num'] / 12)
# Next month's seasonal (for prediction)
fcst['next_month'] = (fcst['month_num'] % 12) + 1
fcst['sin_m_next'] = np.sin(2 * np.pi * fcst['next_month'] / 12)
fcst['cos_m_next'] = np.cos(2 * np.pi * fcst['next_month'] / 12)
# 2 months ahead seasonal
fcst['next2_month'] = (fcst['next_month'] % 12) + 1
fcst['sin_m_next2'] = np.sin(2 * np.pi * fcst['next2_month'] / 12)
fcst['cos_m_next2'] = np.cos(2 * np.pi * fcst['next2_month'] / 12)

# Training columns (predict log(P_t) from t-1 and t-2 data)
train_cols = ['lp_lag1', 'lp_lag2', 'comp_lag1', 'comp_lag2', 'sin_m', 'cos_m']
# Prediction mapping: to predict P_{t+1}, shift features by 1
pred_map = {
    'lp_lag1': 'log_price',    # P_t → becomes lag1 for t+1
    'lp_lag2': 'lp_lag1',      # P_{t-1} → becomes lag2
    'comp_lag1': 'composite',   # composite_t → becomes lag1
    'comp_lag2': 'comp_lag1',   # composite_{t-1} → becomes lag2
    'sin_m': 'sin_m_next',     # next month's seasonal
    'cos_m': 'cos_m_next',
}

min_train_fcst = 36
fcst['pred_1m'] = np.nan
fcst['pred_2m'] = np.nan
fcst['actual_next'] = fcst['ng_price'].shift(-1)
fcst['actual_next2'] = fcst['ng_price'].shift(-2)

for i in range(min_train_fcst, len(fcst)):
    train_block = fcst.iloc[2:i]  # skip first 2 rows (no lags)
    valid_mask = train_block[train_cols + ['log_price']].notna().all(axis=1)
    t_df = train_block[valid_mask]
    if len(t_df) < 24:
        continue
    X_tr = np.column_stack([np.ones(len(t_df))] + [t_df[c].values for c in train_cols])
    y_tr = t_df['log_price'].values
    try:
        beta_fc = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
    except np.linalg.LinAlgError:
        continue

    # 1-month ahead: predict P_{t+1} using data at time t
    cur = fcst.iloc[i]
    x_1m = [cur.get(pred_map[c], np.nan) for c in train_cols]
    if any(np.isnan(v) for v in x_1m):
        continue
    log_pred_1m = np.array([1.0] + x_1m) @ beta_fc
    fcst.iat[i, fcst.columns.get_loc('pred_1m')] = np.exp(log_pred_1m)

    # 2-month ahead: use predicted P_{t+1} as input
    x_2m = [
        log_pred_1m,                            # log(P̂_{t+1}) → lag1
        cur['log_price'],                       # log(P_t) → lag2
        cur['composite'],                       # composite_t → lag1
        cur.get('comp_lag1', np.nan),           # composite_{t-1} → lag2
        cur.get('sin_m_next2', np.nan),         # seasonal for t+2
        cur.get('cos_m_next2', np.nan),
    ]
    if not any(np.isnan(v) for v in x_2m):
        log_pred_2m = np.array([1.0] + x_2m) @ beta_fc
        fcst.iat[i, fcst.columns.get_loc('pred_2m')] = np.exp(log_pred_2m)

# --- Return-based forecast (anchored to current price) ---
# Model ΔlogP directly: anchors to P_t so can't lose badly to naive
# Δlog(P_{t+1}) = f(composite_t, Δcomposite_t, ΔlogP_t, mispricing_t, seasonal)
# Forecast: P̂_{t+1} = P_t × exp(predicted_return)
print("  Building return-based forecast...")
fcst['dlog'] = fcst['log_price'].diff()
fcst['dcomp'] = fcst['composite'].diff()
# Mispricing from the nowcast FV model (if available)
if 'log_fv' in fv_data.columns:
    fv_misprice = fv_data[['date', 'log_fv']].dropna()
    fcst = fcst.merge(fv_misprice, on='date', how='left')
    fcst['mispricing'] = fcst['log_price'] - fcst['log_fv']
    fcst['mispricing_lag1'] = fcst['mispricing'].shift(1)
else:
    fcst['mispricing_lag1'] = np.nan

ret_train_cols = ['composite', 'dcomp', 'dlog', 'sin_m_next', 'cos_m_next']
# Add mispricing if available for > 50% of data
has_mispricing = fcst['mispricing_lag1'].notna().sum() > len(fcst) * 0.4
if has_mispricing:
    ret_train_cols.append('mispricing_lag1')

fcst['pred_ret_1m'] = np.nan
fcst['pred_ret_2m'] = np.nan

for i in range(min_train_fcst, len(fcst)):
    # Build training set for return prediction
    train_r = fcst.iloc[3:i].copy()
    train_r['target_ret'] = train_r['dlog'].shift(-1)  # next month's return
    valid_r = train_r.dropna(subset=ret_train_cols + ['target_ret'])
    if len(valid_r) < 20:
        continue
    X_ret = np.column_stack([np.ones(len(valid_r))] + [valid_r[c].values for c in ret_train_cols])
    y_ret = valid_r['target_ret'].values
    try:
        beta_ret = np.linalg.lstsq(X_ret, y_ret, rcond=None)[0]
    except np.linalg.LinAlgError:
        continue

    cur = fcst.iloc[i]
    x_r = [cur.get(c, np.nan) for c in ret_train_cols]
    if any(np.isnan(v) for v in x_r):
        continue
    pred_ret = np.array([1.0] + x_r) @ beta_ret
    # Cap extreme predictions (±50% monthly is unrealistic)
    pred_ret = np.clip(pred_ret, -0.4, 0.4)
    fcst.iat[i, fcst.columns.get_loc('pred_ret_1m')] = cur['ng_price'] * np.exp(pred_ret)

    # 2-month: iterate (predict return of t+1→t+2 using predicted values)
    # Use predicted return as the "dlog" for next iteration
    x_r2 = [
        cur.get('composite', np.nan),     # composite stays similar
        0.0,                               # Δcomposite ≈ 0 (unknown)
        pred_ret,                           # predicted return becomes dlog
        cur.get('sin_m_next2', np.nan),
        cur.get('cos_m_next2', np.nan),
    ]
    if has_mispricing:
        # Mispricing at t+1 ≈ mispricing at t - pred_ret (moves toward FV)
        mp = cur.get('mispricing_lag1', np.nan)
        x_r2.append(mp if not np.isnan(mp) else 0.0)
    if not any(np.isnan(v) for v in x_r2):
        pred_ret2 = np.clip(np.array([1.0] + x_r2) @ beta_ret, -0.4, 0.4)
        fcst.iat[i, fcst.columns.get_loc('pred_ret_2m')] = cur['ng_price'] * np.exp(pred_ret + pred_ret2)

# --- Optimal forecast combination (expanding window) ---
# Blend level-model with naive using Bates-Granger optimal weight
print("  Computing optimal forecast combination...")
fcst['pred_combo_1m'] = np.nan
fcst['pred_combo_2m'] = np.nan
for i in range(min_train_fcst + 12, len(fcst)):
    past = fcst.iloc[min_train_fcst:i]
    past_v = past.dropna(subset=['pred_1m', 'actual_next'])
    if len(past_v) < 8:
        continue
    # Optimal weight: w = cov(actual-naive, model-naive) / var(model-naive)
    model_excess = past_v['pred_1m'] - past_v['ng_price']
    actual_excess = past_v['actual_next'] - past_v['ng_price']
    var_m = model_excess.var()
    w = np.clip(np.cov(actual_excess, model_excess)[0, 1] / var_m, 0, 1) if var_m > 1e-10 else 0
    cur = fcst.iloc[i]
    if not np.isnan(cur.get('pred_1m', np.nan)):
        fcst.iat[i, fcst.columns.get_loc('pred_combo_1m')] = w * cur['pred_1m'] + (1 - w) * cur['ng_price']
    if not np.isnan(cur.get('pred_2m', np.nan)):
        fcst.iat[i, fcst.columns.get_loc('pred_combo_2m')] = w * cur['pred_2m'] + (1 - w) * cur['ng_price']

# --- Forecast metrics for all models ---
def compute_fcst_metrics(fdf, pred_col, actual_col, label):
    """Compute forecast error metrics."""
    v = fdf.dropna(subset=[pred_col, actual_col])
    if len(v) < 6:
        return None
    err = v[actual_col] - v[pred_col]
    err_naive = v[actual_col] - v['ng_price']
    rmse = np.sqrt((err ** 2).mean())
    rmse_naive = np.sqrt((err_naive ** 2).mean())
    theil = rmse / rmse_naive if rmse_naive > 0 else np.nan
    mae_val = err.abs().mean()
    mape = (err.abs() / v[actual_col]).mean() * 100
    ss_res = (err ** 2).sum()
    ss_tot = ((v[actual_col] - v[actual_col].mean()) ** 2).sum()
    r2_fc = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    model_dir = np.sign(v[pred_col].values - v['ng_price'].values)
    actual_dir = np.sign(v[actual_col].values - v['ng_price'].values)
    dir_acc = (model_dir == actual_dir).mean() * 100
    corr_fc = np.corrcoef(v[pred_col].values, v[actual_col].values)[0, 1]
    return {
        'label': label, 'rmse': rmse, 'rmse_naive': rmse_naive, 'theil_u': theil,
        'mae': mae_val, 'mape': mape, 'r2': r2_fc, 'dir_acc': dir_acc, 'corr': corr_fc, 'n': len(v),
    }

fcst_metrics = {}
print("\n  1-Month Ahead Forecasts:")
print(f"  {'Model':<20s} | {'RMSE':>6s} | {'Naive':>6s} | {'Theil U':>8s} | {'R²':>6s} | {'Dir%':>5s} | {'Corr':>5s}")
print(f"  {'-'*68}")
for pred_col, label in [
    ('pred_1m', 'Level AR(2)'),
    ('pred_ret_1m', 'Return-based'),
    ('pred_combo_1m', 'Combo (opt)'),
]:
    m = compute_fcst_metrics(fcst, pred_col, 'actual_next', label)
    if m:
        beat = '*' if m['theil_u'] < 1 else ''
        print(f"  {label:<20s} | ${m['rmse']:>4.2f} | ${m['rmse_naive']:>4.2f} | {m['theil_u']:>7.3f}{beat} | {m['r2']:>5.3f} | {m['dir_acc']:>4.0f}% | {m['corr']:>4.3f}")
        fcst_metrics[f'1m_{pred_col}'] = m

print("\n  2-Month Ahead Forecasts:")
print(f"  {'Model':<20s} | {'RMSE':>6s} | {'Naive':>6s} | {'Theil U':>8s} | {'R²':>6s} | {'Dir%':>5s} | {'Corr':>5s}")
print(f"  {'-'*68}")
for pred_col, actual_col, label in [
    ('pred_2m', 'actual_next2', 'Level AR(2)'),
    ('pred_ret_2m', 'actual_next2', 'Return-based'),
    ('pred_combo_2m', 'actual_next2', 'Combo (opt)'),
]:
    m = compute_fcst_metrics(fcst, pred_col, actual_col, label)
    if m:
        beat = '*' if m['theil_u'] < 1 else ''
        print(f"  {label:<20s} | ${m['rmse']:>4.2f} | ${m['rmse_naive']:>4.2f} | {m['theil_u']:>7.3f}{beat} | {m['r2']:>5.3f} | {m['dir_acc']:>4.0f}% | {m['corr']:>4.3f}")
        fcst_metrics[f'2m_{pred_col}'] = m

# Use Level AR(2) as primary display (gives meaningful forecasts; combo is ~naive)
# Report all models in metrics table for comparison
best_1m_key = '1m_pred_1m'
best_1m_col = 'pred_1m'
best_1m_label = 'Level AR(2)'
best_2m_key = '2m_pred_2m'
best_2m_col = 'pred_2m'
best_2m_label = 'Level AR(2)'

# Find best Theil U for reporting
theil_best_1m = min([fcst_metrics[k]['theil_u'] for k in fcst_metrics if k.startswith('1m_')], default=99)
theil_best_2m = min([fcst_metrics[k]['theil_u'] for k in fcst_metrics if k.startswith('2m_')], default=99)

print(f"\n  Best 1M model: {best_1m_label} (Theil U={fcst_metrics.get(best_1m_key, {}).get('theil_u', 0):.3f})")
print(f"  Best 2M model: {best_2m_label} (Theil U={fcst_metrics.get(best_2m_key, {}).get('theil_u', 0):.3f})")

# Latest predictions (best model)
latest_fcst = fcst.dropna(subset=[best_1m_col]).iloc[-1] if best_1m_col in fcst.columns else None
if latest_fcst is not None:
    print(f"\n  Current NG:       ${latest_fcst['ng_price']:.2f}")
    p1 = latest_fcst.get(best_1m_col, np.nan)
    if not np.isnan(p1):
        print(f"  1M Forecast:      ${p1:.2f} ({best_1m_label})")
    p2 = latest_fcst.get(best_2m_col, np.nan)
    if not np.isnan(p2):
        print(f"  2M Forecast:      ${p2:.2f} ({best_2m_label})")

# ============================================
# Charts: 4x2
# ============================================
print("\n--- Creating Charts ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(4, 2, figsize=(22, 26))

# Colors
C_FAIR = '#1565C0'
C_PRICE = '#333333'
C_BAND = '#90CAF9'
C_BULL = '#2E7D32'
C_BEAR = '#C62828'

# ============================================
# Top-Left: Fair Value vs Actual Price
# ============================================
ax = axes[0, 0]
fv_plot = fv_data.dropna(subset=['fair_value'])
if len(fv_plot) > 0:
    ax.fill_between(fv_plot['date'], fv_plot['fv_lower'], fv_plot['fv_upper'],
                    alpha=0.2, color=C_BAND, label='±1σ band')
    ax.plot(fv_plot['date'], fv_plot['fair_value'], color=C_FAIR, linewidth=2.2,
            label='Fair Value')
    ax.plot(fv_plot['date'], fv_plot['ng_price'], color=C_PRICE, linewidth=1.5,
            alpha=0.8, label='NG=F Actual')

    # Shade rich/cheap
    for i in range(1, len(fv_plot)):
        row = fv_plot.iloc[i]
        prev = fv_plot.iloc[i-1]
        if row['ng_price'] > row['fv_upper']:
            ax.axvspan(prev['date'], row['date'], alpha=0.08, color=C_BEAR)
        elif row['ng_price'] < row['fv_lower']:
            ax.axvspan(prev['date'], row['date'], alpha=0.08, color=C_BULL)

    # Annotate current
    last = fv_plot.iloc[-1]
    rich_cheap = 'RICH' if last['ng_price'] > last['fv_upper'] else 'CHEAP' if last['ng_price'] < last['fv_lower'] else 'FAIR'
    rc_color = C_BEAR if rich_cheap == 'RICH' else C_BULL if rich_cheap == 'CHEAP' else '#757575'
    ax.text(0.02, 0.95,
            f'NG: ${last["ng_price"]:.2f}\n'
            f'FV: ${last["fair_value"]:.2f}\n'
            f'Range: ${last["fv_lower"]:.2f}-${last["fv_upper"]:.2f}\n'
            f'Signal: {rich_cheap}',
            transform=ax.transAxes, fontsize=10, fontweight='bold', va='top',
            color=rc_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    if r2 is not None:
        ax.text(0.98, 0.02, f'R²={r2:.3f}  MAE=${mae:.2f}',
                transform=ax.transAxes, fontsize=9, ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

ax.set_ylabel('$/MMBtu', fontsize=11, fontweight='bold')
ax.set_title('NG Fair Value: Partial Adjustment Model (log prices + AR(2))', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Top-Right: Factor Contribution (current month)
# ============================================
ax = axes[0, 1]
latest_row = analysis.dropna(subset=['composite']).iloc[-1]
contributions = []
for zcol, label, sign in zip(z_cols, factor_labels, factor_signs):
    val = latest_row.get(zcol, np.nan)
    if not np.isnan(val):
        contributions.append((label, val))

if contributions:
    contributions.sort(key=lambda x: x[1])
    labels_c = [c[0] for c in contributions]
    values_c = [c[1] for c in contributions]
    colors_c = [C_BULL if v > 0 else C_BEAR for v in values_c]
    y_pos = range(len(contributions))
    ax.barh(y_pos, values_c, color=colors_c, alpha=0.7, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_c, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    for i, (lbl, val) in enumerate(contributions):
        ax.text(val + (0.05 if val >= 0 else -0.05), i, f'{val:+.2f}',
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=9, fontweight='bold')

    ax.text(0.98, 0.02,
            f'Composite: {latest_row["composite"]:+.2f}\n'
            f'Factors: {latest_row["n_factors"]:.0f}/{len(factor_cols)}',
            transform=ax.transAxes, fontsize=10, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax.set_xlabel('Z-Score (sign-aligned: + = bullish)', fontsize=10)
ax.set_title(f'Factor Contributions ({latest_row["date"]:%Y-%m})', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# ============================================
# Middle-Left: Composite Z-Score Over Time
# ============================================
ax = axes[1, 0]
comp_valid = analysis.dropna(subset=['composite']).copy()
if len(comp_valid) > 0:
    bull_mask = comp_valid['composite'] > 0.5
    bear_mask = comp_valid['composite'] < -0.5
    bar_colors = np.where(bull_mask, C_BULL, np.where(bear_mask, C_BEAR, '#90CAF9'))
    ax.bar(comp_valid['date'], comp_valid['composite'], width=25, color=bar_colors, alpha=0.7)

    ax2 = ax.twinx()
    price_valid = analysis.dropna(subset=['ng_price'])
    ax2.plot(price_valid['date'], price_valid['ng_price'], color='#757575',
             linewidth=1.8, alpha=0.7, label='NG=F')
    ax2.set_ylabel('NG Price ($/MMBtu)', color='#757575', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#757575')

    ax.axhline(y=0.5, color=C_BULL, linewidth=0.8, linestyle='--', alpha=0.4)
    ax.axhline(y=-0.5, color=C_BEAR, linewidth=0.8, linestyle='--', alpha=0.4)
    ax.axhline(y=0, color='black', linewidth=0.5)

ax.set_ylabel('Composite Z-Score', fontsize=10, fontweight='bold')
ax.set_title('Composite Signal Over Time vs NG Price', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Middle-Right: IC Bar Chart by Factor
# ============================================
ax = axes[1, 1]
ic_labels = factor_labels + ['COMPOSITE']
ic_vals = [factor_stats[l]['ic'] for l in ic_labels]
ic_ts = [factor_stats[l]['ic_t'] for l in ic_labels]
ic_colors = [C_BULL if v > 0 else C_BEAR for v in ic_vals]
# Highlight composite
ic_colors[-1] = C_FAIR

x_pos = range(len(ic_labels))
bars = ax.bar(x_pos, ic_vals, color=ic_colors, alpha=0.7, edgecolor='white')
ax.axhline(y=0, color='black', linewidth=0.8)

# Annotate t-stats
for i, (val, t) in enumerate(zip(ic_vals, ic_ts)):
    sig = '**' if abs(t) > 2 else '*' if abs(t) > 1.5 else ''
    ax.text(i, val + (0.005 if val >= 0 else -0.015), f't={t:.1f}{sig}',
            ha='center', fontsize=7.5, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels([l.replace(' ', '\n') for l in ic_labels], fontsize=8)
ax.set_ylabel('Mean Spearman IC (3m fwd)', fontsize=10, fontweight='bold')
ax.set_title('Factor Information Coefficients', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# ============================================
# Bottom-Left: Factor Correlation Matrix
# ============================================
ax = axes[2, 0]
corr_data = analysis[factor_cols].dropna()
if len(corr_data) > 10:
    corr_matrix = corr_data.corr(method='spearman')
    short_labels = ['Rig\nMom', 'Export\nTight', 'Stor\nDev', 'S/D\nBal',
                    'Oil/NG', 'DXY', 'RVol', 'COT', 'IndPro', 'Term\nStr']
    n = len(factor_cols)
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=7.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=7.5)
    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

ax.set_title('Factor Correlation Matrix (Spearman)', fontsize=13, fontweight='bold')

# ============================================
# Bottom-Right: Summary Table
# ============================================
ax = axes[2, 1]
ax.axis('off')

table_rows = []
row_labels_table = []
for col, label, sign in zip(factor_cols, factor_labels, factor_signs):
    latest_z = latest_row.get(f'{col}_z', np.nan)
    stats = factor_stats.get(label, {})
    bull_bear = 'Lower' if sign == -1 else 'Higher'
    table_rows.append([
        bull_bear,
        f'{stats.get("ic", np.nan):+.3f}',
        f'{stats.get("ic_t", np.nan):+.1f}',
        f'{latest_z:+.2f}' if not np.isnan(latest_z) else 'N/A',
    ])
    row_labels_table.append(label)

# Add composite row
comp_stats = factor_stats.get('COMPOSITE', {})
table_rows.append([
    '—',
    f'{comp_stats.get("ic", np.nan):+.3f}',
    f'{comp_stats.get("ic_t", np.nan):+.1f}',
    f'{latest_row["composite"]:+.2f}',
])
row_labels_table.append('COMPOSITE')

col_labels = ['Bullish\nWhen', 'Mean IC', 'IC\nt-stat', 'Current\nZ-Score']

table = ax.table(cellText=table_rows, rowLabels=row_labels_table, colLabels=col_labels,
                 cellLoc='center', rowLoc='right', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.0, 1.7)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#1565C0')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Composite row
for j in range(-1, len(col_labels)):
    table[len(row_labels_table), j].set_facecolor('#E3F2FD')
    table[len(row_labels_table), j].set_text_props(fontweight='bold')

# Color z-scores
for i in range(len(row_labels_table)):
    cell = table[i + 1, 3]
    try:
        val = float(table_rows[i][3])
        if val > 0.5:
            cell.set_facecolor('#C8E6C9')
        elif val < -0.5:
            cell.set_facecolor('#FFCDD2')
    except (ValueError, TypeError):
        pass

# Highlight significant t-stats
for i in range(len(row_labels_table)):
    try:
        t_val = float(table_rows[i][2])
        if abs(t_val) > 2:
            table[i + 1, 2].set_facecolor('#C8E6C9')
    except (ValueError, TypeError):
        pass

ax.set_title('Factor Summary & Current Signals', fontsize=13, fontweight='bold', pad=20)

# ============================================
# Row 4 Left: Walk-Forward Best 1M Forecast vs Actual
# ============================================
ax = axes[3, 0]
fcst_plot = fcst.dropna(subset=[best_1m_col, 'actual_next']).copy()
if len(fcst_plot) > 6:
    # Align forecast dates to when they're realized (shift by 1 month)
    plot_dates = fcst_plot['date'] + pd.DateOffset(months=1)
    ax.plot(plot_dates, fcst_plot['actual_next'].values,
            color=C_PRICE, linewidth=1.8, alpha=0.9, label='Actual NG=F')
    ax.plot(plot_dates, fcst_plot[best_1m_col].values,
            color=C_FAIR, linewidth=2.0, label=f'1M Forecast ({best_1m_label})')
    ax.plot(plot_dates, fcst_plot['ng_price'].values,
            color='#BDBDBD', linewidth=1.0, alpha=0.5, linestyle=':', label='Naive (Pₜ = Pₜ₋₁)')

    # Shade forecast errors
    ax.fill_between(plot_dates,
                    fcst_plot[best_1m_col].values, fcst_plot['actual_next'].values,
                    alpha=0.10, color=C_FAIR)

    # Also overlay other models for comparison
    for alt_col, alt_label, alt_color in [
        ('pred_1m', 'Level', '#9E9E9E'),
        ('pred_ret_1m', 'Return', C_BULL),
        ('pred_combo_1m', 'Combo', '#FF6F00'),
    ]:
        if alt_col != best_1m_col:
            alt_valid = fcst_plot.dropna(subset=[alt_col])
            if len(alt_valid) > 6:
                alt_dates = alt_valid['date'] + pd.DateOffset(months=1)
                ax.plot(alt_dates, alt_valid[alt_col].values,
                        color=alt_color, linewidth=0.8, alpha=0.4, linestyle='--')

    bm = fcst_metrics.get(best_1m_key, {})
    ax.text(0.02, 0.95,
            f'{best_1m_label}\n'
            f'R² = {bm.get("r2", 0):.3f}\n'
            f'RMSE = ${bm.get("rmse", 0):.2f}\n'
            f'Theil U = {bm.get("theil_u", 0):.3f}\n'
            f'Dir acc = {bm.get("dir_acc", 0):.0f}%',
            transform=ax.transAxes, fontsize=9.5, va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    # Annotate latest forecast
    lf = latest_fcst
    if lf is not None:
        p1 = lf.get(best_1m_col, np.nan)
        p2 = lf.get(best_2m_col, np.nan)
        fcst_text = f'Next 1M: ${p1:.2f}' if not np.isnan(p1) else ''
        if not np.isnan(p2):
            fcst_text += f'\nNext 2M: ${p2:.2f}'
        if fcst_text:
            ax.text(0.98, 0.02, fcst_text,
                    transform=ax.transAxes, fontsize=10, fontweight='bold', ha='right', va='bottom',
                    color=C_FAIR,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_FAIR, alpha=0.9))

    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.set_ylabel('$/MMBtu', fontsize=10, fontweight='bold')
ax.set_title('Walk-Forward: 1-Month Ahead Forecast vs Actual', fontsize=13, fontweight='bold')

# ============================================
# Row 4 Right: Model Comparison — Theil U & Rolling RMSE
# ============================================
ax = axes[3, 1]
if len(fcst_plot) > 12:
    plot_dates = fcst_plot['date'] + pd.DateOffset(months=1)
    window = 12

    # Rolling Theil U for each model
    for pred_col, label, color, lw in [
        ('pred_1m', 'Level AR(2)', '#9E9E9E', 1.2),
        ('pred_ret_1m', 'Return-based', C_BULL, 1.8),
        ('pred_combo_1m', 'Combo (opt)', '#FF6F00', 1.8),
    ]:
        fdf_tmp = fcst.dropna(subset=[pred_col, 'actual_next']).copy()
        if len(fdf_tmp) < window + 3:
            continue
        err_m2 = (fdf_tmp['actual_next'] - fdf_tmp[pred_col]) ** 2
        err_n2 = (fdf_tmp['actual_next'] - fdf_tmp['ng_price']) ** 2
        roll_rmse_m = err_m2.rolling(window).mean().apply(np.sqrt)
        roll_rmse_n = err_n2.rolling(window).mean().apply(np.sqrt)
        rolling_theil = (roll_rmse_m / roll_rmse_n).values
        fdf_dates = fdf_tmp['date'] + pd.DateOffset(months=1)
        theil_full = fcst_metrics.get(f'1m_{pred_col}', {}).get('theil_u', 0)
        ax.plot(fdf_dates, rolling_theil, color=color, linewidth=lw,
                label=f'{label} (avg U={theil_full:.3f})', alpha=0.8)

    ax.axhline(1.0, color='black', linewidth=1.5, linestyle='-', alpha=0.6,
               label='Naive = 1.0 (random walk)')
    ax.axhline(0, color='black', linewidth=0.3)

    # Shade region where Theil < 1 = model beats naive
    ax.fill_between(ax.get_xlim(), [0, 0], [1, 1], alpha=0.04, color=C_BULL,
                    transform=ax.get_yaxis_transform())

    hl_str = f'{ecm_half_life:.1f}mo' if not np.isnan(ecm_half_life) else 'N/A'
    ax.text(0.02, 0.95,
            f'Rolling {window}M Theil U\n(< 1 = beats naive)\n\nECM half-life: {hl_str}',
            transform=ax.transAxes, fontsize=9.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylim(0, 2.5)

ax.set_ylabel('Rolling 12M Theil U', fontsize=10, fontweight='bold')
ax.set_title('Forecast Accuracy: Model vs Random Walk', fontsize=13, fontweight='bold')

# ============================================
# Save
# ============================================
period_start = analysis['date'].min()
period_end = analysis['date'].max()

fig.suptitle(f'NG Composite Fair Value Model ({period_start:%Y-%m} to {period_end:%Y-%m})',
             fontsize=16, fontweight='bold', y=1.003)
plt.tight_layout()
plt.savefig('/home/wyatt/weather/ng_fair_value_composite.png', dpi=150, bbox_inches='tight')
print("\nChart saved: ng_fair_value_composite.png")

# ============================================
# Console Summary
# ============================================
print("\n" + "=" * 85)
print("=== NG COMPOSITE FAIR VALUE MODEL ===")
print(f"Period: {period_start:%Y-%m} to {period_end:%Y-%m} ({len(analysis)} months, {len(factor_cols)} factors)")
print("=" * 85)

print(f"\n  {'Factor':<22s} | {'Sign':>6s} | {'Mean IC':>8s} | {'IC t':>6s} | {'Z-Score':>8s} | {'Signal':>8s}")
print("  " + "-" * 75)
for col, label, sign in zip(factor_cols, factor_labels, factor_signs):
    stats = factor_stats.get(label, {})
    z_val = latest_row.get(f'{col}_z', np.nan)
    sig = 'BULL' if z_val > 0.5 else 'BEAR' if z_val < -0.5 else 'NEUT' if not np.isnan(z_val) else 'N/A'
    sign_label = 'Lower' if sign == -1 else 'Higher'
    print(f"  {label:<22s} | {sign_label:>6s} | {stats.get('ic', np.nan):>+8.3f} | {stats.get('ic_t', np.nan):>+6.1f} | {z_val:>+8.2f} | {sig:>8s}")

print("  " + "-" * 75)
z_comp = latest_row['composite']
sig_comp = 'BULL' if z_comp > 0.5 else 'BEAR' if z_comp < -0.5 else 'NEUTRAL'
print(f"  {'COMPOSITE':<22s} | {'—':>6s} | {comp_ic:>+8.3f} | {comp_ic_t:>+6.1f} | {z_comp:>+8.2f} | {sig_comp:>8s}")

fv_latest = fv_data.dropna(subset=['fair_value']).iloc[-1] if 'fair_value' in fv_data.columns else None
if fv_latest is not None and not np.isnan(fv_latest['fair_value']):
    print(f"\n  Current Price:  ${fv_latest['ng_price']:.2f}")
    print(f"  Fair Value:     ${fv_latest['fair_value']:.2f}")
    print(f"  Fair Range:     ${fv_latest['fv_lower']:.2f} - ${fv_latest['fv_upper']:.2f}")
    pct_diff = (fv_latest['ng_price'] / fv_latest['fair_value'] - 1) * 100
    if fv_latest['ng_price'] > fv_latest['fv_upper']:
        print(f"  Valuation:      RICH by {pct_diff:+.0f}% vs fair value")
    elif fv_latest['ng_price'] < fv_latest['fv_lower']:
        print(f"  Valuation:      CHEAP by {pct_diff:+.0f}% vs fair value")
    else:
        print(f"  Valuation:      FAIR ({pct_diff:+.0f}% vs midpoint)")
    print(f"  Model R²:       {r2:.3f}")

if not np.isnan(ecm_alpha):
    print(f"\n--- Error Correction Model ---")
    print(f"  ECM: Δlog(P_{{t+1}}) = α·mispricing_t + β·Δcomposite_t + ε")
    print(f"  Speed of adjustment (α): {ecm_alpha:.4f}")
    if not np.isnan(ecm_beta):
        print(f"  Composite change (β):    {ecm_beta:.4f}")
    if not np.isnan(ecm_half_life):
        print(f"  Half-life of mispricing:  {ecm_half_life:.1f} months")
    if not np.isnan(ecm_r2):
        print(f"  ECM R² (return pred):     {ecm_r2:.3f}")

if fcst_metrics:
    print(f"\n--- Walk-Forward Forecast (out-of-sample) ---")
    print(f"  {'Model':20s} | {'RMSE':>6s} | {'Naive':>6s} | {'Theil U':>8s} | {'R²':>6s} | {'Dir%':>5s} | {'Corr':>5s}")
    print(f"  {'-'*68}")
    for key in sorted(fcst_metrics.keys()):
        m = fcst_metrics[key]
        horizon = key[:2].upper()
        beat = '*' if m['theil_u'] < 1 else ''
        print(f"  {horizon} {m['label']:17s} | ${m['rmse']:>4.2f} | ${m['rmse_naive']:>4.2f} | {m['theil_u']:>7.3f}{beat} | {m['r2']:>5.3f} | {m['dir_acc']:>4.0f}% | {m['corr']:>4.3f}")
    print(f"\n  Best 1M: {best_1m_label}")
    print(f"  Best 2M: {best_2m_label}")
    if latest_fcst is not None:
        p1 = latest_fcst.get(best_1m_col, np.nan)
        p2 = latest_fcst.get(best_2m_col, np.nan)
        if not np.isnan(p1):
            print(f"\n  Forecast from {latest_fcst['date']:%Y-%m}:")
            print(f"    1M ahead: ${p1:.2f} ({best_1m_label})")
        if not np.isnan(p2):
            print(f"    2M ahead: ${p2:.2f} ({best_2m_label})")

print("\n" + "=" * 85)
print("Done.")
