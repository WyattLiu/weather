#!/usr/bin/env python3
"""
NG Factor Backtest: Quantify Each Signal
Backtests 4 fundamental signals (rig count momentum, export tightening,
storage deviation, S/D balance) for NG forward return predictive power.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import subprocess
import tempfile
import os
from io import BytesIO
from datetime import datetime
import yfinance as yf
from scipy.stats import spearmanr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

print("NG Factor Backtest")
print("=" * 65)

# ============================================
# Data Fetching (copied from ng_production_model.py)
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
        print(f"    Combined: {len(combined)} weeks ({combined['date'].min():%Y-%m-%d} to {combined['date'].max():%Y-%m-%d})")
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
        print(f"    {len(result)} records ({result['date'].min():%Y-%m-%d} to {result['date'].max():%Y-%m-%d})")
        return result
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame(columns=['date', 'storage_bcf'])


def mmcf_to_bcfd(df):
    """Convert MMcf/month to Bcf/d."""
    df = df.copy()
    df['bcfd'] = df['value'] / df['date'].dt.days_in_month / 1000
    return df


# ============================================
# Fetch all data
# ============================================
print("\n--- Monthly EIA Data ---")
production = fetch_monthly(EIA_BASE + 'N9070US2m.xls', 'Dry Gas Production')
consumption = fetch_monthly(EIA_BASE + 'N9140US2m.xls', 'Total Consumption')
lng_exports = fetch_monthly(EIA_BASE + 'N9133US2m.xls', 'LNG Exports')
pipeline_exports = fetch_monthly(EIA_BASE + 'N9132US2m.xls', 'Pipeline Exports (Mexico)')
lng_imports = fetch_monthly(EIA_BASE + 'N9103US2m.xls', 'LNG Imports')
pipeline_imports = fetch_monthly(EIA_BASE + 'N9102US2m.xls', 'Pipeline Imports (Canada)')

print("\n--- Weekly Data ---")
rig_count = fetch_rig_count()
storage = fetch_storage()

print("\n--- NG Price ---")
ng = yf.Ticker("NG=F")
ng_daily = ng.history(period="max", interval="1d")
if ng_daily.index.tz is not None:
    ng_daily.index = ng_daily.index.tz_localize(None)
ng_daily = ng_daily[['Close']].rename(columns={'Close': 'ng_price'})
ng_daily.index.name = 'date'
ng_daily = ng_daily.reset_index()
print(f"  NG=F: {len(ng_daily)} daily bars")

# ============================================
# Convert to Bcf/d and build monthly balance
# ============================================
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

# ============================================
# NG price at monthly frequency (end-of-month)
# ============================================
ng_daily['month'] = ng_daily['date'].dt.to_period('M')
ng_monthly = ng_daily.groupby('month')['ng_price'].last().reset_index()
ng_monthly['date'] = ng_monthly['month'].dt.to_timestamp()
ng_monthly = ng_monthly[['date', 'ng_price']].sort_values('date')

# Forward returns (raw)
for n in [1, 3, 6]:
    ng_monthly[f'fwd_{n}m_raw'] = ng_monthly['ng_price'].shift(-n) / ng_monthly['ng_price'] - 1

# Deseasonalize prices using a seasonal index built from NG price history
# NG has strong seasonal pattern: expensive in winter (heating), cheap in shoulder.
# $3 is cheap in January but normal in March. A seasonal index normalizes this so
# factor analysis measures genuine predictive power, not seasonal correlation.
ng_monthly['cal_month'] = ng_monthly['date'].dt.month

# Build seasonal index: rolling 10-year median price for each calendar month
# Use expanding window per month so we only use past data (no lookahead)
ng_monthly['seasonal_idx'] = np.nan
for m in range(1, 13):
    mask = ng_monthly['cal_month'] == m
    # Expanding median within each calendar month (uses only data up to that point)
    ng_monthly.loc[mask, 'seasonal_idx'] = (
        ng_monthly.loc[mask, 'ng_price']
        .expanding(min_periods=3)
        .median()
        .shift(1)  # lag by 1 year-occurrence to avoid lookahead
    )

# Deseasonalized price = raw price / seasonal index
ng_monthly['ng_deseas'] = ng_monthly['ng_price'] / ng_monthly['seasonal_idx']

# Forward returns on deseasonalized price → excess returns vs seasonal expectation
for n in [1, 3, 6]:
    ng_monthly[f'fwd_{n}m'] = ng_monthly['ng_deseas'].shift(-n) / ng_monthly['ng_deseas'] - 1

# Print seasonal index for reference
seas_idx = ng_monthly.dropna(subset=['seasonal_idx']).groupby('cal_month')['seasonal_idx'].last()
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
seas_str = ', '.join(f'{month_names[m]}=${v:.2f}' for m, v in seas_idx.items())
print(f"  Seasonal index (latest): {seas_str}")
print(f"  Forward returns deseasonalized via price-level seasonal index")

# ============================================
# Factor 1: Rig Count Momentum (3mo/6mo MA ratio)
# ============================================
print("\n--- Building Factors ---")

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

    # Compute 5yr seasonal avg for each week-of-year, rolling
    stor_monthly_list = []
    for _, row in stor.iterrows():
        yr, wk = row['year'], row['week']
        hist = stor[(stor['year'] >= yr - 5) & (stor['year'] < yr) & (stor['week'] == wk)]
        if len(hist) >= 3:
            avg_5yr = hist['storage_bcf'].mean()
            dev = (row['storage_bcf'] - avg_5yr) / avg_5yr * 100
            stor_monthly_list.append({'date': row['date'], 'storage_dev': dev})

    stor_dev = pd.DataFrame(stor_monthly_list)
    # Resample to monthly (last observation of each month)
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
# Factor 4: S/D Balance (already computed)
# ============================================
sd_balance = monthly[['date', 'balance']].dropna().copy()
print(f"  S/D Balance: {len(sd_balance)} months")

# ============================================
# Merge all factors + price data
# ============================================
print("\n--- Merging Factors ---")
cutoff = pd.Timestamp('2015-01-01')

master = ng_monthly[['date', 'ng_price', 'fwd_1m', 'fwd_3m', 'fwd_6m',
                     'fwd_1m_raw', 'fwd_3m_raw', 'fwd_6m_raw']].copy()
master = master[master['date'] >= cutoff]

for fdf, col in [
    (rig_factor, 'rig_momentum'),
    (exp_tight, 'export_tightening'),
    (stor_dev, 'storage_dev'),
    (sd_balance, 'balance'),
]:
    if not fdf.empty:
        master = master.merge(fdf, on='date', how='left')

master = master.sort_values('date').reset_index(drop=True)

factor_cols = ['rig_momentum', 'export_tightening', 'storage_dev', 'balance']
factor_labels = ['Rig Count Momentum', 'Export Tightening', 'Storage Deviation', 'S/D Balance']
# Sign: -1 means "lower = more bullish", +1 means "higher = more bullish"
# Storage deviation is +1: high storage = depressed prices = mean-reversion upside
factor_signs = [-1, 1, 1, -1]

# Drop rows where we have no factors at all
mask = master[factor_cols].notna().any(axis=1) & master['fwd_3m'].notna()
analysis = master[mask].copy()
print(f"  Analysis dataset: {len(analysis)} months ({analysis['date'].min():%Y-%m} to {analysis['date'].max():%Y-%m})")

# ============================================
# Analysis functions
# ============================================

def quintile_returns(df, factor_col, return_cols):
    """Compute mean forward returns by quintile of a factor."""
    valid = df.dropna(subset=[factor_col]).copy()
    if len(valid) < 10:
        return None
    try:
        valid['quintile'] = pd.qcut(valid[factor_col], 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        # Not enough unique values for 5 bins
        valid['quintile'] = pd.qcut(valid[factor_col].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    result = valid.groupby('quintile', observed=True)[return_cols].mean() * 100
    return result


def compute_ic(df, factor_col, return_col):
    """Compute Information Coefficient (Spearman rank correlation) statistics."""
    valid = df.dropna(subset=[factor_col, return_col]).copy()
    if len(valid) < 12:
        return {'mean_ic': np.nan, 'ic_std': np.nan, 'ic_tstat': np.nan, 'pct_positive': np.nan}

    # Rolling 24-month IC
    ic_series = []
    dates = []
    for i in range(23, len(valid)):
        window = valid.iloc[i-23:i+1]
        if len(window) >= 12:
            corr, _ = spearmanr(window[factor_col], window[return_col])
            ic_series.append(corr)
            dates.append(valid.iloc[i]['date'])

    ic_arr = np.array(ic_series)
    ic_arr = ic_arr[~np.isnan(ic_arr)]

    if len(ic_arr) == 0:
        return {'mean_ic': np.nan, 'ic_std': np.nan, 'ic_tstat': np.nan,
                'pct_positive': np.nan, 'ic_dates': [], 'ic_values': []}

    mean_ic = np.mean(ic_arr)
    ic_std = np.std(ic_arr)
    ic_tstat = mean_ic / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
    pct_pos = np.mean(ic_arr > 0) * 100

    return {
        'mean_ic': mean_ic, 'ic_std': ic_std, 'ic_tstat': ic_tstat,
        'pct_positive': pct_pos,
        'ic_dates': dates[:len(ic_arr)], 'ic_values': ic_arr.tolist(),
    }


def hit_rate(df, factor_col, return_col, bullish_sign):
    """Hit rate: when factor in most bullish quintile, % positive returns."""
    valid = df.dropna(subset=[factor_col, return_col]).copy()
    if len(valid) < 10:
        return np.nan, np.nan
    try:
        valid['quintile'] = pd.qcut(valid[factor_col], 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        valid['quintile'] = pd.qcut(valid[factor_col].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

    # Most bullish quintile: Q1 if lower=bullish (sign=-1), Q5 if higher=bullish (sign=1)
    bullish_q = 1 if bullish_sign == -1 else 5
    bullish_subset = valid[valid['quintile'] == bullish_q]
    if len(bullish_subset) == 0:
        return np.nan, np.nan

    hr = (bullish_subset[return_col] > 0).mean() * 100
    base_rate = (valid[return_col] > 0).mean() * 100
    return hr, base_rate


def q1_q5_spread(df, factor_col, return_col):
    """Q1-Q5 spread and t-test."""
    valid = df.dropna(subset=[factor_col, return_col]).copy()
    if len(valid) < 10:
        return np.nan, np.nan
    try:
        valid['quintile'] = pd.qcut(valid[factor_col], 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        valid['quintile'] = pd.qcut(valid[factor_col].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

    q1 = valid[valid['quintile'] == 1][return_col]
    q5 = valid[valid['quintile'] == 5][return_col]
    if len(q1) < 3 or len(q5) < 3:
        return np.nan, np.nan

    spread = (q5.mean() - q1.mean()) * 100  # Q5 - Q1
    _, p_val = ttest_ind(q5, q1)
    t_stat = (q5.mean() - q1.mean()) / np.sqrt(q5.var()/len(q5) + q1.var()/len(q1)) if q5.var() + q1.var() > 0 else 0
    return spread, t_stat


# ============================================
# Run analysis for each factor
# ============================================
print("\n--- Running Factor Analysis ---")

results = {}
quintile_data = {}
ic_data = {}

for fcol, flabel, fsign in zip(factor_cols, factor_labels, factor_signs):
    print(f"\n  {flabel}:")

    # Quintile returns
    qr = quintile_returns(analysis, fcol, ['fwd_1m', 'fwd_3m', 'fwd_6m'])
    quintile_data[flabel] = qr
    if qr is not None:
        print(f"    Quintile returns (3m): Q1={qr.loc[1, 'fwd_3m']:.1f}% Q5={qr.loc[5, 'fwd_3m']:.1f}%")

    # IC
    ic = compute_ic(analysis, fcol, 'fwd_3m')
    ic_data[flabel] = ic
    print(f"    Mean IC: {ic['mean_ic']:.3f}, t-stat: {ic['ic_tstat']:.2f}")

    # Hit rate
    hr, br = hit_rate(analysis, fcol, 'fwd_3m', fsign)
    print(f"    Hit rate (bullish Q): {hr:.1f}%, base rate: {br:.1f}%")

    # Spread
    spread, t = q1_q5_spread(analysis, fcol, 'fwd_3m')
    # Adjust spread sign so positive = bullish direction works
    adj_spread = spread * fsign  # flip if lower=bullish
    print(f"    Q1-Q5 spread (3m): {adj_spread:.1f}%, t={t:.2f}")

    results[flabel] = {
        'mean_ic': ic['mean_ic'],
        'ic_tstat': ic['ic_tstat'],
        'hit_rate': hr,
        'base_rate': br,
        'spread_3m': adj_spread,
        'spread_tstat': t,
    }

# ============================================
# Combined Signal
# ============================================
print("\n--- Combined Signal ---")

# Z-score normalize each factor (rolling 60-month window)
for fcol, fsign in zip(factor_cols, factor_signs):
    valid_mask = analysis[fcol].notna()
    rolling_mean = analysis[fcol].rolling(60, min_periods=24).mean()
    rolling_std = analysis[fcol].rolling(60, min_periods=24).std()
    z = (analysis[fcol] - rolling_mean) / rolling_std
    # Sign-align: multiply by -1 where lower = more bullish
    analysis[f'{fcol}_z'] = z * fsign

z_cols = [f'{c}_z' for c in factor_cols]
analysis['composite'] = analysis[z_cols].mean(axis=1)

# Composite analysis
qr_comp = quintile_returns(analysis, 'composite', ['fwd_1m', 'fwd_3m', 'fwd_6m'])
quintile_data['Composite'] = qr_comp
ic_comp = compute_ic(analysis, 'composite', 'fwd_3m')
ic_data['Composite'] = ic_comp
hr_comp, br_comp = hit_rate(analysis, 'composite', 'fwd_3m', 1)  # higher composite = more bullish
spread_comp, t_comp = q1_q5_spread(analysis, 'composite', 'fwd_3m')

results['Composite'] = {
    'mean_ic': ic_comp['mean_ic'],
    'ic_tstat': ic_comp['ic_tstat'],
    'hit_rate': hr_comp,
    'base_rate': br_comp,
    'spread_3m': spread_comp,
    'spread_tstat': t_comp,
}

if qr_comp is not None:
    print(f"  Quintile returns (3m): Q1={qr_comp.loc[1, 'fwd_3m']:.1f}% Q5={qr_comp.loc[5, 'fwd_3m']:.1f}%")
print(f"  Mean IC: {ic_comp['mean_ic']:.3f}, t-stat: {ic_comp['ic_tstat']:.2f}")
print(f"  Hit rate: {hr_comp:.1f}%, Spread: {spread_comp:.1f}%")

# Cumulative strategy return: long when composite > 0, short when < 0
# Use RAW returns for equity curve (you trade real prices, not deseasonalized)
strat = analysis.dropna(subset=['composite', 'fwd_1m_raw']).copy()
strat['signal'] = np.where(strat['composite'] > 0, 1, -1)
strat['strat_return'] = strat['signal'] * strat['fwd_1m_raw']
strat['cum_strat'] = (1 + strat['strat_return']).cumprod()
strat['cum_bh'] = (1 + strat['fwd_1m_raw']).cumprod()

total_ret = strat['cum_strat'].iloc[-1] - 1 if len(strat) > 0 else 0
ann_ret = (1 + total_ret) ** (12 / max(len(strat), 1)) - 1
ann_vol = strat['strat_return'].std() * np.sqrt(12)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

# ============================================
# Charts: 3x2
# ============================================
print("\n--- Creating Charts ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 2, figsize=(22, 18))

q_colors = ['#C62828', '#EF5350', '#BDBDBD', '#66BB6A', '#2E7D32']

# ============================================
# Top-Left: Quintile Returns by Factor (3m)
# ============================================
ax = axes[0, 0]
factor_keys = [l for l in factor_labels if quintile_data.get(l) is not None]
n_factors = len(factor_keys)
if n_factors > 0:
    x_positions = np.arange(n_factors)
    bar_width = 0.15
    for q_idx in range(5):
        offsets = x_positions + (q_idx - 2) * bar_width
        vals = []
        for fl in factor_keys:
            qr = quintile_data[fl]
            vals.append(qr.loc[q_idx + 1, 'fwd_3m'] if qr is not None else 0)
        ax.bar(offsets, vals, bar_width, color=q_colors[q_idx],
               label=f'Q{q_idx+1}', edgecolor='white', linewidth=0.5)

    # Annotate Q1-Q5 spread for each factor
    for i, fl in enumerate(factor_keys):
        qr = quintile_data[fl]
        if qr is not None:
            fsign = factor_signs[factor_labels.index(fl)]
            spread_val = results[fl]['spread_3m']
            ax.text(i, ax.get_ylim()[1] * 0.85, f'Spread: {spread_val:+.1f}%',
                    ha='center', fontsize=7.5, fontweight='bold',
                    color='#2E7D32' if spread_val > 0 else '#C62828')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([l.replace(' ', '\n') for l in factor_keys], fontsize=8.5)
    ax.set_ylabel('Mean 3-Month Forward Return (%)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, ncol=5, loc='upper left', title='Quintile')
    ax.grid(axis='y', alpha=0.3)

ax.set_title('Quintile Returns by Factor (3-Month Forward)', fontsize=13, fontweight='bold')

# ============================================
# Top-Right: Rolling IC Time Series
# ============================================
ax = axes[0, 1]
ic_colors = ['#1565C0', '#E65100', '#2E7D32', '#7B1FA2']
for i, (fl, color) in enumerate(zip(factor_labels, ic_colors)):
    ic = ic_data.get(fl, {})
    dates = ic.get('ic_dates', [])
    values = ic.get('ic_values', [])
    if len(dates) > 0 and len(values) > 0:
        min_len = min(len(dates), len(values))
        ax.plot(dates[:min_len], values[:min_len], color=color, linewidth=1.3, alpha=0.85,
                label=f'{fl} (IC={ic["mean_ic"]:.3f})')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.axhline(y=0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axhline(y=-0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.fill_between(ax.get_xlim(), 0.1, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.1 else 0.3,
                alpha=0.05, color='green')
ax.fill_between(ax.get_xlim(), ax.get_ylim()[0] if ax.get_ylim()[0] < -0.1 else -0.3, -0.1,
                alpha=0.05, color='red')
ax.set_ylabel('Spearman IC (rolling 24m)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.set_title('Rolling Information Coefficient (vs 3m Forward Return)', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Middle-Left: Combined Signal vs NG Price
# ============================================
ax = axes[1, 0]
comp_valid = analysis.dropna(subset=['composite']).copy()
if len(comp_valid) > 0:
    bull_mask = comp_valid['composite'] > 1
    bear_mask = comp_valid['composite'] < -1
    neutral_mask = ~bull_mask & ~bear_mask

    bar_colors = np.where(bull_mask, '#2E7D32', np.where(bear_mask, '#C62828', '#90CAF9'))
    ax.bar(comp_valid['date'], comp_valid['composite'], width=25, color=bar_colors, alpha=0.7)

    ax2 = ax.twinx()
    price_valid = analysis.dropna(subset=['ng_price'])
    ax2.plot(price_valid['date'], price_valid['ng_price'], color='#757575',
             linewidth=1.8, alpha=0.7, label='NG=F')
    ax2.set_ylabel('NG Price ($/MMBtu)', color='#757575', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#757575')

    ax.axhline(y=1, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=-1, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)

ax.set_ylabel('Composite Z-Score', fontsize=10, fontweight='bold')
ax.set_title('Combined Fundamental Signal vs NG Price', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Middle-Right: Cumulative Return of Composite Strategy
# ============================================
ax = axes[1, 1]
if len(strat) > 0:
    ax.plot(strat['date'], strat['cum_strat'], color='#1565C0', linewidth=2.2,
            label=f'Composite L/S (Sharpe: {sharpe:.2f})')
    ax.plot(strat['date'], strat['cum_bh'], color='#757575', linewidth=1.5,
            linestyle='--', alpha=0.7, label='Buy & Hold NG')
    ax.axhline(y=1, color='black', linewidth=0.5)

    # Annotate total returns
    ax.text(0.02, 0.95,
            f'Strategy: {(strat["cum_strat"].iloc[-1]-1)*100:+.0f}%\n'
            f'Buy & Hold: {(strat["cum_bh"].iloc[-1]-1)*100:+.0f}%\n'
            f'Sharpe: {sharpe:.2f}',
            transform=ax.transAxes, fontsize=10, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax.set_ylabel('Cumulative Return', fontsize=10, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_title('Composite Strategy: Long Bullish / Short Bearish', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Bottom-Left: Factor Correlation Matrix
# ============================================
ax = axes[2, 0]
corr_data = analysis[factor_cols].dropna()
if len(corr_data) > 10:
    corr_matrix = corr_data.corr(method='spearman')
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    short_labels = ['Rig Mom.', 'Export\nTight.', 'Storage\nDev.', 'S/D\nBalance']
    ax.set_xticks(range(4))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(short_labels, fontsize=9)

    for i in range(4):
        for j in range(4):
            val = corr_matrix.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

ax.set_title('Factor Correlation Matrix (Spearman)', fontsize=13, fontweight='bold')

# ============================================
# Bottom-Right: Summary Stats Table
# ============================================
ax = axes[2, 1]
ax.axis('off')

table_rows = []
row_labels = []
for key in factor_labels + ['Composite']:
    r = results.get(key, {})
    table_rows.append([
        f'{r.get("mean_ic", np.nan):.3f}',
        f'{r.get("ic_tstat", np.nan):.2f}',
        f'{r.get("hit_rate", np.nan):.1f}%',
        f'{r.get("spread_3m", np.nan):+.1f}%',
        f'{r.get("spread_tstat", np.nan):.2f}',
    ])
    row_labels.append(key)

col_labels = ['Mean IC', 'IC t-stat', 'Hit Rate', 'Q1-Q5 Spread\n(3m)', 'Spread\nt-stat']

table = ax.table(cellText=table_rows, rowLabels=row_labels, colLabels=col_labels,
                 cellLoc='center', rowLoc='right', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

# Style header row
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#1565C0')
    cell.set_text_props(color='white', fontweight='bold')

# Style composite row (last row)
for j in range(-1, len(col_labels)):
    cell = table[len(row_labels), j]
    cell.set_facecolor('#E3F2FD')
    cell.set_text_props(fontweight='bold')

# Highlight significant t-stats
for i, key in enumerate(row_labels):
    r = results.get(key, {})
    tstat = abs(r.get('ic_tstat', 0))
    if tstat > 2:
        table[i + 1, 1].set_facecolor('#C8E6C9')
    spread_t = abs(r.get('spread_tstat', 0))
    if spread_t > 2:
        table[i + 1, 4].set_facecolor('#C8E6C9')

ax.set_title('Factor Backtest Summary Statistics', fontsize=13, fontweight='bold', pad=20)

# ============================================
# Save
# ============================================
period_start = analysis['date'].min()
period_end = analysis['date'].max()

fig.suptitle(f'NG Factor Backtest: Signal Quantification ({period_start:%Y-%m} to {period_end:%Y-%m})',
             fontsize=16, fontweight='bold', y=1.003)
plt.tight_layout()
plt.savefig('/home/wyatt/weather/ng_backtest_factors.png', dpi=150, bbox_inches='tight')
print("\nChart saved: ng_backtest_factors.png")

# ============================================
# Console Summary
# ============================================
print("\n" + "=" * 95)
print(f"=== NG FACTOR BACKTEST ===")
print(f"Period: {period_start:%Y-%m} to {period_end:%Y-%m} ({len(analysis)} months)")
print("=" * 95)

header = f"{'Factor':<24s} | {'Mean IC':>7s} | {'IC t-stat':>9s} | {'Hit Rate':>8s} | {'Q1-Q5 Spread (3m)':>18s} | {'Spread t-stat':>13s}"
print(header)
print("-" * len(header))

for key in factor_labels:
    r = results.get(key, {})
    print(f"{key:<24s} | {r.get('mean_ic', np.nan):>7.3f} | {r.get('ic_tstat', np.nan):>9.2f} | "
          f"{r.get('hit_rate', np.nan):>7.1f}% | {r.get('spread_3m', np.nan):>+17.1f}% | "
          f"{r.get('spread_tstat', np.nan):>13.2f}")

print("-" * len(header))
r = results.get('Composite', {})
print(f"{'COMPOSITE':<24s} | {r.get('mean_ic', np.nan):>7.3f} | {r.get('ic_tstat', np.nan):>9.2f} | "
      f"{r.get('hit_rate', np.nan):>7.1f}% | {r.get('spread_3m', np.nan):>+17.1f}% | "
      f"{r.get('spread_tstat', np.nan):>13.2f}")

# Find strongest/weakest
factor_tstats = {k: abs(results[k].get('ic_tstat', 0)) for k in factor_labels if k in results}
if factor_tstats:
    strongest = max(factor_tstats, key=factor_tstats.get)
    weakest = min(factor_tstats, key=factor_tstats.get)
    print(f"\nStrongest factor: {strongest} (highest |IC t-stat| = {factor_tstats[strongest]:.2f})")
    print(f"Weakest factor: {weakest} (lowest |IC t-stat| = {factor_tstats[weakest]:.2f})")

# Correlation assessment
if len(corr_data) > 10:
    off_diag = corr_matrix.values[np.triu_indices(4, k=1)]
    avg_corr = np.mean(np.abs(off_diag))
    if avg_corr < 0.3:
        div_label = 'low (good diversification benefit)'
    elif avg_corr < 0.6:
        div_label = 'moderate diversification benefit'
    else:
        div_label = 'high (limited diversification benefit)'
    print(f"Factor correlations: avg |rho| = {avg_corr:.2f} — {div_label}")

print(f"\nComposite strategy: {(strat['cum_strat'].iloc[-1]-1)*100:+.0f}% total, Sharpe {sharpe:.2f}")
print("\n" + "=" * 95)
print("\nDone.")
