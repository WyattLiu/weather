#!/usr/bin/env python3
"""
NG Supply-Demand Production Model
Supply, demand, storage, rig count, and export analysis for natural gas fundamentals.
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
import warnings
warnings.filterwarnings('ignore')

print("NG Supply-Demand Production Model")
print("=" * 65)

# ============================================
# Data Fetching (using curl — EIA blocks Python requests)
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
        # Use cached file if fresh (< 24h old), otherwise re-download
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

        # Find Total Lower 48 column
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
power = fetch_monthly(EIA_BASE + 'N3045US2m.xls', 'Power Sector')
industrial = fetch_monthly(EIA_BASE + 'N3035US2m.xls', 'Industrial')
residential = fetch_monthly(EIA_BASE + 'N3010US2m.xls', 'Residential')
lng_exports = fetch_monthly(EIA_BASE + 'N9133US2m.xls', 'LNG Exports')
pipeline_exports = fetch_monthly(EIA_BASE + 'N9132US2m.xls', 'Pipeline Exports (Mexico)')
lng_imports = fetch_monthly(EIA_BASE + 'N9103US2m.xls', 'LNG Imports')
pipeline_imports = fetch_monthly(EIA_BASE + 'N9102US2m.xls', 'Pipeline Imports (Canada)')

print("\n--- Weekly Data ---")
rig_count = fetch_rig_count()
storage = fetch_storage()

print("\n--- NG Price ---")
ng = yf.Ticker("NG=F")
ng_daily = ng.history(period="5y", interval="1d")
if ng_daily.index.tz is not None:
    ng_daily.index = ng_daily.index.tz_localize(None)
ng_daily = ng_daily[['Close']].rename(columns={'Close': 'ng_price'})
ng_daily.index.name = 'date'
ng_daily = ng_daily.reset_index()
print(f"  NG=F: {len(ng_daily)} daily bars")

# ============================================
# Convert to Bcf/d
# ============================================
cutoff = pd.Timestamp('2021-01-01')

datasets = {
    'production': production, 'consumption': consumption,
    'power': power, 'industrial': industrial, 'residential': residential,
    'lng_exports': lng_exports, 'pipeline_exports': pipeline_exports,
    'lng_imports': lng_imports, 'pipeline_imports': pipeline_imports,
}

for name in datasets:
    df = datasets[name]
    if not df.empty:
        datasets[name] = mmcf_to_bcfd(df)

prod = datasets['production']
cons = datasets['consumption']
pwr = datasets['power']
ind = datasets['industrial']
res = datasets['residential']
lng_exp = datasets['lng_exports']
pipe_exp = datasets['pipeline_exports']
lng_imp = datasets['lng_imports']
pipe_imp = datasets['pipeline_imports']

# ============================================
# Build merged monthly balance
# ============================================
print("\n--- Building S/D Balance ---")

monthly = prod[['date', 'bcfd']].rename(columns={'bcfd': 'prod'}).copy()
for col_name, df in [
    ('cons', cons), ('pwr', pwr), ('ind', ind), ('res', res),
    ('lng_exp', lng_exp), ('pipe_exp', pipe_exp),
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

monthly['prod_yoy'] = monthly['prod'].pct_change(12) * 100
monthly['cons_yoy'] = monthly['cons'].pct_change(12) * 100
monthly['lng_exp_yoy'] = monthly['lng_exp'].pct_change(12) * 100
monthly['pipe_exp_yoy'] = monthly['pipe_exp'].pct_change(12) * 100

m5 = monthly[monthly['date'] >= cutoff].copy()

ng_monthly = ng_daily.copy()
ng_monthly['date'] = ng_monthly['date'].dt.to_period('M').dt.to_timestamp()
ng_monthly = ng_monthly.groupby('date')['ng_price'].mean().reset_index()
m5 = m5.merge(ng_monthly, on='date', how='left')

print(f"  Monthly balance: {len(m5)} months ({m5['date'].min():%Y-%m} to {m5['date'].max():%Y-%m})")

# ============================================
# Rig count processing
# ============================================
rig_5y = rig_count[rig_count['date'] >= cutoff].copy() if not rig_count.empty else pd.DataFrame()

if not rig_count.empty:
    rig_monthly = rig_count.copy()
    rig_monthly['month'] = rig_monthly['date'].dt.to_period('M').dt.to_timestamp()
    rig_monthly = rig_monthly.groupby('month')['value'].mean().reset_index()
    rig_monthly.columns = ['date', 'rig_count']
else:
    rig_monthly = pd.DataFrame(columns=['date', 'rig_count'])

# ============================================
# Storage processing
# ============================================
current_year = datetime.now().year

if not storage.empty:
    storage['year'] = storage['date'].dt.year
    storage['day_of_year'] = storage['date'].dt.dayofyear

    cur_stor = storage[storage['year'] == current_year].copy()
    last_stor = storage[storage['year'] == current_year - 1].copy()

    five_yr_stor = storage[(storage['year'] >= current_year - 5) &
                           (storage['year'] <= current_year - 1)].copy()

    five_yr_stor['week_bin'] = (five_yr_stor['day_of_year'] // 7)
    week_stats = five_yr_stor.groupby('week_bin')['storage_bcf'].agg(
        ['min', 'max', 'mean']).reset_index()
    week_stats['plot_date'] = week_stats['week_bin'].apply(
        lambda w: pd.Timestamp(year=current_year, month=1, day=1) + pd.Timedelta(days=w * 7))

    if len(cur_stor) > 0:
        latest_stor = cur_stor.iloc[-1]
        latest_week_bin = latest_stor['day_of_year'] // 7
        matching = week_stats[week_stats['week_bin'] == latest_week_bin]
        stor_vs_avg = latest_stor['storage_bcf'] - matching.iloc[0]['mean'] if len(matching) > 0 else np.nan
    else:
        latest_stor = None
        stor_vs_avg = np.nan
else:
    cur_stor = last_stor = pd.DataFrame()
    week_stats = pd.DataFrame()
    latest_stor = None
    stor_vs_avg = np.nan

# ============================================
# Create Charts: 3x2
# ============================================
print("\n--- Creating Charts ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 2, figsize=(22, 16))

C_PROD = '#1565C0'
C_CONS = '#B71C1C'
C_POWER = '#FF8F00'
C_IND = '#00897B'
C_RES = '#7B1FA2'
C_LNG_EXP = '#E65100'
C_PIPE_EXP = '#2E7D32'
C_LNG_IMP = '#42A5F5'
C_SURPLUS = '#4CAF50'
C_DEFICIT = '#E53935'
C_NG = '#757575'
C_RIG = '#1565C0'

# ============================================
# Top-Left: Supply Stack
# ============================================
ax = axes[0, 0]
if len(m5) > 0:
    ax.plot(m5['date'], m5['prod'], color=C_PROD, linewidth=2.5, label='Dry Gas Production')
    total_imp = m5['lng_imp'] + m5['pipe_imp']
    ax.fill_between(m5['date'], 0, total_imp, alpha=0.4, color=C_LNG_IMP, label='Imports (LNG + Canada)')
    ax.plot(m5['date'], m5['total_supply'], '--', color=C_PROD, linewidth=1.2, alpha=0.6, label='Total Supply')

    last = m5.dropna(subset=['prod']).iloc[-1]
    ax.annotate(f'{last["prod"]:.1f} Bcf/d', xy=(last['date'], last['prod']),
                xytext=(10, 8), textcoords='offset points', fontsize=10, fontweight='bold',
                color=C_PROD, arrowprops=dict(arrowstyle='->', color=C_PROD, lw=1))

    last_yoy = m5.dropna(subset=['prod_yoy'])
    if len(last_yoy) > 0:
        yoy_val = last_yoy.iloc[-1]['prod_yoy']
        ax.text(0.02, 0.95, f'YoY: {yoy_val:+.1f}%',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=C_SURPLUS if yoy_val > 0 else C_DEFICIT,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax.set_ylabel('Bcf/d', fontsize=11, fontweight='bold')
ax.set_title('Supply: Production + Imports', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Top-Right: Demand Stack
# ============================================
ax = axes[0, 1]
if len(m5) > 0:
    ax.plot(m5['date'], m5['cons'], color=C_CONS, linewidth=2.5, label='Total Consumption')
    for col, color, label in [('pwr', C_POWER, 'Power Sector'),
                               ('ind', C_IND, 'Industrial'),
                               ('res', C_RES, 'Residential')]:
        if col in m5.columns:
            ax.plot(m5['date'], m5[col], color=color, linewidth=1.2, alpha=0.7, label=label)

    ax.plot(m5['date'], m5['lng_exp'], color=C_LNG_EXP, linewidth=2.2, label='LNG Exports')
    ax.plot(m5['date'], m5['pipe_exp'], color=C_PIPE_EXP, linewidth=1.8, label='Mexico Pipeline')

    last_cons = m5.dropna(subset=['cons']).iloc[-1]
    ax.annotate(f'{last_cons["cons"]:.1f}', xy=(last_cons['date'], last_cons['cons']),
                xytext=(8, 0), textcoords='offset points', fontsize=9, fontweight='bold', color=C_CONS)
    last_lng = m5.dropna(subset=['lng_exp']).iloc[-1]
    ax.annotate(f'{last_lng["lng_exp"]:.1f}', xy=(last_lng['date'], last_lng['lng_exp']),
                xytext=(8, 0), textcoords='offset points', fontsize=9, fontweight='bold', color=C_LNG_EXP)

ax.set_ylabel('Bcf/d', fontsize=11, fontweight='bold')
ax.set_title('Demand: Consumption + Exports', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Middle-Left: S/D Balance
# ============================================
ax = axes[1, 0]
if len(m5) > 0:
    balance_valid = m5.dropna(subset=['balance'])
    colors_bar = [C_SURPLUS if b >= 0 else C_DEFICIT for b in balance_valid['balance']]
    ax.bar(balance_valid['date'], balance_valid['balance'], width=25, color=colors_bar, alpha=0.7)

    ax2 = ax.twinx()
    price_valid = m5.dropna(subset=['ng_price'])
    ax2.plot(price_valid['date'], price_valid['ng_price'], '--', color=C_NG, linewidth=1.5, alpha=0.7, label='NG=F')
    ax2.set_ylabel('NG Price ($/MMBtu)', color=C_NG, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=C_NG)

    if len(balance_valid) > 0:
        last_bal = balance_valid.iloc[-1]
        label_text = 'surplus' if last_bal['balance'] >= 0 else 'deficit'
        ax.annotate(f'{last_bal["balance"]:+.1f} Bcf/d ({label_text})',
                    xy=(last_bal['date'], last_bal['balance']),
                    xytext=(0, 15 if last_bal['balance'] >= 0 else -20),
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    color=C_SURPLUS if last_bal['balance'] >= 0 else C_DEFICIT, ha='center')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Bcf/d (surplus/deficit)', fontsize=11, fontweight='bold')
ax.set_title('Supply-Demand Balance', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Middle-Right: Rig Count vs Production
# ============================================
ax = axes[1, 1]
if not rig_5y.empty:
    ax.plot(rig_5y['date'], rig_5y['value'], color=C_RIG, linewidth=1.5, alpha=0.8, label='Gas Rig Count')
    ax.set_ylabel('Gas Rig Count', color=C_RIG, fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=C_RIG)

    latest_rig_val = rig_5y.iloc[-1]
    ax.annotate(f'{latest_rig_val["value"]:.0f}', xy=(latest_rig_val['date'], latest_rig_val['value']),
                xytext=(8, 5), textcoords='offset points', fontsize=10, fontweight='bold', color=C_RIG)

    ax2 = ax.twinx()
    prod_5y = prod[prod['date'] >= cutoff]
    if not prod_5y.empty:
        ax2.plot(prod_5y['date'], prod_5y['bcfd'], color=C_DEFICIT, linewidth=2, alpha=0.8, label='Production (Bcf/d)')
        ax2.set_ylabel('Production (Bcf/d)', color=C_DEFICIT, fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=C_DEFICIT)

        last_p = prod_5y.iloc[-1]
        ax2.annotate(f'{last_p["bcfd"]:.1f}', xy=(last_p['date'], last_p['bcfd']),
                     xytext=(8, -8), textcoords='offset points', fontsize=10, fontweight='bold', color=C_DEFICIT)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    if len(rig_5y) > 26:
        recent_avg = rig_5y.tail(13)['value'].mean()
        six_mo_avg = rig_5y.tail(26)['value'].mean()
        rig_dir = 'RISING' if recent_avg > six_mo_avg else 'FALLING' if recent_avg < six_mo_avg * 0.98 else 'FLAT'
        ax.text(0.98, 0.95, f'Rig trend: {rig_dir}\n~6mo lead on production',
                transform=ax.transAxes, ha='right', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
else:
    ax.text(0.5, 0.5, 'Rig count data unavailable', transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='gray')

ax.set_title('Rig Count vs Production (Leading Indicator)', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Bottom-Left: Export Growth Model
# ============================================
ax = axes[2, 0]
if len(m5) > 0:
    lng_valid = m5.dropna(subset=['lng_exp'])
    pipe_valid = m5.dropna(subset=['pipe_exp'])

    ax.plot(lng_valid['date'], lng_valid['lng_exp'], color=C_LNG_EXP, linewidth=2.5, label='LNG Exports')
    ax.plot(pipe_valid['date'], pipe_valid['pipe_exp'], color=C_PIPE_EXP, linewidth=2, label='Mexico Pipeline')

    both = m5.dropna(subset=['lng_exp', 'pipe_exp'])
    ax.fill_between(both['date'], 0, both['lng_exp'] + both['pipe_exp'],
                    alpha=0.15, color=C_LNG_EXP, label='Total Exports')

    if len(lng_valid) > 0:
        ll = lng_valid.iloc[-1]
        ax.annotate(f'{ll["lng_exp"]:.1f} Bcf/d', xy=(ll['date'], ll['lng_exp']),
                    xytext=(8, 5), textcoords='offset points', fontsize=10,
                    fontweight='bold', color=C_LNG_EXP)
    if len(pipe_valid) > 0:
        lp = pipe_valid.iloc[-1]
        ax.annotate(f'{lp["pipe_exp"]:.1f} Bcf/d', xy=(lp['date'], lp['pipe_exp']),
                    xytext=(8, 5), textcoords='offset points', fontsize=10,
                    fontweight='bold', color=C_PIPE_EXP)

    lng_yoy_valid = m5.dropna(subset=['lng_exp_yoy'])
    if len(lng_yoy_valid) > 0:
        lng_yoy_last = lng_yoy_valid.iloc[-1]['lng_exp_yoy']
        pipe_yoy_valid = m5.dropna(subset=['pipe_exp_yoy'])
        pipe_yoy_last = pipe_yoy_valid.iloc[-1]['pipe_exp_yoy'] if len(pipe_yoy_valid) > 0 else np.nan
        yoy_text = f'LNG YoY: {lng_yoy_last:+.0f}%'
        if not np.isnan(pipe_yoy_last):
            yoy_text += f'\nMexico YoY: {pipe_yoy_last:+.0f}%'
        ax.text(0.02, 0.95, yoy_text, transform=ax.transAxes, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
                va='top')

    # Capacity milestones
    milestones = [
        (pd.Timestamp('2024-12-01'), 'Plaquemines\nPhase 1'),
        (pd.Timestamp('2025-06-01'), 'Corpus Christi\nStage 3'),
        (pd.Timestamp('2025-12-01'), 'Golden Pass\n(expected)'),
    ]
    ymin, ymax = ax.get_ylim()
    for mdate, mlabel in milestones:
        if m5['date'].min() <= mdate <= m5['date'].max() + pd.DateOffset(months=3):
            ax.axvline(x=mdate, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
            ax.text(mdate, ymax * 0.95, mlabel, fontsize=7, color='gray', ha='center',
                    va='top', style='italic')

ax.set_ylabel('Bcf/d', fontsize=11, fontweight='bold')
ax.set_title('Export Growth: LNG + Mexico Pipeline', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Bottom-Right: Storage vs 5-Year Range
# ============================================
ax = axes[2, 1]
if not storage.empty and len(week_stats) > 0:
    ax.fill_between(week_stats['plot_date'], week_stats['min'], week_stats['max'],
                    alpha=0.2, color='gray', label='5yr range')
    ax.plot(week_stats['plot_date'], week_stats['mean'], '--', color='gray',
            linewidth=1.5, label='5yr avg')

    if len(last_stor) > 0:
        last_plot = last_stor.copy()
        last_plot['plot_date'] = last_plot['date'].apply(
            lambda d: d.replace(year=current_year) if not (d.month == 2 and d.day == 29)
            else d.replace(year=current_year, month=2, day=28))
        ax.plot(last_plot['plot_date'], last_plot['storage_bcf'], '--', color='steelblue',
                linewidth=1.5, label=str(current_year - 1))

    if len(cur_stor) > 0:
        ax.plot(cur_stor['date'], cur_stor['storage_bcf'], 'k-', linewidth=2.5,
                label=str(current_year))

        ls = cur_stor.iloc[-1]
        annotation = f'{ls["storage_bcf"]:,.0f} Bcf'
        if not np.isnan(stor_vs_avg):
            above_below = 'above' if stor_vs_avg > 0 else 'below'
            annotation += f'\n{abs(stor_vs_avg):.0f} Bcf {above_below} avg'
        ax.annotate(annotation, xy=(ls['date'], ls['storage_bcf']),
                    xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=1))

ax.set_ylabel('Working Gas (Bcf)', fontsize=11, fontweight='bold')
ax.set_title('Storage vs 5-Year Range', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# ============================================
# Save
# ============================================
fig.suptitle('NG Supply-Demand Production Model', fontsize=16, fontweight='bold', y=1.003)
plt.tight_layout()
plt.savefig('/home/wyatt/weather/ng_production_model.png', dpi=150, bbox_inches='tight')
print("\nChart saved: ng_production_model.png")

# ============================================
# Console Summary
# ============================================
print("\n" + "=" * 65)
print("=== NG SUPPLY-DEMAND MODEL ===")
print("=" * 65)

latest_m = m5.dropna(subset=['prod']).iloc[-1] if len(m5) > 0 else None

if latest_m is not None:
    print(f"\n  Data through: {latest_m['date']:%Y-%m}")
    prod_yoy_val = latest_m.get('prod_yoy', np.nan)
    if not np.isnan(prod_yoy_val):
        print(f"  Production: {latest_m['prod']:.1f} Bcf/d (YoY: {prod_yoy_val:+.1f}%)")
    else:
        print(f"  Production: {latest_m['prod']:.1f} Bcf/d")

lng_latest = m5.dropna(subset=['lng_exp']).iloc[-1] if len(m5) > 0 else None
if lng_latest is not None:
    lng_yoy_v = lng_latest.get('lng_exp_yoy', np.nan)
    yoy_str = f" (YoY: {lng_yoy_v:+.0f}%)" if not np.isnan(lng_yoy_v) else ""
    print(f"  LNG Exports: {lng_latest['lng_exp']:.1f} Bcf/d{yoy_str}")

pipe_latest = m5.dropna(subset=['pipe_exp']).iloc[-1] if len(m5) > 0 else None
if pipe_latest is not None:
    pipe_yoy_v = pipe_latest.get('pipe_exp_yoy', np.nan)
    yoy_str = f" (YoY: {pipe_yoy_v:+.0f}%)" if not np.isnan(pipe_yoy_v) else ""
    print(f"  Mexico Exports: {pipe_latest['pipe_exp']:.1f} Bcf/d{yoy_str}")

cons_latest = m5.dropna(subset=['cons']).iloc[-1] if len(m5) > 0 else None
if cons_latest is not None:
    cons_yoy_v = cons_latest.get('cons_yoy', np.nan)
    yoy_str = f" (YoY: {cons_yoy_v:+.0f}%)" if not np.isnan(cons_yoy_v) else ""
    print(f"  Total Consumption: {cons_latest['cons']:.1f} Bcf/d{yoy_str}")

bal_latest = m5.dropna(subset=['balance']).iloc[-1] if len(m5) > 0 else None
if bal_latest is not None:
    label = 'surplus' if bal_latest['balance'] >= 0 else 'deficit'
    print(f"  S/D Balance: {bal_latest['balance']:+.1f} Bcf/d ({label})")

# Rig count
six_mo = None
if not rig_count.empty:
    latest_rig = rig_count.iloc[-1]
    print(f"\n  Gas Rig Count: {latest_rig['value']:.0f} (as of {latest_rig['date']:%Y-%m-%d})")
    if len(rig_count) > 26:
        six_mo = rig_count.tail(26)['value'].mean()
        direction = 'up' if latest_rig['value'] > six_mo else 'down' if latest_rig['value'] < six_mo * 0.98 else 'flat'
        print(f"  vs 6mo avg: {six_mo:.0f}, direction: {direction}")
        if direction == 'down':
            print("  Implied Production Trend: falling (6-month lag)")
        elif direction == 'up':
            print("  Implied Production Trend: rising (6-month lag)")
        else:
            print("  Implied Production Trend: flat")

# Storage
if latest_stor is not None:
    print(f"\n  Storage: {latest_stor['storage_bcf']:,.0f} Bcf (as of {latest_stor['date']:%Y-%m-%d})")
    if not np.isnan(stor_vs_avg):
        above_below = 'above' if stor_vs_avg > 0 else 'below'
        print(f"  {abs(stor_vs_avg):.0f} Bcf {above_below} 5yr avg")

# ============================================
# Signal Logic
# ============================================
signals = []

if latest_m is not None and not rig_count.empty and len(rig_count) > 26 and six_mo is not None:
    latest_rig = rig_count.iloc[-1]
    rig_dir = 'down' if latest_rig['value'] < six_mo * 0.98 else 'up' if latest_rig['value'] > six_mo else 'flat'
    prod_yoy_s = latest_m.get('prod_yoy', 0)
    if np.isnan(prod_yoy_s):
        prod_yoy_s = 0
    prod_dir = 'up' if prod_yoy_s > 1 else 'down' if prod_yoy_s < -1 else 'flat'
    if rig_dir == 'down' and prod_dir in ('up', 'flat'):
        signals.append(('BULLISH', 'Rigs falling while production flat/rising -> future supply contraction'))
    elif rig_dir == 'up':
        signals.append(('BEARISH', 'Rigs rising -> future supply growth'))

if lng_latest is not None and latest_m is not None:
    lng_g = lng_latest.get('lng_exp_yoy', np.nan)
    prod_g = latest_m.get('prod_yoy', np.nan)
    if not np.isnan(lng_g) and not np.isnan(prod_g) and lng_g > prod_g:
        signals.append(('BULLISH', f'LNG export growth ({lng_g:+.0f}%) > production growth ({prod_g:+.0f}%)'))

if not np.isnan(stor_vs_avg):
    if stor_vs_avg < -50:
        signals.append(('BULLISH', f'Storage {abs(stor_vs_avg):.0f} Bcf below 5yr avg'))
    elif stor_vs_avg > 200:
        signals.append(('BEARISH', f'Storage {stor_vs_avg:.0f} Bcf above 5yr avg'))

if bal_latest is not None and not np.isnan(bal_latest['balance']):
    if bal_latest['balance'] < -1:
        signals.append(('BULLISH', f'Supply deficit of {abs(bal_latest["balance"]):.1f} Bcf/d'))
    elif bal_latest['balance'] > 2:
        signals.append(('BEARISH', f'Supply surplus of {bal_latest["balance"]:.1f} Bcf/d'))

bull = sum(1 for s, _ in signals if s == 'BULLISH')
bear = sum(1 for s, _ in signals if s == 'BEARISH')
if bull > bear:
    overall = 'BULLISH'
elif bear > bull:
    overall = 'BEARISH'
else:
    overall = 'NEUTRAL'

print(f"\n  Signal: {overall}")
for sig, reason in signals:
    print(f"    [{sig}] {reason}")

print("\n" + "=" * 65)
print("\nDone.")
