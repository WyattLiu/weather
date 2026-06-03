#!/usr/bin/env python3
"""
NG CFTC Commitment of Traders (COT) — Comprehensive Analysis
Fetches all disaggregated COT data for NG, plots all meaningful metrics,
and analyzes predictive behavior of positioning vs forward NG returns.
"""
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from matplotlib.patches import Patch

print("NG COT — Comprehensive Analysis")
print("=" * 65)

# ============================================
# Fetch CFTC disaggregated futures-only data
# ============================================
NG_PATTERNS = [
    'NATURAL GAS - NEW YORK MERCANTILE EXCHANGE',
    'NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE',
]

all_ng = []
for year in range(2010, 2027):
    url = f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=20)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, low_memory=False)
        df['Market_and_Exchange_Names'] = df['Market_and_Exchange_Names'].str.strip()
        ng = df[df['Market_and_Exchange_Names'].isin(NG_PATTERNS)].copy()
        if len(ng) > 0:
            all_ng.append(ng)
            print(f"  {year}: {len(ng)} weeks")
    except Exception as e:
        print(f"  {year}: error - {e}")

cot = pd.concat(all_ng, ignore_index=True)

# Parse dates
if 'Report_Date_as_YYYY-MM-DD' in cot.columns:
    cot['date'] = pd.to_datetime(cot['Report_Date_as_YYYY-MM-DD'], errors='coerce')
mask = cot['date'].isna()
if mask.any():
    cot.loc[mask, 'date'] = pd.to_datetime(
        cot.loc[mask, 'As_of_Date_In_Form_YYMMDD'].astype(str), format='%y%m%d', errors='coerce')
cot = cot.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
print(f"\nTotal: {len(cot)} weeks ({cot['date'].min().date()} to {cot['date'].max().date()})")

# ============================================
# Compute ALL meaningful positioning metrics
# ============================================
def safe_float(series):
    return pd.to_numeric(series, errors='coerce').astype(float)

# Managed Money
cot['mm_long'] = safe_float(cot['M_Money_Positions_Long_All'])
cot['mm_short'] = safe_float(cot['M_Money_Positions_Short_All'])
cot['mm_spread'] = safe_float(cot['M_Money_Positions_Spread_All'])
cot['mm_net'] = cot['mm_long'] - cot['mm_short']
cot['mm_gross'] = cot['mm_long'] + cot['mm_short']

# Producer / Merchant
cot['prod_long'] = safe_float(cot['Prod_Merc_Positions_Long_All'])
cot['prod_short'] = safe_float(cot['Prod_Merc_Positions_Short_All'])
cot['prod_net'] = cot['prod_long'] - cot['prod_short']

# Swap Dealers
cot['swap_long'] = safe_float(cot['Swap_Positions_Long_All'])
cot['swap_short'] = safe_float(cot['Swap__Positions_Short_All'])
cot['swap_spread'] = safe_float(cot['Swap__Positions_Spread_All'])
cot['swap_net'] = cot['swap_long'] - cot['swap_short']

# Other Reportable
cot['other_long'] = safe_float(cot['Other_Rept_Positions_Long_All'])
cot['other_short'] = safe_float(cot['Other_Rept_Positions_Short_All'])
cot['other_spread'] = safe_float(cot['Other_Rept_Positions_Spread_All'])
cot['other_net'] = cot['other_long'] - cot['other_short']

# Non-Reportable (small traders below CFTC reporting threshold — NOT purely retail)
cot['nonrept_long'] = safe_float(cot['NonRept_Positions_Long_All'])
cot['nonrept_short'] = safe_float(cot['NonRept_Positions_Short_All'])
cot['nonrept_net'] = cot['nonrept_long'] - cot['nonrept_short']

# Open Interest
cot['oi'] = safe_float(cot['Open_Interest_All'])

# Weekly changes
cot['mm_long_chg'] = safe_float(cot['Change_in_M_Money_Long_All'])
cot['mm_short_chg'] = safe_float(cot['Change_in_M_Money_Short_All'])
cot['mm_net_chg'] = cot['mm_long_chg'] - cot['mm_short_chg']

# Normalized metrics (% of OI)
for col in ['mm_net', 'mm_long', 'mm_short', 'prod_net', 'swap_net', 'other_net', 'nonrept_net']:
    cot[f'{col}_pct_oi'] = cot[col] / cot['oi'] * 100

# Long/Short ratio
cot['mm_ls_ratio'] = cot['mm_long'] / cot['mm_short'].replace(0, np.nan)

# Rolling metrics
cot['mm_net_4w_avg'] = cot['mm_net'].rolling(4, min_periods=2).mean()
cot['mm_net_chg_4w'] = cot['mm_net'].diff(4)  # 4-week change in net
cot['mm_net_chg_13w'] = cot['mm_net'].diff(13)  # 13-week (quarter) change
cot['oi_chg_4w'] = cot['oi'].diff(4)

# Percentile ranks (expanding window)
cot['mm_net_pct_rank'] = cot['mm_net'].expanding().apply(lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() * 100, raw=False)
cot['mm_net_oi_pct_rank'] = cot['mm_net_pct_oi'].expanding().apply(lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() * 100, raw=False)

# Calendar
cot['month'] = cot['date'].dt.month
cot['year'] = cot['date'].dt.year

# ============================================
# Fetch NG=F price and compute forward returns
# ============================================
print("\nFetching NG=F price history...")
ng = yf.Ticker("NG=F")
ng_hist = ng.history(period="max", interval="1d")
if ng_hist.index.tz is not None:
    ng_hist.index = ng_hist.index.tz_localize(None)
print(f"  {len(ng_hist)} daily bars ({ng_hist.index.min().date()} to {ng_hist.index.max().date()})")

# Match COT to NG price + compute forward returns
ng_weekly = ng_hist['Close'].resample('W-TUE').last().dropna()

ng_at_cot, fwd_1w, fwd_2w, fwd_4w, fwd_8w, fwd_13w = [], [], [], [], [], []
for d in cot['date']:
    diffs = abs(ng_weekly.index - d)
    if len(diffs) > 0 and diffs.min() <= pd.Timedelta(days=7):
        idx = diffs.argmin()
        price = ng_weekly.iloc[idx]
        ng_at_cot.append(price)

        # Forward returns (%)
        for offset, store in [(1, fwd_1w), (2, fwd_2w), (4, fwd_4w), (8, fwd_8w), (13, fwd_13w)]:
            if idx + offset < len(ng_weekly):
                ret = (ng_weekly.iloc[idx + offset] / price - 1) * 100
                store.append(ret)
            else:
                store.append(np.nan)
    else:
        ng_at_cot.append(np.nan)
        for store in [fwd_1w, fwd_2w, fwd_4w, fwd_8w, fwd_13w]:
            store.append(np.nan)

cot['ng_price'] = ng_at_cot
cot['fwd_1w'] = fwd_1w
cot['fwd_2w'] = fwd_2w
cot['fwd_4w'] = fwd_4w
cot['fwd_8w'] = fwd_8w
cot['fwd_13w'] = fwd_13w

# ============================================
# PAGE 1: All Positioning Data (4x2)
# ============================================
print("\nCreating charts...")
plt.style.use('seaborn-v0_8-whitegrid')

fig1, axes1 = plt.subplots(4, 2, figsize=(20, 24))
latest = cot.iloc[-1]

# --- 1,0: Managed Money Net vs Price ---
ax = axes1[0, 0]
ax2 = ax.twinx()
ax.bar(cot['date'], cot['mm_net'] / 1000, width=5,
       color=['#2E7D32' if v >= 0 else '#D32F2F' for v in cot['mm_net']], alpha=0.6)
ax2.plot(cot['date'], cot['ng_price'], color='#1565C0', linewidth=1.2, alpha=0.8)
ax.set_ylabel('MM Net (000s)', fontsize=10, fontweight='bold')
ax2.set_ylabel('NG=F', color='#1565C0', fontsize=10)
ax.set_title('Managed Money Net vs NG Price', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Annotate extremes
mm_max_idx = cot['mm_net'].idxmax()
mm_min_idx = cot['mm_net'].idxmin()
ax.annotate(f'Peak: {cot.loc[mm_max_idx, "mm_net"]/1000:+,.0f}K',
            xy=(cot.loc[mm_max_idx, 'date'], cot.loc[mm_max_idx, 'mm_net']/1000),
            fontsize=8, fontweight='bold', color='#2E7D32', ha='center',
            xytext=(0, 8), textcoords='offset points')
ax.annotate(f'Trough: {cot.loc[mm_min_idx, "mm_net"]/1000:+,.0f}K',
            xy=(cot.loc[mm_min_idx, 'date'], cot.loc[mm_min_idx, 'mm_net']/1000),
            fontsize=8, fontweight='bold', color='#D32F2F', ha='center',
            xytext=(0, -12), textcoords='offset points')

# --- 1,1: Producer Net vs Price ---
ax = axes1[0, 1]
ax2 = ax.twinx()
ax.bar(cot['date'], cot['prod_net'] / 1000, width=5, color='#FF8F00', alpha=0.6)
ax2.plot(cot['date'], cot['ng_price'], color='#1565C0', linewidth=1.2, alpha=0.8)
ax.set_ylabel('Producer Net (000s)', fontsize=10, fontweight='bold')
ax2.set_ylabel('NG=F', color='#1565C0', fontsize=10)
ax.set_title('Producer/Merchant Net vs NG Price', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- 2,0: Swap Dealer Net vs Price ---
ax = axes1[1, 0]
ax2 = ax.twinx()
ax.bar(cot['date'], cot['swap_net'] / 1000, width=5, color='#6A0DAD', alpha=0.5)
ax2.plot(cot['date'], cot['ng_price'], color='#1565C0', linewidth=1.2, alpha=0.8)
ax.set_ylabel('Swap Net (000s)', fontsize=10, fontweight='bold')
ax2.set_ylabel('NG=F', color='#1565C0', fontsize=10)
ax.set_title('Swap Dealer Net vs NG Price', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- 2,1: Non-Reportable (Retail) Net vs Price ---
ax = axes1[1, 1]
ax2 = ax.twinx()
ax.bar(cot['date'], cot['nonrept_net'] / 1000, width=5, color='#795548', alpha=0.5)
ax2.plot(cot['date'], cot['ng_price'], color='#1565C0', linewidth=1.2, alpha=0.8)
ax.set_ylabel('Non-Rept Net (000s)', fontsize=10, fontweight='bold')
ax2.set_ylabel('NG=F', color='#1565C0', fontsize=10)
ax.set_title('Non-Reportable (Small Traders) Net vs NG Price', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- 3,0: All Nets as % of OI (stacked area) ---
ax = axes1[2, 0]
net_cols = [('mm_net_pct_oi', '#2E7D32', 'Managed Money'),
            ('prod_net_pct_oi', '#FF8F00', 'Producers'),
            ('swap_net_pct_oi', '#6A0DAD', 'Swap Dealers'),
            ('other_net_pct_oi', '#795548', 'Other Rept'),
            ('nonrept_net_pct_oi', '#9E9E9E', 'Small Traders')]
for col, color, label in net_cols:
    ax.plot(cot['date'], cot[col], color=color, linewidth=1.5, alpha=0.8, label=label)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Net as % of OI', fontsize=10, fontweight='bold')
ax.set_title('All Categories: Net Position as % of Open Interest', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- 3,1: Open Interest + MM Long/Short Ratio ---
ax = axes1[2, 1]
ax2 = ax.twinx()
ax.plot(cot['date'], cot['oi'] / 1000, color='#6A0DAD', linewidth=1.5, alpha=0.8, label='Open Interest')
ax2.plot(cot['date'], cot['mm_ls_ratio'], color='#D32F2F', linewidth=1.5, alpha=0.8, label='MM Long/Short Ratio')
ax2.axhline(y=1.0, color='#D32F2F', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_ylabel('OI (000s)', fontsize=10, fontweight='bold', color='#6A0DAD')
ax.tick_params(axis='y', labelcolor='#6A0DAD')
ax2.set_ylabel('MM L/S Ratio', fontsize=10, fontweight='bold', color='#D32F2F')
ax2.tick_params(axis='y', labelcolor='#D32F2F')
ax.set_title('Open Interest & MM Long/Short Ratio', fontsize=12, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- 4,0: MM Net Weekly Change (momentum) ---
ax = axes1[3, 0]
ax.bar(cot['date'], cot['mm_net_chg'] / 1000, width=5,
       color=['#2E7D32' if v >= 0 else '#D32F2F' for v in cot['mm_net_chg'].fillna(0)], alpha=0.5)
ax.plot(cot['date'], cot['mm_net_chg'].rolling(4).mean() / 1000, color='#1565C0', linewidth=2,
        label='4-week avg')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Weekly Net Change (000s)', fontsize=10, fontweight='bold')
ax.set_xlabel('Date', fontsize=10)
ax.set_title('Managed Money: Weekly Net Change (Flow Momentum)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- 4,1: MM Net Percentile Rank (expanding) vs Price ---
ax = axes1[3, 1]
ax2 = ax.twinx()
ax.fill_between(cot['date'], 50, cot['mm_net_pct_rank'],
                where=cot['mm_net_pct_rank'] >= 50, alpha=0.4, color='#2E7D32')
ax.fill_between(cot['date'], 50, cot['mm_net_pct_rank'],
                where=cot['mm_net_pct_rank'] < 50, alpha=0.4, color='#D32F2F')
ax.plot(cot['date'], cot['mm_net_pct_rank'], color='black', linewidth=0.8, alpha=0.5)
ax2.plot(cot['date'], cot['ng_price'], color='#1565C0', linewidth=1.2, alpha=0.8)
ax.axhline(y=80, color='#D32F2F', linewidth=1, linestyle=':', alpha=0.5, label='Extreme long (80th)')
ax.axhline(y=20, color='#2E7D32', linewidth=1, linestyle=':', alpha=0.5, label='Extreme short (20th)')
ax.axhline(y=50, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax.set_ylabel('MM Net Percentile Rank', fontsize=10, fontweight='bold')
ax2.set_ylabel('NG=F', color='#1565C0', fontsize=10)
ax.set_xlabel('Date', fontsize=10)
ax.set_title('MM Net Positioning Percentile (Expanding) vs Price', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig1.suptitle('NG Commitment of Traders — All Positioning Data', fontsize=16, fontweight='bold', y=1.003)
fig1.tight_layout()
fig1.savefig('/home/wyatt/weather/ng_cot_positions.png', dpi=150, bbox_inches='tight')
print("  Saved: ng_cot_positions.png")

# ============================================
# PAGE 2: Predictive Behavior Analysis (4x2)
# ============================================
fig2, axes2 = plt.subplots(4, 2, figsize=(20, 24))

# --- Quintile analysis: bin MM net into quintiles, show avg forward returns ---
valid = cot.dropna(subset=['mm_net_pct_oi', 'fwd_1w', 'fwd_4w', 'fwd_13w']).copy()
valid['mm_quintile'] = pd.qcut(valid['mm_net_pct_oi'], 5, labels=['Q1\n(Most Short)', 'Q2', 'Q3', 'Q4', 'Q5\n(Most Long)'])

# 1,0: Quintile avg forward returns (multiple horizons)
ax = axes2[0, 0]
quintile_returns = valid.groupby('mm_quintile', observed=True)[['fwd_1w', 'fwd_2w', 'fwd_4w', 'fwd_8w', 'fwd_13w']].mean()
x = np.arange(5)
width = 0.15
colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0', '#0D47A1']
for i, (col, label, c) in enumerate(zip(
    ['fwd_1w', 'fwd_2w', 'fwd_4w', 'fwd_8w', 'fwd_13w'],
    ['1 wk', '2 wk', '4 wk', '8 wk', '13 wk'],
    colors)):
    ax.bar(x + i * width, quintile_returns[col], width, label=label, color=c, edgecolor='white')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(quintile_returns.index, fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Avg Forward Return (%)', fontsize=10, fontweight='bold')
ax.set_title('MM Net Quintile → Avg Forward NG Return', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=3)
ax.grid(axis='y', alpha=0.3)

# 1,1: Quintile hit rate (% of times positive return)
ax = axes2[0, 1]
for i, (col, label, c) in enumerate(zip(
    ['fwd_1w', 'fwd_4w', 'fwd_13w'],
    ['1 wk', '4 wk', '13 wk'],
    ['#90CAF9', '#42A5F5', '#0D47A1'])):
    hit_rate = valid.groupby('mm_quintile', observed=True)[col].apply(lambda x: (x > 0).mean() * 100)
    ax.bar(x + i * 0.25, hit_rate, 0.25, label=label, color=c, edgecolor='white')
ax.axhline(y=50, color='black', linewidth=1, linestyle='--', alpha=0.5, label='50% (coin flip)')
ax.set_xticks(x + 0.25)
ax.set_xticklabels(quintile_returns.index, fontsize=9)
ax.set_ylabel('% Positive Return', fontsize=10, fontweight='bold')
ax.set_title('MM Net Quintile → % Times NG Goes Up', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(axis='y', alpha=0.3)

# 2,0: Extreme positioning signal — returns after >80th and <20th percentile
ax = axes2[1, 0]
extreme_long = valid[valid['mm_net_pct_rank'] >= 80]
extreme_short = valid[valid['mm_net_pct_rank'] <= 20]
neutral = valid[(valid['mm_net_pct_rank'] > 35) & (valid['mm_net_pct_rank'] < 65)]

horizons = ['fwd_1w', 'fwd_2w', 'fwd_4w', 'fwd_8w', 'fwd_13w']
horizon_labels = ['1w', '2w', '4w', '8w', '13w']
x = np.arange(len(horizons))

for data, label, color, offset in [
    (extreme_short, f'After extreme short (<20th, n={len(extreme_short)})', '#2E7D32', -0.25),
    (neutral, f'After neutral (35-65th, n={len(neutral)})', '#9E9E9E', 0),
    (extreme_long, f'After extreme long (>80th, n={len(extreme_long)})', '#D32F2F', 0.25),
]:
    means = [data[h].mean() for h in horizons]
    ax.bar(x + offset, means, 0.24, label=label, color=color, alpha=0.7, edgecolor='white')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(horizon_labels)
ax.set_ylabel('Avg Forward Return (%)', fontsize=10, fontweight='bold')
ax.set_title('Contrarian Signal: Returns After Extreme Positioning', fontsize=12, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# 2,1: Extreme signal — cumulative returns
ax = axes2[1, 1]
for data, label, color in [
    (extreme_short, 'After extreme short', '#2E7D32'),
    (neutral, 'After neutral', '#9E9E9E'),
    (extreme_long, 'After extreme long', '#D32F2F'),
]:
    cum_means = [data[h].mean() for h in horizons]
    ax.plot(range(len(horizons)), cum_means, 'o-', color=color, linewidth=2, markersize=8, label=label)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(range(len(horizons)))
ax.set_xticklabels(horizon_labels)
ax.set_ylabel('Avg Forward Return (%)', fontsize=10, fontweight='bold')
ax.set_title('Contrarian Signal: Return Trajectory After Extremes', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3,0: By-month analysis — does COT signal vary by season?
ax = axes2[2, 0]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
season_colors_list = ['#1565C0', '#1565C0', '#1565C0', '#757575', '#757575', '#FF8F00',
                      '#FF8F00', '#FF8F00', '#FF8F00', '#757575', '#757575', '#1565C0']

# For each month: correlation between MM net and 4w forward return
monthly_corr = []
monthly_n = []
for m in range(1, 13):
    month_data = valid[valid['month'] == m]
    if len(month_data) > 10:
        corr = month_data['mm_net_pct_oi'].corr(month_data['fwd_4w'])
        monthly_corr.append(corr)
        monthly_n.append(len(month_data))
    else:
        monthly_corr.append(0)
        monthly_n.append(0)

ax.bar(range(12), monthly_corr, color=season_colors_list, alpha=0.7, edgecolor='white')
ax.axhline(y=0, color='black', linewidth=0.8)
for i, (corr, n) in enumerate(zip(monthly_corr, monthly_n)):
    ax.text(i, corr + (0.01 if corr >= 0 else -0.03), f'{corr:.2f}\n(n={n})',
            ha='center', fontsize=7, fontweight='bold')
ax.set_xticks(range(12))
ax.set_xticklabels(month_names)
ax.set_ylabel('Correlation (MM Net/OI → 4w Return)', fontsize=10, fontweight='bold')
ax.set_title('Predictive Power by Month (MM Net/OI vs 4-Week Forward Return)', fontsize=12, fontweight='bold')

# 3,1: By-month avg return when specs extreme long vs short
ax = axes2[2, 1]
for m in range(1, 13):
    month_data = valid[valid['month'] == m]
    if len(month_data) < 10:
        continue
    q_low = month_data['mm_net_pct_oi'].quantile(0.25)
    q_high = month_data['mm_net_pct_oi'].quantile(0.75)
    short_ret = month_data[month_data['mm_net_pct_oi'] <= q_low]['fwd_4w'].mean()
    long_ret = month_data[month_data['mm_net_pct_oi'] >= q_high]['fwd_4w'].mean()
    ax.bar(m - 1 - 0.15, short_ret, 0.3, color='#2E7D32', alpha=0.7)
    ax.bar(m - 1 + 0.15, long_ret, 0.3, color='#D32F2F', alpha=0.7)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(12))
ax.set_xticklabels(month_names)
ax.set_ylabel('Avg 4w Fwd Return (%)', fontsize=10, fontweight='bold')
ax.set_title('4-Week Return When Specs Bottom Quartile (green) vs Top Quartile (red)', fontsize=11, fontweight='bold')
ax.legend(handles=[Patch(facecolor='#2E7D32', label='After spec short (bullish?)'),
                    Patch(facecolor='#D32F2F', label='After spec long (bearish?)')],
          fontsize=8, loc='upper right')

# 4,0: Change in positioning vs forward return (flow momentum)
ax = axes2[3, 0]
valid_flow = valid.dropna(subset=['mm_net_chg_4w', 'fwd_4w']).copy()
valid_flow['flow_quintile'] = pd.qcut(valid_flow['mm_net_chg_4w'], 5,
                                       labels=['Q1\n(Selling)', 'Q2', 'Q3', 'Q4', 'Q5\n(Buying)'])
flow_returns = valid_flow.groupby('flow_quintile', observed=True)[['fwd_1w', 'fwd_4w', 'fwd_13w']].mean()
x = np.arange(5)
for i, (col, label, c) in enumerate(zip(
    ['fwd_1w', 'fwd_4w', 'fwd_13w'],
    ['1 wk', '4 wk', '13 wk'],
    ['#90CAF9', '#42A5F5', '#0D47A1'])):
    ax.bar(x + i * 0.25, flow_returns[col], 0.25, label=label, color=c, edgecolor='white')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x + 0.25)
ax.set_xticklabels(flow_returns.index, fontsize=9)
ax.set_ylabel('Avg Forward Return (%)', fontsize=10, fontweight='bold')
ax.set_xlabel('4-Week Flow Direction', fontsize=10)
ax.set_title('MM 4-Week Flow Momentum → Forward NG Return', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=3)
ax.grid(axis='y', alpha=0.3)

# 4,1: Producer positioning predictive power
ax = axes2[3, 1]
valid_prod = valid.dropna(subset=['prod_net', 'fwd_4w']).copy()
valid_prod['prod_quintile'] = pd.qcut(valid_prod['prod_net'], 5,
                                       labels=['Q1\n(Heavy hedge)', 'Q2', 'Q3', 'Q4', 'Q5\n(Light hedge)'])
prod_returns = valid_prod.groupby('prod_quintile', observed=True)[['fwd_1w', 'fwd_4w', 'fwd_13w']].mean()
x = np.arange(5)
for i, (col, label, c) in enumerate(zip(
    ['fwd_1w', 'fwd_4w', 'fwd_13w'],
    ['1 wk', '4 wk', '13 wk'],
    ['#FFE0B2', '#FF9800', '#E65100'])):
    ax.bar(x + i * 0.25, prod_returns[col], 0.25, label=label, color=c, edgecolor='white')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x + 0.25)
ax.set_xticklabels(prod_returns.index, fontsize=9)
ax.set_ylabel('Avg Forward Return (%)', fontsize=10, fontweight='bold')
ax.set_xlabel('Producer Net Position Quintile', fontsize=10)
ax.set_title('Producer Hedge Level → Forward NG Return', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=3)
ax.grid(axis='y', alpha=0.3)

fig2.suptitle('NG COT — Predictive Behavior Analysis', fontsize=16, fontweight='bold', y=1.003)
fig2.tight_layout()
fig2.savefig('/home/wyatt/weather/ng_cot_behavior.png', dpi=150, bbox_inches='tight')
print("  Saved: ng_cot_behavior.png")

# ============================================
# Console Summary
# ============================================
print("\n" + "=" * 65)
print("COT COMPREHENSIVE SUMMARY")
print(f"Report date: {latest['date'].date()} | {len(cot)} weeks of history")
print("=" * 65)

full_pct = lambda col: (cot[col] < latest[col]).mean() * 100

print(f"\n{'Category':<25} {'Net':>12} {'% of OI':>10} {'Pctile':>8}")
print("-" * 60)
for name, net_col, pct_col in [
    ('Managed Money', 'mm_net', 'mm_net_pct_oi'),
    ('Producer/Merchant', 'prod_net', 'prod_net_pct_oi'),
    ('Swap Dealer', 'swap_net', 'swap_net_pct_oi'),
    ('Other Reportable', 'other_net', 'other_net_pct_oi'),
    ('Small Traders (Non-Rept)', 'nonrept_net', 'nonrept_net_pct_oi'),
]:
    net = latest[net_col]
    pct_oi = latest[pct_col]
    pctile = full_pct(net_col)
    print(f"  {name:<23} {net:>+11,.0f}  {pct_oi:>+8.1f}%  {pctile:>6.0f}th")

print(f"\n  Open Interest: {latest['oi']:>12,.0f} ({full_pct('oi'):.0f}th pctile)")
print(f"  MM L/S Ratio: {latest['mm_ls_ratio']:>12.2f} ({full_pct('mm_ls_ratio'):.0f}th pctile)")
print(f"  NG Price: ${latest['ng_price']:.3f}")

print(f"\n{'='*65}")
print("PREDICTIVE ANALYSIS RESULTS")
print(f"{'='*65}")

# Quintile spread (Q1 avg - Q5 avg)
for horizon, label in [('fwd_1w', '1 week'), ('fwd_4w', '4 weeks'), ('fwd_13w', '13 weeks')]:
    q1_ret = quintile_returns.loc[quintile_returns.index[0], horizon]
    q5_ret = quintile_returns.loc[quintile_returns.index[-1], horizon]
    spread = q1_ret - q5_ret
    print(f"\n  {label} forward:")
    print(f"    After spec SHORT (Q1): {q1_ret:+.2f}%")
    print(f"    After spec LONG  (Q5): {q5_ret:+.2f}%")
    print(f"    Contrarian spread: {spread:+.2f}%")
    if spread > 0:
        print(f"    -> Contrarian signal WORKS at this horizon (short specs = higher returns)")
    else:
        print(f"    -> Contrarian signal DOES NOT work at this horizon")

# Extreme signal
print(f"\n  Extreme positioning (>80th / <20th percentile):")
for horizon, label in [('fwd_4w', '4w'), ('fwd_13w', '13w')]:
    short_ret = extreme_short[horizon].mean()
    long_ret = extreme_long[horizon].mean()
    print(f"    {label}: after extreme short {short_ret:+.2f}%, after extreme long {long_ret:+.2f}%")

# Monthly correlation summary
print(f"\n  Monthly predictive power (MM net/OI → 4w return):")
best_month = max(range(12), key=lambda i: abs(monthly_corr[i]))
print(f"    Strongest: {month_names[best_month]} (r={monthly_corr[best_month]:.3f})")
winter_corr = np.mean([monthly_corr[m-1] for m in [12, 1, 2, 3]])
summer_corr = np.mean([monthly_corr[m-1] for m in [6, 7, 8, 9]])
print(f"    Winter avg correlation: {winter_corr:.3f}")
print(f"    Summer avg correlation: {summer_corr:.3f}")

# Current positioning signal
print(f"\n{'='*65}")
print("CURRENT SIGNAL FOR UNG (Apr26)")
print(f"{'='*65}")
mm_pct = full_pct('mm_net')
mm_oi_pct = full_pct('mm_net_pct_oi')
print(f"\n  MM Net: {latest['mm_net']:+,.0f} ({mm_pct:.0f}th percentile)")
print(f"  MM Net/OI: {latest['mm_net_pct_oi']:+.1f}% ({mm_oi_pct:.0f}th percentile)")
print(f"  4-week flow: {latest['mm_net_chg_4w']:+,.0f} contracts")

# What does the quintile analysis say about current?
current_q = None
for q_label in quintile_returns.index:
    q_data = valid[valid['mm_quintile'] == q_label]
    if len(q_data) > 0:
        q_min = q_data['mm_net_pct_oi'].min()
        q_max = q_data['mm_net_pct_oi'].max()
        if q_min <= latest['mm_net_pct_oi'] <= q_max:
            current_q = q_label
            break

if current_q is not None:
    q_ret_4w = quintile_returns.loc[current_q, 'fwd_4w']
    q_ret_13w = quintile_returns.loc[current_q, 'fwd_13w']
    print(f"\n  Current positioning falls in: {current_q.replace(chr(10), ' ')}")
    print(f"    Historical avg 4w return from this quintile: {q_ret_4w:+.2f}%")
    print(f"    Historical avg 13w return from this quintile: {q_ret_13w:+.2f}%")

print("\n" + "=" * 65)

plt.show()
