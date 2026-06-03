#!/usr/bin/env python3
"""
NG ETF Flow Analysis — Retail Sentiment via UNG/BOIL/KOLD Creation/Redemption
Fetches daily shares outstanding + AUM from ProShares (BOIL/KOLD) and
ALPS API via USCF (UNG), computes flows, and compares to NG price.
"""
import urllib.request
import subprocess
import io
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

print("NG ETF Flow Analysis — UNG/BOIL/KOLD Retail Sentiment")
print("=" * 65)

# ============================================
# Fetch ProShares historical NAV data (BOIL/KOLD)
# ============================================
etf_data = {}
for ticker in ['BOIL', 'KOLD']:
    url = f'https://accounts.profunds.com/etfdata/ByFund/{ticker}-historical_nav.csv'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=20)
    df = pd.read_csv(io.BytesIO(resp.read()))
    df['date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    df['shares'] = df['Shares Outstanding (000)'] * 1000
    df['aum'] = df['Assets Under Management']
    df['nav'] = df['NAV']
    df = df[df['shares'] > 0].copy()
    etf_data[ticker] = df[['date', 'shares', 'aum', 'nav']].copy()
    print(f"  {ticker}: {len(etf_data[ticker])} days ({etf_data[ticker]['date'].min().date()} to {etf_data[ticker]['date'].max().date()})")

# ============================================
# Fetch UNG data from ALPS API (via USCF)
# Uses curl subprocess — Python SSL hangs on this server
# ============================================
print("  UNG: fetching from ALPS API...")

# Step 1: get cookies
subprocess.run(['curl', '-s', '-c', '/tmp/ung_cookies.txt', '-o', '/dev/null',
                'https://www.uscfinvestments.com/ung'], timeout=15)

# Step 2: get JWT token
token_result = subprocess.run(
    ['curl', '-s', '-b', '/tmp/ung_cookies.txt',
     'https://www.uscfinvestments.com/site-template/assets/javascript/api_key.php'],
    capture_output=True, text=True, timeout=15)
token_match = re.search(r"var token = '([^']+)'", token_result.stdout)
if not token_match:
    raise RuntimeError("Failed to get ALPS API token")
token = token_match.group(1)

# Step 3: fetch historical NAV (includes navTotal for shares calculation)
nav_result = subprocess.run(
    ['curl', '-s', '-H', f'Authorization: Bearer {token}',
     'https://secure.alpsinc.com/MarketingAPI/api/v1/historicalnav/UNG/inception-today'],
    capture_output=True, text=True, timeout=30)
ung_raw = json.loads(nav_result.stdout)
ung_records = ung_raw[0]['UNG']

# Step 4: get current day's shares outstanding from dailyprice
daily_result = subprocess.run(
    ['curl', '-s', '-H', f'Authorization: Bearer {token}',
     'https://secure.alpsinc.com/MarketingAPI/api/v1/dailyprice/UNG'],
    capture_output=True, text=True, timeout=15)
ung_daily = json.loads(daily_result.stdout)

# Parse UNG historical data
ung_rows = []
for rec in ung_records:
    nav = rec.get('value', 0)
    nav_total = rec.get('navTotal', 0)
    if nav > 0 and nav_total > 0:
        shares = nav_total / nav
        ung_rows.append({
            'date': pd.to_datetime(rec['date']).normalize(),
            'shares': shares,
            'aum': nav_total,
            'nav': nav,
        })

ung_df = pd.DataFrame(ung_rows).sort_values('date').reset_index(drop=True)
etf_data['UNG'] = ung_df
print(f"  UNG: {len(ung_df)} days ({ung_df['date'].min().date()} to {ung_df['date'].max().date()})")

# Print current UNG snapshot
if ung_daily:
    d = ung_daily[0] if isinstance(ung_daily, list) else ung_daily
    so_str = d.get('so', 'N/A')
    cr_str = d.get('cr', 'N/A')
    print(f"  UNG today: shares={so_str}, creation/redemption={cr_str}")

# ============================================
# Compute daily flows for all ETFs
# ============================================
for ticker in ['BOIL', 'KOLD', 'UNG']:
    df = etf_data[ticker]
    df['shares_chg'] = df['shares'].diff()
    df['flow_dollars'] = df['shares_chg'] * df['nav']
    df['flow_5d'] = df['flow_dollars'].rolling(5, min_periods=1).sum()
    df['flow_20d'] = df['flow_dollars'].rolling(20, min_periods=5).sum()
    df['flow_60d'] = df['flow_dollars'].rolling(60, min_periods=20).sum()
    etf_data[ticker] = df

# ============================================
# Merge all three on date
# ============================================
merged = pd.merge(
    etf_data['BOIL'][['date', 'shares', 'aum', 'nav', 'shares_chg', 'flow_dollars', 'flow_5d', 'flow_20d', 'flow_60d']],
    etf_data['KOLD'][['date', 'shares', 'aum', 'nav', 'shares_chg', 'flow_dollars', 'flow_5d', 'flow_20d', 'flow_60d']],
    on='date', suffixes=('_boil', '_kold'), how='inner'
)
merged = pd.merge(
    merged,
    etf_data['UNG'][['date', 'shares', 'aum', 'nav', 'shares_chg', 'flow_dollars', 'flow_5d', 'flow_20d', 'flow_60d']].rename(
        columns={c: f'{c}_ung' for c in ['shares', 'aum', 'nav', 'shares_chg', 'flow_dollars', 'flow_5d', 'flow_20d', 'flow_60d']}),
    on='date', how='inner'
)
merged = merged.sort_values('date').reset_index(drop=True)

# Combined long flow = UNG + BOIL inflow (both bullish instruments)
merged['long_flow'] = merged['flow_dollars_boil'] + merged['flow_dollars_ung']
merged['long_flow_20d'] = merged['flow_20d_boil'] + merged['flow_20d_ung']

# Net bullish flow = (UNG + BOIL inflow) - KOLD inflow
merged['net_bull_flow'] = merged['long_flow'] - merged['flow_dollars_kold']
merged['net_bull_flow_20d'] = merged['long_flow_20d'] - merged['flow_20d_kold']

# Total NG ETF AUM
merged['total_aum'] = merged['aum_boil'] + merged['aum_kold'] + merged['aum_ung']
merged['long_aum'] = merged['aum_boil'] + merged['aum_ung']

# Bull share = (UNG + BOIL) / (UNG + BOIL + KOLD)
merged['bull_share'] = merged['long_aum'] / merged['total_aum']

# BOIL-only share of leveraged (BOIL vs KOLD)
merged['boil_kold_ratio'] = merged['aum_boil'] / merged['aum_kold'].replace(0, np.nan)

print(f"\nMerged (all 3): {len(merged)} days ({merged['date'].min().date()} to {merged['date'].max().date()})")

# ============================================
# Fetch NG=F price
# ============================================
print("\nFetching NG=F price...")
ng = yf.Ticker("NG=F")
ng_hist = ng.history(period="max", interval="1d")
if ng_hist.index.tz is not None:
    ng_hist.index = ng_hist.index.tz_localize(None)
ng_close = ng_hist[['Close']].rename(columns={'Close': 'ng_price'})
ng_close.index.name = 'date'
ng_close = ng_close.reset_index()

merged = pd.merge_asof(merged.sort_values('date'), ng_close.sort_values('date'),
                       on='date', direction='nearest', tolerance=pd.Timedelta(days=3))
merged = merged.dropna(subset=['ng_price'])
print(f"  With NG price: {len(merged)} days")

# Forward returns
merged['ng_fwd_5d'] = merged['ng_price'].shift(-5) / merged['ng_price'] - 1
merged['ng_fwd_20d'] = merged['ng_price'].shift(-20) / merged['ng_price'] - 1
merged['ng_fwd_60d'] = merged['ng_price'].shift(-60) / merged['ng_price'] - 1

# Focus on 2019+ for charts (meaningful AUM era)
recent = merged[merged['date'] >= '2019-01-01'].copy()

# ============================================
# PLOTTING — Page 1: Flow Dashboard (5x2)
# ============================================
print("\nCreating charts...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(5, 2, figsize=(20, 30))
latest = recent.iloc[-1]

# Colors
UNG_COLOR = '#FF8F00'    # amber for UNG
BOIL_COLOR = '#2E7D32'   # green for BOIL (leveraged bull)
KOLD_COLOR = '#D32F2F'   # red for KOLD (leveraged bear)
NG_COLOR = '#1565C0'     # blue for NG price
FLOW_POS = '#4CAF50'
FLOW_NEG = '#E53935'

# --- 1,0: All Three AUM Stacked ---
ax = axes[0, 0]
ax2 = ax.twinx()
ax.stackplot(recent['date'],
             recent['aum_ung'] / 1e6,
             recent['aum_boil'] / 1e6,
             recent['aum_kold'] / 1e6,
             labels=['UNG (1x Long)', 'BOIL (2x Long)', 'KOLD (2x Short)'],
             colors=[UNG_COLOR, BOIL_COLOR, KOLD_COLOR], alpha=0.6)
ax2.plot(recent['date'], recent['ng_price'], color='black', linewidth=1.8, alpha=0.9, label='NG=F')
ax.set_ylabel('AUM ($M)', fontsize=10, fontweight='bold')
ax2.set_ylabel('NG=F', fontsize=10)
ax.set_title('NG ETF AUM: UNG + BOIL + KOLD', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Annotate latest total
ax.annotate(f'Total: ${latest["total_aum"]/1e6:,.0f}M',
            xy=(latest['date'], latest['total_aum']/1e6),
            fontsize=9, fontweight='bold', color='black',
            xytext=(-10, 5), textcoords='offset points', ha='right')

# --- 1,1: Bull Share (UNG+BOIL) / Total vs NG ---
ax = axes[0, 1]
ax2 = ax.twinx()
ax.plot(recent['date'], recent['bull_share'] * 100, color=UNG_COLOR, linewidth=2.5, alpha=0.9)
ax.axhline(y=50, color='black', linewidth=1, linestyle='--', alpha=0.4)
ax.fill_between(recent['date'], 50, recent['bull_share'] * 100,
                where=recent['bull_share'] > 0.5, alpha=0.15, color=BOIL_COLOR)
ax.fill_between(recent['date'], 50, recent['bull_share'] * 100,
                where=recent['bull_share'] <= 0.5, alpha=0.15, color=KOLD_COLOR)
ax2.plot(recent['date'], recent['ng_price'], color=NG_COLOR, linewidth=1.2, alpha=0.7)
ax.set_ylabel('Long Share of Total AUM (%)', fontsize=10, fontweight='bold')
ax2.set_ylabel('NG=F', color=NG_COLOR, fontsize=10)
ax.set_title('Retail Sentiment: (UNG+BOIL) Share of All NG ETF AUM', fontsize=11, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
bull_pct = latest['bull_share'] * 100
ax.annotate(f'{bull_pct:.0f}%', xy=(latest['date'], bull_pct),
            fontsize=12, fontweight='bold', color=UNG_COLOR,
            xytext=(5, 5), textcoords='offset points')

# --- 2,0: UNG AUM + Flow ---
ax = axes[1, 0]
ax2 = ax.twinx()
ax.fill_between(recent['date'], recent['aum_ung'] / 1e6, alpha=0.4, color=UNG_COLOR, label='UNG AUM')
ax2.bar(recent['date'], recent['flow_20d_ung'] / 1e6, width=1, alpha=0.5,
        color=[FLOW_POS if v >= 0 else FLOW_NEG for v in recent['flow_20d_ung'].fillna(0)],
        label='UNG 20d flow')
ax.set_ylabel('UNG AUM ($M)', fontsize=10, fontweight='bold', color=UNG_COLOR)
ax2.set_ylabel('20-Day Flow ($M)', fontsize=10)
ax.set_title('UNG: AUM & 20-Day Rolling Flow', fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.annotate(f'${latest["aum_ung"]/1e6:,.0f}M', xy=(latest['date'], latest['aum_ung']/1e6),
            fontsize=9, fontweight='bold', color=UNG_COLOR, ha='right',
            xytext=(-5, 5), textcoords='offset points')

# --- 2,1: BOIL vs KOLD AUM side by side ---
ax = axes[1, 1]
ax2 = ax.twinx()
ax.fill_between(recent['date'], recent['aum_boil'] / 1e6, alpha=0.5, color=BOIL_COLOR, label='BOIL AUM')
ax.fill_between(recent['date'], recent['aum_kold'] / 1e6, alpha=0.5, color=KOLD_COLOR, label='KOLD AUM')
ax2.plot(recent['date'], recent['ng_price'], color=NG_COLOR, linewidth=1.5, alpha=0.8)
ax.set_ylabel('AUM ($M)', fontsize=10, fontweight='bold')
ax2.set_ylabel('NG=F', color=NG_COLOR, fontsize=10)
ax.set_title('BOIL (2x Bull) vs KOLD (2x Bear) AUM', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.annotate(f'BOIL: ${latest["aum_boil"]/1e6:.0f}M', xy=(latest['date'], latest['aum_boil']/1e6),
            fontsize=8, fontweight='bold', color=BOIL_COLOR, ha='right',
            xytext=(-5, 8), textcoords='offset points')
ax.annotate(f'KOLD: ${latest["aum_kold"]/1e6:.0f}M', xy=(latest['date'], latest['aum_kold']/1e6),
            fontsize=8, fontweight='bold', color=KOLD_COLOR, ha='right',
            xytext=(-5, -12), textcoords='offset points')

# --- 3,0: Combined Net Bullish Flow (UNG+BOIL-KOLD) ---
ax = axes[2, 0]
ax2 = ax.twinx()
ax.bar(recent['date'], recent['net_bull_flow'] / 1e6, width=1,
       color=[FLOW_POS if v >= 0 else FLOW_NEG for v in recent['net_bull_flow'].fillna(0)], alpha=0.3)
ax.plot(recent['date'], recent['net_bull_flow_20d'] / 1e6, color=UNG_COLOR, linewidth=2.5,
        label='20-day net bull flow')
ax2.plot(recent['date'], recent['ng_price'], color=NG_COLOR, linewidth=1.2, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Net Bullish Flow ($M)', fontsize=10, fontweight='bold')
ax2.set_ylabel('NG=F', color=NG_COLOR, fontsize=10)
ax.set_title('Combined Net Bullish Flow: (UNG+BOIL) − KOLD', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- 3,1: Individual 20d flows all three ---
ax = axes[2, 1]
ax.plot(recent['date'], recent['flow_20d_ung'] / 1e6, color=UNG_COLOR, linewidth=2, label='UNG 20d flow')
ax.plot(recent['date'], recent['flow_20d_boil'] / 1e6, color=BOIL_COLOR, linewidth=1.5, label='BOIL 20d flow')
ax.plot(recent['date'], recent['flow_20d_kold'] / 1e6, color=KOLD_COLOR, linewidth=1.5, label='KOLD 20d flow')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('20-Day Rolling Flow ($M)', fontsize=10, fontweight='bold')
ax.set_title('Individual ETF 20-Day Flows', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- 4,0: Share creation/redemption (all three) ---
ax = axes[3, 0]
for ticker, col_suffix, color, label in [
    ('UNG', '_ung', UNG_COLOR, 'UNG'),
    ('BOIL', '_boil', BOIL_COLOR, 'BOIL'),
    ('KOLD', '_kold', KOLD_COLOR, 'KOLD'),
]:
    shares_chg_20d = recent[f'shares_chg{col_suffix}'].rolling(20, min_periods=5).sum() / 1e6
    ax.plot(recent['date'].values, shares_chg_20d.values, color=color, linewidth=1.5, label=f'{label} 20d')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('20-Day Net Shares Created (M)', fontsize=10, fontweight='bold')
ax.set_title('Share Creation/Redemption — All Three ETFs', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- 4,1: UNG shares outstanding history (the raw data) ---
ax = axes[3, 1]
ax2 = ax.twinx()
ung_recent = etf_data['UNG'][etf_data['UNG']['date'] >= '2019-01-01']
ax.plot(ung_recent['date'], ung_recent['shares'] / 1e6, color=UNG_COLOR, linewidth=2, label='UNG Shares Outstanding')
ax2.plot(ung_recent['date'], ung_recent['nav'], color=NG_COLOR, linewidth=1.2, alpha=0.7, label='UNG NAV')
ax.set_ylabel('Shares Outstanding (M)', fontsize=10, fontweight='bold', color=UNG_COLOR)
ax2.set_ylabel('UNG NAV ($)', color=NG_COLOR, fontsize=10)
ax.set_title('UNG Shares Outstanding (Creation/Redemption History)', fontsize=12, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Note the reverse split
ax.annotate('1:4 reverse split\n(Jan 2024)', xy=(pd.Timestamp('2024-01-23'), ung_recent.loc[ung_recent['date'] >= '2024-01-23', 'shares'].iloc[0]/1e6),
            fontsize=8, color='gray', ha='center',
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))

# --- 5,0: Contrarian quintile analysis ---
ax = axes[4, 0]
valid_q = recent.dropna(subset=['bull_share', 'ng_fwd_20d']).copy()
valid_q['sentiment_q'] = pd.qcut(valid_q['bull_share'], 5,
                                  labels=['Q1\n(Most\nBearish)', 'Q2', 'Q3\n(Neutral)', 'Q4', 'Q5\n(Most\nBullish)'])
q_returns = valid_q.groupby('sentiment_q', observed=True)[['ng_fwd_5d', 'ng_fwd_20d', 'ng_fwd_60d']].mean() * 100
x = np.arange(5)
width = 0.25
for i, (col, label, c) in enumerate(zip(
    ['ng_fwd_5d', 'ng_fwd_20d', 'ng_fwd_60d'],
    ['5 day', '20 day', '60 day'],
    ['#90CAF9', '#42A5F5', '#0D47A1'])):
    ax.bar(x + i * width, q_returns[col], width, label=label, color=c, edgecolor='white')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(q_returns.index, fontsize=8)
ax.set_ylabel('Avg Forward NG Return (%)', fontsize=10, fontweight='bold')
ax.set_title('Retail Sentiment Quintile → Forward NG Return', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=3)
ax.grid(axis='y', alpha=0.3)

# --- 5,1: Extreme contrarian signal ---
ax = axes[4, 1]
extreme_bull = valid_q[valid_q['bull_share'] >= valid_q['bull_share'].quantile(0.8)]
extreme_bear = valid_q[valid_q['bull_share'] <= valid_q['bull_share'].quantile(0.2)]
neutral_zone = valid_q[(valid_q['bull_share'] >= valid_q['bull_share'].quantile(0.35)) &
                       (valid_q['bull_share'] <= valid_q['bull_share'].quantile(0.65))]

horizons = ['ng_fwd_5d', 'ng_fwd_20d', 'ng_fwd_60d']
horizon_labels = ['5d', '20d', '60d']
x_h = np.arange(len(horizons))

for data, label, color, offset in [
    (extreme_bear, f'After retail bearish (<20th, n={len(extreme_bear)})', KOLD_COLOR, -0.25),
    (neutral_zone, f'Neutral (35-65th, n={len(neutral_zone)})', '#9E9E9E', 0),
    (extreme_bull, f'After retail bullish (>80th, n={len(extreme_bull)})', BOIL_COLOR, 0.25),
]:
    means = [data[h].mean() * 100 for h in horizons]
    ax.bar(x_h + offset, means, 0.24, label=label, color=color, alpha=0.7, edgecolor='white')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x_h)
ax.set_xticklabels(horizon_labels)
ax.set_ylabel('Avg Forward NG Return (%)', fontsize=10, fontweight='bold')
ax.set_title('Contrarian Signal: Returns After Extreme Retail Sentiment', fontsize=12, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('NG ETF Flow Analysis — UNG / BOIL / KOLD Retail Sentiment', fontsize=16, fontweight='bold', y=1.002)
fig.tight_layout()
fig.savefig('/home/wyatt/weather/ng_etf_flows.png', dpi=150, bbox_inches='tight')
print("  Saved: ng_etf_flows.png")

# ============================================
# Console Summary
# ============================================
print("\n" + "=" * 65)
print("ETF FLOW SUMMARY")
print(f"Data: {recent['date'].min().date()} to {recent['date'].max().date()}")
print("=" * 65)

print(f"\n  CURRENT STATE ({latest['date'].date()}):")
print(f"    UNG  AUM: ${latest['aum_ung']/1e6:,.0f}M")
print(f"    BOIL AUM: ${latest['aum_boil']/1e6:,.0f}M")
print(f"    KOLD AUM: ${latest['aum_kold']/1e6:,.0f}M")
print(f"    Total:    ${latest['total_aum']/1e6:,.0f}M")
print(f"    Long share (UNG+BOIL): {latest['bull_share']*100:.1f}%  |  KOLD short share: {(1-latest['bull_share'])*100:.1f}%")
print(f"    NG Price: ${latest['ng_price']:.3f}")

# Recent flows
last_5 = recent.tail(5)
last_20 = recent.tail(20)
print(f"\n  RECENT FLOWS (dollar):")
for label, data in [('Last 5 days', last_5), ('Last 20 days', last_20)]:
    ung_f = data['flow_dollars_ung'].sum()
    boil_f = data['flow_dollars_boil'].sum()
    kold_f = data['flow_dollars_kold'].sum()
    net = ung_f + boil_f - kold_f
    print(f"    {label}:  UNG ${ung_f/1e6:+,.1f}M  |  BOIL ${boil_f/1e6:+,.1f}M  |  KOLD ${kold_f/1e6:+,.1f}M  |  Net bull ${net/1e6:+,.1f}M")

# Percentile
bull_share_pct = (recent['bull_share'] < latest['bull_share']).mean() * 100
net_flow_pct = (recent['net_bull_flow_20d'].dropna() < latest['net_bull_flow_20d']).mean() * 100
print(f"\n  PERCENTILE RANK (since 2019):")
print(f"    Long share: {bull_share_pct:.0f}th percentile")
print(f"    20d net bull flow: {net_flow_pct:.0f}th percentile")

# Contrarian
print(f"\n  CONTRARIAN ANALYSIS (incl UNG+BOIL vs KOLD):")
for horizon, label in [('ng_fwd_5d', '5-day'), ('ng_fwd_20d', '20-day'), ('ng_fwd_60d', '60-day')]:
    bull_ret = extreme_bull[horizon].mean() * 100
    bear_ret = extreme_bear[horizon].mean() * 100
    print(f"    {label}: after retail bullish {bull_ret:+.1f}%  |  after retail bearish {bear_ret:+.1f}%")
    if bear_ret > bull_ret:
        print(f"      -> Contrarian works: fade retail {bear_ret - bull_ret:+.1f}% edge")
    else:
        print(f"      -> Momentum works at this horizon")

# Current signal
print(f"\n  CURRENT SIGNAL:")
if latest['bull_share'] > 0.7:
    print(f"    Retail is BULLISH ({latest['bull_share']*100:.0f}% long share) — contrarian bearish?")
elif latest['bull_share'] < 0.55:
    print(f"    Retail is BEARISH ({latest['bull_share']*100:.0f}% long share) — contrarian bullish?")
else:
    print(f"    Retail is NEUTRAL ({latest['bull_share']*100:.0f}% long share)")

print("\n" + "=" * 65)

plt.show()
