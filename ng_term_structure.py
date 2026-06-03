#!/usr/bin/env python3
"""
NG Term Structure Analysis
Shows forward curve, curve evolution, calendar spreads, individual contracts,
historical comparison, and inflation/trend-adjusted rich/cheap analysis.
"""
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

print("NG Term Structure Analysis")
print("=" * 65)

# ============================================
# Contract month codes and generation
# ============================================
MONTH_CODES = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
               7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
MONTH_NAMES = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


def get_season(month):
    if month in (12, 1, 2, 3):
        return 'winter'
    elif month in (6, 7, 8, 9):
        return 'summer'
    else:
        return 'shoulder'


SEASON_COLORS = {'winter': '#1565C0', 'summer': '#FF8F00', 'shoulder': '#757575'}

today = datetime.now()
contracts = []
for offset in range(24):
    m = (today.month - 1 + offset) % 12 + 1
    y = today.year + (today.month - 1 + offset) // 12
    code = MONTH_CODES[m]
    yy = y % 100
    ticker = f"NG{code}{yy:02d}.NYM"
    label = f"{MONTH_NAMES[m]}{yy:02d}"
    contracts.append({'ticker': ticker, 'label': label, 'month': m, 'year': y,
                      'season': get_season(m), 'expiry_approx': datetime(y, m, 1)})

# ============================================
# Fetch data
# ============================================
print(f"\nFetching {len(contracts)} individual contracts...")
contract_data = {}
for c in contracts:
    try:
        t = yf.Ticker(c['ticker'])
        hist = t.history(period="90d", interval="1d")
        if len(hist) > 0:
            hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index
            contract_data[c['ticker']] = {
                'hist': hist, 'label': c['label'], 'month': c['month'],
                'year': c['year'], 'season': c['season'],
                'expiry_approx': c['expiry_approx'],
                'latest_price': hist['Close'].iloc[-1],
                'latest_date': hist.index[-1],
            }
            print(f"  {c['label']} ({c['ticker']}): {len(hist)} bars, latest ${hist['Close'].iloc[-1]:.3f}")
    except Exception:
        pass

print(f"\nFetched {len(contract_data)} contracts with data")

print("Fetching NG=F continuous (90d + 10yr)...")
ng_cont = yf.Ticker("NG=F")
ng_cont_hist = ng_cont.history(period="90d", interval="1d")
if ng_cont_hist.index.tz is not None:
    ng_cont_hist.index = ng_cont_hist.index.tz_localize(None)

ng_long_hist = ng_cont.history(period="10y", interval="1d")
if ng_long_hist.index.tz is not None:
    ng_long_hist.index = ng_long_hist.index.tz_localize(None)
print(f"  NG=F 90d: {len(ng_cont_hist)} bars | 10yr: {len(ng_long_hist)} bars ({ng_long_hist.index.min().date()} to {ng_long_hist.index.max().date()})")

# ============================================
# Trend fitting on 10yr NG=F
# ============================================
print("\nFitting trend models...")
ng_long_hist['month'] = ng_long_hist.index.month
ng_long_hist['year'] = ng_long_hist.index.year
ng_long_hist['days_from_start'] = (ng_long_hist.index - ng_long_hist.index[0]).days

# Remove outliers for fitting (clip at 1st/99th percentile)
p01 = ng_long_hist['Close'].quantile(0.01)
p99 = ng_long_hist['Close'].quantile(0.99)
fit_mask = (ng_long_hist['Close'] >= p01) & (ng_long_hist['Close'] <= p99)
fit_data = ng_long_hist[fit_mask]

t_days = fit_data['days_from_start'].values
log_prices = np.log(fit_data['Close'].values)

# Model 1: Exponential trend (log-linear): log(P) = a + b*t
exp_coeffs = np.polyfit(t_days, log_prices, 1)
exp_trend_all = np.exp(exp_coeffs[1] + exp_coeffs[0] * ng_long_hist['days_from_start'].values)
annual_drift_pct = (np.exp(exp_coeffs[0] * 365.25) - 1) * 100

# Model 2: Linear trend: P = a + b*t
lin_coeffs = np.polyfit(t_days, fit_data['Close'].values, 1)
lin_trend_all = lin_coeffs[1] + lin_coeffs[0] * ng_long_hist['days_from_start'].values

# Model 3: 3yr rolling median (adapts to regime changes)
rolling_3yr = ng_long_hist['Close'].rolling(window=756, min_periods=200, center=True).median()

# Compute detrended prices (ratio to exponential trend)
ng_long_hist['exp_trend'] = exp_trend_all
ng_long_hist['lin_trend'] = lin_trend_all
ng_long_hist['rolling_3yr'] = rolling_3yr
ng_long_hist['detrended_exp'] = ng_long_hist['Close'] / ng_long_hist['exp_trend']
ng_long_hist['detrended_roll'] = ng_long_hist['Close'] / ng_long_hist['rolling_3yr']

print(f"  Exponential trend: {annual_drift_pct:+.1f}%/yr compound drift")
print(f"  Linear trend: {lin_coeffs[0]*365.25:+.3f} $/yr drift")
print(f"  Current exp trend value: ${exp_trend_all[-1]:.3f}")

# ============================================
# Build percentiles: RAW and TREND-ADJUSTED
# ============================================
hist_percentiles_raw = {}
hist_percentiles_adj = {}
for m in range(1, 13):
    mask = ng_long_hist['month'] == m
    month_data = ng_long_hist[mask]
    raw_prices = month_data['Close'].values
    adj_ratios = month_data['detrended_exp'].dropna().values

    if len(raw_prices) > 20:
        hist_percentiles_raw[m] = {
            'p10': np.percentile(raw_prices, 10), 'p25': np.percentile(raw_prices, 25),
            'p50': np.percentile(raw_prices, 50), 'p75': np.percentile(raw_prices, 75),
            'p90': np.percentile(raw_prices, 90), 'all': raw_prices,
        }
    if len(adj_ratios) > 20:
        hist_percentiles_adj[m] = {
            'p10': np.percentile(adj_ratios, 10), 'p25': np.percentile(adj_ratios, 25),
            'p50': np.percentile(adj_ratios, 50), 'p75': np.percentile(adj_ratios, 75),
            'p90': np.percentile(adj_ratios, 90), 'all': adj_ratios,
        }

# Historical winter-summer spread distribution
dec_prices_hist = ng_long_hist[ng_long_hist['month'] == 12].groupby('year')['Close'].mean()
jun_prices_hist = ng_long_hist[ng_long_hist['month'] == 6].groupby('year')['Close'].mean()
common_years = dec_prices_hist.index.intersection(jun_prices_hist.index)
hist_winter_summer_spreads = dec_prices_hist.loc[common_years] - jun_prices_hist.loc[common_years]

# ============================================
# Sort contracts, find key contracts
# ============================================
sorted_tickers = sorted(contract_data.keys(), key=lambda t: contract_data[t]['expiry_approx'])
if len(sorted_tickers) < 1:
    print("ERROR: No contract data available")
    exit(1)

front_ticker = sorted_tickers[0]
front = contract_data[front_ticker]


def find_contract(month, year):
    for tk in sorted_tickers:
        c = contract_data[tk]
        if c['month'] == month and c['year'] == year:
            return tk
    return None


m1_ticker = sorted_tickers[0] if len(sorted_tickers) >= 1 else None
m2_ticker = sorted_tickers[1] if len(sorted_tickers) >= 2 else None
dec26_ticker = find_contract(12, 2026)
jun26_ticker = find_contract(6, 2026)

next_winter_ticker = None
for tk in sorted_tickers:
    c = contract_data[tk]
    if c['month'] == 12 and c['expiry_approx'] > today:
        next_winter_ticker = tk
        break

shoulder_ticker = None
for tk in sorted_tickers[1:]:
    if contract_data[tk]['season'] == 'shoulder':
        shoulder_ticker = tk
        break

# ============================================
# Compute rich/cheap: RAW and TREND-ADJUSTED
# ============================================
labels = [contract_data[tk]['label'] for tk in sorted_tickers]
prices = [contract_data[tk]['latest_price'] for tk in sorted_tickers]
seasons = [contract_data[tk]['season'] for tk in sorted_tickers]

# Current trend value (extrapolate to each contract's delivery month)
days_from_start_now = (pd.Timestamp(today) - ng_long_hist.index[0]).days

rich_cheap = []
for tk in sorted_tickers:
    c = contract_data[tk]
    m = c['month']
    price = c['latest_price']

    # Project trend to contract delivery date
    delivery_date = c['expiry_approx']
    days_to_delivery = (pd.Timestamp(delivery_date) - ng_long_hist.index[0]).days
    trend_at_delivery = np.exp(exp_coeffs[1] + exp_coeffs[0] * days_to_delivery)
    current_ratio = price / trend_at_delivery

    entry = {'ticker': tk, 'label': c['label'], 'month': m, 'season': c['season'],
             'price': price, 'trend_price': trend_at_delivery, 'detrended_ratio': current_ratio,
             'raw_pct': np.nan, 'adj_pct': np.nan,
             'raw_p50': np.nan, 'adj_p50': np.nan}

    if m in hist_percentiles_raw:
        entry['raw_pct'] = (hist_percentiles_raw[m]['all'] < price).mean() * 100
        entry['raw_p50'] = hist_percentiles_raw[m]['p50']
    if m in hist_percentiles_adj:
        entry['adj_pct'] = (hist_percentiles_adj[m]['all'] < current_ratio).mean() * 100
        entry['adj_p50'] = hist_percentiles_adj[m]['p50']

    rich_cheap.append(entry)

# ============================================
# Create chart: 4x2 layout
# ============================================
print("\nCreating chart...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(4, 2, figsize=(20, 26))

# ============================================
# Row 1, Left: Forward Curve Snapshot
# ============================================
ax = axes[0, 0]
bar_colors = [SEASON_COLORS[s] for s in seasons]
ax.bar(range(len(labels)), prices, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Price ($/MMBtu)', fontsize=11, fontweight='bold')
ax.set_title('Forward Curve Snapshot (Today)', fontsize=13, fontweight='bold')

ax.annotate(f'Front: ${prices[0]:.3f}', xy=(0, prices[0]),
            xytext=(15, 15), textcoords='offset points', fontsize=9, fontweight='bold',
            color=SEASON_COLORS[seasons[0]],
            arrowprops=dict(arrowstyle='->', color=SEASON_COLORS[seasons[0]], lw=1.2))

winter_indices = [i for i, s in enumerate(seasons) if s == 'winter' and i > 0]
if winter_indices:
    peak_idx, peak_price = max([(i, prices[i]) for i in winter_indices], key=lambda x: x[1])
    ax.annotate(f'Winter peak: ${peak_price:.3f}', xy=(peak_idx, peak_price),
                xytext=(15, 15), textcoords='offset points', fontsize=9, fontweight='bold',
                color=SEASON_COLORS['winter'],
                arrowprops=dict(arrowstyle='->', color=SEASON_COLORS['winter'], lw=1.2))

if len(prices) >= 2:
    m1m2 = prices[0] - prices[1]
    ax.text(0.98, 0.95, f'M1-M2: {m1m2:+.3f} ({"back" if m1m2 > 0 else "contango"})',
            transform=ax.transAxes, ha='right', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax.legend(handles=[Patch(facecolor=SEASON_COLORS[s], label=l) for s, l in
                    [('winter', 'Winter'), ('summer', 'Summer'), ('shoulder', 'Shoulder')]],
          loc='upper left', fontsize=8, framealpha=0.9)

# ============================================
# Row 1, Right: Curve Evolution Spaghetti
# ============================================
ax = axes[0, 1]
all_dates = sorted(set(d for tk in sorted_tickers for d in contract_data[tk]['hist'].index.tolist()))

snapshot_dates = []
if len(all_dates) > 0:
    date_range = (all_dates[-1] - all_dates[0]).days
    if date_range > 0:
        step = max(date_range // 10, 1)
        current = all_dates[0]
        while current <= all_dates[-1]:
            closest = min(all_dates, key=lambda d: abs((d - current).total_seconds()))
            if closest not in snapshot_dates:
                snapshot_dates.append(closest)
            current += timedelta(days=step)
        if all_dates[-1] not in snapshot_dates:
            snapshot_dates.append(all_dates[-1])
    else:
        snapshot_dates = [all_dates[-1]]

    cmap = plt.cm.YlOrRd
    for i, snap_date in enumerate(snapshot_dates):
        frac = i / max(len(snapshot_dates) - 1, 1)
        curve_x, curve_y = [], []
        for j, tk in enumerate(sorted_tickers):
            valid = contract_data[tk]['hist']
            valid = valid[valid.index <= snap_date]
            if len(valid) > 0:
                curve_x.append(j)
                curve_y.append(valid['Close'].iloc[-1])
        if len(curve_x) >= 2:
            lbl = snap_date.strftime('%b %d') if i in (0, len(snapshot_dates) - 1) else None
            ax.plot(curve_x, curve_y, '-o', markersize=2, color=cmap(0.2 + 0.7 * frac),
                    alpha=0.3 + 0.7 * frac, linewidth=1.0 + 1.5 * frac, label=lbl)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Price ($/MMBtu)', fontsize=11, fontweight='bold')
ax.set_title('Curve Evolution (Weekly Snapshots)', fontsize=13, fontweight='bold')
if snapshot_dates:
    ax.text(0.98, 0.95, f'{snapshot_dates[0].strftime("%b %d")} -> {snapshot_dates[-1].strftime("%b %d")}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

# ============================================
# Row 2, Left: NG=F 10yr History + Trend Fits
# ============================================
ax = axes[1, 0]
ax.plot(ng_long_hist.index, ng_long_hist['Close'], color='#555555', linewidth=0.8, alpha=0.7, label='NG=F daily')
ax.plot(ng_long_hist.index, exp_trend_all, color='#D32F2F', linewidth=2.5, linestyle='--',
        label=f'Exp trend ({annual_drift_pct:+.1f}%/yr)')
ax.plot(ng_long_hist.index, lin_trend_all, color='#1565C0', linewidth=2, linestyle=':',
        label=f'Linear trend ({lin_coeffs[0]*365.25:+.2f} $/yr)')
ax.plot(ng_long_hist.index, rolling_3yr, color='#2E7D32', linewidth=2, alpha=0.8,
        label='3yr rolling median')

# Mark current front month
ax.scatter([ng_long_hist.index[-1]], [ng_long_hist['Close'].iloc[-1]], color='#D32F2F',
           s=100, zorder=5, edgecolors='black', linewidth=1)
ax.annotate(f'${ng_long_hist["Close"].iloc[-1]:.2f}',
            xy=(ng_long_hist.index[-1], ng_long_hist['Close'].iloc[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#D32F2F')

ax.set_ylabel('Price ($/MMBtu)', fontsize=11, fontweight='bold')
ax.set_title('NG=F 10yr History with Trend Models', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Annotate key structural events
ax.annotate('COVID\ncrash', xy=(pd.Timestamp('2020-06-01'), 1.5),
            fontsize=8, ha='center', color='gray', style='italic')
ax.annotate('2022\nspike', xy=(pd.Timestamp('2022-08-01'), 9.0),
            fontsize=8, ha='center', color='gray', style='italic')

# ============================================
# Row 2, Right: Detrended Price (ratio to exponential trend)
# ============================================
ax = axes[1, 1]
detrended = ng_long_hist['detrended_exp'].dropna()
ax.plot(detrended.index, detrended.values, color='#555555', linewidth=0.8, alpha=0.7)
ax.axhline(y=1.0, color='black', linewidth=1.5, linestyle='-', alpha=0.6, label='On trend (1.0x)')
ax.axhline(y=detrended.quantile(0.75), color='#D32F2F', linewidth=1, linestyle=':', alpha=0.5, label='75th pct')
ax.axhline(y=detrended.quantile(0.25), color='#2E7D32', linewidth=1, linestyle=':', alpha=0.5, label='25th pct')

# Fill zones
ax.fill_between(detrended.index, detrended.quantile(0.25), detrended.quantile(0.75),
                alpha=0.08, color='steelblue')

# Mark current
current_detrended = ng_long_hist['detrended_exp'].iloc[-1]
current_pct_global = (detrended < current_detrended).mean() * 100
ax.scatter([detrended.index[-1]], [current_detrended], color='#D32F2F', s=100, zorder=5,
           edgecolors='black', linewidth=1)
ax.annotate(f'{current_detrended:.2f}x trend\n({current_pct_global:.0f}th pct)',
            xy=(detrended.index[-1], current_detrended),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#D32F2F')

ax.set_ylabel('Price / Trend Ratio', fontsize=11, fontweight='bold')
ax.set_title('NG Detrended (Actual / Exponential Trend)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ============================================
# Row 3, Left: Current Curve vs Trend-Adjusted Percentile Bands
# ============================================
ax = axes[2, 0]
x_pos = list(range(len(sorted_tickers)))

# Build trend-projected "fair value" curve (exp trend extrapolated to each delivery month)
trend_curve = [r['trend_price'] for r in rich_cheap]

# Build adjusted percentile bands (rescale back to dollar terms)
adj_p10, adj_p25, adj_p50, adj_p75, adj_p90 = [], [], [], [], []
for i, tk in enumerate(sorted_tickers):
    m = contract_data[tk]['month']
    tp = rich_cheap[i]['trend_price']
    if m in hist_percentiles_adj:
        hp = hist_percentiles_adj[m]
        adj_p10.append(hp['p10'] * tp)
        adj_p25.append(hp['p25'] * tp)
        adj_p50.append(hp['p50'] * tp)
        adj_p75.append(hp['p75'] * tp)
        adj_p90.append(hp['p90'] * tp)
    else:
        adj_p10.append(np.nan); adj_p25.append(np.nan); adj_p50.append(np.nan)
        adj_p75.append(np.nan); adj_p90.append(np.nan)

ax.fill_between(x_pos, adj_p10, adj_p90, alpha=0.12, color='#D32F2F', label='Trend-adj 10-90th')
ax.fill_between(x_pos, adj_p25, adj_p75, alpha=0.22, color='#D32F2F', label='Trend-adj 25-75th')
ax.plot(x_pos, adj_p50, '--', color='#D32F2F', linewidth=1.5, alpha=0.7, label='Trend-adj median')
ax.plot(x_pos, trend_curve, ':', color='#2E7D32', linewidth=2, alpha=0.8, label='Exp trend projection')

# Overlay current curve
for i, r in enumerate(rich_cheap):
    pct = r['adj_pct']
    if pct >= 70:
        c = '#D32F2F'
    elif pct <= 30:
        c = '#2E7D32'
    else:
        c = '#FF8F00'
    ax.scatter(i, r['price'], color=c, s=60, zorder=5, edgecolors='black', linewidth=0.5)

ax.plot(x_pos, prices, '-', color='black', linewidth=2, alpha=0.8, label='Current curve')

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Price ($/MMBtu)', fontsize=11, fontweight='bold')
ax.set_title(f'Current Curve vs Trend-Adjusted Percentiles ({annual_drift_pct:+.1f}%/yr bias removed)',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9, ncol=2)

# ============================================
# Row 3, Right: Raw vs Trend-Adjusted Percentile Comparison
# ============================================
ax = axes[2, 1]
if rich_cheap:
    rc_labels = [r['label'] for r in rich_cheap]
    raw_pcts = [r['raw_pct'] for r in rich_cheap]
    adj_pcts = [r['adj_pct'] for r in rich_cheap]

    x = np.arange(len(rc_labels))
    width = 0.38

    bars_raw = ax.bar(x - width/2, raw_pcts, width, color='#B71C1C', alpha=0.5, label='Raw 10yr')
    bars_adj = ax.bar(x + width/2, adj_pcts, width, color='#1565C0', alpha=0.7, label='Trend-adjusted')

    ax.axhline(y=50, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.axhline(y=30, color='#2E7D32', linewidth=0.8, linestyle=':', alpha=0.4)
    ax.axhline(y=70, color='#D32F2F', linewidth=0.8, linestyle=':', alpha=0.4)

    # Label the adjusted bars
    for i, (raw, adj) in enumerate(zip(raw_pcts, adj_pcts)):
        ax.text(i + width/2, adj + 1.5, f'{adj:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold',
                color='#1565C0')

    ax.set_xticks(x)
    ax.set_xticklabels(rc_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Percentile Rank', fontsize=11, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.set_title('Raw vs Trend-Adjusted Rich/Cheap', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Summary box
    avg_raw = np.nanmean(raw_pcts)
    avg_adj = np.nanmean(adj_pcts)
    ax.text(0.02, 0.95, f'Raw avg: {avg_raw:.0f}th pct\nTrend-adj avg: {avg_adj:.0f}th pct\nDrift: {annual_drift_pct:+.1f}%/yr',
            transform=ax.transAxes, ha='left', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

# ============================================
# Row 4, Left: Calendar Spreads Over Time
# ============================================
ax = axes[3, 0]
spread_colors = {'m1m2': '#B71C1C', 'winter_summer': '#1565C0', 'dec_front': '#2E7D32'}

if m1_ticker and m2_ticker:
    m1_hist = contract_data[m1_ticker]['hist']['Close']
    m2_hist = contract_data[m2_ticker]['hist']['Close']
    common_idx = m1_hist.index.intersection(m2_hist.index)
    if len(common_idx) > 0:
        spread = m1_hist.loc[common_idx] - m2_hist.loc[common_idx]
        ax.plot(common_idx, spread, color=spread_colors['m1m2'], linewidth=2,
                label=f'M1-M2 ({contract_data[m1_ticker]["label"]}-{contract_data[m2_ticker]["label"]})')
        ax.annotate(f'{spread.iloc[-1]:+.3f}', xy=(common_idx[-1], spread.iloc[-1]),
                    xytext=(8, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=spread_colors['m1m2'], va='center')

if dec26_ticker and jun26_ticker:
    dec_hist = contract_data[dec26_ticker]['hist']['Close']
    jun_hist = contract_data[jun26_ticker]['hist']['Close']
    common_idx = dec_hist.index.intersection(jun_hist.index)
    if len(common_idx) > 0:
        spread = dec_hist.loc[common_idx] - jun_hist.loc[common_idx]
        ax.plot(common_idx, spread, color=spread_colors['winter_summer'], linewidth=2,
                label='Dec26-Jun26 (winter-summer)')
        ax.annotate(f'{spread.iloc[-1]:+.3f}', xy=(common_idx[-1], spread.iloc[-1]),
                    xytext=(8, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=spread_colors['winter_summer'], va='center')
        if len(hist_winter_summer_spreads) > 3:
            ws_med = hist_winter_summer_spreads.median()
            ws_p25 = hist_winter_summer_spreads.quantile(0.25)
            ws_p75 = hist_winter_summer_spreads.quantile(0.75)
            ax.axhspan(ws_p25, ws_p75, alpha=0.08, color='steelblue')
            ax.axhline(y=ws_med, color='steelblue', linewidth=1, linestyle=':', alpha=0.5,
                       label=f'Hist Dec-Jun median: ${ws_med:.2f}')

if dec26_ticker and m1_ticker and dec26_ticker != m1_ticker:
    dec_hist = contract_data[dec26_ticker]['hist']['Close']
    front_hist = contract_data[m1_ticker]['hist']['Close']
    common_idx = dec_hist.index.intersection(front_hist.index)
    if len(common_idx) > 0:
        spread = dec_hist.loc[common_idx] - front_hist.loc[common_idx]
        ax.plot(common_idx, spread, color=spread_colors['dec_front'], linewidth=2,
                label=f'Dec26-{contract_data[m1_ticker]["label"]} premium')
        ax.annotate(f'{spread.iloc[-1]:+.3f}', xy=(common_idx[-1], spread.iloc[-1]),
                    xytext=(8, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=spread_colors['dec_front'], va='center')

ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_ylabel('Spread ($/MMBtu)', fontsize=11, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_title('Calendar Spreads Over Time', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=8, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

# ============================================
# Row 4, Right: Individual Contracts vs Continuous
# ============================================
ax = axes[3, 1]
key_contracts = []
if m1_ticker:
    key_contracts.append((m1_ticker, '#B71C1C', 2.0, '-'))
if shoulder_ticker:
    key_contracts.append((shoulder_ticker, '#FF8F00', 2.0, '-'))
if next_winter_ticker and next_winter_ticker != m1_ticker:
    key_contracts.append((next_winter_ticker, '#1565C0', 2.0, '-'))
if m2_ticker and m2_ticker not in [t[0] for t in key_contracts]:
    key_contracts.append((m2_ticker, '#6A0DAD', 1.8, '-'))

for tk, color, lw, ls in key_contracts:
    hist = contract_data[tk]['hist']
    ax.plot(hist.index, hist['Close'], color=color, linewidth=lw, linestyle=ls,
            label=contract_data[tk]['label'])
    ax.annotate(f'${hist["Close"].iloc[-1]:.3f}', xy=(hist.index[-1], hist['Close'].iloc[-1]),
                xytext=(8, 0), textcoords='offset points',
                fontsize=9, fontweight='bold', color=color, va='center')

if len(ng_cont_hist) > 0:
    ax.plot(ng_cont_hist.index, ng_cont_hist['Close'], color='gray', linewidth=1.5,
            linestyle='--', alpha=0.6, label='NG=F (continuous)')
    ax.annotate(f'${ng_cont_hist["Close"].iloc[-1]:.3f}',
                xy=(ng_cont_hist.index[-1], ng_cont_hist['Close'].iloc[-1]),
                xytext=(8, -12), textcoords='offset points',
                fontsize=9, fontweight='bold', color='gray', va='center')

ax.set_ylabel('Price ($/MMBtu)', fontsize=11, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_title('Individual Contracts vs Continuous', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

# ============================================
# Save
# ============================================
fig.suptitle('NG Term Structure Analysis (with Inflation/Trend Adjustment)', fontsize=16, fontweight='bold', y=1.003)
plt.tight_layout()
plt.savefig('/home/wyatt/weather/ng_term_structure.png', dpi=150, bbox_inches='tight')
print("\nChart saved as 'ng_term_structure.png'")

# ============================================
# Console Summary
# ============================================
print("\n" + "=" * 65)
print("TERM STRUCTURE SUMMARY")
print("=" * 65)

print(f"\nForward Curve ({len(sorted_tickers)} contracts):")
print(f"  Front month: {contract_data[front_ticker]['label']} = ${front['latest_price']:.3f}")
if len(sorted_tickers) >= 2:
    back = contract_data[sorted_tickers[-1]]
    print(f"  Back month:  {back['label']} = ${back['latest_price']:.3f}")
if len(prices) >= 2:
    avg_near, avg_far = np.mean(prices[:3]), np.mean(prices[-3:])
    structure = "BACKWARDATION" if avg_near > avg_far else "CONTANGO"
    print(f"  Structure: {structure} (near ${avg_near:.3f} vs far ${avg_far:.3f})")

# ============================================
# Trend Analysis
# ============================================
print(f"\n{'='*65}")
print("INFLATION / TREND ANALYSIS")
print(f"{'='*65}")
print(f"\n  Exponential trend: {annual_drift_pct:+.1f}% per year compound")
print(f"  Linear trend: {lin_coeffs[0]*365.25:+.3f} $/yr")
print(f"  Current exp trend value: ${exp_trend_all[-1]:.3f}")
print(f"  3yr rolling median: ${rolling_3yr.dropna().iloc[-1]:.3f}")
print(f"  Current price / trend: {current_detrended:.2f}x ({current_pct_global:.0f}th pct globally)")

# ============================================
# Rich/Cheap: Raw vs Adjusted
# ============================================
print(f"\n{'='*65}")
print("RICH/CHEAP: RAW vs TREND-ADJUSTED")
print(f"{'='*65}")

print(f"\n{'Contract':<10} {'Price':>8} {'Trend':>8} {'Raw%':>6} {'Adj%':>6} {'Raw':>8} {'Adjusted':>10}")
print("-" * 68)
for r in rich_cheap:
    raw_v = "RICH" if r['raw_pct'] >= 70 else "CHEAP" if r['raw_pct'] <= 30 else "FAIR"
    adj_v = "RICH" if r['adj_pct'] >= 70 else "CHEAP" if r['adj_pct'] <= 30 else "FAIR"
    print(f"  {r['label']:<8} ${r['price']:>6.3f}  ${r['trend_price']:>6.3f}  {r['raw_pct']:>5.0f}%  {r['adj_pct']:>5.0f}%  {raw_v:>6}  {adj_v:>8}")

avg_raw = np.nanmean([r['raw_pct'] for r in rich_cheap])
avg_adj = np.nanmean([r['adj_pct'] for r in rich_cheap])
print(f"\n  Overall raw:      {avg_raw:.0f}th percentile")
print(f"  Overall adjusted: {avg_adj:.0f}th percentile")
print(f"  Trend bias:       {avg_raw - avg_adj:+.0f} pct points")

# Season breakdown
for season_name, season_key in [('Winter', 'winter'), ('Summer', 'summer'), ('Shoulder', 'shoulder')]:
    sr = [r for r in rich_cheap if r['season'] == season_key]
    if sr:
        raw_avg = np.nanmean([r['raw_pct'] for r in sr])
        adj_avg = np.nanmean([r['adj_pct'] for r in sr])
        print(f"  {season_name:10s}: raw {raw_avg:.0f}th -> adj {adj_avg:.0f}th (bias: {raw_avg - adj_avg:+.0f})")

# Highlight UNG/Apr position
apr_rc = next((r for r in rich_cheap if r['label'] == 'Apr26'), None)
if apr_rc:
    print(f"\n  >>> YOUR POSITION: UNG (Apr26)")
    print(f"      Price: ${apr_rc['price']:.3f}")
    print(f"      Trend fair value: ${apr_rc['trend_price']:.3f}")
    print(f"      Raw percentile: {apr_rc['raw_pct']:.0f}th (vs all 10yr Aprils)")
    print(f"      Trend-adjusted: {apr_rc['adj_pct']:.0f}th (removing {annual_drift_pct:+.1f}%/yr drift)")
    if apr_rc['adj_pct'] <= 30:
        print(f"      -> CHEAP after trend adjustment")
    elif apr_rc['adj_pct'] <= 50:
        print(f"      -> BELOW MEDIAN after trend adjustment (fair-to-cheap)")
    elif apr_rc['adj_pct'] <= 70:
        print(f"      -> FAIR after trend adjustment")
    else:
        print(f"      -> Still RICH even after trend adjustment")

# Spread context
if dec26_ticker and jun26_ticker:
    current_ws = contract_data[dec26_ticker]['latest_price'] - contract_data[jun26_ticker]['latest_price']
    ws_pct = (hist_winter_summer_spreads < current_ws).mean() * 100
    print(f"\n  Winter-Summer Spread: ${current_ws:.3f} ({ws_pct:.0f}th pct vs 10yr)")

print("\n" + "=" * 65)

plt.show()
