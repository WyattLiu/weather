#!/usr/bin/env python3
"""
HDD vs Natural Gas Price Comparison
Shows 3 lines:
1. UNG/NG hourly price (right axis)
2. ECMWF 4-day accumulated HDD (left axis)
3. GFS 4-day accumulated HDD (left axis)
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.dates as mdates

print("Loading HDD and NG price data...")

# Configuration
base_temp = 18.3  # 65°F in Celsius
us_lat_min, us_lat_max = 25, 50
us_lon_min, us_lon_max = -125, -65


def calculate_accumulated_hdd_ecmwf(ds, base_temp=18.3):
    """Calculate accumulated HDD from ECMWF 6-hourly data for multiple timeframes."""
    temp_c = ds['t2m'] - 273.15
    steps_hours = [int(s / np.timedelta64(1, 'h')) for s in ds['step'].values]
    max_hours = max(steps_hours) if steps_hours else 0
    max_days = max_hours // 24

    daily_hdds = []
    for day in range(max_days + 1):
        start_h = day * 24
        end_h = (day + 1) * 24
        day_steps = [i for i, h in enumerate(steps_hours) if start_h <= h < end_h]
        if day_steps:
            day_temp = temp_c.isel(step=day_steps).mean(dim='step')
            day_hdd = np.maximum(0, base_temp - day_temp)
            day_hdd_mean = float(day_hdd.mean().values)
            daily_hdds.append(day_hdd_mean)

    # Return accumulated HDD for different periods (matching GFS)
    return {
        '1-4d': sum(daily_hdds[:4]) if len(daily_hdds) >= 4 else None,
        '1-5d': sum(daily_hdds[:5]) if len(daily_hdds) >= 5 else None,
        '6-10d': sum(daily_hdds[5:10]) if len(daily_hdds) >= 10 else None,
        'daily': daily_hdds
    }


def calculate_accumulated_hdd_gfs(ds, base_temp=18.3, max_days=16):
    """Calculate accumulated HDD from GFS daily data for multiple timeframes."""
    # Find temperature variable
    temp_var = None
    for var in ds.data_vars:
        if 'TMP' in var.upper() or 'T2M' in var.upper() or var == 't':
            temp_var = var
            break
    if temp_var is None:
        return None, []

    temp_k = ds[temp_var]
    temp_c = temp_k - 273.15

    # Get number of steps
    if 'step' in ds.dims:
        n_steps = min(ds.sizes['step'], max_days)
    else:
        n_steps = 1

    daily_hdds = []
    for i in range(n_steps):
        try:
            if 'step' in ds.dims:
                day_temp = temp_c.isel(step=i)
            else:
                day_temp = temp_c
            day_hdd = np.maximum(0, base_temp - day_temp)
            day_hdd_mean = float(day_hdd.mean().values)
            daily_hdds.append(day_hdd_mean)
        except:
            break

    # Return accumulated HDD for different periods
    # Key periods for NG trading: 1-5d (short), 6-10d (key), 8-14d (extended)
    return {
        '1-5d': sum(daily_hdds[:5]) if len(daily_hdds) >= 5 else None,
        '6-10d': sum(daily_hdds[5:10]) if len(daily_hdds) >= 10 else None,
        '8-14d': sum(daily_hdds[7:14]) if len(daily_hdds) >= 14 else None,
        'daily': daily_hdds
    }


def load_and_filter_us(filepath, source='ecmwf'):
    """Load GRIB2 file and filter to US region."""
    try:
        if source == 'gfs':
            ds = xr.open_dataset(filepath, engine='cfgrib',
                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
        else:
            ds = xr.open_dataset(filepath, engine='cfgrib')
    except:
        try:
            ds = xr.open_dataset(filepath, engine='cfgrib')
        except Exception as e:
            return None, None

    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

    if ds[lon_name].max() > 180:
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)

    ds_us = ds.where(
        (ds[lat_name] >= us_lat_min) & (ds[lat_name] <= us_lat_max) &
        (ds[lon_name] >= us_lon_min) & (ds[lon_name] <= us_lon_max),
        drop=True
    )

    if 'time' in ds.coords:
        forecast_time = ds['time'].values
    else:
        forecast_time = None

    return ds_us, forecast_time


# ============================================
# Load UNG/NG hourly prices + TTF + EUR/USD
# ============================================
print("Fetching NG hourly prices...")
ng = yf.Ticker("NG=F")
ng_hourly = ng.history(period="60d", interval="1h")
print(f"  Loaded {len(ng_hourly)} hourly NG price bars")

print("Fetching TTF (EUR/MWh) and EUR/USD...")
ttf_ticker = yf.Ticker("TTF=F")
ttf_hourly = ttf_ticker.history(period="60d", interval="1h")
eurusd_ticker = yf.Ticker("EURUSD=X")
eurusd_hourly = eurusd_ticker.history(period="60d", interval="1h")
print(f"  Loaded {len(ttf_hourly)} TTF bars, {len(eurusd_hourly)} EUR/USD bars")

# Convert TTF from EUR/MWh to USD/MMBtu
# 1 MWh = 3.412 MMBtu
MWH_PER_MMBTU = 3.412
if len(ttf_hourly) > 0 and len(eurusd_hourly) > 0:
    ttf_times_raw = ttf_hourly.index.tz_localize(None) if ttf_hourly.index.tz else ttf_hourly.index
    eurusd_times_raw = eurusd_hourly.index.tz_localize(None) if eurusd_hourly.index.tz else eurusd_hourly.index

    # Align EUR/USD to TTF timestamps via nearest match
    ttf_usd_mmbtu = []
    ttf_times_aligned = []
    for i, t in enumerate(ttf_times_raw):
        diffs = abs(eurusd_times_raw - t)
        nearest_idx = diffs.argmin()
        if diffs[nearest_idx] < pd.Timedelta(hours=6):
            eur_price = ttf_hourly['Close'].iloc[i]
            fx_rate = eurusd_hourly['Close'].iloc[nearest_idx]
            usd_mmbtu = eur_price * fx_rate / MWH_PER_MMBTU
            ttf_usd_mmbtu.append(usd_mmbtu)
            ttf_times_aligned.append(t)

    ttf_usd_mmbtu = np.array(ttf_usd_mmbtu)
    print(f"  TTF converted: {len(ttf_usd_mmbtu)} points, latest=${ttf_usd_mmbtu[-1]:.3f}/MMBtu")
else:
    ttf_usd_mmbtu = np.array([])
    ttf_times_aligned = []

# ============================================
# Load ECMWF data
# ============================================
print("\nLoading ECMWF forecasts...")
ecmwf_files = sorted(glob.glob('/home/wyatt/weather/forecast_historical_*z.grib2'))
print(f"  Found {len(ecmwf_files)} ECMWF files")

ecmwf_data = {'1-4d': [], '6-10d': []}
for hfile in ecmwf_files:
    basename = os.path.basename(hfile)
    if basename.count('_') < 3:
        continue

    ds_us, forecast_time = load_and_filter_us(hfile, 'ecmwf')
    if ds_us is None:
        continue

    try:
        hdd_results = calculate_accumulated_hdd_ecmwf(ds_us, base_temp)
        forecast_date = str(forecast_time).split('T')[0]
        forecast_hour = str(forecast_time).split('T')[1][:2]
        release_dt = pd.to_datetime(f"{forecast_date} {forecast_hour}:00")

        if hdd_results['1-4d'] is not None:
            ecmwf_data['1-4d'].append({'datetime': release_dt, 'acc_hdd': hdd_results['1-4d']})
        if hdd_results['6-10d'] is not None:
            ecmwf_data['6-10d'].append({'datetime': release_dt, 'acc_hdd': hdd_results['6-10d']})
    except:
        continue

for period in ecmwf_data:
    ecmwf_data[period] = sorted(ecmwf_data[period], key=lambda x: x['datetime'])
print(f"  Processed ECMWF forecasts: 1-4d={len(ecmwf_data['1-4d'])}, 6-10d={len(ecmwf_data['6-10d'])}")

# ============================================
# Load GFS data (all cycles, will smooth for visualization)
# ============================================
print("\nLoading GFS forecasts...")
all_gfs_files = sorted(glob.glob('/home/wyatt/weather/gfs_*.grib2'))
gfs_files = all_gfs_files
print(f"  Found {len(gfs_files)} GFS files")

gfs_data = {'1-5d': [], '6-10d': [], '8-14d': []}
for gfile in gfs_files:
    basename = os.path.basename(gfile)
    ds_us, forecast_time = load_and_filter_us(gfile, 'gfs')
    if ds_us is None:
        continue

    try:
        hdd_results = calculate_accumulated_hdd_gfs(ds_us, base_temp)
        if hdd_results is None:
            continue

        # Parse date from filename
        parts = basename.replace('.grib2', '').split('_')
        for part in parts:
            if len(part) == 8 and part.isdigit():
                date_str = part
                break
        else:
            continue

        hour_str = "00"
        for part in parts:
            if part.endswith('z') and part[:-1].isdigit():
                hour_str = part[:-1]
                break

        release_dt = pd.to_datetime(f"{date_str} {hour_str}:00")

        # Store each timeframe separately
        for period in ['1-5d', '6-10d', '8-14d']:
            if hdd_results[period] is not None:
                gfs_data[period].append({'datetime': release_dt, 'acc_hdd': hdd_results[period]})
    except:
        continue

# Sort each timeframe
for period in gfs_data:
    gfs_data[period] = sorted(gfs_data[period], key=lambda x: x['datetime'])
print(f"  Processed GFS forecasts: 1-5d={len(gfs_data['1-5d'])}, 6-10d={len(gfs_data['6-10d'])}, 8-14d={len(gfs_data['8-14d'])}")

# ============================================
# Create the chart - overlay (dual axis)
# ============================================
print("\nCreating chart...")

plt.style.use('seaborn-v0_8-whitegrid')

fig, ax1 = plt.subplots(figsize=(20, 10))

# Determine date range
all_times = []
for period in ecmwf_data:
    all_times += [d['datetime'] for d in ecmwf_data[period]]
for period in gfs_data:
    all_times += [d['datetime'] for d in gfs_data[period]]
if all_times:
    start_time = min(all_times) - pd.Timedelta(days=1)
    end_time = max(all_times) + pd.Timedelta(days=1)

    # Filter NG data
    if ng_hourly.index.tz is not None:
        start_tz = start_time.tz_localize('America/New_York')
        end_tz = end_time.tz_localize('America/New_York')
        ng_filtered = ng_hourly[(ng_hourly.index >= start_tz) & (ng_hourly.index <= end_tz)]
    else:
        ng_filtered = ng_hourly[(ng_hourly.index >= start_time) & (ng_hourly.index <= end_time)]

    ng_times = ng_filtered.index.tz_localize(None) if ng_filtered.index.tz else ng_filtered.index
    ng_prices = ng_filtered['Close'].values
else:
    ng_times = ng_hourly.index.tz_localize(None) if ng_hourly.index.tz else ng_hourly.index
    ng_prices = ng_hourly['Close'].values

# ---- Color palette (distinct, no conflicts) ----
colors = {
    'ecmwf_14d': '#6A0DAD',  # Deep purple
    'ecmwf_610d': '#E040FB',  # Pink-purple
    'gfs_15d': '#1565C0',     # Dark blue
    'gfs_610d': '#2E7D32',    # Forest green
    'gfs_814d': '#E65100',    # Deep orange
    'ng_price': '#B71C1C',    # Dark red
    'ttf_price': '#FF6F00',   # Amber
}

# Normalize HDD by period (daily average HDD)
ecmwf_days = {'1-4d': 4, '6-10d': 5}
gfs_days = {'1-5d': 5, '6-10d': 5, '8-14d': 7}

# ---- Right axis #1: Henry Hub NG Price ----
ax2 = ax1.twinx()
ax2.plot(ng_times, ng_prices, '-', color=colors['ng_price'], linewidth=1.3, alpha=0.5, label='HH Futures (1h)')
ax2.set_ylabel('Henry Hub ($/MMBtu)', color=colors['ng_price'], fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=colors['ng_price'], labelsize=10)

if len(ng_prices) > 0:
    ng_pad = (max(ng_prices) - min(ng_prices)) * 0.1
    ax2.set_ylim(min(ng_prices) - ng_pad, max(ng_prices) + ng_pad)
    ax2.annotate(f'HH ${ng_prices[-1]:.3f}', xy=(ng_times[-1], ng_prices[-1]),
                 xytext=(8, 0), textcoords='offset points',
                 fontsize=9, fontweight='bold', color=colors['ng_price'],
                 va='center', zorder=5)

# ---- Right axis #2: TTF Price (offset to the right) ----
if len(ttf_usd_mmbtu) > 0:
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.08))
    ax3.plot(ttf_times_aligned, ttf_usd_mmbtu, '-', color=colors['ttf_price'],
             linewidth=1.3, alpha=0.55, label='TTF (USD/MMBtu)')
    ax3.set_ylabel('TTF ($/MMBtu)', color=colors['ttf_price'], fontsize=11, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=colors['ttf_price'], labelsize=10)
    ttf_pad = (max(ttf_usd_mmbtu) - min(ttf_usd_mmbtu)) * 0.1
    ax3.set_ylim(min(ttf_usd_mmbtu) - ttf_pad, max(ttf_usd_mmbtu) + ttf_pad)
    ax3.annotate(f'TTF ${ttf_usd_mmbtu[-1]:.2f}', xy=(ttf_times_aligned[-1], ttf_usd_mmbtu[-1]),
                 xytext=(8, 0), textcoords='offset points',
                 fontsize=9, fontweight='bold', color=colors['ttf_price'],
                 va='center', zorder=5)

# ---- Left axis: HDD Forecasts ----
ax1.set_ylabel('Avg HDD / day', fontsize=12, fontweight='bold')
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# Plot ECMWF HDD
ecmwf_style = {
    '1-4d': {'color': colors['ecmwf_14d'], 'label': 'ECMWF 1-4d', 'lw': 2.0, 'marker': 'o', 'ms': 3},
    '6-10d': {'color': colors['ecmwf_610d'], 'label': 'ECMWF 6-10d', 'lw': 1.8, 'marker': 's', 'ms': 3},
}
for period in ['1-4d', '6-10d']:
    if ecmwf_data[period]:
        times = [d['datetime'] for d in ecmwf_data[period]]
        hdd = [d['acc_hdd'] / ecmwf_days[period] for d in ecmwf_data[period]]
        s = ecmwf_style[period]
        ax1.plot(times, hdd, marker=s['marker'], markersize=s['ms'], linewidth=s['lw'],
                 color=s['color'], alpha=0.85, label=s['label'], zorder=4)
        ax1.annotate(f'{hdd[-1]:.1f}', xy=(times[-1], hdd[-1]),
                     xytext=(8, 0), textcoords='offset points',
                     fontsize=9, fontweight='bold', color=s['color'],
                     va='center', zorder=5)

# Plot GFS HDD
gfs_style = {
    '1-5d': {'color': colors['gfs_15d'], 'label': 'GFS 1-5d', 'lw': 1.8, 'ls': '-'},
    '6-10d': {'color': colors['gfs_610d'], 'label': 'GFS 6-10d (KEY)', 'lw': 2.8, 'ls': '-'},
    '8-14d': {'color': colors['gfs_814d'], 'label': 'GFS 8-14d', 'lw': 1.8, 'ls': '--'},
}
for period in ['1-5d', '6-10d', '8-14d']:
    if gfs_data[period]:
        times = [d['datetime'] for d in gfs_data[period]]
        hdd = [d['acc_hdd'] / gfs_days[period] for d in gfs_data[period]]
        s = gfs_style[period]

        # Smooth if enough points
        if len(hdd) >= 4:
            df = pd.DataFrame({'time': times, 'hdd': hdd}).set_index('time')
            smoothed = df['hdd'].rolling(window=4, center=True, min_periods=2).mean()
            plot_times, plot_hdd = smoothed.index, smoothed.values
        else:
            plot_times, plot_hdd = times, hdd

        ax1.plot(plot_times, plot_hdd, linestyle=s['ls'], linewidth=s['lw'],
                 color=s['color'], alpha=0.85, label=s['label'], zorder=3)
        last_val = plot_hdd[-1] if isinstance(plot_hdd, list) else float(plot_hdd[-1])
        last_time = plot_times[-1] if isinstance(plot_times, list) else plot_times[-1]
        ax1.annotate(f'{last_val:.1f}', xy=(last_time, last_val),
                     xytext=(8, 0), textcoords='offset points',
                     fontsize=9, fontweight='bold', color=s['color'],
                     va='center', zorder=5)

# ---- X-axis formatting ----
if all_times:
    span_days = (end_time - start_time).days
    if span_days > 45:
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
    elif span_days > 14:
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
    else:
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
ax1.set_xlabel('Date', fontsize=11)

# ---- Legend (combine all axes) ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines_all, labels_all = lines1 + lines2, labels1 + labels2
if len(ttf_usd_mmbtu) > 0:
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines_all += lines3
    labels_all += labels3
ax1.legend(lines_all, labels_all, loc='upper left', fontsize=10, framealpha=0.9, ncol=3)

ax1.set_title('HDD Forecasts (ECMWF & GFS) vs Natural Gas Futures',
              fontsize=15, fontweight='bold', pad=12)
ax1.margins(x=0.02)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/wyatt/weather/hdd_ng_comparison.png', dpi=150, bbox_inches='tight')
print("\nChart saved as 'hdd_ng_comparison.png'")

# ============================================
# Print summary
# ============================================
print("\n" + "=" * 65)
print("HDD vs NATURAL GAS SUMMARY")
print("=" * 65)

print(f"\nECMWF Daily Avg HDD:")
for period, label, days in [('1-4d', '1-4 day', 4), ('6-10d', '6-10 day', 5)]:
    if ecmwf_data[period]:
        latest = ecmwf_data[period][-1]['acc_hdd'] / days
        mean_val = np.mean([d['acc_hdd'] / days for d in ecmwf_data[period]])
        print(f"  {label}: {latest:.1f} HDD/day (latest), {mean_val:.1f} HDD/day (mean)")

print(f"\nGFS Daily Avg HDD (key trading windows):")
for period, label, days in [('1-5d', '1-5 day (priced in)', 5), ('6-10d', '6-10 day (KEY)', 5), ('8-14d', '8-14 day (extended)', 7)]:
    if gfs_data[period]:
        latest = gfs_data[period][-1]['acc_hdd'] / days
        mean_val = np.mean([d['acc_hdd'] / days for d in gfs_data[period]])
        print(f"  {label}: {latest:.1f} HDD/day (latest), {mean_val:.1f} HDD/day (mean)")

if len(ng_prices) > 0:
    print(f"\nNG Price (Henry Hub):")
    print(f"  Latest: ${ng_prices[-1]:.3f}/MMBtu")
    print(f"  Mean:   ${np.mean(ng_prices):.3f}/MMBtu")

if len(ttf_usd_mmbtu) > 0:
    print(f"\nTTF Price (converted to USD/MMBtu):")
    print(f"  Latest: ${ttf_usd_mmbtu[-1]:.3f}/MMBtu")
    print(f"  Mean:   ${np.mean(ttf_usd_mmbtu):.3f}/MMBtu")
    if len(ng_prices) > 0:
        spread = ttf_usd_mmbtu[-1] - ng_prices[-1]
        print(f"  TTF-HH Spread: ${spread:.3f}/MMBtu")

# Correlation
if ecmwf_data['1-4d'] and len(ng_prices) > 0:
    ng_at_ecmwf = []
    for d in ecmwf_data['1-4d']:
        dt = d['datetime']
        try:
            diffs = abs(ng_filtered.index.tz_localize(None) - dt)
            if len(diffs) > 0:
                ng_at_ecmwf.append(ng_filtered.iloc[diffs.argmin()]['Close'])
        except:
            pass

    if len(ng_at_ecmwf) == len(ecmwf_data['1-4d']):
        corr = np.corrcoef([d['acc_hdd'] for d in ecmwf_data['1-4d']], ng_at_ecmwf)[0, 1]
        print(f"\nCorrelation (ECMWF 1-4d HDD vs NG): {corr:.3f}")

print("=" * 65)

plt.show()
