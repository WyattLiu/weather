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
    """Calculate accumulated HDD from ECMWF 6-hourly data."""
    temp_c = ds['t2m'] - 273.15
    steps_hours = [int(s / np.timedelta64(1, 'h')) for s in ds['step'].values]

    daily_hdds = []
    for day in range(4):
        start_h = day * 24
        end_h = (day + 1) * 24
        day_steps = [i for i, h in enumerate(steps_hours) if start_h <= h < end_h]
        if day_steps:
            day_temp = temp_c.isel(step=day_steps).mean(dim='step')
            day_hdd = np.maximum(0, base_temp - day_temp)
            day_hdd_mean = float(day_hdd.mean().values)
            daily_hdds.append(day_hdd_mean)

    return sum(daily_hdds), daily_hdds


def calculate_accumulated_hdd_gfs(ds, base_temp=18.3):
    """Calculate accumulated HDD from GFS daily data."""
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
        n_steps = min(ds.dims['step'], 4)
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

    return sum(daily_hdds[:4]), daily_hdds[:4]


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
# Load UNG/NG hourly prices
# ============================================
print("Fetching NG hourly prices...")
ng = yf.Ticker("NG=F")
ng_hourly = ng.history(period="60d", interval="1h")
print(f"  Loaded {len(ng_hourly)} hourly NG price bars")

# ============================================
# Load ECMWF data
# ============================================
print("\nLoading ECMWF forecasts...")
ecmwf_files = sorted(glob.glob('/home/wyatt/weather/forecast_historical_*z.grib2'))
print(f"  Found {len(ecmwf_files)} ECMWF files")

ecmwf_data = []
for hfile in ecmwf_files:
    basename = os.path.basename(hfile)
    if basename.count('_') < 3:
        continue

    ds_us, forecast_time = load_and_filter_us(hfile, 'ecmwf')
    if ds_us is None:
        continue

    try:
        acc_hdd, _ = calculate_accumulated_hdd_ecmwf(ds_us, base_temp)
        forecast_date = str(forecast_time).split('T')[0]
        forecast_hour = str(forecast_time).split('T')[1][:2]
        release_dt = pd.to_datetime(f"{forecast_date} {forecast_hour}:00")

        ecmwf_data.append({'datetime': release_dt, 'acc_hdd': acc_hdd})
    except:
        continue

ecmwf_data = sorted(ecmwf_data, key=lambda x: x['datetime'])
print(f"  Processed {len(ecmwf_data)} ECMWF forecasts")

# ============================================
# Load GFS data
# ============================================
print("\nLoading GFS forecasts...")
gfs_files = sorted(glob.glob('/home/wyatt/weather/gfs_*.grib2'))
print(f"  Found {len(gfs_files)} GFS files")

gfs_data = []
for gfile in gfs_files:
    basename = os.path.basename(gfile)
    ds_us, forecast_time = load_and_filter_us(gfile, 'gfs')
    if ds_us is None:
        continue

    try:
        acc_hdd, _ = calculate_accumulated_hdd_gfs(ds_us, base_temp)
        if acc_hdd is None:
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
        gfs_data.append({'datetime': release_dt, 'acc_hdd': acc_hdd})
    except:
        continue

gfs_data = sorted(gfs_data, key=lambda x: x['datetime'])
print(f"  Processed {len(gfs_data)} GFS forecasts")

# ============================================
# Create the chart
# ============================================
print("\nCreating chart...")

fig, ax1 = plt.subplots(figsize=(18, 9))

# Determine date range
all_times = [d['datetime'] for d in ecmwf_data] + [d['datetime'] for d in gfs_data]
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

# Right axis: NG prices
ax2 = ax1.twinx()
ax2.plot(ng_times, ng_prices, '-', color='red', linewidth=1.5, alpha=0.5, label='NG Price (hourly)')
ax2.set_ylabel('Natural Gas Price ($/MMBtu)', color='red', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='red')

# Left axis: HDD data
ax1.set_xlabel('Date & Time', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accumulated HDD (4-day total)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot ECMWF HDD (blue)
if ecmwf_data:
    ecmwf_times = [d['datetime'] for d in ecmwf_data]
    ecmwf_hdd = [d['acc_hdd'] for d in ecmwf_data]
    ax1.plot(ecmwf_times, ecmwf_hdd, 'o-', color='blue', markersize=8, linewidth=2,
             alpha=0.8, label=f'ECMWF 4-day HDD', zorder=3)

# Plot GFS HDD (green)
if gfs_data:
    gfs_times = [d['datetime'] for d in gfs_data]
    gfs_hdd = [d['acc_hdd'] for d in gfs_data]
    ax1.plot(gfs_times, gfs_hdd, 's-', color='green', markersize=8, linewidth=2,
             alpha=0.8, label=f'GFS 4-day HDD', zorder=3)

# Add mean lines
if ecmwf_data:
    ecmwf_mean = np.mean([d['acc_hdd'] for d in ecmwf_data])
    ax1.axhline(y=ecmwf_mean, color='blue', linestyle='--', linewidth=1, alpha=0.3)

if gfs_data:
    gfs_mean = np.mean([d['acc_hdd'] for d in gfs_data])
    ax1.axhline(y=gfs_mean, color='green', linestyle='--', linewidth=1, alpha=0.3)

if len(ng_prices) > 0:
    ng_mean = np.mean(ng_prices)
    ax2.axhline(y=ng_mean, color='red', linestyle='--', linewidth=1, alpha=0.3)

# Format x-axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)

# Title and legend
plt.title('ECMWF & GFS Accumulated HDD (4-Day) vs Natural Gas Hourly Prices',
          fontsize=14, fontweight='bold', pad=20)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('/home/wyatt/weather/hdd_ng_comparison.png', dpi=150, bbox_inches='tight')
print("\nChart saved as 'hdd_ng_comparison.png'")

# ============================================
# Print summary
# ============================================
print("\n" + "=" * 65)
print("HDD vs NATURAL GAS SUMMARY")
print("=" * 65)

if ecmwf_data:
    print(f"\nECMWF (4-day HDD):")
    print(f"  Latest: {ecmwf_data[-1]['acc_hdd']:.1f} HDD ({ecmwf_data[-1]['datetime']})")
    print(f"  Mean:   {np.mean([d['acc_hdd'] for d in ecmwf_data]):.1f} HDD")

if gfs_data:
    print(f"\nGFS (4-day HDD):")
    print(f"  Latest: {gfs_data[-1]['acc_hdd']:.1f} HDD ({gfs_data[-1]['datetime']})")
    print(f"  Mean:   {np.mean([d['acc_hdd'] for d in gfs_data]):.1f} HDD")

if len(ng_prices) > 0:
    print(f"\nNG Price:")
    print(f"  Latest: ${ng_prices[-1]:.3f}/MMBtu")
    print(f"  Mean:   ${np.mean(ng_prices):.3f}/MMBtu")

# Correlation
if ecmwf_data and len(ng_prices) > 0:
    ng_at_ecmwf = []
    for d in ecmwf_data:
        dt = d['datetime']
        try:
            diffs = abs(ng_filtered.index.tz_localize(None) - dt)
            if len(diffs) > 0:
                ng_at_ecmwf.append(ng_filtered.iloc[diffs.argmin()]['Close'])
        except:
            pass

    if len(ng_at_ecmwf) == len(ecmwf_data):
        corr = np.corrcoef([d['acc_hdd'] for d in ecmwf_data], ng_at_ecmwf)[0, 1]
        print(f"\nCorrelation (ECMWF HDD vs NG): {corr:.3f}")

print("=" * 65)

plt.show()
