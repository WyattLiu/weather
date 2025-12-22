#!/usr/bin/env python3
"""
GFS Historical HDD Analysis - Plot accumulated HDD over time as a line chart.
Shows trends in heating demand forecasts with NG price overlay.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
from datetime import datetime
import yfinance as yf

print("Loading GFS historical data for HDD trend analysis...")

# Configuration
base_temp = 18.3  # 65°F in Celsius
us_lat_min, us_lat_max = 25, 50
us_lon_min, us_lon_max = -125, -65


def calculate_accumulated_hdd_gfs(ds, base_temp=18.3, max_days=16):
    """Calculate accumulated HDD from GFS forecast."""
    # Find temperature variable
    temp_var = None
    for var in ds.data_vars:
        if 'TMP' in var.upper() or 'T2M' in var.upper() or var == 't':
            temp_var = var
            break

    if temp_var is None:
        raise ValueError(f"No temperature variable found in {list(ds.data_vars)}")

    temp_k = ds[temp_var]
    temp_c = temp_k - 273.15

    # Get number of time steps
    if 'step' in ds.dims:
        n_steps = ds.dims['step']
    elif 'time' in ds.dims:
        n_steps = ds.dims['time']
    else:
        n_steps = 1

    # Calculate daily HDD (each step = 1 day for our historical data)
    daily_hdds = []
    for i in range(min(n_steps, max_days)):
        try:
            if 'step' in ds.dims:
                day_temp = temp_c.isel(step=i)
            elif n_steps > 1:
                day_temp = temp_c.isel(time=i)
            else:
                day_temp = temp_c

            day_hdd = np.maximum(0, base_temp - day_temp)
            day_hdd_mean = float(day_hdd.mean().values)
            daily_hdds.append(day_hdd_mean)
        except Exception:
            break

    if not daily_hdds:
        return None

    return {
        'daily': daily_hdds,
        '4_day': sum(daily_hdds[:4]) if len(daily_hdds) >= 4 else None,
        '7_day': sum(daily_hdds[:7]) if len(daily_hdds) >= 7 else None,
        '14_day': sum(daily_hdds[:14]) if len(daily_hdds) >= 14 else None,
        'total': sum(daily_hdds),
        'num_days': len(daily_hdds)
    }


def load_gfs_file(filepath):
    """Load GFS GRIB2 file and filter to US region."""
    try:
        ds = xr.open_dataset(filepath, engine='cfgrib',
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
    except Exception:
        try:
            ds = xr.open_dataset(filepath, engine='cfgrib')
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            return None, None

    # Get coordinate names
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

    # Convert longitude if needed
    if ds[lon_name].max() > 180:
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)

    # Filter to US region
    ds_us = ds.where(
        (ds[lat_name] >= us_lat_min) & (ds[lat_name] <= us_lat_max) &
        (ds[lon_name] >= us_lon_min) & (ds[lon_name] <= us_lon_max),
        drop=True
    )

    # Get forecast date from filename
    basename = os.path.basename(filepath)
    # Pattern: gfs_historical_YYYYMMDD_HHz.grib2 or gfs_YYYYMMDD_HHz.grib2
    parts = basename.replace('.grib2', '').split('_')
    for part in parts:
        if len(part) == 8 and part.isdigit():
            date_str = part
            break
    else:
        date_str = datetime.now().strftime("%Y%m%d")

    hour_str = "00"
    for part in parts:
        if part.endswith('z') and part[:-1].isdigit():
            hour_str = part[:-1]
            break

    forecast_time = pd.to_datetime(f"{date_str} {hour_str}:00")
    return ds_us, forecast_time


# Find all GFS files (both current and historical)
gfs_files = sorted(glob.glob('/home/wyatt/weather/gfs_*.grib2') +
                   glob.glob('/home/wyatt/weather/gfs_historical_*.grib2'))
print(f"Found {len(gfs_files)} GFS forecast files\n")

if len(gfs_files) == 0:
    print("No GFS data found. Run fetch_gfs.py or fetch_gfs_historical.py first.")
    exit(1)

# Process each file
all_data = []
for gfile in gfs_files:
    basename = os.path.basename(gfile)
    print(f"Processing {basename}...", end=' ', flush=True)

    ds_us, forecast_time = load_gfs_file(gfile)
    if ds_us is None:
        print("SKIP")
        continue

    try:
        hdd_results = calculate_accumulated_hdd_gfs(ds_us, base_temp)
        if hdd_results is None:
            print("NO DATA")
            continue

        all_data.append({
            'datetime': forecast_time,
            'hdd_4day': hdd_results['4_day'],
            'hdd_7day': hdd_results['7_day'],
            'hdd_14day': hdd_results['14_day'],
            'daily': hdd_results['daily'],
            'num_days': hdd_results['num_days']
        })
        print(f"OK (4d:{hdd_results['4_day']:.0f} 7d:{hdd_results['7_day']:.0f} 14d:{hdd_results['14_day']:.0f})" if hdd_results['14_day'] else "OK")

    except Exception as e:
        print(f"ERROR: {e}")

if not all_data:
    print("\nNo data processed.")
    exit(1)

# Sort by date
all_data = sorted(all_data, key=lambda x: x['datetime'])

print(f"\nProcessed {len(all_data)} forecasts")
print(f"Date range: {all_data[0]['datetime'].strftime('%Y-%m-%d')} to {all_data[-1]['datetime'].strftime('%Y-%m-%d')}")

# Fetch NG prices for the same period
print("\nFetching NG price data...")
ng = yf.Ticker("NG=F")
start_date = all_data[0]['datetime'] - pd.Timedelta(days=1)
end_date = all_data[-1]['datetime'] + pd.Timedelta(days=1)

# Get daily NG data for longer time series
ng_daily = ng.history(start=start_date, end=end_date, interval="1d")
print(f"Loaded {len(ng_daily)} daily NG price bars")

# Create the chart
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# Prepare data
dates = [d['datetime'] for d in all_data]
hdd_4day = [d['hdd_4day'] for d in all_data]
hdd_7day = [d['hdd_7day'] for d in all_data]
hdd_14day = [d['hdd_14day'] for d in all_data]

# Top plot: HDD trends
ax1 = axes[0]
ax1.plot(dates, hdd_4day, 'o-', color='blue', linewidth=2, markersize=6, label='4-Day HDD', alpha=0.8)
if any(h is not None for h in hdd_7day):
    valid_dates = [d for d, h in zip(dates, hdd_7day) if h is not None]
    valid_hdd = [h for h in hdd_7day if h is not None]
    ax1.plot(valid_dates, valid_hdd, 's-', color='green', linewidth=2, markersize=5, label='7-Day HDD', alpha=0.8)
if any(h is not None for h in hdd_14day):
    valid_dates = [d for d, h in zip(dates, hdd_14day) if h is not None]
    valid_hdd = [h for h in hdd_14day if h is not None]
    ax1.plot(valid_dates, valid_hdd, '^-', color='orange', linewidth=2, markersize=5, label='14-Day HDD', alpha=0.8)

ax1.set_ylabel('Accumulated HDD', fontsize=12, fontweight='bold')
ax1.set_title('GFS Accumulated HDD Forecast Trends - US Continental\nBase Temperature: 65°F (18.3°C)',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Add mean lines
if hdd_4day:
    ax1.axhline(y=np.mean(hdd_4day), color='blue', linestyle='--', alpha=0.3, label=f'4d Mean: {np.mean(hdd_4day):.0f}')
if any(h is not None for h in hdd_14day):
    valid = [h for h in hdd_14day if h is not None]
    ax1.axhline(y=np.mean(valid), color='orange', linestyle='--', alpha=0.3)

# Bottom plot: NG prices
ax2 = axes[1]
if len(ng_daily) > 0:
    ng_times = ng_daily.index.tz_localize(None) if ng_daily.index.tz else ng_daily.index
    ax2.plot(ng_times, ng_daily['Close'], '-', color='red', linewidth=2, label='NG Price')
    ax2.fill_between(ng_times, ng_daily['Close'], alpha=0.2, color='red')
    ax2.axhline(y=ng_daily['Close'].mean(), color='darkred', linestyle='--', alpha=0.5,
                label=f'Mean: ${ng_daily["Close"].mean():.2f}')

ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Natural Gas Price ($/MMBtu)', fontsize=12, fontweight='bold', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Format x-axis
import matplotlib.dates as mdates
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//20)))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('/home/wyatt/weather/gfs_hdd_history.png', dpi=150, bbox_inches='tight')
print("\nChart saved as 'gfs_hdd_history.png'")

# Calculate correlation
if len(ng_daily) > 0:
    # Match HDD dates with NG prices
    hdd_ng_pairs = []
    for d in all_data:
        dt = d['datetime']
        # Find closest NG price
        if dt in ng_daily.index or pd.Timestamp(dt) in ng_daily.index:
            ng_price = ng_daily.loc[pd.Timestamp(dt), 'Close'] if pd.Timestamp(dt) in ng_daily.index else None
        else:
            # Find nearest date
            diffs = abs(ng_daily.index - pd.Timestamp(dt))
            if len(diffs) > 0:
                nearest_idx = diffs.argmin()
                if diffs[nearest_idx] < pd.Timedelta(days=2):
                    ng_price = ng_daily.iloc[nearest_idx]['Close']
                else:
                    ng_price = None
            else:
                ng_price = None

        if ng_price is not None and d['hdd_14day'] is not None:
            hdd_ng_pairs.append((d['hdd_14day'], ng_price))

    if len(hdd_ng_pairs) > 5:
        hdd_arr = np.array([p[0] for p in hdd_ng_pairs])
        ng_arr = np.array([p[1] for p in hdd_ng_pairs])
        correlation = np.corrcoef(hdd_arr, ng_arr)[0, 1]

        print("\n" + "=" * 65)
        print("GFS HDD vs NATURAL GAS CORRELATION ANALYSIS")
        print("=" * 65)
        print(f"Period: {all_data[0]['datetime'].strftime('%Y-%m-%d')} to {all_data[-1]['datetime'].strftime('%Y-%m-%d')}")
        print(f"Data points: {len(hdd_ng_pairs)}")
        print(f"\n14-Day HDD Statistics:")
        print(f"  Mean:  {np.mean(hdd_arr):.1f} HDD")
        print(f"  Min:   {np.min(hdd_arr):.1f} HDD")
        print(f"  Max:   {np.max(hdd_arr):.1f} HDD")
        print(f"\nNG Price Statistics:")
        print(f"  Mean:  ${np.mean(ng_arr):.2f}/MMBtu")
        print(f"  Min:   ${np.min(ng_arr):.2f}/MMBtu")
        print(f"  Max:   ${np.max(ng_arr):.2f}/MMBtu")
        print(f"\nCorrelation (14-day HDD vs NG): {correlation:.3f}")
        if correlation > 0.5:
            print("  -> Strong positive: Higher HDD -> Higher NG prices")
        elif correlation > 0.2:
            print("  -> Moderate positive correlation")
        elif correlation > -0.2:
            print("  -> Weak/No correlation")
        else:
            print("  -> Negative correlation")
        print("=" * 65)

plt.show()
