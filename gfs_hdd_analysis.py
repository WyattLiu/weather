#!/usr/bin/env python3
"""
GFS HDD Analysis - Calculate accumulated HDD from GFS 16-day forecast.
Produces the same style output as ECMWF HDD analysis for comparison.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
from datetime import datetime
import yfinance as yf

print("Loading GFS forecast data for HDD analysis...")

# Configuration
base_temp = 18.3  # 65°F in Celsius

# US bounding box
us_lat_min, us_lat_max = 25, 50
us_lon_min, us_lon_max = -125, -65  # Will convert GFS 0-360 to -180-180


def calculate_accumulated_hdd_gfs(ds, base_temp=18.3, max_days=16):
    """
    Calculate accumulated HDD from GFS forecast.
    GFS provides daily snapshots, so we calculate HDD for each day.
    Returns: dict with HDD by period (4-day, 7-day, 10-day, 14-day, 16-day)
    """
    # GFS uses TMP variable name
    if 'TMP_P0_L103_GLL0' in ds.data_vars:
        temp_k = ds['TMP_P0_L103_GLL0']  # Temperature in Kelvin
    elif 't2m' in ds.data_vars:
        temp_k = ds['t2m']
    elif 'TMP' in ds.data_vars:
        temp_k = ds['TMP']
    else:
        # Try to find temperature variable
        temp_vars = [v for v in ds.data_vars if 'TMP' in v.upper() or 'T2M' in v.upper()]
        if temp_vars:
            temp_k = ds[temp_vars[0]]
        else:
            raise ValueError(f"Cannot find temperature variable. Available: {list(ds.data_vars)}")

    temp_c = temp_k - 273.15

    # Get forecast steps
    if 'step' in ds.coords:
        steps = ds['step'].values
        steps_hours = [int(s / np.timedelta64(1, 'h')) for s in steps]
    elif 'forecast_time0' in ds.coords:
        steps_hours = list(range(0, (max_days + 1) * 24, 24))
    else:
        # Assume daily data
        steps_hours = list(range(0, len(temp_c) * 24, 24))

    # Calculate daily HDD
    daily_hdds = []
    for i, hours in enumerate(steps_hours):
        day = hours // 24
        if day >= max_days:
            break

        if 'step' in ds.coords:
            day_temp = temp_c.isel(step=i)
        else:
            day_temp = temp_c.isel(time=i) if 'time' in ds.dims else temp_c

        day_hdd = np.maximum(0, base_temp - day_temp)
        day_hdd_mean = float(day_hdd.mean().values)
        daily_hdds.append(day_hdd_mean)

    # Calculate accumulated HDD for different periods
    results = {
        'daily': daily_hdds,
        '4_day': sum(daily_hdds[:4]) if len(daily_hdds) >= 4 else None,
        '7_day': sum(daily_hdds[:7]) if len(daily_hdds) >= 7 else None,
        '10_day': sum(daily_hdds[:10]) if len(daily_hdds) >= 10 else None,
        '14_day': sum(daily_hdds[:14]) if len(daily_hdds) >= 14 else None,
        '16_day': sum(daily_hdds[:16]) if len(daily_hdds) >= 16 else None,
        'total': sum(daily_hdds),
        'num_days': len(daily_hdds)
    }

    return results


def load_gfs_file(filepath):
    """Load GFS GRIB2 file and filter to US region."""
    try:
        # Try cfgrib first
        ds = xr.open_dataset(filepath, engine='cfgrib')
    except Exception:
        try:
            # Try with filter_by_keys for specific level
            ds = xr.open_dataset(
                filepath,
                engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}}
            )
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None

    # Get coordinate names
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

    # Convert longitude from 0-360 to -180-180 if needed
    if ds[lon_name].max() > 180:
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)

    # Filter to US region
    ds_us = ds.where(
        (ds[lat_name] >= us_lat_min) & (ds[lat_name] <= us_lat_max) &
        (ds[lon_name] >= us_lon_min) & (ds[lon_name] <= us_lon_max),
        drop=True
    )

    # Extract time info from filename or dataset
    if 'time' in ds.coords:
        forecast_time = ds['time'].values
        if hasattr(forecast_time, '__iter__') and len(forecast_time) > 0:
            forecast_time = forecast_time[0] if len(forecast_time.shape) > 0 else forecast_time
    else:
        # Parse from filename: gfs_YYYYMMDD_HHz.grib2
        basename = os.path.basename(filepath)
        parts = basename.replace('.grib2', '').split('_')
        if len(parts) >= 3:
            date_str = parts[1]
            hour_str = parts[2].replace('z', '')
            forecast_time = pd.to_datetime(f"{date_str} {hour_str}:00")
        else:
            forecast_time = datetime.now()

    return ds_us, forecast_time


# Find all GFS files
gfs_files = sorted(glob.glob('/home/wyatt/weather/gfs_*.grib2'))
print(f"Found {len(gfs_files)} GFS forecast files\n")

if len(gfs_files) == 0:
    print("No GFS data found. Run fetch_gfs.py first.")
    exit(1)

# Process each file
all_data = []
for gfile in gfs_files:
    print(f"Processing {os.path.basename(gfile)}...")

    ds_us, forecast_time = load_gfs_file(gfile)
    if ds_us is None:
        continue

    try:
        hdd_results = calculate_accumulated_hdd_gfs(ds_us, base_temp)

        # Parse time for labeling
        if isinstance(forecast_time, (np.datetime64, pd.Timestamp)):
            dt = pd.to_datetime(forecast_time)
        else:
            dt = forecast_time

        all_data.append({
            'file': os.path.basename(gfile),
            'datetime': dt,
            'hdd_4day': hdd_results['4_day'],
            'hdd_7day': hdd_results['7_day'],
            'hdd_10day': hdd_results['10_day'],
            'hdd_14day': hdd_results['14_day'],
            'hdd_16day': hdd_results['16_day'],
            'daily_hdds': hdd_results['daily'],
            'num_days': hdd_results['num_days']
        })

        print(f"  4-day: {hdd_results['4_day']:.1f} HDD | 7-day: {hdd_results['7_day']:.1f} HDD | "
              f"14-day: {hdd_results['14_day']:.1f} HDD | 16-day: {hdd_results['16_day']:.1f} HDD")

    except Exception as e:
        print(f"  Error: {e}")

if not all_data:
    print("\nNo data processed successfully.")
    exit(1)

# Sort by datetime
all_data = sorted(all_data, key=lambda x: x['datetime'])

# Get latest forecast for detailed output
latest = all_data[-1]

print("\n" + "=" * 70)
print("GFS ACCUMULATED HDD SUMMARY - US CONTINENTAL")
print("=" * 70)
print(f"Base Temperature: 65°F (18.3°C)")
print(f"Region: {us_lat_min}°-{us_lat_max}°N, {abs(us_lon_min)}°-{abs(us_lon_max)}°W")
print(f"Latest Forecast: {latest['datetime']}")
print(f"Forecast Days: {latest['num_days']}")
print()

print("Daily HDD Breakdown:")
for i, hdd in enumerate(latest['daily_hdds']):
    print(f"  Day {i+1:2d}: {hdd:5.1f} HDD")
print()

print("Accumulated HDD by Period:")
print(f"   4-day:  {latest['hdd_4day']:6.1f} HDD  (comparable to ECMWF)")
print(f"   7-day:  {latest['hdd_7day']:6.1f} HDD")
print(f"  10-day:  {latest['hdd_10day']:6.1f} HDD")
print(f"  14-day:  {latest['hdd_14day']:6.1f} HDD")
print(f"  16-day:  {latest['hdd_16day']:6.1f} HDD")

# ── PERSIST forecast history (added 2026-06-11) ──────────────────────
# The HDD-forecast-surprise factor needs a HISTORY of forecasts; until
# now these numbers existed only in this process and the PNG. Appends
# one row per run → /home/wyatt/weather/hdd_forecast_history.csv
try:
    import csv as _csv
    import datetime as _dt
    _hist = '/home/wyatt/weather/hdd_forecast_history.csv'
    _new = not __import__('os').path.exists(_hist)
    with open(_hist, 'a', newline='') as _f:
        _w = _csv.writer(_f)
        if _new:
            _w.writerow(['run_ts', 'forecast_init', 'hdd_4day', 'hdd_7day',
                         'hdd_10day', 'hdd_14day', 'hdd_16day'])
        _w.writerow([_dt.datetime.now().isoformat(timespec='seconds'),
                     str(latest['datetime']),
                     round(latest['hdd_4day'], 1), round(latest['hdd_7day'], 1),
                     round(latest['hdd_10day'], 1), round(latest['hdd_14day'], 1),
                     round(latest['hdd_16day'], 1)])
    print(f"\n[history] appended to {_hist}")
except Exception as _e:
    print(f"\n[history] persist failed: {_e}")
print("=" * 70)

# Create comparison chart if we have multiple forecasts
if len(all_data) > 1:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Different accumulation periods over time
    ax1 = axes[0]
    times = [d['datetime'] for d in all_data]

    for period, label, color in [
        ('hdd_4day', '4-day', 'blue'),
        ('hdd_7day', '7-day', 'green'),
        ('hdd_14day', '14-day', 'orange'),
        ('hdd_16day', '16-day', 'red'),
    ]:
        values = [d[period] for d in all_data if d[period] is not None]
        valid_times = [d['datetime'] for d in all_data if d[period] is not None]
        if values:
            ax1.plot(valid_times, values, 'o-', label=f'{label} HDD', color=color, markersize=8)

    ax1.set_xlabel('Forecast Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accumulated HDD', fontsize=12, fontweight='bold')
    ax1.set_title('GFS Accumulated HDD by Forecast Period - US Continental', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Daily HDD for latest forecast
    ax2 = axes[1]
    days = range(1, len(latest['daily_hdds']) + 1)
    ax2.bar(days, latest['daily_hdds'], color='steelblue', alpha=0.7, edgecolor='navy')
    ax2.axhline(y=np.mean(latest['daily_hdds']), color='red', linestyle='--',
                label=f"Mean: {np.mean(latest['daily_hdds']):.1f} HDD/day")
    ax2.set_xlabel('Forecast Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Daily HDD', fontsize=12, fontweight='bold')
    ax2.set_title(f'GFS Daily HDD Forecast - {latest["datetime"].strftime("%Y-%m-%d %H")}z', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/wyatt/weather/gfs_hdd_forecast.png', dpi=150, bbox_inches='tight')
    print("\nChart saved as 'gfs_hdd_forecast.png'")

else:
    # Single forecast - just show daily breakdown
    fig, ax = plt.subplots(figsize=(14, 6))
    days = range(1, len(latest['daily_hdds']) + 1)
    ax.bar(days, latest['daily_hdds'], color='steelblue', alpha=0.7, edgecolor='navy')
    ax.axhline(y=np.mean(latest['daily_hdds']), color='red', linestyle='--',
               label=f"Mean: {np.mean(latest['daily_hdds']):.1f} HDD/day")
    ax.set_xlabel('Forecast Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily HDD', fontsize=12, fontweight='bold')
    ax.set_title(f'GFS 16-Day HDD Forecast - US Continental\n{latest["datetime"]}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/wyatt/weather/gfs_hdd_forecast.png', dpi=150, bbox_inches='tight')
    print("\nChart saved as 'gfs_hdd_forecast.png'")

plt.show()
