#!/usr/bin/env python3
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from datetime import datetime

print("Loading current and historical forecast data...")

# Calculate HDD (Heating Degree Days)
base_temp = 18.3  # 65°F in Celsius

# US bounding box (approximate continental US)
us_lat_min, us_lat_max = 25, 50   # Latitude range
us_lon_min, us_lon_max = -125, -65  # Longitude range

def calculate_accumulated_hdd(ds, base_temp=18.3):
    """
    Calculate accumulated HDD over the forecast period (US standard method).
    Groups 6-hourly data into daily averages, calculates HDD per day, then sums.
    Returns: accumulated HDD and daily breakdown
    """
    temp_c = ds['t2m'] - 273.15
    steps_hours = [int(s / np.timedelta64(1, 'h')) for s in ds['step'].values]

    daily_hdds = []
    for day in range(4):  # 4-day forecast
        start_h = day * 24
        end_h = (day + 1) * 24
        day_steps = [i for i, h in enumerate(steps_hours) if start_h <= h < end_h]
        if day_steps:
            # Average temp for the day across all 6-hourly steps
            day_temp = temp_c.isel(step=day_steps).mean(dim='step')
            # HDD for the day
            day_hdd = np.maximum(0, base_temp - day_temp)
            # Spatial mean over US region
            day_hdd_mean = float(day_hdd.mean().values)
            daily_hdds.append(day_hdd_mean)

    accumulated = sum(daily_hdds)
    return accumulated, daily_hdds

def load_and_filter_us(filepath):
    """Load GRIB2 file and filter to US region."""
    ds = xr.open_dataset(filepath, engine='cfgrib')

    # Convert longitude from 0-360 to -180-180 if needed
    if ds['longitude'].max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby('longitude')

    # Filter to US region
    ds_us = ds.where(
        (ds.latitude >= us_lat_min) & (ds.latitude <= us_lat_max) &
        (ds.longitude >= us_lon_min) & (ds.longitude <= us_lon_max),
        drop=True
    )
    return ds_us, ds['time'].values

# Load current forecast
current_ds_us, current_time = load_and_filter_us('forecast_data.grib2')
current_date = str(current_time).split('T')[0]
current_hour = str(current_time).split('T')[1][:2]

print(f"US region grid points: {current_ds_us.latitude.size} x {current_ds_us.longitude.size} = {current_ds_us.latitude.size * current_ds_us.longitude.size} total")

# Calculate accumulated HDD for current forecast
current_acc_hdd, current_daily = calculate_accumulated_hdd(current_ds_us, base_temp)

# Load historical forecasts
historical_files = sorted(glob.glob('forecast_historical_*.grib2'))
print(f"Found {len(historical_files)} historical forecast files\n")

if len(historical_files) == 0:
    print("No historical data found. Cannot create comparison.")
    exit(1)

# Calculate for each historical forecast
historical_data = []
for hfile in historical_files:
    print(f"  Loading {os.path.basename(hfile)}...")
    ds_us, forecast_time = load_and_filter_us(hfile)

    acc_hdd, daily_hdds = calculate_accumulated_hdd(ds_us, base_temp)

    forecast_date = str(forecast_time).split('T')[0]
    forecast_hour = str(forecast_time).split('T')[1][:2]

    historical_data.append({
        'date': forecast_date,
        'hour': forecast_hour,
        'acc_hdd': acc_hdd,
        'daily_hdds': daily_hdds,
        'label': f"{forecast_date} {forecast_hour}z"
    })

# Sort by date and hour
historical_data = sorted(historical_data, key=lambda x: (x['date'], x['hour']))

# Calculate historical statistics
historical_acc_hdds = [d['acc_hdd'] for d in historical_data]
historical_mean = np.mean(historical_acc_hdds)
historical_std = np.std(historical_acc_hdds)

# Create the chart
fig, ax = plt.subplots(figsize=(14, 7))

# Combine all data
all_data = historical_data + [{
    'date': current_date,
    'hour': current_hour,
    'acc_hdd': current_acc_hdd,
    'daily_hdds': current_daily,
    'is_current': True
}]

# Remove duplicates
seen = set()
unique_data = []
for d in all_data:
    key = (d['date'], d['hour'])
    if key not in seen:
        seen.add(key)
        unique_data.append(d)

unique_data = sorted(unique_data, key=lambda x: (x['date'], x['hour']))

# Prepare data for plotting
positions = range(len(unique_data))
labels = [f"{d['date'][5:10]}\n{d['hour']}z" for d in unique_data]
values = [d['acc_hdd'] for d in unique_data]

# Plot line connecting all points
ax.plot(positions, values, 'o-', linewidth=2.5, markersize=8, color='steelblue', alpha=0.7, label='Forecast Releases', zorder=2)

# Highlight current forecast point
current_idx = next((i for i, d in enumerate(unique_data) if d.get('is_current')), len(unique_data) - 1)
ax.plot(current_idx, values[current_idx], 'o', markersize=12, color='red', alpha=0.9, label=f'Current ({current_date[5:10]} {current_hour}z)', zorder=3)

# Add horizontal line for historical mean
historical_values = [d['acc_hdd'] for d in unique_data if not d.get('is_current', False)]
if historical_values:
    hist_mean = np.mean(historical_values)
    ax.axhline(y=hist_mean, color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {hist_mean:.1f} HDD', zorder=1)

# Formatting
ax.set_xlabel('Forecast Date & Time', fontsize=12, fontweight='bold')
ax.set_ylabel('Accumulated HDD (4-day total)', fontsize=12, fontweight='bold')
ax.set_title(f'Accumulated Heating Degree Days - US Continental (4-Day Outlook)\nBase Temperature: 65°F (18.3°C)', fontsize=14, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9, rotation=0)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

# Add value labels on points
for i, (pos, val) in enumerate(zip(positions, values)):
    if unique_data[i].get('is_current'):
        ax.text(pos, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
    elif i % 2 == 0:  # Show every other label to reduce clutter
        ax.text(pos, val, f'{val:.1f}', ha='center', va='bottom', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('hdd_forecast.png', dpi=150, bbox_inches='tight')
print("\nChart saved as 'hdd_forecast.png'\n")

# Print summary
print("="*65)
print("ACCUMULATED HDD SUMMARY - US CONTINENTAL (4-DAY OUTLOOK)")
print("="*65)
print(f"Base Temperature: 65°F (18.3°C)")
print(f"Region: {us_lat_min}°-{us_lat_max}°N, {abs(us_lon_min)}°-{abs(us_lon_max)}°W")
print(f"Outlook: 4 days (96 hours)\n")

print("Current Forecast Daily Breakdown:")
for i, hdd in enumerate(current_daily):
    print(f"  Day {i+1}: {hdd:.1f} HDD")
print(f"  ─────────────────")
print(f"  Total: {current_acc_hdd:.1f} HDD\n")

print("Historical Forecasts (Accumulated 4-day HDD):")
for d in historical_data[-10:]:  # Show last 10
    print(f"  {d['date']} {d['hour']:>2}z: {d['acc_hdd']:5.1f} HDD")
if len(historical_data) > 10:
    print(f"  ... ({len(historical_data) - 10} earlier forecasts)")

print(f"\nStatistics:")
print(f"  Historical Mean: {historical_mean:.1f} HDD")
print(f"  Historical Std:  {historical_std:.1f} HDD")
print(f"  Current:         {current_acc_hdd:.1f} HDD")
print(f"  Difference:      {current_acc_hdd - historical_mean:+.1f} HDD ({((current_acc_hdd - historical_mean) / historical_mean * 100):+.1f}%)")

if current_acc_hdd > historical_mean + historical_std:
    print("\n>>> CURRENT FORECAST IS SIGNIFICANTLY COLDER THAN AVERAGE <<<")
elif current_acc_hdd < historical_mean - historical_std:
    print("\n>>> CURRENT FORECAST IS SIGNIFICANTLY WARMER THAN AVERAGE <<<")
else:
    print("\nCurrent forecast is within normal range")

print("="*65)

plt.show()
