#!/usr/bin/env python3
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the GRIB2 data
print("Loading forecast data...")
ds = xr.open_dataset('forecast_data.grib2', engine='cfgrib')

print(f"Data loaded. Available variables: {list(ds.data_vars)}")
print(f"Coordinates: {list(ds.coords)}")

# Get temperature data (in Kelvin)
temp_k = ds['t2m']
print(f"Temperature shape: {temp_k.shape}")

# Convert to Celsius
temp_c = temp_k - 273.15

# HDD base temperature (typically 65°F = 18.3°C)
base_temp = 18.3

# Calculate HDD for each grid point and time step
# HDD = max(0, base_temp - daily_avg_temp)
hdd = np.maximum(0, base_temp - temp_c)

# Average HDD across all grid points (global average)
hdd_global = hdd.mean(dim=['latitude', 'longitude'])

# Get time information
time_steps = ds['step'].values
valid_times = ds['valid_time'].values

# Create a figure with HDD analysis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: HDD over forecast period
ax1.plot(range(len(time_steps)), hdd_global.values, marker='o', linewidth=2, markersize=6)
ax1.set_xlabel('Forecast Step (hours)', fontsize=12)
ax1.set_ylabel('Global Average HDD (°C)', fontsize=12)
ax1.set_title('Heating Degree Days (HDD) - Global Average Forecast', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(len(time_steps)))
ax1.set_xticklabels([f"{int(h/3600000000000)}" for h in time_steps], rotation=45)

# Calculate cumulative HDD gain
cumulative_hdd = np.cumsum(hdd_global.values)

# Plot 2: Cumulative HDD gain/loss
ax2.plot(range(len(time_steps)), cumulative_hdd, marker='s', linewidth=2, markersize=6, color='red')
ax2.set_xlabel('Forecast Step (hours)', fontsize=12)
ax2.set_ylabel('Cumulative HDD (°C)', fontsize=12)
ax2.set_title('Cumulative Heating Degree Days Over Forecast Period', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(len(time_steps)))
ax2.set_xticklabels([f"{int(h/3600000000000)}" for h in time_steps], rotation=45)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('hdd_forecast.png', dpi=150, bbox_inches='tight')
print("\nChart saved as 'hdd_forecast.png'")

# Print summary statistics
print("\n=== HDD Summary Statistics ===")
print(f"Base Temperature: {base_temp}°C (65°F)")
print(f"\nForecast Period: {len(time_steps)} steps")
print(f"Average HDD per step: {hdd_global.mean().values:.2f}°C")
print(f"Total Cumulative HDD: {cumulative_hdd[-1]:.2f}°C")
print(f"Maximum HDD: {hdd_global.max().values:.2f}°C at step {hdd_global.argmax().values}")
print(f"Minimum HDD: {hdd_global.min().values:.2f}°C at step {hdd_global.argmin().values}")

# Show regional analysis (sample a few locations)
print("\n=== Sample Regional HDD Values (latest forecast) ===")
# Get latest forecast step
latest_hdd = hdd.isel(step=-1)
print(f"Northern latitudes (>60°N): {latest_hdd.where(latest_hdd.latitude > 60).mean().values:.2f}°C")
print(f"Mid latitudes (30-60°N): {latest_hdd.where((latest_hdd.latitude > 30) & (latest_hdd.latitude <= 60)).mean().values:.2f}°C")
print(f"Tropics (±30°): {latest_hdd.where((latest_hdd.latitude >= -30) & (latest_hdd.latitude <= 30)).mean().values:.2f}°C")

plt.show()
