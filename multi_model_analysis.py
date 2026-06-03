#!/usr/bin/env python3
"""
Multi-Model HDD Analysis vs Natural Gas Prices
Compares GFS, ECMWF IFS, and AIFS forecasts against NG front month.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timedelta

print("=" * 70)
print("MULTI-MODEL HDD vs NATURAL GAS ANALYSIS")
print("Models: GFS, ECMWF IFS, AIFS")
print("=" * 70)

# Configuration
BASE_TEMP = 18.3  # 65°F in Celsius
OUTPUT_DIR = "/home/wyatt/weather"


def load_model_csv(filepath):
    """Load model CSV and calculate HDD for different periods."""
    try:
        df = pd.read_csv(filepath)
        if 'temp_c' not in df.columns:
            return None

        # Calculate daily HDD from temperature
        df['hdd'] = np.maximum(0, BASE_TEMP - df['temp_c'])

        # Map forecast hours to days
        df['day'] = df['forecast_hour'] // 24

        # Calculate HDD for different periods (matching GFS trading windows)
        daily_hdds = df.groupby('day')['hdd'].mean().to_dict()

        periods = {}

        # 1-5d (days 1-5)
        d1_5 = [daily_hdds.get(d, 0) for d in range(1, 6)]
        if len([h for h in d1_5 if h > 0]) >= 3:
            periods['1-5d'] = sum(d1_5)

        # 6-10d (days 6-10) - KEY trading window
        d6_10 = [daily_hdds.get(d, 0) for d in range(6, 11)]
        if len([h for h in d6_10 if h > 0]) >= 3:
            periods['6-10d'] = sum(d6_10)

        # 8-14d (days 8-14)
        d8_14 = [daily_hdds.get(d, 0) for d in range(8, 15)]
        if len([h for h in d8_14 if h > 0]) >= 3:
            periods['8-14d'] = sum(d8_14)

        return periods

    except Exception as e:
        return None


def parse_datetime_from_filename(filename):
    """Extract datetime from model filename like gfs_20241201_00z.csv"""
    basename = os.path.basename(filename)
    parts = basename.replace('.csv', '').split('_')

    date_str = None
    hour_str = "00"

    for part in parts:
        if len(part) == 8 and part.isdigit():
            date_str = part
        elif part.endswith('z') and part[:-1].isdigit():
            hour_str = part[:-1]

    if date_str:
        return pd.to_datetime(f"{date_str} {hour_str}:00")
    return None


# ============================================
# Load all model data
# ============================================
print("\nLoading model data...")

models = {
    'gfs': {'files': [], 'data': {'1-5d': [], '6-10d': [], '8-14d': []}, 'color': '#0000FF', 'label': 'GFS'},
    'ifs': {'files': [], 'data': {'1-5d': [], '6-10d': [], '8-14d': []}, 'color': '#8B00FF', 'label': 'ECMWF IFS'},
    'aifs': {'files': [], 'data': {'1-5d': [], '6-10d': [], '8-14d': []}, 'color': '#00CED1', 'label': 'AIFS'}
}

for model_name in models:
    pattern = os.path.join(OUTPUT_DIR, f"{model_name}_*.csv")
    files = sorted(glob.glob(pattern))
    models[model_name]['files'] = files
    print(f"  {model_name.upper()}: Found {len(files)} files")

    for filepath in files:
        dt = parse_datetime_from_filename(filepath)
        if dt is None:
            continue

        periods = load_model_csv(filepath)
        if periods is None:
            continue

        for period in ['1-5d', '6-10d', '8-14d']:
            if period in periods:
                models[model_name]['data'][period].append({
                    'datetime': dt,
                    'acc_hdd': periods[period]
                })

    # Sort by datetime
    for period in models[model_name]['data']:
        models[model_name]['data'][period] = sorted(
            models[model_name]['data'][period],
            key=lambda x: x['datetime']
        )

# Also load existing GRIB2-based data
print("\n  Loading GRIB2 data (existing)...")
grib_gfs = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'gfs_*.grib2')))
grib_ecmwf = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'forecast_historical_*.grib2')))
print(f"    GFS GRIB2: {len(grib_gfs)} files")
print(f"    ECMWF GRIB2: {len(grib_ecmwf)} files")

# ============================================
# Load NG prices
# ============================================
print("\nFetching NG front month prices...")
ng = yf.Ticker("NG=F")
ng_data = ng.history(period="60d", interval="1h")
print(f"  Loaded {len(ng_data)} hourly NG price bars")

# ============================================
# Determine common date range
# ============================================
all_times = []
for model_name in models:
    for period in models[model_name]['data']:
        all_times += [d['datetime'] for d in models[model_name]['data'][period]]

if not all_times:
    print("\nNo model data found. Run fetch_all_models.py first.")
    exit(1)

start_time = min(all_times)
end_time = max(all_times)
print(f"\nData range: {start_time} to {end_time}")

# Filter NG data
if ng_data.index.tz is not None:
    ng_data.index = ng_data.index.tz_localize(None)
ng_filtered = ng_data[(ng_data.index >= start_time - pd.Timedelta(days=1)) &
                      (ng_data.index <= end_time + pd.Timedelta(days=1))]

# ============================================
# Create charts
# ============================================
print("\nCreating charts...")

# Normalization factors (days per period)
period_days = {'1-5d': 5, '6-10d': 5, '8-14d': 7}

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

for idx, (period, ax) in enumerate(zip(['1-5d', '6-10d', '8-14d'], axes)):
    # Right axis for NG
    ax2 = ax.twinx()
    ax2.plot(ng_filtered.index, ng_filtered['Close'], '-', color='red',
             linewidth=1, alpha=0.4, label='NG Price')
    ax2.set_ylabel('NG Price ($/MMBtu)', color='red', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='red')

    # Left axis for HDD
    for model_name, model_info in models.items():
        data = model_info['data'][period]
        if data:
            times = [d['datetime'] for d in data]
            hdds = [d['acc_hdd'] / period_days[period] for d in data]  # Normalize

            # Smooth
            if len(hdds) >= 4:
                df = pd.DataFrame({'time': times, 'hdd': hdds}).set_index('time')
                smoothed = df['hdd'].rolling(window=4, center=True, min_periods=2).mean()
                ax.plot(smoothed.index, smoothed.values, '-',
                        color=model_info['color'], linewidth=2, alpha=0.8,
                        label=model_info['label'])
            else:
                ax.plot(times, hdds, 'o-', color=model_info['color'],
                        linewidth=2, alpha=0.8, label=model_info['label'])

    ax.set_ylabel(f'Daily Avg HDD ({period})', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    period_label = {'1-5d': '1-5 Day (Priced In)',
                    '6-10d': '6-10 Day (KEY Trading Window)',
                    '8-14d': '8-14 Day (Extended)'}
    ax.set_title(period_label[period], fontsize=12, fontweight='bold')

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')

plt.suptitle('Multi-Model HDD Forecasts vs Natural Gas Front Month\n(GFS, ECMWF IFS, AIFS)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'multi_model_hdd_ng.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: multi_model_hdd_ng.png")

# ============================================
# Calculate correlations
# ============================================
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

for period in ['1-5d', '6-10d', '8-14d']:
    print(f"\n{period} Period:")
    for model_name, model_info in models.items():
        data = model_info['data'][period]
        if len(data) < 5:
            continue

        # Match NG prices to forecast times
        ng_at_forecast = []
        hdds = []

        for d in data:
            dt = d['datetime']
            try:
                diffs = abs(ng_filtered.index - dt)
                closest_idx = diffs.argmin()
                if diffs[closest_idx] < pd.Timedelta(hours=2):
                    ng_at_forecast.append(ng_filtered.iloc[closest_idx]['Close'])
                    hdds.append(d['acc_hdd'])
            except:
                pass

        if len(ng_at_forecast) >= 5:
            corr = np.corrcoef(hdds, ng_at_forecast)[0, 1]
            print(f"  {model_info['label']}: r = {corr:.3f} (n={len(ng_at_forecast)})")

# ============================================
# Model comparison summary
# ============================================
print("\n" + "=" * 70)
print("MODEL COMPARISON (Latest Values)")
print("=" * 70)

for period in ['1-5d', '6-10d', '8-14d']:
    print(f"\n{period} Daily Avg HDD:")
    for model_name, model_info in models.items():
        data = model_info['data'][period]
        if data:
            latest = data[-1]['acc_hdd'] / period_days[period]
            mean_val = np.mean([d['acc_hdd'] / period_days[period] for d in data])
            print(f"  {model_info['label']:12s}: {latest:.1f} (latest), {mean_val:.1f} (mean)")

if len(ng_filtered) > 0:
    print(f"\nNG Front Month:")
    print(f"  Latest: ${ng_filtered['Close'].iloc[-1]:.3f}/MMBtu")
    print(f"  Mean:   ${ng_filtered['Close'].mean():.3f}/MMBtu")

print("=" * 70)
plt.show()
