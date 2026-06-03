#!/usr/bin/env python3
"""
Natural Gas Fair Value Model
Calculates theoretical NG price based on HDD forecasts.
Uses regression to find the HDD-NG relationship and identify mispricings.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

print("=" * 70)
print("NATURAL GAS FAIR VALUE MODEL")
print("Based on HDD Forecast Regression")
print("=" * 70)

OUTPUT_DIR = "/home/wyatt/weather"
BASE_TEMP = 18.3

# Period normalization
PERIOD_DAYS = {'1-5d': 5, '6-10d': 5, '8-14d': 7}


def load_csv_data():
    """Load HDD data from CSV files."""
    print("\nLoading forecast data...")

    models = ['gfs', 'ifs', 'aifs']
    all_data = []

    for model in models:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{model}_*.csv")))

        for filepath in files:
            try:
                df = pd.read_csv(filepath)
                if 'temp_c' not in df.columns:
                    continue

                # Parse datetime from filename
                basename = os.path.basename(filepath)
                parts = basename.replace('.csv', '').split('_')
                date_str = hour_str = None
                for part in parts:
                    if len(part) == 8 and part.isdigit():
                        date_str = part
                    elif part.endswith('z'):
                        hour_str = part[:-1]

                if not date_str:
                    continue

                dt = pd.to_datetime(f"{date_str} {hour_str or '00'}:00")

                # Calculate HDD by day
                df['hdd'] = np.maximum(0, BASE_TEMP - df['temp_c'])
                df['day'] = df['forecast_hour'] // 24
                daily_hdd = df.groupby('day')['hdd'].mean()

                # Calculate period HDDs (normalized to daily avg)
                record = {'datetime': dt, 'model': model}

                # 1-5d
                d1_5 = [daily_hdd.get(d, 0) for d in range(1, 6)]
                if len([h for h in d1_5 if h > 0]) >= 3:
                    record[f'{model}_1_5d'] = np.mean(d1_5)

                # 6-10d (KEY)
                d6_10 = [daily_hdd.get(d, 0) for d in range(6, 11)]
                if len([h for h in d6_10 if h > 0]) >= 3:
                    record[f'{model}_6_10d'] = np.mean(d6_10)

                # 8-14d
                d8_14 = [daily_hdd.get(d, 0) for d in range(8, 15)]
                if len([h for h in d8_14 if h > 0]) >= 3:
                    record[f'{model}_8_14d'] = np.mean(d8_14)

                all_data.append(record)

            except Exception as e:
                continue

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Pivot to get one row per datetime
    pivoted = df.pivot_table(
        index='datetime',
        values=[c for c in df.columns if c not in ['datetime', 'model']],
        aggfunc='first'
    ).reset_index()

    # Calculate ensemble mean for 6-10d (key period)
    cols_6_10 = [c for c in pivoted.columns if '6_10d' in c]
    if cols_6_10:
        pivoted['ensemble_6_10d'] = pivoted[cols_6_10].mean(axis=1)
        pivoted['model_spread'] = pivoted[cols_6_10].std(axis=1)

    return pivoted.sort_values('datetime')


def load_ng_prices():
    """Load NG front month prices."""
    print("Fetching NG prices...")
    ng = yf.Ticker("NG=F")
    ng_data = ng.history(period="max", interval="1d")

    if ng_data.index.tz is not None:
        ng_data.index = ng_data.index.tz_localize(None)

    print(f"  Loaded {len(ng_data)} daily bars")
    return ng_data


def build_fair_value_model(hdd_df, ng_df):
    """Build regression model: NG = f(HDD)"""
    print("\nBuilding fair value model...")

    # Match HDD forecasts to NG prices
    matched_data = []

    for idx, row in hdd_df.iterrows():
        dt = row['datetime']

        # Find closest NG price
        try:
            time_diffs = abs(ng_df.index - dt)
            closest_idx = time_diffs.argmin()
            if time_diffs[closest_idx] > pd.Timedelta(days=1):
                continue

            ng_price = ng_df.iloc[closest_idx]['Close']

            if 'ensemble_6_10d' in row and not pd.isna(row['ensemble_6_10d']):
                matched_data.append({
                    'datetime': dt,
                    'hdd_6_10d': row['ensemble_6_10d'],
                    'model_spread': row.get('model_spread', 0),
                    'ng_price': ng_price
                })
        except:
            continue

    if len(matched_data) < 10:
        print("  Not enough data for regression")
        return None, None, None

    matched_df = pd.DataFrame(matched_data)

    # Fit linear regression: NG = a + b * HDD
    X = matched_df[['hdd_6_10d']].values
    y = matched_df['ng_price'].values

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    intercept = model.intercept_
    slope = model.coef_[0]

    print(f"  Regression: NG = {intercept:.3f} + {slope:.4f} * HDD_6-10d")
    print(f"  R² = {r2:.3f}")
    print(f"  Samples: {len(matched_df)}")

    # Calculate fair value and residuals
    matched_df['fair_value'] = model.predict(X)
    matched_df['residual'] = matched_df['ng_price'] - matched_df['fair_value']
    matched_df['residual_pct'] = matched_df['residual'] / matched_df['fair_value'] * 100

    return model, matched_df, {'r2': r2, 'intercept': intercept, 'slope': slope}


def plot_fair_value(matched_df, model_stats, hdd_df, ng_df):
    """Create fair value visualization."""
    print("\nCreating charts...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # ============================================
    # Chart 1: NG Price vs Fair Value over time
    # ============================================
    ax1 = axes[0]

    ax1.plot(matched_df['datetime'], matched_df['ng_price'],
             'b-', linewidth=2, label='Actual NG Price', alpha=0.8)
    ax1.plot(matched_df['datetime'], matched_df['fair_value'],
             'g--', linewidth=2, label='Fair Value (HDD-based)', alpha=0.8)

    # Fill between to show over/undervalued
    ax1.fill_between(matched_df['datetime'],
                     matched_df['fair_value'], matched_df['ng_price'],
                     where=matched_df['ng_price'] > matched_df['fair_value'],
                     color='red', alpha=0.3, label='Overvalued')
    ax1.fill_between(matched_df['datetime'],
                     matched_df['fair_value'], matched_df['ng_price'],
                     where=matched_df['ng_price'] < matched_df['fair_value'],
                     color='green', alpha=0.3, label='Undervalued')

    ax1.set_ylabel('NG Price ($/MMBtu)', fontsize=11)
    ax1.set_title(f'Natural Gas: Actual vs Fair Value (R² = {model_stats["r2"]:.3f})',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    # ============================================
    # Chart 2: Residual (mispricing) over time
    # ============================================
    ax2 = axes[1]

    colors = ['green' if r < 0 else 'red' for r in matched_df['residual_pct']]
    ax2.bar(matched_df['datetime'], matched_df['residual_pct'],
            color=colors, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add +/- 1 std bands
    std = matched_df['residual_pct'].std()
    ax2.axhline(y=std, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-std, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(matched_df['datetime'], -std, std, color='gray', alpha=0.1)

    ax2.set_ylabel('Mispricing (%)', fontsize=11)
    ax2.set_title('NG Mispricing: Actual - Fair Value (Green = Undervalued, Red = Overvalued)',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    # ============================================
    # Chart 3: Scatter plot - HDD vs NG with regression line
    # ============================================
    ax3 = axes[2]

    scatter = ax3.scatter(matched_df['hdd_6_10d'], matched_df['ng_price'],
                         c=matched_df['datetime'].astype(np.int64),
                         cmap='viridis', alpha=0.7, s=50)

    # Regression line
    hdd_range = np.linspace(matched_df['hdd_6_10d'].min(),
                            matched_df['hdd_6_10d'].max(), 100)
    fair_line = model_stats['intercept'] + model_stats['slope'] * hdd_range
    ax3.plot(hdd_range, fair_line, 'r-', linewidth=2,
             label=f"Fair Value: ${model_stats['intercept']:.2f} + ${model_stats['slope']:.3f}×HDD")

    # Mark latest point
    latest = matched_df.iloc[-1]
    ax3.scatter([latest['hdd_6_10d']], [latest['ng_price']],
                color='red', s=200, marker='*', zorder=5, label='Latest')

    ax3.set_xlabel('6-10d Daily Avg HDD', fontsize=11)
    ax3.set_ylabel('NG Price ($/MMBtu)', fontsize=11)
    ax3.set_title('HDD vs NG Price Relationship', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax3, label='Date')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ng_fair_value.png'), dpi=150, bbox_inches='tight')
    print("  Saved: ng_fair_value.png")


def main():
    # Load data
    hdd_df = load_csv_data()
    if hdd_df.empty:
        print("No HDD data found")
        return

    print(f"  Loaded {len(hdd_df)} forecasts")

    ng_df = load_ng_prices()

    # Build fair value model
    model, matched_df, model_stats = build_fair_value_model(hdd_df, ng_df)

    if model is None:
        return

    # Create visualization
    plot_fair_value(matched_df, model_stats, hdd_df, ng_df)

    # ============================================
    # Current Signal
    # ============================================
    print("\n" + "=" * 70)
    print("CURRENT FAIR VALUE ANALYSIS")
    print("=" * 70)

    latest = matched_df.iloc[-1]
    latest_hdd = hdd_df.iloc[-1]

    print(f"\nLatest Forecast: {latest['datetime']}")
    print(f"  6-10d HDD: {latest['hdd_6_10d']:.1f} HDD/day")
    if 'model_spread' in latest and not pd.isna(latest['model_spread']):
        print(f"  Model Spread: ±{latest['model_spread']:.1f} HDD/day")

    print(f"\nNG Price:")
    print(f"  Actual:     ${latest['ng_price']:.3f}/MMBtu")
    print(f"  Fair Value: ${latest['fair_value']:.3f}/MMBtu")
    print(f"  Mispricing: {latest['residual_pct']:+.1f}%")

    # Trading signal
    residual_std = matched_df['residual_pct'].std()
    z_score = latest['residual_pct'] / residual_std

    print(f"\nSignal Strength: {abs(z_score):.2f} std deviations")

    if z_score > 1.5:
        signal = "STRONG SELL"
        reason = "significantly overvalued vs HDD forecast"
    elif z_score > 0.5:
        signal = "SELL"
        reason = "overvalued vs HDD forecast"
    elif z_score < -1.5:
        signal = "STRONG BUY"
        reason = "significantly undervalued vs HDD forecast"
    elif z_score < -0.5:
        signal = "BUY"
        reason = "undervalued vs HDD forecast"
    else:
        signal = "NEUTRAL"
        reason = "fairly valued"

    print(f"\nSIGNAL: {signal}")
    print(f"Reason: NG is {reason}")

    # Fair value range
    fair_low = model_stats['intercept'] + model_stats['slope'] * (latest['hdd_6_10d'] - 2)
    fair_high = model_stats['intercept'] + model_stats['slope'] * (latest['hdd_6_10d'] + 2)
    print(f"\nFair Value Range (±2 HDD): ${fair_low:.3f} - ${fair_high:.3f}")

    print("\n" + "=" * 70)
    print("DISCLAIMER: This is for educational purposes only.")
    print("Not financial advice. Weather is just one factor affecting NG prices.")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
