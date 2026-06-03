#!/usr/bin/env python3
"""
Long-Term Fair Value Analysis
Uses historical HDD data (2021+) to build robust fair value model.
Analyzes prediction accuracy, error distribution, and regime changes.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("LONG-TERM FAIR VALUE ANALYSIS")
print("HDD vs Natural Gas (2021-Present)")
print("=" * 70)

HIST_DIR = "/home/wyatt/weather/historical"
OUTPUT_DIR = "/home/wyatt/weather"
BASE_TEMP = 18.3


def load_historical_hdd():
    """Load all historical HDD data."""
    print("\nLoading historical HDD data...")

    all_data = []

    # Load GFS historical
    gfs_files = sorted(glob.glob(os.path.join(HIST_DIR, "gfs_hist_*.csv")))
    print(f"  GFS files: {len(gfs_files)}")

    for f in gfs_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                row = df.iloc[0]
                date = pd.to_datetime(str(int(row['date'])), format='%Y%m%d')
                all_data.append({
                    'date': date,
                    'source': 'gfs',
                    'hdd_1_5d': row.get('hdd_1_5d', np.nan),
                    'hdd_6_10d': row.get('hdd_6_10d', np.nan),
                    'hdd_8_14d': row.get('hdd_8_14d', np.nan)
                })
        except:
            continue

    # Load IFS historical
    ifs_files = sorted(glob.glob(os.path.join(HIST_DIR, "ifs_hist_*.csv")))
    print(f"  IFS files: {len(ifs_files)}")

    for f in ifs_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                row = df.iloc[0]
                date = pd.to_datetime(str(int(row['date'])), format='%Y%m%d')
                all_data.append({
                    'date': date,
                    'source': 'ifs',
                    'hdd_1_5d': row.get('hdd_1_5d', np.nan),
                    'hdd_6_10d': row.get('hdd_6_10d', np.nan),
                    'hdd_8_14d': row.get('hdd_8_14d', np.nan) if 'hdd_8_14d' in row else np.nan
                })
        except:
            continue

    # Also load recent CSV data
    recent_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "gfs_*.csv")))
    print(f"  Recent GFS files: {len(recent_files)}")

    for f in recent_files:
        try:
            df = pd.read_csv(f)
            if 'temp_c' not in df.columns:
                continue

            basename = os.path.basename(f)
            parts = basename.replace('.csv', '').split('_')
            date_str = None
            for part in parts:
                if len(part) == 8 and part.isdigit():
                    date_str = part
                    break

            if not date_str:
                continue

            date = pd.to_datetime(date_str, format='%Y%m%d')

            df['hdd'] = np.maximum(0, BASE_TEMP - df['temp_c'])
            df['day'] = df['forecast_hour'] // 24
            daily_hdd = df.groupby('day')['hdd'].mean()

            days_1_5 = [daily_hdd.get(d, 0) for d in range(1, 6)]
            days_6_10 = [daily_hdd.get(d, 0) for d in range(6, 11)]
            days_8_14 = [daily_hdd.get(d, 0) for d in range(8, 15)]

            all_data.append({
                'date': date,
                'source': 'gfs_recent',
                'hdd_1_5d': np.mean(days_1_5) if days_1_5 else np.nan,
                'hdd_6_10d': np.mean(days_6_10) if days_6_10 else np.nan,
                'hdd_8_14d': np.mean(days_8_14) if days_8_14 else np.nan
            })
        except:
            continue

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.sort_values('date')

    # Aggregate by date (take mean if multiple sources)
    agg = df.groupby('date').agg({
        'hdd_1_5d': 'mean',
        'hdd_6_10d': 'mean',
        'hdd_8_14d': 'mean'
    }).reset_index()

    return agg


def load_ng_history():
    """Load full NG price history."""
    print("\nLoading NG price history...")
    ng = yf.Ticker("NG=F")
    ng_data = ng.history(period="max", interval="1d")

    if ng_data.index.tz is not None:
        ng_data.index = ng_data.index.tz_localize(None)

    print(f"  Loaded {len(ng_data)} daily bars")
    print(f"  Range: {ng_data.index.min().date()} to {ng_data.index.max().date()}")

    return ng_data


def build_fair_value_model(hdd_df, ng_df):
    """Build and analyze fair value regression model."""
    print("\nBuilding fair value model...")

    # Match HDD to NG prices
    matched = []
    for idx, row in hdd_df.iterrows():
        date = row['date']

        try:
            # Find NG price on that date
            if date in ng_df.index:
                ng_price = ng_df.loc[date, 'Close']
            else:
                # Find closest
                diffs = abs(ng_df.index - date)
                closest_idx = diffs.argmin()
                if diffs[closest_idx] > pd.Timedelta(days=3):
                    continue
                ng_price = ng_df.iloc[closest_idx]['Close']

            if not pd.isna(row['hdd_6_10d']) and ng_price > 0:
                matched.append({
                    'date': date,
                    'hdd_6_10d': row['hdd_6_10d'],
                    'hdd_1_5d': row.get('hdd_1_5d', np.nan),
                    'ng_price': ng_price,
                    'month': date.month,
                    'year': date.year
                })
        except:
            continue

    if len(matched) < 20:
        print(f"  Only {len(matched)} matched samples - not enough")
        return None, None

    df = pd.DataFrame(matched)
    print(f"  Matched {len(df)} HDD forecasts to NG prices")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Simple linear regression: NG = a + b * HDD
    X = df[['hdd_6_10d']].values
    y = df['ng_price'].values

    model = LinearRegression()
    model.fit(X, y)

    # Predictions and residuals
    df['fair_value'] = model.predict(X)
    df['residual'] = df['ng_price'] - df['fair_value']
    df['residual_pct'] = df['residual'] / df['fair_value'] * 100
    df['abs_error'] = abs(df['residual'])
    df['abs_error_pct'] = abs(df['residual_pct'])

    # Model statistics
    r2 = model.score(X, y)
    mae = mean_absolute_error(y, df['fair_value'])
    rmse = np.sqrt(mean_squared_error(y, df['fair_value']))
    mae_pct = df['abs_error_pct'].mean()

    stats = {
        'intercept': model.intercept_,
        'slope': model.coef_[0],
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mae_pct': mae_pct,
        'n_samples': len(df)
    }

    print(f"\n  Regression: NG = ${stats['intercept']:.3f} + ${stats['slope']:.4f} × HDD")
    print(f"  R² = {r2:.3f}")
    print(f"  MAE = ${mae:.3f} ({mae_pct:.1f}%)")
    print(f"  RMSE = ${rmse:.3f}")

    return df, stats


def analyze_errors(df, stats):
    """Analyze prediction errors and identify patterns."""
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    # Error distribution
    print(f"\nResidual Statistics:")
    print(f"  Mean: {df['residual'].mean():+.3f}")
    print(f"  Std:  {df['residual'].std():.3f}")
    print(f"  Min:  {df['residual'].min():.3f}")
    print(f"  Max:  {df['residual'].max():.3f}")

    print(f"\nPercentage Error Distribution:")
    print(f"  Mean: {df['residual_pct'].mean():+.1f}%")
    print(f"  Std:  {df['residual_pct'].std():.1f}%")
    print(f"  Within ±5%:  {(df['abs_error_pct'] <= 5).mean()*100:.1f}% of samples")
    print(f"  Within ±10%: {(df['abs_error_pct'] <= 10).mean()*100:.1f}% of samples")
    print(f"  Within ±20%: {(df['abs_error_pct'] <= 20).mean()*100:.1f}% of samples")
    print(f"  Within ±30%: {(df['abs_error_pct'] <= 30).mean()*100:.1f}% of samples")

    # Error by season
    print(f"\nError by Season:")
    df['season'] = df['month'].apply(lambda m: 'Winter' if m in [11,12,1,2,3] else 'Summer' if m in [5,6,7,8,9] else 'Shoulder')
    for season in ['Winter', 'Shoulder', 'Summer']:
        season_df = df[df['season'] == season]
        if len(season_df) > 0:
            print(f"  {season:10s}: MAE = ${season_df['abs_error'].mean():.3f} ({season_df['abs_error_pct'].mean():.1f}%), n={len(season_df)}")

    # Error by year
    print(f"\nError by Year:")
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        if len(year_df) >= 5:
            print(f"  {year}: MAE = ${year_df['abs_error'].mean():.3f} ({year_df['abs_error_pct'].mean():.1f}%), n={len(year_df)}")

    # Worst predictions
    print(f"\nTop 10 Worst Predictions:")
    worst = df.nlargest(10, 'abs_error_pct')[['date', 'hdd_6_10d', 'ng_price', 'fair_value', 'residual_pct']]
    for _, row in worst.iterrows():
        print(f"  {row['date'].date()}: HDD={row['hdd_6_10d']:.1f}, Actual=${row['ng_price']:.2f}, Fair=${row['fair_value']:.2f}, Error={row['residual_pct']:+.1f}%")

    return df


def create_visualizations(df, stats):
    """Create comprehensive visualizations."""
    print("\nCreating visualizations...")

    fig = plt.figure(figsize=(18, 16))

    # 1. Time series: Actual vs Fair Value
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(df['date'], df['ng_price'], 'b-', linewidth=1.5, alpha=0.8, label='Actual NG')
    ax1.plot(df['date'], df['fair_value'], 'g--', linewidth=1.5, alpha=0.8, label='Fair Value')
    ax1.fill_between(df['date'], df['fair_value'], df['ng_price'],
                     where=df['ng_price'] > df['fair_value'],
                     color='red', alpha=0.2, label='Overvalued')
    ax1.fill_between(df['date'], df['fair_value'], df['ng_price'],
                     where=df['ng_price'] < df['fair_value'],
                     color='green', alpha=0.2, label='Undervalued')
    ax1.set_ylabel('NG Price ($/MMBtu)')
    ax1.set_title(f'Natural Gas: Actual vs Fair Value (R² = {stats["r2"]:.3f})', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Residual over time
    ax2 = fig.add_subplot(3, 2, 2)
    colors = ['green' if r < 0 else 'red' for r in df['residual_pct']]
    ax2.bar(df['date'], df['residual_pct'], color=colors, alpha=0.7, width=5)
    ax2.axhline(y=0, color='black', linewidth=1)
    std = df['residual_pct'].std()
    ax2.axhline(y=std, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-std, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=2*std, color='gray', linestyle=':', alpha=0.3)
    ax2.axhline(y=-2*std, color='gray', linestyle=':', alpha=0.3)
    ax2.set_ylabel('Mispricing (%)')
    ax2.set_title('Mispricing Over Time (1σ and 2σ bands)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Scatter: HDD vs NG
    ax3 = fig.add_subplot(3, 2, 3)
    scatter = ax3.scatter(df['hdd_6_10d'], df['ng_price'],
                         c=df['date'].astype(np.int64),
                         cmap='viridis', alpha=0.6, s=30)
    hdd_range = np.linspace(df['hdd_6_10d'].min(), df['hdd_6_10d'].max(), 100)
    fair_line = stats['intercept'] + stats['slope'] * hdd_range
    ax3.plot(hdd_range, fair_line, 'r-', linewidth=2,
             label=f"FV = ${stats['intercept']:.2f} + ${stats['slope']:.3f}×HDD")
    ax3.set_xlabel('6-10d Daily Avg HDD')
    ax3.set_ylabel('NG Price ($/MMBtu)')
    ax3.set_title('HDD vs NG Price (Color = Date)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Date')

    # 4. Error histogram
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.hist(df['residual_pct'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.axvline(x=df['residual_pct'].mean(), color='green', linestyle='-', linewidth=2, label=f"Mean: {df['residual_pct'].mean():.1f}%")
    ax4.set_xlabel('Prediction Error (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution (Std: {df["residual_pct"].std():.1f}%)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Error by month
    ax5 = fig.add_subplot(3, 2, 5)
    monthly = df.groupby('month')['abs_error_pct'].mean()
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colors = ['blue' if m in [11,12,1,2,3] else 'orange' if m in [5,6,7,8,9] else 'gray' for m in range(1,13)]
    ax5.bar(range(1, 13), [monthly.get(m, 0) for m in range(1, 13)], color=colors, alpha=0.7)
    ax5.set_xticks(range(1, 13))
    ax5.set_xticklabels(months, rotation=45)
    ax5.set_ylabel('Mean Absolute Error (%)')
    ax5.set_title('Prediction Error by Month (Blue=Winter, Orange=Summer)', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Rolling accuracy
    ax6 = fig.add_subplot(3, 2, 6)
    df_sorted = df.sort_values('date')
    rolling_mae = df_sorted['abs_error_pct'].rolling(window=10, min_periods=5).mean()
    ax6.plot(df_sorted['date'], rolling_mae, 'b-', linewidth=2)
    ax6.axhline(y=df['abs_error_pct'].mean(), color='red', linestyle='--',
                label=f'Overall Mean: {df["abs_error_pct"].mean():.1f}%')
    ax6.set_ylabel('Rolling MAE (%)')
    ax6.set_title('10-Sample Rolling Mean Absolute Error', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fair_value_longterm.png'), dpi=150, bbox_inches='tight')
    print("  Saved: fair_value_longterm.png")


def main():
    # Load data
    hdd_df = load_historical_hdd()
    if hdd_df.empty:
        print("\nNo historical HDD data found. Run fetch_historical_deep.py first.")
        print("Checking recent data only...")
        # Still continue with recent data

    ng_df = load_ng_history()

    # Build model
    result_df, stats = build_fair_value_model(hdd_df, ng_df)

    if result_df is None:
        print("\nInsufficient data for analysis.")
        return

    # Analyze errors
    result_df = analyze_errors(result_df, stats)

    # Create visualizations
    create_visualizations(result_df, stats)

    # Current signal
    print("\n" + "=" * 70)
    print("CURRENT SIGNAL")
    print("=" * 70)

    latest = result_df.iloc[-1]
    residual_std = result_df['residual_pct'].std()
    z_score = latest['residual_pct'] / residual_std

    print(f"\nLatest ({latest['date'].date()}):")
    print(f"  HDD 6-10d:  {latest['hdd_6_10d']:.1f}")
    print(f"  NG Actual:  ${latest['ng_price']:.3f}")
    print(f"  Fair Value: ${latest['fair_value']:.3f}")
    print(f"  Mispricing: {latest['residual_pct']:+.1f}% ({z_score:+.2f}σ)")

    # Historical context
    pct_rank = (result_df['residual_pct'] < latest['residual_pct']).mean() * 100
    print(f"\n  Historical Percentile: {pct_rank:.0f}%")
    print(f"  (100% = most overvalued historically)")

    if z_score > 2:
        signal = "STRONG SELL"
    elif z_score > 1:
        signal = "SELL"
    elif z_score < -2:
        signal = "STRONG BUY"
    elif z_score < -1:
        signal = "BUY"
    else:
        signal = "NEUTRAL"

    print(f"\n  SIGNAL: {signal}")

    # Model reliability warning
    print(f"\n  Model Reliability:")
    print(f"    - Explains {stats['r2']*100:.0f}% of price variation")
    print(f"    - Average error: ±{stats['mae_pct']:.0f}%")
    print(f"    - Based on {stats['n_samples']} historical samples")

    print("\n" + "=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
