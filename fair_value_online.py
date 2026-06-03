#!/usr/bin/env python3
"""
Online Fair Value Algorithm with Error Bars
Shows predictions as they would have been made in real-time,
with confidence intervals based on recent model accuracy.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.linear_model import Ridge
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

HIST_DIR = "/home/wyatt/weather/historical"
OUTPUT_DIR = "/home/wyatt/weather"
BASE_TEMP = 18.3


def load_all_data():
    """Load all historical HDD and NG data."""
    print("Loading data...")

    # Load GFS historical
    all_hdd = []
    gfs_files = sorted(glob.glob(os.path.join(HIST_DIR, "gfs_hist_*.csv")))

    for f in gfs_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                row = df.iloc[0]
                date = pd.to_datetime(str(int(row['date'])), format='%Y%m%d')
                all_hdd.append({
                    'date': date,
                    'hdd_6_10d': row.get('hdd_6_10d', np.nan),
                })
        except:
            continue

    # Also load recent CSV data
    recent_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "gfs_*.csv")))
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

            days_6_10 = [daily_hdd.get(d, 0) for d in range(6, 11)]

            all_hdd.append({
                'date': date,
                'hdd_6_10d': np.mean(days_6_10) if days_6_10 else np.nan,
            })
        except:
            continue

    hdd_df = pd.DataFrame(all_hdd)
    hdd_df = hdd_df.drop_duplicates('date').sort_values('date').reset_index(drop=True)

    # Load NG prices
    ng = yf.Ticker("NG=F")
    ng_df = ng.history(period="max", interval="1d")
    if ng_df.index.tz is not None:
        ng_df.index = ng_df.index.tz_localize(None)

    return hdd_df, ng_df


def merge_data(hdd_df, ng_df):
    """Merge HDD with NG prices."""
    matched = []
    for idx, row in hdd_df.iterrows():
        date = row['date']

        try:
            if date in ng_df.index:
                ng_price = ng_df.loc[date, 'Close']
            else:
                diffs = abs(ng_df.index - date)
                closest_idx = diffs.argmin()
                if diffs[closest_idx] > pd.Timedelta(days=3):
                    continue
                ng_price = ng_df.iloc[closest_idx]['Close']

            if not pd.isna(row['hdd_6_10d']) and ng_price > 0:
                matched.append({
                    'date': date,
                    'hdd_6_10d': row['hdd_6_10d'],
                    'ng_price': ng_price,
                })
        except:
            continue

    return pd.DataFrame(matched).sort_values('date').reset_index(drop=True)


def online_algorithm(df, train_window=26, error_window=12):
    """
    Online prediction algorithm.
    At each time step, only uses data available up to that point.

    train_window: weeks of data to train regression
    error_window: weeks to compute rolling error estimate
    """
    print(f"Running online algorithm (train={train_window}w, error={error_window}w)...")

    results = []

    for i in range(len(df)):
        current = df.iloc[i]

        # Training data: only past observations
        if i < train_window:
            train_df = df.iloc[:i]
        else:
            train_df = df.iloc[i-train_window:i]

        if len(train_df) < 10:
            # Not enough data - skip
            results.append({
                'date': current['date'],
                'hdd_6_10d': current['hdd_6_10d'],
                'ng_actual': current['ng_price'],
                'ng_predicted': np.nan,
                'error_estimate': np.nan,
                'lower_1sigma': np.nan,
                'upper_1sigma': np.nan,
                'lower_2sigma': np.nan,
                'upper_2sigma': np.nan,
            })
            continue

        # Fit model on past data only
        X_train = train_df[['hdd_6_10d']].values
        y_train = train_df['ng_price'].values

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Predict current price (this is the "online" prediction)
        prediction = model.predict([[current['hdd_6_10d']]])[0]

        # Estimate error from recent predictions
        if i >= error_window:
            recent_results = results[-(error_window):]
            recent_errors = [r['ng_actual'] - r['ng_predicted']
                           for r in recent_results
                           if not np.isnan(r['ng_predicted'])]
            if len(recent_errors) >= 5:
                error_std = np.std(recent_errors)
            else:
                # Fall back to training residuals
                train_pred = model.predict(X_train)
                error_std = np.std(y_train - train_pred)
        else:
            # Use training residuals
            train_pred = model.predict(X_train)
            error_std = np.std(y_train - train_pred)

        results.append({
            'date': current['date'],
            'hdd_6_10d': current['hdd_6_10d'],
            'ng_actual': current['ng_price'],
            'ng_predicted': prediction,
            'error_estimate': error_std,
            'lower_1sigma': prediction - error_std,
            'upper_1sigma': prediction + error_std,
            'lower_2sigma': prediction - 2*error_std,
            'upper_2sigma': prediction + 2*error_std,
        })

    return pd.DataFrame(results)


def create_visualization(df, ng_daily=None):
    """Create the online algorithm visualization with error bars."""
    print("Creating visualization...")

    valid = df.dropna(subset=['ng_predicted'])

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # =========================================
    # Plot 1: Full time series with error bands
    # =========================================
    ax1 = axes[0]

    # Plot daily NG prices as background if available
    if ng_daily is not None:
        ng_plot = ng_daily[(ng_daily.index >= valid['date'].min()) &
                           (ng_daily.index <= valid['date'].max())]
        ax1.plot(ng_plot.index, ng_plot['Close'], color='gray', linewidth=0.8,
                 alpha=0.5, label='Daily NG', zorder=1)

    # 2-sigma band (95% confidence)
    ax1.fill_between(valid['date'], valid['lower_2sigma'], valid['upper_2sigma'],
                     color='lightblue', alpha=0.4, label='95% CI (±2σ)')

    # 1-sigma band (68% confidence)
    ax1.fill_between(valid['date'], valid['lower_1sigma'], valid['upper_1sigma'],
                     color='cornflowerblue', alpha=0.5, label='68% CI (±1σ)')

    # Prediction line with markers
    ax1.plot(valid['date'], valid['ng_predicted'], 'b-', linewidth=1.5,
             label='Online Prediction', zorder=3)
    ax1.scatter(valid['date'], valid['ng_predicted'], color='blue', s=15, zorder=3, alpha=0.7)

    # Actual prices at prediction points
    ax1.scatter(valid['date'], valid['ng_actual'], color='black', s=20,
             alpha=0.8, label='Actual NG (weekly)', zorder=4, marker='o')

    ax1.set_ylabel('NG Price ($/MMBtu)', fontsize=11)
    ax1.set_title('Online Fair Value Algorithm: Predictions with Confidence Intervals',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # =========================================
    # Plot 2: Zoomed recent period (last 52 weeks)
    # =========================================
    ax2 = axes[1]

    recent = valid.tail(52)  # Last year

    # Daily NG prices as background
    if ng_daily is not None:
        ng_recent = ng_daily[(ng_daily.index >= recent['date'].min()) &
                             (ng_daily.index <= recent['date'].max())]
        ax2.plot(ng_recent.index, ng_recent['Close'], color='gray', linewidth=1,
                 alpha=0.6, label='Daily NG', zorder=1)

    # 2-sigma band
    ax2.fill_between(recent['date'], recent['lower_2sigma'], recent['upper_2sigma'],
                     color='lightblue', alpha=0.4, label='95% CI')

    # 1-sigma band
    ax2.fill_between(recent['date'], recent['lower_1sigma'], recent['upper_1sigma'],
                     color='cornflowerblue', alpha=0.5, label='68% CI')

    # Prediction line with markers
    ax2.plot(recent['date'], recent['ng_predicted'], 'b-', linewidth=2,
             label='Online Prediction', zorder=3)
    ax2.scatter(recent['date'], recent['ng_predicted'], color='blue', s=30, zorder=3)

    # Actual prices at prediction points
    ax2.scatter(recent['date'], recent['ng_actual'], color='black', s=40,
             alpha=0.8, label='Actual (weekly)', zorder=4, marker='o')

    # Highlight when actual is outside confidence band
    for _, row in recent.iterrows():
        if row['ng_actual'] > row['upper_2sigma']:
            ax2.scatter(row['date'], row['ng_actual'], color='red', s=60,
                       zorder=5, marker='^', edgecolors='darkred', linewidths=1)
        elif row['ng_actual'] < row['lower_2sigma']:
            ax2.scatter(row['date'], row['ng_actual'], color='green', s=60,
                       zorder=5, marker='v', edgecolors='darkgreen', linewidths=1)

    ax2.set_ylabel('NG Price ($/MMBtu)', fontsize=11)
    ax2.set_title('Last 12 Months: Red▲=Overvalued, Green▼=Undervalued (outside 95% CI)',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # =========================================
    # Plot 3: Prediction error over time
    # =========================================
    ax3 = axes[2]

    valid['error'] = valid['ng_actual'] - valid['ng_predicted']
    valid['error_pct'] = valid['error'] / valid['ng_predicted'] * 100

    # Color by sign
    colors = ['green' if e < 0 else 'red' for e in valid['error_pct']]
    ax3.bar(valid['date'], valid['error_pct'], color=colors, alpha=0.6, width=5)

    # Rolling MAE
    valid['abs_error_pct'] = valid['error_pct'].abs()
    rolling_mae = valid['abs_error_pct'].rolling(window=12, min_periods=4).mean()
    ax3.plot(valid['date'], rolling_mae, 'purple', linewidth=2.5,
             label=f'Rolling 12-week MAE')
    ax3.plot(valid['date'], -rolling_mae, 'purple', linewidth=2.5)

    # Reference lines
    ax3.axhline(y=0, color='black', linewidth=1)
    overall_mae = valid['abs_error_pct'].mean()
    ax3.axhline(y=overall_mae, color='gray', linestyle='--', alpha=0.7,
                label=f'Overall MAE: {overall_mae:.1f}%')
    ax3.axhline(y=-overall_mae, color='gray', linestyle='--', alpha=0.7)

    ax3.set_ylabel('Prediction Error (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Online Algorithm Error: Green=Undervalued, Red=Overvalued',
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fair_value_online.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: fair_value_online.png")

    return valid


def print_statistics(df):
    """Print algorithm performance statistics."""
    valid = df.dropna(subset=['ng_predicted'])

    print("\n" + "=" * 70)
    print("ONLINE ALGORITHM PERFORMANCE")
    print("=" * 70)

    valid['error'] = valid['ng_actual'] - valid['ng_predicted']
    valid['error_pct'] = valid['error'] / valid['ng_predicted'] * 100
    valid['abs_error_pct'] = valid['error_pct'].abs()

    print(f"\nOverall Statistics (n={len(valid)}):")
    print(f"  MAE: ${valid['error'].abs().mean():.3f} ({valid['abs_error_pct'].mean():.1f}%)")
    print(f"  RMSE: ${np.sqrt((valid['error']**2).mean()):.3f}")
    print(f"  Median Error: {valid['abs_error_pct'].median():.1f}%")

    print(f"\nConfidence Interval Coverage:")
    within_1sigma = ((valid['ng_actual'] >= valid['lower_1sigma']) &
                     (valid['ng_actual'] <= valid['upper_1sigma'])).mean() * 100
    within_2sigma = ((valid['ng_actual'] >= valid['lower_2sigma']) &
                     (valid['ng_actual'] <= valid['upper_2sigma'])).mean() * 100
    print(f"  Within 1σ band: {within_1sigma:.1f}% (expected: 68%)")
    print(f"  Within 2σ band: {within_2sigma:.1f}% (expected: 95%)")

    print(f"\nPrediction Accuracy Buckets:")
    print(f"  Within ±5%:  {(valid['abs_error_pct'] <= 5).mean()*100:.1f}%")
    print(f"  Within ±10%: {(valid['abs_error_pct'] <= 10).mean()*100:.1f}%")
    print(f"  Within ±15%: {(valid['abs_error_pct'] <= 15).mean()*100:.1f}%")
    print(f"  Within ±20%: {(valid['abs_error_pct'] <= 20).mean()*100:.1f}%")
    print(f"  Within ±30%: {(valid['abs_error_pct'] <= 30).mean()*100:.1f}%")

    print(f"\nBy Year:")
    valid['year'] = valid['date'].dt.year
    for year in sorted(valid['year'].unique()):
        year_df = valid[valid['year'] == year]
        if len(year_df) >= 5:
            mae = year_df['error'].abs().mean()
            mae_pct = year_df['abs_error_pct'].mean()
            within_2s = ((year_df['ng_actual'] >= year_df['lower_2sigma']) &
                        (year_df['ng_actual'] <= year_df['upper_2sigma'])).mean() * 100
            print(f"  {year}: MAE=${mae:.2f} ({mae_pct:.1f}%), 95% coverage={within_2s:.0f}%, n={len(year_df)}")

    # Current prediction
    latest = valid.iloc[-1]
    print(f"\n" + "=" * 70)
    print("CURRENT PREDICTION")
    print("=" * 70)
    print(f"\nDate: {latest['date'].date()}")
    print(f"HDD 6-10d: {latest['hdd_6_10d']:.1f}")
    print(f"\nActual NG:    ${latest['ng_actual']:.3f}")
    print(f"Predicted:    ${latest['ng_predicted']:.3f}")
    print(f"Error:        {latest['error_pct']:+.1f}%")
    print(f"\n68% CI: ${latest['lower_1sigma']:.2f} - ${latest['upper_1sigma']:.2f}")
    print(f"95% CI: ${latest['lower_2sigma']:.2f} - ${latest['upper_2sigma']:.2f}")

    if latest['ng_actual'] > latest['upper_2sigma']:
        signal = "OVERVALUED (outside 95% CI)"
    elif latest['ng_actual'] < latest['lower_2sigma']:
        signal = "UNDERVALUED (outside 95% CI)"
    elif latest['ng_actual'] > latest['upper_1sigma']:
        signal = "Slightly overvalued (outside 68% CI)"
    elif latest['ng_actual'] < latest['lower_1sigma']:
        signal = "Slightly undervalued (outside 68% CI)"
    else:
        signal = "FAIR VALUE (within 68% CI)"

    print(f"\nSignal: {signal}")
    print("=" * 70)


def main():
    # Load and merge data
    hdd_df, ng_df = load_all_data()
    df = merge_data(hdd_df, ng_df)

    print(f"  Total samples: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Run online algorithm
    results = online_algorithm(df, train_window=26, error_window=12)

    # Create visualization
    create_visualization(results, ng_daily=ng_df)

    # Print statistics
    print_statistics(results)

    plt.show()


if __name__ == "__main__":
    main()
