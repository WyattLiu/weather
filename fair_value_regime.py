#!/usr/bin/env python3
"""
Regime-Aware Fair Value Model
Accounts for structural market regimes in NG pricing.
Uses rolling windows to adapt to changing market conditions.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("REGIME-AWARE FAIR VALUE ANALYSIS")
print("=" * 70)

HIST_DIR = "/home/wyatt/weather/historical"
OUTPUT_DIR = "/home/wyatt/weather"
BASE_TEMP = 18.3


def load_all_data():
    """Load all historical HDD and NG data."""
    print("\nLoading data...")

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
                    'hdd_1_5d': row.get('hdd_1_5d', np.nan),
                    'hdd_6_10d': row.get('hdd_6_10d', np.nan),
                    'hdd_8_14d': row.get('hdd_8_14d', np.nan)
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

            days_1_5 = [daily_hdd.get(d, 0) for d in range(1, 6)]
            days_6_10 = [daily_hdd.get(d, 0) for d in range(6, 11)]
            days_8_14 = [daily_hdd.get(d, 0) for d in range(8, 15)]

            all_hdd.append({
                'date': date,
                'hdd_1_5d': np.mean(days_1_5) if days_1_5 else np.nan,
                'hdd_6_10d': np.mean(days_6_10) if days_6_10 else np.nan,
                'hdd_8_14d': np.mean(days_8_14) if days_8_14 else np.nan
            })
        except:
            continue

    hdd_df = pd.DataFrame(all_hdd)
    hdd_df = hdd_df.drop_duplicates('date').sort_values('date').reset_index(drop=True)
    print(f"  HDD samples: {len(hdd_df)}")

    # Load NG prices
    ng = yf.Ticker("NG=F")
    ng_df = ng.history(period="max", interval="1d")
    if ng_df.index.tz is not None:
        ng_df.index = ng_df.index.tz_localize(None)
    print(f"  NG prices: {len(ng_df)} bars")

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
                    'hdd_1_5d': row['hdd_1_5d'],
                    'hdd_6_10d': row['hdd_6_10d'],
                    'hdd_8_14d': row['hdd_8_14d'],
                    'ng_price': ng_price,
                    'month': date.month,
                    'year': date.year
                })
        except:
            continue

    df = pd.DataFrame(matched)
    print(f"  Matched samples: {len(df)}")
    return df


def add_features(df):
    """Add derived features for regime detection."""
    df = df.sort_values('date').reset_index(drop=True)

    # Rolling price statistics (regime indicators)
    df['ng_ma_30'] = df['ng_price'].rolling(window=4, min_periods=1).mean()  # ~30 days (weekly data)
    df['ng_ma_90'] = df['ng_price'].rolling(window=12, min_periods=1).mean()  # ~90 days
    df['ng_volatility'] = df['ng_price'].rolling(window=8, min_periods=1).std()

    # Price momentum
    df['ng_pct_change'] = df['ng_price'].pct_change()

    # Season indicator
    df['is_winter'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)
    df['is_summer'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)

    # HDD change from previous period
    df['hdd_6_10d_change'] = df['hdd_6_10d'].diff()

    return df


def rolling_regression_model(df, window=26):
    """
    Rolling window regression - adapts to changing market regimes.
    Uses past 'window' weeks to estimate current fair value.
    """
    print(f"\nBuilding rolling {window*7}-day regression model...")

    df = df.sort_values('date').reset_index(drop=True)

    # Store results
    fair_values = []
    slopes = []
    intercepts = []
    r2_scores = []

    for i in range(len(df)):
        if i < window:
            # Not enough history - use expanding window
            train_df = df.iloc[:i+1]
        else:
            # Use rolling window
            train_df = df.iloc[i-window:i]

        if len(train_df) < 10:
            fair_values.append(np.nan)
            slopes.append(np.nan)
            intercepts.append(np.nan)
            r2_scores.append(np.nan)
            continue

        X = train_df[['hdd_6_10d']].values
        y = train_df['ng_price'].values

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        # Predict current fair value
        current_hdd = df.iloc[i]['hdd_6_10d']
        fv = model.predict([[current_hdd]])[0]

        fair_values.append(fv)
        slopes.append(model.coef_[0])
        intercepts.append(model.intercept_)
        r2_scores.append(r2_score(y, model.predict(X)))

    df['fair_value_rolling'] = fair_values
    df['slope_rolling'] = slopes
    df['intercept_rolling'] = intercepts
    df['r2_rolling'] = r2_scores

    # Calculate residuals
    df['residual_rolling'] = df['ng_price'] - df['fair_value_rolling']
    df['residual_pct_rolling'] = df['residual_rolling'] / df['fair_value_rolling'] * 100

    return df


def seasonal_regime_model(df):
    """
    Separate models for different seasons and price regimes.
    """
    print("\nBuilding seasonal/regime-aware model...")

    # Define regimes based on price level
    df['regime'] = pd.cut(df['ng_ma_90'],
                          bins=[0, 3, 5, 8, 15],
                          labels=['Low (<$3)', 'Normal ($3-5)', 'High ($5-8)', 'Crisis (>$8)'])

    # Fit separate models per season
    results = []

    for season in ['winter', 'summer', 'shoulder']:
        if season == 'winter':
            mask = df['is_winter'] == 1
        elif season == 'summer':
            mask = df['is_summer'] == 1
        else:
            mask = (df['is_winter'] == 0) & (df['is_summer'] == 0)

        season_df = df[mask]
        if len(season_df) < 15:
            continue

        X = season_df[['hdd_6_10d']].values
        y = season_df['ng_price'].values

        model = LinearRegression()
        model.fit(X, y)

        r2 = model.score(X, y)
        mae = mean_absolute_error(y, model.predict(X))

        results.append({
            'season': season,
            'samples': len(season_df),
            'intercept': model.intercept_,
            'slope': model.coef_[0],
            'r2': r2,
            'mae': mae,
            'mae_pct': mae / y.mean() * 100
        })

        print(f"  {season.capitalize():10s}: NG = ${model.intercept_:.2f} + ${model.coef_[0]:.3f}×HDD, R²={r2:.3f}, MAE={mae:.2f} ({mae/y.mean()*100:.0f}%), n={len(season_df)}")

    return pd.DataFrame(results)


def analyze_model_accuracy(df):
    """Analyze accuracy of rolling model."""
    print("\n" + "=" * 70)
    print("ROLLING MODEL ACCURACY ANALYSIS")
    print("=" * 70)

    # Filter to where we have predictions
    valid = df.dropna(subset=['fair_value_rolling', 'residual_pct_rolling'])

    print(f"\nOverall (n={len(valid)}):")
    print(f"  MAE: ${valid['residual_rolling'].abs().mean():.3f} ({valid['residual_pct_rolling'].abs().mean():.1f}%)")
    print(f"  RMSE: ${np.sqrt((valid['residual_rolling']**2).mean()):.3f}")
    print(f"  Median Error: {valid['residual_pct_rolling'].abs().median():.1f}%")

    print(f"\nAccuracy Distribution:")
    print(f"  Within ±5%:  {(valid['residual_pct_rolling'].abs() <= 5).mean()*100:.1f}%")
    print(f"  Within ±10%: {(valid['residual_pct_rolling'].abs() <= 10).mean()*100:.1f}%")
    print(f"  Within ±15%: {(valid['residual_pct_rolling'].abs() <= 15).mean()*100:.1f}%")
    print(f"  Within ±20%: {(valid['residual_pct_rolling'].abs() <= 20).mean()*100:.1f}%")

    print(f"\nBy Year:")
    for year in sorted(valid['year'].unique()):
        year_df = valid[valid['year'] == year]
        if len(year_df) >= 5:
            mae = year_df['residual_rolling'].abs().mean()
            mae_pct = year_df['residual_pct_rolling'].abs().mean()
            avg_r2 = year_df['r2_rolling'].mean()
            print(f"  {year}: MAE=${mae:.2f} ({mae_pct:.1f}%), Avg R²={avg_r2:.3f}, n={len(year_df)}")

    print(f"\nBy Season:")
    valid['season'] = valid['month'].apply(lambda m: 'Winter' if m in [11,12,1,2,3] else 'Summer' if m in [5,6,7,8,9] else 'Shoulder')
    for season in ['Winter', 'Shoulder', 'Summer']:
        season_df = valid[valid['season'] == season]
        if len(season_df) > 0:
            mae = season_df['residual_rolling'].abs().mean()
            mae_pct = season_df['residual_pct_rolling'].abs().mean()
            print(f"  {season:10s}: MAE=${mae:.2f} ({mae_pct:.1f}%), n={len(season_df)}")

    # Worst predictions
    print(f"\nTop 10 Worst Predictions (Rolling Model):")
    worst = valid.nlargest(10, 'residual_pct_rolling', keep='first')[['date', 'hdd_6_10d', 'ng_price', 'fair_value_rolling', 'residual_pct_rolling']]
    for _, row in worst.iterrows():
        print(f"  {row['date'].date()}: HDD={row['hdd_6_10d']:.1f}, Actual=${row['ng_price']:.2f}, FV=${row['fair_value_rolling']:.2f}, Error={row['residual_pct_rolling']:+.1f}%")

    return valid


def create_visualizations(df):
    """Create comprehensive visualizations."""
    print("\nCreating visualizations...")

    valid = df.dropna(subset=['fair_value_rolling'])

    fig = plt.figure(figsize=(18, 14))

    # 1. Price vs Fair Value over time
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(valid['date'], valid['ng_price'], 'b-', linewidth=1.5, alpha=0.8, label='Actual NG')
    ax1.plot(valid['date'], valid['fair_value_rolling'], 'g--', linewidth=1.5, alpha=0.8, label='Fair Value (Rolling)')
    ax1.fill_between(valid['date'], valid['fair_value_rolling'], valid['ng_price'],
                     where=valid['ng_price'] > valid['fair_value_rolling'],
                     color='red', alpha=0.2, label='Overvalued')
    ax1.fill_between(valid['date'], valid['fair_value_rolling'], valid['ng_price'],
                     where=valid['ng_price'] < valid['fair_value_rolling'],
                     color='green', alpha=0.2, label='Undervalued')
    ax1.set_ylabel('NG Price ($/MMBtu)')
    ax1.set_title('Natural Gas: Actual vs Rolling Fair Value', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Rolling R² over time
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(valid['date'], valid['r2_rolling'], 'purple', linewidth=1.5)
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good fit (0.5)')
    ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Weak fit (0.2)')
    ax2.set_ylabel('R² (Rolling Window)')
    ax2.set_title('Model Fit Quality Over Time', fontweight='bold')
    ax2.set_ylim(-0.1, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Rolling slope over time (HDD sensitivity)
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(valid['date'], valid['slope_rolling'], 'darkblue', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_ylabel('HDD Sensitivity ($/HDD)')
    ax3.set_title('Rolling HDD Price Sensitivity', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Mispricing over time
    ax4 = fig.add_subplot(3, 2, 4)
    colors = ['green' if r < 0 else 'red' for r in valid['residual_pct_rolling']]
    ax4.bar(valid['date'], valid['residual_pct_rolling'], color=colors, alpha=0.7, width=5)
    ax4.axhline(y=0, color='black', linewidth=1)
    std = valid['residual_pct_rolling'].std()
    ax4.axhline(y=std, color='gray', linestyle='--', alpha=0.5, label=f'±1σ ({std:.0f}%)')
    ax4.axhline(y=-std, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=2*std, color='gray', linestyle=':', alpha=0.3, label=f'±2σ ({2*std:.0f}%)')
    ax4.axhline(y=-2*std, color='gray', linestyle=':', alpha=0.3)
    ax4.set_ylabel('Mispricing (%)')
    ax4.set_title('Mispricing Over Time (Rolling Model)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Error distribution
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.hist(valid['residual_pct_rolling'], bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
    mean_err = valid['residual_pct_rolling'].mean()
    ax5.axvline(x=mean_err, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_err:.1f}%')
    ax5.set_xlabel('Prediction Error (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Error Distribution (Std: {valid["residual_pct_rolling"].std():.1f}%)', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Error by month
    ax6 = fig.add_subplot(3, 2, 6)
    monthly = valid.groupby('month')['residual_pct_rolling'].apply(lambda x: x.abs().mean())
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colors = ['blue' if m in [11,12,1,2,3] else 'orange' if m in [5,6,7,8,9] else 'gray' for m in range(1,13)]
    ax6.bar(range(1, 13), [monthly.get(m, 0) for m in range(1, 13)], color=colors, alpha=0.7)
    ax6.set_xticks(range(1, 13))
    ax6.set_xticklabels(months, rotation=45)
    ax6.set_ylabel('Mean Absolute Error (%)')
    ax6.set_title('Prediction Error by Month (Blue=Winter, Orange=Summer)', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fair_value_regime.png'), dpi=150, bbox_inches='tight')
    print("  Saved: fair_value_regime.png")


def current_signal(df):
    """Generate current trading signal."""
    print("\n" + "=" * 70)
    print("CURRENT SIGNAL")
    print("=" * 70)

    valid = df.dropna(subset=['fair_value_rolling'])
    latest = valid.iloc[-1]

    residual_std = valid['residual_pct_rolling'].std()
    z_score = latest['residual_pct_rolling'] / residual_std

    print(f"\nLatest ({latest['date'].date()}):")
    print(f"  HDD 6-10d:    {latest['hdd_6_10d']:.1f}")
    print(f"  NG Actual:    ${latest['ng_price']:.3f}")
    print(f"  Fair Value:   ${latest['fair_value_rolling']:.3f}")
    print(f"  Mispricing:   {latest['residual_pct_rolling']:+.1f}% ({z_score:+.2f}σ)")

    # Model parameters for current regime
    print(f"\n  Current Model Parameters:")
    print(f"    Intercept: ${latest['intercept_rolling']:.3f}")
    print(f"    Slope:     ${latest['slope_rolling']:.4f}/HDD")
    print(f"    R²:        {latest['r2_rolling']:.3f}")

    # Historical context
    pct_rank = (valid['residual_pct_rolling'] < latest['residual_pct_rolling']).mean() * 100
    print(f"\n  Historical Percentile: {pct_rank:.0f}%")

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

    # Model confidence
    if latest['r2_rolling'] > 0.5:
        confidence = "HIGH"
    elif latest['r2_rolling'] > 0.2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    print(f"  Model Confidence: {confidence} (R²={latest['r2_rolling']:.2f})")

    # Recent accuracy
    recent = valid.tail(12)  # Last 3 months
    recent_mae = recent['residual_pct_rolling'].abs().mean()
    print(f"  Recent (12-week) MAE: {recent_mae:.1f}%")


def main():
    # Load data
    hdd_df, ng_df = load_all_data()

    # Merge
    df = merge_data(hdd_df, ng_df)
    if len(df) < 30:
        print("Insufficient data")
        return

    # Add features
    df = add_features(df)

    # Seasonal model analysis
    seasonal_results = seasonal_regime_model(df)

    # Rolling regression
    df = rolling_regression_model(df, window=26)  # 6-month rolling window

    # Analyze accuracy
    analyze_model_accuracy(df)

    # Visualizations
    create_visualizations(df)

    # Current signal
    current_signal(df)

    print("\n" + "=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
