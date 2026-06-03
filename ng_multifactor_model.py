#!/usr/bin/env python3
"""
Multi-Factor NG Fair Value Model
Combines: Storage levels, HDD forecasts, seasonality, and momentum
Tests whether storage adds predictive power over naive baselines
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

HIST_DIR = "/home/wyatt/weather/historical"
OUTPUT_DIR = "/home/wyatt/weather"


def load_all_data():
    """Load storage, HDD, and NG price data."""
    print("Loading data...")

    # 1. Storage data
    storage = pd.read_csv(os.path.join(OUTPUT_DIR, 'ng_storage_weekly.csv'))
    storage['date'] = pd.to_datetime(storage['date'])
    storage = storage.set_index('date')
    print(f"  Storage: {len(storage)} weekly obs ({storage.index.min().date()} to {storage.index.max().date()})")

    # 2. HDD data
    all_hdd = []
    for f in sorted(glob.glob(os.path.join(HIST_DIR, "gfs_hist_*.csv"))):
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

    hdd_df = pd.DataFrame(all_hdd).drop_duplicates('date').sort_values('date')
    hdd_df = hdd_df.set_index('date')
    print(f"  HDD: {len(hdd_df)} weekly obs ({hdd_df.index.min().date()} to {hdd_df.index.max().date()})")

    # 3. NG prices (daily)
    ng = yf.Ticker("NG=F")
    ng_df = ng.history(period="max", interval="1d")
    if ng_df.index.tz is not None:
        ng_df.index = ng_df.index.tz_localize(None)
    print(f"  NG prices: {len(ng_df)} daily obs")

    return storage, hdd_df, ng_df


def merge_all_data(storage, hdd_df, ng_df):
    """Merge all data sources on weekly basis."""
    print("\nMerging data...")

    # Resample NG to weekly (Friday close)
    ng_weekly = ng_df['Close'].resample('W-FRI').last()

    # Create master dataframe
    master = pd.DataFrame(index=ng_weekly.index)
    master['ng_price'] = ng_weekly

    # Merge storage (find closest date within 5 days)
    storage_values = []
    for date in master.index:
        closest = storage.index[storage.index.get_indexer([date], method='nearest')]
        if len(closest) > 0:
            diff = abs((closest[0] - date).days)
            if diff <= 7:
                storage_values.append(storage.loc[closest[0], 'storage_bcf'])
            else:
                storage_values.append(np.nan)
        else:
            storage_values.append(np.nan)
    master['storage_bcf'] = storage_values

    # Merge HDD (find closest date within 5 days)
    hdd_values = []
    for date in master.index:
        closest = hdd_df.index[hdd_df.index.get_indexer([date], method='nearest')]
        if len(closest) > 0:
            diff = abs((closest[0] - date).days)
            if diff <= 7:
                hdd_values.append(hdd_df.loc[closest[0], 'hdd_6_10d'])
            else:
                hdd_values.append(np.nan)
        else:
            hdd_values.append(np.nan)
    master['hdd_6_10d'] = hdd_values

    # Drop rows with missing data
    master = master.dropna()
    print(f"  Merged dataset: {len(master)} observations")
    print(f"  Date range: {master.index.min().date()} to {master.index.max().date()}")

    return master


def add_features(df):
    """Add derived features."""
    df = df.copy()

    # Storage features
    df['storage_weekly_change'] = df['storage_bcf'].diff()  # Weekly change (BCF)

    # Calculate storage vs seasonal norm (simpler approach)
    df['week_of_year'] = df.index.isocalendar().week.astype(int)

    # Use historical average for each week of year
    week_means = df.groupby('week_of_year')['storage_bcf'].transform('mean')
    df['storage_vs_avg'] = (df['storage_bcf'] - week_means) / week_means * 100

    # Seasonality
    df['month'] = df.index.month
    df['is_winter'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)
    df['is_injection'] = df['month'].isin([4, 5, 6, 7, 8, 9, 10]).astype(int)

    # Price momentum (shorter windows)
    df['ng_ma_4w'] = df['ng_price'].rolling(4, min_periods=2).mean()
    df['ng_momentum'] = df['ng_price'] / df['ng_ma_4w'] - 1

    # Lagged price
    df['ng_lag_1'] = df['ng_price'].shift(1)

    # Drop only rows with critical NaN values
    critical_cols = ['ng_price', 'storage_bcf', 'hdd_6_10d', 'ng_lag_1']
    df = df.dropna(subset=critical_cols)

    # Fill remaining NaN with 0 for optional features
    df = df.fillna(0)

    return df


def test_model_validity(df):
    """Test whether features have predictive power."""
    print("\n" + "=" * 70)
    print("FEATURE PREDICTIVE POWER ANALYSIS")
    print("=" * 70)

    features = ['storage_bcf', 'storage_vs_avg', 'storage_weekly_change',
                'hdd_6_10d', 'is_winter', 'ng_lag_1']

    print("\nCorrelation with NG Price:")
    for feat in features:
        if feat in df.columns:
            corr, p = stats.pearsonr(df[feat], df['ng_price'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:25s}: r={corr:+.3f}, p={p:.4f} {sig}")

    print("\nCorrelation with NG Price CHANGE:")
    df['ng_change'] = df['ng_price'].pct_change()
    df_valid = df.dropna()
    for feat in features:
        if feat in df_valid.columns:
            feat_change = df_valid[feat].pct_change() if feat != 'ng_lag_1' else df_valid[feat]
            valid = pd.concat([feat_change, df_valid['ng_change']], axis=1).dropna()
            if len(valid) > 30:
                corr, p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  Δ{feat:24s}: r={corr:+.3f}, p={p:.4f} {sig}")


def online_backtest(df, train_window=52):
    """
    Online backtest comparing multiple models.
    At each point, only uses data available up to that time.
    """
    print("\n" + "=" * 70)
    print(f"ONLINE BACKTEST (train_window={train_window} weeks)")
    print("=" * 70)

    results = []

    feature_cols = ['storage_bcf', 'storage_vs_avg', 'hdd_6_10d', 'is_winter', 'ng_lag_1']

    for i in range(train_window, len(df)):
        train = df.iloc[i-train_window:i]
        test_row = df.iloc[i]

        actual = test_row['ng_price']

        # Model 1: Naive - just use last price
        pred_naive = train['ng_price'].iloc[-1]

        # Model 2: Rolling mean
        pred_mean = train['ng_price'].mean()

        # Model 3: HDD only
        X_hdd = train[['hdd_6_10d']].values
        y = train['ng_price'].values
        model_hdd = Ridge(alpha=1.0)
        model_hdd.fit(X_hdd, y)
        pred_hdd = model_hdd.predict([[test_row['hdd_6_10d']]])[0]

        # Model 4: Storage only
        X_stor = train[['storage_bcf']].values
        model_stor = Ridge(alpha=1.0)
        model_stor.fit(X_stor, y)
        pred_stor = model_stor.predict([[test_row['storage_bcf']]])[0]

        # Model 5: Multi-factor (Storage + HDD + Lag)
        X_multi = train[feature_cols].values
        model_multi = Ridge(alpha=1.0)
        model_multi.fit(X_multi, y)
        pred_multi = model_multi.predict([test_row[feature_cols].values])[0]

        # Model 6: Gradient Boosting
        model_gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        model_gb.fit(X_multi, y)
        pred_gb = model_gb.predict([test_row[feature_cols].values])[0]

        results.append({
            'date': df.index[i],
            'actual': actual,
            'pred_naive': pred_naive,
            'pred_mean': pred_mean,
            'pred_hdd': pred_hdd,
            'pred_storage': pred_stor,
            'pred_multi': pred_multi,
            'pred_gb': pred_gb,
        })

    results_df = pd.DataFrame(results)

    # Calculate errors
    models = ['naive', 'mean', 'hdd', 'storage', 'multi', 'gb']
    print("\nModel Performance (Out-of-Sample):")
    print("-" * 70)
    print(f"{'Model':<20} {'MAE ($)':<12} {'MAE (%)':<12} {'vs Naive':<12} {'Direction':<10}")
    print("-" * 70)

    naive_mae = (results_df['actual'] - results_df['pred_naive']).abs().mean()

    for model in models:
        pred_col = f'pred_{model}'
        error = (results_df['actual'] - results_df[pred_col]).abs()
        mae = error.mean()
        mae_pct = (error / results_df['actual']).mean() * 100

        # Directional accuracy
        actual_dir = (results_df['actual'].diff() > 0).astype(int)
        pred_dir = (results_df[pred_col].diff() > 0).astype(int)
        dir_acc = (actual_dir == pred_dir).mean() * 100

        improvement = (naive_mae - mae) / naive_mae * 100 if model != 'naive' else 0

        print(f"{model:<20} ${mae:<11.3f} {mae_pct:<11.1f}% {improvement:+.1f}%{'':<6} {dir_acc:.1f}%")

    return results_df


def analyze_feature_importance(df, train_window=52):
    """Analyze which features matter most."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    feature_cols = ['storage_bcf', 'storage_vs_avg', 'hdd_6_10d', 'is_winter', 'ng_lag_1']

    # Use last portion of data for final model
    train = df.iloc[-train_window*2:-train_window]
    test = df.iloc[-train_window:]

    X_train = train[feature_cols].values
    y_train = train['ng_price'].values

    # Fit Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nRandom Forest Feature Importance:")
    for _, row in importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:25s}: {row['importance']:.3f} {bar}")

    # Also show Ridge coefficients (standardized)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y_train)

    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': ridge.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    print("\nRidge Regression Coefficients (standardized):")
    for _, row in coef_df.iterrows():
        print(f"  {row['feature']:25s}: {row['coefficient']:+.3f}")


def create_visualization(df, results_df):
    """Create visualization of model performance."""
    print("\nCreating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Plot 1: Storage vs NG Price over time
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ax1.plot(df.index, df['ng_price'], 'b-', linewidth=1.5, label='NG Price')
    ax1_twin.plot(df.index, df['storage_bcf'], 'g-', linewidth=1.5, alpha=0.7, label='Storage')

    ax1.set_ylabel('NG Price ($/MMBtu)', color='blue')
    ax1_twin.set_ylabel('Storage (BCF)', color='green')
    ax1.set_title('Natural Gas Price vs Storage Levels', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Model predictions comparison (recent period)
    ax2 = axes[1]
    recent = results_df.tail(104)  # Last 2 years

    ax2.plot(recent['date'], recent['actual'], 'k-', linewidth=2, label='Actual', zorder=5)
    ax2.plot(recent['date'], recent['pred_naive'], 'gray', linewidth=1, alpha=0.5, label='Naive (last price)')
    ax2.plot(recent['date'], recent['pred_multi'], 'b-', linewidth=1.5, label='Multi-factor', zorder=4)
    ax2.plot(recent['date'], recent['pred_gb'], 'r--', linewidth=1.5, label='Gradient Boosting', zorder=3)

    ax2.set_ylabel('NG Price ($/MMBtu)')
    ax2.set_title('Model Predictions: Last 2 Years', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction errors
    ax3 = axes[2]

    recent['error_naive'] = recent['actual'] - recent['pred_naive']
    recent['error_multi'] = recent['actual'] - recent['pred_multi']

    ax3.bar(recent['date'], recent['error_naive'], alpha=0.4, color='gray', label='Naive Error', width=5)
    ax3.bar(recent['date'], recent['error_multi'], alpha=0.6, color='blue', label='Multi-factor Error', width=5)
    ax3.axhline(y=0, color='black', linewidth=1)

    ax3.set_ylabel('Prediction Error ($)')
    ax3.set_xlabel('Date')
    ax3.set_title('Prediction Errors: Multi-factor vs Naive', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ng_multifactor.png'), dpi=150, bbox_inches='tight')
    print("  Saved: ng_multifactor.png")


def current_signal(df, results_df):
    """Generate current trading signal."""
    print("\n" + "=" * 70)
    print("CURRENT SIGNAL")
    print("=" * 70)

    latest = df.iloc[-1]
    latest_result = results_df.iloc[-1]

    print(f"\nDate: {df.index[-1].date()}")
    print(f"\nInputs:")
    print(f"  Storage:    {latest['storage_bcf']:.0f} BCF")
    print(f"  Storage vs 5yr avg: {latest['storage_vs_avg']:+.1f}%")
    print(f"  HDD 6-10d:  {latest['hdd_6_10d']:.1f}")
    print(f"  NG Price:   ${latest['ng_price']:.3f}")

    print(f"\nModel Predictions:")
    print(f"  Multi-factor: ${latest_result['pred_multi']:.3f}")
    print(f"  Gradient Boost: ${latest_result['pred_gb']:.3f}")

    avg_pred = (latest_result['pred_multi'] + latest_result['pred_gb']) / 2
    mispricing = (latest['ng_price'] - avg_pred) / avg_pred * 100

    print(f"\n  Avg Prediction: ${avg_pred:.3f}")
    print(f"  Mispricing: {mispricing:+.1f}%")

    if mispricing > 15:
        signal = "SELL (overvalued)"
    elif mispricing > 5:
        signal = "Slight sell bias"
    elif mispricing < -15:
        signal = "BUY (undervalued)"
    elif mispricing < -5:
        signal = "Slight buy bias"
    else:
        signal = "NEUTRAL (fair value)"

    print(f"\n  Signal: {signal}")


def main():
    # Load data
    storage, hdd_df, ng_df = load_all_data()

    # Merge
    df = merge_all_data(storage, hdd_df, ng_df)

    # Add features
    df = add_features(df)
    print(f"\nFinal dataset: {len(df)} observations with {len(df.columns)} features")

    # Test feature validity
    test_model_validity(df)

    # Run backtest
    results_df = online_backtest(df, train_window=52)

    # Feature importance
    analyze_feature_importance(df)

    # Visualize
    create_visualization(df, results_df)

    # Current signal
    current_signal(df, results_df)

    print("\n" + "=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
