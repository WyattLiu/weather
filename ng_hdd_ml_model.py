#!/usr/bin/env python3
"""
Machine Learning Model for Natural Gas Price Prediction
Based on HDD forecasts from GFS, ECMWF IFS, and AIFS.

Features:
- HDD forecasts from multiple models/timeframes
- HDD changes (momentum)
- Model agreement/disagreement
- Seasonal factors

Target:
- NG price change over next 24h, 48h, 1 week
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("NG PRICE PREDICTION MODEL")
print("Based on Multi-Model HDD Forecasts")
print("=" * 70)

OUTPUT_DIR = "/home/wyatt/weather"
BASE_TEMP = 18.3

# US bounding box
US_LAT_MIN, US_LAT_MAX = 25, 50
US_LON_MIN, US_LON_MAX = -125, -65


# ============================================
# Load HDD Data from GRIB2 files (existing data)
# ============================================
def load_grib2_data():
    """Load HDD data from existing GRIB2 files."""
    import xarray as xr

    print("\nLoading GRIB2 data...")
    all_data = []

    # Load GFS GRIB2 files
    gfs_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'gfs_*.grib2')))
    print(f"  GFS GRIB2: {len(gfs_files)} files")

    for filepath in gfs_files:
        try:
            ds = xr.open_dataset(filepath, engine='cfgrib',
                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})

            # Find temp variable
            temp_var = None
            for var in ds.data_vars:
                if 'TMP' in var.upper() or 'T2M' in var.upper() or var == 't':
                    temp_var = var
                    break
            if temp_var is None:
                continue

            temp_k = ds[temp_var]
            temp_c = temp_k - 273.15

            # Filter to US
            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'

            if ds[lon_name].max() > 180:
                ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
                ds = ds.sortby(lon_name)
                temp_c = ds[temp_var] - 273.15

            temp_us = temp_c.where(
                (ds[lat_name] >= US_LAT_MIN) & (ds[lat_name] <= US_LAT_MAX) &
                (ds[lon_name] >= US_LON_MIN) & (ds[lon_name] <= US_LON_MAX),
                drop=True
            )

            # Calculate daily HDD
            if 'step' in ds.dims:
                n_steps = min(ds.sizes['step'], 16)
            else:
                n_steps = 1

            daily_hdds = []
            for i in range(n_steps):
                try:
                    if 'step' in ds.dims:
                        day_temp = temp_us.isel(step=i)
                    else:
                        day_temp = temp_us
                    day_hdd = np.maximum(0, BASE_TEMP - day_temp)
                    daily_hdds.append(float(day_hdd.mean().values))
                except:
                    break

            # Parse datetime from filename
            basename = os.path.basename(filepath)
            parts = basename.replace('.grib2', '').split('_')
            date_str = hour_str = None
            for part in parts:
                if len(part) == 8 and part.isdigit():
                    date_str = part
                elif part.endswith('z'):
                    hour_str = part[:-1]

            if date_str and len(daily_hdds) >= 5:
                dt = pd.to_datetime(f"{date_str} {hour_str or '00'}:00")
                record = {
                    'datetime': dt,
                    'gfs_1_5d': sum(daily_hdds[:5]) if len(daily_hdds) >= 5 else np.nan,
                    'gfs_6_10d': sum(daily_hdds[5:10]) if len(daily_hdds) >= 10 else np.nan,
                    'gfs_8_14d': sum(daily_hdds[7:14]) if len(daily_hdds) >= 14 else np.nan,
                }
                all_data.append(record)

            ds.close()

        except Exception as e:
            continue

    # Load ECMWF GRIB2 files
    ecmwf_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'forecast_historical_*.grib2')))
    print(f"  ECMWF GRIB2: {len(ecmwf_files)} files")

    for filepath in ecmwf_files:
        try:
            ds = xr.open_dataset(filepath, engine='cfgrib')

            if 't2m' not in ds.data_vars:
                continue

            temp_c = ds['t2m'] - 273.15

            # Get step hours
            steps_hours = [int(s / np.timedelta64(1, 'h')) for s in ds['step'].values]

            # Calculate daily HDD for 4 days
            daily_hdds = []
            for day in range(10):  # Get up to 10 days if available
                start_h = day * 24
                end_h = (day + 1) * 24
                day_steps = [i for i, h in enumerate(steps_hours) if start_h <= h < end_h]
                if day_steps:
                    day_temp = temp_c.isel(step=day_steps).mean(dim='step')

                    # Filter to US
                    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
                    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

                    if ds[lon_name].max() > 180:
                        lon_min, lon_max = 235, 295
                    else:
                        lon_min, lon_max = -125, -65

                    day_temp_us = day_temp.where(
                        (ds[lat_name] >= US_LAT_MIN) & (ds[lat_name] <= US_LAT_MAX) &
                        (ds[lon_name] >= lon_min) & (ds[lon_name] <= lon_max),
                        drop=True
                    )

                    day_hdd = np.maximum(0, BASE_TEMP - day_temp_us)
                    daily_hdds.append(float(day_hdd.mean().values))

            # Parse datetime
            if 'time' in ds.coords:
                forecast_time = ds['time'].values
                forecast_date = str(forecast_time).split('T')[0]
                forecast_hour = str(forecast_time).split('T')[1][:2]
                dt = pd.to_datetime(f"{forecast_date} {forecast_hour}:00")

                if len(daily_hdds) >= 4:
                    record = {
                        'datetime': dt,
                        'ifs_1_5d': sum(daily_hdds[:5]) if len(daily_hdds) >= 5 else sum(daily_hdds[:4]),
                        'ifs_6_10d': sum(daily_hdds[5:10]) if len(daily_hdds) >= 10 else np.nan,
                    }
                    all_data.append(record)

            ds.close()

        except Exception as e:
            continue

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Group by datetime and aggregate
    grouped = df.groupby('datetime').first().reset_index()
    return grouped.sort_values('datetime')


# ============================================
# Load HDD Data from CSV files
# ============================================
def load_csv_data():
    """Load HDD data from Herbie-fetched CSV files."""
    print("\nLoading CSV data...")

    models = ['gfs', 'ifs', 'aifs']
    all_data = []

    for model in models:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{model}_*.csv")))
        print(f"  {model.upper()}: {len(files)} files")

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

                # Calculate period HDDs
                periods = {
                    f'{model}_1_5d': daily_hdd.loc[1:5].sum() if len(daily_hdd) > 5 else np.nan,
                    f'{model}_6_10d': daily_hdd.loc[6:10].sum() if len(daily_hdd) > 10 else np.nan,
                    f'{model}_8_14d': daily_hdd.loc[8:14].sum() if len(daily_hdd) > 14 else np.nan,
                }

                record = {'datetime': dt, 'model': model}
                record.update(periods)
                all_data.append(record)

            except Exception as e:
                continue

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Pivot to have one row per datetime with all model columns
    pivoted = df.pivot_table(
        index='datetime',
        values=[c for c in df.columns if c not in ['datetime', 'model']],
        aggfunc='first'
    ).reset_index()

    return pivoted.sort_values('datetime')


# ============================================
# Load NG Price Data
# ============================================
def load_ng_data(start_date, end_date):
    """Load NG front month prices."""
    print("\nFetching NG prices...")
    ng = yf.Ticker("NG=F")

    # Get max available data
    ng_data = ng.history(period="max", interval="1d")  # Daily for longer history

    if ng_data.index.tz is not None:
        ng_data.index = ng_data.index.tz_localize(None)

    # Filter to date range
    ng_data = ng_data[(ng_data.index >= start_date) & (ng_data.index <= end_date)]

    print(f"  Loaded {len(ng_data)} daily bars from {ng_data.index.min()} to {ng_data.index.max()}")
    return ng_data


# ============================================
# Feature Engineering
# ============================================
def create_features(hdd_df, ng_df):
    """Create ML features from HDD and NG data."""
    print("\nCreating features...")

    features = []

    for idx, row in hdd_df.iterrows():
        dt = row['datetime']

        # Find closest NG price
        try:
            time_diffs = abs(ng_df.index - dt)
            closest_idx = time_diffs.argmin()
            if time_diffs[closest_idx] > pd.Timedelta(days=1):
                continue

            ng_price = ng_df.iloc[closest_idx]['Close']

            # Future prices for target
            future_24h_idx = ng_df.index.get_indexer([dt + pd.Timedelta(hours=24)], method='nearest')[0]
            future_48h_idx = ng_df.index.get_indexer([dt + pd.Timedelta(hours=48)], method='nearest')[0]

            if future_24h_idx >= len(ng_df) or future_48h_idx >= len(ng_df):
                continue

            ng_24h = ng_df.iloc[future_24h_idx]['Close']
            ng_48h = ng_df.iloc[future_48h_idx]['Close']

            # Feature record
            record = {
                'datetime': dt,
                'ng_price': ng_price,
                'ng_change_24h': (ng_24h - ng_price) / ng_price * 100,  # % change
                'ng_change_48h': (ng_48h - ng_price) / ng_price * 100,
            }

            # HDD features (normalized to daily)
            for col in hdd_df.columns:
                if col != 'datetime' and not pd.isna(row[col]):
                    # Normalize by period days
                    if '1_5d' in col:
                        record[col] = row[col] / 5
                    elif '6_10d' in col:
                        record[col] = row[col] / 5
                    elif '8_14d' in col:
                        record[col] = row[col] / 7
                    else:
                        record[col] = row[col]

            features.append(record)

        except Exception as e:
            continue

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)

    # Add derived features
    # Model agreement (std across models for same period)
    for period in ['1_5d', '6_10d', '8_14d']:
        cols = [c for c in df.columns if period in c and any(m in c for m in ['gfs', 'ifs', 'aifs'])]
        if len(cols) >= 2:
            df[f'model_std_{period}'] = df[cols].std(axis=1)
            df[f'model_mean_{period}'] = df[cols].mean(axis=1)

    # HDD momentum (change from previous forecast)
    df = df.sort_values('datetime')
    for col in df.columns:
        if any(period in col for period in ['1_5d', '6_10d', '8_14d']) and 'std' not in col and 'mean' not in col:
            df[f'{col}_change'] = df[col].diff()

    # Seasonal features
    df['month'] = df['datetime'].dt.month
    df['is_winter'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)

    return df.dropna()


# ============================================
# Train Models
# ============================================
def train_and_evaluate(df, target_col='ng_change_24h'):
    """Train and evaluate ML models."""
    print(f"\nTraining models for {target_col}...")

    # Feature columns
    feature_cols = [c for c in df.columns if c not in [
        'datetime', 'ng_price', 'ng_change_24h', 'ng_change_48h'
    ]]

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")

    if len(X) < 10:
        print("  Not enough data for training!")
        return None, None, None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 3))

    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    }

    results = {}
    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
            mean_score = scores.mean()
            results[name] = {
                'r2_mean': mean_score,
                'r2_std': scores.std()
            }
            print(f"  {name}: R2 = {mean_score:.3f} (+/- {scores.std():.3f})")

            if mean_score > best_score:
                best_score = mean_score
                best_model = model

        except Exception as e:
            print(f"  {name}: Error - {e}")

    # Train best model on all data
    if best_model is not None:
        best_model.fit(X_scaled, y)

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\n  Top 10 Features:")
            for _, row in importances.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.3f}")

        elif hasattr(best_model, 'coef_'):
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': np.abs(best_model.coef_)
            }).sort_values('importance', ascending=False)

            print("\n  Top 10 Features (by coefficient magnitude):")
            for _, row in importances.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.3f}")

    return best_model, scaler, feature_cols


# ============================================
# Make Predictions
# ============================================
def make_prediction(model, scaler, feature_cols, latest_hdd_row, current_ng_price):
    """Make forward-looking prediction."""
    if model is None:
        return None

    # Prepare features
    features = {}
    for col in feature_cols:
        if col in latest_hdd_row:
            features[col] = latest_hdd_row[col]
        else:
            features[col] = 0

    X = pd.DataFrame([features])[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    return {
        'predicted_change_pct': prediction,
        'predicted_price': current_ng_price * (1 + prediction / 100)
    }


# ============================================
# Main
# ============================================
def main():
    # Load data from both sources
    csv_df = load_csv_data()
    grib_df = load_grib2_data()

    # Combine data sources
    if not csv_df.empty and not grib_df.empty:
        hdd_df = pd.concat([csv_df, grib_df], ignore_index=True)
        hdd_df = hdd_df.groupby('datetime').first().reset_index()
        hdd_df = hdd_df.sort_values('datetime')
    elif not csv_df.empty:
        hdd_df = csv_df
    elif not grib_df.empty:
        hdd_df = grib_df
    else:
        print("\nNo HDD data found. Run fetch_all_models.py first.")
        return

    print(f"\nCombined: {len(hdd_df)} HDD forecasts")
    print(f"Date range: {hdd_df['datetime'].min()} to {hdd_df['datetime'].max()}")

    # Load NG data
    start_date = hdd_df['datetime'].min() - pd.Timedelta(days=7)
    end_date = hdd_df['datetime'].max() + pd.Timedelta(days=7)
    ng_df = load_ng_data(start_date, end_date)

    if ng_df.empty:
        print("\nNo NG data found.")
        return

    # Create features
    feature_df = create_features(hdd_df, ng_df)

    if feature_df.empty:
        print("\nCouldn't create features - no overlapping data.")
        return

    print(f"\nCreated {len(feature_df)} feature records")

    # Train models
    print("\n" + "=" * 70)
    print("24-HOUR PREDICTION MODEL")
    print("=" * 70)
    model_24h, scaler_24h, features_24h = train_and_evaluate(feature_df, 'ng_change_24h')

    print("\n" + "=" * 70)
    print("48-HOUR PREDICTION MODEL")
    print("=" * 70)
    model_48h, scaler_48h, features_48h = train_and_evaluate(feature_df, 'ng_change_48h')

    # Make prediction with latest data
    print("\n" + "=" * 70)
    print("FORWARD PREDICTIONS")
    print("=" * 70)

    latest_hdd = hdd_df.iloc[-1].to_dict()
    latest_ng = ng_df['Close'].iloc[-1]

    print(f"\nLatest NG Price: ${latest_ng:.3f}/MMBtu")
    print(f"Latest Forecast Time: {latest_hdd['datetime']}")

    if model_24h is not None:
        pred_24h = make_prediction(model_24h, scaler_24h, features_24h, latest_hdd, latest_ng)
        if pred_24h:
            direction = "UP" if pred_24h['predicted_change_pct'] > 0 else "DOWN"
            print(f"\n24h Prediction: {direction} {abs(pred_24h['predicted_change_pct']):.2f}%")
            print(f"   Target: ${pred_24h['predicted_price']:.3f}/MMBtu")

    if model_48h is not None:
        pred_48h = make_prediction(model_48h, scaler_48h, features_48h, latest_hdd, latest_ng)
        if pred_48h:
            direction = "UP" if pred_48h['predicted_change_pct'] > 0 else "DOWN"
            print(f"\n48h Prediction: {direction} {abs(pred_48h['predicted_change_pct']):.2f}%")
            print(f"   Target: ${pred_48h['predicted_price']:.3f}/MMBtu")

    # Trading signals
    print("\n" + "=" * 70)
    print("TRADING SIGNALS")
    print("=" * 70)

    # Check model agreement
    hdd_cols = [c for c in latest_hdd.keys() if '6_10d' in str(c)]
    if hdd_cols:
        values = [latest_hdd[c] for c in hdd_cols if pd.notna(latest_hdd.get(c))]
        if values:
            mean_6_10 = np.mean(values)
            std_6_10 = np.std(values)
            print(f"\n6-10d HDD (KEY): {mean_6_10:.1f} +/- {std_6_10:.1f}")

            if std_6_10 < 1:
                print("  Models AGREE - higher confidence signal")
            else:
                print("  Models DIVERGE - lower confidence")

            # Simple HDD-based signal
            if mean_6_10 > 12:
                print("  Above-normal heating demand -> BULLISH for NG")
            elif mean_6_10 < 8:
                print("  Below-normal heating demand -> BEARISH for NG")
            else:
                print("  Normal heating demand -> NEUTRAL")

    print("\n" + "=" * 70)
    print("DISCLAIMER: This is for educational purposes only.")
    print("Not financial advice. Past performance doesn't guarantee future results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
