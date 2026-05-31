"""Seasonal-detrended Z-score.

The naive z-score `(storage - 252day_mean) / std` is dominated by the
annual storage cycle (March trough, November peak). The TRUE alpha
signal is storage SURPRISE vs seasonal expectation.

This module computes proper detrended z-scores for use by both the
backtest engine and the live dashboard.
"""
import pandas as pd
import numpy as np


def fit_seasonal_storage(df: pd.DataFrame) -> pd.Series:
    """For each row, compute expected storage based on day-of-year.

    Uses median storage per DOY across all available history. Returns
    a series indexed by df.index with the seasonal expectation.
    """
    if 'eia_storage_weekly' not in df.columns:
        return pd.Series(dtype=float, index=df.index)

    # Group by day-of-year and take median
    df2 = df.copy()
    df2['doy'] = pd.DatetimeIndex(df2.index).dayofyear
    storage_by_doy = df2.groupby('doy')['eia_storage_weekly'].median()

    # Smooth (rolling 14-day) to remove noise
    storage_by_doy = storage_by_doy.rolling(14, min_periods=1, center=True).mean()

    # Map each date back to its expected storage
    expected = df2['doy'].map(storage_by_doy)
    return expected


def compute_seasonal_z(series: pd.Series, df: pd.DataFrame,
                       lookback_years: float = 3.0) -> pd.Series:
    """Detrended z: how far is the series from its DOY seasonal expectation?

    Returns z = (actual - seasonal_expected) / seasonal_std
    where seasonal_std is the typical deviation around the seasonal cycle.
    """
    expected = fit_seasonal_storage(df)
    surprise = series - expected
    # Rolling std of the surprise (not the raw series)
    window = int(lookback_years * 252)
    surprise_std = surprise.rolling(window, min_periods=60).std()
    z = surprise / (surprise_std + 1e-9)
    return z


def compute_seasonal_days_supply_z(df: pd.DataFrame) -> pd.Series:
    """Same idea for days_supply."""
    if 'days_supply' not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    ds = df['days_supply']
    df2 = df.copy()
    df2['doy'] = pd.DatetimeIndex(df2.index).dayofyear
    ds_by_doy = df2.groupby('doy')['days_supply'].median()
    ds_by_doy = ds_by_doy.rolling(14, min_periods=1, center=True).mean()
    expected = df2['doy'].map(ds_by_doy)
    surprise = ds - expected
    window = 252 * 3
    std = surprise.rolling(window, min_periods=60).std()
    return surprise / (std + 1e-9)


def add_seasonal_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns: storage_surprise_z, days_supply_surprise_z, seasonal_storage_expected.
    Returns NEW dataframe with added columns."""
    out = df.copy()
    out['seasonal_storage_expected'] = fit_seasonal_storage(out)
    out['storage_surprise_z'] = compute_seasonal_z(out['eia_storage_weekly'], out)
    out['days_supply_surprise_z'] = compute_seasonal_days_supply_z(out)
    return out


if __name__ == '__main__':
    # Validate
    import os
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'cache/master_dataset.csv'),
        parse_dates=['Date'], index_col=0,
    )
    df = df.dropna(subset=['UNG'])
    df = add_seasonal_factors(df)
    print("=== Seasonal Detrending Validation ===")
    print(f"Rows: {len(df)}")
    print()
    print("Naive storage z (last 10 days):")
    naive_z = (df['eia_storage_weekly'] - df['eia_storage_weekly'].rolling(252).mean()) \
              / df['eia_storage_weekly'].rolling(252).std()
    print(naive_z.tail(10).to_string())
    print()
    print("Seasonal-detrended storage surprise z (last 10 days):")
    print(df['storage_surprise_z'].tail(10).to_string())
    print()
    print(f"Naive z mean: {naive_z.mean():.2f}, std: {naive_z.std():.2f}")
    print(f"Surprise z mean: {df['storage_surprise_z'].mean():.2f}, std: {df['storage_surprise_z'].std():.2f}")
    print()
    # Correlation with seasonal cycle
    import math
    df['seasonal_pred'] = 2500 + 600 * np.sin(2*math.pi * (pd.DatetimeIndex(df.index).dayofyear - 60) / 365)
    print(f"Naive z corr with seasonal sine: {naive_z.corr(df['seasonal_pred']):.3f}")
    print(f"Surprise z corr with seasonal sine: {df['storage_surprise_z'].corr(df['seasonal_pred']):.3f}")
