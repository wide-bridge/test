"""
Time-series DataFrame transformation utilities using pandas.
Includes resampling, rolling statistics, lag features, and EWM smoothing.
"""

import pandas as pd
import numpy as np


def create_sample_timeseries() -> pd.DataFrame:
    """Generate sample time-series data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=365, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "value": np.cumsum(np.random.randn(365)) + 100,
        "volume": np.random.randint(50, 500, size=365),
        "category": np.random.choice(["A", "B"], size=365),
    })
    return df.set_index("timestamp")


def add_lag_features(df: pd.DataFrame, column: str, lags: list[int]) -> pd.DataFrame:
    """Add lagged versions of a column (e.g., t-1, t-7, t-30)."""
    result = df.copy()
    for lag in lags:
        result[f"{column}_lag_{lag}"] = result[column].shift(lag)
    return result


def add_rolling_features(
    df: pd.DataFrame, column: str, windows: list[int]
) -> pd.DataFrame:
    """Add rolling mean, std, min, max for given window sizes."""
    result = df.copy()
    for w in windows:
        rolling = result[column].rolling(window=w)
        result[f"{column}_rmean_{w}"] = rolling.mean()
        result[f"{column}_rstd_{w}"] = rolling.std()
        result[f"{column}_rmin_{w}"] = rolling.min()
        result[f"{column}_rmax_{w}"] = rolling.max()
    return result


def add_ewm_features(
    df: pd.DataFrame, column: str, spans: list[int]
) -> pd.DataFrame:
    """Add exponentially weighted moving averages for given spans."""
    result = df.copy()
    for span in spans:
        result[f"{column}_ewm_{span}"] = result[column].ewm(span=span).mean()
    return result


def add_diff_and_pct_change(
    df: pd.DataFrame, column: str, periods: list[int]
) -> pd.DataFrame:
    """Add differenced and percent-change features."""
    result = df.copy()
    for p in periods:
        result[f"{column}_diff_{p}"] = result[column].diff(p)
        result[f"{column}_pct_{p}"] = result[column].pct_change(p)
    return result


def resample_ohlcv(df: pd.DataFrame, value_col: str, volume_col: str, rule: str) -> pd.DataFrame:
    """Resample to OHLCV-style aggregation (Open, High, Low, Close, Volume)."""
    resampled = df.resample(rule).agg(
        open=(value_col, "first"),
        high=(value_col, "max"),
        low=(value_col, "min"),
        close=(value_col, "last"),
        volume=(volume_col, "sum"),
    )
    return resampled


def grouped_resample_ffill(
    df: pd.DataFrame, group_col: str, rule: str
) -> pd.DataFrame:
    """Resample within groups and forward-fill missing values."""
    return df.groupby(group_col).resample(rule).ffill()


def build_timeseries_features(
    df: pd.DataFrame,
    value_col: str = "value",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Full pipeline: lag + rolling + EWM + diff features on a time-series DataFrame."""
    result = df.copy()

    # Lag features
    result = add_lag_features(result, value_col, lags=[1, 6, 12, 24])

    # Rolling statistics
    result = add_rolling_features(result, value_col, windows=[6, 12, 24])

    # Exponentially weighted moving averages
    result = add_ewm_features(result, value_col, spans=[6, 12, 24])

    # Differences and percent changes
    result = add_diff_and_pct_change(result, value_col, periods=[1, 6, 24])

    return result


if __name__ == "__main__":
    # Create sample data
    df = create_sample_timeseries()
    print("Raw data shape:", df.shape)
    print(df.head())
    print()

    # Build all features
    featured = build_timeseries_features(df)
    print("Featured data shape:", featured.shape)
    print("Columns:", list(featured.columns))
    print(featured.dropna().head())
    print()

    # Resample to daily OHLCV
    daily = resample_ohlcv(df, "value", "volume", rule="D")
    print("Daily OHLCV:")
    print(daily.head())
    print()

    # Grouped resample with forward fill
    filled = grouped_resample_ffill(df[["value", "category"]], "category", "3h")
    print("Grouped resample (3h, ffill):")
    print(filled.head(10))
