"""
app/ml/features.py
──────────────────
Feature engineering from a device's rolling raw readings.

Input  : list of RawReading ORM objects (last N hours from DB)
Output : FeatureVector Pydantic object, ready for anomaly model inference.

All features are computed from time-indexed pandas DataFrames so that
window-based aggregations (1-min, 5-min, 15-min) are accurate regardless
of the polling interval.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from app.config import settings
from app.models.orm import RawReading
from app.schemas.readings import FeatureVector

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    device_id: str,
    email: str,
    readings: Sequence[RawReading],
    reference_time: Optional[datetime] = None,
) -> FeatureVector:
    """
    Compute the full feature vector for a device given its recent readings.

    Parameters
    ----------
    device_id      : Unique device identifier.
    email          : Owner email.
    readings       : Ordered list of RawReading ORM objects (oldest → newest).
    reference_time : The "now" used for time-context features. Defaults to utcnow.

    Returns
    -------
    FeatureVector with all available features populated.
    Missing values are None (e.g. not enough history for rolling windows).
    """
    now = reference_time or datetime.now(timezone.utc)

    if not readings:
        logger.debug("No readings for device %s – returning empty feature vector.", device_id)
        return _empty_vector(device_id, email, now)

    df = _to_dataframe(readings)

    if df.empty:
        return _empty_vector(device_id, email, now)

    return FeatureVector(
        device_id=device_id,
        email=email,
        # Rolling power aggregates
        rolling_avg_power_1m=_rolling_mean(df, "power", "1min"),
        rolling_avg_power_5m=_rolling_mean(df, "power", "5min"),
        rolling_std_power_5m=_rolling_std(df, "power", "5min"),
        # Rolling current aggregates
        rolling_avg_current_5m=_rolling_mean(df, "current", "5min"),
        rolling_max_current_5m=_rolling_max(df, "current", "5min"),
        # Rate-of-change: difference from previous reading
        delta_power=_latest_delta(df, "power"),
        delta_current=_latest_delta(df, "current"),
        # Voltage behaviour
        voltage_deviation=_voltage_deviation(df),
        voltage_out_of_range=_voltage_out_of_range(df),
        # Spike frequency
        spike_count_15m=_spike_count(df, window_minutes=15),
        # Time context
        hour_of_day=now.hour,
        day_of_week=now.weekday(),
        night_flag=_is_night(now.hour),
    )


def build_training_matrix(
    all_readings: Sequence[RawReading],
) -> pd.DataFrame:
    """
    Build a wide feature matrix from ALL readings (across all devices)
    suitable for training the Isolation Forest model.

    Returns a DataFrame where each row is one feature vector computed
    from a sliding window around that reading's timestamp.
    This is a simplified approach: for each reading we build the feature
    vector using the 5-minute history up to that reading's timestamp.
    """
    if not all_readings:
        return pd.DataFrame()

    df_all = _to_dataframe(all_readings)
    df_all["device_id_col"] = [r.device_id for r in all_readings]

    rows = []
    for device_id, group in df_all.groupby("device_id_col"):
        group = group.sort_index()
        for ts in group.index:
            window = group.loc[:ts].tail(60)  # up to ~5 min at 5s intervals
            if len(window) < 3:
                continue
            fv = _compute_from_df(str(device_id), "", window, ts)
            rows.append(_feature_vector_to_dict(fv))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Columns we care about from RawReading
_NUMERIC_COLS = ["voltage", "current", "power", "kwh"]


def _to_dataframe(readings: Sequence[RawReading]) -> pd.DataFrame:
    """Convert RawReading ORM objects to a timezone-aware time-indexed DataFrame."""
    records = []
    for r in readings:
        ts = r.reading_ts
        # Ensure timezone-aware
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        records.append({
            "ts": ts,
            "device_id": r.device_id,
            "voltage": r.voltage,
            "current": r.current,
            "power": r.power,
            "kwh": r.kwh,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("ts").set_index("ts")

    # Coerce to numeric, fill zeros for missing where appropriate
    for col in _NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _compute_from_df(
    device_id: str,
    email: str,
    df: pd.DataFrame,
    reference_time: pd.Timestamp,
) -> FeatureVector:
    """Feature computation on a pre-sliced DataFrame."""
    dt = reference_time.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return FeatureVector(
        device_id=device_id,
        email=email,
        rolling_avg_power_1m=_rolling_mean(df, "power", "1min"),
        rolling_avg_power_5m=_rolling_mean(df, "power", "5min"),
        rolling_std_power_5m=_rolling_std(df, "power", "5min"),
        rolling_avg_current_5m=_rolling_mean(df, "current", "5min"),
        rolling_max_current_5m=_rolling_max(df, "current", "5min"),
        delta_power=_latest_delta(df, "power"),
        delta_current=_latest_delta(df, "current"),
        voltage_deviation=_voltage_deviation(df),
        voltage_out_of_range=_voltage_out_of_range(df),
        spike_count_15m=_spike_count(df, window_minutes=15),
        hour_of_day=dt.hour,
        day_of_week=dt.weekday(),
        night_flag=_is_night(dt.hour),
    )


def _rolling_mean(df: pd.DataFrame, col: str, window: str) -> Optional[float]:
    """Mean of `col` over the trailing `window` duration. None if column absent."""
    if col not in df.columns or df[col].isna().all():
        return None
    series = df[col].dropna()
    if series.empty:
        return None
    rolled = series.rolling(window, min_periods=1).mean()
    val = rolled.iloc[-1]
    return float(val) if not np.isnan(val) else None


def _rolling_std(df: pd.DataFrame, col: str, window: str) -> Optional[float]:
    if col not in df.columns or df[col].isna().all():
        return None
    series = df[col].dropna()
    if len(series) < 2:
        return 0.0
    rolled = series.rolling(window, min_periods=2).std()
    val = rolled.iloc[-1]
    return float(val) if not np.isnan(val) else None


def _rolling_max(df: pd.DataFrame, col: str, window: str) -> Optional[float]:
    if col not in df.columns or df[col].isna().all():
        return None
    series = df[col].dropna()
    if series.empty:
        return None
    rolled = series.rolling(window, min_periods=1).max()
    val = rolled.iloc[-1]
    return float(val) if not np.isnan(val) else None


def _latest_delta(df: pd.DataFrame, col: str) -> Optional[float]:
    """Difference between the latest and second-latest reading of `col`."""
    if col not in df.columns:
        return None
    series = df[col].dropna()
    if len(series) < 2:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-2])


def _voltage_deviation(df: pd.DataFrame) -> Optional[float]:
    """Deviation of the latest voltage reading from its 5-minute rolling mean."""
    if "voltage" not in df.columns or df["voltage"].isna().all():
        return None
    series = df["voltage"].dropna()
    if series.empty:
        return None
    rolling_mean = series.rolling("5min", min_periods=1).mean().iloc[-1]
    if np.isnan(rolling_mean) or rolling_mean == 0:
        return None
    return float(series.iloc[-1] - rolling_mean)


def _voltage_out_of_range(df: pd.DataFrame) -> Optional[bool]:
    if "voltage" not in df.columns or df["voltage"].isna().all():
        return None
    latest = df["voltage"].dropna().iloc[-1]
    return not (settings.VOLTAGE_MIN <= latest <= settings.VOLTAGE_MAX)


def _spike_count(df: pd.DataFrame, window_minutes: int = 15) -> Optional[int]:
    """
    Count how many times power exceeded POWER_SPIKE_MULTIPLIER × rolling-5min-mean
    in the last `window_minutes` minutes.
    """
    if "power" not in df.columns or df["power"].isna().all():
        return None

    series = df["power"].dropna()
    if series.empty:
        return None

    now = series.index[-1]
    cutoff = now - pd.Timedelta(minutes=window_minutes)
    recent = series[series.index >= cutoff]

    if recent.empty:
        return 0

    rolling_avg = series.rolling("5min", min_periods=1).mean()
    threshold = rolling_avg * settings.POWER_SPIKE_MULTIPLIER

    spikes = (recent > threshold.reindex(recent.index, method="nearest")).sum()
    return int(spikes)


def _is_night(hour: int) -> bool:
    """Return True if hour (0-23) falls in the night period 22:00–06:00."""
    return hour >= 22 or hour < 6


def _empty_vector(device_id: str, email: str, now: datetime) -> FeatureVector:
    return FeatureVector(
        device_id=device_id,
        email=email,
        hour_of_day=now.hour,
        day_of_week=now.weekday(),
        night_flag=_is_night(now.hour),
    )


# ── ML training matrix helper ─────────────────────────────────────────────────

# Feature columns in the exact order used for ML training/inference
FEATURE_COLUMNS = [
    "rolling_avg_power_1m",
    "rolling_avg_power_5m",
    "rolling_std_power_5m",
    "rolling_avg_current_5m",
    "rolling_max_current_5m",
    "delta_power",
    "delta_current",
    "voltage_deviation",
    "spike_count_15m",
    "hour_of_day",
    "day_of_week",
    "night_flag",
]


def _feature_vector_to_dict(fv: FeatureVector) -> dict:
    return {
        "rolling_avg_power_1m": fv.rolling_avg_power_1m,
        "rolling_avg_power_5m": fv.rolling_avg_power_5m,
        "rolling_std_power_5m": fv.rolling_std_power_5m,
        "rolling_avg_current_5m": fv.rolling_avg_current_5m,
        "rolling_max_current_5m": fv.rolling_max_current_5m,
        "delta_power": fv.delta_power,
        "delta_current": fv.delta_current,
        "voltage_deviation": fv.voltage_deviation,
        "spike_count_15m": fv.spike_count_15m,
        "hour_of_day": fv.hour_of_day,
        "day_of_week": fv.day_of_week,
        "night_flag": int(fv.night_flag) if fv.night_flag is not None else None,
    }


def feature_vector_to_numpy(fv: FeatureVector) -> Optional[np.ndarray]:
    """
    Convert a FeatureVector to a 1×N numpy array for model inference.
    Returns None if critical features are all missing.
    """
    row = _feature_vector_to_dict(fv)
    values = [row.get(col) for col in FEATURE_COLUMNS]

    # If ALL numeric features are None we can't score
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None

    # Replace remaining None with 0 (neutral imputation for inference)
    arr = np.array([0.0 if v is None else float(v) for v in values], dtype=np.float64)
    return arr.reshape(1, -1)
