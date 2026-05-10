"""
app/ml/forecasting.py
─────────────────────
Per-device 24-hour energy consumption forecasting.

Model
-----
GradientBoostingRegressor (scikit-learn) trained on a device's hourly_summaries.
Falls back to baseline-average (per hour-of-day) when fewer than
MIN_FORECAST_SAMPLES hourly rows are available.

Feature set (9 features per sample)
-------------------------------------
  0. hour_of_day       – 0–23
  1. day_of_week       – 0=Mon … 6=Sun
  2. is_weekend        – 0/1
  3. lag_1h            – avg_power from 1 hour ago
  4. lag_2h            – avg_power from 2 hours ago
  5. lag_3h            – avg_power from 3 hours ago
  6. lag_24h           – avg_power 24 hours ago (same hour yesterday)
  7. rolling_mean_6h   – mean of last 6 hourly power values
  8. rolling_mean_24h  – mean of last 24 hourly power values

Forecast chaining
-----------------
For horizons > 1 hour, each predicted value is appended to the rolling
history so subsequent steps use it as a lag feature (recursive strategy).

Usage
-----
    from app.ml.forecasting import generate_forecast

    result = generate_forecast(
        device_id   = "ECO-00007",
        email       = "user@example.com",
        hourly_records = [(dt1, 450.0), (dt2, 510.0), ...],   # oldest → newest
        baselines   = {0: 120.0, 1: 115.0, ..., 23: 400.0},  # {hour_of_day: avg_power}
        now         = datetime.now(timezone.utc),
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_FORECAST_SAMPLES: int = 48          # minimum hourly rows to enable ML model
FORECAST_HORIZON: int = 24             # predict this many hours into the future
N_FEATURES: int = 9

# In-memory model cache: device_id → (model, scaler, n_training_samples)
_model_cache: dict[str, tuple[GradientBoostingRegressor, StandardScaler, int]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HourlyForecastPoint:
    """One predicted hour in the 24-hour forecast."""
    hour_offset: int           # 1 = next hour, 2 = +2 h, … 24 = +24 h
    hour_bucket: datetime      # absolute UTC datetime of that hour (on the hour)
    predicted_power_w: float   # average power expected during that hour (watts)
    predicted_kwh: float       # energy for that hour (= predicted_power_w / 1000)
    confidence: str            # "high" | "medium" | "low"


@dataclass
class ForecastResult:
    """Full 24-hour forecast for one device."""
    device_id: str
    email: str
    generated_at: datetime
    model_type: str            # "gradient_boosting" | "baseline_fallback"
    training_samples: int      # 0 when using fallback
    points: list[HourlyForecastPoint] = field(default_factory=list)

    @property
    def total_predicted_kwh(self) -> float:
        return round(sum(p.predicted_kwh for p in self.points), 4)

    @property
    def peak_hours_kwh(self) -> float:
        """kWh predicted during peak hours (7:00–23:00 UTC)."""
        return round(
            sum(p.predicted_kwh for p in self.points if 7 <= p.hour_bucket.hour < 23),
            4,
        )

    @property
    def offpeak_hours_kwh(self) -> float:
        return round(self.total_predicted_kwh - self.peak_hours_kwh, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_vector(
    power_history: list[float],    # ordered list of hourly avg_power values (oldest → newest)
    target_dt: datetime,           # the datetime of the hour we're predicting
) -> np.ndarray:
    """
    Build a 9-element feature vector for one prediction point.

    `power_history` should contain all power values available *before*
    the target hour (i.e. the target's value itself is NOT included).
    Values are in Watts.
    """
    n = len(power_history)

    def _lag(k: int) -> float:
        """k-step lag: k=1 → last element, k=24 → 24 elements ago."""
        idx = n - k
        if idx < 0 or idx >= n:
            # Insufficient history – use the global mean as a neutral fill
            return float(np.mean(power_history)) if power_history else 0.0
        return float(power_history[idx])

    recent_6 = power_history[-6:] if n >= 6 else power_history
    recent_24 = power_history[-24:] if n >= 24 else power_history

    return np.array(
        [
            float(target_dt.hour),               # 0 – hour_of_day
            float(target_dt.weekday()),           # 1 – day_of_week
            float(target_dt.weekday() >= 5),      # 2 – is_weekend
            _lag(1),                              # 3 – lag_1h
            _lag(2),                              # 4 – lag_2h
            _lag(3),                              # 5 – lag_3h
            _lag(24),                             # 6 – lag_24h
            float(np.mean(recent_6)) if recent_6 else 0.0,   # 7 – rolling_mean_6h
            float(np.mean(recent_24)) if recent_24 else 0.0, # 8 – rolling_mean_24h
        ],
        dtype=np.float64,
    )


def _build_training_matrix(
    records: list[tuple[datetime, float]],  # (hour_bucket, avg_power) oldest → newest
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over historical records to create (X, y) training pairs.

    For each record at index i >= 24:
        X[i] = features built from records[0..i-1]
        y[i] = records[i].avg_power   (target)

    Requires >= 24 records to produce at least one training sample.
    """
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []

    for i in range(24, len(records)):
        target_dt, target_power = records[i]
        if target_power is None or np.isnan(target_power):
            continue

        power_history = [p for _, p in records[:i] if p is not None]
        if not power_history:
            continue

        fv = _build_feature_vector(power_history, target_dt)
        X_rows.append(fv)
        y_rows.append(float(target_power))

    if not X_rows:
        return np.empty((0, N_FEATURES), dtype=np.float64), np.empty(0, dtype=np.float64)

    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def _train_or_load(
    device_id: str,
    records: list[tuple[datetime, float]],
) -> tuple[Optional[GradientBoostingRegressor], Optional[StandardScaler], int]:
    """
    Return a trained (model, scaler) pair for this device.
    Uses the in-memory cache and only retrains when data has grown
    significantly (>= 10 new samples since last training).

    Returns (None, None, 0) if training data is insufficient.
    """
    n = len(records)
    cached = _model_cache.get(device_id)

    # Decide whether to retrain
    if cached is not None:
        _, _, cached_n = cached
        # Retrain if we have at least 10% more samples than last time
        if n - cached_n < max(10, int(cached_n * 0.10)):
            model, scaler, trained_n = cached
            return model, scaler, trained_n

    # Build training matrix
    X, y = _build_training_matrix(records)
    if len(y) < MIN_FORECAST_SAMPLES:
        logger.debug(
            "[%s] Forecast: only %d training samples – using baseline fallback.",
            device_id, len(y),
        )
        return None, None, 0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.85,
        loss="squared_error",
        random_state=42,
    )
    model.fit(X_scaled, y)

    _model_cache[device_id] = (model, scaler, len(y))
    logger.info(
        "[%s] Forecast model trained: %d samples, features=%d.",
        device_id, len(y), N_FEATURES,
    )
    return model, scaler, len(y)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence classification
# ─────────────────────────────────────────────────────────────────────────────

def _confidence(n_samples: int, hour_offset: int) -> str:
    """
    Estimate prediction confidence based on training data size and horizon.
    The further into the future, the lower the confidence.
    """
    if n_samples == 0:
        return "low"
    if n_samples >= 336 and hour_offset <= 6:   # 14 days of data, short horizon
        return "high"
    if n_samples >= 168 and hour_offset <= 12:  # 7 days of data, medium horizon
        return "high"
    if n_samples >= 72:                          # 3 days of data
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_forecast(
    device_id: str,
    email: str,
    hourly_records: list[tuple[datetime, float | None]],
    baselines: dict[int, float],
    now: datetime,
) -> ForecastResult:
    """
    Generate a 24-hour hourly power consumption forecast for one device.

    Parameters
    ----------
    device_id :
        Unique device identifier.
    email :
        Associated user email.
    hourly_records :
        List of (hour_bucket, avg_power_watts) tuples, ordered oldest → newest.
        avg_power may be None for missing hours (they are skipped during training).
    baselines :
        Mapping of {hour_of_day: avg_power_watts} from the Baseline table.
        Used as the fallback when not enough data exists for ML training.
    now :
        Current UTC datetime (the forecast starts from the next hour after this).

    Returns
    -------
    ForecastResult with 24 HourlyForecastPoints.
    """
    # Base hour: truncate 'now' to the start of the current hour
    base_hour = now.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

    # Filter records to only those with a valid power value
    clean_records: list[tuple[datetime, float]] = [
        (dt, p) for dt, p in hourly_records if p is not None and not np.isnan(p)
    ]

    # Try to train / load a model
    model, scaler, training_samples = _train_or_load(device_id, clean_records)
    use_ml = model is not None and scaler is not None
    model_type = "gradient_boosting" if use_ml else "baseline_fallback"

    # Rolling power history (watts); will be extended with predictions as we go
    rolling_power: list[float] = [p for _, p in clean_records]

    points: list[HourlyForecastPoint] = []

    for offset in range(1, FORECAST_HORIZON + 1):
        target_dt = base_hour + timedelta(hours=offset)
        hour_of_day = target_dt.hour

        if use_ml:
            fv = _build_feature_vector(rolling_power, target_dt)
            fv_scaled = scaler.transform(fv.reshape(1, -1))
            predicted_w = float(model.predict(fv_scaled)[0])
            predicted_w = max(0.0, predicted_w)     # clamp: power ≥ 0
        else:
            # Fallback: use baseline for this hour-of-day, or 0
            predicted_w = float(baselines.get(hour_of_day) or 0.0)
            predicted_w = max(0.0, predicted_w)

        predicted_kwh = round(predicted_w / 1000.0, 4)

        points.append(
            HourlyForecastPoint(
                hour_offset=offset,
                hour_bucket=target_dt,
                predicted_power_w=round(predicted_w, 2),
                predicted_kwh=predicted_kwh,
                confidence=_confidence(training_samples, offset),
            )
        )

        # Extend rolling history with this prediction for subsequent lags
        rolling_power.append(predicted_w)

    return ForecastResult(
        device_id=device_id,
        email=email,
        generated_at=now,
        model_type=model_type,
        training_samples=training_samples,
        points=points,
    )


def clear_model_cache(device_id: Optional[str] = None) -> None:
    """
    Clear the in-memory model cache.
    Pass a device_id to evict only that device; omit to clear all.
    """
    if device_id:
        _model_cache.pop(device_id, None)
    else:
        _model_cache.clear()
