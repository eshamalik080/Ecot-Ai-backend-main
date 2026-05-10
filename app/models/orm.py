"""
app/models/orm.py
─────────────────
All SQLAlchemy ORM models.

Tables
------
raw_readings        – Rolling 3-hour window of ingested PZEM readings
feature_snapshots   – Latest computed feature vector per device
ai_results          – Latest anomaly / theft scoring result per device
hourly_summaries    – Aggregated stats kept indefinitely (feeds baselines)
baselines           – Per-device per-hour-of-day learned normal behaviour
suspicious_events   – Audit log of anomaly / theft events
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

# JSON / Text columns that store serialised Python objects use Text.
# We use Text everywhere (no dialect-specific JSON type) for portability.

from app.db import Base


# ─────────────────────────────────────────────────────────────────────────────
# 1. raw_readings
# ─────────────────────────────────────────────────────────────────────────────

class RawReading(Base):
    """
    One row per PZEM poll per device. Rows older than DATA_RETENTION_HOURS
    are automatically deleted by the cleanup job.
    """
    __tablename__ = "raw_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # PZEM electrical measurements
    voltage: Mapped[float | None] = mapped_column(Float, nullable=True)
    current: Mapped[float | None] = mapped_column(Float, nullable=True)
    power: Mapped[float | None] = mapped_column(Float, nullable=True)
    kwh: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Optional fields (may be present in future firmware versions)
    power_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    frequency: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Load relay states from Firebase
    load1: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    load2: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    load3: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    overall: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Timestamp of the actual reading (from Firebase if available, else server time)
    reading_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    # When this row was inserted into our DB
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # Composite index for fast per-device time-range queries
        Index("ix_raw_readings_device_ts", "device_id", "reading_ts"),
        # Index for cleanup job (purges by inserted_at)
        Index("ix_raw_readings_inserted_at", "inserted_at"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. feature_snapshots
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSnapshot(Base):
    """
    Latest rolling-window feature vector computed for a device.
    One row per device; upserted on every poll cycle.
    """
    __tablename__ = "feature_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)

    # Rolling averages
    rolling_avg_power_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_avg_power_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_std_power_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_avg_current_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_max_current_5m: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Delta (rate-of-change) features
    delta_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta_current: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Voltage behaviour
    voltage_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)   # vs rolling mean
    voltage_out_of_range: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Spike count in last 15 minutes
    spike_count_15m: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Time-context features
    hour_of_day: Mapped[int | None] = mapped_column(Integer, nullable=True)
    day_of_week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    night_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # 22:00–06:00

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. ai_results
# ─────────────────────────────────────────────────────────────────────────────

class AIResult(Base):
    """
    Latest anomaly detection and theft scoring result for each device.
    One row per device; upserted after every inference run.
    """
    __tablename__ = "ai_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)

    # ML model output (0.0 – 1.0; higher = more anomalous)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Whether rule-based checks triggered
    rule_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    rule_reasons: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list

    # Theft scoring
    theft_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0.0 – 1.0

    # Human-readable classification
    # Values: "normal" | "anomaly" | "suspicious" | "possible_theft"
    status: Mapped[str] = mapped_column(String(32), default="normal")

    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. hourly_summaries
# ─────────────────────────────────────────────────────────────────────────────

class HourlySummary(Base):
    """
    Aggregated statistics for each (device, hour bucket).
    Kept indefinitely to feed baseline learning without storing raw rows.
    """
    __tablename__ = "hourly_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)

    # Truncated to the start of the hour (UTC)
    hour_bucket: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    avg_voltage: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_current: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    kwh_delta: Mapped[float | None] = mapped_column(Float, nullable=True)  # kWh growth in this hour
    reading_count: Mapped[int] = mapped_column(Integer, default=0)
    anomaly_count: Mapped[int] = mapped_column(Integer, default=0)
    spike_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_hourly_summaries_device_hour", "device_id", "hour_bucket", unique=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. baselines
# ─────────────────────────────────────────────────────────────────────────────

class Baseline(Base):
    """
    Learned normal behaviour for each (device, hour_of_day) slot.
    Updated periodically from hourly_summaries.
    Used by theft scoring as the reference for "what's normal".
    """
    __tablename__ = "baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    hour_of_day: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-23

    # Learned averages and standard deviations
    avg_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_current: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_current: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_kwh_growth: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Historical spike & anomaly rate for this hour slot
    avg_spike_freq: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_anomaly_freq: Mapped[float | None] = mapped_column(Float, nullable=True)

    # How many hourly summaries were used to build this baseline
    sample_count: Mapped[int] = mapped_column(Integer, default=0)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_baselines_device_hour", "device_id", "hour_of_day", unique=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. suspicious_events
# ─────────────────────────────────────────────────────────────────────────────

class SuspiciousEvent(Base):
    """
    Audit log. One row is written every time a device crosses the
    'suspicious' or 'possible_theft' threshold.
    """
    __tablename__ = "suspicious_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False)  # suspicious | possible_theft
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    theft_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Raw snapshot at the time of the event
    voltage: Mapped[float | None] = mapped_column(Float, nullable=True)
    current: Mapped[float | None] = mapped_column(Float, nullable=True)
    power: Mapped[float | None] = mapped_column(Float, nullable=True)
    kwh: Mapped[float | None] = mapped_column(Float, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. device_forecasts
# ─────────────────────────────────────────────────────────────────────────────

class DeviceForecast(Base):
    """
    Latest 24-hour energy consumption forecast for each device.
    One row per device; upserted each time the forecasting job runs (hourly).

    Stores the full per-hour breakdown as JSON so the API can serve it
    without a secondary query.
    """
    __tablename__ = "device_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)

    # When this forecast was generated (UTC)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # ML model used: "gradient_boosting" | "baseline_fallback"
    model_type: Mapped[str] = mapped_column(String(32), default="baseline_fallback")
    training_samples: Mapped[int] = mapped_column(Integer, default=0)

    # Serialised list[dict] – 24 hourly forecast points.
    # Each dict: {hour_offset, hour_bucket_iso, predicted_power_w, predicted_kwh,
    #             tariff_label, tariff_rate, cost_rs, confidence}
    hourly_forecasts: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 24-hour aggregate totals
    total_predicted_kwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost_rs: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_cost_rs: Mapped[float | None] = mapped_column(Float, nullable=True)
    offpeak_cost_rs: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Tariff optimisation outputs
    cheapest_hours: Mapped[str | None] = mapped_column(Text, nullable=True)      # JSON list[int]
    most_expensive_hours: Mapped[str | None] = mapped_column(Text, nullable=True) # JSON list[int]
    potential_savings_rs: Mapped[float | None] = mapped_column(Float, nullable=True)
    shiftable_kwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    optimization_tips: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON list[str]
    alert_level: Mapped[str] = mapped_column(String(32), default="info")

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
