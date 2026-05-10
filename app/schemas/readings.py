"""
app/schemas/readings.py
───────────────────────
Pydantic v2 schemas used for:
  - Validating/parsing raw Firebase data
  - FastAPI request/response bodies
  - Internal service layer data transfer
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Firebase raw data
# ─────────────────────────────────────────────────────────────────────────────

class FirebaseReading(BaseModel):
    """
    Represents a single user node read from Firebase Realtime Database.
    All PZEM fields are optional – Firebase may omit them when device is offline.
    """
    device_id: str = Field(..., min_length=1, max_length=64)
    email: str = Field(..., min_length=3, max_length=255)

    voltage: Optional[float] = None
    current: Optional[float] = None
    power: Optional[float] = None
    kwh: Optional[float] = None
    power_factor: Optional[float] = None
    frequency: Optional[float] = None

    load1: Optional[bool] = None
    load2: Optional[bool] = None
    load3: Optional[bool] = None
    overall: Optional[bool] = None

    # If Firebase supplies a Unix timestamp; otherwise we generate one
    timestamp: Optional[int] = None

    # Non-electrical metadata (informational only)
    name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None

    @field_validator("voltage", "current", "power", "kwh", mode="before")
    @classmethod
    def coerce_numeric(cls, v: Any) -> Optional[float]:
        """Accept int / str numbers from Firebase and coerce to float."""
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("load1", "load2", "load3", "overall", mode="before")
    @classmethod
    def coerce_bool(cls, v: Any) -> Optional[bool]:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    @model_validator(mode="before")
    @classmethod
    def extract_device_id(cls, data: Any) -> Any:
        """
        Firebase user nodes typically don't have device_id at the top level
        in all cases – fall back to email-derived ID if missing.
        """
        if isinstance(data, dict):
            if not data.get("device_id") and data.get("email"):
                data["device_id"] = data["email"].replace("@", "_").replace(".", "_")
        return data


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector
# ─────────────────────────────────────────────────────────────────────────────

class FeatureVector(BaseModel):
    """Computed rolling-window features for a device, ready for ML inference."""
    device_id: str
    email: str

    rolling_avg_power_1m: Optional[float] = None
    rolling_avg_power_5m: Optional[float] = None
    rolling_std_power_5m: Optional[float] = None
    rolling_avg_current_5m: Optional[float] = None
    rolling_max_current_5m: Optional[float] = None

    delta_power: Optional[float] = None
    delta_current: Optional[float] = None

    voltage_deviation: Optional[float] = None
    voltage_out_of_range: Optional[bool] = None

    spike_count_15m: Optional[int] = None

    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    night_flag: Optional[bool] = None


# ─────────────────────────────────────────────────────────────────────────────
# AI result
# ─────────────────────────────────────────────────────────────────────────────

class AIResultOut(BaseModel):
    """API response schema for the AI result of a device."""
    device_id: str
    email: str
    anomaly_score: Optional[float] = None
    theft_score: Optional[float] = None
    status: str = "normal"
    reason: Optional[str] = None
    explanation: Optional[str] = None
    rule_triggered: bool = False
    rule_reasons: Optional[list[str]] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Live reading response
# ─────────────────────────────────────────────────────────────────────────────

class LiveReadingOut(BaseModel):
    """Most recent raw reading for a device."""
    id: int
    device_id: str
    email: str
    voltage: Optional[float] = None
    current: Optional[float] = None
    power: Optional[float] = None
    kwh: Optional[float] = None
    reading_ts: datetime
    inserted_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Suspicious event response
# ─────────────────────────────────────────────────────────────────────────────

class SuspiciousEventOut(BaseModel):
    id: int
    device_id: str
    email: str
    status: str
    anomaly_score: Optional[float] = None
    theft_score: Optional[float] = None
    reason: Optional[str] = None
    voltage: Optional[float] = None
    current: Optional[float] = None
    power: Optional[float] = None
    kwh: Optional[float] = None
    occurred_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Hourly summary response
# ─────────────────────────────────────────────────────────────────────────────

class HourlySummaryOut(BaseModel):
    id: int
    device_id: str
    hour_bucket: datetime
    avg_voltage: Optional[float] = None
    avg_current: Optional[float] = None
    avg_power: Optional[float] = None
    max_power: Optional[float] = None
    kwh_delta: Optional[float] = None
    reading_count: int
    anomaly_count: int
    spike_count: int

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Health / model status
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    firebase_connected: bool
    model_loaded: bool
    version: str


class ModelStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_loaded: bool
    model_path: Optional[str] = None
    training_samples: Optional[int] = None
    trained_at: Optional[str] = None
    contamination: float


# ─────────────────────────────────────────────────────────────────────────────
# Forecast & tariff response schemas
# ─────────────────────────────────────────────────────────────────────────────

class HourlyForecastItem(BaseModel):
    """One hour in the 24-hour forecast (as returned by the API)."""
    hour_offset: int                   # 1 = next hour, … 24 = +24 h
    hour_bucket: str                   # ISO-8601 UTC datetime string
    predicted_power_w: float           # watts
    predicted_kwh: float               # kWh for this 1-hour slot
    tariff_label: str                  # "peak" | "off_peak"
    tariff_rate: float                 # Rs / kWh
    cost_rs: float                     # cost for this slot
    confidence: str                    # "high" | "medium" | "low"


class DeviceForecastOut(BaseModel):
    """API response for GET /devices/{device_id}/forecast."""
    device_id: str
    email: str
    generated_at: Optional[datetime] = None
    model_type: str = "baseline_fallback"
    training_samples: int = 0

    # Per-hour breakdown
    hourly_forecasts: list[HourlyForecastItem] = []

    # 24-hour totals
    total_predicted_kwh: Optional[float] = None
    total_cost_rs: Optional[float] = None
    peak_cost_rs: Optional[float] = None
    offpeak_cost_rs: Optional[float] = None

    # Tariff optimisation
    cheapest_hours: list[int] = []
    most_expensive_hours: list[int] = []
    potential_savings_rs: Optional[float] = None
    shiftable_kwh: Optional[float] = None
    optimization_tips: list[str] = []
    alert_level: str = "info"
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
