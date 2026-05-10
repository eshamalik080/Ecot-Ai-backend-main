"""
app/config.py
─────────────
Central settings object backed by pydantic-settings.
All secrets and tuneable parameters are read from environment variables
or the .env file. Nothing is hard-coded.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "ECOT AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://ecot:ecot_pass@localhost:5432/ecot_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── Firebase ──────────────────────────────────────────────────────────────
    FIREBASE_CREDENTIALS_PATH: str = "firebase_credentials.json"
    FIREBASE_DATABASE_URL: str = ""
    # Root node to monitor. "/" means all users under root.
    FIREBASE_ROOT_NODE: str = "/"

    # ── Polling / Scheduling ──────────────────────────────────────────────────
    POLLING_INTERVAL_SECONDS: int = 5
    DATA_RETENTION_HOURS: int = 3
    CLEANUP_INTERVAL_SECONDS: int = 300          # 5 minutes
    BASELINE_UPDATE_INTERVAL_SECONDS: int = 3600  # 1 hour

    # ── ML ────────────────────────────────────────────────────────────────────
    MODEL_DIR: str = "models"
    ANOMALY_CONTAMINATION: float = 0.05
    # Decision boundary: raw Isolation Forest score < threshold → anomaly
    # (sklearn's score_samples returns negative values; we negate & normalise)
    ANOMALY_THRESHOLD: float = 0.5
    MIN_TRAINING_SAMPLES: int = 50   # minimum rows before auto-training is allowed

    # ── Rule-based Safety Thresholds ─────────────────────────────────────────
    VOLTAGE_MIN: float = 180.0
    VOLTAGE_MAX: float = 260.0
    CURRENT_MAX: float = 30.0    # Amperes
    POWER_MAX: float = 5000.0    # Watts
    # A power reading is a spike if it exceeds N × rolling 5-min avg
    POWER_SPIKE_MULTIPLIER: float = 3.0

    # ── Forecasting ───────────────────────────────────────────────────────────
    # How often to regenerate 24-hour forecasts (seconds). Default = 1 hour.
    FORECAST_UPDATE_INTERVAL_SECONDS: int = 3600
    # Minimum hourly summaries required before switching from baseline to ML.
    MIN_FORECAST_SAMPLES: int = 48

    # ── Tariff – NEPRA Pakistan Residential ──────────────────────────────────
    # Peak window in LOCAL time (PKT = UTC+5). Code converts internally.
    TARIFF_PEAK_START_HOUR: int = 17     # 5:00 PM PKT
    TARIFF_PEAK_END_HOUR: int = 5        # 5:00 AM PKT  (wraps midnight)
    TARIFF_CURRENCY: str = "Rs"
    TARIFF_TIMEZONE_OFFSET: int = 5      # PKT = UTC + 5

    # Sanctioned load determines which tariff schedule applies:
    #   < 5 kW  → Slab-based tariff (NEPRA residential slabs)
    #   >= 5 kW → Time-of-Use (TOU): peak Rs 46.85 / off-peak Rs 34.53
    SANCTIONED_LOAD_KW: float = 5.0

    # Set True for protected/lifeline consumers (very low income, ≤50 units/month)
    TARIFF_PROTECTED: bool = False

    # TOU rates for sanctioned load >= 5 kW (Rs/kWh)
    TARIFF_PEAK_RATE: float = 46.85
    TARIFF_OFF_PEAK_RATE: float = 34.53

    @field_validator("ANOMALY_CONTAMINATION")
    @classmethod
    def check_contamination(cls, v: float) -> float:
        if not 0.0 < v < 0.5:
            raise ValueError("ANOMALY_CONTAMINATION must be between 0 and 0.5")
        return v


# Singleton – import this everywhere instead of re-instantiating.
settings = Settings()
