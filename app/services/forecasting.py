"""
app/services/forecasting.py
────────────────────────────
Forecast service: generates 24-hour power consumption forecasts with
tariff-based cost optimization for every known device and persists the
results to PostgreSQL and Firebase.

Called by the APScheduler `forecasting_job` (default: every hour).

Pipeline per device
-------------------
1. Query up to 720 hourly_summaries (30 days) ordered oldest → newest.
2. Query per-device baselines for the fallback path.
3. Call generate_forecast()  →  ForecastResult (24 HourlyForecastPoints).
4. Call compute_tariff_optimization()  →  TariffOptimization.
5. Serialise and upsert into device_forecasts table.
6. Write a compact forecast payload to Firebase under /<user_key>/forecast/.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import firebase_client as fb
from app.config import settings
from app.db import get_db
from app.ml.forecasting import generate_forecast
from app.ml.tariff import compute_tariff_optimization, TariffOptimization
from app.models.orm import AIResult, Baseline, DeviceForecast, HourlySummary

logger = logging.getLogger(__name__)

_MAX_HISTORY_HOURS = 720   # 30 days of hourly summaries


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point  (called by APScheduler)
# ─────────────────────────────────────────────────────────────────────────────

def run_forecast_cycle() -> None:
    """
    Regenerate 24-hour forecasts for every device that has AI results.
    Exceptions per device are caught and logged so one bad device cannot
    block the others.
    """
    now = datetime.now(timezone.utc)
    logger.info("Forecast cycle started at %s.", now.isoformat())

    with get_db() as db:
        # Find all known devices from the ai_results table
        device_rows = db.execute(select(AIResult.device_id, AIResult.email)).all()

        if not device_rows:
            logger.info("Forecast cycle: no known devices yet.")
            return

        logger.info("Forecast cycle: running for %d device(s).", len(device_rows))

        for device_id, email in device_rows:
            try:
                _forecast_device(db, device_id, email, now)
            except Exception as exc:
                logger.error(
                    "Forecast error for device %s: %s", device_id, exc, exc_info=True
                )

        db.commit()

    logger.info("Forecast cycle complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Per-device forecast pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _forecast_device(
    db: Session,
    device_id: str,
    email: str,
    now: datetime,
) -> None:
    """End-to-end forecast pipeline for one device."""

    # ── Step 1: Load hourly summaries ─────────────────────────────────────────
    stmt = (
        select(HourlySummary.hour_bucket, HourlySummary.avg_power)
        .where(HourlySummary.device_id == device_id)
        .order_by(HourlySummary.hour_bucket.asc())
        .limit(_MAX_HISTORY_HOURS)
    )
    rows = db.execute(stmt).all()

    # hourly_records: list of (datetime, float | None)
    hourly_records = [(row.hour_bucket, row.avg_power) for row in rows]

    # ── Step 2: Load baselines for fallback ───────────────────────────────────
    bl_stmt = select(Baseline.hour_of_day, Baseline.avg_power).where(
        Baseline.device_id == device_id
    )
    bl_rows = db.execute(bl_stmt).all()
    baselines: dict[int, float] = {
        row.hour_of_day: float(row.avg_power)
        for row in bl_rows
        if row.avg_power is not None
    }

    # ── Step 3: Generate 24-hour forecast ─────────────────────────────────────
    forecast = generate_forecast(
        device_id=device_id,
        email=email,
        hourly_records=hourly_records,
        baselines=baselines,
        now=now,
    )

    # ── Step 4: Compute tariff optimisation ───────────────────────────────────
    # Convert PKT peak hours to UTC for the forecasting engine
    # (hourly_summaries hour_bucket values are stored in UTC)
    tz_offset = settings.TARIFF_TIMEZONE_OFFSET  # 5 for PKT
    peak_start_utc = (settings.TARIFF_PEAK_START_HOUR - tz_offset) % 24  # 17-5=12
    peak_end_utc   = (settings.TARIFF_PEAK_END_HOUR   - tz_offset) % 24  # 5-5=0

    tariff_opt: TariffOptimization = compute_tariff_optimization(
        forecast=forecast,
        peak_start_hour=peak_start_utc,
        peak_end_hour=peak_end_utc,
        peak_rate=settings.TARIFF_PEAK_RATE,
        offpeak_rate=settings.TARIFF_OFF_PEAK_RATE,
        currency=settings.TARIFF_CURRENCY,
    )

    logger.info(
        "[%s] Forecast: model=%s samples=%d total_kwh=%.3f cost=%s%.2f savings=%s%.2f",
        device_id,
        forecast.model_type,
        forecast.training_samples,
        tariff_opt.total_predicted_kwh,
        settings.TARIFF_CURRENCY,
        tariff_opt.total_cost_rs,
        settings.TARIFF_CURRENCY,
        tariff_opt.potential_savings_rs,
    )

    # ── Step 5: Serialise hourly breakdown ────────────────────────────────────
    hourly_json = json.dumps(
        [
            {
                "hour_offset": slot.hour_offset,
                "hour_bucket": slot.hour_bucket.isoformat(),
                "predicted_power_w": slot.predicted_power_w,
                "predicted_kwh": slot.predicted_kwh,
                "tariff_label": slot.label,
                "tariff_rate": slot.rate_per_kwh,
                "cost_rs": slot.cost_rs,
                "confidence": slot.confidence,
            }
            for slot in tariff_opt.slots
        ]
    )

    # ── Step 6: Upsert into device_forecasts ──────────────────────────────────
    _upsert_device_forecast(db, device_id, email, forecast, tariff_opt, hourly_json, now)

    # ── Step 7: Write to Firebase ─────────────────────────────────────────────
    firebase_key = fb.email_to_firebase_key(email)
    _write_forecast_to_firebase(firebase_key, device_id, forecast, tariff_opt, now)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_device_forecast(
    db: Session,
    device_id: str,
    email: str,
    forecast,
    tariff_opt: TariffOptimization,
    hourly_json: str,
    now: datetime,
) -> None:
    """Insert or update the device_forecasts row for this device."""
    stmt = select(DeviceForecast).where(DeviceForecast.device_id == device_id)
    existing = db.execute(stmt).scalar_one_or_none()

    cheapest_json = json.dumps(tariff_opt.cheapest_hours)
    expensive_json = json.dumps(tariff_opt.most_expensive_hours)
    tips_json = json.dumps(tariff_opt.tips)

    if existing:
        existing.email = email
        existing.generated_at = now
        existing.model_type = forecast.model_type
        existing.training_samples = forecast.training_samples
        existing.hourly_forecasts = hourly_json
        existing.total_predicted_kwh = tariff_opt.total_predicted_kwh
        existing.total_cost_rs = tariff_opt.total_cost_rs
        existing.peak_cost_rs = tariff_opt.peak_cost_rs
        existing.offpeak_cost_rs = tariff_opt.offpeak_cost_rs
        existing.cheapest_hours = cheapest_json
        existing.most_expensive_hours = expensive_json
        existing.potential_savings_rs = tariff_opt.potential_savings_rs
        existing.shiftable_kwh = tariff_opt.shiftable_kwh
        existing.optimization_tips = tips_json
        existing.alert_level = tariff_opt.alert_level
        existing.updated_at = now
    else:
        row = DeviceForecast(
            device_id=device_id,
            email=email,
            generated_at=now,
            model_type=forecast.model_type,
            training_samples=forecast.training_samples,
            hourly_forecasts=hourly_json,
            total_predicted_kwh=tariff_opt.total_predicted_kwh,
            total_cost_rs=tariff_opt.total_cost_rs,
            peak_cost_rs=tariff_opt.peak_cost_rs,
            offpeak_cost_rs=tariff_opt.offpeak_cost_rs,
            cheapest_hours=cheapest_json,
            most_expensive_hours=expensive_json,
            potential_savings_rs=tariff_opt.potential_savings_rs,
            shiftable_kwh=tariff_opt.shiftable_kwh,
            optimization_tips=tips_json,
            alert_level=tariff_opt.alert_level,
            updated_at=now,
        )
        db.add(row)


# ─────────────────────────────────────────────────────────────────────────────
# Firebase write
# ─────────────────────────────────────────────────────────────────────────────

def _write_forecast_to_firebase(
    firebase_key: str,
    device_id: str,
    forecast,
    tariff_opt: TariffOptimization,
    now: datetime,
) -> None:
    """Write a compact forecast node to Firebase for mobile app consumption."""

    # Include only the next 6 hours in the Firebase payload to keep it lean.
    # The full 24-hour breakdown is available via the REST API.
    next_6 = [
        {
            "hour_offset": slot.hour_offset,
            "hour_bucket": slot.hour_bucket.isoformat(),
            "predicted_power_w": slot.predicted_power_w,
            "predicted_kwh": slot.predicted_kwh,
            "tariff_label": slot.label,
            "tariff_rate_per_kwh": slot.rate_per_kwh,
            "cost_rs": slot.cost_rs,
            "confidence": slot.confidence,
        }
        for slot in tariff_opt.slots[:6]
    ]

    payload = {
        # Model info
        "model_type": forecast.model_type,
        "training_samples": forecast.training_samples,
        "generated_at": int(now.timestamp()),

        # 24h totals
        "total_predicted_kwh_24h": tariff_opt.total_predicted_kwh,
        "total_cost_rs_24h": tariff_opt.total_cost_rs,
        "peak_cost_rs_24h": tariff_opt.peak_cost_rs,
        "offpeak_cost_rs_24h": tariff_opt.offpeak_cost_rs,

        # Tariff optimisation summary
        "potential_savings_rs": tariff_opt.potential_savings_rs,
        "shiftable_kwh": tariff_opt.shiftable_kwh,
        "cheapest_hours": tariff_opt.cheapest_hours[:4],
        "alert_level": tariff_opt.alert_level,
        "primary_tip": tariff_opt.tips[0] if tariff_opt.tips else "",

        # Short-horizon hourly breakdown
        "next_6h": next_6,
    }

    fb.write_forecast(firebase_key, payload)
