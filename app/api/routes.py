"""
app/api/routes.py
─────────────────
All FastAPI route handlers, grouped with APIRouter prefixes.

Endpoints
---------
GET  /health                          – Health check
GET  /model/status                    – ML model status
POST /model/retrain                   – Manually trigger retraining

GET  /devices/{device_id}/result      – Latest AI result for a device
GET  /devices/{device_id}/readings    – Latest raw readings for a device
GET  /devices/{device_id}/summaries   – Hourly summaries for a device

GET  /events/suspicious               – Paginated suspicious event log
GET  /devices                         – List all known devices
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app import firebase_client as fb
from app.config import settings
from app.db import get_db_dep, verify_connection
from app.ml.anomaly import isolation_forest_model
from app.models.orm import AIResult, DeviceForecast, HourlySummary, RawReading, SuspiciousEvent
from app.schemas.readings import (
    AIResultOut,
    DeviceForecastOut,
    HealthResponse,
    HourlyForecastItem,
    HourlySummaryOut,
    LiveReadingOut,
    ModelStatusResponse,
    SuspiciousEventOut,
)
from app.tasks.scheduler import get_scheduler, trigger_job_now

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db_dep)) -> HealthResponse:
    """Returns system health: DB connection, Firebase connection, model status."""
    db_ok = False
    try:
        db.execute(select(RawReading).limit(1))
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (db_ok and fb.is_firebase_ready()) else "degraded",
        db_connected=db_ok,
        firebase_connected=fb.is_firebase_ready(),
        model_loaded=isolation_forest_model.is_loaded,
        version=settings.APP_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/model/status", response_model=ModelStatusResponse, tags=["Model"])
def model_status() -> ModelStatusResponse:
    """Returns the current ML model status and metadata."""
    meta = isolation_forest_model.meta
    return ModelStatusResponse(
        model_loaded=isolation_forest_model.is_loaded,
        model_path=isolation_forest_model.model_path,
        training_samples=meta.get("training_samples"),
        trained_at=meta.get("trained_at"),
        contamination=settings.ANOMALY_CONTAMINATION,
    )


@router.post("/model/retrain", tags=["Model"])
def trigger_retrain() -> dict:
    """
    Manually trigger the training pipeline in a background thread.
    Uses the same training script logic (runs via a scheduler job or directly).
    """
    try:
        # Import here to avoid circular dependency
        from scripts.train import train_from_db
        import threading

        thread = threading.Thread(target=train_from_db, daemon=True)
        thread.start()

        return {"status": "training_started", "message": "Training running in background."}
    except Exception as exc:
        logger.error("Retrain trigger failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Devices list
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/devices", tags=["Devices"])
def list_devices(db: Session = Depends(get_db_dep)) -> list[dict]:
    """Return all known device IDs and their latest AI result status."""
    stmt = select(AIResult).order_by(AIResult.device_id)
    results = db.execute(stmt).scalars().all()
    return [
        {
            "device_id": r.device_id,
            "email": r.email,
            "status": r.status,
            "theft_score": r.theft_score,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in results
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Per-device
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/devices/{device_id}/result", response_model=AIResultOut, tags=["Devices"])
def get_ai_result(device_id: str, db: Session = Depends(get_db_dep)) -> AIResultOut:
    """Latest AI anomaly/theft result for a device."""
    stmt = select(AIResult).where(AIResult.device_id == device_id)
    row = db.execute(stmt).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No AI result found for device '{device_id}'")

    import json
    rule_reasons = None
    if row.rule_reasons:
        try:
            rule_reasons = json.loads(row.rule_reasons)
        except (ValueError, TypeError):
            rule_reasons = [row.rule_reasons]

    return AIResultOut(
        device_id=row.device_id,
        email=row.email,
        anomaly_score=row.anomaly_score,
        theft_score=row.theft_score,
        status=row.status,
        reason=row.reason,
        explanation=row.explanation,
        rule_triggered=row.rule_triggered,
        rule_reasons=rule_reasons,
        updated_at=row.updated_at,
    )


@router.get(
    "/devices/{device_id}/readings",
    response_model=list[LiveReadingOut],
    tags=["Devices"],
)
def get_latest_readings(
    device_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db_dep),
) -> list[LiveReadingOut]:
    """Latest raw readings for a device (most recent first)."""
    stmt = (
        select(RawReading)
        .where(RawReading.device_id == device_id)
        .order_by(desc(RawReading.reading_ts))
        .limit(limit)
    )
    rows = db.execute(stmt).scalars().all()
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No readings found for device '{device_id}'",
        )
    return [LiveReadingOut.model_validate(r) for r in rows]


@router.get(
    "/devices/{device_id}/summaries",
    response_model=list[HourlySummaryOut],
    tags=["Devices"],
)
def get_hourly_summaries(
    device_id: str,
    limit: int = Query(default=24, ge=1, le=720),  # max 30 days
    db: Session = Depends(get_db_dep),
) -> list[HourlySummaryOut]:
    """Hourly summaries for a device (most recent first)."""
    stmt = (
        select(HourlySummary)
        .where(HourlySummary.device_id == device_id)
        .order_by(desc(HourlySummary.hour_bucket))
        .limit(limit)
    )
    rows = db.execute(stmt).scalars().all()
    return [HourlySummaryOut.model_validate(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/events/suspicious",
    response_model=list[SuspiciousEventOut],
    tags=["Events"],
)
def get_suspicious_events(
    device_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None, description="Filter: suspicious | possible_theft"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db_dep),
) -> list[SuspiciousEventOut]:
    """Paginated list of suspicious / possible_theft events. Optional device filter."""
    stmt = select(SuspiciousEvent).order_by(desc(SuspiciousEvent.occurred_at))

    if device_id:
        stmt = stmt.where(SuspiciousEvent.device_id == device_id)
    if status:
        stmt = stmt.where(SuspiciousEvent.status == status)

    stmt = stmt.offset(offset).limit(limit)
    rows = db.execute(stmt).scalars().all()
    return [SuspiciousEventOut.model_validate(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Forecast & tariff
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/devices/{device_id}/forecast",
    response_model=DeviceForecastOut,
    tags=["Forecast"],
)
def get_device_forecast(
    device_id: str,
    db: Session = Depends(get_db_dep),
) -> DeviceForecastOut:
    """
    Latest 24-hour energy consumption forecast and tariff-based cost
    optimization for a device.

    The forecast is regenerated by the background `forecasting_job` every hour.
    Use POST /forecast/trigger to force an immediate refresh.
    """
    import json as _json

    stmt = select(DeviceForecast).where(DeviceForecast.device_id == device_id)
    row = db.execute(stmt).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No forecast found for device '{device_id}'. "
                "The forecasting job runs hourly; wait for the first cycle "
                "or trigger it via POST /forecast/trigger."
            ),
        )

    # Deserialise JSON fields
    hourly: list[HourlyForecastItem] = []
    if row.hourly_forecasts:
        try:
            raw_items = _json.loads(row.hourly_forecasts)
            hourly = [HourlyForecastItem(**item) for item in raw_items]
        except Exception:
            hourly = []

    cheapest = []
    if row.cheapest_hours:
        try:
            cheapest = _json.loads(row.cheapest_hours)
        except Exception:
            pass

    most_expensive = []
    if row.most_expensive_hours:
        try:
            most_expensive = _json.loads(row.most_expensive_hours)
        except Exception:
            pass

    tips = []
    if row.optimization_tips:
        try:
            tips = _json.loads(row.optimization_tips)
        except Exception:
            pass

    return DeviceForecastOut(
        device_id=row.device_id,
        email=row.email,
        generated_at=row.generated_at,
        model_type=row.model_type,
        training_samples=row.training_samples,
        hourly_forecasts=hourly,
        total_predicted_kwh=row.total_predicted_kwh,
        total_cost_rs=row.total_cost_rs,
        peak_cost_rs=row.peak_cost_rs,
        offpeak_cost_rs=row.offpeak_cost_rs,
        cheapest_hours=cheapest,
        most_expensive_hours=most_expensive,
        potential_savings_rs=row.potential_savings_rs,
        shiftable_kwh=row.shiftable_kwh,
        optimization_tips=tips,
        alert_level=row.alert_level,
        updated_at=row.updated_at,
    )


@router.post("/forecast/trigger", tags=["Forecast"])
def trigger_forecast_now() -> dict:
    """
    Immediately trigger the forecasting job for all devices in a background thread.
    Useful after accumulating new hourly summaries or during development.
    """
    import threading
    from app.services.forecasting import run_forecast_cycle

    thread = threading.Thread(target=run_forecast_cycle, daemon=True)
    thread.start()
    return {"status": "forecast_started", "message": "Forecast cycle running in background."}


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler control (dev/ops use)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/scheduler/trigger/{job_id}", tags=["System"])
def trigger_scheduler_job(job_id: str) -> dict:
    """
    Immediately trigger a scheduled job by its ID.
    Available job IDs: ingestion_job, cleanup_job, baseline_job, forecasting_job
    """
    success = trigger_job_now(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {"status": "triggered", "job_id": job_id}


@router.get("/scheduler/jobs", tags=["System"])
def list_scheduler_jobs() -> list[dict]:
    """List all registered APScheduler jobs and their next run time."""
    scheduler = get_scheduler()
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append(
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
        )
    return jobs
