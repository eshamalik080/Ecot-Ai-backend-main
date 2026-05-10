"""
app/services/cleanup.py
───────────────────────
Data retention policy enforcement.

Runs every CLEANUP_INTERVAL_SECONDS seconds (default: 5 minutes).

Policy:
  raw_readings          → keep only last DATA_RETENTION_HOURS hours (default 3h)
  feature_snapshots     → one row per device, no cleanup needed
  ai_results            → one row per device, no cleanup needed
  hourly_summaries      → kept indefinitely (feeds baselines)
  baselines             → kept indefinitely
  suspicious_events     → kept indefinitely (audit log)

Before deleting raw rows, the cleanup job creates hourly_summaries for any
completed hours that haven't been summarised yet, so we never lose history.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import math
from collections import defaultdict

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models.orm import HourlySummary, RawReading

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_cleanup() -> None:
    """
    1. Summarise any raw data for hours that are about to be purged.
    2. Delete raw readings older than DATA_RETENTION_HOURS.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=settings.DATA_RETENTION_HOURS)

    with get_db() as db:
        _summarise_before_purge(db, cutoff)
        deleted = _purge_old_readings(db, cutoff)
        db.commit()

    logger.info(
        "Cleanup complete: deleted %d raw_reading rows older than %s",
        deleted,
        cutoff.isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summarise before purge
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_before_purge(db: Session, cutoff: datetime) -> None:
    """
    For each device, aggregate the rows that are about to be deleted into
    hourly_summaries, skipping hours that are already summarised.
    """
    stmt = select(RawReading).where(RawReading.inserted_at < cutoff)
    rows = list(db.execute(stmt).scalars())

    by_bucket: dict[tuple[str, str, datetime], list[RawReading]] = defaultdict(list)
    for reading in rows:
        hour_bucket = reading.reading_ts.replace(minute=0, second=0, microsecond=0)
        by_bucket[(reading.device_id, reading.email, hour_bucket)].append(reading)

    for (device_id, email, hour_bucket), bucket_rows in by_bucket.items():
        _ensure_hourly_summary(db, device_id, email, hour_bucket, bucket_rows)


def _ensure_hourly_summary(
    db: Session,
    device_id: str,
    email: str,
    hour_bucket: datetime,
    rows: list[RawReading],
) -> None:
    """Create or update the hourly summary for (device_id, hour_bucket)."""
    # Check if summary already exists
    existing_stmt = select(HourlySummary).where(
        HourlySummary.device_id == device_id,
        HourlySummary.hour_bucket == hour_bucket,
    )
    existing = db.execute(existing_stmt).scalar_one_or_none()
    if existing:
        return  # already summarised

    if not rows:
        return

    voltages = [row.voltage for row in rows if row.voltage is not None]
    currents = [row.current for row in rows if row.current is not None]
    powers = [row.power for row in rows if row.power is not None]
    kwh_values = [row.kwh for row in rows if row.kwh is not None]

    avg_voltage = sum(voltages) / len(voltages) if voltages else None
    avg_current = sum(currents) / len(currents) if currents else None
    avg_power = sum(powers) / len(powers) if powers else None
    max_power = max(powers) if powers else None
    min_power = min(powers) if powers else None
    std_power = _sample_stddev(powers)
    kwh_delta = (max(kwh_values) - min(kwh_values)) if len(kwh_values) >= 2 else 0.0 if kwh_values else None

    summary = HourlySummary(
        device_id=device_id,
        email=email,
        hour_bucket=hour_bucket,
        avg_voltage=avg_voltage,
        avg_current=avg_current,
        avg_power=avg_power,
        max_power=max_power,
        min_power=min_power,
        std_power=std_power,
        kwh_delta=kwh_delta,
        reading_count=len(rows),
        anomaly_count=0,  # updated separately by result_writer
        spike_count=0,
    )
    db.add(summary)
    logger.debug(
        "Created hourly summary for device=%s hour=%s (readings=%d)",
        device_id,
        hour_bucket.isoformat(),
        len(rows),
    )


def _sample_stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


# ─────────────────────────────────────────────────────────────────────────────
# Purge
# ─────────────────────────────────────────────────────────────────────────────

def _purge_old_readings(db: Session, cutoff: datetime) -> int:
    """Delete raw_readings older than cutoff. Returns count deleted."""
    result = db.execute(
        delete(RawReading).where(RawReading.inserted_at < cutoff)
    )
    return result.rowcount or 0
