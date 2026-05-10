"""
app/services/baseline.py
────────────────────────
Baseline computation and update service.

Runs every BASELINE_UPDATE_INTERVAL_SECONDS (default: 1 hour).

Reads all available hourly_summaries for each device and computes
per-(device, hour_of_day) statistics. These baselines are what the
theft scoring uses to judge whether current consumption is "normal".

Because we never delete hourly_summaries, baselines become more
accurate over time without needing to store raw readings long-term.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.orm import Baseline, HourlySummary

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_update() -> None:
    """Recompute baselines from all hourly summaries. Called periodically."""
    with get_db() as db:
        devices = _get_all_devices(db)

        updated = 0
        for device_id in devices:
            try:
                n = _update_device_baselines(db, device_id)
                updated += n
            except Exception as exc:
                logger.error(
                    "Baseline update failed for device %s: %s", device_id, exc
                )

        db.commit()
        logger.info("Baseline update complete: updated %d (device, hour) slots.", updated)


# ─────────────────────────────────────────────────────────────────────────────
# Per-device baseline computation
# ─────────────────────────────────────────────────────────────────────────────

def _get_all_devices(db: Session) -> list[str]:
    """Return all distinct device IDs that have hourly summaries."""
    stmt = select(HourlySummary.device_id).distinct()
    return list(db.execute(stmt).scalars())


def _update_device_baselines(db: Session, device_id: str) -> int:
    """
    Compute per-hour-of-day baselines for a device from all available summaries.
    Returns number of baseline rows updated/created.
    """
    stmt = select(HourlySummary).where(HourlySummary.device_id == device_id)
    summaries = list(db.execute(stmt).scalars())

    if not summaries:
        return 0

    # Group summaries by hour_of_day
    by_hour: dict[int, list[HourlySummary]] = {}
    for s in summaries:
        if s.hour_bucket is None:
            continue
        hour = s.hour_bucket.hour
        by_hour.setdefault(hour, []).append(s)

    n_updated = 0
    for hour, hour_summaries in by_hour.items():
        _upsert_baseline(db, device_id, hour, hour_summaries)
        n_updated += 1

    return n_updated


def _upsert_baseline(
    db: Session, device_id: str, hour: int, summaries: list[HourlySummary]
) -> None:
    """Compute and persist (or update) a baseline row for (device_id, hour)."""
    # Extract arrays, ignoring None values
    powers = [s.avg_power for s in summaries if s.avg_power is not None]
    currents = [s.avg_current for s in summaries if s.avg_current is not None]
    kwh_deltas = [s.kwh_delta for s in summaries if s.kwh_delta is not None]
    spike_counts = [s.spike_count for s in summaries if s.spike_count is not None]
    anomaly_counts = [s.anomaly_count for s in summaries if s.anomaly_count is not None]
    reading_counts = [s.reading_count for s in summaries if s.reading_count]

    avg_power = float(np.mean(powers)) if powers else None
    std_power = float(np.std(powers)) if len(powers) > 1 else None
    avg_current = float(np.mean(currents)) if currents else None
    std_current = float(np.std(currents)) if len(currents) > 1 else None
    avg_kwh_growth = float(np.mean(kwh_deltas)) if kwh_deltas else None

    # Spike frequency: average spikes per reading (spikes/hour normalised by readings)
    total_readings = sum(reading_counts) if reading_counts else 1
    total_spikes = sum(spike_counts) if spike_counts else 0
    avg_spike_freq = total_spikes / total_readings if total_readings else 0.0

    total_anomalies = sum(anomaly_counts) if anomaly_counts else 0
    avg_anomaly_freq = total_anomalies / total_readings if total_readings else 0.0

    # Upsert
    stmt = select(Baseline).where(
        Baseline.device_id == device_id,
        Baseline.hour_of_day == hour,
    )
    existing = db.execute(stmt).scalar_one_or_none()

    now = datetime.now(timezone.utc)

    if existing:
        existing.avg_power = avg_power
        existing.std_power = std_power
        existing.avg_current = avg_current
        existing.std_current = std_current
        existing.avg_kwh_growth = avg_kwh_growth
        existing.avg_spike_freq = avg_spike_freq
        existing.avg_anomaly_freq = avg_anomaly_freq
        existing.sample_count = len(summaries)
        existing.updated_at = now
    else:
        baseline = Baseline(
            device_id=device_id,
            hour_of_day=hour,
            avg_power=avg_power,
            std_power=std_power,
            avg_current=avg_current,
            std_current=std_current,
            avg_kwh_growth=avg_kwh_growth,
            avg_spike_freq=avg_spike_freq,
            avg_anomaly_freq=avg_anomaly_freq,
            sample_count=len(summaries),
            updated_at=now,
        )
        db.add(baseline)
