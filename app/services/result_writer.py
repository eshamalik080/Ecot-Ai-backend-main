"""
app/services/result_writer.py
──────────────────────────────
Utility for writing AI results to Firebase and updating hourly summaries
with anomaly/spike counts. Separated from ingestion.py for clarity.

Suggested Firebase schema
─────────────────────────

/
├── <user_key>/               ← live device data (written by firmware)
│   ├── device_id
│   ├── email
│   ├── voltage
│   ├── current
│   ├── power
│   ├── kwh
│   ├── load1 / load2 / load3 / overall
│   └── ai_result/            ← written by this backend
│       ├── anomaly_score
│       ├── theft_score
│       ├── status
│       ├── reason
│       ├── explanation
│       ├── rule_triggered
│       ├── rule_reasons      ← array
│       └── updated_at        ← unix timestamp
│
└── alerts/                   ← optional: push here for FCM notifications
    └── <user_key>/
        └── <alert_id>/
            ├── status
            ├── theft_score
            ├── reason
            └── occurred_at
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.orm import HourlySummary
from sqlalchemy import select, func

logger = logging.getLogger(__name__)


def increment_hourly_anomaly_count(db: Session, device_id: str, ts: datetime) -> None:
    """
    Increment the anomaly_count for the hourly_summary row that contains ts.
    Called after an anomaly is confirmed.
    """
    from datetime import timedelta

    hour_bucket = ts.replace(minute=0, second=0, microsecond=0)
    stmt = select(HourlySummary).where(
        HourlySummary.device_id == device_id,
        HourlySummary.hour_bucket == hour_bucket,
    )
    row = db.execute(stmt).scalar_one_or_none()
    if row:
        row.anomaly_count = (row.anomaly_count or 0) + 1
        logger.debug(
            "anomaly_count incremented for device=%s hour=%s → %d",
            device_id,
            hour_bucket,
            row.anomaly_count,
        )


def increment_hourly_spike_count(db: Session, device_id: str, ts: datetime) -> None:
    """Increment spike_count for the hourly summary row that contains ts."""
    hour_bucket = ts.replace(minute=0, second=0, microsecond=0)
    stmt = select(HourlySummary).where(
        HourlySummary.device_id == device_id,
        HourlySummary.hour_bucket == hour_bucket,
    )
    row = db.execute(stmt).scalar_one_or_none()
    if row:
        row.spike_count = (row.spike_count or 0) + 1
