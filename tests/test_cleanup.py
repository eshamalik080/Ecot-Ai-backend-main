from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models.orm import HourlySummary, RawReading
from app.services.cleanup import _purge_old_readings, _summarise_before_purge
from conftest import make_reading


def test_cleanup_summarises_old_rows_before_purging(db_session) -> None:
    hour_start = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
    rows = [
        make_reading(ts=hour_start + timedelta(minutes=0), power=100.0, current=1.0, kwh=10.0),
        make_reading(ts=hour_start + timedelta(minutes=10), power=130.0, current=1.5, kwh=10.4),
        make_reading(ts=hour_start + timedelta(minutes=20), power=160.0, current=2.0, kwh=10.9),
    ]
    db_session.add_all(rows)
    db_session.commit()

    cutoff = hour_start + timedelta(hours=4)
    _summarise_before_purge(db_session, cutoff)
    deleted = _purge_old_readings(db_session, cutoff)
    db_session.commit()

    summary = db_session.query(HourlySummary).one()
    remaining_rows = db_session.query(RawReading).count()

    assert deleted == 3
    assert remaining_rows == 0
    assert summary.device_id == "dev-1"
    assert summary.hour_bucket == hour_start
    assert summary.reading_count == 3
    assert round(summary.avg_power or 0.0, 2) == 130.0
    assert summary.max_power == 160.0
    assert summary.min_power == 100.0
    assert round(summary.kwh_delta or 0.0, 2) == 0.9
