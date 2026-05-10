"""
Tests for app/services/result_writer.py – hourly counter increments.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.models.orm import HourlySummary
from app.services.result_writer import increment_hourly_anomaly_count, increment_hourly_spike_count


def _make_summary(db_session, *, device_id: str = "dev-1", hour: int = 10) -> HourlySummary:
    ts = datetime(2026, 4, 14, hour, 0, tzinfo=timezone.utc)
    s = HourlySummary(
        device_id=device_id,
        email="u@e.com",
        hour_bucket=ts,
        avg_voltage=220.0,
        avg_current=5.0,
        avg_power=1000.0,
        max_power=1100.0,
        min_power=900.0,
        std_power=50.0,
        kwh_delta=0.5,
        reading_count=10,
        anomaly_count=0,
        spike_count=0,
    )
    db_session.add(s)
    db_session.commit()
    return s


def test_increment_anomaly_count(db_session) -> None:
    _make_summary(db_session)
    ts = datetime(2026, 4, 14, 10, 30, tzinfo=timezone.utc)  # within hour 10
    increment_hourly_anomaly_count(db_session, "dev-1", ts)
    db_session.commit()

    row = db_session.query(HourlySummary).filter_by(device_id="dev-1").one()
    assert row.anomaly_count == 1


def test_increment_anomaly_count_multiple_times(db_session) -> None:
    _make_summary(db_session)
    ts = datetime(2026, 4, 14, 10, 15, tzinfo=timezone.utc)
    increment_hourly_anomaly_count(db_session, "dev-1", ts)
    increment_hourly_anomaly_count(db_session, "dev-1", ts)
    db_session.commit()

    row = db_session.query(HourlySummary).filter_by(device_id="dev-1").one()
    assert row.anomaly_count == 2


def test_increment_spike_count(db_session) -> None:
    _make_summary(db_session)
    ts = datetime(2026, 4, 14, 10, 45, tzinfo=timezone.utc)
    increment_hourly_spike_count(db_session, "dev-1", ts)
    db_session.commit()

    row = db_session.query(HourlySummary).filter_by(device_id="dev-1").one()
    assert row.spike_count == 1


def test_increment_no_matching_summary_does_not_raise(db_session) -> None:
    """When there is no summary row for the given hour, the function is a no-op."""
    ts = datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)
    # No summary added – should not raise
    increment_hourly_anomaly_count(db_session, "dev-ghost", ts)
    increment_hourly_spike_count(db_session, "dev-ghost", ts)


def test_increment_anomaly_for_correct_hour_only(db_session) -> None:
    _make_summary(db_session, hour=10)
    _make_summary(db_session, hour=11)

    ts = datetime(2026, 4, 14, 10, 5, tzinfo=timezone.utc)  # belongs to hour 10
    increment_hourly_anomaly_count(db_session, "dev-1", ts)
    db_session.commit()

    hour10 = db_session.query(HourlySummary).filter_by(device_id="dev-1", hour_bucket=datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)).one()
    hour11 = db_session.query(HourlySummary).filter_by(device_id="dev-1", hour_bucket=datetime(2026, 4, 14, 11, 0, tzinfo=timezone.utc)).one()
    assert hour10.anomaly_count == 1
    assert hour11.anomaly_count == 0
