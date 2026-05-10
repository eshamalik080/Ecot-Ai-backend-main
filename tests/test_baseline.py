"""
Tests for app/services/baseline.py – baseline computation from hourly summaries.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from app.models.orm import Baseline, HourlySummary
from app.services.baseline import run_baseline_update, _update_device_baselines


def _make_summary(
    db: Session,
    *,
    device_id: str = "dev-1",
    email: str = "u@e.com",
    hour: int,
    avg_power: float,
    avg_current: float = 5.0,
    kwh_delta: float = 0.5,
    spike_count: int = 0,
    anomaly_count: int = 0,
    reading_count: int = 10,
) -> HourlySummary:
    ts = datetime(2026, 4, 14, hour, 0, tzinfo=timezone.utc)
    s = HourlySummary(
        device_id=device_id,
        email=email,
        hour_bucket=ts,
        avg_voltage=220.0,
        avg_current=avg_current,
        avg_power=avg_power,
        max_power=avg_power + 100,
        min_power=avg_power - 100,
        std_power=20.0,
        kwh_delta=kwh_delta,
        reading_count=reading_count,
        anomaly_count=anomaly_count,
        spike_count=spike_count,
    )
    db.add(s)
    return s


def test_baseline_created_for_device(db_session) -> None:
    _make_summary(db_session, hour=10, avg_power=400.0)
    _make_summary(db_session, hour=10, avg_power=600.0)
    db_session.commit()

    n = _update_device_baselines(db_session, "dev-1")
    db_session.commit()

    assert n == 1  # one hour slot
    baseline = db_session.query(Baseline).filter_by(device_id="dev-1", hour_of_day=10).one()
    assert baseline.avg_power == pytest.approx(500.0)
    assert baseline.sample_count == 2


def test_baseline_created_for_multiple_hours(db_session) -> None:
    for h in [8, 9, 10]:
        _make_summary(db_session, hour=h, avg_power=300.0 + h * 10)
    db_session.commit()

    n = _update_device_baselines(db_session, "dev-1")
    db_session.commit()

    assert n == 3
    baselines = db_session.query(Baseline).filter_by(device_id="dev-1").all()
    assert {b.hour_of_day for b in baselines} == {8, 9, 10}


def test_baseline_updated_on_second_run(db_session) -> None:
    _make_summary(db_session, hour=12, avg_power=300.0)
    db_session.commit()
    _update_device_baselines(db_session, "dev-1")
    db_session.commit()

    # Add another summary for same hour
    _make_summary(db_session, hour=12, avg_power=500.0)
    db_session.commit()
    _update_device_baselines(db_session, "dev-1")
    db_session.commit()

    baselines = db_session.query(Baseline).filter_by(device_id="dev-1", hour_of_day=12).all()
    assert len(baselines) == 1  # upsert, not duplicate
    assert baselines[0].avg_power == pytest.approx(400.0)


def test_baseline_no_summaries_returns_zero(db_session) -> None:
    n = _update_device_baselines(db_session, "dev-no-data")
    assert n == 0


def test_run_baseline_update_multiple_devices(db_session, monkeypatch) -> None:
    for dev in ("dev-A", "dev-B"):
        _make_summary(db_session, device_id=dev, hour=14, avg_power=200.0)
    db_session.commit()

    import app.services.baseline as bl_module
    from contextlib import contextmanager

    @contextmanager
    def fake_get_db():
        yield db_session

    monkeypatch.setattr(bl_module, "get_db", fake_get_db)
    run_baseline_update()

    baselines = db_session.query(Baseline).all()
    devices = {b.device_id for b in baselines}
    assert "dev-A" in devices
    assert "dev-B" in devices


def test_baseline_spike_and_anomaly_freq(db_session) -> None:
    _make_summary(db_session, hour=6, avg_power=300.0, spike_count=5, anomaly_count=2, reading_count=10)
    _make_summary(db_session, hour=6, avg_power=300.0, spike_count=3, anomaly_count=1, reading_count=10)
    db_session.commit()

    _update_device_baselines(db_session, "dev-1")
    db_session.commit()

    baseline = db_session.query(Baseline).filter_by(device_id="dev-1", hour_of_day=6).one()
    # total spikes = 8, total readings = 20 → freq = 0.4
    assert baseline.avg_spike_freq == pytest.approx(0.4)
    assert baseline.avg_anomaly_freq == pytest.approx(0.15)
