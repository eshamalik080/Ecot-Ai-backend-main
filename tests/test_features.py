from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.ml.features import compute_features
from conftest import make_reading


def test_compute_features_builds_expected_rolling_values() -> None:
    start = datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)
    readings = [
        make_reading(ts=start + timedelta(minutes=index), power=100.0 * (index + 1), current=1.0 + index, voltage=220.0 + index)
        for index in range(5)
    ]

    features = compute_features(
        device_id="dev-1",
        email="user@example.com",
        readings=readings,
        reference_time=start + timedelta(minutes=4),
    )

    assert features.rolling_avg_power_1m == 500.0
    assert features.rolling_avg_power_5m == 300.0
    assert round(features.delta_power or 0.0, 2) == 100.0
    assert features.rolling_max_current_5m == 5.0
    assert features.hour_of_day == 10
    assert features.night_flag is False


def test_compute_features_returns_empty_vector_without_readings() -> None:
    now = datetime(2026, 4, 14, 23, 0, tzinfo=timezone.utc)

    features = compute_features(
        device_id="dev-1",
        email="user@example.com",
        readings=[],
        reference_time=now,
    )

    assert features.rolling_avg_power_5m is None
    assert features.hour_of_day == 23
    assert features.night_flag is True
