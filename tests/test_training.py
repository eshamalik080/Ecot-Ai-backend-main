from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from app.ml.anomaly import isolation_forest_model
from app.models.orm import RawReading
from conftest import make_reading
from scripts.train import train_from_db


def test_train_from_db_trains_and_saves_model(db_session, monkeypatch, tmp_path) -> None:
    start = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)

    readings: list[RawReading] = []
    for index in range(80):
        ts = start + timedelta(minutes=index)
        readings.append(
            make_reading(
                device_id="dev-1",
                email="user@example.com",
                ts=ts,
                voltage=220.0 + (index % 3),
                current=2.0 + (index % 5) * 0.1,
                power=400.0 + index * 3.0,
                kwh=10.0 + index * 0.05,
            )
        )

    db_session.add_all(readings)
    db_session.commit()

    @contextmanager
    def fake_get_db():
        yield db_session

    monkeypatch.setattr("scripts.train.get_db", fake_get_db)
    monkeypatch.setattr("scripts.train.isolation_forest_model._loaded", False)
    monkeypatch.setattr("scripts.train.isolation_forest_model._model", None)
    monkeypatch.setattr("scripts.train.isolation_forest_model._meta", {})
    monkeypatch.setattr("app.config.settings.MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("app.config.settings.MIN_TRAINING_SAMPLES", 10)

    meta = train_from_db()

    assert meta["training_samples"] >= 10
    assert meta["features"]
    assert (tmp_path / "isolation_forest.joblib").exists()
    assert isolation_forest_model.is_loaded is True
