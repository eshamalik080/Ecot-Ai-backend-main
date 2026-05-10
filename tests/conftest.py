from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models.orm import RawReading


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def make_reading(
    *,
    device_id: str = "dev-1",
    email: str = "user@example.com",
    ts: datetime,
    voltage: float | None = 220.0,
    current: float | None = 5.0,
    power: float | None = 1000.0,
    kwh: float | None = 10.0,
) -> RawReading:
    return RawReading(
        device_id=device_id,
        email=email,
        voltage=voltage,
        current=current,
        power=power,
        kwh=kwh,
        reading_ts=ts,
        inserted_at=ts + timedelta(seconds=1),
    )
