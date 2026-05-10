"""
app/db.py
─────────
SQLAlchemy engine, session factory, and Base declarative class.

Usage (in any service/route):

    from app.db import SessionLocal, engine
    with SessionLocal() as session:
        session.add(obj)
        session.commit()
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


# ── Engine ────────────────────────────────────────────────────────────────────

def _build_engine():
    database_url = settings.DATABASE_URL
    engine_kwargs = {"echo": settings.DEBUG}

    if database_url.startswith("sqlite"):
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        engine_kwargs.update(
            {
                "pool_size": settings.DATABASE_POOL_SIZE,
                "max_overflow": settings.DATABASE_MAX_OVERFLOW,
                "pool_pre_ping": True,
                "pool_recycle": 1800,
            }
        )

    return create_engine(database_url, **engine_kwargs)


engine = _build_engine()


# ── Declarative Base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Session Factory ───────────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # keep attributes accessible after commit
)


# ── Context Manager Helper ────────────────────────────────────────────────────

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Yields a database session and guarantees rollback on error and close on exit.
    Use this in background tasks and scripts.

        with get_db() as db:
            db.add(obj)
            db.commit()
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── FastAPI Dependency ────────────────────────────────────────────────────────

def get_db_dep() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a DB session per request.

        @router.get("/")
        def my_route(db: Session = Depends(get_db_dep)):
            ...
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ── Table Creation ────────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables defined in ORM models. Called at startup."""
    # Import models so SQLAlchemy is aware of them before create_all
    from app.models import orm  # noqa: F401

    logger.info("Creating database tables if they don't exist …")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")


def verify_connection() -> None:
    """Quick connectivity check – raises on failure."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("PostgreSQL connection verified.")
