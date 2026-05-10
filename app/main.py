"""
app/main.py
───────────
FastAPI application entry point.

Startup sequence
────────────────
1. Configure logging.
2. Verify PostgreSQL connection and create tables.
3. Initialise Firebase Admin SDK.
4. Load the ML model from disk (if available).
5. Start the APScheduler background jobs.

Shutdown sequence
─────────────────
1. Stop APScheduler gracefully.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.logging_config import setup_logging

# Configure logging before anything else
setup_logging()

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI-powered anomaly detection and theft detection backend "
            "for live PZEM electricity data from Firebase Realtime Database."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Adjust origins for production; allow all in dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    from app.api.routes import router
    app.include_router(router, prefix="/api/v1")

    # ── Lifecycle events ──────────────────────────────────────────────────────
    @app.on_event("startup")
    def startup_event() -> None:
        logger.info("=" * 60)
        logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
        logger.info("=" * 60)

        # 1. Database
        try:
            from app.db import create_tables, verify_connection
            verify_connection()
            create_tables()
        except Exception as exc:
            logger.critical("Database startup failed: %s", exc)
            raise RuntimeError(f"Cannot connect to PostgreSQL: {exc}") from exc

        # 2. Firebase
        try:
            from app.firebase_client import init_firebase
            init_firebase()
        except Exception as exc:
            logger.critical("Firebase startup failed: %s", exc)
            raise RuntimeError(f"Cannot initialise Firebase: {exc}") from exc

        # 3. ML Model (non-fatal – system works without a pre-trained model)
        try:
            from app.ml.anomaly import isolation_forest_model
            loaded = isolation_forest_model.load()
            if loaded:
                logger.info("ML model loaded successfully.")
            else:
                logger.warning(
                    "No trained model found. Rule-based detection only until "
                    "training is run (POST /api/v1/model/retrain or scripts/train.py)."
                )
        except Exception as exc:
            logger.warning("ML model load error (non-fatal): %s", exc)

        # 4. Scheduler
        try:
            from app.tasks.scheduler import start_scheduler
            start_scheduler()
        except Exception as exc:
            logger.critical("Scheduler startup failed: %s", exc)
            raise RuntimeError(f"Cannot start scheduler: {exc}") from exc

        logger.info("Startup complete. API listening on http://0.0.0.0:8000")
        logger.info("Swagger UI: http://localhost:8000/docs")

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        logger.info("Shutting down …")
        from app.tasks.scheduler import stop_scheduler
        stop_scheduler()
        logger.info("Shutdown complete.")

    return app


# ── App instance ───────────────────────────────────────────────────────────────
# Uvicorn imports this: `uvicorn app.main:app`
app = create_app()
