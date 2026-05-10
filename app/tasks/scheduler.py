"""
app/tasks/scheduler.py
──────────────────────
APScheduler configuration and job registration.

Jobs registered
---------------
1. ingestion_job    – polls Firebase every POLLING_INTERVAL_SECONDS
2. cleanup_job      – purges old raw readings every CLEANUP_INTERVAL_SECONDS
3. baseline_job     – rebuilds baselines every BASELINE_UPDATE_INTERVAL_SECONDS

The scheduler uses BackgroundScheduler (runs in a daemon thread), which is
the right choice for WSGI/sync FastAPI. It starts on application startup
and shuts down cleanly on application shutdown.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level scheduler instance
_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    """Return the module-level scheduler, creating it if needed."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone="UTC")
    return _scheduler


def start_scheduler() -> None:
    """Register all jobs and start the scheduler. Called at FastAPI startup."""
    # Import services here to avoid circular imports at module load time
    from app.services.ingestion import run_ingestion_cycle
    from app.services.cleanup import run_cleanup
    from app.services.baseline import run_baseline_update
    from app.services.forecasting import run_forecast_cycle

    scheduler = get_scheduler()

    # ── Job 1: Firebase polling ────────────────────────────────────────────
    scheduler.add_job(
        func=run_ingestion_cycle,
        trigger=IntervalTrigger(seconds=settings.POLLING_INTERVAL_SECONDS),
        id="ingestion_job",
        name="Firebase ingestion",
        replace_existing=True,
        max_instances=1,        # never overlap; skip if previous run still going
        coalesce=True,
    )
    logger.info(
        "Scheduled: ingestion_job every %ds", settings.POLLING_INTERVAL_SECONDS
    )

    # ── Job 2: Data retention cleanup ─────────────────────────────────────
    scheduler.add_job(
        func=run_cleanup,
        trigger=IntervalTrigger(seconds=settings.CLEANUP_INTERVAL_SECONDS),
        id="cleanup_job",
        name="Data retention cleanup",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    logger.info(
        "Scheduled: cleanup_job every %ds", settings.CLEANUP_INTERVAL_SECONDS
    )

    # ── Job 3: Baseline update ─────────────────────────────────────────────
    scheduler.add_job(
        func=run_baseline_update,
        trigger=IntervalTrigger(seconds=settings.BASELINE_UPDATE_INTERVAL_SECONDS),
        id="baseline_job",
        name="Baseline update",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    logger.info(
        "Scheduled: baseline_job every %ds", settings.BASELINE_UPDATE_INTERVAL_SECONDS
    )

    # ── Job 4: 24-hour energy forecast + tariff optimisation ──────────────
    scheduler.add_job(
        func=run_forecast_cycle,
        trigger=IntervalTrigger(seconds=settings.FORECAST_UPDATE_INTERVAL_SECONDS),
        id="forecasting_job",
        name="Energy forecast & tariff optimization",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    logger.info(
        "Scheduled: forecasting_job every %ds", settings.FORECAST_UPDATE_INTERVAL_SECONDS
    )

    scheduler.start()
    logger.info("APScheduler started with %d jobs.", len(scheduler.get_jobs()))


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler. Called at FastAPI shutdown."""
    scheduler = get_scheduler()
    if scheduler.running:
        scheduler.shutdown(wait=True)
        logger.info("APScheduler stopped.")


def trigger_job_now(job_id: str) -> bool:
    """
    Manually trigger a scheduled job by ID.
    Available IDs: ingestion_job, cleanup_job, baseline_job, forecasting_job
    Returns True if job was found and triggered, False otherwise.
    """
    scheduler = get_scheduler()
    job = scheduler.get_job(job_id)
    if job is None:
        logger.warning("trigger_job_now: job '%s' not found.", job_id)
        return False
    job.modify(next_run_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    logger.info("Manually triggered job: %s", job_id)
    return True
