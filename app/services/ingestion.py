"""
app/services/ingestion.py
─────────────────────────
Firebase polling loop and full ingestion pipeline.

Each poll cycle (every POLLING_INTERVAL_SECONDS seconds):
  1. Read all device nodes from Firebase.
  2. Validate each reading with Pydantic.
  3. Persist new raw readings to PostgreSQL.
  4. Compute feature vectors for each device.
  5. Run anomaly detection (rule-based + ML).
  6. Run theft scoring.
  7. Persist AI results to PostgreSQL.
  8. Write AI results back to Firebase.
  9. Log any suspicious events.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app import firebase_client as fb
from app.config import settings
from app.db import get_db
from app.ml.anomaly import detect_anomaly
from app.ml.features import compute_features
from app.ml.theft import compute_theft_score
from app.models.orm import AIResult, FeatureSnapshot, RawReading, SuspiciousEvent
from app.schemas.readings import FirebaseReading, FeatureVector
from app.services.result_writer import increment_hourly_anomaly_count, increment_hourly_spike_count

logger = logging.getLogger(__name__)

# Guard: prevent multiple concurrent auto-training threads
_auto_training_running: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point (called by APScheduler)
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion_cycle() -> None:
    """
    Full ingestion cycle. Called every POLLING_INTERVAL_SECONDS.
    Exceptions are caught per-device so one bad device cannot block others.
    After processing, triggers auto-training if the Isolation Forest model
    is not yet loaded and enough raw readings have been collected.
    """
    readings = fb.read_all_devices()

    if not readings:
        logger.debug("Ingestion cycle: no data from Firebase.")
        return

    logger.info("Ingestion cycle: processing %d device(s) …", len(readings))

    with get_db() as db:
        for reading in readings:
            try:
                _process_device(db, reading)
            except Exception as exc:
                logger.error(
                    "Error processing device %s: %s", reading.device_id, exc, exc_info=True
                )
        db.commit()

    # Auto-train the ML model once enough data has accumulated
    _maybe_auto_train()


# ─────────────────────────────────────────────────────────────────────────────
# Per-device pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _process_device(db: Session, reading: FirebaseReading) -> None:
    """End-to-end pipeline for one device reading."""
    now = datetime.now(timezone.utc)

    # ── Step 1: Determine reading timestamp ───────────────────────────────────
    if reading.timestamp:
        try:
            reading_ts = datetime.fromtimestamp(reading.timestamp, tz=timezone.utc)
        except (OSError, OverflowError):
            reading_ts = now
    else:
        reading_ts = now

    # ── Step 2: Persist raw reading ───────────────────────────────────────────
    raw = RawReading(
        device_id=reading.device_id,
        email=reading.email,
        voltage=reading.voltage,
        current=reading.current,
        power=reading.power,
        kwh=reading.kwh,
        power_factor=reading.power_factor,
        frequency=reading.frequency,
        load1=reading.load1,
        load2=reading.load2,
        load3=reading.load3,
        overall=reading.overall,
        reading_ts=reading_ts,
    )
    db.add(raw)
    db.flush()  # get the id without full commit

    # ── Step 3: Fetch recent readings for feature computation ─────────────────
    from datetime import timedelta
    from sqlalchemy import select

    cutoff = now - timedelta(hours=settings.DATA_RETENTION_HOURS)
    stmt = (
        select(RawReading)
        .where(
            RawReading.device_id == reading.device_id,
            RawReading.reading_ts >= cutoff,
        )
        .order_by(RawReading.reading_ts.asc())
    )
    recent_readings = list(db.execute(stmt).scalars())

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    features = compute_features(
        device_id=reading.device_id,
        email=reading.email,
        readings=recent_readings,
        reference_time=now,
    )
    _upsert_feature_snapshot(db, features, now)

    # ── Step 5: Anomaly detection ─────────────────────────────────────────────
    anomaly_result = detect_anomaly(reading, features)

    # ── Step 6: Fetch baseline and monthly kWh ───────────────────────────────
    baseline = _get_baseline(db, reading.device_id, now.hour)
    monthly_kwh = _get_monthly_kwh(db, reading.device_id, now)

    # ── Step 7: Theft scoring ─────────────────────────────────────────────────
    theft = compute_theft_score(reading, features, anomaly_result, baseline)

    logger.info(
        "[%s] status=%s  anomaly=%.2f  theft=%.2f  %s",
        reading.device_id,
        theft.status,
        anomaly_result.combined_score,
        theft.score,
        f"[{'; '.join(anomaly_result.rule.reasons[:1])}]" if anomaly_result.rule.reasons else "",
    )

    # ── Step 8: Persist AI result ─────────────────────────────────────────────
    rule_reasons_json = json.dumps(anomaly_result.rule.reasons) if anomaly_result.rule.reasons else None
    _upsert_ai_result(db, reading, anomaly_result, theft, rule_reasons_json, now)

    # ── Step 9: Update hourly anomaly / spike counters ───────────────────────
    if theft.status in ("anomaly", "suspicious", "possible_theft"):
        increment_hourly_anomaly_count(db, reading.device_id, now)
    if anomaly_result.rule.triggered and any(
        "spike" in r.lower() for r in anomaly_result.rule.reasons
    ):
        increment_hourly_spike_count(db, reading.device_id, now)

    # ── Step 10: Log suspicious events ───────────────────────────────────────
    if theft.status in ("suspicious", "possible_theft"):
        event = SuspiciousEvent(
            device_id=reading.device_id,
            email=reading.email,
            status=theft.status,
            anomaly_score=anomaly_result.combined_score,
            theft_score=theft.score,
            reason=theft.reason,
            explanation=theft.explanation,
            voltage=reading.voltage,
            current=reading.current,
            power=reading.power,
            kwh=reading.kwh,
            occurred_at=now,
        )
        db.add(event)

    # ── Step 11: Write results back to Firebase ──────────────────────────────
    firebase_key = fb.email_to_firebase_key(reading.email)
    fb.write_ai_result(
        firebase_key,
        {
            "anomaly_score": round(anomaly_result.combined_score, 4),
            "theft_score": round(theft.score, 4),
            "status": theft.status,
            "reason": theft.reason,
            "explanation": theft.explanation,
            "rule_triggered": anomaly_result.rule.triggered,
            "rule_reasons": anomaly_result.rule.reasons,
            "updated_at": int(now.timestamp()),
        },
    )

    # ── Step 12: Write suggestion node to Firebase ───────────────────────────
    suggestion_payload = _build_suggestion(reading, anomaly_result, theft, now, monthly_kwh)
    fb.write_suggestion(firebase_key, suggestion_payload)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_feature_snapshot(
    db: Session, fv: FeatureVector, now: datetime
) -> None:
    """Insert or update the feature snapshot for a device."""
    from sqlalchemy import select

    stmt = select(FeatureSnapshot).where(FeatureSnapshot.device_id == fv.device_id)
    existing = db.execute(stmt).scalar_one_or_none()

    if existing:
        existing.email = fv.email
        existing.rolling_avg_power_1m = fv.rolling_avg_power_1m
        existing.rolling_avg_power_5m = fv.rolling_avg_power_5m
        existing.rolling_std_power_5m = fv.rolling_std_power_5m
        existing.rolling_avg_current_5m = fv.rolling_avg_current_5m
        existing.rolling_max_current_5m = fv.rolling_max_current_5m
        existing.delta_power = fv.delta_power
        existing.delta_current = fv.delta_current
        existing.voltage_deviation = fv.voltage_deviation
        existing.voltage_out_of_range = fv.voltage_out_of_range
        existing.spike_count_15m = fv.spike_count_15m
        existing.hour_of_day = fv.hour_of_day
        existing.day_of_week = fv.day_of_week
        existing.night_flag = fv.night_flag
        existing.computed_at = now
    else:
        snapshot = FeatureSnapshot(
            device_id=fv.device_id,
            email=fv.email,
            rolling_avg_power_1m=fv.rolling_avg_power_1m,
            rolling_avg_power_5m=fv.rolling_avg_power_5m,
            rolling_std_power_5m=fv.rolling_std_power_5m,
            rolling_avg_current_5m=fv.rolling_avg_current_5m,
            rolling_max_current_5m=fv.rolling_max_current_5m,
            delta_power=fv.delta_power,
            delta_current=fv.delta_current,
            voltage_deviation=fv.voltage_deviation,
            voltage_out_of_range=fv.voltage_out_of_range,
            spike_count_15m=fv.spike_count_15m,
            hour_of_day=fv.hour_of_day,
            day_of_week=fv.day_of_week,
            night_flag=fv.night_flag,
            computed_at=now,
        )
        db.add(snapshot)


def _upsert_ai_result(
    db: Session,
    reading: FirebaseReading,
    anomaly_result,
    theft,
    rule_reasons_json: Optional[str],
    now: datetime,
) -> None:
    """Insert or update the AI result for a device."""
    from sqlalchemy import select

    stmt = select(AIResult).where(AIResult.device_id == reading.device_id)
    existing = db.execute(stmt).scalar_one_or_none()

    if existing:
        existing.email = reading.email
        existing.anomaly_score = anomaly_result.combined_score
        existing.rule_triggered = anomaly_result.rule.triggered
        existing.rule_reasons = rule_reasons_json
        existing.theft_score = theft.score
        existing.status = theft.status
        existing.reason = theft.reason
        existing.explanation = theft.explanation
        existing.updated_at = now
    else:
        result = AIResult(
            device_id=reading.device_id,
            email=reading.email,
            anomaly_score=anomaly_result.combined_score,
            rule_triggered=anomaly_result.rule.triggered,
            rule_reasons=rule_reasons_json,
            theft_score=theft.score,
            status=theft.status,
            reason=theft.reason,
            explanation=theft.explanation,
            updated_at=now,
        )
        db.add(result)


def _get_monthly_kwh(db: Session, device_id: str, now: datetime) -> float:
    """
    Return total kWh consumed so far this calendar month for a device,
    by summing kwh_delta from hourly_summaries.
    Returns 0.0 when no summaries exist yet.
    """
    from sqlalchemy import func, select
    from app.models.orm import HourlySummary

    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    result = db.execute(
        select(func.sum(HourlySummary.kwh_delta)).where(
            HourlySummary.device_id == device_id,
            HourlySummary.hour_bucket >= month_start,
        )
    ).scalar()
    return float(result or 0.0)


def _get_tariff_context(
    power_w: Optional[float],
    utc_hour: int,
    monthly_kwh: float,
) -> dict:
    """
    Compute the current Pakistan (PKT = UTC+5) NEPRA tariff context.

    Tariff schedule applied
    -----------------------
    Sanctioned load < 5 kW  → NEPRA residential slabs based on monthly units
    Sanctioned load >= 5 kW → TOU: Peak Rs 46.85 / Off-peak Rs 34.53

    Peak window (both schedules): 5 PM – 5 AM PKT

    Returns a dict with:
        pkt_hour        – local Pakistan hour (0-23)
        is_peak         – True during 5 PM – 5 AM PKT
        period          – "Peak" | "Off-Peak"
        rate            – applicable Rs / kWh
        slab_label      – e.g. "101–200 units" or "TOU"
        schedule        – "slab" | "tou"
        monthly_kwh     – kWh consumed this month so far
        window_str      – "5 PM – 5 AM" | "5 AM – 5 PM"
        hourly_cost_rs  – estimated cost at current power for 1 hour
        cost_str        – formatted sentence (empty if power unknown)
        currency        – "Rs"
    """
    from app.ml.tariff import get_nepra_slab_rate, NEPRA_TOU_PEAK_RATE, NEPRA_TOU_OFFPEAK_RATE

    pkt_hour   = (utc_hour + settings.TARIFF_TIMEZONE_OFFSET) % 24
    peak_start = settings.TARIFF_PEAK_START_HOUR   # 17
    peak_end   = settings.TARIFF_PEAK_END_HOUR     # 5
    cur        = settings.TARIFF_CURRENCY

    # Peak window wraps midnight (17 > 5)
    is_peak    = pkt_hour >= peak_start or pkt_hour < peak_end
    window_str = "5 PM – 5 AM" if is_peak else "5 AM – 5 PM"
    period     = "Peak" if is_peak else "Off-Peak"

    # ── Choose tariff schedule ────────────────────────────────────────────────
    if settings.SANCTIONED_LOAD_KW >= 5.0:
        # TOU schedule
        rate       = NEPRA_TOU_PEAK_RATE if is_peak else NEPRA_TOU_OFFPEAK_RATE
        slab_label = "TOU"
        schedule   = "tou"
    else:
        # Slab schedule — rate determined by monthly consumption
        rate, slab_label = get_nepra_slab_rate(monthly_kwh, settings.TARIFF_PROTECTED)
        schedule = "slab"

    # ── Estimated hourly cost at current power ────────────────────────────────
    hourly_cost: Optional[float] = None
    cost_str = ""
    if power_w is not None and power_w > 0:
        hourly_cost = round(power_w / 1000.0 * rate, 2)
        cost_str = (
            f"At current load ({power_w:.0f} W), "
            f"estimated cost: {cur} {hourly_cost:.2f}/hr."
        )

    return {
        "pkt_hour":       pkt_hour,
        "is_peak":        is_peak,
        "period":         period,
        "rate":           rate,
        "slab_label":     slab_label,
        "schedule":       schedule,
        "monthly_kwh":    round(monthly_kwh, 3),
        "window_str":     window_str,
        "hourly_cost_rs": hourly_cost,
        "cost_str":       cost_str,
        "currency":       cur,
    }


def _build_suggestion(
    reading: FirebaseReading,
    anomaly_result,
    theft,
    now: datetime,
    monthly_kwh: float = 0.0,
) -> dict:
    """
    Build the Firebase suggestion payload for a device.

    The node is always written (overwritten), so the mobile app sees
    the latest state on every poll cycle.

    Structure
    ---------
    {
        "theft_score":          float,
        "anomaly_score":        float,
        "status":               str,         # normal|anomaly|suspicious|possible_theft
        "alert_level":          str,         # info|warning|critical
        "suggestion":           str,
        "action_items":         list[str],
        "contributing_factors": list[str],
        "reason":               str,
        "tariff":               dict,        # Pakistan TOU tariff context
        "updated_at":           int
    }
    """
    status = theft.status
    score  = theft.score

    # ── Pakistan tariff context (NEPRA) ──────────────────────────────────────
    tariff = _get_tariff_context(reading.power, now.hour, monthly_kwh)

    # ── Alert level ───────────────────────────────────────────────────────────
    if status == "possible_theft":
        alert_level = "critical"
    elif status in ("suspicious", "anomaly"):
        alert_level = "warning"
    else:
        alert_level = "info"

    # ── Tariff action item (appended to every status) ─────────────────────────
    cur = tariff["currency"]
    rate = tariff["rate"]
    period = tariff["period"]
    window_str = tariff["window_str"]

    slab_label = tariff["slab_label"]
    schedule   = tariff["schedule"]

    # Slab context line (only for slab schedule)
    if schedule == "slab":
        slab_info = (
            f"Monthly consumption so far: {tariff['monthly_kwh']:.1f} kWh "
            f"→ Slab: {slab_label} @ {cur} {rate:.2f}/kWh."
        )
    else:
        slab_info = f"TOU Schedule (sanctioned load ≥ 5 kW)."

    if tariff["is_peak"]:
        tariff_action = (
            f"Tariff: {period} ({window_str} PKT) — {cur} {rate:.2f}/kWh. "
            f"{slab_info} "
            "Avoid running heavy appliances (AC, washing machine, water heater) during peak hours."
        )
    else:
        tariff_action = (
            f"Tariff: {period} ({window_str} PKT) — {cur} {rate:.2f}/kWh. "
            f"{slab_info} "
            "Good time to run high-consumption appliances and save on your bill."
        )

    if tariff["cost_str"]:
        tariff_action += f" {tariff['cost_str']}"

    # ── Suggestion text & action items ────────────────────────────────────────
    if status == "possible_theft":
        suggestion = (
            "High electricity theft risk detected on your meter. "
            "Unusual patterns strongly suggest unauthorized energy use or meter tampering. "
            "Immediate action is recommended."
        )
        action_items = [
            "Contact your electricity provider and report a suspected tampering incident.",
            "Do not attempt to inspect the meter yourself — call a licensed technician.",
            "Check that no unauthorized connections are attached to your supply line.",
            "Review recent high-consumption appliances for unexpected operation.",
            f"Current theft risk score: {score:.0%}.",
        ]

        # Peak-hour theft is a stronger indicator
        if tariff["is_peak"]:
            action_items.insert(
                0,
                f"Suspicious usage detected during PEAK tariff hours ({window_str} PKT) "
                f"at {cur} {rate:.0f}/kWh — unauthorized consumption during expensive hours is a common theft pattern.",
            )

    elif status == "suspicious":
        suggestion = (
            "Suspicious electricity usage pattern detected. "
            "Your meter readings deviate from your normal consumption profile. "
            "Please review your recent appliance usage."
        )
        action_items = [
            "Check for appliances running unintentionally (heaters, ACs, pumps).",
            "Verify that all high-power devices are switched off when not in use.",
            "If suspicious activity persists, contact your electricity provider.",
            f"Current theft risk score: {score:.0%}.",
        ]

    elif status == "anomaly":
        triggered_rules = anomaly_result.rule.reasons
        if triggered_rules:
            rule_summary = "; ".join(triggered_rules[:2])
            suggestion = (
                f"An electrical anomaly was detected: {rule_summary}. "
                "This may indicate a transient fault, a large appliance switching on, "
                "or an equipment issue. Monitor your system closely."
            )
        else:
            suggestion = (
                "An unusual electrical pattern was detected by the AI model. "
                "This may be a transient event. Monitor your system and check "
                "that all appliances are operating normally."
            )
        action_items = [
            "Inspect high-power appliances (e.g., AC units, water heaters, motors) for faults.",
            "Check for loose wiring or overloaded circuits.",
            "If readings normalise within the next few minutes, no further action is needed.",
            "Persistent anomalies should be reported to a qualified electrician.",
            f"Current anomaly score: {anomaly_result.combined_score:.0%}.",
        ]

    else:  # normal
        if tariff["is_peak"]:
            suggestion = (
                "Your electricity usage is normal. "
                f"You are currently in PEAK tariff hours ({window_str} PKT) "
                f"at {cur} {rate:.0f}/kWh — consider shifting heavy loads to off-peak (5 AM – 5 PM) to save money."
            )
        else:
            suggestion = (
                "Your electricity usage is normal. "
                f"You are in OFF-PEAK tariff hours ({window_str} PKT) "
                f"at {cur} {rate:.0f}/kWh — a great time to run high-consumption appliances."
            )
        action_items = [
            "Continue monitoring your energy usage.",
            "Use the load breakdown to identify high-consumption appliances.",
            f"Current risk score: {score:.0%} — all clear.",
        ]

    # ── Append tariff line to action items (all statuses) ─────────────────────
    action_items.append(tariff_action)

    # ── Peak-hour high-consumption theft warning ──────────────────────────────
    if (
        status in ("suspicious", "possible_theft")
        and reading.power is not None
        and reading.power > 200
        and tariff["is_peak"]
    ):
        pkt_h = tariff["pkt_hour"]
        action_items.insert(
            0,
            f"High power usage ({reading.power:.0f} W) during PEAK hours "
            f"({pkt_h:02d}:00 PKT) — this is a common electricity theft indicator.",
        )

    return {
        "theft_score":          round(score, 4),
        "anomaly_score":        round(anomaly_result.combined_score, 4),
        "status":               status,
        "alert_level":          alert_level,
        "suggestion":           suggestion,
        "action_items":         action_items,
        "contributing_factors": theft.contributing_factors,
        "reason":               theft.reason,
        "tariff": {
            "period":                   tariff["period"],
            "rate_per_kwh":             tariff["rate"],
            "currency":                 tariff["currency"],
            "window":                   f"{window_str} PKT",
            "is_peak":                  tariff["is_peak"],
            "pkt_hour":                 tariff["pkt_hour"],
            "schedule":                 tariff["schedule"],       # "slab" | "tou"
            "slab_label":               tariff["slab_label"],     # e.g. "101–200 units"
            "monthly_kwh":              tariff["monthly_kwh"],    # kWh used this month
            "estimated_hourly_cost_rs": tariff["hourly_cost_rs"],
        },
        "updated_at": int(now.timestamp()),
    }


def _get_baseline(db: Session, device_id: str, hour: int):
    """Fetch the Baseline row for this device/hour, or None."""
    from sqlalchemy import select
    from app.models.orm import Baseline

    stmt = select(Baseline).where(
        Baseline.device_id == device_id,
        Baseline.hour_of_day == hour,
    )
    return db.execute(stmt).scalar_one_or_none()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-training
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_auto_train() -> None:
    """
    Trigger Isolation Forest training in a background thread when:
      - The model is not yet loaded (no saved .joblib file), AND
      - At least MIN_TRAINING_SAMPLES raw readings exist in the DB, AND
      - No training thread is already running.

    This runs at the end of every ingestion cycle so the model activates
    automatically once enough live data has been collected — no manual
    API call required.
    """
    global _auto_training_running

    from app.ml.anomaly import isolation_forest_model

    # Skip if model already loaded or a training run is in progress
    if isolation_forest_model.is_loaded or _auto_training_running:
        return

    # Count available raw readings
    from sqlalchemy import func, select
    with get_db() as db:
        count: int = db.execute(select(func.count(RawReading.id))).scalar() or 0

    if count < settings.MIN_TRAINING_SAMPLES:
        logger.debug(
            "Auto-train: %d/%d samples collected. Waiting for more data.",
            count, settings.MIN_TRAINING_SAMPLES,
        )
        return

    # Enough data — launch training in the background
    import threading

    def _run() -> None:
        global _auto_training_running
        try:
            logger.info(
                "Auto-training Isolation Forest (%d raw readings available) …", count
            )
            from scripts.train import train_from_db
            meta = train_from_db()
            logger.info(
                "Auto-training complete. samples=%s trained_at=%s",
                meta.get("training_samples"),
                meta.get("trained_at"),
            )
        except Exception as exc:
            logger.error("Auto-training failed: %s", exc, exc_info=True)
        finally:
            _auto_training_running = False

    _auto_training_running = True
    thread = threading.Thread(target=_run, name="auto-train", daemon=True)
    thread.start()
