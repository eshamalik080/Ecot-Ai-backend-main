"""
Tests for app/ml/theft.py – theft scoring and status classification.
"""

from __future__ import annotations

from app.ml.anomaly import AnomalyResult, MLInferenceResult, RuleCheckResult
from app.ml.theft import compute_theft_score, _THRESHOLD_SUSPICIOUS, _THRESHOLD_POSSIBLE_THEFT
from app.schemas.readings import FirebaseReading, FeatureVector


def _clean_anomaly() -> AnomalyResult:
    return AnomalyResult(
        rule=RuleCheckResult(triggered=False, reasons=[]),
        ml=MLInferenceResult(model_used=False),
    )


def _high_anomaly() -> AnomalyResult:
    return AnomalyResult(
        rule=RuleCheckResult(triggered=True, reasons=["Power spike: …"]),
        ml=MLInferenceResult(score=0.9, is_anomaly=True, model_used=True),
    )


def _make_reading(**kwargs) -> FirebaseReading:
    defaults = dict(device_id="dev-1", email="u@e.com", voltage=220.0, current=5.0, power=500.0, kwh=5.0)
    defaults.update(kwargs)
    return FirebaseReading(**defaults)


def _make_features(**kwargs) -> FeatureVector:
    defaults = dict(device_id="dev-1", email="u@e.com", night_flag=False, spike_count_15m=0, hour_of_day=10)
    defaults.update(kwargs)
    return FeatureVector(**defaults)


# ── Normal scenario ───────────────────────────────────────────────────────────

def test_normal_reading_yields_normal_status() -> None:
    result = compute_theft_score(
        reading=_make_reading(power=300.0, kwh=1.0),
        features=_make_features(),
        anomaly_result=_clean_anomaly(),
    )
    assert result.status == "normal"
    assert result.score < _THRESHOLD_SUSPICIOUS


# ── Anomaly status ────────────────────────────────────────────────────────────

def test_rule_trigger_without_high_score_yields_anomaly() -> None:
    anomaly = AnomalyResult(
        rule=RuleCheckResult(triggered=True, reasons=["Voltage too low: 150V"]),
        ml=MLInferenceResult(model_used=False),
    )
    # combined_score = 1.0 but theft weighting may stay below suspicious
    result = compute_theft_score(
        reading=_make_reading(power=100.0, kwh=0.1),
        features=_make_features(spike_count_15m=0),
        anomaly_result=anomaly,
    )
    # At minimum the score should reflect the anomaly signal (35% * 1.0 = 0.35)
    assert result.score >= 0.30
    assert result.status in ("anomaly", "suspicious", "possible_theft")


# ── Night-time signal ─────────────────────────────────────────────────────────

def test_night_high_power_adds_to_score() -> None:
    day_result = compute_theft_score(
        reading=_make_reading(power=1000.0, kwh=2.0),
        features=_make_features(night_flag=False),
        anomaly_result=_clean_anomaly(),
    )
    night_result = compute_theft_score(
        reading=_make_reading(power=1000.0, kwh=2.0),
        features=_make_features(night_flag=True, hour_of_day=2),
        anomaly_result=_clean_anomaly(),
    )
    assert night_result.score > day_result.score


# ── Spike frequency signal ────────────────────────────────────────────────────

def test_high_spike_count_increases_score() -> None:
    low = compute_theft_score(
        reading=_make_reading(),
        features=_make_features(spike_count_15m=0),
        anomaly_result=_clean_anomaly(),
    )
    high = compute_theft_score(
        reading=_make_reading(),
        features=_make_features(spike_count_15m=5),
        anomaly_result=_clean_anomaly(),
    )
    assert high.score > low.score


# ── Possible theft classification ─────────────────────────────────────────────

def test_possible_theft_status_with_all_signals() -> None:
    result = compute_theft_score(
        reading=_make_reading(power=1500.0, kwh=80.0),
        features=_make_features(night_flag=True, spike_count_15m=5, hour_of_day=2),
        anomaly_result=_high_anomaly(),
    )
    assert result.status == "possible_theft"
    assert result.score >= _THRESHOLD_POSSIBLE_THEFT


# ── Baseline deviation signal ─────────────────────────────────────────────────

def test_baseline_deviation_increases_score(db_session) -> None:
    from app.models.orm import Baseline
    from datetime import datetime, timezone

    baseline = Baseline(
        device_id="dev-1",
        hour_of_day=10,
        avg_power=300.0,
        std_power=20.0,
        avg_current=3.0,
        std_current=0.5,
        avg_kwh_growth=0.1,
        avg_spike_freq=0.0,
        avg_anomaly_freq=0.0,
        sample_count=5,
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(baseline)
    db_session.commit()

    # Power 900W vs baseline 300W ± 20W → extreme deviation
    result = compute_theft_score(
        reading=_make_reading(power=900.0, kwh=1.0),
        features=_make_features(),
        anomaly_result=_clean_anomaly(),
        baseline=baseline,
    )
    no_baseline_result = compute_theft_score(
        reading=_make_reading(power=900.0, kwh=1.0),
        features=_make_features(),
        anomaly_result=_clean_anomaly(),
    )
    assert result.score > no_baseline_result.score


# ── Score clamp ───────────────────────────────────────────────────────────────

def test_score_is_clamped_to_0_1() -> None:
    result = compute_theft_score(
        reading=_make_reading(power=5000.0, kwh=500.0),
        features=_make_features(night_flag=True, spike_count_15m=10, hour_of_day=3),
        anomaly_result=_high_anomaly(),
    )
    assert 0.0 <= result.score <= 1.0


# ── Output fields ─────────────────────────────────────────────────────────────

def test_theft_score_has_all_required_fields() -> None:
    result = compute_theft_score(
        reading=_make_reading(),
        features=_make_features(),
        anomaly_result=_clean_anomaly(),
    )
    assert result.status in ("normal", "anomaly", "suspicious", "possible_theft")
    assert isinstance(result.reason, str) and result.reason
    assert isinstance(result.explanation, str) and result.explanation
    assert isinstance(result.contributing_factors, list)
