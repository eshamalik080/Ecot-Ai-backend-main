"""
Tests for app/ml/anomaly.py – rule-based checks and combined scoring.
"""

from __future__ import annotations

from app.ml.anomaly import AnomalyResult, MLInferenceResult, RuleCheckResult, run_rule_checks
from app.schemas.readings import FirebaseReading, FeatureVector


def _make_reading(**kwargs) -> FirebaseReading:
    defaults = dict(
        device_id="dev-1",
        email="user@example.com",
        voltage=220.0,
        current=5.0,
        power=1000.0,
        kwh=10.0,
    )
    defaults.update(kwargs)
    return FirebaseReading(**defaults)


def _make_features(**kwargs) -> FeatureVector:
    defaults = dict(
        device_id="dev-1",
        email="user@example.com",
        rolling_avg_power_5m=1000.0,
        rolling_avg_current_5m=5.0,
        spike_count_15m=0,
        delta_power=0.0,
        voltage_deviation=0.0,
    )
    defaults.update(kwargs)
    return FeatureVector(**defaults)


# ── Voltage ───────────────────────────────────────────────────────────────────

def test_rule_voltage_too_low() -> None:
    reading = _make_reading(voltage=150.0)
    result = run_rule_checks(reading, _make_features())
    assert result.triggered
    assert any("too low" in r for r in result.reasons)


def test_rule_voltage_too_high() -> None:
    reading = _make_reading(voltage=280.0)
    result = run_rule_checks(reading, _make_features())
    assert result.triggered
    assert any("too high" in r for r in result.reasons)


def test_rule_voltage_normal() -> None:
    reading = _make_reading(voltage=220.0)
    result = run_rule_checks(reading, _make_features())
    assert not any("Voltage" in r for r in result.reasons)


# ── Current ───────────────────────────────────────────────────────────────────

def test_rule_current_exceeded() -> None:
    reading = _make_reading(current=35.0)
    result = run_rule_checks(reading, _make_features(rolling_avg_current_5m=10.0))
    assert result.triggered
    assert any("Current exceeded" in r for r in result.reasons)


# ── Power ─────────────────────────────────────────────────────────────────────

def test_rule_power_exceeded() -> None:
    reading = _make_reading(power=6000.0)
    result = run_rule_checks(reading, _make_features(rolling_avg_power_5m=1000.0))
    assert result.triggered
    assert any("Power exceeded" in r for r in result.reasons)


def test_rule_power_spike() -> None:
    # power = 4x rolling avg → spike (threshold 3x)
    reading = _make_reading(power=4000.0)
    result = run_rule_checks(reading, _make_features(rolling_avg_power_5m=1000.0))
    assert result.triggered
    assert any("spike" in r.lower() for r in result.reasons)


def test_rule_power_no_spike_when_avg_is_zero() -> None:
    """Division-by-zero guard: no spike rule fires when avg is 0."""
    reading = _make_reading(power=9999.0)
    result = run_rule_checks(reading, _make_features(rolling_avg_power_5m=0.0))
    # Only power-exceeded rule may fire, not spike
    assert not any("spike" in r.lower() for r in result.reasons)


# ── Repeated spikes ───────────────────────────────────────────────────────────

def test_rule_repeated_spikes() -> None:
    result = run_rule_checks(_make_reading(), _make_features(spike_count_15m=5))
    assert result.triggered
    assert any("Repeated spikes" in r for r in result.reasons)


# ── Sudden power jump ─────────────────────────────────────────────────────────

def test_rule_sudden_power_jump() -> None:
    result = run_rule_checks(_make_reading(), _make_features(delta_power=1500.0))
    assert result.triggered
    assert any("power jump" in r.lower() for r in result.reasons)


# ── Voltage deviation ─────────────────────────────────────────────────────────

def test_rule_voltage_deviation() -> None:
    result = run_rule_checks(_make_reading(), _make_features(voltage_deviation=25.0))
    assert result.triggered
    assert any("Voltage deviation" in r for r in result.reasons)


# ── All-clear ─────────────────────────────────────────────────────────────────

def test_rule_all_clear_normal_reading() -> None:
    reading = _make_reading(voltage=220.0, current=5.0, power=1000.0)
    features = _make_features(
        rolling_avg_power_5m=1000.0,
        rolling_avg_current_5m=5.0,
        spike_count_15m=0,
        delta_power=0.0,
        voltage_deviation=0.0,
    )
    result = run_rule_checks(reading, features)
    assert not result.triggered
    assert result.reasons == []


# ── AnomalyResult.combined_score ─────────────────────────────────────────────

def test_combined_score_no_model_no_rule() -> None:
    ar = AnomalyResult(
        rule=RuleCheckResult(triggered=False, reasons=[]),
        ml=MLInferenceResult(model_used=False),
    )
    assert ar.combined_score == 0.0


def test_combined_score_no_model_rule_triggered() -> None:
    ar = AnomalyResult(
        rule=RuleCheckResult(triggered=True, reasons=["test"]),
        ml=MLInferenceResult(model_used=False),
    )
    assert ar.combined_score == 1.0


def test_combined_score_blend_with_ml() -> None:
    ar = AnomalyResult(
        rule=RuleCheckResult(triggered=True, reasons=["test"]),
        ml=MLInferenceResult(score=0.8, is_anomaly=True, model_used=True),
    )
    expected = round(0.7 * 0.8 + 0.3 * 1.0, 4)
    assert ar.combined_score == expected


def test_combined_score_ml_only_no_rule() -> None:
    ar = AnomalyResult(
        rule=RuleCheckResult(triggered=False, reasons=[]),
        ml=MLInferenceResult(score=0.6, is_anomaly=True, model_used=True),
    )
    expected = round(0.7 * 0.6 + 0.3 * 0.0, 4)
    assert ar.combined_score == expected
