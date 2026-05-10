"""
app/ml/theft.py
───────────────
Theft risk scoring.

IMPORTANT: This module does NOT claim theft with certainty.
It computes a probabilistic risk score and assigns one of four status labels:
  normal       – nothing suspicious
  anomaly      – ML or rule-based anomaly, but not yet suspicious
  suspicious   – elevated risk; warrants attention
  possible_theft – high-confidence risk signal

The score is a weighted combination of multiple signals, each grounded in
known electricity theft indicators from utility literature:
  - tampered/bypassed meters show abnormal power patterns
  - illegal connections appear as spikes during off-hours
  - rapid energy growth with small apparent load is a red flag
  - unusual usage at night (when legitimate demand is low)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.ml.anomaly import AnomalyResult
from app.ml.features import FeatureVector
from app.models.orm import Baseline
from app.schemas.readings import FirebaseReading

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

# Status thresholds (tunable via config in a future version)
_THRESHOLD_SUSPICIOUS = 0.45
_THRESHOLD_POSSIBLE_THEFT = 0.70


@dataclass
class TheftScore:
    score: float                      # 0.0–1.0
    status: str                       # normal | anomaly | suspicious | possible_theft
    reason: str
    explanation: str
    contributing_factors: list[str]   # detailed breakdown for UI/logging


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

def compute_theft_score(
    reading: FirebaseReading,
    features: FeatureVector,
    anomaly_result: AnomalyResult,
    baseline: Optional[Baseline] = None,
) -> TheftScore:
    """
    Compute a theft risk score from 0.0 to 1.0.

    Signals used
    ------------
    1. Anomaly score (ML + rule combined)           weight 0.35
    2. Night-time high usage                         weight 0.15
    3. Repeated spikes (spike frequency)             weight 0.15
    4. Deviation from learned baseline               weight 0.25
    5. KWh growth anomaly                            weight 0.10

    Parameters
    ----------
    reading        : Most recent FirebaseReading.
    features       : Computed feature vector.
    anomaly_result : Output of detect_anomaly().
    baseline       : Learned Baseline for this device/hour, or None if unavailable.
    """
    factors: list[str] = []
    weighted_score = 0.0

    # ── Signal 1: Anomaly score (35%) ────────────────────────────────────────
    anomaly_contribution = anomaly_result.combined_score * 0.35
    weighted_score += anomaly_contribution
    if anomaly_result.combined_score > 0.5:
        factors.append(
            f"High anomaly score: {anomaly_result.combined_score:.2f} "
            f"(ML: {'yes' if anomaly_result.ml.model_used else 'no model'}, "
            f"Rules: {'triggered' if anomaly_result.rule.triggered else 'clear'})"
        )

    # ── Signal 2: Night-time high usage (15%) ────────────────────────────────
    night_contribution = 0.0
    if features.night_flag and reading.power is not None and reading.power > 200:
        # Suspicious if power is substantial during 22:00–06:00
        night_ratio = min(reading.power / 2000.0, 1.0)  # normalise against 2kW
        night_contribution = night_ratio * 0.15
        weighted_score += night_contribution
        if night_ratio > 0.3:
            factors.append(
                f"High night-time consumption: {reading.power:.1f}W at hour {features.hour_of_day}"
            )

    # ── Signal 3: Spike frequency (15%) ──────────────────────────────────────
    spike_contribution = 0.0
    if features.spike_count_15m is not None and features.spike_count_15m > 0:
        spike_contribution = min(features.spike_count_15m / 5.0, 1.0) * 0.15
        weighted_score += spike_contribution
        if features.spike_count_15m >= 2:
            factors.append(
                f"Repeated power spikes: {features.spike_count_15m} in last 15 min"
            )

    # ── Signal 4: Baseline deviation (25%) ───────────────────────────────────
    baseline_contribution = 0.0
    if baseline is not None and reading.power is not None:
        baseline_contribution = _baseline_deviation_signal(reading, baseline) * 0.25
        weighted_score += baseline_contribution
        if baseline_contribution > 0.1:
            factors.append(
                f"Deviation from learned baseline: "
                f"current power {reading.power:.1f}W vs "
                f"baseline avg {baseline.avg_power:.1f}W ± {baseline.std_power:.1f}W "
                f"for hour {baseline.hour_of_day}"
            )

    # ── Signal 5: KWh growth anomaly (10%) ───────────────────────────────────
    kwh_contribution = 0.0
    if reading.power is not None and reading.kwh is not None:
        kwh_contribution = _kwh_growth_signal(reading) * 0.10
        weighted_score += kwh_contribution
        if kwh_contribution > 0.05:
            factors.append(
                f"Anomalous kWh growth pattern detected (kwh={reading.kwh:.3f})"
            )

    # ── Final score ──────────────────────────────────────────────────────────
    final_score = round(min(max(weighted_score, 0.0), 1.0), 4)

    status, reason, explanation = _classify(
        final_score, factors, anomaly_result, features
    )

    return TheftScore(
        score=final_score,
        status=status,
        reason=reason,
        explanation=explanation,
        contributing_factors=factors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signal sub-functions
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_deviation_signal(reading: FirebaseReading, baseline: Baseline) -> float:
    """
    How far is current power from the learned baseline for this hour of day?
    Returns a 0–1 signal where 1 = extremely far from baseline.
    """
    if baseline.avg_power is None or baseline.std_power is None:
        return 0.0
    if baseline.std_power == 0:
        baseline_std = baseline.avg_power * 0.1 or 50.0  # assume 10% tolerance
    else:
        baseline_std = baseline.std_power

    sigma = baseline_std or 50.0
    deviation_sigmas = abs((reading.power or 0.0) - baseline.avg_power) / sigma

    # Map sigma distance to [0,1]: 0σ → 0.0, ≥3σ → 1.0
    return min(deviation_sigmas / 3.0, 1.0)


def _kwh_growth_signal(reading: FirebaseReading) -> float:
    """
    Simple heuristic: if kwh is very high compared to expected cumulative
    energy from current power level, flag it.
    This is approximate – a full implementation would compare against
    hourly_summaries. For now, detect very high kwh values.
    """
    if reading.kwh is None:
        return 0.0
    # kWh > 50 is unusually high for a single residential device in short observation
    # This is a placeholder; refine once time-series of kWh growth is available.
    return min(reading.kwh / 100.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify(
    score: float,
    factors: list[str],
    anomaly_result: AnomalyResult,
    features: FeatureVector,
) -> tuple[str, str, str]:
    """Map score to (status, reason, explanation) strings."""

    if score >= _THRESHOLD_POSSIBLE_THEFT:
        status = "possible_theft"
        reason = "Multiple high-confidence theft indicators detected"
        explanation = (
            f"Theft risk score {score:.2f} exceeds threshold {_THRESHOLD_POSSIBLE_THEFT}. "
            f"Signals: {'; '.join(factors[:3]) if factors else 'composite score elevated'}. "
            "This is a risk indicator only – not a confirmed theft event. "
            "Manual review is recommended."
        )

    elif score >= _THRESHOLD_SUSPICIOUS:
        status = "suspicious"
        reason = "Elevated risk indicators – anomalous usage pattern detected"
        explanation = (
            f"Theft risk score {score:.2f} is elevated. "
            f"Signals: {'; '.join(factors[:2]) if factors else 'composite score elevated'}. "
            "Monitor this device for further anomalies."
        )

    elif anomaly_result.rule.triggered or anomaly_result.ml.is_anomaly:
        status = "anomaly"
        reason = (
            anomaly_result.rule.reasons[0]
            if anomaly_result.rule.reasons
            else "ML model flagged an unusual pattern"
        )
        explanation = (
            f"Anomaly detected (score {score:.2f}) but below theft risk threshold. "
            f"Rule triggers: {', '.join(anomaly_result.rule.reasons) or 'none'}. "
            f"ML anomaly: {'yes' if anomaly_result.ml.is_anomaly else 'no'}."
        )

    else:
        status = "normal"
        reason = "No anomalies detected"
        explanation = (
            f"All signals within normal range. "
            f"Risk score: {score:.2f}. "
            f"Night flag: {features.night_flag}, "
            f"Spike count (15m): {features.spike_count_15m or 0}."
        )

    return status, reason, explanation
