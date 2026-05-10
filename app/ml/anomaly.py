"""
app/ml/anomaly.py
─────────────────
Two-layer anomaly detection:

  Layer 1 – Rule-based  : Deterministic checks on current sensor values.
                          Fast, interpretable, no training required.
  Layer 2 – ML-based    : Isolation Forest model for pattern-level anomalies.
                          Requires training data but catches subtle deviations.

Both layers return structured results that are merged by the scoring pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from app.config import settings
from app.ml.features import FEATURE_COLUMNS, FeatureVector, feature_vector_to_numpy
from app.schemas.readings import FirebaseReading

logger = logging.getLogger(__name__)

# ── Model file path ────────────────────────────────────────────────────────────
_MODEL_FILENAME = "isolation_forest.joblib"
_META_FILENAME = "model_meta.json"


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleCheckResult:
    triggered: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class MLInferenceResult:
    score: Optional[float] = None      # Normalised 0–1; higher = more anomalous
    is_anomaly: bool = False
    model_used: bool = False           # False if model not loaded yet


@dataclass
class AnomalyResult:
    """Combined result from both detection layers."""
    rule: RuleCheckResult
    ml: MLInferenceResult

    @property
    def combined_score(self) -> float:
        """
        Best-effort anomaly score (0–1).
        Uses ML score when available; falls back to rule-based binary signal.
        """
        if self.ml.model_used and self.ml.score is not None:
            # Blend: 70% ML score + 30% rule signal
            rule_signal = 1.0 if self.rule.triggered else 0.0
            return round(0.7 * self.ml.score + 0.3 * rule_signal, 4)
        # No ML model: use rule signal only
        return 1.0 if self.rule.triggered else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest model manager (singleton)
# ─────────────────────────────────────────────────────────────────────────────

class IsolationForestModel:
    """
    Wraps an sklearn IsolationForest with load/save/inference helpers.
    Thread-safe for reading (the model object is immutable after training).
    """

    def __init__(self) -> None:
        self._model: Optional[IsolationForest] = None
        self._meta: dict = {}
        self._model_path: Optional[str] = None
        self._loaded = False

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, model_dir: Optional[str] = None) -> bool:
        """
        Try to load a previously saved model.
        Returns True if successful, False if no model file exists yet.
        """
        dir_path = model_dir or settings.MODEL_DIR
        model_path = os.path.join(dir_path, _MODEL_FILENAME)
        meta_path = os.path.join(dir_path, _META_FILENAME)

        if not os.path.exists(model_path):
            logger.info(
                "No saved model found at %s. "
                "Will fall back to rule-based detection until training runs.",
                model_path,
            )
            return False

        try:
            self._model = joblib.load(model_path)
            self._model_path = model_path
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self._meta = json.load(f)
            self._loaded = True
            logger.info(
                "Isolation Forest loaded from %s  (samples=%s, trained_at=%s)",
                model_path,
                self._meta.get("training_samples", "unknown"),
                self._meta.get("trained_at", "unknown"),
            )
            return True
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", model_path, exc)
            return False

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, model: IsolationForest, meta: dict, model_dir: Optional[str] = None) -> None:
        """Persist model and metadata to disk."""
        dir_path = model_dir or settings.MODEL_DIR
        os.makedirs(dir_path, exist_ok=True)

        model_path = os.path.join(dir_path, _MODEL_FILENAME)
        meta_path = os.path.join(dir_path, _META_FILENAME)

        joblib.dump(model, model_path)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._model = model
        self._meta = meta
        self._model_path = model_path
        self._loaded = True
        logger.info("Model saved to %s", model_path)

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(self, feature_vector: FeatureVector) -> MLInferenceResult:
        """
        Run inference on a FeatureVector.

        Returns an MLInferenceResult with:
          score      : 0–1 normalised anomaly score (higher = more anomalous)
          is_anomaly : True if score exceeds ANOMALY_THRESHOLD
          model_used : False when no model is loaded
        """
        if not self._loaded or self._model is None:
            return MLInferenceResult(model_used=False)

        arr = feature_vector_to_numpy(feature_vector)
        if arr is None:
            logger.debug("Not enough features for ML scoring – skipping.")
            return MLInferenceResult(model_used=False)

        try:
            # score_samples returns negative values; more negative = more anomalous
            raw_score = float(self._model.score_samples(arr)[0])
            # Normalise to 0–1: typical range is roughly -0.7 to 0.1
            # We map: score <= -0.7 → 1.0 (very anomalous), score >= 0.1 → 0.0
            normalised = _normalise_score(raw_score, low=-0.7, high=0.1)
            return MLInferenceResult(
                score=round(normalised, 4),
                is_anomaly=normalised >= settings.ANOMALY_THRESHOLD,
                model_used=True,
            )
        except Exception as exc:
            logger.error("ML inference error: %s", exc)
            return MLInferenceResult(model_used=False)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        contamination: Optional[float] = None,
        model_dir: Optional[str] = None,
    ) -> dict:
        """
        Train a new Isolation Forest on the given feature matrix X.
        Saves the model automatically.

        Returns a metrics dict.
        """
        if X.shape[0] < settings.MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Not enough training samples: {X.shape[0]} < {settings.MIN_TRAINING_SAMPLES}"
            )

        cont = contamination or settings.ANOMALY_CONTAMINATION
        model = IsolationForest(
            n_estimators=200,
            contamination=cont,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)

        # Score training set to compute fraction flagged
        preds = model.predict(X)  # +1 = inlier, -1 = outlier
        n_outliers = int((preds == -1).sum())

        meta = {
            "training_samples": int(X.shape[0]),
            "features": FEATURE_COLUMNS,
            "contamination": cont,
            "n_outliers_found": n_outliers,
            "outlier_fraction": round(n_outliers / X.shape[0], 4),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        self.save(model, meta, model_dir=model_dir)
        logger.info(
            "Training complete. Samples=%d, outliers found=%d (%.1f%%)",
            X.shape[0],
            n_outliers,
            100 * meta["outlier_fraction"],
        )
        return meta

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path


# Module-level singleton used by all services
isolation_forest_model = IsolationForestModel()


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based detector
# ─────────────────────────────────────────────────────────────────────────────

def run_rule_checks(
    reading: FirebaseReading,
    features: FeatureVector,
) -> RuleCheckResult:
    """
    Deterministic rule-based anomaly checks.
    Returns a RuleCheckResult with triggered=True and reasons if any rule fires.
    """
    reasons: list[str] = []

    # ── 1. Voltage out of safe range ──────────────────────────────────────────
    if reading.voltage is not None:
        if reading.voltage < settings.VOLTAGE_MIN:
            reasons.append(
                f"Voltage too low: {reading.voltage:.1f}V < {settings.VOLTAGE_MIN}V"
            )
        elif reading.voltage > settings.VOLTAGE_MAX:
            reasons.append(
                f"Voltage too high: {reading.voltage:.1f}V > {settings.VOLTAGE_MAX}V"
            )

    # ── 2. Current exceeded safety limit ──────────────────────────────────────
    if reading.current is not None and reading.current > settings.CURRENT_MAX:
        reasons.append(
            f"Current exceeded limit: {reading.current:.2f}A > {settings.CURRENT_MAX}A"
        )

    # ── 3. Power exceeded safety limit ────────────────────────────────────────
    if reading.power is not None and reading.power > settings.POWER_MAX:
        reasons.append(
            f"Power exceeded limit: {reading.power:.1f}W > {settings.POWER_MAX}W"
        )

    # ── 4. Power spike vs rolling average ─────────────────────────────────────
    if (
        reading.power is not None
        and features.rolling_avg_power_5m is not None
        and features.rolling_avg_power_5m > 0
    ):
        spike_ratio = reading.power / features.rolling_avg_power_5m
        if spike_ratio > settings.POWER_SPIKE_MULTIPLIER:
            reasons.append(
                f"Power spike: {reading.power:.1f}W is {spike_ratio:.1f}× "
                f"the 5-min rolling avg ({features.rolling_avg_power_5m:.1f}W)"
            )

    # ── 5. Current spike vs rolling average ───────────────────────────────────
    if (
        reading.current is not None
        and features.rolling_avg_current_5m is not None
        and features.rolling_avg_current_5m > 0
    ):
        current_ratio = reading.current / features.rolling_avg_current_5m
        if current_ratio > settings.POWER_SPIKE_MULTIPLIER:
            reasons.append(
                f"Current spike: {reading.current:.2f}A is {current_ratio:.1f}× "
                f"the 5-min rolling avg"
            )

    # ── 6. Repeated spikes in last 15 minutes ────────────────────────────────
    if features.spike_count_15m is not None and features.spike_count_15m >= 3:
        reasons.append(
            f"Repeated spikes: {features.spike_count_15m} spike(s) in last 15 min"
        )

    # ── 7. Abnormal delta (sudden jump) ───────────────────────────────────────
    if features.delta_power is not None and abs(features.delta_power) > 1000:
        reasons.append(
            f"Sudden power jump: Δ{features.delta_power:+.1f}W between readings"
        )

    # ── 8. Voltage deviation from rolling mean ────────────────────────────────
    if features.voltage_deviation is not None and abs(features.voltage_deviation) > 20:
        reasons.append(
            f"Voltage deviation: {features.voltage_deviation:+.1f}V from rolling mean"
        )

    return RuleCheckResult(triggered=bool(reasons), reasons=reasons)


# ─────────────────────────────────────────────────────────────────────────────
# Combined detector entry point
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomaly(
    reading: FirebaseReading,
    features: FeatureVector,
) -> AnomalyResult:
    """
    Run both rule-based and ML anomaly detection.
    Returns an AnomalyResult with all scores and flags populated.
    """
    rule_result = run_rule_checks(reading, features)
    ml_result = isolation_forest_model.score(features)

    return AnomalyResult(rule=rule_result, ml=ml_result)


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_score(raw: float, low: float = -0.7, high: float = 0.1) -> float:
    """
    Map raw Isolation Forest score_samples output to [0, 1].
    raw <= low  → 1.0 (most anomalous)
    raw >= high → 0.0 (most normal)
    """
    if raw <= low:
        return 1.0
    if raw >= high:
        return 0.0
    return (high - raw) / (high - low)
