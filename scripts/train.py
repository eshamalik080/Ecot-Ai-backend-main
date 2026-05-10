"""
Training entry point for the Isolation Forest anomaly model.

This module is used both by the API (`POST /api/v1/model/retrain`) and as a
standalone script:

    python -m scripts.train
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from app.db import get_db
from app.ml.anomaly import isolation_forest_model
from app.ml.features import build_training_matrix
from app.models.orm import RawReading
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def train_from_db() -> dict:
    """
    Build a training matrix from all stored raw readings and train the model.

    Returns training metadata from `IsolationForestModel.train()`.
    Raises `ValueError` when there is not enough data to train.
    """
    with get_db() as db:
        readings = list(
            db.execute(
                select(RawReading).order_by(RawReading.device_id, RawReading.reading_ts)
            ).scalars()
        )

    matrix = build_training_matrix(readings)
    if matrix.empty:
        raise ValueError("No usable feature rows could be built from raw_readings.")

    return isolation_forest_model.train(matrix.to_numpy())


def main() -> None:
    setup_logging()
    meta = train_from_db()
    logger.info(
        "Training finished successfully. samples=%s trained_at=%s",
        meta.get("training_samples"),
        meta.get("trained_at"),
    )


if __name__ == "__main__":
    main()
