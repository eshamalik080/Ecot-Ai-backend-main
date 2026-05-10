"""
app/firebase_client.py
──────────────────────
Firebase Admin SDK wrapper.

Responsibilities
----------------
1. Initialize the Firebase app once using a service account credentials file.
2. Read all user/device nodes from the Realtime Database.
3. Parse and validate each node into a FirebaseReading object.
4. Write AI results back to each user's node under ai_result/.
5. Handle missing / malformed nodes gracefully without crashing the poll loop.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import firebase_admin
from firebase_admin import credentials, db as firebase_db
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.schemas.readings import FirebaseReading

logger = logging.getLogger(__name__)

# Module-level flag so we only initialise once per process
_firebase_initialised: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_firebase() -> None:
    """
    Initialise Firebase Admin SDK. Safe to call multiple times – skips if
    already initialised. Raises RuntimeError on bad credentials.
    """
    global _firebase_initialised

    if _firebase_initialised:
        return

    if not settings.FIREBASE_DATABASE_URL:
        raise RuntimeError(
            "FIREBASE_DATABASE_URL is not set. Check your .env file."
        )

    try:
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(
            cred,
            {"databaseURL": settings.FIREBASE_DATABASE_URL},
        )
        _firebase_initialised = True
        logger.info(
            "Firebase initialised → database: %s",
            settings.FIREBASE_DATABASE_URL,
        )
    except Exception as exc:
        raise RuntimeError(f"Firebase initialisation failed: {exc}") from exc


def is_firebase_ready() -> bool:
    return _firebase_initialised


# ─────────────────────────────────────────────────────────────────────────────
# Read from Firebase
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=False,
)
def _fetch_root_node() -> Optional[dict[str, Any]]:
    """
    Fetch the entire root node (or a sub-node if FIREBASE_ROOT_NODE is set)
    from Firebase Realtime Database. Returns None on failure.
    """
    root = settings.FIREBASE_ROOT_NODE.strip("/") or None
    ref = firebase_db.reference(root) if root else firebase_db.reference("/")
    return ref.get()


def read_all_devices() -> list[FirebaseReading]:
    """
    Read all user/device nodes from Firebase.

    Expected structure:
        /
        └── <user_key> (e.g. umer23538183@gmail_com)
            ├── device_id: "ECO-00007"
            ├── email: "umer@gmail.com"
            ├── voltage: 221.6
            └── ...

    Returns a list of validated FirebaseReading objects.
    Missing or invalid nodes are logged and skipped.
    """
    try:
        raw_data = _fetch_root_node()
    except RetryError as exc:
        logger.error("Failed to fetch Firebase data after retries: %s", exc)
        return []

    if not raw_data or not isinstance(raw_data, dict):
        logger.debug("Firebase root node is empty or not a dict.")
        return []

    readings: list[FirebaseReading] = []

    for user_key, node in raw_data.items():
        if not isinstance(node, dict):
            # Skip non-dict children (e.g. if someone wrote a scalar at root)
            continue

        # Inject the user_key as a fallback identifier
        node.setdefault("_user_key", user_key)

        reading = _parse_node(user_key, node)
        if reading is not None:
            readings.append(reading)

    logger.debug("Read %d device(s) from Firebase.", len(readings))
    return readings


def _parse_node(user_key: str, node: dict[str, Any]) -> Optional[FirebaseReading]:
    """Parse and validate a single Firebase user node. Returns None on error."""
    try:
        # Normalise: Firebase keys use underscores for dots/@ in email paths
        email = node.get("email") or user_key.replace("_", ".").replace("@.", "@")

        # Skip non-user nodes (e.g. "inventory", "config", metadata buckets).
        # A valid user key must resolve to something that looks like an email.
        if "@" not in email:
            logger.debug("Skipping non-user node '%s' (no email address).", user_key)
            return None

        device_id = node.get("device_id") or email.replace("@", "_").replace(".", "_")

        payload = {
            "device_id": device_id,
            "email": email,
            "voltage": node.get("voltage"),
            "current": node.get("current"),
            "power": node.get("power"),
            "kwh": node.get("kwh"),
            "power_factor": node.get("power_factor"),
            "frequency": node.get("frequency"),
            "load1": node.get("load1"),
            "load2": node.get("load2"),
            "load3": node.get("load3"),
            "overall": node.get("overall"),
            "timestamp": node.get("timestamp"),
            "name": node.get("name"),
            "city": node.get("city"),
            "country": node.get("country"),
        }
        return FirebaseReading(**payload)

    except Exception as exc:
        logger.warning("Skipping malformed Firebase node '%s': %s", user_key, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Write AI results back to Firebase
# ─────────────────────────────────────────────────────────────────────────────

def write_ai_result(user_key: str, result: dict[str, Any]) -> bool:
    """
    Write AI result dict to:
        /<user_key>/ai_result/

    The user_key must match the node key in Firebase (e.g. email with @ → _).

    Returns True on success, False on failure.
    """
    try:
        ref = firebase_db.reference(f"/{user_key}/ai_result")
        ref.set(result)
        logger.debug("AI result written to Firebase for key: %s", user_key)
        return True
    except Exception as exc:
        logger.error(
            "Failed to write AI result to Firebase for key '%s': %s", user_key, exc
        )
        return False


def write_suggestion(user_key: str, payload: dict[str, Any]) -> bool:
    """
    Write (overwrite) the suggestion node for a user to:
        /<user_key>/suggestion/

    This is always a full replace so the mobile app always sees
    the latest state without needing to merge stale fields.

    Returns True on success, False on failure.
    """
    try:
        ref = firebase_db.reference(f"/{user_key}/suggestion")
        ref.set(payload)
        logger.debug("Suggestion written to Firebase for key: %s", user_key)
        return True
    except Exception as exc:
        logger.error(
            "Failed to write suggestion to Firebase for key '%s': %s", user_key, exc
        )
        return False


def write_forecast(user_key: str, payload: dict[str, Any]) -> bool:
    """
    Write (overwrite) the forecast node for a user to:
        /<user_key>/forecast/

    Payload contains the 24-hour energy forecast and tariff optimization
    summary. The mobile app reads this to display upcoming consumption
    and cost predictions.

    Returns True on success, False on failure.
    """
    try:
        ref = firebase_db.reference(f"/{user_key}/forecast")
        ref.set(payload)
        logger.debug("Forecast written to Firebase for key: %s", user_key)
        return True
    except Exception as exc:
        logger.error(
            "Failed to write forecast to Firebase for key '%s': %s", user_key, exc
        )
        return False


def email_to_firebase_key(email: str) -> str:
    """
    Convert an email address to the Firebase-safe key format.
    Firebase does not allow '.', '#', '$', '[', ']' in keys,
    so the Firebase SDK or the app typically encodes them.
    Convention used: replace '@' with '@' kept, '.' with '_dot_' or just '_'.
    We use: replace '.' with '_' and '@' kept as '@'.

    Match this to however your Firebase data was actually written.
    Adjust if your convention differs.
    """
    return email.replace(".", "_")
