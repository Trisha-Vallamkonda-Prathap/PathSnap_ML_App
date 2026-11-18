"""Trip registry: which trips the API serves, and where their files live."""
from pathlib import Path
from typing import List

from .config import TRIPS_DIR


class TripNotFound(Exception):
    """Raised when a trip ID is not in the configured trip folder."""


def list_trip_ids() -> List[str]:
    """Return the trip IDs (sub-folder names) in TRIPS_DIR, sorted.

    IDs stay strings so leading zeros like "00000007" are kept.
    """
    if not TRIPS_DIR.is_dir():
        return []
    return sorted(p.name for p in TRIPS_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))


def trip_folder(trip_id: str) -> Path:
    """Return the folder for a trip. Only IDs from the registry are accepted,
    which also blocks tricks like trip_id = "../train/00000001"."""
    if trip_id not in list_trip_ids():
        raise TripNotFound(trip_id)
    return TRIPS_DIR / trip_id
