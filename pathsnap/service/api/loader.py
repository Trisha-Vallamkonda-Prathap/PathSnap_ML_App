"""Inference loader: reads ONLY the prediction inputs (.track, .nodes, .arcs).

Uses Member 1's loader (pathsnap.data.loading.load_trip), which also projects
every point to metres. The ground-truth .route file is NOT read here; it is read
only by ground_truth.py, after the prediction is made.
"""
from pathlib import Path

from pathsnap.data.loading import Trip, load_trip


class TripDataError(Exception):
    """A trip file is missing or cannot be parsed."""


def load_trip_inputs(folder: Path) -> Trip:
    """Load one trip folder with Member 1's loader and turn its errors into TripDataError."""
    try:
        return load_trip(folder)
    except (OSError, ValueError) as err:  # missing file, bad row, invalid node ID, ...
        raise TripDataError(f"Trip {folder.name}: {err}") from err
