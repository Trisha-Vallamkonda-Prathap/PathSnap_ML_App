"""SERVER-SIDE SCORING ONLY. The only module that reads the ground-truth .route file.

Rules:
  * Call score_trip() only AFTER predict_trip() has produced its prediction.
  * Per-point reference arcs stay inside Member 1's scoring code. Only aggregate
    counts come back, so ground truth never reaches the predictor or the UI.

Uses Member 1's alignment and scoring (pathsnap.data.reference_alignment.scoring):
each GPS point is aligned to the reference route as "aligned", "ambiguous" or
"unaligned"; accuracy = correct / aligned, and an unmatched point counts as wrong.
"""
from pathlib import Path
from typing import Dict, Optional

from pathsnap.data.loading import Trip
from pathsnap.data.reference_alignment.scoring import score_trip as member1_score_trip


def score_trip(trip: Trip, folder: Path, predictions: Dict[int, Optional[int]]) -> Optional[Dict]:
    """Aggregate scores for one prediction (obs_idx -> arc_id or None),
    or None when the trip has no usable reference."""
    if not (folder / f"{trip.trip_id}.route").is_file():
        return None  # e.g. a trip without a .route file: nothing to score against
    result = member1_score_trip(trip, folder, predictions)
    if not result["aligned"]:
        return None
    # Only aggregate numbers leave this function.
    return {
        "accuracy": result["accuracy"],
        "correct": result["correct"],
        "aligned": result["aligned"],
        "ambiguous": result["ambiguous"],
        "unaligned": result["unaligned"],
        "macro_f1": result["macro_f1"],
    }
