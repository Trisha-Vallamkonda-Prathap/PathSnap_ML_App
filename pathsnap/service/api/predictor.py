"""The real PathSnap prediction pipeline, run on Laptop 1.

    GPS + road network
      -> 1. candidate road arcs near each GPS point   (Member 1, pathsnap.data.candidates)
      -> 2. ML score for every candidate              (Member 2, saved LightGBM ranker)
      -> 3. HMM + Viterbi decoding into one route     (Member 3, pathsnap.decoder)

The predictor receives ONLY the inference inputs (the Trip from loader.py).
It never sees the ground-truth route.
"""
import dataclasses
from functools import lru_cache
from typing import Dict, Optional, Tuple

import pandas as pd

from pathsnap.data.candidates import CandidateConfig, generate_candidates
from pathsnap.data.loading import Trip
from pathsnap.decoder import DecodedTrip, DecoderConfig, decode_trip
from pathsnap.models.training.scorer import CandidateScorer

from .config import MODEL_PATH

# Same settings the team used for development (Member 1 and Member 3 defaults).
CANDIDATE_CONFIG = CandidateConfig()   # 60 m search radius, up to 24 candidates per point
DECODER_CONFIG = DecoderConfig()       # 30 s GPS-gap threshold, 45 m/s max speed, ...


class ModelNotAvailable(Exception):
    """The saved model file is missing or is only a Git LFS pointer."""


@lru_cache(maxsize=1)
def load_scorer() -> CandidateScorer:
    """Load Member 2's saved model once and keep it in memory for every request."""
    if not MODEL_PATH.is_file():
        raise ModelNotAvailable(f"Model file not found: {MODEL_PATH} (set PATHSNAP_MODEL_PATH).")
    if MODEL_PATH.stat().st_size < 1024:  # a Git LFS pointer is ~130 bytes, the real model ~2 MB
        raise ModelNotAvailable(f"{MODEL_PATH.name} is a Git LFS pointer. Run `git lfs pull` first.")
    return CandidateScorer.load(MODEL_PATH)


def predict_trip(trip: Trip) -> Tuple[DecodedTrip, Dict[int, Optional[int]]]:
    """Return (decoded, classifier_only).

    decoded: Member 3's DecodedTrip, one match per GPS point plus the connected route parts.
    classifier_only: obs_idx -> the model's Top-1 arc before HMM decoding (None if no candidates),
    kept so the server can report "classifier only vs classifier + HMM" for the same trip.
    """
    # 1. Candidate road arcs for every GPS point (Member 1).
    candidates = generate_candidates(trip, CANDIDATE_CONFIG)
    trip_record = trip.to_record()  # the shared v1 trip format the decoder expects

    # 2. Score the candidates with the trained model (Member 2).
    #    Output: each candidate gets an emission probability (they sum to 1 per GPS point).
    candidate_table = pd.DataFrame([dataclasses.asdict(c) for c in candidates])
    observation_table = pd.DataFrame(trip_record["observations"]).assign(trip_id=trip.trip_id)
    classifier_only: Dict[int, Optional[int]] = {i: None for i in range(len(trip.track))}
    if candidate_table.empty:
        scored_rows = []  # no road near any point: the decoder returns an all-unmatched trip
    else:
        scored, greedy_matches = load_scorer().score(candidate_table, observation_table)
        scored_rows = scored.to_dict("records")
        for row in greedy_matches.itertuples():
            classifier_only[int(row.obs_idx)] = None if pd.isna(row.arc_id) else int(row.arc_id)

    # 3. HMM + Viterbi decoding over the whole trip (Member 3).
    return decode_trip(trip_record, scored_rows, DECODER_CONFIG), classifier_only
