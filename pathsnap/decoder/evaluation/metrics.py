"""Evaluation metrics for observation-level and sequence-level route performance."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.models import (
    ClassifierMatch,
    Match,
    Observation,
    Reference,
    RoutePart,
    ScoredCandidate,
)
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph


def compute_top1_accuracy(
    matches: List[Union[Match, ClassifierMatch, Dict[str, Any]]],
    references: List[Union[Reference, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute observation-level Top-1 accuracy over aligned reference observations.

    Rules:
        - Only observations with alignment_status == 'aligned' contribute to the denominator.
        - Missing, null, or incorrect predictions count as incorrect.
        - Reports counts of aligned, ambiguous, and unaligned observations.

    Returns:
        Dict with keys:
            accuracy: float | None (None if aligned count is 0)
            correct: int
            aligned: int
            ambiguous: int
            unaligned: int
    """
    ref_objs = [
        r if isinstance(r, Reference) else Reference.from_dict(r) for r in references
    ]
    ref_by_obs = {r.obs_idx: r for r in ref_objs}

    pred_by_obs: Dict[int, Optional[int]] = {}
    for m in matches:
        if isinstance(m, (Match, ClassifierMatch)):
            pred_by_obs[m.obs_idx] = m.arc_id
        elif isinstance(m, dict):
            pred_by_obs[int(m["obs_idx"])] = (
                int(m["arc_id"]) if m.get("arc_id") is not None else None
            )

    aligned_count = 0
    ambiguous_count = 0
    unaligned_count = 0
    correct_count = 0

    for r in ref_objs:
        if r.alignment_status == "aligned":
            aligned_count += 1
            pred_arc = pred_by_obs.get(r.obs_idx)
            if pred_arc is not None and r.ref_arc_id is not None and pred_arc == r.ref_arc_id:
                correct_count += 1
        elif r.alignment_status == "ambiguous":
            ambiguous_count += 1
        elif r.alignment_status == "unaligned":
            unaligned_count += 1

    accuracy = (correct_count / aligned_count) if aligned_count > 0 else None

    return {
        "accuracy": accuracy,
        "correct": correct_count,
        "aligned": aligned_count,
        "ambiguous": ambiguous_count,
        "unaligned": unaligned_count,
    }


def compute_candidate_miss_rate(
    candidates: List[Union[ScoredCandidate, Dict[str, Any]]],
    references: List[Union[Reference, Dict[str, Any]]],
) -> Optional[float]:
    """Compute candidate miss rate against aligned reference observations.

    Defined as the number of aligned observations whose reference arc is missing
    from their candidate set divided by the total number of aligned observations.

    Returns:
        Float miss rate in [0, 1], or None if aligned count is 0.
    """
    ref_objs = [
        r if isinstance(r, Reference) else Reference.from_dict(r) for r in references
    ]
    cand_arc_ids_by_obs: Dict[int, Set[int]] = {}
    for c in candidates:
        obs_idx = int(c.obs_idx if hasattr(c, "obs_idx") else c["obs_idx"])
        arc_id = int(c.arc_id if hasattr(c, "arc_id") else c["arc_id"])
        if obs_idx not in cand_arc_ids_by_obs:
            cand_arc_ids_by_obs[obs_idx] = set()
        cand_arc_ids_by_obs[obs_idx].add(arc_id)

    aligned_count = 0
    miss_count = 0

    for r in ref_objs:
        if r.alignment_status == "aligned":
            aligned_count += 1
            cand_set = cand_arc_ids_by_obs.get(r.obs_idx, set())
            if r.ref_arc_id is not None and r.ref_arc_id not in cand_set:
                miss_count += 1

    if aligned_count == 0:
        return None
    return miss_count / aligned_count


def compute_macro_f1(
    matches: List[Union[Match, ClassifierMatch, Dict[str, Any]]],
    references: List[Union[Reference, Dict[str, Any]]],
    trip_id: str,
) -> Optional[float]:
    """Compute Macro F1 across trip-qualified road arc IDs for aligned observations.

    Formula:
        Classes are all unique true reference arc IDs in aligned observations.
        Unmatched / null predictions are counted as false negatives for true classes,
        but the unmatched sentinel itself is not included in the class average.
    """
    ref_objs = [
        r if isinstance(r, Reference) else Reference.from_dict(r)
        for r in references
        if (isinstance(r, Reference) and r.trip_id == trip_id)
        or (isinstance(r, dict) and str(r.get("trip_id")) == trip_id)
    ]

    pred_by_obs: Dict[int, Optional[int]] = {}
    for m in matches:
        if isinstance(m, (Match, ClassifierMatch)):
            pred_by_obs[m.obs_idx] = m.arc_id
        elif isinstance(m, dict):
            pred_by_obs[int(m["obs_idx"])] = (
                int(m["arc_id"]) if m.get("arc_id") is not None else None
            )

    aligned_refs = [r for r in ref_objs if r.alignment_status == "aligned" and r.ref_arc_id is not None]
    if not aligned_refs:
        return None

    true_classes = {f"{trip_id}_{r.ref_arc_id}" for r in aligned_refs}
    if not true_classes:
        return None

    f1_scores: List[float] = []

    for c in sorted(true_classes):
        # Extract the arc ID for class c
        target_arc_id = int(c.split("_")[-1])

        tp = 0
        fp = 0
        fn = 0

        for r in aligned_refs:
            pred_arc = pred_by_obs.get(r.obs_idx)
            is_true = (r.ref_arc_id == target_arc_id)
            is_pred = (pred_arc == target_arc_id)

            if is_true and is_pred:
                tp += 1
            elif not is_true and is_pred:
                fp += 1
            elif is_true and not is_pred:
                fn += 1

        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else None


def compute_route_continuity(
    matches: List[Union[Match, ClassifierMatch, Dict[str, Any]]],
    observations: List[Union[Observation, Dict[str, Any]]],
    graph: RoadNetworkGraph,
    config: Optional[DecoderConfig] = None,
) -> Dict[str, Any]:
    """Measure sequential route continuity across consecutive matched observations.

    Applies reachability, time, and gap rules:
        - Intervals where an observation is unmatched or null are recorded as unresolved.
        - Intervals where delta_t > gps_gap_threshold_s are recorded as excluded gaps.
        - Eligible intervals are tested for road network reachability within max speed.

    Returns:
        Dict with keys:
            continuity_rate: float | None
            eligible_transitions: int
            continuous_transitions: int
            breaks: int
            excluded_gaps: int
            unresolved_points: int
    """
    if config is None:
        config = DecoderConfig()

    obs_objs = sorted(
        [o if isinstance(o, Observation) else Observation.from_dict(o) for o in observations],
        key=lambda x: x.obs_idx,
    )
    obs_map = {o.obs_idx: o for o in obs_objs}

    pred_by_obs: Dict[int, Optional[int]] = {}
    fraction_by_obs: Dict[int, Optional[float]] = {}
    for m in matches:
        if isinstance(m, (Match, ClassifierMatch)):
            pred_by_obs[m.obs_idx] = m.arc_id
            fraction_by_obs[m.obs_idx] = m.along_arc_fraction
        elif isinstance(m, dict):
            pred_by_obs[int(m["obs_idx"])] = (
                int(m["arc_id"]) if m.get("arc_id") is not None else None
            )
            fraction_by_obs[int(m["obs_idx"])] = m.get("along_arc_fraction")

    eligible_count = 0
    continuous_count = 0
    breaks_count = 0
    excluded_gaps_count = 0
    unresolved_count = 0

    for i in range(1, len(obs_objs)):
        prev_obs = obs_objs[i - 1]
        curr_obs = obs_objs[i]

        prev_arc = pred_by_obs.get(prev_obs.obs_idx)
        curr_arc = pred_by_obs.get(curr_obs.obs_idx)

        # Check for unresolved match
        if prev_arc is None or curr_arc is None:
            unresolved_count += 1
            continue

        # Check for GPS gap
        dt = curr_obs.time_s - prev_obs.time_s
        if dt > config.gps_gap_threshold_s:
            excluded_gaps_count += 1
            continue

        # Eligible transition interval
        eligible_count += 1

        # Decoded matches carry their projected fractions; legacy/classifier matches fall back to arc midpoints.
        prev_fraction = fraction_by_obs.get(prev_obs.obs_idx)
        curr_fraction = fraction_by_obs.get(curr_obs.obs_idx)
        d_net, _ = graph.get_candidate_distance_and_path(
            from_arc_id=prev_arc,
            from_frac=prev_fraction if prev_fraction is not None else 0.5,
            to_arc_id=curr_arc,
            to_frac=curr_fraction if curr_fraction is not None else 0.5,
        )

        is_reachable = False
        if d_net is not None:
            if dt > 0:
                speed = d_net / dt
                if speed <= config.max_speed_mps:
                    is_reachable = True
            else:
                is_reachable = True

        if is_reachable:
            continuous_count += 1
        else:
            breaks_count += 1

    continuity_rate = (
        (continuous_count / eligible_count) if eligible_count > 0 else None
    )

    return {
        "continuity_rate": continuity_rate,
        "eligible_transitions": eligible_count,
        "continuous_transitions": continuous_count,
        "breaks": breaks_count,
        "excluded_gaps": excluded_gaps_count,
        "unresolved_points": unresolved_count,
    }


def compute_sequence_edit_distance(
    route_parts: List[Union[RoutePart, Dict[str, Any]]],
    reference_route: List[int],
) -> Dict[str, Any]:
    """Compute Levenshtein edit distance between the decoded arc sequence and reference route.

    Args:
        route_parts: Decoded route parts containing ordered lists of arc IDs.
        reference_route: Chronological sequence of ground-truth reference arc IDs.

    Returns:
        Dict with keys:
            raw_edit_distance: int
            normalized_edit_distance: float
            predicted_length: int
            reference_length: int
    """
    predicted_arcs: List[int] = []
    for rp in route_parts:
        arcs = rp.arc_ids if isinstance(rp, RoutePart) else rp.get("arc_ids", [])
        predicted_arcs.extend(arcs)

    m = len(predicted_arcs)
    n = len(reference_route)

    # Dynamic programming matrix for Levenshtein distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted_arcs[i - 1] == reference_route[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    raw_dist = dp[m][n]
    max_len = max(m, n, 1)
    norm_dist = raw_dist / max_len

    return {
        "raw_edit_distance": raw_dist,
        "normalized_edit_distance": norm_dist,
        "predicted_length": m,
        "reference_length": n,
    }
