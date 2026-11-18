"""Viterbi trellis decoder for HMM route reconstruction."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.models import (
    DecodedTrip,
    Match,
    Observation,
    RoadArc,
    RoadNode,
    RoutePart,
    ScoredCandidate,
    Trip,
)
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph
from pathsnap.decoder.viterbi.probabilities import (
    compute_log_emission,
    compute_log_transition,
    haversine_distance_m,
)


def decode_trip(
    trip: Union[Trip, Dict[str, Any]],
    scored_candidates: List[Union[ScoredCandidate, Dict[str, Any]]],
    config: Optional[DecoderConfig] = None,
) -> DecodedTrip:
    """Reconstruct a coherent road network route from noisy observations and candidate scores.

    Treats candidate directed arc IDs at each observation as hidden states in a Hidden
    Markov Model and decodes the most likely sequence using log-domain Viterbi. Handles
    GPS time gaps, missing candidate sets, unreachable network transitions, and recovers
    intermediate road arcs connecting observation matches.

    Args:
        trip: Trip object or dict containing trip_id, observations, nodes, and arcs.
        scored_candidates: List of ScoredCandidate objects or dicts from the ML scorer.
        config: Optional DecoderConfig with tuning thresholds and hyperparameters.

    Returns:
        DecodedTrip conforming to the shared decoder-to-API contract, containing
        per-observation matches and contiguous route_parts.
    """
    if config is None:
        config = DecoderConfig()

    trip_obj = trip if isinstance(trip, Trip) else Trip.from_dict(trip)

    # Standardize scored candidates
    cand_objs: List[ScoredCandidate] = []
    for sc in scored_candidates:
        cand_objs.append(sc if isinstance(sc, ScoredCandidate) else ScoredCandidate.from_dict(sc))

    # Build directed road network graph
    graph = RoadNetworkGraph(
        nodes=trip_obj.nodes,
        arcs=trip_obj.arcs,
        max_search_distance_m=config.max_network_search_m,
        same_arc_reverse_tolerance_m=config.same_arc_reverse_tolerance_m,
    )

    # Group candidate records by obs_idx, sorted by arc_id
    cands_by_obs: Dict[int, List[ScoredCandidate]] = {}
    for c in cand_objs:
        if c.obs_idx not in cands_by_obs:
            cands_by_obs[c.obs_idx] = []
        cands_by_obs[c.obs_idx].append(c)

    for obs_idx in cands_by_obs:
        cands_by_obs[obs_idx].sort(key=lambda x: x.arc_id)

    # Sort observations in source chronological order
    observations = sorted(trip_obj.observations, key=lambda o: o.obs_idx)
    num_obs = len(observations)

    if num_obs == 0:
        return DecodedTrip(trip_id=trip_obj.trip_id, matches=[], route_parts=[])

    # Observation index to Observation lookup
    obs_lookup: Dict[int, Observation] = {o.obs_idx: o for o in observations}

    matches: List[Match] = []
    route_parts: List[RoutePart] = []

    # Segment tracking for Viterbi decoding
    # We break the sequence into independent Viterbi segments when gaps, empty candidates,
    # or invalid transitions occur.
    seg_obs_indices: List[int] = []
    seg_break_before: Optional[str] = None
    prev_obs: Optional[Observation] = None

    def finalize_segment(
        obs_idx_list: List[int],
        break_before: Optional[str],
    ) -> None:
        """Run Viterbi on a contiguous segment of observations and append results."""
        if not obs_idx_list:
            return

        seg_matches, seg_route_parts = _decode_contiguous_segment(
            obs_indices=obs_idx_list,
            obs_lookup=obs_lookup,
            cands_by_obs=cands_by_obs,
            graph=graph,
            config=config,
            break_before=break_before,
        )
        matches.extend(seg_matches)
        route_parts.extend(seg_route_parts)

    for obs in observations:
        idx = obs.obs_idx
        cands = cands_by_obs.get(idx, [])

        # Check for GPS gap
        is_gps_gap = False
        if prev_obs is not None:
            dt = obs.time_s - prev_obs.time_s
            if dt > config.gps_gap_threshold_s:
                is_gps_gap = True

        if is_gps_gap:
            # Finalize current segment
            finalize_segment(seg_obs_indices, seg_break_before)
            seg_obs_indices = []
            seg_break_before = "gps_gap"

        # Check if candidate set is empty
        if not cands:
            # Finalize current segment before the empty candidate observation
            finalize_segment(seg_obs_indices, seg_break_before)
            seg_obs_indices = []
            # Record unmatched observation
            matches.append(
                Match(obs_idx=idx, arc_id=None, unmatched_reason="no_candidates")
            )
            # Next segment will be marked with break_before="no_candidates"
            seg_break_before = "no_candidates"
            prev_obs = obs
            continue

        # Non-empty candidate set: add to current segment
        seg_obs_indices.append(idx)
        prev_obs = obs

    # Finalize any trailing segment
    finalize_segment(seg_obs_indices, seg_break_before)

    # Sort matches by obs_idx to guarantee exact input source ordering
    matches.sort(key=lambda m: m.obs_idx)

    # Set the very first route part's break_before to None according to the spec:
    # "The first part has break_before: null. Later parts give the reason for the preceding break."
    if route_parts:
        route_parts[0].break_before = None

    return DecodedTrip(
        trip_id=trip_obj.trip_id,
        matches=matches,
        route_parts=route_parts,
    )


def _decode_contiguous_segment(
    obs_indices: List[int],
    obs_lookup: Dict[int, Observation],
    cands_by_obs: Dict[int, List[ScoredCandidate]],
    graph: RoadNetworkGraph,
    config: DecoderConfig,
    break_before: Optional[str],
) -> Tuple[List[Match], List[RoutePart]]:
    """Decode a contiguous sequence of observations that all possess candidate road arcs.

    If an invalid transition disconnects all states at step t, this function splits
    the segment at step t, marking the subsequent segment with break_before='invalid_transition'.
    """
    matches: List[Match] = []
    route_parts: List[RoutePart] = []

    if not obs_indices:
        return matches, route_parts

    # Sub-segment boundaries: start a subsegment
    sub_start = 0
    current_break_before = break_before

    while sub_start < len(obs_indices):
        # We find the longest subsegment starting at sub_start where transitions remain possible
        # Viterbi state: step -> dict of {cand_idx: (viterbi_log_prob, backptr_cand_idx)}
        trellis: List[Dict[int, Tuple[float, Optional[int]]]] = []

        # Step 0 of subsegment
        t0_obs_idx = obs_indices[sub_start]
        t0_cands = cands_by_obs[t0_obs_idx]

        # Prior distribution: equal priors + emission
        prior_step: Dict[int, Tuple[float, Optional[int]]] = {}
        for c_idx, cand in enumerate(t0_cands):
            log_em = compute_log_emission(cand.emission_prob, floor=config.emission_floor)
            prior_step[c_idx] = (log_em, None)

        trellis.append(prior_step)

        sub_end = sub_start
        for step_idx in range(sub_start + 1, len(obs_indices)):
            prev_obs_idx = obs_indices[step_idx - 1]
            curr_obs_idx = obs_indices[step_idx]

            prev_obs = obs_lookup[prev_obs_idx]
            curr_obs = obs_lookup[curr_obs_idx]

            prev_cands = cands_by_obs[prev_obs_idx]
            curr_cands = cands_by_obs[curr_obs_idx]

            d_gps = haversine_distance_m(
                prev_obs.lon, prev_obs.lat, curr_obs.lon, curr_obs.lat
            )
            delta_t = curr_obs.time_s - prev_obs.time_s

            curr_step: Dict[int, Tuple[float, Optional[int]]] = {}
            any_valid_transition = False

            for c_curr_idx, c_curr in enumerate(curr_cands):
                log_em = compute_log_emission(
                    c_curr.emission_prob, floor=config.emission_floor
                )

                best_prob = float("-inf")
                best_prev_idx: Optional[int] = None
                best_prev_arc_id: Optional[int] = None

                for c_prev_idx, (prev_prob, _) in trellis[-1].items():
                    if prev_prob == float("-inf"):
                        continue

                    c_prev = prev_cands[c_prev_idx]
                    d_net, _ = graph.get_candidate_distance_and_path(
                        c_prev.arc_id,
                        c_prev.along_arc_fraction,
                        c_curr.arc_id,
                        c_curr.along_arc_fraction,
                    )

                    log_trans = compute_log_transition(
                        d_net=d_net,
                        d_gps=d_gps,
                        delta_t=delta_t,
                        config=config,
                    )

                    if log_trans == float("-inf"):
                        continue

                    candidate_total = prev_prob + log_trans + log_em

                    # Deterministic tie-breaking favoring smaller arc_id of predecessor
                    is_better = False
                    if candidate_total > best_prob + 1e-9:
                        is_better = True
                    elif abs(candidate_total - best_prob) <= 1e-9:
                        # Tie-break by smaller predecessor arc_id
                        if best_prev_arc_id is None or c_prev.arc_id < best_prev_arc_id:
                            is_better = True

                    if is_better:
                        best_prob = candidate_total
                        best_prev_idx = c_prev_idx
                        best_prev_arc_id = c_prev.arc_id

                curr_step[c_curr_idx] = (best_prob, best_prev_idx)
                if best_prob > float("-inf"):
                    any_valid_transition = True

            if not any_valid_transition:
                # Transition failure between step_idx - 1 and step_idx
                # Current subsegment ends at step_idx - 1
                break

            trellis.append(curr_step)
            sub_end = step_idx

        # Backtrack optimal path for subsegment from sub_start to sub_end
        last_step = trellis[-1]
        best_final_prob = float("-inf")
        best_final_cand_idx: Optional[int] = None
        best_final_arc_id: Optional[int] = None

        final_obs_cands = cands_by_obs[obs_indices[sub_end]]
        for c_idx, (prob, _) in last_step.items():
            cand = final_obs_cands[c_idx]
            is_better = False
            if prob > best_final_prob + 1e-9:
                is_better = True
            elif abs(prob - best_final_prob) <= 1e-9:
                if best_final_arc_id is None or cand.arc_id < best_final_arc_id:
                    is_better = True

            if is_better:
                best_final_prob = prob
                best_final_cand_idx = c_idx
                best_final_arc_id = cand.arc_id

        # In case all states at end are -inf (e.g. single isolated point with -inf emission),
        # pick highest emission candidate with tie break
        if best_final_cand_idx is None:
            best_final_cand_idx = min(
                range(len(final_obs_cands)),
                key=lambda i: (-final_obs_cands[i].emission_prob, final_obs_cands[i].arc_id),
            )

        # Backtrack backpointers
        chosen_cand_indices: List[int] = [best_final_cand_idx]
        for t in range(len(trellis) - 1, 0, -1):
            curr_cand_idx = chosen_cand_indices[-1]
            prev_cand_idx = trellis[t][curr_cand_idx][1]
            if prev_cand_idx is None:
                # Fallback if backpointer is somehow missing
                prev_obs_cands = cands_by_obs[obs_indices[sub_start + t - 1]]
                prev_cand_idx = min(
                    range(len(prev_obs_cands)),
                    key=lambda i: (-prev_obs_cands[i].emission_prob, prev_obs_cands[i].arc_id),
                )
            chosen_cand_indices.append(prev_cand_idx)

        chosen_cand_indices.reverse()

        # Build matches and route part for this subsegment
        sub_cands: List[ScoredCandidate] = []
        for i, obs_idx in enumerate(obs_indices[sub_start : sub_end + 1]):
            cand = cands_by_obs[obs_idx][chosen_cand_indices[i]]
            sub_cands.append(cand)
            matches.append(Match(obs_idx=obs_idx, arc_id=cand.arc_id, unmatched_reason=None,
                                 along_arc_fraction=cand.along_arc_fraction))

        # Reconstruct route part arc IDs including intermediate arcs
        route_arc_ids: List[int] = []
        for i, cand in enumerate(sub_cands):
            if i == 0:
                route_arc_ids.append(cand.arc_id)
            else:
                prev_cand = sub_cands[i - 1]
                _, intermediate = graph.get_candidate_distance_and_path(
                    prev_cand.arc_id,
                    prev_cand.along_arc_fraction,
                    cand.arc_id,
                    cand.along_arc_fraction,
                )
                for inter_arc in intermediate:
                    route_arc_ids.append(inter_arc)
                if cand.arc_id != route_arc_ids[-1]:
                    route_arc_ids.append(cand.arc_id)

        route_parts.append(
            RoutePart(
                start_obs_idx=obs_indices[sub_start],
                end_obs_idx=obs_indices[sub_end],
                arc_ids=route_arc_ids,
                break_before=current_break_before,
            )
        )

        # Advance to next subsegment
        sub_start = sub_end + 1
        current_break_before = "invalid_transition"

    return matches, route_parts
