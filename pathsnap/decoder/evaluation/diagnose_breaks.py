"""Diagnostic script: categorize every disconnected interval across 10 student_test trips.

For each consecutive matched-observation pair that counts as a continuity break:
  1. Retrieve the two arc IDs and their projected fractions.
  2. Query the road network graph for the shortest directed path.
  3. Categorize the failure.
  4. Compute key statistics (elapsed time, GPS distance, implied speed if path found).

Categories:
  A. gps_gap          : delta_t > gps_gap_threshold_s  -> excluded from denominator
  B. no_candidates    : one or both sides has arc_id=None
  C. graph_disconnect : graph returns None (no directed path exists within max_search_m)
  D. speed_violation  : path exists but d_net/delta_t > max_speed_mps
  E. feasible         : d_net not None, speed OK -> should have been continuous (bug?)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from pathsnap.data.candidates import CandidateConfig, generate_candidates
from pathsnap.data.loading import load_trip
from pathsnap.data.reference_alignment import AlignmentConfig, align_references, read_reference_route
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.evaluation.comparator import compare_classifier_vs_viterbi
from pathsnap.decoder.models import Trip
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph
from pathsnap.decoder.viterbi.probabilities import haversine_distance_m
from pathsnap.models.training.scorer import CandidateScorer


def categorize_break(
    prev_arc: int | None,
    curr_arc: int | None,
    prev_frac: float | None,
    curr_frac: float | None,
    dt: float,
    graph: RoadNetworkGraph,
    config: DecoderConfig,
) -> dict[str, Any]:
    if dt > config.gps_gap_threshold_s:
        return {"category": "gps_gap", "d_net": None, "speed": None}
    if prev_arc is None or curr_arc is None:
        return {"category": "no_candidates", "d_net": None, "speed": None}

    pf = prev_frac if prev_frac is not None else 0.5
    cf = curr_frac if curr_frac is not None else 0.5
    d_net, path = graph.get_candidate_distance_and_path(prev_arc, pf, curr_arc, cf)

    if d_net is None:
        return {"category": "graph_disconnect", "d_net": None, "speed": None, "inter_arcs": []}

    speed = (d_net / dt) if dt > 0 else 0.0
    if speed > config.max_speed_mps:
        return {"category": "speed_violation", "d_net": d_net, "speed": speed, "inter_arcs": path}

    return {"category": "feasible", "d_net": d_net, "speed": speed, "inter_arcs": path}


def diagnose_all_trips(
    dataset_dir: str = "Student Folder/student_test",
    model_path: str = "artifacts/models/pathsnap_scorer.joblib",
    out_dir: str = "reports/decoder-comparison",
):
    config = DecoderConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scorer = CandidateScorer.load(model_path)
    for _, step in scorer.pipeline.steps:
        if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
            step._fill_dtype = step._fit_dtype

    cand_config = CandidateConfig()
    align_config = AlignmentConfig()

    trip_dirs = sorted([d for d in Path(dataset_dir).iterdir() if d.is_dir()])

    all_break_rows = []
    summary = {}

    for trip_dir in trip_dirs:
        trip_id = trip_dir.name
        print(f"Diagnosing {trip_id}...", flush=True)

        trip_raw = load_trip(trip_dir)
        cands = generate_candidates(trip_raw, cand_config)
        cands_df = pd.DataFrame([asdict(c) for c in cands])
        obs_df = pd.DataFrame([
            {"trip_id": trip_raw.trip_id, "obs_idx": i, "lon": lon, "lat": lat, "time_s": t}
            for i, (lon, lat, t) in enumerate(trip_raw.track)
        ])
        scored_df, matches_df = scorer.score(cands_df, obs_df)
        ref_route = read_reference_route(trip_raw, trip_dir)
        refs = align_references(trip_raw, ref_route, align_config)

        dec_trip = Trip.from_dict(trip_raw.to_record())
        scored_cands = scored_df.to_dict(orient="records")
        classifier_matches = matches_df.to_dict(orient="records")
        ref_dicts = [asdict(r) for r in refs]

        comp = compare_classifier_vs_viterbi(
            trip=dec_trip,
            scored_candidates=scored_cands,
            classifier_matches=classifier_matches,
            references=ref_dicts,
            reference_route=ref_route,
            config=config,
        )
        decoded_trip = comp["decoded_trip"]

        # Build graph from trip
        graph = RoadNetworkGraph(
            nodes=dec_trip.nodes,
            arcs=dec_trip.arcs,
            max_search_distance_m=config.max_network_search_m,
        )

        # Build lookup tables from decoded matches
        arc_by_obs: dict[int, int | None] = {m.obs_idx: m.arc_id for m in decoded_trip.matches}

        # Build scored candidate fraction lookup: (obs_idx, arc_id) -> along_arc_fraction
        frac_by_key: dict[tuple[int, int], float] = {}
        for sc in scored_cands:
            obs_idx = int(sc["obs_idx"])
            arc_id = sc["arc_id"]
            if arc_id is not None:
                frac_by_key[(obs_idx, int(arc_id))] = float(sc["along_arc_fraction"])

        # Build observation sorted list
        obs_sorted = sorted(dec_trip.observations, key=lambda o: o.obs_idx)
        obs_map = {o.obs_idx: o for o in obs_sorted}

        trip_breaks = {"gps_gap": 0, "no_candidates": 0, "graph_disconnect": 0,
                       "speed_violation": 0, "feasible": 0}

        for i in range(1, len(obs_sorted)):
            prev_obs = obs_sorted[i - 1]
            curr_obs = obs_sorted[i]
            dt = curr_obs.time_s - prev_obs.time_s

            prev_arc = arc_by_obs.get(prev_obs.obs_idx)
            curr_arc = arc_by_obs.get(curr_obs.obs_idx)

            if dt > config.gps_gap_threshold_s:
                trip_breaks["gps_gap"] += 1
                continue

            if prev_arc is None or curr_arc is None:
                trip_breaks["no_candidates"] += 1
                continue

            prev_frac = frac_by_key.get((prev_obs.obs_idx, prev_arc), 0.5)
            curr_frac = frac_by_key.get((curr_obs.obs_idx, curr_arc), 0.5)

            d_gps = haversine_distance_m(
                prev_obs.lon, prev_obs.lat, curr_obs.lon, curr_obs.lat
            )

            info = categorize_break(
                prev_arc, curr_arc, prev_frac, curr_frac, dt, graph, config
            )

            if info["category"] == "feasible":
                # This interval is feasible - it's continuous, NOT a break
                continue

            # It's genuinely a break or a disconnect
            trip_breaks[info["category"]] += 1

            all_break_rows.append({
                "trip_id": trip_id,
                "prev_obs_idx": prev_obs.obs_idx,
                "curr_obs_idx": curr_obs.obs_idx,
                "prev_arc": prev_arc,
                "curr_arc": curr_arc,
                "prev_frac": prev_frac,
                "curr_frac": curr_frac,
                "dt_s": dt,
                "d_gps_m": d_gps,
                "d_net_m": info.get("d_net"),
                "speed_mps": info.get("speed"),
                "inter_arcs": str(info.get("inter_arcs", [])),
                "category": info["category"],
            })

        summary[trip_id] = trip_breaks
        cont = comp["viterbi"]["continuity_rate"]
        print(f"  Breaks: {trip_breaks}  | continuity={cont:.4f}")

    df_breaks = pd.DataFrame(all_break_rows)
    df_breaks.to_csv(out / "break_diagnosis.csv", index=False)

    print("\n=== SUMMARY ===")
    total = {k: 0 for k in ["gps_gap", "no_candidates", "graph_disconnect", "speed_violation", "feasible"]}
    for tid, counts in summary.items():
        for k, v in counts.items():
            total[k] += v
    print(json.dumps(total, indent=2))

    # Speed violation stats
    if not df_breaks.empty:
        sv = df_breaks[df_breaks["category"] == "speed_violation"]
        gd = df_breaks[df_breaks["category"] == "graph_disconnect"]
        print(f"\nSpeed violations: {len(sv)}")
        if len(sv) > 0:
            print(f"  d_net range: {sv['d_net_m'].min():.1f} - {sv['d_net_m'].max():.1f} m")
            print(f"  speed range: {sv['speed_mps'].min():.1f} - {sv['speed_mps'].max():.1f} m/s")
            print(f"  dt range: {sv['dt_s'].min():.1f} - {sv['dt_s'].max():.1f} s")
        print(f"\nGraph disconnects: {len(gd)}")
        if len(gd) > 0:
            print(f"  d_gps range: {gd['d_gps_m'].min():.1f} - {gd['d_gps_m'].max():.1f} m")
            print(f"  dt range: {gd['dt_s'].min():.1f} - {gd['dt_s'].max():.1f} s")

    print(f"\nFull break table: {out / 'break_diagnosis.csv'}")
    return df_breaks, summary, total


if __name__ == "__main__":
    diagnose_all_trips()
