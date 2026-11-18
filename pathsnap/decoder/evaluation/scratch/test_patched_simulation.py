"""Test the effect of allowing small reverse tolerance on same arc in RoadNetworkGraph."""
from pathlib import Path
from dataclasses import asdict
import pandas as pd

from pathsnap.data.loading import load_trip
from pathsnap.data.candidates import CandidateConfig, generate_candidates
from pathsnap.data.reference_alignment import AlignmentConfig, align_references, read_reference_route
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.evaluation.comparator import compare_classifier_vs_viterbi
from pathsnap.decoder.models import Trip
from pathsnap.models.training.scorer import CandidateScorer
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph

scorer = CandidateScorer.load("artifacts/models/pathsnap_scorer.joblib")
for _, step in scorer.pipeline.steps:
    if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
        step._fill_dtype = step._fit_dtype

cand_config = CandidateConfig()
align_config = AlignmentConfig()

for trip_name in ["00000007", "00000077", "00000040"]:
    trip_dir = Path(f"Student Folder/student_test/{trip_name}")
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

    # Baseline run (current config)
    comp_base = compare_classifier_vs_viterbi(
        trip=dec_trip,
        scored_candidates=scored_cands,
        classifier_matches=classifier_matches,
        references=ref_dicts,
        reference_route=ref_route,
        config=DecoderConfig(),
    )

    # Monkey-patch get_candidate_distance_and_path to support 15m reverse tolerance on same arc
    orig_fn = RoadNetworkGraph.get_candidate_distance_and_path

    def patched_fn(self, from_arc_id, from_frac, to_arc_id, to_frac):
        if from_arc_id not in self.arcs or to_arc_id not in self.arcs:
            return None, []
        if from_arc_id == to_arc_id:
            from_arc = self.arcs[from_arc_id]
            frac_diff = to_frac - from_frac
            # Forward movement
            if frac_diff >= -1e-6:
                return max(0.0, frac_diff) * from_arc.length_m, []
            # Backward movement within GPS noise tolerance (e.g. 15m)
            bwd_m = abs(frac_diff) * from_arc.length_m
            if bwd_m <= 15.0:
                return 0.0, []  # Stationary / GPS jitter on same segment
            # Beyond tolerance: genuine backward violation -> cycle or None
            cycle_dist, intermediate_arcs = self._find_shortest_node_path(
                from_arc.to_node_id, from_arc.from_node_id
            )
            if cycle_dist is None:
                return None, []
            total_dist = (1.0 - from_frac) * from_arc.length_m + cycle_dist + to_frac * from_arc.length_m
            if total_dist > self.max_search_distance_m:
                return None, []
            return total_dist, intermediate_arcs

        return orig_fn(self, from_arc_id, from_frac, to_arc_id, to_frac)

    RoadNetworkGraph.get_candidate_distance_and_path = patched_fn

    comp_patched = compare_classifier_vs_viterbi(
        trip=dec_trip,
        scored_candidates=scored_cands,
        classifier_matches=classifier_matches,
        references=ref_dicts,
        reference_route=ref_route,
        config=DecoderConfig(),
    )

    # Restore
    RoadNetworkGraph.get_candidate_distance_and_path = orig_fn

    print(f"\n=== Trip {trip_name} ===")
    print(f"  CURRENT:  acc={comp_base['viterbi']['top1_accuracy']:.4f}, cont={comp_base['viterbi']['continuity_rate']:.4f}")
    print(f"  PATCHED:  acc={comp_patched['viterbi']['top1_accuracy']:.4f}, cont={comp_patched['viterbi']['continuity_rate']:.4f}")
    print(f"  BASELINE: acc={comp_base['baseline']['top1_accuracy']:.4f}, cont={comp_base['baseline']['continuity_rate']:.4f}")
