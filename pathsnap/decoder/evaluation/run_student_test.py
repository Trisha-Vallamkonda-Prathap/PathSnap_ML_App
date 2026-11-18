"""Run route decoding and evaluation across all 10 student_test trips.

Connects Member 1's real candidate generation and road network graphs,
Member 2's final saved LightGBM candidate scorer, and Member 3's HMM/Viterbi decoder.
Generates per-trip and aggregate comparison reports in reports/decoder-comparison/.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pathsnap.data.candidates import CandidateConfig, generate_candidates
from pathsnap.data.loading import load_trip
from pathsnap.data.reference_alignment import AlignmentConfig, align_references, read_reference_route
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.evaluation.comparator import compare_classifier_vs_viterbi
from pathsnap.decoder.models import Trip
from pathsnap.models.training.scorer import CandidateScorer


def run_all_student_test(
    dataset_dir: str | Path = "Student Folder/student_test",
    model_artifact: str | Path = "artifacts/models/pathsnap_scorer.joblib",
    out_dir: str | Path = "reports/decoder-comparison",
    config: DecoderConfig | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = DecoderConfig()

    dataset_path = Path(dataset_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load saved model
    print(f"Loading candidate scorer from {model_artifact}...")
    scorer = CandidateScorer.load(model_artifact)

    # Patch scikit-learn 1.8 compatibility if needed
    for name, step in scorer.pipeline.steps:
        if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
            step._fill_dtype = step._fit_dtype

    cand_config = CandidateConfig()
    align_config = AlignmentConfig()

    trip_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    print(f"Found {len(trip_dirs)} trips in {dataset_path}: {[d.name for d in trip_dirs]}")

    trip_results: List[Dict[str, Any]] = []

    total_t0 = time.time()

    for trip_dir in trip_dirs:
        trip_id = trip_dir.name
        t_start = time.time()
        print(f"\nProcessing trip {trip_id}...", flush=True)

        # Member 1 load and candidate generation
        trip_raw = load_trip(trip_dir)
        cands = generate_candidates(trip_raw, cand_config)
        cands_df = pd.DataFrame([asdict(c) for c in cands])

        obs_df = pd.DataFrame([
            {"trip_id": trip_raw.trip_id, "obs_idx": i, "lon": lon, "lat": lat, "time_s": time_s}
            for i, (lon, lat, time_s) in enumerate(trip_raw.track)
        ])

        # Member 2 scoring
        scored_df, matches_df = scorer.score(cands_df, obs_df)

        # Member 1 ground truth references
        ref_route = read_reference_route(trip_raw, trip_dir)
        refs = align_references(trip_raw, ref_route, align_config)

        # Member 3 decoding & comparison
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

        elapsed = time.time() - t_start

        row = {
            "trip_id": trip_id,
            "observations": len(trip_raw.track),
            "candidates": len(cands),
            "aligned_obs": comp["aligned_count"],
            "candidate_miss_rate": comp["candidate_miss_rate"],
            # Baseline
            "base_accuracy": comp["baseline"]["top1_accuracy"],
            "base_correct": comp["baseline"]["correct_count"],
            "base_macro_f1": comp["baseline"]["macro_f1"],
            "base_continuity": comp["baseline"]["continuity_rate"],
            "base_continuous": comp["baseline"]["continuous_transitions"],
            "base_eligible": comp["baseline"]["eligible_transitions"],
            "base_breaks": comp["baseline"]["breaks"],
            # Viterbi
            "viterbi_accuracy": comp["viterbi"]["top1_accuracy"],
            "viterbi_correct": comp["viterbi"]["correct_count"],
            "viterbi_macro_f1": comp["viterbi"]["macro_f1"],
            "viterbi_continuity": comp["viterbi"]["continuity_rate"],
            "viterbi_continuous": comp["viterbi"]["continuous_transitions"],
            "viterbi_eligible": comp["viterbi"]["eligible_transitions"],
            "viterbi_breaks": comp["viterbi"]["breaks"],
            "viterbi_route_parts": comp["viterbi"]["route_parts_count"],
            "viterbi_route_breaks": comp["viterbi"]["route_breaks_count"],
            "viterbi_edit_distance": comp["viterbi"]["sequence_edit_distance"]["raw_edit_distance"],
            "viterbi_norm_edit_dist": comp["viterbi"]["sequence_edit_distance"]["normalized_edit_distance"],
            # Deltas
            "acc_improvement": comp["delta"]["accuracy_improvement"],
            "f1_improvement": comp["delta"]["macro_f1_improvement"],
            "continuity_improvement": comp["delta"]["continuity_improvement"],
            "elapsed_seconds": elapsed,
        }
        trip_results.append(row)
        print(
            f"Trip {trip_id} done in {elapsed:.2f}s | "
            f"Acc: {row['base_accuracy']:.4f} -> {row['viterbi_accuracy']:.4f} | "
            f"Cont: {row['base_continuity']:.4f} -> {row['viterbi_continuity']:.4f} "
            f"(+{row['continuity_improvement']*100:.2f}pp)",
            flush=True,
        )

    total_time = time.time() - total_t0
    df = pd.DataFrame(trip_results)
    df.to_csv(out_path / "student_test_comparison.csv", index=False)

    # Compute aggregate metrics
    total_aligned = df["aligned_obs"].sum()
    total_base_correct = df["base_correct"].sum()
    total_viterbi_correct = df["viterbi_correct"].sum()

    agg_base_acc = total_base_correct / total_aligned if total_aligned > 0 else 0.0
    agg_vit_acc = total_viterbi_correct / total_aligned if total_aligned > 0 else 0.0

    total_eligible = df["base_eligible"].sum()
    total_base_continuous = df["base_continuous"].sum()
    total_viterbi_continuous = df["viterbi_continuous"].sum()

    agg_base_cont = total_base_continuous / total_eligible if total_eligible > 0 else 0.0
    agg_vit_cont = total_viterbi_continuous / total_eligible if total_eligible > 0 else 0.0

    agg_summary = {
        "num_trips": len(trip_dirs),
        "total_observations": int(df["observations"].sum()),
        "total_aligned_observations": int(total_aligned),
        "aggregate_baseline_accuracy": float(agg_base_acc),
        "aggregate_viterbi_accuracy": float(agg_vit_acc),
        "accuracy_delta": float(agg_vit_acc - agg_base_acc),
        "mean_baseline_macro_f1": float(df["base_macro_f1"].mean()),
        "mean_viterbi_macro_f1": float(df["viterbi_macro_f1"].mean()),
        "macro_f1_delta": float(df["viterbi_macro_f1"].mean() - df["base_macro_f1"].mean()),
        "total_eligible_transitions": int(total_eligible),
        "aggregate_baseline_continuity": float(agg_base_cont),
        "aggregate_viterbi_continuity": float(agg_vit_cont),
        "continuity_delta": float(agg_vit_cont - agg_base_cont),
        "mean_candidate_miss_rate": float(df["candidate_miss_rate"].mean()),
        "mean_normalized_edit_distance": float(df["viterbi_norm_edit_dist"].mean()),
        "total_runtime_seconds": float(total_time),
    }

    (out_path / "student_test_summary.json").write_text(json.dumps(agg_summary, indent=2))

    print("\n" + "=" * 60)
    print("ALL 10 STUDENT TEST TRIPS EVALUATED SUCCESSFULLY")
    print(f"Total Observations: {agg_summary['total_observations']}")
    print(f"Total Aligned Obs:  {agg_summary['total_aligned_observations']}")
    print(f"Baseline Accuracy:  {agg_summary['aggregate_baseline_accuracy']:.4f}")
    print(f"Viterbi Accuracy:   {agg_summary['aggregate_viterbi_accuracy']:.4f} ({agg_summary['accuracy_delta']:+.4f})")
    print(f"Baseline Continuity:{agg_summary['aggregate_baseline_continuity']:.4f}")
    print(f"Viterbi Continuity: {agg_summary['aggregate_viterbi_continuity']:.4f} ({agg_summary['continuity_delta']:+.4f})")
    print(f"Total Runtime:      {total_time:.2f}s")
    print(f"Results written to: {out_path}")
    print("=" * 60)

    return agg_summary


if __name__ == "__main__":
    run_all_student_test()
