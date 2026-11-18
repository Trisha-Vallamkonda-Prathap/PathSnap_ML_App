"""Investigate trip 00000077: compare viterbi arc choices vs reference arcs."""
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

config = DecoderConfig()
scorer = CandidateScorer.load("artifacts/models/pathsnap_scorer.joblib")
for _, step in scorer.pipeline.steps:
    if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
        step._fill_dtype = step._fit_dtype

trip_dir = Path("Student Folder/student_test/00000077")
trip_raw = load_trip(trip_dir)
cand_config = CandidateConfig()
align_config = AlignmentConfig()

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

# Build a comparison table: obs_idx, ref_arc, baseline_arc, viterbi_arc
ref_map = {r["obs_idx"]: r["ref_arc_id"] for r in ref_dicts}
base_map = {m["obs_idx"]: m["arc_id"] for m in classifier_matches}
vit_map = {m.obs_idx: m.arc_id for m in decoded_trip.matches}

rows = []
for obs_idx in sorted(set(ref_map) | set(base_map) | set(vit_map)):
    r = ref_map.get(obs_idx)
    b = base_map.get(obs_idx)
    v = vit_map.get(obs_idx)
    rows.append({
        "obs_idx": obs_idx,
        "ref_arc": r,
        "base_arc": b,
        "vit_arc": v,
        "base_correct": r == b if r is not None else None,
        "vit_correct": r == v if r is not None else None,
        "base_ok_vit_wrong": (r == b) and (r != v) if r is not None else False,
    })
df77 = pd.DataFrame(rows)

print("=== TRIP 00000077 ===")
print(f"Total obs: {len(df77)}")
print(f"Baseline correct: {df77['base_correct'].sum()} / {df77['base_correct'].count()}")
print(f"Viterbi correct: {df77['vit_correct'].sum()} / {df77['vit_correct'].count()}")
print(f"Baseline correct but Viterbi wrong: {df77['base_ok_vit_wrong'].sum()}")

# Look at cases where viterbi is wrong but baseline is right
regressions = df77[df77["base_ok_vit_wrong"]].head(30)
print(f"\nTop 30 regressions (base right, viterbi wrong):")
print(regressions[["obs_idx","ref_arc","base_arc","vit_arc"]].to_string())

# What arcs does viterbi choose when it's wrong?
wrong_vit = df77[(df77["vit_correct"]==False) & (df77["vit_arc"].notna())]
print(f"\nViterbi wrong choice distribution (arc_id counts):")
print(wrong_vit["vit_arc"].value_counts().head(20).to_string())

print(f"\nRef arc distribution for viterbi-wrong observations:")
print(wrong_vit["ref_arc"].value_counts().head(20).to_string())
