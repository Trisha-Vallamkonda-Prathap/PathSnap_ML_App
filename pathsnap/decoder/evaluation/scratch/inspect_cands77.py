from pathlib import Path
from dataclasses import asdict
import pandas as pd
from pathsnap.data.loading import load_trip
from pathsnap.data.candidates import CandidateConfig, generate_candidates
from pathsnap.models.training.scorer import CandidateScorer

scorer = CandidateScorer.load("artifacts/models/pathsnap_scorer.joblib")
for _, step in scorer.pipeline.steps:
    if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
        step._fill_dtype = step._fit_dtype

trip_raw = load_trip(Path("Student Folder/student_test/00000077"))
cands = generate_candidates(trip_raw, CandidateConfig())
cands_df = pd.DataFrame([asdict(c) for c in cands])
obs_df = pd.DataFrame([
    {"trip_id": trip_raw.trip_id, "obs_idx": i, "lon": lon, "lat": lat, "time_s": t}
    for i, (lon, lat, t) in enumerate(trip_raw.track)
])
scored_df, matches_df = scorer.score(cands_df, obs_df)

for obs_i in [2, 12, 78]:
    sub = scored_df[scored_df["obs_idx"] == obs_i].sort_values("emission_prob", ascending=False)
    print(f"\nObs {obs_i} candidates:")
    print(sub[["arc_id", "along_arc_fraction", "distance_m", "emission_prob"]].head(6).to_string())
