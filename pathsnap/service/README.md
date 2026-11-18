# PathSnap service (Member 4): API + UI runbook

The API on Laptop 1 runs the full team pipeline for one trip:

1. `pathsnap.data.loading.load_trip` and `pathsnap.data.candidates.generate_candidates` (Member 1)
2. `CandidateScorer` with `artifacts/models/pathsnap_scorer.joblib`, the tuned LightGBM ranker (Member 2)
3. `pathsnap.decoder.decode_trip`, HMM + Viterbi (Member 3)
4. Server-side scoring with `pathsnap.data.reference_alignment.scoring.score_trip` (Member 1), run only after
   prediction. Only aggregate numbers are returned; reference arcs never leave Laptop 1.

## Setup (once, on each laptop)

Python 3.10 or newer is required (3.12 recommended). macOS's built-in Python 3.9 is too old.

```bash
git lfs install && git lfs pull          # downloads the real .joblib model files (~2 MB each)
python3.12 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the trips in `data/student_test/<trip_id>/` (ignored by Git), or point `PATHSNAP_TRIPS_DIR` elsewhere.

## Laptop 1: API

```bash
python -m uvicorn pathsnap.service.api:app --host 0.0.0.0 --port 8000
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `PATHSNAP_TRIPS_DIR` | `data/student_test` | Trip folders served by `/trips`. Point it at the evaluator trips on demo day. |
| `PATHSNAP_MODEL_PATH` | `artifacts/models/pathsnap_scorer.joblib` | Member 2's saved scorer |

At start-up the console prints the trip folder, the number of trips, and `Model loaded | lightgbm_ranker`.
Every request prints one log line; `/predict` also prints accuracy, continuity and inference time.

## Laptop 2: UI

```bash
export PATHSNAP_API_URL=http://LAPTOP1_IP:8000      # Windows PowerShell: $env:PATHSNAP_API_URL="http://LAPTOP1_IP:8000"
streamlit run pathsnap/service/ui/app.py
```

Find `LAPTOP1_IP` with `ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows). Both laptops must be on the
same Wi-Fi, and Laptop 1's firewall must allow port 8000.

## Check the connection

```bash
PATHSNAP_API_URL=http://LAPTOP1_IP:8000 python -m pathsnap.service.smoke_test
```

It calls `/health`, `/trips`, `/predict` for every trip, the 404 and 422 cases, and fails if a response
contains any field outside the agreed contract.
