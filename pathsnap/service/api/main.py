"""PathSnap API endpoints and request logging (runs on Laptop 1)."""
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from . import ground_truth, route, trips
from .config import MODEL_PATH, TRIPS_DIR
from .loader import TripDataError, load_trip_inputs
from .predictor import DECODER_CONFIG, ModelNotAvailable, load_scorer, predict_trip

# ---- Logging: one clean line per request, shown to the evaluator during the demo ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("pathsnap.api")
# Uvicorn prints its own access line too; hide it so each request is logged once, in our format.
logging.getLogger("uvicorn.access").disabled = True

app = FastAPI(title="PathSnap API")
log.info("PathSnap API starting | trip folder: %s | %d trips found", TRIPS_DIR, len(trips.list_trip_ids()))

# Load the model once at start-up so the first request is not slow, and so a missing
# model is reported immediately in the console.
try:
    _scorer = load_scorer()
    log.info("Model loaded | %s | %s", _scorer.model_name, MODEL_PATH)
except ModelNotAvailable as err:
    log.error("MODEL NOT LOADED | %s", err)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Time every request and log: method, path, trip_id, status code, duration, caller IP."""
    start = time.perf_counter()
    status = 500  # used if the endpoint crashes
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        ms = (time.perf_counter() - start) * 1000
        trip_id = getattr(request.state, "trip_id", "-")  # endpoints set this when a trip is involved
        caller = request.client.host if request.client else "-"
        log.info("%-4s %-9s | trip=%-8s | status=%d | %8.1f ms | from %s",
                 request.method, request.url.path, trip_id, status, ms, caller)


@app.get("/health")
def health():
    """Simple liveness check: Laptop 2 calls this to see if the API is reachable."""
    return {"status": "ok"}


@app.get("/trips")
def get_trips():
    """List the trip IDs in the configured trip folder (contract: {"trips": [{"trip_id": ...}]})."""
    if not TRIPS_DIR.is_dir():
        raise HTTPException(status_code=500,
                            detail=f"Trip folder not found: {TRIPS_DIR}. Set PATHSNAP_TRIPS_DIR.")
    return {"trips": [{"trip_id": t} for t in trips.list_trip_ids()]}


class PredictRequest(BaseModel):
    trip_id: str  # a body without a string trip_id is rejected automatically with HTTP 422


@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    """Run the full pipeline for one trip and score it on the server."""
    trip_id = body.trip_id
    request.state.trip_id = trip_id  # shown in the request log line

    # 1. Resolve the trip through the configured trip list.
    try:
        folder = trips.trip_folder(trip_id)
    except trips.TripNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown trip_id '{trip_id}'. See GET /trips.")

    # 2. Load the prediction inputs only (GPS + road network; no ground truth).
    try:
        trip = load_trip_inputs(folder)
    except TripDataError as err:
        raise HTTPException(status_code=500, detail=str(err))
    points = trip.track  # (N, 3): lon, lat, time_s. Row = obs_idx

    # 3. Run the pipeline: candidates -> model scores -> HMM/Viterbi. Never sees ground truth.
    start = time.perf_counter()
    try:
        decoded, classifier_only = predict_trip(trip)
    except ModelNotAvailable as err:
        raise HTTPException(status_code=500, detail=str(err))
    inference_ms = (time.perf_counter() - start) * 1000
    if len(decoded.matches) != len(points):
        raise HTTPException(status_code=500, detail="Decoder returned the wrong number of matches.")

    # 4. Build the drawable route from the decoded route parts.
    geometry = route.parts_to_geometry(decoded, trip)
    continuity = route.route_continuity(decoded, trip, DECODER_CONFIG)

    # 5. Score AFTER predicting, on the server. Only aggregate counts come back.
    try:
        decoded_arcs = {m.obs_idx: m.arc_id for m in decoded.matches}
        metrics = ground_truth.score_trip(trip, folder, decoded_arcs)
        # Same scoring for the model's Top-1 picks before HMM decoding (for the comparison).
        classifier_metrics = ground_truth.score_trip(trip, folder, classifier_only)
    except (OSError, ValueError) as err:
        raise HTTPException(status_code=500, detail=f"Scoring failed for trip {trip_id}: {err}")

    matched = sum(m.arc_id is not None for m in decoded.matches)
    breaks = sum(p.break_before is not None for p in decoded.route_parts)
    if matched == 0:
        status = "no_match"
    elif matched < len(points) or breaks > 0:
        status = "partial"
    else:
        status = "ok"

    acc_text = (f"{metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['aligned']})"
                if metrics else "n/a")
    cont_text = (f"{continuity['value']:.1%} ({continuity['connected']}/{continuity['transitions']})"
                 if continuity["value"] is not None else "n/a")
    clf_text = (f"{classifier_metrics['accuracy']:.1%}" if classifier_metrics else "n/a")
    log.info("     predict %s | %d points | matched %d | breaks %d | accuracy %s (classifier only %s) "
             "| continuity %s | inference %.0f ms",
             trip_id, len(points), matched, breaks, acc_text, clf_text, cont_text, inference_ms)

    return {
        # --- shared API-to-UI contract (handoffs/04) ---
        "trip_id": trip_id,
        "status": status,
        "route_geometry": geometry,
        "counts": {"observations": len(points), "matched": matched, "route_breaks": breaks},
        "metrics": metrics,
        # --- extras for the UI ---
        "route_continuity": continuity,
        "classifier_only_metrics": classifier_metrics,  # model Top-1 before HMM, aggregate only
        "gps_points": points[:, :2].tolist(),  # [[lon, lat], ...] observed input, not ground truth
        "matches": [{"obs_idx": m.obs_idx, "arc_id": m.arc_id, "unmatched_reason": m.unmatched_reason}
                    for m in decoded.matches],
        "inference_ms": round(inference_ms, 1),
    }
