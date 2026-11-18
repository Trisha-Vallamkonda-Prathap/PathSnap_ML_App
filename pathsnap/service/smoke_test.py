"""Smoke test for a running PathSnap API.

Usage (API must already be running):
    python -m pathsnap.service.smoke_test
    PATHSNAP_API_URL=http://<laptop1-ip>:8000 python -m pathsnap.service.smoke_test

Checks /health, /trips, /predict for EVERY configured trip, the error codes,
and that responses contain only the agreed fields (no ground truth).
Exit code 0 = all passed, 1 = something failed.
"""
import os
import sys

import requests

API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Every key /predict is allowed to return. Anything else (e.g. reference arcs) is a failure.
ALLOWED_KEYS = {"trip_id", "status", "route_geometry", "counts", "metrics",
                "route_continuity", "classifier_only_metrics", "gps_points", "matches", "inference_ms"}
ALLOWED_MATCH_KEYS = {"obs_idx", "arc_id", "unmatched_reason"}

failures = []


def check(ok, message):
    """Print PASS/FAIL for one check and remember failures."""
    print(("  PASS  " if ok else "  FAIL  ") + message)
    if not ok:
        failures.append(message)


def main():
    print(f"PathSnap smoke test against {API_URL}\n")
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
    except requests.exceptions.RequestException as err:
        print(f"  FAIL  cannot reach the API: {err}")
        return 1
    check(health.status_code == 200 and health.json() == {"status": "ok"}, "GET /health -> {'status': 'ok'}")

    trips = requests.get(f"{API_URL}/trips", timeout=10)
    trip_ids = [t["trip_id"] for t in trips.json().get("trips", [])] if trips.ok else []
    check(trips.status_code == 200 and len(trip_ids) > 0, f"GET /trips -> {len(trip_ids)} trips")

    for trip_id in trip_ids:
        resp = requests.post(f"{API_URL}/predict", json={"trip_id": trip_id}, timeout=300)
        if resp.status_code != 200:
            check(False, f"POST /predict {trip_id} -> HTTP {resp.status_code}: {resp.text[:200]}")
            continue
        body = resp.json()
        n = body["counts"]["observations"]
        extra = set(body) - ALLOWED_KEYS
        extra_match = set().union(*(set(m) for m in body["matches"])) - ALLOWED_MATCH_KEYS
        ok = (body["trip_id"] == trip_id and len(body["matches"]) == n == len(body["gps_points"])
              and not extra and not extra_match)
        m = body["metrics"]
        acc = f"{m['accuracy']:.1%} ({m['correct']}/{m['aligned']})" if m else "n/a"
        check(ok, f"POST /predict {trip_id} -> {body['status']}, {n} points, accuracy {acc}"
                  + (f", UNEXPECTED KEYS {sorted(extra | extra_match)}" if extra or extra_match else ""))

    resp = requests.post(f"{API_URL}/predict", json={"trip_id": "does-not-exist"}, timeout=10)
    check(resp.status_code == 404, f"unknown trip_id -> HTTP {resp.status_code} (expected 404)")
    resp = requests.post(f"{API_URL}/predict", json={"wrong_field": 1}, timeout=10)
    check(resp.status_code == 422, f"invalid body -> HTTP {resp.status_code} (expected 422)")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
