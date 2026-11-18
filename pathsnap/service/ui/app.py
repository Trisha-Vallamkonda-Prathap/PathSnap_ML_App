"""PathSnap evaluator UI (Streamlit, runs on Laptop 2).

Start it with:  streamlit run pathsnap/service/ui/app.py
The UI only talks to the API over HTTP. It never reads data files or ground truth.
"""
import os

import folium
import requests
import streamlit as st
from branca.element import MacroElement, Template
from streamlit_folium import st_folium

# Address of the API on Laptop 1. Set it before starting the UI, e.g.
#   export PATHSNAP_API_URL=http://192.168.1.20:8000
API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_TIMEOUT_S = 120  # big trips can take a while once the real model is plugged in

GPS_COLOUR = "#1f77b4"    # blue
ROUTE_COLOUR = "#d62728"  # red


# ---------------------------------------------------------------- API calls
def get_trips():
    """Ask the API for the trip list. Returns (trip_ids, error_message)."""
    try:
        resp = requests.get(f"{API_URL}/trips", timeout=10)
        resp.raise_for_status()
        return [t["trip_id"] for t in resp.json()["trips"]], None
    except requests.exceptions.ConnectionError:
        return [], (f"Cannot reach the PathSnap API at {API_URL}. Check that the API is running on "
                    "Laptop 1, both laptops are on the same Wi-Fi, and PATHSNAP_API_URL is correct.")
    except requests.exceptions.Timeout:
        return [], f"The API at {API_URL} did not answer in time."
    except requests.exceptions.RequestException as err:
        return [], f"The API returned an error: {err}"


def run_prediction(trip_id):
    """Send the selected trip ID to POST /predict. Returns (result, error_message)."""
    try:
        resp = requests.post(f"{API_URL}/predict", json={"trip_id": trip_id}, timeout=PREDICT_TIMEOUT_S)
    except requests.exceptions.ConnectionError:
        return None, f"Lost connection to the API at {API_URL}. Is it still running?"
    except requests.exceptions.Timeout:
        return None, f"The prediction took longer than {PREDICT_TIMEOUT_S} s and was stopped."
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return None, f"API error {resp.status_code}: {detail}"
    return resp.json(), None


# ---------------------------------------------------------------- map
LEGEND_HTML = """
{% macro html(this, kwargs) %}
<div style="position: absolute; bottom: 24px; left: 12px; z-index: 9999; background: white;
            padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;
            font-family: sans-serif; color: #222;">
  <b>Legend</b><br>
  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
               background:""" + GPS_COLOUR + """;"></span>&nbsp;GPS points (input)<br>
  <span style="display:inline-block;width:18px;height:4px;vertical-align:middle;
               background:""" + ROUTE_COLOUR + """;"></span>&nbsp;Predicted route
</div>
{% endmacro %}
"""


def build_map(result):
    """Folium map with the GPS points and the predicted route as two toggleable layers."""
    gps = result["gps_points"]                  # [[lon, lat], ...]
    latlon = [(lat, lon) for lon, lat in gps]   # Folium wants (lat, lon)

    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
    lats = [lat for lat, _ in latlon]
    lons = [lon for _, lon in latlon]
    fmap.fit_bounds([(min(lats), min(lons)), (max(lats), max(lons))])  # zoom to the trip's bounding box

    # Route first, GPS points second, so the points are drawn on top of the line.
    route_layer = folium.FeatureGroup(name="Predicted route")
    geometry = result["route_geometry"]
    if geometry:  # null when nothing was matched
        for line in geometry["coordinates"]:    # one line per route part, so breaks show as gaps
            folium.PolyLine([(lat, lon) for lon, lat in line], color=ROUTE_COLOUR,
                            weight=3, opacity=0.6).add_to(route_layer)
    route_layer.add_to(fmap)

    gps_layer = folium.FeatureGroup(name="GPS points (input)")
    for lat, lon in latlon:
        folium.CircleMarker((lat, lon), radius=2, color=GPS_COLOUR, fill=True, fill_opacity=0.9,
                            weight=0).add_to(gps_layer)
    gps_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    legend = MacroElement()
    legend._template = Template(LEGEND_HTML)
    fmap.add_child(legend)
    return fmap


# ---------------------------------------------------------------- results
def show_metrics(result):
    """Show the aggregate scores returned by the API, with their denominators."""
    counts, metrics, cont = result["counts"], result["metrics"], result["route_continuity"]

    col1, col2, col3 = st.columns(3)
    if metrics:
        col1.metric("Pointwise accuracy", f"{metrics['accuracy']:.1%}")
        col1.caption(f"{metrics['correct']} correct of {metrics['aligned']} aligned GPS points"
                     + (f" · macro F1 {metrics['macro_f1']:.3f}" if metrics.get("macro_f1") is not None else ""))
        clf = result.get("classifier_only_metrics")
        if clf:
            col1.caption(f"Classifier only (before HMM): {clf['accuracy']:.1%} "
                         f"({clf['correct']}/{clf['aligned']})")
    else:
        col1.metric("Pointwise accuracy", "n/a")
        col1.caption("No reference available for this trip")
    if cont["value"] is not None:
        col2.metric("Route continuity", f"{cont['value']:.1%}")
        col2.caption(f"{cont['connected']} of {cont['transitions']} GPS-to-GPS steps are reachable on the road network")
    else:
        col2.metric("Route continuity", "n/a")
    col3.metric("Status", result["status"])
    col3.caption(f"Inference time: {result['inference_ms']:.0f} ms")

    col4, col5, col6 = st.columns(3)
    col4.metric("GPS points", counts["observations"])
    col5.metric("Matched points", counts["matched"])
    col6.metric("Route breaks", counts["route_breaks"])


# ---------------------------------------------------------------- page
st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching and route reconstruction")
st.caption(f"API: {API_URL}")

trip_ids, error = get_trips()
if error:
    st.error(error)
    if st.button("Try again"):
        st.rerun()
    st.stop()
if not trip_ids:
    st.warning("The API is running but has no trips configured (check PATHSNAP_TRIPS_DIR on Laptop 1).")
    st.stop()

left, right = st.columns([3, 1])
trip_id = left.selectbox("Trip", trip_ids)
right.write("")  # spacer to line the button up with the dropdown
if right.button("Predict", type="primary", use_container_width=True):
    with st.spinner(f"Running the PathSnap pipeline for trip {trip_id} on the API..."):
        result, error = run_prediction(trip_id)
    # Keep the result across Streamlit reruns (e.g. when the map is moved).
    st.session_state["result"], st.session_state["error"] = result, error

if st.session_state.get("error"):
    st.error(st.session_state["error"])
result = st.session_state.get("result")
if result:
    st.subheader(f"Trip {result['trip_id']}")
    show_metrics(result)
    # returned_objects=[] stops the map from triggering a page rerun on every pan/zoom.
    st_folium(build_map(result), height=600, use_container_width=True, returned_objects=[])
