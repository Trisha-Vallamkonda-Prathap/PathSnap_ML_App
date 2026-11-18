"""Turn the decoded route into map geometry and a continuity score.

Uses only the prediction and the road network (no ground truth).
"""
from typing import Dict, Optional

from pathsnap.data.loading import Trip
from pathsnap.decoder import DecodedTrip, DecoderConfig, RoadNetworkGraph
from pathsnap.decoder.evaluation.metrics import compute_route_continuity


def parts_to_geometry(decoded: DecodedTrip, trip: Trip) -> Optional[Dict]:
    """GeoJSON MultiLineString with one line per route part, coordinates as [lon, lat].

    Each decoder route part is a connected directed path (it already includes the
    in-between arcs). Separate lines mean the map shows a gap at each break instead
    of a false straight line.
    """
    lines = []
    for part in decoded.route_parts:
        if not part.arc_ids:
            continue
        first_node = trip.arc_nodes[part.arc_ids[0], 0]
        node_ids = [first_node] + [trip.arc_nodes[a, 1] for a in part.arc_ids]  # from-node, then each to-node
        lines.append([[float(trip.node_lonlat[n, 0]), float(trip.node_lonlat[n, 1])] for n in node_ids])
    return {"type": "MultiLineString", "coordinates": lines} if lines else None


def route_continuity(decoded: DecodedTrip, trip: Trip, config: DecoderConfig) -> Dict:
    """Member 3's continuity metric: of the consecutive matched GPS steps (GPS gaps excluded),
    how many are reachable on the directed road network at a plausible speed."""
    record = trip.to_record()
    graph = RoadNetworkGraph(record["nodes"], record["arcs"],
                             max_search_distance_m=config.max_network_search_m,
                             same_arc_reverse_tolerance_m=config.same_arc_reverse_tolerance_m)
    result = compute_route_continuity(decoded.matches, record["observations"], graph, config)
    return {
        "value": result["continuity_rate"],              # None = not measurable
        "connected": result["continuous_transitions"],
        "transitions": result["eligible_transitions"],
        "excluded_gaps": result["excluded_gaps"],
        "unresolved_points": result["unresolved_points"],
    }
