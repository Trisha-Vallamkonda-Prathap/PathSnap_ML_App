"""Investigate arc lengths and graph.py logic for same-arc speed violations."""
import sys, math
from dataclasses import asdict
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[4]))

from pathsnap.data.loading import load_trip
from pathsnap.decoder.models import Trip
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph
from pathsnap.decoder.config import DecoderConfig

config = DecoderConfig()

# Focus on trip 00000007 - arc 285 same-arc violations ~470m d_net in 1s
# and trip 00000040 - arc 3411/3449 same-arc violations ~180m d_net in 1s

def inspect_arc(trip_dir_str, arc_ids_to_check):
    trip_dir = Path(trip_dir_str)
    trip_raw = load_trip(trip_dir)
    dec_trip = Trip.from_dict(trip_raw.to_record())
    
    graph = RoadNetworkGraph(
        nodes=dec_trip.nodes,
        arcs=dec_trip.arcs,
        max_search_distance_m=config.max_network_search_m,
    )
    
    print(f"\nTrip: {trip_dir.name}")
    for arc_id in arc_ids_to_check:
        arc_data = graph._arcs.get(arc_id)
        if arc_data:
            print(f"  Arc {arc_id}: length={arc_data['length_m']:.1f}m, from={arc_data['from_node']} to={arc_data['to_node']}")
        else:
            print(f"  Arc {arc_id}: NOT FOUND in graph")
    
    # Test: what is the same-arc d_net at frac 0.0 to 1.0?
    for arc_id in arc_ids_to_check:
        arc_data = graph._arcs.get(arc_id)
        if arc_data:
            L = arc_data['length_m']
            # from frac 0.9 to frac 0.1 (backward on same arc) - should be impossible
            d, path = graph.get_candidate_distance_and_path(arc_id, 0.9, arc_id, 0.1)
            print(f"  Arc {arc_id}: frac 0.9->0.1 (backward): d_net={d}, path={path}")
            # forward: frac 0.1 to 0.9
            d2, path2 = graph.get_candidate_distance_and_path(arc_id, 0.1, arc_id, 0.9)
            print(f"  Arc {arc_id}: frac 0.1->0.9 (forward): d_net={d2:.1f}m, path={path2}")
            # full arc: 0.0 to 1.0
            d3, path3 = graph.get_candidate_distance_and_path(arc_id, 0.0, arc_id, 1.0)
            print(f"  Arc {arc_id}: frac 0.0->1.0: d_net={d3:.1f}m")

# Trip 7, arc 285
inspect_arc("Student Folder/student_test/00000007", [285, 282, 281, 280, 268])

# Trip 40, arcs 3411, 3449, 3412
inspect_arc("Student Folder/student_test/00000040", [3411, 3449, 3412, 3386])

# Trip 77, arcs 8345, 8342, 8343
inspect_arc("Student Folder/student_test/00000077", [8345, 8342, 8343, 8341, 8344, 8347, 8339])
