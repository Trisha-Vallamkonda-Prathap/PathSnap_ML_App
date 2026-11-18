"""Inspect arc lengths and same-arc backward routing behavior."""
from pathlib import Path
from dataclasses import asdict
import pandas as pd

from pathsnap.data.loading import load_trip
from pathsnap.decoder.models import Trip
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph
from pathsnap.decoder.config import DecoderConfig

config = DecoderConfig()

def inspect_arcs(trip_dir_str, arc_ids):
    trip_dir = Path(trip_dir_str)
    trip_raw = load_trip(trip_dir)
    dec_trip = Trip.from_dict(trip_raw.to_record())
    
    graph = RoadNetworkGraph(
        nodes=dec_trip.nodes,
        arcs=dec_trip.arcs,
        max_search_distance_m=config.max_network_search_m,
    )
    
    print(f"\n=== Trip: {trip_dir.name} ===")
    for arc_id in arc_ids:
        arc_obj = graph.arcs.get(arc_id)
        if arc_obj:
            print(f"  Arc {arc_id}: length={arc_obj.length_m:.1f}m, from={arc_obj.from_node_id} -> to={arc_obj.to_node_id}")
            
            # Test forward (0.1 -> 0.9): should be small
            d_fwd, _ = graph.get_candidate_distance_and_path(arc_id, 0.1, arc_id, 0.9)
            # Test backward (0.9 -> 0.1): triggers cycle search
            d_bwd, bwd_path = graph.get_candidate_distance_and_path(arc_id, 0.9, arc_id, 0.1)
            print(f"    frac 0.1->0.9 (fwd): d_net={d_fwd:.1f}m")
            if d_bwd is not None:
                print(f"    frac 0.9->0.1 (bwd): d_net={d_bwd:.1f}m  <-- CYCLE COST: {bwd_path}")
            else:
                print(f"    frac 0.9->0.1 (bwd): None (no cycle)")
        else:
            print(f"  Arc {arc_id}: NOT FOUND")

# Trip 7 - arc 285 same-arc violations ~470m
inspect_arcs("Student Folder/student_test/00000007", [285, 282, 281, 280, 268])

# Trip 40 - arc 3411, 3449 same-arc violations ~180m
inspect_arcs("Student Folder/student_test/00000040", [3411, 3449, 3412])

# Trip 77 - arc 8345, 8342, 8343
inspect_arcs("Student Folder/student_test/00000077", [8345, 8342, 8343, 8341, 8344, 8347])

print("\nDONE")
