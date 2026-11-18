from pathlib import Path
from pathsnap.data.loading import load_trip
from pathsnap.decoder.models import Trip

trip_dir = Path("Student Folder/student_test/00000077")
trip_raw = load_trip(trip_dir)
dec_trip = Trip.from_dict(trip_raw.to_record())
arcs = {a.arc_id: a for a in dec_trip.arcs}

pairs = [(44, 45), (4305, 4306), (4417, 4418), (5085, 5094), (8344, 8348), (8345, 8342)]
for a1, a2 in pairs:
    o1 = arcs.get(a1)
    o2 = arcs.get(a2)
    print(f"Arc {a1}: {o1.from_node_id} -> {o1.to_node_id}, length={o1.length_m:.1f}m" if o1 else f"{a1} missing")
    print(f"Arc {a2}: {o2.from_node_id} -> {o2.to_node_id}, length={o2.length_m:.1f}m" if o2 else f"{a2} missing")
    if o1 and o2:
        is_reverse = (o1.from_node_id == o2.to_node_id and o1.to_node_id == o2.from_node_id)
        print(f"  Are opposite directions of same road segment? {is_reverse}")
