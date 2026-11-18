import pandas as pd
from pathlib import Path
from pathsnap.data.loading import load_trip
from pathsnap.decoder.models import Trip
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph
from pathsnap.decoder.config import DecoderConfig

df = pd.read_csv('reports/decoder-comparison/break_diagnosis.csv')
sv = df[df['category'] == 'speed_violation']
same_arc_sv = sv[sv['prev_arc'] == sv['curr_arc']]

print(f"Total speed violations on same arc: {len(same_arc_sv)}")
print(f"prev_frac vs curr_frac on same-arc speed violations:")
print(f"Number where curr_frac < prev_frac: {(same_arc_sv['curr_frac'] < same_arc_sv['prev_frac']).sum()}")
print(f"Fraction differences (curr - prev):")
diffs = same_arc_sv['curr_frac'] - same_arc_sv['prev_frac']
print(diffs.describe())
