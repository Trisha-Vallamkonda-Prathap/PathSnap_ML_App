import pandas as pd

df = pd.read_csv('reports/decoder-comparison/break_diagnosis.csv')
diff_sv = df[(df['category'] == 'speed_violation') & (df['prev_arc'] != df['curr_arc'])]

print(f"Total diff-arc speed violations: {len(diff_sv)}")
print(diff_sv[['trip_id', 'prev_obs_idx', 'curr_obs_idx', 'prev_arc', 'curr_arc', 'dt_s', 'd_gps_m', 'd_net_m', 'speed_mps']].head(25).to_string())

print("\nDiff-arc summary by trip:")
print(diff_sv.groupby('trip_id').size())

print("\nd_gps vs d_net for diff-arc violations:")
print("d_gps:", diff_sv['d_gps_m'].describe())
print("d_net:", diff_sv['d_net_m'].describe())
