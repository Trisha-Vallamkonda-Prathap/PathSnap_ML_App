import pandas as pd

df = pd.read_csv('reports/decoder-comparison/break_diagnosis.csv')

# Trip 77 speed violations
sv77 = df[(df['category']=='speed_violation') & (df['trip_id']==77)].copy()
sv77 = sv77.sort_values('prev_obs_idx')

print("=== TRIP 00000077: Speed violations ===")
print(f"Total violations: {len(sv77)}")
print(f"d_net range: {sv77['d_net_m'].min():.1f} - {sv77['d_net_m'].max():.1f} m")
print(f"speed range: {sv77['speed_mps'].min():.1f} - {sv77['speed_mps'].max():.1f} m/s")

same_arc77 = sv77[sv77['prev_arc'] == sv77['curr_arc']]
diff_arc77 = sv77[sv77['prev_arc'] != sv77['curr_arc']]
print(f"  Same arc (prev==curr): {len(same_arc77)}")
print(f"  Different arc: {len(diff_arc77)}")
print("\nFirst 20 violations:")
print(sv77[['prev_obs_idx','curr_obs_idx','prev_arc','curr_arc','d_net_m','speed_mps']].head(20).to_string())

# Trip 7 same arc analysis
print("\n=== TRIP 00000007: Speed violations ===")
sv7 = df[(df['category']=='speed_violation') & (df['trip_id']==7)].copy()
same7 = sv7[sv7['prev_arc'] == sv7['curr_arc']]
diff7 = sv7[sv7['prev_arc'] != sv7['curr_arc']]
print(f"  Same arc: {len(same7)}, Different arc: {len(diff7)}")
if len(same7):
    print(same7[['prev_obs_idx','curr_obs_idx','prev_arc','d_net_m','speed_mps']].head(20).to_string())

# Trip 40 same arc analysis
print("\n=== TRIP 00000040: Speed violations ===")
sv40 = df[(df['category']=='speed_violation') & (df['trip_id']==40)].copy()
same40 = sv40[sv40['prev_arc'] == sv40['curr_arc']]
diff40 = sv40[sv40['prev_arc'] != sv40['curr_arc']]
print(f"  Same arc: {len(same40)}, Different arc: {len(diff40)}")
print(sv40[['prev_obs_idx','curr_obs_idx','prev_arc','curr_arc','d_net_m','speed_mps']].head(30).to_string())

# Global: how many same-arc violations
sv_all = df[df['category']=='speed_violation']
same_all = sv_all[sv_all['prev_arc'] == sv_all['curr_arc']]
diff_all = sv_all[sv_all['prev_arc'] != sv_all['curr_arc']]
print(f"\n=== GLOBAL BREAKDOWN ===")
print(f"  Same-arc violations: {len(same_all)} / {len(sv_all)}")
print(f"  Diff-arc violations: {len(diff_all)} / {len(sv_all)}")
print(f"  Same-arc d_net stats: mean={same_all['d_net_m'].mean():.1f}, max={same_all['d_net_m'].max():.1f}")
print(f"  Diff-arc d_net stats: mean={diff_all['d_net_m'].mean():.1f}, max={diff_all['d_net_m'].max():.1f}")
