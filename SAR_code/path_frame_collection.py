import asf_search as asf
import pandas as pd

df_dir = r"D:\SAR-Intelligence\s1_temporal_stacks.csv"
df = pd.read_csv(df_dir)
# p=22 f=446
lis_df_col = df.columns  #(imp_index,imp_value):(2,path),(3,frame)

def num_scene_exist(df:pd.DataFrame,path:int,frame:int,lis:str):
    lis_df_col = df.columns
    df_row = df[(df[lis_df_col[2]]==path) & (df[lis_df_col[3]]==frame)]
    if lis=='len':
        return f"Total Scenes:{len(df_row)}"
    else:
        return df_row
    
print(num_scene_exist(df,16,209,'len'))

path = 16
frame = 209

print(f"Identified as Path:{path}")
print(f"Identified as Frame:{frame}")

inventory_results = asf.search(
    platform=asf.PLATFORM.SENTINEL1,
    relativeOrbit=path,
    frame=frame,
    processingLevel='GRD_HD',
    beamMode='IW'
)

total_snapshots = len(inventory_results)

print("SUCCESS")
print(f"Totall available snapshots for Path {path}/Frame {frame}: {total_snapshots}")

dates = sorted([r.properties['startTime'] for r in inventory_results])
print(f"Data range:{dates[0][:10]} to {dates[-1][:10]}")

results_2020 = asf.search(
    platform=asf.PLATFORM.SENTINEL1,
    relativeOrbit=path,
    frame=frame,
    processingLevel='GRD_HD',
    beamMode='IW',
    start='2020-01-01T00:00:00Z',  # Start of 2020
    end='2020-01-31T23:59:59Z'
)
count = len(results_2020)
print(f"\nFound {count} snapshots in 2020.")

    # Optional: List the exact dates found
print("Dates acquired:")
for r in results_2020:
    print(f" - {r.properties['startTime'][:10]}")