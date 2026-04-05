import asf_search as asf
import pandas as pd
from dateutil.parser import parse
from datetime import timedelta

product_dir = r"D:\SAR-Intelligence\data_engine\SAR_Data_csv_info\ESA_xView3_sceneName_mapping.csv"
product_df = pd.read_csv(product_dir)

scene_names = list(product_df[list(product_df.columns)[0]])
print(f"Fetching metadata for {len(scene_names)} scenes...")

results = asf.granule_search(scene_names)
print(f"Metadata retrieved for {len(results)} scenes")

metadata = []
count=0
for r in results:
    if r.properties['sceneName'] in scene_names:
        # print(f"{}"}
        meta = {
            'sceneName':r.properties['sceneName'],
            'startTime':r.properties['startTime'],
            'path':r.properties['pathNumber'],
            'frame':r.properties['frameNumber'],
            'direction':r.properties['flightDirection'],
            'geometry':r.geometry
        } 
        metadata.append(meta)
    else:
        print(f"{r.properties['sceneName']} not in the list")

df = pd.DataFrame(metadata)
df = df.drop_duplicates(subset=['sceneName'],keep='first')
print(f"Dataset Cleaned total entries:{len(df)}")
df['startTime'] = pd.to_datetime(df['startTime'])

stacks = df.groupby(['path','frame','direction'])

print(f"Found {len(stacks)} unique spatial stacks.")
print("-"*30)

count = 1
for (path,frame,direction),stack_df in stacks:
    print(count)
    print(f"Stack:Path {path} | Frame:{frame} | {direction}")
    print(f"Count:{len(stack_df)} scenes")

    start = stack_df['startTime'].min()
    end = stack_df['startTime'].max()
    duration = (end-start).days

    print(f"Timeline:{start.date()} to {end.date()}({duration} days)")
    print("-"*30)
    count+=1

# Saving Stack Info
df.to_csv("s1_temporal_stacks.csv", index=False)
