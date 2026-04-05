import pandas as pd
import asf_search as asf
import geopandas as gpd
from shapely import wkt
from shapely.geometry import shape
import ast

df_dir = r"D:\SAR-Intelligence\s1_temporal_stacks.csv"
df = pd.read_csv(df_dir)
row_geometry_str = df[list(df.columns)[-1]][0]

polygon_dict = ast.literal_eval(row_geometry_str)
aoi_wkt = shape(polygon_dict).wkt
print(f"Searching Area:{aoi_wkt}")

results_geometry = asf.search(
    platform=asf.PLATFORM.SENTINEL1,
    processingLevel='GRD_HD',
    intersectsWith=aoi_wkt,
    start='2020-01-01T00:00:00Z',  # Start of 2020
    end='2020-01-31T23:59:59Z'
)

count_geometry = len(results_geometry)
print(f"Found {count_geometry} snapshot in 2020-01 for Geometry:{aoi_wkt}")

timeline_buckets = {}

for scene in results_geometry:
    path = scene.properties['pathNumber']
    direction = scene.properties['flightDirection']

    bucket_id = f"Path {path}({direction})"
    if bucket_id not in timeline_buckets:
        timeline_buckets[bucket_id] = []
    
    timeline_buckets[bucket_id].append(scene)

# print(timeline_buckets)

print(f"-"*30)
for bucket, scene in timeline_buckets.items():
    print(f"{bucket}:{len(scene)} frames")
    dates = sorted([s.properties['startTime'][:10] for s in scene])
    print(f"Dates:{dates[:5]}")

def analyze_scene_overlaps(asf_results,target_aoi_wkt):
    data = []
    for scene in asf_results:
        props = scene.properties
        data.append({
            'sceneName':props['sceneName'],
            'startTime':props['startTime'],
            'path':props['pathNumber'],
            'frame':props['frameNumber'],
            'direction':props['flightDirection'],
            'geometry':shape(scene.geometry)
        })