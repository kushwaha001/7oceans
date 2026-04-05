import pandas as pd
import asf_search as asf
from shapely.geometry import shape
import ast
from typing import Optional,List
import geopandas as gpd
from shapely import wkt

df_dir = r"D:\SAR-Intelligence\s1_temporal_stacks.csv"
df = pd.read_csv(df_dir)

lis_df_col = df.columns

def num_scene_exist(df:pd.DataFrame,path:int,frame:int,lis:str):
    lis_df_col = df.columns
    df_row = df[(df[lis_df_col[2]]==path) & (df[lis_df_col[3]]==frame)]
    if lis=='len':
        return f"Total Scenes:{len(df_row)}"
    else:
        return df_row

print(num_scene_exist(df,22,446,'len'))

def get_coordinate(df:pd.DataFrame,path:int,frame:int,lis:str):
    lis_df_col = df.columns
    df_row = df[(df[lis_df_col[2]]==path) & (df[lis_df_col[3]]==frame)]
    if lis == 'list_coordinate':
        return list(df_row[lis_df_col[-1]])
    elif lis == 'single':
        return df_row[0]
    
# lis = get_coordinate(df,22,446,'list_coordinate')

path = 22
frame = 446

lis = get_coordinate(df,path,frame,'list_coordinate')
row_geometry_str  = lis[0]   #type:ignore

polygon_dict = ast.literal_eval(row_geometry_str)
aoi_wkt = shape(polygon_dict).wkt

print(f"Searching Area:{aoi_wkt}")

result_gemotry = asf.search(
    platform=asf.PLATFORM.SENTINEL1,
    processingLevel='GRD_HD',
    intersectsWith=aoi_wkt,
    start='2020-03-01T00:00:00Z',  # Start of 2020
    end='2020-03-31T23:59:59Z'
)


results_path_frame = asf.search(
    platform=asf.PLATFORM.SENTINEL1,
    relativeOrbit=path,
    frame=frame,
    processingLevel='GRD_HD',
    beamMode='IW',
    start='2020-03-01T00:00:00Z',
    end='2020-03-31T23:59:59Z'
)

count_path_frame = len(results_path_frame)
count_geometry = len(result_gemotry)

print(f"Found {count_path_frame} snapshots in 2020-01 for the Path:{path} and Frame:{frame}")
print(f"Found {count_geometry} snapshot in 2020-01 for the Geometry:{aoi_wkt}")


print("-"*50)
print(f"Dates acquired for Frame:{frame} Path:{path}")
for r in results_path_frame:
    print(f"-{r.properties['startTime'][:10]}")
print("-"*50)

print("\n")

print("-"*50)
print(f"Dates acquired for Geometry:{aoi_wkt}")
for r in result_gemotry:
    print(f"-{r.properties['startTime'][:10]}")
print("-"*50)

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
    
    gdf_scenes = gpd.GeoDataFrame(data,geometry='geometry')

    gdf_scenes.set_crs(epsg=4326,inplace=True)

    target_geom = wkt.loads(target_aoi_wkt)
    gdf_target = gpd.GeoDataFrame({'geometry': [target_geom]}, crs="EPSG:4326")

    gdf_scenes_metric = gdf_scenes.to_crs(epsg=3857)
    gdf_target_metric = gdf_target.to_crs(epsg=3857)
    
    target_area = gdf_target_metric.geometry.iloc[0].area # type:ignore

    overlaps = []
    
    for index, row in gdf_scenes_metric.iterrows():
        intersection = row['geometry'].intersection(gdf_target_metric.geometry.iloc[0])
        intersect_area = intersection.area
        
        overlap_pct = (intersect_area / target_area) * 100
        overlaps.append(overlap_pct)

    gdf_scenes['overlap_pct'] = overlaps

    # 6. Formatting the Output
    # Sort by highest overlap first, then by date
    result_df = gdf_scenes[[
        'startTime', 'path', 'direction', 'overlap_pct', 'sceneName','geometry'
    ]].sort_values(by=['overlap_pct', 'startTime'], ascending=[False, True])

    return result_df

# --- USAGE EXAMPLE ---

# 1. Your AOI (From your CSV)
my_aoi = "POLYGON ((15.492675 43.069744, 15.888587 44.570686, 12.588525 44.9786, 12.274636 43.478409, 15.492675 43.069744))"

# 2. Run the analysis (assuming 'results' is your list from asf.search)
df_analysis = analyze_scene_overlaps(result_gemotry, aoi_wkt)

# 3. Filter for the "Good Stuff"
# Only keep scenes that cover at least 90% of your corridor
good_scenes = df_analysis[df_analysis['overlap_pct']>0]

print(f"Total Scenes: {len(df_analysis)}")
# print(f"Usable Full-Coverage Scenes: {len(good_scenes)}")
print(good_scenes[:])
good_scenes.to_csv("Overlap_analysis.csv")