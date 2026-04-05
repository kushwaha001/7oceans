import pandas as pd
import geopandas as gpd
import ast
from shapely.geometry import shape

df = pd.read_csv(r"D:\SAR-Intelligence\s1_temporal_stacks.csv")

def parse_geometry(geom_str):
    geom_dict = ast.literal_eval(geom_str)
    return shape(geom_dict)

df["geometry"] = df["geometry"].apply(parse_geometry) # type:ignore
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf_proj = gdf.to_crs(epsg=32629)

gdf_proj["area_m2"] = gdf_proj.area
gdf_proj["area_km2"] = gdf_proj["area_m2"] / 1e6

print(gdf_proj[["sceneName", "area_km2"]])

gdf_equal_area = gdf.to_crs("EPSG:6933")  # World Equal Area
gdf_equal_area["area_m2"] = gdf_equal_area.area

total_area = gdf_proj["area_m2"].sum()
print("Total Area (km²):", total_area / 1e6)