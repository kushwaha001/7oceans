import pandas as pd
import asf_search as asf
from shapely.geometry import shape
import ast

df_dir = r"D:\SAR-Intelligence\s1_temporal_stacks.csv"
df = pd.read_csv(df_dir)
row_geometry_str = df[list(df.columns)[-1]][0]
print(type(row_geometry_str))

polygon_dict = ast.literal_eval(row_geometry_str)
aoi_wkt = shape(polygon_dict).wkt

print(f"Searching Area:{aoi_wkt}")

results = asf.search(
    platform='Sentinel-1',
    processingLevel='GRD_HD',
    intersectsWith=aoi_wkt,
    start='2020-01-01T00:00:00Z',  # Start of 2020
    end='2020-01-31T23:59:59Z'
)

print(f"Total Scene Found:{len(results)}")