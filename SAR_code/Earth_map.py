import ast
import geopandas as gpd
import geodatasets
from shapely.geometry import shape
import pandas as pd
import matplotlib.pyplot as plt

def plot_footprints(df, title="Sentinel-1 SAR Footprints"):
    geometries = []
    for geom in df['Location']:
        if isinstance(geom, str):
            geom_dict = ast.literal_eval(geom)
        else:
            geom_dict = geom
        geometries.append(shape(geom_dict))

    # 1. Create the footprints dataframe
    gdf_footprints = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    
    # 2. Load the world map using the CORRECT geodatasets key
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    
    fig, ax = plt.subplots(figsize=(10, 8))

    # 3. Plot the WORLD map (background)
    world.plot(ax=ax, color='lightgrey', edgecolor='white')

    # 4. Plot the SAR FOOTPRINTS (the red overlays) - This was missing!
    gdf_footprints.plot(ax=ax, facecolor='red', alpha=0.2, edgecolor='darkred', linewidth=1.5)

    # 5. Zoom to the data
    minx, miny, maxx, maxy = gdf_footprints.total_bounds
    buffer = 3
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# --- Execution ---
data_dir = r"D:\SAR-Intelligence\data_engine\Analysis_output\Same_footprint_Overlap_dataset.csv3"
data_dir2 = r"D:\SAR-Intelligence\s1_temporal_stacks.csv"
data_dir3 = r"D:\SAR-Intelligence\Overlap_analysis.csv"
data_dir4 = r"D:\SAR-Intelligence\data_engine\Analysis_output\Same_footprint_Overlap_dataset.csv"
data_frame = pd.read_csv(data_dir2)

plot_footprints(data_frame)