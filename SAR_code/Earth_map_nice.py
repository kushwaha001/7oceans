import ast
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx  # The new basemap library!

def plot_footprints_satellite(df, title="Sentinel-1 SAR Footprints"):
    geometries = []
    for geom in df['Location']:
        if isinstance(geom, str):
            geom_dict = ast.literal_eval(geom)
        else:
            geom_dict = geom
        geometries.append(shape(geom_dict))

    # 1. Create the footprints dataframe (EPSG:4326 is standard Lat/Lon)
    gdf_footprints = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    
    # 2. Project to Web Mercator (EPSG:3857) - REQUIRED for satellite basemaps!
    gdf_footprints = gdf_footprints.to_crs(epsg=3857)
    
    fig, ax = plt.subplots(figsize=(10, 8))

    # 3. Plot the SAR FOOTPRINTS (slightly higher alpha since the background is dark)
    gdf_footprints.plot(ax=ax, facecolor='red', alpha=0.35, edgecolor='darkred', linewidth=1.5)

    # 4. Add the beautiful Esri Satellite Basemap!
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)  #type:ignore
    

    # 5. Zoom to the data (Buffer is now in meters, not degrees. 500000 = 500km)
    minx, miny, maxx, maxy = gdf_footprints.total_bounds
    buffer = 500000 
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)

    # 6. Formatting (Turning off axes because Web Mercator numbers look messy)
    ax.set_axis_off()
    plt.title(title, fontsize=16, pad=15)
    
    plt.tight_layout()
    plt.show()

# --- Execution ---
data_dir = r"D:\SAR-Intelligence\data_engine\Analysis_output\Same_footprint_from_dataBase.csv"
data_dir2 = r"D:\SAR-Intelligence\s1_temporal_stacks.csv"
data_dir3 = r"D:\SAR-Intelligence\Overlap_analysis.csv"
data_frame = pd.read_csv(data_dir2)

plot_footprints_satellite(data_frame)