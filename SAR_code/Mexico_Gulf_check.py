import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
from shapely.geometry import box

def plot_bbox(min_lat, min_lon, max_lat, max_lon, title="Bounding Box"):
    
    # Create bounding box geometry
    bbox = box(min_lon, min_lat, max_lon, max_lat)
    
    # Convert to GeoDataFrame
    gdf_bbox = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
    
    # Load world map
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    world.plot(ax=ax, color='lightgrey', edgecolor='white')
    gdf_bbox.boundary.plot(ax=ax, edgecolor='red', linewidth=2)
    
    # Zoom to bbox
    buffer = 1
    minx, miny, maxx, maxy = gdf_bbox.total_bounds
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)
    
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()

plot_bbox(
    min_lat=17.4068,
    min_lon=-98.0539,
    max_lat=31.4648,
    max_lon=-80.433,
    title="Gulf of mexico Bounding Box"
)