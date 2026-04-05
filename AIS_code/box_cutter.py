from itertools import product
from typing import List,Tuple
import geopandas as gpd
import geodatasets
from shapely.geometry import box
import matplotlib.pyplot as plt

def generate_overlapping_bboxes(min_lat, min_lon, max_lat, max_lon, bin_size, overlap_pct)->List[Tuple[int,int,int,int]]:
    """
    Generates a list of overlapping bounding box tuples.
    
    Parameters:
    - min_lat, min_lon, max_lat, max_lon: Float coordinates of the main bounding box.
    - bin_size: The size of the grid cell in degrees (e.g., 0.5).
    - overlap_pct: Float representing overlap percentage (e.g., 0.25 for 25%).
    
    Returns:
    - A list of tuples: (lat_min, lon_min, lat_max, lon_max)
    """
    # Calculate how far to step for each new bin
    stride = bin_size * (1.0 - overlap_pct)
    
    def get_starts(start, end):
        starts = []
        current = start
        while current < end:
            starts.append(current)
            # Break early if the current bin fully covers the maximum boundary
            if current + bin_size >= end:
                break
            current += stride
        return starts

    # Generate 1D arrays of starting coordinates
    lat_starts = get_starts(min_lat, max_lat)
    lon_starts = get_starts(min_lon, max_lon)
    
    # Use itertools.product to efficiently generate the 2D grid combinations
    bboxes = [
        (lat, lon, lat + bin_size, lon + bin_size)
        for lat, lon in product(lat_starts, lon_starts)
    ]
    
    return bboxes

def plot_multiple_bboxes(bboxes_list, title="Grid Bounding Boxes"):
    
    # Extract tuples and create a list of shapely box geometries.
    # Note: shapely box takes (minx, miny, maxx, maxy) which is (lon, lat, lon, lat)
    geometries = [
        box(lon_min, lat_min, lon_max, lat_max) 
        for lat_min, lon_min, lat_max, lon_max in bboxes_list
    ]
    
    # Convert all geometries into a single GeoDataFrame (much faster than looping)
    gdf_bboxes = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    
    # Load world map
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 8)) # Slightly larger figure for clarity
    
    world.plot(ax=ax, color='lightgrey', edgecolor='white')
    
    # Plot the grid boxes
    # We use a thinner linewidth and alpha (transparency) so the overlapping edges don't become a solid red blob
    gdf_bboxes.boundary.plot(ax=ax, edgecolor='red', linewidth=0.5, alpha=0.6)
    
    # Zoom to the total bounds of ALL boxes combined
    buffer = 1
    minx, miny, maxx, maxy = gdf_bboxes.total_bounds
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)
    
    plt.title(f"{title} ({len(bboxes_list)} cells)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()
# --- Example Usage ---
# Gulf of Mexico coordinates
gulf_min_lat = 17.4068
gulf_min_lon = -98.0539
gulf_max_lat = 31.4648
gulf_max_lon = -80.4330

# 0.5 degree bins with a 20% overlap
grid_boxes = generate_overlapping_bboxes(
    min_lat=gulf_min_lat, 
    min_lon=gulf_min_lon, 
    max_lat=gulf_max_lat, 
    max_lon=gulf_max_lon, 
    bin_size=0.5, 
    overlap_pct=0.0
)

print(f"Generated {len(grid_boxes)} bounding boxes.")
print(f"First box: {grid_boxes[0]}")
print(f"Last box: {grid_boxes[-1]}")

plot_multiple_bboxes(grid_boxes, title="Gulf of Mexico - Overlapping Spatial Grid")