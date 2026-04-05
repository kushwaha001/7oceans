import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
import matplotlib.patches as mpatches

# 1. Load and prepare the data
df_dir = r"D:\SAR-Intelligence\Overlap_analysis.csv"
df = pd.read_csv(df_dir)

lis_columns = list(df.columns)
date_col = lis_columns[1]  # Assuming index 1 is 'startTime'
geom_col = lis_columns[-1] # Assuming the last column is 'geometry'

# Convert WKT strings to shapely geometries
df[geom_col] = df[geom_col].apply(wkt.loads) #type:ignore

# Convert the time column to datetime (UTC) ONCE for the whole dataframe 
# This is more efficient than doing it inside the function every time
df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')

# Create the main GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs="EPSG:4326")

# 2. Extract the Base Patch (Row 0)
# We keep it as a GeoDataFrame (using [[0]]) so it's easy to plot
base_patch = gdf.iloc[[0]]
base_geom = base_patch.geometry.iloc[0]

print(f"Base Patch Scene: {base_patch.iloc[0]['sceneName']}")

# 3. Define your temporal filter function
def get_scene_for_date(date_str, input_gdf, date_column):
    start = pd.to_datetime(date_str, utc=True)
    end = start + pd.Timedelta(days=1)
    
    # Filter the GeoDataFrame based on the time window
    filtered_gdf = input_gdf[(input_gdf[date_column] >= start) & (input_gdf[date_column] < end)]
    
    # Optional: Exclude the base patch itself if it happens to fall on the same date
    filtered_gdf = filtered_gdf[filtered_gdf.index != base_patch.index[0]]
    
    return filtered_gdf

# 4. Filter for your specific day
target_date = '2020-01-10' 
filter_gdf = get_scene_for_date(target_date, gdf, date_col)
print(f"Found {len(filter_gdf)} patches for {target_date}")

# ---------------------------------------------------------
# 5. Visualization: The Overall Picture (Base vs. Filtered)
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 10))

# Plot the Base Patch solidly
base_patch.plot(ax=ax, color='lightgray', alpha=0.3, zorder=1)
base_patch.plot(ax=ax, color='none', edgecolor='black', linewidth=3, zorder=5)

# Setup the legend, starting with the base patch
base_name_short = base_patch.iloc[0]['sceneName'].split('_')[5] if '_' in base_patch.iloc[0]['sceneName'] else 'Base Patch'
legend_handles = [
    mpatches.Patch(facecolor='lightgray', edgecolor='black', linewidth=3, label=f"BASE: {base_name_short}")
]

# Set up colors for the daily patches
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Loop through the filtered daily patches to draw outlines and overlaps
for idx, (index, row) in enumerate(filter_gdf.iterrows()):
    color = colors[idx % len(colors)]
    other_geom = row[geom_col]
    
    # Temporarily isolate the row to plot it
    temp_gdf = gpd.GeoDataFrame({'geometry': [other_geom]}, crs="EPSG:4326")
    
    # Plot the full extent of the daily patch (dashed, overall picture)
    temp_gdf.plot(ax=ax, color='none', edgecolor=color, linestyle='--', linewidth=2, alpha=0.6, zorder=2)
    
    # Calculate overlap strictly with the base patch
    overlap = base_geom.intersection(other_geom)  #type:ignore
    
    if not overlap.is_empty:
        overlap_gdf = gpd.GeoDataFrame({'geometry': [overlap]}, crs="EPSG:4326")
        
        # Highlight where it intersects the base footprint
        overlap_gdf.plot(ax=ax, color=color, alpha=0.5, hatch='\\\\', zorder=3)
        
        # Build a clean label for the legend
        scene_name = row['sceneName']
        short_name = scene_name.split('_')[5] if '_' in scene_name else f"Patch {idx}"
        direction = row['direction'] if 'direction' in row else 'UNKNOWN'
        
        # Calculate actual overlap percentage dynamically
        overlap_pct = (overlap.area / base_geom.area) * 100 #type:ignore
         
        legend_handles.append(
            mpatches.Patch(
                facecolor=color, alpha=0.5, hatch='\\\\', edgecolor=color, 
                label=f"{short_name}\n({direction} | {overlap_pct:.1f}% cover)"
            )
        )

# Format the final map
plt.title(f'SAR Coverage on {target_date} vs Base Footprint', fontsize=16, pad=15)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

# Place legend outside the map so it doesn't block the large footprints
plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Satellite Passes")

plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()