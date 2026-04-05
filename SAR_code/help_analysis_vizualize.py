import ast
from shapely.geometry import shape
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df_dir = r"D:\SAR-Intelligence\data_engine\Analysis_output\Same_footprint_Overlap_dataset.csv"
df = pd.read_csv(df_dir)

# Identifying the columns
date_col = 'startTime'
geom_col = 'Location' # Using the specific column name

# Parsing logiv to handle the dictionary style string
def parse_geojson_geom(val):
    if isinstance(val,str):
        # converting the string representation of a dict into an actual dict
        dict_geom = ast.literal_eval(val)
        # converting the dict into a shapely geometry
        return shape(dict_geom)
    return val

df[geom_col] = df[geom_col].apply(parse_geojson_geom) # type:ignore
df[date_col] = pd.to_datetime(df[date_col],utc=True,errors='coerce')

print(df.head())

# Creating the main GeoDataFrame
gdf = gpd.GeoDataFrame(df,geometry=geom_col,crs="EPSG:4326")

# Extracting the base Patch (row 0)
# We keep it as a GeoDataframe so its easy to plot
base_index_number = 20
base_patch = gdf.iloc[[base_index_number]]
base_geom = base_patch.geometry.iloc[0]
print(f"Base Patch Scene:{base_patch.iloc[0]['sceneName']}")

# Defining the temporal filter function
def get_scene_for_date(date_str,input_gdf,date_column):
    start = pd.to_datetime(date_str,utc=True)
    end = start + pd.Timedelta(days=1)

    # Filter the GeoDataFrame based on the time window
    filtered_gdf = input_gdf[(input_gdf[date_column]>=start) & (input_gdf[date_column]<end)]

    # Optional:Excude the base patch itself if it happens to fall on the same date
    filtered_gdf = filtered_gdf[filtered_gdf.index != base_patch.index[0]]

    return filtered_gdf

# filter for specific day
target_date = '2020-01-09'
filter_gdf = get_scene_for_date(target_date,gdf,date_col)
print(f"Found {len(filter_gdf)} patches for {target_date}")

# -------------------------------------------------------------------------------
# Visualizaling the overall patches
# -------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14,10))

# Plot the base patch solidly
base_patch.plot(ax=ax,color='lightgray',alpha=0.3,zorder=1)
base_patch.plot(ax=ax,color='none',edgecolor='black',linewidth=3,zorder=5)

# Setup the lengend,starting with the base patch
base_name_short = base_patch.iloc[0]['sceneName'].split('_')[5] if '_' in base_patch.iloc[0]['sceneName'] else 'Base Patch'
legend_handles = [
    mpatches.Patch(facecolor='lightgray',edgecolor='black',linewidth=3,label=f"Base:{base_name_short}")
]

# Setting up colors for daily patches
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Looping through the filtered daily patches to draw outlines and overlaps
for idx,(index,row) in enumerate(filter_gdf.iterrows()):
    color = colors[idx % len(colors)]
    other_geom = row[geom_col]

    # Temporarily isolate the row to plot it
    temp_gdf = gpd.GeoDataFrame({'geometry':[other_geom]},crs="EPSG:4326")

    # Plot the full extent of the dail patch
    temp_gdf.plot(ax=ax,color = 'none',edgecolor=color,linestyle='--',linewidth=2,alpha=0.6,zorder=2)

    # Calculate overlap strictly with the base patch
    overlap = base_geom.intersection(other_geom) # type:ignore

    if not overlap.is_empty:
        overlap_gdf = gpd.GeoDataFrame({'geometry':[overlap]},crs="EPSG:4326")

        # Highlight where it intersects the base footprint
        overlap_gdf.plot(ax=ax, color=color, alpha=0.5, hatch='\\\\', zorder=3)
        
        # Build a clean label for the legend
        scene_name = row['sceneName']
        short_name = scene_name.split('_')[5] if '_' in scene_name else f"Patch {idx}"
        
        # Changed 'direction' to 'flightDirection' to match your CSV
        direction = row['flightDirection'] if 'flightDirection' in row else 'UNKNOWN'
        
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
