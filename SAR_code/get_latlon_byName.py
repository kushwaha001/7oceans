import asf_search as asf
from geopy.geocoders import Nominatim
import pandas as pd
import time

def get_sar_by_name(location_name: str, start_date: str, end_date: str):
    # 1. Ask OpenStreetMap for the coordinates of the name
    geolocator = Nominatim(user_agent="mda_sar_pipeline")
    location = geolocator.geocode(location_name)
    
    if not location:
        raise ValueError(f"Could not find coordinates for {location_name}")
        
    # 2. Extract the bounding box [lat_min, lat_max, lon_min, lon_max]
    bbox = location.raw['boundingbox'] # type:ignore
    min_lat, max_lat = bbox[0], bbox[1]
    min_lon, max_lon = bbox[2], bbox[3]
    
    print(f"Found '{location_name}' at Bounding Box: Lat({min_lat} to {max_lat}), Lon({min_lon} to {max_lon})")
    
    # 3. Create the WKT Polygon
    wkt_polygon = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    
    # 4. Search ASF
    print("Querying ASF Archive...")
    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,
        intersectsWith=wkt_polygon,
        start=start_date,
        end=end_date
    )
    
    print(f"Found {len(results)} SAR scenes for {location_name}.")
    return results


def get_location_name_from_scene(scene_name: str):
    print(f"1. Querying ASF for Scene: {scene_name}...")
    
    # Step 1: Ask ASF for the center coordinates
    results = asf.granule_search(scene_name)
    
    if len(results) == 0:
        return "Error: Scene not found in the ASF archive."
        
    scene = results[0]
    center_lat = scene.properties.get('centerLat')
    center_lon = scene.properties.get('centerLon')
    
    print(f"   -> Found Coordinates: Lat {center_lat}, Lon {center_lon}")
    
    # Step 2: Ask OpenStreetMap for the name of those coordinates
    print("2. Translating coordinates into a location name...")
    
    # We use Nominatim (OpenStreetMap's geocoder). 
    # Always provide a unique user_agent name so they don't block your script.
    geolocator = Nominatim(user_agent="mda_sar_pipeline_v1")
    
    try:
        # zoom=10 gets the region/sea/bay level. zoom=18 gets the exact street.
        location = geolocator.reverse((center_lat, center_lon), language='en', zoom=10) #type:ignore
        
        if location:
            print("\n✅ MATCH FOUND:")
            return location.address # type:ignore
        else:
            # If it's in the extreme deep ocean, OpenStreetMap might not have a name for it
            return "⚠️ Deep Open Ocean (No specific regional name found)"
            
    except Exception as e:
        return f"Geocoding Error: {e}"


# --- Example Usage ---
# scenes = get_sar_by_name("Bay of Biscay", "2022-01-01", "2022-01-31")

data_dir = r"D:\SAR-Intelligence\data_engine\Analysis_output\Same_footprint_Overlap_dataset.csv"
df = pd.read_csv(data_dir)
scene_col = list(df.columns)[0]
sceneNames = list(df[scene_col])

location_name1 = get_location_name_from_scene(sceneNames[0])
location_name2 = get_location_name_from_scene(sceneNames[1])
location_name3 = get_location_name_from_scene(sceneNames[2])
location_name4 = get_location_name_from_scene(sceneNames[3])
location_name5 = get_location_name_from_scene(sceneNames[4])

print(f"location 1 = {location_name1}")
print(f"location 2 = {location_name2}")
print(f"location 3 = {location_name3}")
print(f"location 4 = {location_name4}")
print(f"location 5 = {location_name5}")