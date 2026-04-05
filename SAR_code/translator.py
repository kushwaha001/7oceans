import asf_search as asf
from dateutil.parser import parse
from datetime import timedelta

# 1. The list of names you got from Copernicus Browser
esa_product_names = [
    "S1B_IW_GRDH_1SDV_20200407T185759_20200407T185824_021042_027EAA_107A",
    # Add more ESA names here...
]

print(f"Translating {len(esa_product_names)} ESA names to ASF format...")

for esa_name in esa_product_names:
    # A. Parse the Start Time from the string
    # The time is always after the 4th underscore (e.g., 20200407T185759)
    try:
        time_part = esa_name.split('_')[4] 
        start_time = parse(time_part)
    except:
        print(f"Could not parse time from: {esa_name}")
        continue

    # B. Define a small search window (Time +/- 10 seconds)
    # This accounts for slight clock differences between archives
    search_start = (start_time - timedelta(seconds=10)).isoformat()
    search_end = (start_time + timedelta(seconds=10)).isoformat()

    # C. Search ASF by TIME, not NAME
    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,
        start=search_start,
        end=search_end
    )

    # D. Match Found
    if len(results) > 0:
        asf_item = results[0]
        asf_name = asf_item.properties['sceneName']
        print(f"MATCH FOUND!")
        print(f"  ESA Name: {esa_name}")
        print(f"  ASF Name: {asf_name}")
        
        # Now you can use this 'asf_item' to find the stack!
        # stack = asf.stack_from_id(asf_name) <--- This will fail for GRD (see previous answer)
        # Use the geometry search method I gave you previously using 'asf_item.geometry'
    else:
        print(f"  No matching scene found in ASF for {esa_name}")
