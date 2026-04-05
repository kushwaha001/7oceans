import asf_search as asf
import pandas as pd
from dateutil.parser import parse
from datetime import timedelta

product_dir = r"D:\SAR-Intelligence\data_engine\SAR_Data_csv_info\ESA_xView3_sceneName_mapping.csv"
product_df = pd.read_csv(product_dir)

esa_product_names = list(product_df[list(product_df.columns)[0]])
print(len(esa_product_names))

total_scene = len(esa_product_names)
count=0

for esa_name in esa_product_names:
    try:
        time_part = esa_name.split('_')[4]
        start_time = parse(time_part)
    except:
        print(f"Could not parse time from:{esa_name}")
        continue
    search_start = (start_time-timedelta(seconds=10)).isoformat()
    search_end = (start_time+timedelta(seconds=10)).isoformat()

    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,
        start=search_start,
        end=search_end
    )

    if len(results)>0:
        # print(f"Total Results:{len(results)}")
        asf_item = results[0]
        asf_item1 = results[1]
        asf_name = asf_item.properties['sceneName']
        asf_name1 = asf_item1.properties['sceneName']
        print(f"Match Found!!")
        print(f"Total Product found:{len(results)}")
        print(f"ESA Name:{esa_name}")
        print(f"ASF Name:{asf_name}")
        print(f"ASf Name1:{asf_name1}")
    else:
        print(f"No matching scene found in ASF for {esa_name}")
        count+=1
        total_scene-=1
print(f"Total scene found:{total_scene}")
print(f"Total scene not found:{count}")
