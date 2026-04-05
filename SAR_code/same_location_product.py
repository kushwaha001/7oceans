import os
import numpy as np
import pandas as pd

product_dir = r"D:\SAR-Intelligence\data_engine\SAR_Data_csv_info\ESA_xView3_sceneName_mapping.csv"
location_dir = r"D:\SAR-Intelligence\data_engine\SAR_Data_csv_info\scene_split.csv"
output_dir = r"D:\SAR-Intelligence\data_engine\SAR_Data_csv_info\scene_location_split.csv"
product_df = pd.read_csv(product_dir)
location_df = pd.read_csv(location_dir)

location_product_df = location_df.copy()

lis_product = list(product_df.columns)      # (0: ESA_scenename, 2: xview_scenename)
lis_location = list(location_df.columns)    # (0: xview_scenename, 1: location)

print(location_product_df.head())

# Create mapping
mapping = dict(zip(product_df[lis_product[2]], product_df[lis_product[0]]))

# Apply mapping to the CORRECT dataframe
location_product_df['Esa_scenename'] = location_product_df[lis_location[0]].map(mapping)

print(location_product_df.head())
location_product_df.to_csv(output_dir)
print("DataFrame Saved Successfully" if os.path.exists(output_dir) else "Problem occured")