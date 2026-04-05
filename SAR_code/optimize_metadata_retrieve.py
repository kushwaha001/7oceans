import asf_search as asf
import pandas as pd

product_dir = r"D:\SAR-Intelligence\data_engine\SAR_Data_csv_info\ESA_xView3_sceneName_mapping.csv"
product_df = pd.read_csv(product_dir)

scene_names = list(product_df[list(product_df.columns)[0]])

print(f"Fetching metadata for {len(scene_names)} scenes...")

results =  asf.granule_search(scene_names)
print(f"MetaData retrieved for {len(results)} scenes")