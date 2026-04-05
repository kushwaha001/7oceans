import pandas as pd
import numpy as np

data_dir = r"D:\SAR-Intelligence\s1_temporal_stacks.csv"
df = pd.read_csv(data_dir)
print(list(df.columns))
analysis_row = df[(df.frame == 145) & (df.path == 147)]
print(analysis_row)