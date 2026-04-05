import polars as pl
import os

data_dir1 = r"D:\AIS_project\data\CSV_files\AIS_2022_01_01.csv"
data_dir2 = r"D:\AIS_project\data\CSV_files\ais-2022-01-01.csv"

df1 = pl.scan_csv(data_dir1)
df2 = pl.scan_csv(data_dir2)

count1 = df1.select(pl.len()).collect().item()
count2 = df2.select(pl.len()).collect().item()

print(count1)
print(count2)