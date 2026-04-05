# import polars as pl
# import os
# from tqdm import tqdm

# # ===============================================
# # Get number of lens
# # ===============================================

# # Get number of row in single column
# data_dir = r"D:\SAR-Intelligence\AIS_system\data\CSV_files\AIS_2022_01_01.csv"
# df = pl.scan_csv(data_dir)
# row_count = df.select(pl.len()).collect().item()
# print(row_count)

# column_name = df.columns
# print(column_name)

# # Get total number of row in each csv file and grand total
# dataset_location = r"D:\SAR-Intelligence\AIS_system\data\CSV_files"
# dir_lis = os.listdir(dataset_location)
# grand_total = 0
# obj = tqdm(dir_lis)

# show_each = False

# for file in obj:
#     file_path = os.path.join(dataset_location,file)
#     df = pl.scan_csv(file_path)
#     row_count = df.select(pl.len()).collect().item()
#     grand_total+=row_count
#     print(f"{file} = {row_count} rows") if show_each else ""

# print(f"total number of the row for 1 month {grand_total} row")

# df = pl.scan_csv(r"D:\SAR-Intelligence\AIS_system\data\CSV_files\*.csv")

# total = df.select(pl.len()).collect().item()

# print(total)


import polars as pl
from collections import OrderedDict

# Valid MID (Maritime Identification Digits) ranges by region — digits 2-4 of a vessel MMSI
VALID_MIDS = set(range(201, 776))  # Full ITU MID range

def _classify_mmsi(mmsi: int) -> str | None:
    """
    Classify a 9-digit MMSI into its ITU category.
    Returns the category name or None if invalid.
    """
    s = str(mmsi)

    if len(s) != 9:
        return None

    first = s[0]
    first2 = s[:2]
    first3 = s[:3]

    # --- Coastal Station / Port (MID + 00XXXX) ---
    # Format: 00MIDXXXX  → starts with "00"
    if first2 == "00":
        mid = int(s[2:5])
        if mid in VALID_MIDS:
            return "coast_station"
        return None

    # --- Group of ships (0MIDXXXXX) ---
    # Format: 0MID... → starts with "0" but not "00"
    if first == "0" and first2 != "00":
        mid = int(s[1:4])
        if mid in VALID_MIDS:
            return "group"
        return None

    # --- SAR Aircraft (111MIDXXX) ---
    if first3 == "111":
        mid = int(s[3:6])
        if mid in VALID_MIDS:
            return "sar_aircraft"
        return None

    # --- Aids to Navigation (99MIDXXXX) ---
    if first2 == "99":
        mid = int(s[2:5])
        if mid in VALID_MIDS:
            return "aids_to_navigation"
        return None

    # --- Search and Rescue Transponder / MOB / EPIRB (97XXXXX) ---
    if first2 == "97":
        return "sart_mob_epirb"

    # --- Man Overboard devices and similar (972XXXXXX) ---
    # Subset of 97X, already handled above under sart_mob_epirb

    # --- Handheld VHF (98MIDXXXX) ---
    if first2 == "98":
        mid = int(s[2:5])
        if mid in VALID_MIDS:
            return "handheld_vhf"
        return None

    # --- Craft associated with a parent ship (8XXXXXXXX) ---
    if first == "8":
        return "craft_associated"

    # --- Standard Vessel (MIDXXXXXX) ---
    # Format: MID (3 digits, 200-775) followed by 6 digits
    mid = int(first3)
    if mid in VALID_MIDS:
        return "vessel"

    return None  # Does not match any known category → invalid


def read_mmsi(path: str, group: bool = False) -> list[int] | OrderedDict[str, list[int]]:
    """
    Read and validate MMSI values from a large CSV file using Polars.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file containing an 'mmsi' column.
    group : bool, default False
        - False → returns a flat list of all valid MMSI integers.
        - True  → returns an OrderedDict mapping each ITU category name
                  to a list of valid MMSI integers in that category.
                  Categories with no MMSIs are still included (empty list).

    Returns
    -------
    list[int]
        All valid MMSI integers (when group=False).
    OrderedDict[str, list[int]]
        Category → list of valid MMSI integers (when group=True).

    Category Keys (when group=True)
    --------------------------------
    - "vessel"              : Standard ship/vessel (MID + 6 digits)
    - "coast_station"       : Coast station / port (00 + MID + 4 digits)
    - "group"               : Group of ships (0 + MID + 5 digits)
    - "sar_aircraft"        : SAR aircraft (111 + MID + 3 digits)
    - "aids_to_navigation"  : Aids to navigation (99 + MID + 4 digits)
    - "craft_associated"    : Craft assoc. with parent ship (8 + 8 digits)
    - "handheld_vhf"        : Handheld VHF radio (98 + MID + 4 digits)
    - "sart_mob_epirb"      : SART / MOB device / EPIRB (97 + 7 digits)
    """

    # All categories in ITU priority order
    ALL_CATEGORIES = [
        "vessel",
        "coast_station",
        "group",
        "sar_aircraft",
        "aids_to_navigation",
        "craft_associated",
        "handheld_vhf",
        "sart_mob_epirb",
    ]

    # Read only the mmsi column lazily, cast to string for digit checks
    lf = (
        pl.scan_csv(path)
        .select(pl.col("mmsi"))
        .filter(pl.col("mmsi").is_not_null())
        .with_columns(
            pl.col("mmsi").cast(pl.Int64)  # drop floats / bad rows early
        )
        .filter(
            (pl.col("mmsi") >= 1_000_000_00) &  # min 9-digit: 010000000
            (pl.col("mmsi") <= 999_999_999)      # max 9-digit
        )
    )

    # Collect to a plain Python list for classification
    raw: list[int] = lf.collect()["mmsi"].to_list()

    if not group:
        # Validate each MMSI and return as flat list
        return [mmsi for mmsi in raw if _classify_mmsi(mmsi) is not None]

    # Build ordered dict with empty lists for every category
    result: OrderedDict[str, list[int]] = OrderedDict(
        (cat, []) for cat in ALL_CATEGORIES
    )

    for mmsi in raw:
        category = _classify_mmsi(mmsi)
        if category is not None:
            result[category].append(mmsi)

    return result

print("-"*50)
data_dir = r"D:\AIS_project\AnalysisBrandNew_results\Unique_mmsi_all_file.csv"
df = pl.scan_csv(data_dir)
total_mmsi = df.select(pl.len()).collect().item()
print(f"Total number MMSI:{total_mmsi}")
print("-"*50)
print("\n")
print("-"*50)
lis = read_mmsi(data_dir)
print(f"Total valide MMSI:{len(lis)}")
print("-"*50)
print("\n")
print("-"*50)
dict_lis = read_mmsi(data_dir,group=True)
count = 0
for keys in dict_lis:
    num = len(dict_lis[keys]) # type:ignore
    print(f"{keys}:Valid mmsi found {len(dict_lis[keys])}") # type:ignore
    count += num

print(f"Total Count:{count}")
print("-"*50)
# \\wsl.localhost\buntu-24.04\home\water\cuopt_vrp_demo\demo_vrp.py