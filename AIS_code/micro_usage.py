"""
MicroAnalysis — complete usage script (OPTIMIZED + AUTO-SPLIT)
===============================================================
Automatically splits data based on however many files exist in
your directory. Works with 1 month, 6 months, or any range.

Pipeline order (must never be changed):
    Step 0  →  clean_vessel_windows       (optional: filter invalid rows)
    Step 1  →  compute_micro_norm_stats   (training split only)
    Step 2  →  create_micro_tensors       (train / val / test separately)
    Step 3  →  validate_micro_tensors     (validation gate)
    Step 4  →  build_dataset_index        (pair micro ↔ macro)
    Step 5  →  visualisation spot-checks

Changes from original:
    - DYNAMIC SPLITTING: scans directory → splits by ratio (default 70/15/15)
    - DAILY BUNDLES: one .pt per day instead of 100k+ individual files
    - VECTORIZED FEATURES: _build_ping_features 50-100× faster
    - BUNDLE VALIDATION: validates windows inside each bundle
    - UPDATED PLOTTING: accepts bundle_index parameter
"""

import os
import torch
from .Micro_Analysis import MicroAnalysis
from .Splitter import auto_split

# =====================================================================
#  CONFIGURATION
# =====================================================================

# --- directory paths (adjust to your system) -------------------------
DATASETS_LOCATION   = r"D:\AIS_project\data\CSV_files\GulfMexico_AIS"
DEFAULT_CSV         = r"D:\AIS_project\data\CSV_files\GulfMexico_AIS\Gulf_of_Mexico_ais-2022-01-01.csv"
ANALYSIS_RESULT_DIR = r"D:\AIS_project\AnalysisMicroMacro_results"

# output directories
NORM_STATS_DIR      = r"D:\AIS_project\processed"
MICRO_TENSOR_DIR    = r"D:\AIS_project\processed\micro_tensors"
MACRO_TENSOR_DIR    = r"D:\AIS_project\processed\macro_tensors"
CLEANED_CSV_DIR     = r"D:\AIS_project\processed\cleaned_csvs"
DATASET_INDEX_DIR   = r"D:\AIS_project\processed\dataset_indices"

# ─────────────────────────────────────────────────────────────────────────────
#  SPLIT RATIO — change this to adjust train/val/test proportions
#  Default: 70% train, 15% val, 15% test
#  Examples:
#    (0.80, 0.10, 0.10) → 80/10/10
#    (0.70, 0.15, 0.15) → 70/15/15
#    (0.75, 0.125, 0.125) → 75/12.5/12.5
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_RATIOS = (0.70, 0.15, 0.15)


# =====================================================================
#  AUTO-SPLIT — scan CSV directory and split by ratio
#
#  This replaces the old hardcoded month ranges. It works with any
#  number of files: 31 (one month), 181 (six months), 365, etc.
#
#  The split is CHRONOLOGICAL — training always gets the earliest
#  dates, test always gets the latest.
# =====================================================================

print("=" * 70)
print("AUTO-SPLIT — Scanning CSV directory")
print("=" * 70)

splits = auto_split(
    data_dir  = DATASETS_LOCATION,
    extension = ".csv",
    ratios    = SPLIT_RATIOS,
)

TRAIN_STEMS = splits["train"]
VAL_STEMS   = splits["val"]
TEST_STEMS  = splits["test"]


# =====================================================================
#  INSTANTIATE
# =====================================================================

micro = MicroAnalysis(
    datasets_location=DATASETS_LOCATION,
    dataset_dir=DEFAULT_CSV,
    analysis_result_dir=ANALYSIS_RESULT_DIR,
)


# =====================================================================
#  STEP 0 — (Optional) Clean raw CSVs
# =====================================================================

print("=" * 70)
print("STEP 0: Cleaning raw CSVs")
print("=" * 70)

cleaned_files = micro.clean_vessel_windows(
    dataset_dir=DEFAULT_CSV,
    output_dir=CLEANED_CSV_DIR,
    for_group=True,
    create_new=True,
)
print(f"  Cleaned files: {len(cleaned_files)}")


# =====================================================================
#  STEP 1 — Compute norm stats (TRAINING DATA ONLY)
# =====================================================================

print("\n" + "=" * 70)
print("STEP 1: Computing micro normalisation stats (training only)")
print("=" * 70)

norm_stats = micro.compute_micro_norm_stats(
    output_dir=NORM_STATS_DIR,
    training_date_stems=TRAIN_STEMS,
    for_group=True,
)

NORM_STATS_PATH = os.path.join(NORM_STATS_DIR, "micro_norm_stats.json")
print(f"  Norm stats saved to: {NORM_STATS_PATH}")
print(f"  MEAN_LENGTH  : {norm_stats['MEAN_LENGTH']}")
print(f"  MEAN_DRAFT   : {norm_stats['MEAN_DRAFT']}")
print(f"  MAX_TYPE_CODE: {norm_stats['MAX_TYPE_CODE']}")


# =====================================================================
#  STEP 2a — Single-file test block
# =====================================================================

print("\n" + "=" * 70)
print("STEP 2a: Single-file test run")
print("=" * 70)

test_windows = micro.extract_vessel_windows(
    dataset_dir=DEFAULT_CSV,
    for_group=False,
)
print(f"  Windows extracted from single file: {len(test_windows)}")

if test_windows:
    w = test_windows[0]
    print(f"  First window: MMSI={w['mmsi']}  date={w['date']}  "
          f"hour={w['window_hour']}  pings={w['ping_count']}")

test_tensor_dir = os.path.join(MICRO_TENSOR_DIR, "_single_test")
test_created = micro.create_micro_tensors(
    output_dir=test_tensor_dir,
    norm_stats_path=NORM_STATS_PATH,
    for_group=False,
    dataset_dir=DEFAULT_CSV,
)
print(f"  Files created from single file: {len(test_created)}")

if test_created:
    test_result = micro.validate_micro_tensors(
        tensor_dir=test_tensor_dir,
        check_index_files=True,
    )
    assert test_result["failed_windows"] == 0, (
        f"Single-file test FAILED: {test_result['failed_windows']} windows had errors.\n"
        f"Errors: {test_result['errors']}"
    )
    print(f"  Single-file test PASSED — "
          f"{test_result['total_windows']} windows in "
          f"{test_result['total_bundles']} bundle(s), all valid.")


# =====================================================================
#  STEP 2b-d — Create tensors for all non-empty splits
# =====================================================================

tensor_files = {}

for split_name, stems in [("train", TRAIN_STEMS), ("val", VAL_STEMS), ("test", TEST_STEMS)]:
    if not stems:
        print(f"\n{'=' * 70}")
        print(f"STEP 2: {split_name.upper()} — SKIPPED (no files in this split)")
        print(f"{'=' * 70}")
        tensor_files[split_name] = []
        continue

    print(f"\n{'=' * 70}")
    print(f"STEP 2: Creating micro tensors — {split_name.upper()} ({len(stems)} days)")
    print(f"{'=' * 70}")

    split_tensor_dir = os.path.join(MICRO_TENSOR_DIR, split_name)

    files = micro.create_micro_tensors(
        output_dir=split_tensor_dir,
        norm_stats_path=NORM_STATS_PATH,
        for_group=True,
        date_stems=stems,
    )
    tensor_files[split_name] = files

    bundles = [f for f in files if f.endswith("_micro_bundle.pt")]
    print(f"  {split_name.capitalize()} bundles created: {len(bundles):,}")
    print(f"  Total files (bundles + indexes): {len(files):,}")


# =====================================================================
#  STEP 3 — VALIDATION GATE (must pass before training)
# =====================================================================

print("\n" + "=" * 70)
print("STEP 3: Validation gate — checking all splits")
print("=" * 70)

all_passed = True

for split_name, stems in [("train", TRAIN_STEMS), ("val", VAL_STEMS), ("test", TEST_STEMS)]:
    if not stems:
        continue

    split_tensor_dir = os.path.join(MICRO_TENSOR_DIR, split_name)
    print(f"\n  --- Validating {split_name} ---")

    result = micro.validate_micro_tensors(
        tensor_dir=split_tensor_dir,
        expected_feature_dim=11,
        value_min=-1.1,
        value_max=1.1,
        check_index_files=True,
    )

    print(f"  Bundles: {result['total_bundles']}  "
          f"Windows: {result['total_windows']}  "
          f"Passed: {result['passed_windows']}  "
          f"Failed: {result['failed_windows']}")

    if result["failed_windows"] > 0:
        print(f"  FAIL: {split_name} has {result['failed_windows']} bad windows!")
        all_passed = False
    if result["missing_bundles"]:
        print(f"  WARN: {split_name} has {len(result['missing_bundles'])} missing bundles")
        all_passed = False

if all_passed:
    print("\n  VALIDATION GATE PASSED — all splits are clean.")
else:
    raise RuntimeError(
        "VALIDATION GATE FAILED — fix the errors above before training."
    )


# =====================================================================
#  STEP 4 — Build dataset indices (pair micro ↔ macro)
# =====================================================================

print("\n" + "=" * 70)
print("STEP 4: Building dataset indices")
print("=" * 70)

for split_name, stems in [("train", TRAIN_STEMS), ("val", VAL_STEMS), ("test", TEST_STEMS)]:
    if not stems:
        print(f"  {split_name}: SKIPPED (no files)")
        continue

    micro_dir = os.path.join(MICRO_TENSOR_DIR, split_name)
    macro_dir = os.path.join(MACRO_TENSOR_DIR, split_name)

    index_path = micro.build_dataset_index(
        micro_tensor_dir=micro_dir,
        macro_tensor_dir=macro_dir,
        output_dir=DATASET_INDEX_DIR,
        split_name=split_name,
        date_stems=stems,
    )
    print(f"  {split_name} index → {index_path}")


# =====================================================================
#  STEP 5 — (Optional) Visualisation / spot-checks
# =====================================================================

print("\n" + "=" * 70)
print("STEP 5: Visualisation spot-checks")
print("=" * 70)

train_bundles = [f for f in tensor_files.get("train", []) if f.endswith("_micro_bundle.pt")]

if train_bundles:
    sample_bundle_path = train_bundles[0]

    micro.plot_vessel_trajectory(
        tensor_path=sample_bundle_path,
        output_file_name="sample_vessel_trajectory.png",
        bundle_index=0,
    )

    micro.plot_window_sog_profile(
        tensor_path=sample_bundle_path,
        output_file_name="sample_sog_profile.png",
        bundle_index=0,
    )

print("\n" + "=" * 70)
print("MICRO PREPROCESSING PIPELINE COMPLETE")
print(f"  Split: {splits['summary']['ratios']}  "
      f"({len(TRAIN_STEMS)} train / {len(VAL_STEMS)} val / {len(TEST_STEMS)} test)")
print("=" * 70)


# =====================================================================
#  APPENDIX — How to load bundles in a PyTorch Dataset
# =====================================================================
#
#   import json
#   import torch
#   from torch.utils.data import Dataset
#
#   class AISDataset(Dataset):
#       def __init__(self, index_path: str):
#           with open(index_path) as f:
#               index = json.load(f)
#           self.pairs = index["pairs"]
#           self._bundle_cache = {}
#
#       def __len__(self):
#           return len(self.pairs)
#
#       def _load_bundle(self, path: str):
#           if path not in self._bundle_cache:
#               self._bundle_cache[path] = torch.load(
#                   path, map_location="cpu", weights_only=False
#               )
#           return self._bundle_cache[path]
#
#       def __getitem__(self, idx):
#           p = self.pairs[idx]
#           macro = torch.load(p["macro"], map_location="cpu", weights_only=True)
#           bundle = self._load_bundle(p["micro_bundle"])
#           micro  = bundle["windows"][p["bundle_index"]]
#           return {
#               "macro":       macro,               # [N_cells, 10]
#               "micro":       micro,               # [N_pings, 11]
#               "mmsi":        p["mmsi"],
#               "date":        p["date"],
#               "window_hour": p["window_hour"],
#           }