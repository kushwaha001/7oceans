"""
MacroAnalysis — complete usage script (OPTIMIZED + AUTO-SPLIT)
===============================================================
Automatically splits data based on however many files exist in
your directory. Works with 1 month, 6 months, or any range.

Pipeline order (must never be changed):
    Step 0  →  create_parquet_json        (1 CSV read/day — 100-800× faster)
    Step 1  →  compute_norm_stats         (training split only)
    Step 2  →  normalise_and_save_tensors (train / val / test separately)
    Step 3  →  validate_macro_tensors     (run after each split)
"""

import os
from .Macro_Analysis import MacroAnalysis
from .Splitter import auto_split, auto_split_parquet

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  —  change these to match your machine
# ─────────────────────────────────────────────────────────────────────────────

# Folder that contains all daily Gulf-of-Mexico CSVs
DATASETS_LOCATION = r"D:\AIS_project\data\CSV_files\GulfMexico_AIS"

# A single CSV used as the default when for_group=False
DATASET_DIR = r"D:\AIS_project\data\CSV_files\GulfMexico_AIS\Gulf_of_Mexico_ais-2022-01-01.csv"

# Where Analysis base-class results go
ANALYSIS_RESULT_DIR = r"D:\AIS_project\AnalysisMicroMacro_results"

# Where create_parquet_json saves the *_macro_raw.parquet files
RAW_PARQUET_DIR = r"D:\AIS_project\processed\macro_raw"

# Where norm_stats.json will be written (and later read from)
STATS_DIR = r"D:\AIS_project\processed"

# Where the final normalised .pt tensor files will be saved
TENSOR_DIR = r"D:\AIS_project\processed\macro_tensors"

# ─────────────────────────────────────────────────────────────────────────────
#  SPLIT RATIO — change this to adjust train/val/test proportions
#  Default: 70% train, 15% val, 15% test
#  Examples:
#    (0.80, 0.10, 0.10) → 80/10/10
#    (0.70, 0.15, 0.15) → 70/15/15
#    (0.75, 0.125, 0.125) → 75/12.5/12.5
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_RATIOS = (0.70, 0.15, 0.15)


# ─────────────────────────────────────────────────────────────────────────────
#  INSTANTIATE
# ─────────────────────────────────────────────────────────────────────────────

macro = MacroAnalysis(
    datasets_location   = DATASETS_LOCATION,
    dataset_dir         = DATASET_DIR,
    analysis_result_dir = ANALYSIS_RESULT_DIR,
    min_lat  = 17.4068,
    min_lon  = -98.0539,
    max_lat  = 31.4648,
    max_lon  = -80.4330,
    bin_size = 0.5,
    overlap_pct = 0.0,
)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 — create_parquet_json
#  Skip if raw parquets already exist.
# ─────────────────────────────────────────────────────────────────────────────

RUN_STEP_0 = True   # set True to regenerate raw parquets from scratch

if RUN_STEP_0:
    print("=" * 60)
    print("STEP 0 — Creating raw macro parquet files (OPTIMIZED)")
    print("=" * 60)

    created = macro.create_parquet_json(
        output_dir  = RAW_PARQUET_DIR,
        for_group   = True,
        dataset_dir = None,
    )
    print(f"\nStep 0 done — {len(created)} files created.")


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-SPLIT — scan RAW_PARQUET_DIR and split by ratio
#
#  This replaces the old hardcoded month ranges. It works with any number
#  of files: 31 (one month), 181 (six months), 365 (full year), etc.
#
#  The split is CHRONOLOGICAL — training gets the earliest dates, test
#  gets the latest. This prevents future data leaking into training.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("AUTO-SPLIT — Scanning parquet directory")
print("=" * 60)

splits = auto_split_parquet(
    parquet_dir = RAW_PARQUET_DIR,
    ratios      = SPLIT_RATIOS,
)

TRAIN_STEMS = splits["train"]
VAL_STEMS   = splits["val"]
TEST_STEMS  = splits["test"]


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — compute_norm_stats (training split only)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Computing normalisation statistics (training only)")
print("=" * 60)

stats = macro.compute_norm_stats(
    raw_parquet_dir      = RAW_PARQUET_DIR,
    output_dir           = STATS_DIR,
    training_date_stems  = TRAIN_STEMS,
)

print(f"\nStats returned: {stats}")
NORM_STATS_PATH = os.path.join(STATS_DIR, "norm_stats.json")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — normalise_and_save_tensors (all non-empty splits)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 2 — Normalising and saving tensors")
print("=" * 60)

tensor_results = {}

for split_name, stems in [("train", TRAIN_STEMS), ("val", VAL_STEMS), ("test", TEST_STEMS)]:
    if not stems:
        print(f"\n--- {split_name} split: SKIPPED (no files) ---")
        tensor_results[split_name] = []
        continue

    print(f"\n--- {split_name} split ({len(stems)} files) ---")
    split_tensor_dir = os.path.join(TENSOR_DIR, split_name)

    tensors = macro.normalise_and_save_tensors(
        raw_parquet_dir    = RAW_PARQUET_DIR,
        tensor_output_dir  = split_tensor_dir,
        norm_stats_path    = NORM_STATS_PATH,
        date_stems         = stems,
    )
    tensor_results[split_name] = tensors
    print(f"{split_name.capitalize()} tensors saved: {len(tensors)}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — validate_macro_tensors (all non-empty splits)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 3 — Validating tensors")
print("=" * 60)

validation_results = {}

for split_name, stems in [("train", TRAIN_STEMS), ("val", VAL_STEMS), ("test", TEST_STEMS)]:
    if not stems:
        validation_results[split_name] = {"failed": 0, "missing_dates": []}
        continue

    split_tensor_dir = os.path.join(TENSOR_DIR, split_name)
    print(f"\n--- Validating {split_name} split ---")

    result = macro.validate_macro_tensors(
        tensor_dir           = split_tensor_dir,
        expected_date_stems  = stems,
        expected_feature_dim = 10,
        value_min            = -1.1,
        value_max            =  1.1,
    )
    validation_results[split_name] = result


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL GATE
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("FINAL VALIDATION GATE")
print("=" * 60)

all_passed = True
for split_name, result in validation_results.items():
    if result["failed"] > 0:
        print(f"  FAIL: {split_name} has {result['failed']} file(s) failed")
        all_passed = False
    if result.get("missing_dates"):
        print(f"  WARN: {split_name} missing dates: {result['missing_dates']}")
        all_passed = False

if all_passed:
    print("\nAll splits passed validation.")
    print("Macro preprocessing is complete — safe to proceed to training.")
    print(f"\nSplit summary:")
    print(f"  Train : {len(TRAIN_STEMS):>4} days  →  {os.path.join(TENSOR_DIR, 'train')}")
    print(f"  Val   : {len(VAL_STEMS):>4} days  →  {os.path.join(TENSOR_DIR, 'val')}")
    print(f"  Test  : {len(TEST_STEMS):>4} days  →  {os.path.join(TENSOR_DIR, 'test')}")
    print(f"\nNorm stats : {NORM_STATS_PATH}")
else:
    print("\nValidation FAILED — do NOT start training.")


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK SINGLE-FILE TEST  (useful during development)
# ─────────────────────────────────────────────────────────────────────────────

RUN_SINGLE_FILE_TEST = False

if RUN_SINGLE_FILE_TEST:
    print("=" * 60)
    print("SINGLE FILE TEST")
    print("=" * 60)

    SINGLE_CSV        = DATASET_DIR
    SINGLE_RAW_DIR    = r"D:\AIS_project\processed\test_run\raw"
    SINGLE_TENSOR_DIR = r"D:\AIS_project\processed\test_run\tensors"
    SINGLE_STATS_DIR  = r"D:\AIS_project\processed\test_run"

    macro.create_parquet_json(output_dir=SINGLE_RAW_DIR, for_group=False, dataset_dir=SINGLE_CSV)
    macro.compute_norm_stats(raw_parquet_dir=SINGLE_RAW_DIR, output_dir=SINGLE_STATS_DIR, training_date_stems=None)
    macro.normalise_and_save_tensors(
        raw_parquet_dir=SINGLE_RAW_DIR,
        tensor_output_dir=SINGLE_TENSOR_DIR,
        norm_stats_path=os.path.join(SINGLE_STATS_DIR, "norm_stats.json"),
        date_stems=None,
    )
    result = macro.validate_macro_tensors(tensor_dir=SINGLE_TENSOR_DIR, expected_date_stems=None, expected_feature_dim=10)
    print(f"\nSingle file test result: {result}")