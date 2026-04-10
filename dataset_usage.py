"""
dataset_usage.py — HB-Mamba v2.0 Dataset & DataLoader Usage Demonstration
==========================================================================

Run from the project root:
    conda run -n hb_mamba python dataset_usage.py

Demonstrates:
    1.  Building dataloaders with build_dataloaders()
    2.  Inspecting dataset attributes (grid dims, sample counts)
    3.  Iterating a single sample via __getitem__
    4.  Iterating a full batch and reading every field
    5.  Verifying the gap-mask properties (length, type distribution)
    6.  Verifying the padding-mask / mask non-overlap invariant
    7.  Verifying macro cache hit behaviour
    8.  Custom GapMaskConfig usage
    9.  Timing one epoch of the train loader

    """

import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path("/home/hpc25/AIS/SAR_AIS_analysis/data_genration_and_raw_data/raw_data")
DATASET_INDEX_DIR = PROJECT_ROOT / "dataset_index"
NORM_STATS_PATH   = PROJECT_ROOT / "preprocessing" / "norm_stats" / "norm_stats.json"

# Add project root to sys.path so the import below always works regardless of
# the directory from which this script is invoked.
sys.path.insert(0, str(Path(__file__).parent))

from hb_mamba_dataset import (
    GapMaskConfig,
    HBMambaDataset,
    build_dataloaders,
    hb_mamba_collate_fn,
)

SEP  = "=" * 70
SEP2 = "-" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — build_dataloaders
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("SECTION 1 — build_dataloaders()")
print(SEP)

dataloaders = build_dataloaders(
    dataset_index_dir = str(DATASET_INDEX_DIR),
    norm_stats_path   = str(NORM_STATS_PATH),
    batch_size        = 16,
    num_workers       = 0,       # 0 keeps output clean in a demo script
    pin_memory        = False,
)

print(f"Splits found : {list(dataloaders.keys())}")
for split, loader in dataloaders.items():
    ds = loader.dataset
    print(f"  {split:6s}  samples={ds.n_samples:>6d}  batches={len(loader):>5d}  "
          f"shuffle={'yes' if split=='train' else 'no ':3s}  "
          f"drop_last={'yes' if split=='train' else 'no '}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Dataset attributes (grid dimensions from norm_stats)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 2 — Dataset attributes")
print(SEP)

train_ds: HBMambaDataset = dataloaders["train"].dataset
print(f"n_lat_steps   : {train_ds.n_lat_steps}")
print(f"n_lon_steps   : {train_ds.n_lon_steps}")
print(f"n_total_cells : {train_ds.n_total_cells}  (= {train_ds.n_lat_steps} × {train_ds.n_lon_steps})")
print(f"bin_size      : {train_ds.bin_size} degrees")
print(f"split         : {train_ds.split}")
print(f"cache_macro   : {train_ds.cache_macro}")
print(f"cache_micro   : {train_ds.cache_micro}")
print(f"gap_config    : {train_ds.gap_config}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Single-sample inspection via __getitem__
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 3 — Single sample  (train_ds[0])")
print(SEP)

sample = train_ds[0]
for key, val in sample.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key:20s} shape={tuple(val.shape)}  dtype={val.dtype}")
    else:
        print(f"  {key:20s} = {val!r}")

print(SEP2)
print(f"  Masked pings   : {sample['mask'].sum().item()} / {sample['n_day']}")
print(f"  Gap type       : {sample['gap_type']}")
print(f"  macro lat_idx range : [{sample['macro_lat_idx'].min().item()}, {sample['macro_lat_idx'].max().item()}]")
print(f"  macro lon_idx range : [{sample['macro_lon_idx'].min().item()}, {sample['macro_lon_idx'].max().item()}]")
print(f"  micro value range   : [{sample['micro_tokens'].min():.4f}, {sample['micro_tokens'].max():.4f}]")
print(f"  macro value range   : [{sample['macro_features'].min():.4f}, {sample['macro_features'].max():.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Full batch field-by-field walkthrough
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 4 — First training batch")
print(SEP)

train_loader = dataloaders["train"]
batch = next(iter(train_loader))

print("Tensor fields:")
for key in ("macro_features", "macro_lat_idx", "macro_lon_idx",
            "micro_tokens", "mask", "padding_mask", "n_day"):
    t = batch[key]
    print(f"  {key:20s}  shape={tuple(t.shape)}  dtype={t.dtype}")

print("\nList fields:")
print(f"  {'mmsi':20s} = {batch['mmsi']}")
print(f"  {'date':20s} = {batch['date']}")
print(f"  {'gap_type':20s} = {batch['gap_type']}")

print(SEP2)
n_day_vec = batch["n_day"]
print(f"  n_day per sample : {n_day_vec.tolist()}")
print(f"  max_n_day (padded to) : {batch['micro_tokens'].shape[1]}")
print(f"  masked pings per sample : {batch['mask'].sum(dim=1).tolist()}")

# Padding stats
pad_counts = batch["padding_mask"].sum(dim=1).tolist()
print(f"  padded positions per sample : {[int(p) for p in pad_counts]}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Gap-mask property verification (100 samples)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 5 — Gap-mask properties over 100 samples")
print(SEP)

gap_types   = Counter()
gap_lengths = []
cfg = train_ds.gap_config

for i in range(100):
    s = train_ds[i]
    gap_lengths.append(int(s["mask"].sum()))
    gap_types[s["gap_type"]] += 1

    # Each call must produce a contiguous masked region
    indices = s["mask"].nonzero(as_tuple=True)[0]
    if len(indices) > 1:
        diffs = (indices[1:] - indices[:-1])
        assert (diffs == 1).all(), f"Non-contiguous gap at sample {i}"

    # Gap must be within [min_gap_len, n_day - 2] masked pings
    assert int(s["mask"].sum()) >= cfg.min_gap_len, \
        f"Gap too short at sample {i}: {int(s['mask'].sum())}"
    assert int(s["mask"].sum()) <= s["n_day"] - 2, \
        f"Gap leaves fewer than 2 visible pings at sample {i}"

avg_gap = sum(gap_lengths) / len(gap_lengths)
print(f"  Samples checked : 100")
print(f"  Gap type counts : {dict(gap_types)}")
print(f"  Avg masked pings: {avg_gap:.1f}")
print(f"  Min masked pings: {min(gap_lengths)}")
print(f"  Max masked pings: {max(gap_lengths)}")
print(f"  All gaps contiguous          : yes")
print(f"  All gaps >= min_gap_len ({cfg.min_gap_len}) : yes")
print(f"  All gaps leave >= 2 visible  : yes")

# Verify fresh gap per call (same idx called twice → different masks with high prob)
masks_differ = sum(
    1 for i in range(50)
    if not torch.equal(train_ds[i]["mask"], train_ds[i]["mask"])
    # call twice for same idx — they are independent random draws
)
# Re-draw properly: call twice and compare
differ_count = 0
for i in range(50):
    m1 = train_ds[i]["mask"]
    m2 = train_ds[i]["mask"]
    if not torch.equal(m1, m2):
        differ_count += 1
print(f"  Fresh gap per call (differ/50 same-idx calls): {differ_count}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — mask / padding_mask non-overlap invariant (full first batch)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 6 — mask & padding_mask non-overlap invariant")
print(SEP)

overlap = (batch["mask"] & batch["padding_mask"]).sum().item()
print(f"  Overlap positions in batch : {overlap}  (must be 0)")
assert overlap == 0, "FAIL: mask and padding_mask overlap!"

# Show that padding starts exactly at n_day for every sample
for i in range(len(batch["n_day"])):
    nd = int(batch["n_day"][i])
    real_region    = ~batch["padding_mask"][i, :nd]
    padding_region =  batch["padding_mask"][i, nd:]
    assert real_region.all(),    f"Sample {i}: padding_mask=True inside real region"
    assert padding_region.all(), f"Sample {i}: padding_mask=False inside padded region"
print(f"  Padding boundary exact for all {len(batch['n_day'])} samples : yes")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Macro cache hit behaviour
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 7 — Macro cache hit behaviour")
print(SEP)

fresh_ds = HBMambaDataset(
    dataset_index_path = str(DATASET_INDEX_DIR / "train_dataset_index.json"),
    norm_stats_path    = str(NORM_STATS_PATH),
    split              = "train",
    gap_config         = GapMaskConfig(),
    cache_macro        = True,
    cache_micro        = False,
)

print(f"  Macro cache size before any access : {len(fresh_ds._macro_cache)}")
_ = fresh_ds[0]
_ = fresh_ds[1]   # same date as idx 0 in the index (2022-01-01) — should hit cache
print(f"  Macro cache size after 2 accesses  : {len(fresh_ds._macro_cache)}")
# Both idx 0 and 1 belong to 2022-01-01, so only one entry should be cached
dates_accessed = {fresh_ds.pairs[0]["date"], fresh_ds.pairs[1]["date"]}
expected_cache = len(dates_accessed)
print(f"  Distinct dates accessed            : {dates_accessed}")
print(f"  Expected cache entries             : {expected_cache}")
assert len(fresh_ds._macro_cache) == expected_cache, "Cache size mismatch"
print(f"  Cache size matches                 : yes")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Custom GapMaskConfig
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 8 — Custom GapMaskConfig (extrapolation-only, large gaps)")
print(SEP)

extrap_cfg = GapMaskConfig(
    min_gap_frac = 0.20,
    max_gap_frac = 0.50,
    interp_prob  = 0.0,    # always extrapolation
    min_gap_len  = 5,
)

extrap_loaders = build_dataloaders(
    dataset_index_dir = str(DATASET_INDEX_DIR),
    norm_stats_path   = str(NORM_STATS_PATH),
    batch_size        = 8,
    num_workers       = 0,
    gap_config        = extrap_cfg,
    pin_memory        = False,
)

extrap_batch = next(iter(extrap_loaders["train"]))
gap_type_counts = Counter(extrap_batch["gap_type"])
print(f"  gap_config      : {extrap_cfg}")
print(f"  gap_type counts : {dict(gap_type_counts)}")
assert all(g == "extrapolation" for g in extrap_batch["gap_type"]), \
    "Expected all extrapolation gaps"
print(f"  All gaps extrapolation : yes")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Epoch timing (train split, batch_size=64, num_workers=4)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 9 — Epoch timing (100 batches, batch_size=32, num_workers=2)")
print(SEP)

timed_loaders = build_dataloaders(
    dataset_index_dir = str(DATASET_INDEX_DIR),
    norm_stats_path   = str(NORM_STATS_PATH),
    batch_size        = 32,
    num_workers       = 2,
    pin_memory        = False,
)

timed_train = timed_loaders["train"]
N_BATCHES = 100
t0 = time.perf_counter()
for step, b in enumerate(timed_train):
    if step + 1 >= N_BATCHES:
        break
elapsed = time.perf_counter() - t0
samples_done = N_BATCHES * 32

print(f"  Batches timed       : {N_BATCHES}")
print(f"  Samples processed   : {samples_done}")
print(f"  Wall time           : {elapsed:.2f}s")
print(f"  Throughput          : {samples_done / elapsed:.0f} samples/s")

# ─────────────────────────────────────────────────────────────────────────────
# Section 9b — Bottleneck diagnosis
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 9b — Bottleneck diagnosis")
print(SEP)

import time

# Test 1: pure __getitem__ speed (no DataLoader overhead, no workers)
t0 = time.perf_counter()
for i in range(200):
    _ = train_ds[i]
getitem_time = time.perf_counter() - t0
print(f"  Pure __getitem__ (200 samples, cache_micro=False) : {getitem_time:.2f}s  →  {200/getitem_time:.0f} samples/s")

# Test 2: same but with cache_micro=True
cached_micro_ds = HBMambaDataset(
    dataset_index_path = str(DATASET_INDEX_DIR / "train_dataset_index.json"),
    norm_stats_path    = str(NORM_STATS_PATH),
    split              = "train",
    gap_config         = GapMaskConfig(),
    cache_macro        = True,
    cache_micro        = True,   # <── difference
)
t0 = time.perf_counter()
for i in range(200):
    _ = cached_micro_ds[i]
cached_time = time.perf_counter() - t0
print(f"  Pure __getitem__ (200 samples, cache_micro=True)  : {cached_time:.2f}s  →  {200/cached_time:.0f} samples/s")

# Test 3: how many unique micro bundle files are accessed in 200 samples?
bundle_paths = {train_ds.pairs[i]["micro_bundle"] for i in range(200)}
print(f"  Unique micro bundles in first 200 samples : {len(bundle_paths)}")

# Test 4: how large is one micro bundle file on disk?
import os
bundle_sizes = [os.path.getsize(p) / 1024 / 1024 for p in list(bundle_paths)[:5]]
print(f"  First 5 micro bundle sizes (MB): {[f'{s:.1f}' for s in bundle_sizes]}")

# Test 5: raw torch.load speed for one bundle
sample_bundle_path = train_ds.pairs[0]["micro_bundle"]
t0 = time.perf_counter()
for _ in range(20):
    _ = torch.load(sample_bundle_path, map_location="cpu", weights_only=False)
load_time = (time.perf_counter() - t0) / 20
print(f"  Raw torch.load for one micro bundle (avg 20 runs) : {load_time*1000:.1f}ms")

# Test 6: DataLoader with 0 workers vs 4 workers
for nw in [0, 2, 4, 8]:
    tl = build_dataloaders(
        dataset_index_dir = str(DATASET_INDEX_DIR),
        norm_stats_path   = str(NORM_STATS_PATH),
        batch_size        = 32,
        num_workers       = nw,
        pin_memory        = False,
    )["train"]
    t0 = time.perf_counter()
    for step, b in enumerate(tl):
        if step + 1 >= 50:
            break
    elapsed = time.perf_counter() - t0
    print(f"  num_workers={nw:2d}  50 batches × 32  →  {50*32/elapsed:.0f} samples/s")
# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("ALL SECTIONS COMPLETE — dataset_usage.py finished successfully.")
print(SEP)
