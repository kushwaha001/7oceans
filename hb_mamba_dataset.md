# HB-Mamba v2.0 — Dataset & DataLoader Documentation

`hb_mamba_dataset.py` — single-file module providing the full data pipeline
from on-disk tensors to model-ready batches.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Layout on Disk](#data-layout-on-disk)
3. [File Format Reference](#file-format-reference)
4. [Classes & Functions](#classes--functions)
   - [GapMaskConfig](#gapmaskconfig)
   - [HBMambaDataset](#hbmambadataset)
   - [hb_mamba_collate_fn](#hb_mamba_collate_fn)
   - [build_dataloaders](#build_dataloaders)
5. [Batch Dictionary Reference](#batch-dictionary-reference)
6. [Mask Conventions](#mask-conventions)
7. [Gap Masking Strategy](#gap-masking-strategy)
8. [Caching Behaviour](#caching-behaviour)
9. [Usage Examples](#usage-examples)
10. [Running the Smoke-Test](#running-the-smoke-test)
11. [Running the Usage Demo](#running-the-usage-demo)
12. [Observed Statistics (Gulf of Mexico, Jan 2022)](#observed-statistics)

---

## Overview

The module bridges the pre-processed AIS tensor files and the HB-Mamba model.
It handles:

- Loading macro (grid-level) and micro (per-vessel trajectory) tensors.
- Applying a random contiguous **gap mask** every `__getitem__` call so the
  model sees a different masking pattern each epoch.
- Variable-length sequence padding and the construction of a separate
  `padding_mask` so the model can ignore padded positions.
- Macro tensor caching so all vessels on the same date share a single loaded
  tensor.

---

## Data Layout on Disk

```
PROJECT_ROOT/     (= .../data_genration_and_raw_data/raw_data/)
├── tensors/
│   ├── macro/
│   │   ├── 2022-01-01_macro.pt
│   │   ├── 2022-01-02_macro.pt
│   │   └── ...   (one file per day)
│   └── micro/
│       ├── 2022-01-01_micro_bundle.pt
│       ├── 2022-01-01_micro_index.json
│       └── ...   (one bundle + index per day)
├── preprocessing/
│   └── norm_stats/
│       └── norm_stats.json
└── dataset_index/
    ├── train_dataset_index.json
    ├── val_dataset_index.json
    └── test_dataset_index.json
```

> **All tensors are already normalised.** The dataset loads and serves them
> as-is. No re-normalisation is applied.

---

## File Format Reference

### Macro tensor — `{date}_macro.pt`

Loaded with `torch.load(..., weights_only=False)`.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `features` | `[N_cells, 10]` | float32 | Per-cell feature vector (all grid cells present; empty cells = zeros) |
| `lat_idx` | `[N_cells]` | int64 | Latitude grid index `[0, N_LAT_STEPS)` |
| `lon_idx` | `[N_cells]` | int64 | Longitude grid index `[0, N_LON_STEPS)` |

`N_cells` is read from the tensor shape — never hardcoded.

---

### Micro bundle — `{date}_micro_bundle.pt`

Loaded with `torch.load(..., weights_only=False)`.

| Key | Type | Description |
|-----|------|-------------|
| `date` | str | Date string (e.g. `"2022-01-01"`) |
| `windows` | `list[Tensor]` | One `[N_day, 11]` float32 tensor per vessel |
| `metadata` | `list[dict]` | Per-vessel metadata (see below) |

Each metadata entry:
```json
{"mmsi": 205089000, "ping_count": 473, "bundle_index": 0, "date": "2022-01-01"}
```

`N_day` is the number of AIS pings for that vessel on that day (typically
40–200, minimum 10). It varies per vessel.

---

### Dataset index — `{split}_dataset_index.json`

```json
{
  "split": "train",
  "total_samples": 98782,
  "pairs": [
    {
      "date": "2022-01-01",
      "mmsi": 205089000,
      "bundle_index": 0,
      "ping_count": 473,
      "micro_bundle": "/abs/path/2022-01-01_micro_bundle.pt",
      "macro":        "/abs/path/2022-01-01_macro.pt"
    }
  ]
}
```

Each entry in `pairs` is one training sample (one vessel on one day).

---

### Norm stats — `norm_stats.json`

Relevant keys used by the dataset:

| Key | Example value | Description |
|-----|---------------|-------------|
| `N_LAT_STEPS` | 29 | Number of latitude grid bins |
| `N_LON_STEPS` | 36 | Number of longitude grid bins |
| `N_TOTAL_CELLS` | 1044 | Total cells = N_LAT_STEPS × N_LON_STEPS |
| `BIN_SIZE` | 0.5 | Grid resolution in degrees |

---

## Classes & Functions

### `GapMaskConfig`

```python
@dataclass
class GapMaskConfig:
    min_gap_frac: float = 0.05   # minimum gap = 5% of N_day
    max_gap_frac: float = 0.40   # maximum gap = 40% of N_day
    interp_prob:  float = 0.70   # P(interpolation gap) vs P(extrapolation)
    min_gap_len:  int   = 2      # never mask fewer than 2 pings
```

Controls the random gap sampling applied on every `__getitem__` call.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_gap_frac` | float | 0.05 | Minimum fraction of the sequence to mask |
| `max_gap_frac` | float | 0.40 | Maximum fraction of the sequence to mask |
| `interp_prob` | float | 0.70 | Probability the gap is placed in the interior (interpolation) rather than at the end (extrapolation) |
| `min_gap_len` | int | 2 | Hard lower bound on gap length in pings |

---

### `HBMambaDataset`

```python
HBMambaDataset(
    dataset_index_path: str,
    norm_stats_path:    str,
    split:              str,
    gap_config:         GapMaskConfig,
    cache_macro:        bool = True,
    cache_micro:        bool = False,
)
```

#### Constructor parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_index_path` | str | Absolute path to `{split}_dataset_index.json` |
| `norm_stats_path` | str | Absolute path to `norm_stats.json` |
| `split` | str | Split label (`"train"`, `"val"`, `"test"`) |
| `gap_config` | GapMaskConfig | Gap masking parameters |
| `cache_macro` | bool | Cache macro tensors by date (recommended `True`) |
| `cache_micro` | bool | Cache micro bundles by date (keep `False` for large datasets) |

#### Public attributes (read by the model)

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_lat_steps` | int | Number of latitude grid bins |
| `n_lon_steps` | int | Number of longitude grid bins |
| `n_total_cells` | int | Total grid cells |
| `bin_size` | float | Grid resolution in degrees |
| `n_samples` | int | Total number of samples in this split |
| `split` | str | Split name |

#### `__len__`

Returns `self.n_samples`.

#### `__getitem__(idx) -> dict`

Returns one sample dict. **A fresh random gap is sampled on every call**,
so the same `idx` will produce a different mask each time it is accessed.

Return dict keys:

| Key | Shape / Type | Description |
|-----|--------------|-------------|
| `macro_features` | `Tensor[N_cells, 10]` float32 | Daily grid snapshot |
| `macro_lat_idx` | `Tensor[N_cells]` int64 | Lat grid index per cell |
| `macro_lon_idx` | `Tensor[N_cells]` int64 | Lon grid index per cell |
| `micro_tokens` | `Tensor[N_day, 11]` float32 | Full (unmasked) trajectory |
| `mask` | `BoolTensor[N_day]` | `True` = ping to predict |
| `gap_type` | str | `"interpolation"` or `"extrapolation"` |
| `mmsi` | int | Vessel identifier |
| `date` | str | Date string |
| `n_day` | int | Number of pings (= sequence length before padding) |

> Padding is NOT done here. That is the collate function's responsibility.

---

### `hb_mamba_collate_fn`

```python
hb_mamba_collate_fn(batch: list[dict]) -> dict
```

Custom collate function for variable-length trajectories. Pads
`micro_tokens` and `mask` to `max_n_day` within the batch.

#### What it does

1. Finds `max_n_day = max(item["n_day"] for item in batch)`.
2. Zero-pads `micro_tokens` → `[B, max_n_day, 11]`.
3. False-pads `mask` → `[B, max_n_day]`.
4. Builds `padding_mask[i, j] = True` for all `j >= n_day[i]`.
5. Stacks macro tensors to `[B, N_cells, ...]`.
6. Asserts `(mask & padding_mask).any() == False`.

Pass to `DataLoader` via `collate_fn=hb_mamba_collate_fn`.

---

### `build_dataloaders`

```python
build_dataloaders(
    dataset_index_dir: str,
    norm_stats_path:   str,
    batch_size:        int = 64,
    num_workers:       int = 4,
    gap_config:        GapMaskConfig | None = None,
    cache_macro:       bool = True,
    cache_micro:       bool = False,
    pin_memory:        bool = True,
) -> dict[str, DataLoader]
```

Convenience factory. Scans `dataset_index_dir` for `*_dataset_index.json`
files, creates one `HBMambaDataset` and one `DataLoader` per split found.
Split names are derived from filenames — nothing is hardcoded.

#### DataLoader settings

| Split | `shuffle` | `drop_last` |
|-------|-----------|-------------|
| `train` | `True` | `True` |
| any other | `False` | `False` |

`drop_last=True` for training ensures a consistent batch size for
contrastive losses that depend on negative pair mining.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_index_dir` | — | Directory with `*_dataset_index.json` files |
| `norm_stats_path` | — | Path to `norm_stats.json` |
| `batch_size` | 64 | Samples per batch |
| `num_workers` | 4 | DataLoader worker processes |
| `gap_config` | `GapMaskConfig()` | Gap masking config |
| `cache_macro` | `True` | Cache macro tensors |
| `cache_micro` | `False` | Cache micro bundles |
| `pin_memory` | `True` | Pin host memory (set `False` when running without GPU) |

---

## Batch Dictionary Reference

After `hb_mamba_collate_fn`, a batch contains:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `macro_features` | `[B, N_cells, 10]` | float32 | Grid feature snapshots |
| `macro_lat_idx` | `[B, N_cells]` | int64 | Lat index for geo-embedding |
| `macro_lon_idx` | `[B, N_cells]` | int64 | Lon index for geo-embedding |
| `micro_tokens` | `[B, max_n_day, 11]` | float32 | Padded trajectory sequences |
| `mask` | `[B, max_n_day]` | bool | `True` = masked ping (predict) |
| `padding_mask` | `[B, max_n_day]` | bool | `True` = padded (ignore) |
| `n_day` | `[B]` | int64 | True sequence length per sample |
| `mmsi` | `list[int]` | — | Vessel IDs |
| `date` | `list[str]` | — | Date strings |
| `gap_type` | `list[str]` | — | `"interpolation"` or `"extrapolation"` |

---

## Mask Conventions

```
micro_tokens index:   0   1   2   3   4   5   6   7  ... n_day-1 | (padded)
mask:                 F   F   T   T   T   F   F   F  ...    F    |   F
padding_mask:         F   F   F   F   F   F   F   F  ...    F    |   T
```

- **`mask = True`** — this ping is **hidden from the encoder**; the model
  must reconstruct it. Used to compute the reconstruction loss.
- **`padding_mask = True`** — this position contains no real data (zero-filled
  padding). The model must **ignore** it entirely. This matches PyTorch's
  `key_padding_mask` convention (e.g. `nn.MultiheadAttention`).
- **Invariant**: `(mask & padding_mask).any() == False` — these two never
  overlap. A masked ping is always a real ping.

---

## Gap Masking Strategy

On every `__getitem__` call:

```
gap_len  = max(min_gap_len, int(N_day × Uniform(min_gap_frac, max_gap_frac)))
gap_len  = min(gap_len, N_day - 2)          # keep ≥ 2 visible pings

if Bernoulli(interp_prob):
    gap_type  = "interpolation"
    gap_start = randint(1, N_day - gap_len - 1)   # interior, never touches ends
else:
    gap_type  = "extrapolation"
    gap_start = N_day - gap_len                    # tail of the trajectory
```

**Pings are never re-ordered.** Timestamp order is preserved. Only the
DataLoader (training split) shuffles sample order.

Because the gap is re-sampled every call, the same vessel trajectory is
presented with a different masked segment on each epoch, acting as implicit
data augmentation.

---

## Caching Behaviour

### Macro cache (`cache_macro=True`, recommended)

Multiple vessels on the same date share the same daily macro tensor. Without
caching, the same `2022-01-01_macro.pt` would be loaded once per vessel per
epoch. With caching, it is loaded once and held in `self._macro_cache` (keyed
by date string).

Memory cost: one `[1044, 10]` float32 tensor per unique date ≈ 40 KB per day
(negligible).

### Micro cache (`cache_micro=False`, default)

Micro bundles are large (thousands of vessels per day). Keeping them all in
RAM across the dataset is expensive. Leave `cache_micro=False` unless you
have sufficient RAM and want maximum throughput.

---

## Usage Examples

### Minimal

```python
from hb_mamba_dataset import build_dataloaders

dataloaders = build_dataloaders(
    dataset_index_dir = "/home/hpc25/AIS/SAR_AIS_analysis/data_genration_and_raw_data/raw_data/dataset_index",
    norm_stats_path   = "/home/hpc25/AIS/SAR_AIS_analysis/data_genration_and_raw_data/raw_data/preprocessing/norm_stats/norm_stats.json",
)

for batch in dataloaders["train"]:
    macro  = batch["macro_features"]   # [64, 1044, 10]
    micro  = batch["micro_tokens"]     # [64, max_n_day, 11]
    mask   = batch["mask"]             # [64, max_n_day]  — predict here
    pmask  = batch["padding_mask"]     # [64, max_n_day]  — ignore here
    break
```

### Custom gap config

```python
from hb_mamba_dataset import build_dataloaders, GapMaskConfig

gap_cfg = GapMaskConfig(
    min_gap_frac = 0.10,
    max_gap_frac = 0.50,
    interp_prob  = 1.0,    # interpolation only
    min_gap_len  = 3,
)

dataloaders = build_dataloaders(
    dataset_index_dir = "...",
    norm_stats_path   = "...",
    batch_size        = 32,
    gap_config        = gap_cfg,
)
```

### Accessing dataset attributes for model config

```python
dataloaders = build_dataloaders(...)
ds = dataloaders["train"].dataset

model = HBMamba(
    n_lat_steps   = ds.n_lat_steps,    # 29
    n_lon_steps   = ds.n_lon_steps,    # 36
    n_total_cells = ds.n_total_cells,  # 1044
)
```

### Single-sample inspection

```python
ds    = dataloaders["train"].dataset
item  = ds[0]

print(item["micro_tokens"].shape)   # e.g. [473, 11]
print(item["mask"].sum())           # masked ping count
print(item["gap_type"])             # "interpolation" or "extrapolation"
```

---

## Running the Smoke-Test

The `if __name__ == "__main__":` block in `hb_mamba_dataset.py` accepts
`PROJECT_ROOT` as an optional command-line argument:

```bash
conda run -n hb_mamba python hb_mamba_dataset.py \
    /home/hpc25/AIS/SAR_AIS_analysis/data_genration_and_raw_data/raw_data
```

It iterates the first batch of each split and checks:

- Macro `N_cells` matches `n_total_cells` from norm_stats
- `lat_idx` and `lon_idx` are within grid bounds
- Micro feature dimension is 11
- Every sample has at least 1 masked ping
- `padding_mask` shape matches `mask`
- `mask & padding_mask` overlap is zero

Ends with `✓ Dataset smoke-test passed`.

---

## Running the Usage Demo

`dataset_usage.py` walks through nine demonstrations:

```bash
conda run -n hb_mamba python dataset_usage.py
```

| Section | What it shows |
|---------|---------------|
| 1 | `build_dataloaders()` — splits discovered, sample/batch counts |
| 2 | Dataset attributes read from `norm_stats.json` |
| 3 | Single sample shapes, value ranges, mask stats |
| 4 | Full batch tensor shapes and list fields |
| 5 | Gap-mask properties over 100 samples (contiguity, length bounds, type distribution) |
| 6 | `mask` / `padding_mask` non-overlap invariant, exact padding boundary |
| 7 | Macro cache hit behaviour (multi-vessel same date) |
| 8 | Custom `GapMaskConfig` (extrapolation-only, large gaps) |
| 9 | Throughput timing (100 batches, batch_size=32, num_workers=2) |

---

## Observed Statistics

Gulf of Mexico region, January 2022 (30 days):

| Split | Samples | Batches (bs=64) |
|-------|---------|-----------------|
| train | 98,782 | 1,543 |
| val | 18,618 | 291 |
| test | 22,488 | 351 |

Grid: 29 lat × 36 lon = **1,044 cells** at 0.5° resolution.

Micro trajectory lengths range from 10 to ~1,200 pings per vessel-day.
Macro feature vectors are in `[-1, 1]` (normalised). Micro token values
are in `[-1, 1]` (normalised).

Throughput on CPU (batch_size=32, num_workers=2): ~15 samples/s.
With GPU and `pin_memory=True`, transfer overhead is minimised.
