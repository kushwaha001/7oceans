# Claude Code Prompt: HB-Mamba v2.0 — Dataset & DataLoader Implementation

---

## CONTEXT

You are implementing the `Dataset` and `DataLoader` for the **HB-Mamba v2.0** architecture — a Hierarchical Bidirectional Mamba model for AIS vessel trajectory learning. Read every specification below before writing any code.

---

## PROJECT FOLDER STRUCTURE (already on disk — do NOT create these)

```
PROJECT_ROOT/
├── tensors/
│   ├── macro/
│   │   ├── 2022-01-01_macro.pt
│   │   └── ...
│   └── micro/
│       ├── 2022-01-01_micro_bundle.pt
│       ├── 2022-01-01_micro_index.json
│       └── ...
├── preprocessing/
│   └── norm_stats/
│       └── norm_stats.json
└── dataset_index/
    ├── train_dataset_index.json
    ├── val_dataset_index.json
    └── test_dataset_index.json
```

---

## FILE FORMAT SPECIFICATIONS

### 1. Macro tensor file: `{date_stem}_macro.pt`
Load with: `torch.load(path, map_location="cpu", weights_only=False)`

Keys:
- `"features"` → `Tensor[N_cells, 10]` float32 — all grid cells always present (empty = zeros)
- `"lat_idx"` → `Tensor[N_cells]` int64 — latitude grid index per cell (0..n_lat_steps-1)
- `"lon_idx"` → `Tensor[N_cells]` int64 — longitude grid index per cell (0..n_lon_steps-1)

**N_cells is derived from the tensor shape — NEVER hardcode it (e.g. 1044 for Gulf of Mexico at 0.5 deg).**

### 2. Micro bundle file: `{date_stem}_micro_bundle.pt`
Load with: `torch.load(path, map_location="cpu", weights_only=False)`

Keys:
- `"date"` → str
- `"windows"` → `list[Tensor]`, each `[N_day, 11]` float32 — one tensor per vessel, variable N_day (typically 40–200, minimum 10)
- `"metadata"` → `list[dict]` with keys: `"mmsi"` (int), `"ping_count"` (int), `"bundle_index"` (int), `"date"` (str)

### 3. Master dataset index: `{split}_dataset_index.json`
```json
{
  "total_samples": 12345,
  "pairs": [
    {
      "date": "2022-01-01",
      "mmsi": 123456789,
      "bundle_index": 0,
      "ping_count": 87,
      "micro_bundle": "/abs/path/to/2022-01-01_micro_bundle.pt",
      "macro": "/abs/path/to/2022-01-01_macro.pt"
    }
  ]
}
```

### 4. Norm stats file: `norm_stats.json`
Relevant keys:
- `"N_LAT_STEPS"` → int (e.g. 28)
- `"N_LON_STEPS"` → int (e.g. 36)
- `"N_TOTAL_CELLS"` → int (= N_LAT_STEPS × N_LON_STEPS, e.g. 1044)
- `"BIN_SIZE"` → float (e.g. 0.5)

**These drive geo-embedding table sizes. Always read from file — never hardcode.**

---

## WHAT TO BUILD

Create a single file: `hb_mamba_dataset.py`

Containing:
1. `GapMaskConfig` — dataclass for gap masking parameters
2. `HBMambaDataset` — `torch.utils.data.Dataset`
3. `hb_mamba_collate_fn` — custom collate function
4. `build_dataloaders(...)` — convenience factory
5. `if __name__ == "__main__":` smoke-test block

---

## DETAILED SPECIFICATIONS

### `GapMaskConfig` (dataclass)

```python
@dataclass
class GapMaskConfig:
    min_gap_frac: float = 0.05    # minimum gap = 5% of N_day
    max_gap_frac: float = 0.40    # maximum gap = 40% of N_day
    interp_prob:  float = 0.70    # probability of interpolation gap (vs extrapolation)
    min_gap_len:  int   = 2       # never mask fewer than 2 pings
```

---

### `HBMambaDataset(Dataset)`

#### Constructor:
```python
def __init__(
    self,
    dataset_index_path: str,
    norm_stats_path: str,
    split: str,
    gap_config: GapMaskConfig,
    cache_macro: bool = True,
    cache_micro: bool = False,
):
```

Constructor must:
1. Load `dataset_index_path` JSON → store `self.pairs` (the list of sample dicts).
2. Load `norm_stats_path` JSON → store `self.n_lat_steps`, `self.n_lon_steps`, `self.n_total_cells`, `self.bin_size` as instance attributes. **These are exposed for model config.**
3. Init empty dicts: `self._macro_cache = {}` (keyed by date str), `self._micro_cache = {}` (keyed by date str).
4. Store `self.split`, `self.gap_config`, `self.n_samples = len(self.pairs)`.

#### Public attributes (model config reads these):
```
self.n_lat_steps    int
self.n_lon_steps    int
self.n_total_cells  int
self.n_samples      int
self.split          str
```

#### `__len__`: returns `self.n_samples`

#### `__getitem__(self, idx) -> dict`:

Return dict with exactly these keys:
```python
{
    "macro_features":  Tensor[N_cells, 10],   # float32
    "macro_lat_idx":   Tensor[N_cells],        # int64
    "macro_lon_idx":   Tensor[N_cells],        # int64
    "micro_tokens":    Tensor[N_day, 11],      # float32, unmasked full sequence
    "mask":            BoolTensor[N_day],      # True = this ping is MASKED (must predict)
    "gap_type":        str,                    # "interpolation" or "extrapolation"
    "mmsi":            int,
    "date":            str,
    "n_day":           int,                    # actual ping count (= N_day, pre-padding)
}
```

Loading:
- Macro: if `cache_macro=True` and date already in `self._macro_cache`, reuse. Otherwise load from `pairs[idx]["macro"]`.
- Micro: Load bundle from `pairs[idx]["micro_bundle"]` (cache if `cache_micro=True`, keyed by date). Take `bundle["windows"][pairs[idx]["bundle_index"]]`.

Gap masking (apply EVERY `__getitem__` call — fresh random gap each time):
1. `gap_len = max(min_gap_len, int(N_day * uniform(min_gap_frac, max_gap_frac)))`
2. `gap_len = min(gap_len, N_day - 2)` — always keep ≥ 2 visible pings
3. With probability `interp_prob`: `gap_type = "interpolation"`, `gap_start = randint(1, N_day - gap_len - 1)` (gap never touches first or last ping)
4. Otherwise: `gap_type = "extrapolation"`, `gap_start = N_day - gap_len` (gap at the end)
5. Build `mask = torch.zeros(N_day, dtype=torch.bool)`, set `mask[gap_start : gap_start + gap_len] = True`

**Do NOT pad micro_tokens here. Padding is the collate_fn's job.**

---

### `hb_mamba_collate_fn(batch: list[dict]) -> dict`

1. `max_n_day = max(item["n_day"] for item in batch)`
2. Pad `micro_tokens` with zeros to `[B, max_n_day, 11]`
3. Pad `mask` with `False` to `[B, max_n_day]`
4. Build `padding_mask: BoolTensor[B, max_n_day]` — `True` at positions `>= item["n_day"]` (**True = padded/ignored**, matching PyTorch `key_padding_mask` convention)
5. Stack: `macro_features → [B, N_cells, 10]`, `macro_lat_idx → [B, N_cells]`, `macro_lon_idx → [B, N_cells]`
6. `n_day → LongTensor[B]`
7. `mmsi → list[int]`, `date → list[str]`, `gap_type → list[str]` (keep as lists)

Return:
```python
{
    "macro_features":  Tensor[B, N_cells, 10],
    "macro_lat_idx":   Tensor[B, N_cells],
    "macro_lon_idx":   Tensor[B, N_cells],
    "micro_tokens":    Tensor[B, max_n_day, 11],
    "mask":            BoolTensor[B, max_n_day],   # True = masked ping (to predict)
    "padding_mask":    BoolTensor[B, max_n_day],   # True = padded position (ignore)
    "n_day":           LongTensor[B],
    "mmsi":            list[int],
    "date":            list[str],
    "gap_type":        list[str],
}
```

**Mask convention note (add as comment):**
- `mask`: `True` = this is a MASKED ping the model must reconstruct
- `padding_mask`: `True` = this is a PADDED position (no real data, ignore completely)
- These two must never overlap: `assert not (mask & padding_mask).any()`

---

### `build_dataloaders(...) -> dict[str, DataLoader]`

```python
def build_dataloaders(
    dataset_index_dir: str,
    norm_stats_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    gap_config: GapMaskConfig | None = None,
    cache_macro: bool = True,
    cache_micro: bool = False,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
```

1. Scan `dataset_index_dir` for `*_dataset_index.json` — derive split name from filename prefix (e.g. `train_dataset_index.json` → `"train"`). **Do NOT hardcode split names.**
2. Create `HBMambaDataset` per split found.
3. Create `DataLoader` per split:
   - Training: `shuffle=True`, `drop_last=True`
   - Val/Test: `shuffle=False`, `drop_last=False`
   - All: `collate_fn=hb_mamba_collate_fn`, `num_workers=num_workers`, `pin_memory=pin_memory`
4. Return dict of only the splits that exist.

---

## SMOKE-TEST BLOCK (`if __name__ == "__main__":`)

Accept `PROJECT_ROOT` as `sys.argv[1]` (fallback: current directory).

For each split found:
- Print split name, total samples, batch count
- Load first batch, print all tensor shapes
- Print `n_lat_steps`, `n_lon_steps`, `n_total_cells`
- Print gap_type distribution in the batch
- Check for NaNs in `macro_features` and `micro_tokens`

Assertions to run:
```python
assert batch["macro_features"].shape[1] == dataset.n_total_cells
assert int(batch["macro_lat_idx"].max()) < dataset.n_lat_steps
assert int(batch["macro_lon_idx"].max()) < dataset.n_lon_steps
assert batch["micro_tokens"].shape[2] == 11
assert batch["mask"].sum(dim=1).min() >= 1          # at least 1 masked ping per sample
assert batch["padding_mask"].shape == batch["mask"].shape
assert not (batch["mask"] & batch["padding_mask"]).any()   # no overlap
```

End with: `print("✓ Dataset smoke-test passed")`

---

## CRITICAL CONSTRAINTS

1. **Nothing hardcoded.** No cell counts, no grid dims, no date ranges. N_cells from tensor shape; grid dims from norm_stats.json. Feature dims (10 macro, 11 micro) are the only safe constants.
2. **Gap masking in `__getitem__`** — fresh random gap every call = fresh gap every epoch.
3. **Pings are NEVER shuffled.** Timestamp order is sacred. Only sample ORDER is shuffled by DataLoader.
4. **`weights_only=False`** in all `torch.load` calls — bundles contain Python dicts with lists.
5. **Macro cache keyed by date string** — multiple vessel samples on the same date share one loaded macro tensor.
6. **No re-normalisation** — tensors are already normalised from preprocessing. Load and serve as-is.
7. **`drop_last=True` for training** — contrastive loss needs consistent batch size for negative pair mining.
8. Use `pathlib.Path` for all file operations.
9. Type hints throughout. Docstrings on every class and function.
10. Dependencies: only `torch`, `json`, `random`, `pathlib`, `dataclasses`, `typing`, `os`, `re`. Nothing else.

---

## USAGE EXAMPLE (for the module docstring)

```python
from hb_mamba_dataset import build_dataloaders, GapMaskConfig

dataloaders = build_dataloaders(
    dataset_index_dir = "/path/AIS_project/dataset_index",
    norm_stats_path   = "/path/AIS_project/preprocessing/norm_stats/norm_stats.json",
    batch_size        = 64,
    num_workers       = 4,
)

train_ds = dataloaders["train"].dataset
print(train_ds.n_lat_steps)    # e.g. 28
print(train_ds.n_lon_steps)    # e.g. 36
print(train_ds.n_total_cells)  # e.g. 1044

for batch in dataloaders["train"]:
    # batch["macro_features"]  → [64, n_total_cells, 10]
    # batch["micro_tokens"]    → [64, max_n_day, 11]
    # batch["mask"]            → [64, max_n_day] bool  (True = predict here)
    # batch["padding_mask"]    → [64, max_n_day] bool  (True = ignore, padded)
    # batch["n_day"]           → [64] int
    pass
```

---

*End of prompt. Implement `hb_mamba_dataset.py` exactly as specified.*
