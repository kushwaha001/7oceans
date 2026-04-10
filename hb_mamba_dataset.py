"""
HB-Mamba v2.0 — Dataset & DataLoader
======================================

Provides `HBMambaDataset`, `hb_mamba_collate_fn`, and `build_dataloaders`
for the Hierarchical Bidirectional Mamba vessel trajectory model.

Usage example::

    from hb_mamba_dataset import build_dataloaders, GapMaskConfig

    dataloaders = build_dataloaders(
        dataset_index_dir = "/home/hpc25/AIS/SAR_AIS_analysis/data_genration_and_raw_data/raw_data/dataset_index",
        norm_stats_path   = "/home/hpc25/AIS/SAR_AIS_analysis/data_genration_and_raw_data/raw_data/preprocessing/norm_stats/norm_stats.json",
        batch_size        = 64,
        num_workers       = 4,
    )

    train_ds = dataloaders["train"].dataset
    print(train_ds.n_lat_steps)    # e.g. 29
    print(train_ds.n_lon_steps)    # e.g. 36
    print(train_ds.n_total_cells)  # e.g. 1044

    for batch in dataloaders["train"]:
        # batch["macro_features"]  → [B, n_total_cells, 10]
        # batch["micro_tokens"]    → [B, max_n_day, 11]
        # batch["mask"]            → [B, max_n_day] bool  (True = predict here)
        # batch["padding_mask"]    → [B, max_n_day] bool  (True = ignore, padded)
        # batch["n_day"]           → [B] int
        pass
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# GapMaskConfig
# ---------------------------------------------------------------------------

@dataclass
class GapMaskConfig:
    """
    Parameters that control the random gap-masking strategy.

    Attributes
    ----------
    min_gap_frac : float
        Minimum gap length as a fraction of the trajectory length (N_day).
    max_gap_frac : float
        Maximum gap length as a fraction of the trajectory length (N_day).
    interp_prob : float
        Probability of placing the gap in the interior of the sequence
        (interpolation) vs at the tail (extrapolation).
    min_gap_len : int
        Hard lower bound — never mask fewer than this many pings.
    """
    min_gap_frac: float = 0.05
    max_gap_frac: float = 0.40
    interp_prob:  float = 0.70
    min_gap_len:  int   = 2


# ---------------------------------------------------------------------------
# HBMambaDataset
# ---------------------------------------------------------------------------

class HBMambaDataset(Dataset):
    
    """
    PyTorch Dataset for HB-Mamba v2.0 vessel trajectory learning.

    Each sample pairs one vessel's full-day trajectory (micro) with the
    corresponding daily grid snapshot (macro). A random gap mask is applied
    on every `__getitem__` call so the model sees a different masking
    pattern each epoch.

    Parameters
    ----------
    dataset_index_path : str
        Absolute path to a ``{split}_dataset_index.json`` file.
    norm_stats_path : str
        Absolute path to ``norm_stats.json``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"`` (informational).
    gap_config : GapMaskConfig
        Gap masking hyper-parameters.
    cache_macro : bool
        If True, macro tensors are loaded once per date and reused.
    cache_micro : bool
        If True, micro bundles are loaded once per date and reused.
        Keep False for large datasets to avoid OOM.
    """

    def __init__(
        self,
        dataset_index_path: str,
        norm_stats_path: str,
        split: str,
        gap_config: GapMaskConfig,
        cache_macro: bool = True,
        cache_micro: bool = False,
    ) -> None:
        super().__init__()

        # ── Load dataset index ──────────────────────────────────────────────
        idx_path = Path(dataset_index_path)
        if not idx_path.exists():
            raise FileNotFoundError(f"Dataset index not found: {idx_path}")
        with idx_path.open("r") as fh:
            index = json.load(fh)
        self.pairs: List[Dict] = index["pairs"]

        # ── Load norm stats ─────────────────────────────────────────────────
        ns_path = Path(norm_stats_path)
        if not ns_path.exists():
            raise FileNotFoundError(f"Norm stats not found: {ns_path}")
        with ns_path.open("r") as fh:
            ns = json.load(fh)
        self.n_lat_steps: int   = int(ns["N_LAT_STEPS"])
        self.n_lon_steps: int   = int(ns["N_LON_STEPS"])
        self.n_total_cells: int = int(ns["N_TOTAL_CELLS"])
        self.bin_size: float    = float(ns["BIN_SIZE"])

        # ── Misc state ──────────────────────────────────────────────────────
        self.split        = split
        self.gap_config   = gap_config
        self.n_samples    = len(self.pairs)
        self.cache_macro  = cache_macro
        self.cache_micro  = cache_micro

        # Caches keyed by date string
        self._macro_cache: Dict[str, Dict[str, Tensor]] = {}
        self._micro_cache: Dict[str, Dict]              = {}

    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_samples

    # -----------------------------------------------------------------------
    def _load_macro(self, pair: Dict) -> Dict[str, Tensor]:
        """Load (or retrieve from cache) the macro tensor for a given date."""
        date = pair["date"]
        if self.cache_macro and date in self._macro_cache:
            return self._macro_cache[date]
        data = torch.load(pair["macro"], map_location="cpu", weights_only=False)
        if self.cache_macro:
            self._macro_cache[date] = data
        return data

    def _load_micro_bundle(self, pair: Dict) -> Dict:
        """Load (or retrieve from cache) the micro bundle for a given date."""
        date = pair["date"]
        if self.cache_micro and date in self._micro_cache:
            return self._micro_cache[date]
        data = torch.load(pair["micro_bundle"], map_location="cpu", weights_only=False)
        if self.cache_micro:
            self._micro_cache[date] = data
        return data

    # -----------------------------------------------------------------------
    def _apply_gap_mask(self, n_day: int) -> tuple[torch.BoolTensor, str]:
        """
        Sample a random contiguous gap and return the boolean mask and type.

        Returns
        -------
        mask : BoolTensor[n_day]
            True at positions that are MASKED (model must reconstruct).
        gap_type : str
            "interpolation" or "extrapolation".
        """
        cfg = self.gap_config

        # Compute gap length
        gap_len = max(cfg.min_gap_len,
                      int(n_day * random.uniform(cfg.min_gap_frac, cfg.max_gap_frac)))
        gap_len = min(gap_len, n_day - 2)  # always keep ≥ 2 visible pings

        # Decide gap type and position
        if random.random() < cfg.interp_prob:
            gap_type  = "interpolation"
            gap_start = random.randint(1, n_day - gap_len - 1)
        else:
            gap_type  = "extrapolation"
            gap_start = n_day - gap_len

        mask = torch.zeros(n_day, dtype=torch.bool)
        mask[gap_start : gap_start + gap_len] = True
        return mask, gap_type

    # -----------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        """
        Return a single training sample.

        Returns
        -------
        dict with keys:
            macro_features  Tensor[N_cells, 10]   float32
            macro_lat_idx   Tensor[N_cells]        int64
            macro_lon_idx   Tensor[N_cells]        int64
            micro_tokens    Tensor[N_day, 11]      float32  (unmasked)
            mask            BoolTensor[N_day]      True = masked ping to predict
            gap_type        str
            mmsi            int
            date            str
            n_day           int
        """
        pair = self.pairs[idx]

        # ── Macro ────────────────────────────────────────────────────────────
        macro = self._load_macro(pair)
        macro_features: Tensor = macro["features"]   # [N_cells, 10]
        macro_lat_idx:  Tensor = macro["lat_idx"]    # [N_cells]
        macro_lon_idx:  Tensor = macro["lon_idx"]    # [N_cells]

        # ── Micro ────────────────────────────────────────────────────────────
        bundle = self._load_micro_bundle(pair)
        micro_tokens: Tensor = bundle["windows"][pair["bundle_index"]]  # [N_day, 11]

        n_day = micro_tokens.shape[0]

        # ── Gap masking ──────────────────────────────────────────────────────
        mask, gap_type = self._apply_gap_mask(n_day)

        return {
            "macro_features": macro_features,
            "macro_lat_idx":  macro_lat_idx,
            "macro_lon_idx":  macro_lon_idx,
            "micro_tokens":   micro_tokens,
            "mask":           mask,
            "gap_type":       gap_type,
            "mmsi":           int(pair["mmsi"]),
            "date":           pair["date"],
            "n_day":          n_day,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def hb_mamba_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate for variable-length micro trajectories.

    Mask conventions
    ----------------
    - ``mask``         : True  = this ping is MASKED (model must reconstruct it).
    - ``padding_mask`` : True  = this position is PADDED (no real data, ignore completely).
    - These two must NEVER overlap: ``(mask & padding_mask).any() == False``.

    Parameters
    ----------
    batch : list[dict]
        List of dicts returned by `HBMambaDataset.__getitem__`.

    Returns
    -------
    dict with tensors stacked / padded to batch size B:
        macro_features  [B, N_cells, 10]
        macro_lat_idx   [B, N_cells]
        macro_lon_idx   [B, N_cells]
        micro_tokens    [B, max_n_day, 11]   zero-padded
        mask            [B, max_n_day]       bool, False at padded positions
        padding_mask    [B, max_n_day]       bool, True at padded positions
        n_day           LongTensor[B]
        mmsi            list[int]
        date            list[str]
        gap_type        list[str]
    """
    B = len(batch)
    max_n_day = max(item["n_day"] for item in batch)

    # Pre-allocate padded tensors
    feat_dim  = batch[0]["micro_tokens"].shape[1]   # 11
    n_cells   = batch[0]["macro_features"].shape[0]

    micro_tokens_padded  = torch.zeros(B, max_n_day, feat_dim,  dtype=torch.float32)
    mask_padded          = torch.zeros(B, max_n_day,             dtype=torch.bool)
    padding_mask         = torch.ones( B, max_n_day,             dtype=torch.bool)   # default: all padded

    macro_features_list: List[Tensor] = []
    macro_lat_idx_list:  List[Tensor] = []
    macro_lon_idx_list:  List[Tensor] = []
    n_day_list:          List[int]    = []
    mmsi_list:           List[int]    = []
    date_list:           List[str]    = []
    gap_type_list:       List[str]    = []

    for i, item in enumerate(batch):
        nd = item["n_day"]

        micro_tokens_padded[i, :nd, :]  = item["micro_tokens"]
        mask_padded[i, :nd]             = item["mask"]
        padding_mask[i, :nd]            = False  # real data positions → not padded

        macro_features_list.append(item["macro_features"])
        macro_lat_idx_list.append(item["macro_lat_idx"])
        macro_lon_idx_list.append(item["macro_lon_idx"])
        n_day_list.append(nd)
        mmsi_list.append(item["mmsi"])
        date_list.append(item["date"])
        gap_type_list.append(item["gap_type"])

    # Mask convention assertion: masked and padding must not overlap
    assert not (mask_padded & padding_mask).any(), (
        "Invariant violated: mask and padding_mask overlap at the same positions. "
        "Check that gap masking only touches real (non-padded) pings."
    )

    return {
        "macro_features": torch.stack(macro_features_list, dim=0),   # [B, N_cells, 10]
        "macro_lat_idx":  torch.stack(macro_lat_idx_list,  dim=0),   # [B, N_cells]
        "macro_lon_idx":  torch.stack(macro_lon_idx_list,  dim=0),   # [B, N_cells]
        "micro_tokens":   micro_tokens_padded,                        # [B, max_n_day, 11]
        "mask":           mask_padded,                                 # [B, max_n_day]
        "padding_mask":   padding_mask,                                # [B, max_n_day]
        "n_day":          torch.tensor(n_day_list, dtype=torch.long), # [B]
        "mmsi":           mmsi_list,
        "date":           date_list,
        "gap_type":       gap_type_list,
    }


# ---------------------------------------------------------------------------
# build_dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_index_dir: str,
    norm_stats_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    gap_config: Optional[GapMaskConfig] = None,
    cache_macro: bool = True,
    cache_micro: bool = False,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Scan ``dataset_index_dir`` for ``*_dataset_index.json`` files and build
    one `DataLoader` per split found.

    Split name is derived from the filename prefix, e.g.
    ``train_dataset_index.json`` → ``"train"``.
    Split names are NOT hardcoded — any prefix is accepted.

    Parameters
    ----------
    dataset_index_dir : str
        Directory containing ``{split}_dataset_index.json`` files.
    norm_stats_path : str
        Absolute path to ``norm_stats.json``.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        DataLoader worker processes.
    gap_config : GapMaskConfig | None
        Gap masking config. Defaults to `GapMaskConfig()` if None.
    cache_macro : bool
        Forward to `HBMambaDataset`.
    cache_micro : bool
        Forward to `HBMambaDataset`.
    pin_memory : bool
        Pin host memory for faster GPU transfers.

    Returns
    -------
    dict[str, DataLoader]
        Keys are split names (e.g. ``"train"``, ``"val"``, ``"test"``).
        Only splits whose index files exist are included.
    """
    if gap_config is None:
        gap_config = GapMaskConfig()

    index_dir = Path(dataset_index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"Dataset index directory not found: {index_dir}")

    # Discover index files; derive split name from filename prefix
    pattern = re.compile(r"^(.+)_dataset_index\.json$")
    index_files: List[tuple[str, Path]] = []
    for fp in sorted(index_dir.glob("*_dataset_index.json")):
        m = pattern.match(fp.name)
        if m:
            index_files.append((m.group(1), fp))

    if not index_files:
        raise FileNotFoundError(
            f"No '*_dataset_index.json' files found in: {index_dir}"
        )

    dataloaders: Dict[str, DataLoader] = {}

    for split, idx_path in index_files:
        dataset = HBMambaDataset(
            dataset_index_path = str(idx_path),
            norm_stats_path    = norm_stats_path,
            split              = split,
            gap_config         = gap_config,
            cache_macro        = cache_macro,
            cache_micro        = cache_micro,
        )

        is_train = (split == "train")
        loader = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = is_train,
            drop_last   = is_train,
            collate_fn  = hb_mamba_collate_fn,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )
        dataloaders[split] = loader

    return dataloaders


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from collections import Counter

    PROJECT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    DATASET_INDEX_DIR = os.path.join(PROJECT_ROOT, "dataset_index")
    NORM_STATS_PATH   = os.path.join(PROJECT_ROOT, "preprocessing", "norm_stats", "norm_stats.json")

    print(f"\nPROJECT_ROOT      : {PROJECT_ROOT}")
    print(f"DATASET_INDEX_DIR : {DATASET_INDEX_DIR}")
    print(f"NORM_STATS_PATH   : {NORM_STATS_PATH}\n")

    dataloaders = build_dataloaders(
        dataset_index_dir = DATASET_INDEX_DIR,
        norm_stats_path   = NORM_STATS_PATH,
        batch_size        = 8,
        num_workers       = 0,   # 0 for smoke-test (avoids multiprocessing overhead)
        pin_memory        = False,
    )

    for split, loader in dataloaders.items():
        dataset = loader.dataset

        print("=" * 60)
        print(f"Split            : {split}")
        print(f"Total samples    : {dataset.n_samples}")
        print(f"Batch count      : {len(loader)}")
        print(f"n_lat_steps      : {dataset.n_lat_steps}")
        print(f"n_lon_steps      : {dataset.n_lon_steps}")
        print(f"n_total_cells    : {dataset.n_total_cells}")

        batch = next(iter(loader))

        print("\nBatch tensor shapes:")
        for key in ("macro_features", "macro_lat_idx", "macro_lon_idx",
                    "micro_tokens", "mask", "padding_mask", "n_day"):
            print(f"  {key:20s} : {tuple(batch[key].shape)}")
        print(f"  {'mmsi':20s} : list len {len(batch['mmsi'])}")
        print(f"  {'date':20s} : list len {len(batch['date'])}")
        print(f"  {'gap_type':20s} : list len {len(batch['gap_type'])}")

        gap_counts = Counter(batch["gap_type"])
        print(f"\nGap type distribution: {dict(gap_counts)}")

        macro_has_nan = torch.isnan(batch["macro_features"]).any().item()
        micro_has_nan = torch.isnan(batch["micro_tokens"]).any().item()
        print(f"NaN in macro_features: {macro_has_nan}")
        print(f"NaN in micro_tokens  : {micro_has_nan}")

        # ── Assertions ──────────────────────────────────────────────────────
        assert batch["macro_features"].shape[1] == dataset.n_total_cells, \
            f"macro N_cells mismatch: {batch['macro_features'].shape[1]} vs {dataset.n_total_cells}"

        assert int(batch["macro_lat_idx"].max()) < dataset.n_lat_steps, \
            f"lat_idx {int(batch['macro_lat_idx'].max())} >= n_lat_steps {dataset.n_lat_steps}"

        assert int(batch["macro_lon_idx"].max()) < dataset.n_lon_steps, \
            f"lon_idx {int(batch['macro_lon_idx'].max())} >= n_lon_steps {dataset.n_lon_steps}"

        assert batch["micro_tokens"].shape[2] == 11, \
            f"micro feature dim expected 11, got {batch['micro_tokens'].shape[2]}"

        assert batch["mask"].sum(dim=1).min() >= 1, \
            "At least 1 masked ping required per sample"

        assert batch["padding_mask"].shape == batch["mask"].shape, \
            "padding_mask and mask shape mismatch"

        assert not (batch["mask"] & batch["padding_mask"]).any(), \
            "mask and padding_mask overlap — invariant violated"

        print(f"\n✓ All assertions passed for split '{split}'")

    print("\n✓ Dataset smoke-test passed")