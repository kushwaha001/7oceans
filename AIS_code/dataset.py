"""
AISDataset — PyTorch Dataset for HB-Mamba v2.0
Paired macro [936, 10] + micro [N_day, 11] tensors with gap simulation.

Key rule:
  - train split: fresh random gap every __getitem__ call (diversity)
  - val/test split: gap seeded by sample index (deterministic — same gap every epoch)
"""

import json
import torch
from torch.utils.data import Dataset


class AISDataset(Dataset):
    def __init__(self, index_path: str, split: str):
        """
        Parameters
        ----------
        index_path : str
            Path to {split}_dataset_index.json built by build_dataset_index.
        split : str
            "train", "val", or "test".
        """
        with open(index_path, "r") as f:
            idx = json.load(f)
        self.pairs = idx["pairs"]
        self.split = split
        self._bundle_cache: dict = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_bundle(self, path: str) -> dict:
        if path not in self._bundle_cache:
            self._bundle_cache[path] = torch.load(
                path, map_location="cpu", weights_only=False
            )
        return self._bundle_cache[path]

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        # ── load tensors ──────────────────────────────────────────────
        macro = torch.load(pair["macro"], map_location="cpu", weights_only=True)
        bundle = self._load_bundle(pair["micro_bundle"])
        micro = bundle["windows"][pair["bundle_index"]]   # [N_day, 11]

        # ── gap simulation ────────────────────────────────────────────
        rng = torch.Generator()
        if self.split == "train":
            pass  # no seed — fresh random gap every epoch
        else:
            rng.manual_seed(idx)  # FIXED gap: same gap every epoch for val/test

        N = micro.shape[0]

        # Gap length: 5%–40% of sequence length
        min_gap = max(1, int(0.05 * N))
        max_gap = max(min_gap + 1, int(0.40 * N))
        gap_len = int(torch.randint(min_gap, max_gap, (1,), generator=rng).item())

        # Gap type: 70% interpolation (middle), 30% extrapolation (end)
        is_extrapolation = torch.rand(1, generator=rng).item() < 0.30
        if is_extrapolation or N - gap_len <= 0:
            gap_start = max(0, N - gap_len)
        else:
            gap_start = int(torch.randint(0, N - gap_len, (1,), generator=rng).item())

        mask = torch.zeros(N, dtype=torch.bool)
        mask[gap_start: gap_start + gap_len] = True

        return {
            "macro":   macro,                  # [936, 10]
            "micro":   micro,                  # [N_day, 11]
            "mask":    mask,                   # [N_day]  True = masked (gap)
            "mmsi":    pair["mmsi"],
            "date":    pair["date"],
        }


def collate_fn(batch: list) -> dict:
    """
    Pad micro tensors to max N_day in batch.
    All vessels have different daily ping counts — padding to batch max.
    Pads with zeros. pad_mask marks which positions are padding (not gap).
    """
    max_n = max(item["micro"].shape[0] for item in batch)

    micro_padded = torch.zeros(len(batch), max_n, 11)
    mask_padded  = torch.zeros(len(batch), max_n, dtype=torch.bool)
    pad_mask     = torch.ones(len(batch), max_n, dtype=torch.bool)  # True = padded pos

    for i, item in enumerate(batch):
        n = item["micro"].shape[0]
        micro_padded[i, :n] = item["micro"]
        mask_padded[i, :n]  = item["mask"]
        pad_mask[i, :n]     = False  # real positions

    return {
        "macro":    torch.stack([item["macro"] for item in batch]),  # [B, 936, 10]
        "micro":    micro_padded,                                    # [B, max_n, 11]
        "mask":     mask_padded,                                     # [B, max_n]
        "pad_mask": pad_mask,                                        # [B, max_n]
    }
