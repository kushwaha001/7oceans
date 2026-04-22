"""
inference.py — HB-Mamba v2.0  Inference Pipeline
==================================================

Loads a trained checkpoint (default: checkpoints/best.pt) and runs
HBMamba.predict() over a dataset split to extract trajectory embeddings.

Usage:
    # Embed the val split using best checkpoint
    python inference.py

    # Embed a specific split / checkpoint
    python inference.py --split train --checkpoint checkpoints/epoch_025.pt

    # Also save per-step hidden states and attention weights
    python inference.py --save_hidden --save_attn

    # Custom output directory and batch size
    python inference.py --output my_embeddings --batch_size 128

Outputs (saved to --output directory):
    embeddings.npz
        H           float32  [N, 256]   trajectory-level embedding
        mmsi        int64    [N]        vessel MMSI
        date        str      [N]        date string  (YYYY-MM-DD)
        n_day       int64    [N]        trajectory length
        epoch       int                checkpoint epoch
        val_loss    float              checkpoint val_loss

    hidden_states.npz  (--save_hidden)
        h_fwd       float32  [N, T_max, 256]   forward Mamba states (zero-padded)
        h_bwd       float32  [N, T_max, 256]   backward Mamba states
        padding_mask bool    [N, T_max]         True = padded position

    attn_weights.npz   (--save_attn)
        attn_weights float32 [N, T_max, K]     cross-scale attention (zero-padded)
        topk_idx     int64   [N, T_max, K]     grid cell indices for top-K

Key flags:
    --checkpoint  PATH    Checkpoint file            [checkpoints/best.pt]
    --split       STR     Dataset split to embed     [val]
    --output      DIR     Output directory           [embeddings]
    --batch_size  N       Batch size                 [64]
    --num_workers N       DataLoader workers         [4]
    --device      STR     torch device string        [cuda:0]
    --no_amp              Disable bfloat16 autocast
    --save_hidden         Save h_fwd / h_bwd arrays
    --save_attn           Save attn_weights / topk_idx arrays
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Paths (mirrors train.py)
# ---------------------------------------------------------------------------

RAW_ROOT = ROOT / "data_genration_and_raw_data" / "raw_data"
NS_PATH  = str(RAW_ROOT / "preprocessing" / "norm_stats" / "norm_stats.json")
IDX_DIR  = str(RAW_ROOT / "dataset_index")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="HB-Mamba v2.0 — inference / embedding extraction")
    p.add_argument("--checkpoint",   default="checkpoints/best.pt",
                   help="Path to checkpoint  [checkpoints/best.pt]")
    p.add_argument("--split",        default="val",
                   help="Dataset split to embed  [val]")
    p.add_argument("--output",       default="embeddings",
                   help="Output directory  [embeddings]")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--device",       default="cuda:0",
                   help="torch device string  [cuda:0]")
    p.add_argument("--no_amp",       action="store_true",
                   help="Disable bfloat16 autocast")
    p.add_argument("--save_hidden",  action="store_true",
                   help="Save h_fwd / h_bwd per-step hidden states")
    p.add_argument("--save_attn",    action="store_true",
                   help="Save attn_weights / topk_idx arrays")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    """
    Load HBMamba from a checkpoint saved by training/trainer.py.

    Checkpoint format (from _save_checkpoint):
        model_state  : state_dict (may have 'module.' prefix if saved from DDP)
        model_config : dict of HBMamba.__init__ kwargs
        epoch        : int
        val_loss     : float  (val_total at save time)
    """
    from model.hb_mamba import HBMamba

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Strip DDP 'module.' prefix if present
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    model = HBMamba(**ckpt["model_config"]).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"loss_heads.feature_weights"}
    unknown = [k for k in missing if k not in allowed_missing]
    if unknown or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch.  missing={unknown}  unexpected={unexpected}"
        )
    model.eval()

    meta = {
        "epoch"    : ckpt.get("epoch",    -1),
        "val_loss" : ckpt.get("val_loss", float("nan")),
        "config"   : ckpt["model_config"],
    }
    return model, meta


# ---------------------------------------------------------------------------
# Build inference DataLoader (no gap masking randomness needed)
# ---------------------------------------------------------------------------

def build_loader(split: str, batch_size: int, num_workers: int):
    """
    Build a DataLoader for the given split.

    For inference we use a deterministic gap seed (val/test style) regardless
    of split name — the mask is ignored downstream anyway since predict()
    does not use it.
    """
    from hb_mamba_dataset import HBMambaDataset, hb_mamba_collate_fn, GapMaskConfig
    from pathlib import Path as P

    idx_dir = P(IDX_DIR)
    # Prefer exact split match, fall back to val, then test
    candidates = [
        idx_dir / f"{split}_dataset_index.json",
        idx_dir / "val_dataset_index.json",
        idx_dir / "test_dataset_index.json",
    ]
    idx_path = None
    for c in candidates:
        if c.exists():
            idx_path = c
            actual_split = c.stem.replace("_dataset_index", "")
            break

    if idx_path is None:
        # Fallback: use any index we can find
        found = sorted(idx_dir.glob("*_dataset_index.json"))
        if not found:
            raise FileNotFoundError(f"No dataset index files found in {idx_dir}")
        idx_path = found[0]
        actual_split = idx_path.stem.replace("_dataset_index", "")

    if actual_split != split:
        print(f"  [warn] requested split '{split}' not found — using '{actual_split}'")

    dataset = HBMambaDataset(
        dataset_index_path = str(idx_path),
        norm_stats_path    = NS_PATH,
        split              = actual_split,
        gap_config         = GapMaskConfig(),
        cache_macro        = True,
        cache_micro        = False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = False,
        collate_fn  = hb_mamba_collate_fn,
        num_workers = num_workers,
        pin_memory  = True,
    )
    return loader, actual_split


# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device, use_amp: bool, save_hidden: bool, save_attn: bool):
    """
    Iterate over the loader and collect model outputs.

    Returns
    -------
    dict with numpy arrays:
        H            float32  [N, d_model]
        mmsi         int64    [N]
        date         [N]  object array of strings
        n_day        int64    [N]
        h_fwd        float32  [N, T_max, d_model]   only if save_hidden
        h_bwd        float32  [N, T_max, d_model]   only if save_hidden
        padding_mask bool     [N, T_max]             only if save_hidden
        attn_weights float32  [N, T_max, K]          only if save_attn
        topk_idx     int64    [N, T_max, K]          only if save_attn
    """
    H_list            = []
    mmsi_list         = []
    date_list         = []
    n_day_list        = []
    h_fwd_list        = []
    h_bwd_list        = []
    pad_mask_list     = []
    attn_list         = []
    topk_list         = []

    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        macro_features = batch["macro_features"].to(device, non_blocking=True)
        macro_lat_idx  = batch["macro_lat_idx"].to(device,  non_blocking=True)
        macro_lon_idx  = batch["macro_lon_idx"].to(device,  non_blocking=True)
        micro_tokens   = batch["micro_tokens"].to(device,   non_blocking=True)
        padding_mask   = batch["padding_mask"].to(device,   non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model.predict(
                macro_features,
                macro_lat_idx,
                macro_lon_idx,
                micro_tokens,
                padding_mask,
            )

        # H is always float32 after autocast — cast explicitly to be safe
        H_list.append(out["H"].float().cpu())

        mmsi_list.extend(batch["mmsi"])
        date_list.extend(batch["date"])
        n_day_list.append(batch["n_day"])   # LongTensor[B]

        if save_hidden:
            h_fwd_list.append(out["h_fwd"].float().cpu())
            h_bwd_list.append(out["h_bwd"].float().cpu())
            pad_mask_list.append(padding_mask.cpu())

        if save_attn:
            attn_list.append(out["attn_weights"].float().cpu())
            topk_list.append(out["topk_idx"].cpu())

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
            print(f"  batch {batch_idx + 1:4d} / {n_batches}")

    # ── Concatenate ──────────────────────────────────────────────────────────
    result = {
        "H"    : torch.cat(H_list,     dim=0).numpy(),        # [N, 256]
        "mmsi" : np.array(mmsi_list,   dtype=np.int64),       # [N]
        "date" : np.array(date_list,   dtype=object),         # [N]
        "n_day": torch.cat(n_day_list, dim=0).numpy().astype(np.int64),  # [N]
    }

    if save_hidden:
        # Pad to common T_max across batches (they may differ in seq length)
        result["h_fwd"],     T_max = _pad_sequence_list(h_fwd_list)
        result["h_bwd"],     _     = _pad_sequence_list(h_bwd_list)
        result["padding_mask"], _  = _pad_sequence_list(pad_mask_list, bool_pad=True)

    if save_attn:
        result["attn_weights"], _ = _pad_sequence_list(attn_list)
        result["topk_idx"],     _ = _pad_sequence_list(topk_list, int_pad=True)

    return result


def _pad_sequence_list(tensor_list, bool_pad: bool = False, int_pad: bool = False):
    """
    Concatenate a list of [B, T, ...] tensors along dim=0, padding T dimension
    to the maximum T seen across all batches.
    """
    T_max = max(t.shape[1] for t in tensor_list)
    padded = []
    for t in tensor_list:
        B, T = t.shape[0], t.shape[1]
        if T < T_max:
            pad_shape = (B, T_max - T) + t.shape[2:]
            if bool_pad:
                pad = torch.ones(pad_shape, dtype=torch.bool)  # True = padded
            elif int_pad:
                pad = torch.zeros(pad_shape, dtype=torch.long)
            else:
                pad = torch.zeros(pad_shape, dtype=t.dtype)
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0).numpy(), T_max


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("HB-Mamba v2.0 — Inference")
    print(f"  checkpoint  : {ckpt_path}")
    print(f"  device      : {device}")
    print(f"  amp         : {use_amp}")
    print(f"  split       : {args.split}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  save_hidden : {args.save_hidden}")
    print(f"  save_attn   : {args.save_attn}")
    print(f"  output dir  : {out_dir}")
    print("=" * 65)

    # ── Set CUDA device context before any mamba_ssm / Triton import ─────────
    # Triton auto-tunes its kernels against the active CUDA device.
    # Must call set_device() here — same reason train.py defers the HBMamba
    # import until after torch.cuda.set_device(_rank()).
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from {ckpt_path.name} ...")
    model, meta = load_model(str(ckpt_path), device)

    print(f"  checkpoint epoch   : {meta['epoch']}")
    print(f"  checkpoint val_loss: {meta['val_loss']:.6f}")
    print(f"\n{model}\n")

    # ── Build DataLoader ──────────────────────────────────────────────────────
    print(f"Building DataLoader (split='{args.split}') ...")
    loader, actual_split = build_loader(args.split, args.batch_size, args.num_workers)
    print(f"  samples : {len(loader.dataset)}")
    print(f"  batches : {len(loader)}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\nRunning inference ...")
    result = run_inference(
        model, loader, device, use_amp,
        save_hidden = args.save_hidden,
        save_attn   = args.save_attn,
    )

    N = result["H"].shape[0]
    print(f"\n  samples processed : {N}")
    print(f"  H shape           : {result['H'].shape}")
    print(f"  H mean L2 norm    : {np.linalg.norm(result['H'], axis=1).mean():.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # embeddings.npz — always
    emb_path = out_dir / "embeddings.npz"
    np.savez(
        emb_path,
        H        = result["H"],
        mmsi     = result["mmsi"],
        date     = result["date"],
        n_day    = result["n_day"],
        epoch    = np.array(meta["epoch"]),
        val_loss = np.array(meta["val_loss"]),
    )
    print(f"\n  saved: {emb_path}  ({emb_path.stat().st_size / 1024:.1f} KB)")

    # hidden_states.npz — optional
    if args.save_hidden:
        hidden_path = out_dir / "hidden_states.npz"
        np.savez_compressed(
            hidden_path,
            h_fwd        = result["h_fwd"],
            h_bwd        = result["h_bwd"],
            padding_mask = result["padding_mask"],
        )
        print(f"  saved: {hidden_path}  ({hidden_path.stat().st_size / 1024:.1f} KB)")

    # attn_weights.npz — optional
    if args.save_attn:
        attn_path = out_dir / "attn_weights.npz"
        np.savez_compressed(
            attn_path,
            attn_weights = result["attn_weights"],
            topk_idx     = result["topk_idx"],
        )
        print(f"  saved: {attn_path}  ({attn_path.stat().st_size / 1024:.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
