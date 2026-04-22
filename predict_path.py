"""
predict_path.py — HB-Mamba v2.0  One-Day Path Prediction Inference
====================================================================

Loads a trained checkpoint and runs the full gap-prediction pipeline:
    1. Feed the gap-masked micro trajectory through the encoder
    2. Call ReconstructionHead(h_fwd, h_bwd) to predict features at gap positions
    3. Denormalise predictions back to real coordinates (degrees, knots, …)
    4. Compute per-sample errors and save structured results

Usage:
    python predict_path.py
    python predict_path.py --split val --checkpoint checkpoints/best.pt
    python predict_path.py --output path_predictions --batch_size 32 --max_samples 500

Outputs (saved to --output directory):
    path_predictions.npz
        mmsi          int64    [N]              vessel MMSI
        date          object   [N]              date string (YYYY-MM-DD)
        n_day         int64    [N]              trajectory length (# pings)
        gap_start     int64    [N]              gap start index (inclusive)
        gap_end       int64    [N]              gap end index (exclusive)
        gap_len       int64    [N]              number of gap pings
        gap_type      object   [N]              "interpolation" or "extrapolation"

        pred_lat      float32  [N_gap_total]    predicted latitude  (degrees)
        pred_lon      float32  [N_gap_total]    predicted longitude (degrees)
        pred_sog      float32  [N_gap_total]    predicted SOG       (knots)
        pred_cog      float32  [N_gap_total]    predicted COG       (degrees 0-360)

        true_lat      float32  [N_gap_total]    ground-truth latitude  (degrees)
        true_lon      float32  [N_gap_total]    ground-truth longitude (degrees)
        true_sog      float32  [N_gap_total]    ground-truth SOG       (knots)
        true_cog      float32  [N_gap_total]    ground-truth COG       (degrees 0-360)

        sample_idx    int64    [N_gap_total]    sample index each gap ping belongs to

        mae_lat_deg   float32  [N]              per-sample MAE in latitude  (degrees)
        mae_lon_deg   float32  [N]              per-sample MAE in longitude (degrees)
        mae_sog_kt    float32  [N]              per-sample MAE in SOG       (knots)
        mae_cog_deg   float32  [N]              per-sample MAE in COG       (degrees)
        dist_err_km   float32  [N]              per-sample mean Haversine error (km)

Key flags:
    --checkpoint  PATH    Checkpoint file           [checkpoints/best.pt]
    --split       STR     Dataset split             [val]
    --output      DIR     Output directory          [path_predictions]
    --batch_size  N       Batch size                [32]
    --num_workers N       DataLoader workers        [4]
    --device      STR     torch device string       [cuda:0]
    --no_amp              Disable bfloat16 autocast
    --max_samples N       Stop after N samples (0 = all)  [0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_ROOT = ROOT / "data_genration_and_raw_data" / "raw_data"
NS_PATH  = str(RAW_ROOT / "preprocessing" / "norm_stats" / "norm_stats.json")
MNS_PATH = str(RAW_ROOT / "preprocessing" / "norm_stats" / "micro_norm_stats.json")
IDX_DIR  = str(RAW_ROOT / "dataset_index")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="HB-Mamba v2.0 — one-day path prediction inference"
    )
    p.add_argument("--checkpoint",   default="checkpoints/best.pt")
    p.add_argument("--split",        default="val")
    p.add_argument("--output",       default="path_predictions")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--no_amp",       action="store_true")
    p.add_argument("--max_samples",  type=int, default=0,
                   help="Stop after this many samples (0 = all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    from model.hb_mamba import HBMamba

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    model = HBMamba(**ckpt["model_config"]).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys (will use random init): {missing}")
    if unexpected:
        print(f"  [warn] unexpected keys (ignored): {unexpected}")
    model.eval()

    meta = {
        "epoch"    : ckpt.get("epoch",    -1),
        "val_loss" : ckpt.get("val_loss", float("nan")),
        "config"   : ckpt["model_config"],
    }
    return model, meta


# ---------------------------------------------------------------------------
# Build DataLoader
# ---------------------------------------------------------------------------

def build_loader(split: str, batch_size: int, num_workers: int):
    from hb_mamba_dataset import HBMambaDataset, hb_mamba_collate_fn, GapMaskConfig
    from pathlib import Path as P

    idx_dir    = P(IDX_DIR)
    candidates = [
        idx_dir / f"{split}_dataset_index.json",
        idx_dir / "val_dataset_index.json",
        idx_dir / "test_dataset_index.json",
    ]
    idx_path = None
    for c in candidates:
        if c.exists():
            idx_path     = c
            actual_split = c.stem.replace("_dataset_index", "")
            break

    if idx_path is None:
        found = sorted(idx_dir.glob("*_dataset_index.json"))
        if not found:
            raise FileNotFoundError(f"No dataset index files found in {idx_dir}")
        idx_path     = found[0]
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
# Denormalisation helpers
# ---------------------------------------------------------------------------

import json as _json

def _load_norm_bounds():
    with open(NS_PATH)  as f: ns  = _json.load(f)
    with open(MNS_PATH) as f: mns = _json.load(f)
    return {
        "lat_min" : float(ns["GULF_LAT_MIN"]),
        "lat_max" : float(ns["GULF_LAT_MAX"]),
        "lon_min" : float(ns["GULF_LON_MIN"]),
        "lon_max" : float(ns["GULF_LON_MAX"]),
        "max_sog" : float(mns["MAX_SOG"]),
    }


def denorm_features(normed: np.ndarray, bounds: dict) -> dict:
    """
    Denormalise the first-8 micro features.

    Parameters
    ----------
    normed : float32  [M, 8]
        Normalised features: [lat, lon, sog, cog_sin, cog_cos, hdg_sin, hdg_cos, delta_t]

    Returns
    -------
    dict with keys: lat, lon, sog, cog, hdg  — all float32 arrays of shape [M]
    """
    lat = normed[:, 0] * (bounds["lat_max"] - bounds["lat_min"]) + bounds["lat_min"]
    lon = normed[:, 1] * (bounds["lon_max"] - bounds["lon_min"]) + bounds["lon_min"]
    sog = normed[:, 2] * bounds["max_sog"]

    # Reconstruct COG from sin/cos
    cog_rad = np.arctan2(normed[:, 3].astype(np.float64),
                         normed[:, 4].astype(np.float64))
    cog     = (np.degrees(cog_rad) % 360).astype(np.float32)

    # Reconstruct heading from sin/cos
    hdg_rad = np.arctan2(normed[:, 5].astype(np.float64),
                         normed[:, 6].astype(np.float64))
    hdg     = (np.degrees(hdg_rad) % 360).astype(np.float32)

    return {
        "lat" : lat.astype(np.float32),
        "lon" : lon.astype(np.float32),
        "sog" : sog.astype(np.float32),
        "cog" : cog,
        "hdg" : hdg,
    }


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    R   = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cog_error_deg(pred_cog, true_cog):
    """Circular mean absolute error for COG (handles 0/360 wrap)."""
    diff = np.abs(pred_cog - true_cog) % 360
    diff = np.where(diff > 180, 360 - diff, diff)
    return diff


# ---------------------------------------------------------------------------
# Gap position extraction
# ---------------------------------------------------------------------------

def extract_gap_info(mask_np: np.ndarray, padding_mask_np: np.ndarray):
    """
    Given a 1-D boolean mask (True = gap) and padding_mask, return
    (gap_start, gap_end, gap_len) for the contiguous gap block.

    If the gap is fragmented (shouldn't happen under current GapMaskConfig)
    we return the first and last masked position as start/end.
    """
    real_mask = mask_np & ~padding_mask_np      # exclude padded positions
    pos       = np.where(real_mask)[0]
    if len(pos) == 0:
        return 0, 0, 0
    return int(pos[0]), int(pos[-1]) + 1, len(pos)


# ---------------------------------------------------------------------------
# Run path prediction inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_prediction(model, loader, device, use_amp: bool, bounds: dict,
                   max_samples: int = 0):
    """
    Iterate over the loader, run model, collect gap predictions.

    Returns a dict of lists/arrays ready for np.savez.
    """
    # Per-sample scalar lists
    mmsi_list       = []
    date_list       = []
    n_day_list      = []
    gap_start_list  = []
    gap_end_list    = []
    gap_len_list    = []
    gap_type_list   = []
    mae_lat_list    = []
    mae_lon_list    = []
    mae_sog_list    = []
    mae_cog_list    = []
    dist_err_list   = []

    # Per-gap-ping lists
    pred_lat_list   = []
    pred_lon_list   = []
    pred_sog_list   = []
    pred_cog_list   = []
    true_lat_list   = []
    true_lon_list   = []
    true_sog_list   = []
    true_cog_list   = []
    sample_idx_list = []

    n_total    = 0
    n_batches  = len(loader)
    sample_cnt = 0   # global sample counter (across batches)

    for batch_idx, batch in enumerate(loader):
        # ── Move to device ────────────────────────────────────────────────
        macro_features        = batch["macro_features"].to(device, non_blocking=True)
        macro_lat_idx         = batch["macro_lat_idx"].to(device,  non_blocking=True)
        macro_lon_idx         = batch["macro_lon_idx"].to(device,  non_blocking=True)
        micro_tokens_masked   = batch["micro_tokens_masked"].to(device, non_blocking=True)
        micro_tokens_full     = batch["micro_tokens"].to(device,   non_blocking=True)
        mask                  = batch["mask"].to(device,            non_blocking=True)
        padding_mask          = batch["padding_mask"].to(device,    non_blocking=True)

        B = macro_features.shape[0]

        # ── Forward pass — encoder only ───────────────────────────────────
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            # Pass the gap-masked input to the encoder (v2.1 style)
            enc_out = model.predict(
                macro_features,
                macro_lat_idx,
                macro_lon_idx,
                micro_tokens_masked,   # gap positions are zeroed
                padding_mask,
                mask,
            )
            h_fwd = enc_out["h_fwd"]   # [B, T, 256]
            h_bwd = enc_out["h_bwd"]   # [B, T, 256]

            # ── Reconstruction head ───────────────────────────────────────
            # [B, T, 8] — predictions at ALL positions, we slice to gap only.
            # v2.3 delta head: pass linear-interp baseline so the head only
            # predicts the residual from the chord between visible neighbours.
            from model.loss_heads import compute_linear_interp_baseline
            baseline = compute_linear_interp_baseline(
                micro_tokens_masked, mask, padding_mask
            )   # [B, T, 2]
            y_hat = model.loss_heads.recon_head(h_fwd, h_bwd, baseline)

        # Cast to float32 for numpy
        y_hat_np     = y_hat.float().cpu().numpy()                  # [B, T, 8]
        micro_np     = micro_tokens_full.float().cpu().numpy()      # [B, T, 11]
        mask_np      = mask.cpu().numpy()                           # [B, T] bool
        pad_np       = padding_mask.cpu().numpy()                   # [B, T] bool

        # ── Per-sample extraction ─────────────────────────────────────────
        for b in range(B):
            g_start, g_end, g_len = extract_gap_info(mask_np[b], pad_np[b])

            if g_len == 0:
                # No gap found — skip (shouldn't happen)
                continue

            # Predicted features at gap positions
            pred_norm = y_hat_np[b, g_start:g_end, :8]   # [g_len, 8]
            true_norm = micro_np[b, g_start:g_end, :8]    # [g_len, 8]

            pred_d = denorm_features(pred_norm, bounds)
            true_d = denorm_features(true_norm, bounds)

            # Per-sample errors
            lat_err = np.mean(np.abs(pred_d["lat"] - true_d["lat"]))
            lon_err = np.mean(np.abs(pred_d["lon"] - true_d["lon"]))
            sog_err = np.mean(np.abs(pred_d["sog"] - true_d["sog"]))
            cog_err = np.mean(cog_error_deg(pred_d["cog"], true_d["cog"]))
            dist_km = np.mean(haversine_km(
                true_d["lat"], true_d["lon"],
                pred_d["lat"], pred_d["lon"],
            ))

            # Append scalar stats
            mmsi_list.append(int(batch["mmsi"][b]))
            date_list.append(str(batch["date"][b]))
            n_day_list.append(int(batch["n_day"][b]))
            gap_start_list.append(g_start)
            gap_end_list.append(g_end)
            gap_len_list.append(g_len)
            gap_type_list.append(str(batch["gap_type"][b]))
            mae_lat_list.append(float(lat_err))
            mae_lon_list.append(float(lon_err))
            mae_sog_list.append(float(sog_err))
            mae_cog_list.append(float(cog_err))
            dist_err_list.append(float(dist_km))

            # Append per-gap-ping arrays
            pred_lat_list.append(pred_d["lat"])
            pred_lon_list.append(pred_d["lon"])
            pred_sog_list.append(pred_d["sog"])
            pred_cog_list.append(pred_d["cog"])
            true_lat_list.append(true_d["lat"])
            true_lon_list.append(true_d["lon"])
            true_sog_list.append(true_d["sog"])
            true_cog_list.append(true_d["cog"])
            sample_idx_list.append(
                np.full(g_len, n_total, dtype=np.int64)
            )

            n_total += 1

        sample_cnt += B

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
            print(f"  batch {batch_idx + 1:4d} / {n_batches}  "
                  f"({n_total} samples collected)")

        if max_samples > 0 and n_total >= max_samples:
            print(f"  --max_samples {max_samples} reached — stopping early.")
            break

    # ── Assemble output dict ──────────────────────────────────────────────────
    N = n_total

    result = {
        # Per-sample scalars
        "mmsi"        : np.array(mmsi_list,      dtype=np.int64),
        "date"        : np.array(date_list,       dtype=object),
        "n_day"       : np.array(n_day_list,      dtype=np.int64),
        "gap_start"   : np.array(gap_start_list,  dtype=np.int64),
        "gap_end"     : np.array(gap_end_list,    dtype=np.int64),
        "gap_len"     : np.array(gap_len_list,    dtype=np.int64),
        "gap_type"    : np.array(gap_type_list,   dtype=object),
        "mae_lat_deg" : np.array(mae_lat_list,    dtype=np.float32),
        "mae_lon_deg" : np.array(mae_lon_list,    dtype=np.float32),
        "mae_sog_kt"  : np.array(mae_sog_list,    dtype=np.float32),
        "mae_cog_deg" : np.array(mae_cog_list,    dtype=np.float32),
        "dist_err_km" : np.array(dist_err_list,   dtype=np.float32),
        # Per-gap-ping arrays
        "pred_lat"    : np.concatenate(pred_lat_list).astype(np.float32)  if pred_lat_list else np.array([], np.float32),
        "pred_lon"    : np.concatenate(pred_lon_list).astype(np.float32)  if pred_lon_list else np.array([], np.float32),
        "pred_sog"    : np.concatenate(pred_sog_list).astype(np.float32)  if pred_sog_list else np.array([], np.float32),
        "pred_cog"    : np.concatenate(pred_cog_list).astype(np.float32)  if pred_cog_list else np.array([], np.float32),
        "true_lat"    : np.concatenate(true_lat_list).astype(np.float32)  if true_lat_list else np.array([], np.float32),
        "true_lon"    : np.concatenate(true_lon_list).astype(np.float32)  if true_lon_list else np.array([], np.float32),
        "true_sog"    : np.concatenate(true_sog_list).astype(np.float32)  if true_sog_list else np.array([], np.float32),
        "true_cog"    : np.concatenate(true_cog_list).astype(np.float32)  if true_cog_list else np.array([], np.float32),
        "sample_idx"  : np.concatenate(sample_idx_list).astype(np.int64)  if sample_idx_list else np.array([], np.int64),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
    print("HB-Mamba v2.0 — One-Day Path Prediction Inference")
    print(f"  checkpoint  : {ckpt_path}")
    print(f"  device      : {device}")
    print(f"  amp         : {use_amp}")
    print(f"  split       : {args.split}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  max_samples : {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"  output dir  : {out_dir}")
    print("=" * 65)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from {ckpt_path.name} ...")
    model, meta = load_model(str(ckpt_path), device)
    print(f"  checkpoint epoch   : {meta['epoch']}")
    print(f"  checkpoint val_loss: {meta['val_loss']:.6f}")
    print(f"\n{model}\n")

    # ── Norm stats for denormalisation ────────────────────────────────────────
    bounds = _load_norm_bounds()
    print("Normalisation bounds:")
    print(f"  lat  : [{bounds['lat_min']:.4f}, {bounds['lat_max']:.4f}] deg")
    print(f"  lon  : [{bounds['lon_min']:.4f}, {bounds['lon_max']:.4f}] deg")
    print(f"  sog  : [0, {bounds['max_sog']:.1f}] knots\n")

    # ── Build DataLoader ──────────────────────────────────────────────────────
    print(f"Building DataLoader (split='{args.split}') ...")
    loader, actual_split = build_loader(
        args.split, args.batch_size, args.num_workers
    )
    n_samples = len(loader.dataset)
    if args.max_samples > 0:
        n_samples = min(n_samples, args.max_samples)
    print(f"  samples : {len(loader.dataset)}  (running {n_samples})")
    print(f"  batches : {len(loader)}\n")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("Running path prediction ...")
    result = run_prediction(
        model, loader, device, use_amp, bounds,
        max_samples = args.max_samples,
    )

    N = len(result["mmsi"])
    print(f"\n  total samples processed : {N}")

    if N == 0:
        print("ERROR: no samples collected — check dataset split.")
        sys.exit(1)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("Aggregate gap-prediction errors (mean ± std):")
    for key, unit in [
        ("mae_lat_deg", "deg"),
        ("mae_lon_deg", "deg"),
        ("mae_sog_kt",  "kn"),
        ("mae_cog_deg", "deg"),
        ("dist_err_km", "km"),
    ]:
        v = result[key]
        print(f"  {key:14s}: {v.mean():.4f} ± {v.std():.4f}  {unit}")

    interp = result["gap_type"] == "interpolation"
    extrap = result["gap_type"] == "extrapolation"
    if interp.any():
        print(f"\n  Interpolation ({interp.sum()} gaps)  "
              f"dist_err = {result['dist_err_km'][interp].mean():.4f} km")
    if extrap.any():
        print(f"  Extrapolation ({extrap.sum()} gaps)  "
              f"dist_err = {result['dist_err_km'][extrap].mean():.4f} km")

    gap_len_arr = result["gap_len"]
    print(f"\n  Gap length  min={gap_len_arr.min()}  "
          f"mean={gap_len_arr.mean():.1f}  max={gap_len_arr.max()}  pings")
    print(f"  Total gap pings evaluated : {len(result['pred_lat'])}")
    print("─" * 55)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / "path_predictions.npz"
    np.savez(out_path, **result)
    print(f"\n  saved: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print("\nDone.")


if __name__ == "__main__":
    main()
