"""
visualize_trajectory_comparison.py — HB-Mamba Trajectory Visualizer + Diagnostics
===================================================================================

Produces two outputs:
  1. FIGURES  — PNG with continuous actual vs predicted trajectory lines
  2. DIAGNOSTICS — Three files for AI-assisted error tracing:
        gap_ping_details.csv   one row per predicted ping, all raw values
        sample_summary.csv     one row per sample, aggregate stats + baseline
        diagnostics_report.txt human+AI readable analysis of where the model fails

Figure:
  BLUE solid     — actual AIS path (all pings, time-ordered)
  ORANGE dashed  — model path (visible pings unchanged, gap filled with predictions)
  The two lines agree in visible regions and diverge inside the gap.

HOW INFERENCE WORKS ON SINGLE-DAY DATA
---------------------------------------
Each sample = ONE vessel × ONE 24-hour rolling window.
N_day = number of AIS pings sent by that vessel in that window (20–200+).
There is NO cross-day memory; each window is processed independently.

The model does BERT-style masked reconstruction — NOT autoregressive:
  1. A contiguous block (20-45% of pings) is chosen as the gap.
  2. Dynamic features [lat, lon, sog, cog_sin, cog_cos, hdg_sin, hdg_cos]
     at gap positions are zeroed; delta_t and static vessel features kept.
  3. The entire sequence (visible + zeroed gap) goes through ONE forward pass.
  4. CrossScaleInjection replaces zeroed gap pings with a learned [MASK] token.
  5. Bidirectional Mamba2 processes the sequence; both forward and backward
     context inform every gap position prediction simultaneously.
  6. ReconstructionHead outputs 8 features for every position;
     only gap-position outputs are taken as predictions.
  Visible positions in the plotted "predicted path" are actual ground-truth
  pings — the model had access to those and they are not predicted.

Usage:
    python visualize_trajectory_comparison.py \\
        --checkpoint checkpoints/best.pt --n_samples 6 --min_dist_km 200

    # compare two checkpoints side by side
    python visualize_trajectory_comparison.py \\
        --checkpoint checkpoints/best.pt \\
        --checkpoint2 checkpoints_old/best.pt \\
        --n_samples 4
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_ROOT = ROOT / "data_genration_and_raw_data" / "raw_data"
NS_PATH  = str(RAW_ROOT / "preprocessing" / "norm_stats" / "norm_stats.json")
IDX_DIR  = str(RAW_ROOT / "dataset_index")

LAT_MIN = 17.4068;  LAT_MAX = 31.4648;  LAT_RNG = LAT_MAX - LAT_MIN
LON_MIN = -98.0539; LON_MAX = -80.4330; LON_RNG = LON_MAX - LON_MIN
SOG_MAX = 30.0

def denorm_lat(v): return np.asarray(v, dtype=np.float64) * LAT_RNG + LAT_MIN
def denorm_lon(v): return np.asarray(v, dtype=np.float64) * LON_RNG + LON_MIN
def denorm_sog(v): return np.asarray(v, dtype=np.float64) * SOG_MAX


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/best.pt")
    p.add_argument("--checkpoint2", default=None,
                   help="Optional second checkpoint for side-by-side comparison.")
    p.add_argument("--split",       default="val")
    p.add_argument("--n_samples",   type=int, default=6)
    p.add_argument("--min_dist_km", type=float, default=150.0,
                   help="Min total path distance (km) to qualify as a long path.")
    p.add_argument("--min_pings",   type=int,   default=50)
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--output",      default="figs")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--pool",        type=int, default=600,
                   help="Candidate pool size before length filtering.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    d = np.radians
    a = (np.sin((d(lat2) - d(lat1)) / 2) ** 2
         + np.cos(d(lat1)) * np.cos(d(lat2))
         * np.sin((d(lon2) - d(lon1)) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def path_distance_km(lat_arr, lon_arr):
    if len(lat_arr) < 2:
        return 0.0
    return float(haversine_km(lat_arr[:-1], lon_arr[:-1],
                               lat_arr[1:],  lon_arr[1:]).sum())


def linear_interp(lat_a, lon_a, lat_b, lon_b, n_steps):
    """Linearly interpolate n_steps positions between (lat_a,lon_a) and (lat_b,lon_b)."""
    t = np.linspace(0, 1, n_steps + 2)[1:-1]  # exclude endpoints
    return lat_a + t * (lat_b - lat_a), lon_a + t * (lon_b - lon_a)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    from model.hb_mamba import HBMamba
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model = HBMamba(**ckpt["model_config"]).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"loss_heads.feature_weights"}
    unknown = [k for k in missing if k not in allowed_missing]
    if unknown or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch.  missing={unknown}  unexpected={unexpected}")
    model.eval()
    return model, ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan"))


# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------

def get_long_samples(split, n_samples, seed, min_dist_km, min_pings, pool):
    from hb_mamba_dataset import HBMambaDataset, GapMaskConfig

    idx_dir = Path(IDX_DIR)
    candidates = [
        idx_dir / f"{split}_dataset_index.json",
        idx_dir / "val_dataset_index.json",
        idx_dir / "test_dataset_index.json",
    ]
    idx_path = next((c for c in candidates if c.exists()), None)
    if idx_path is None:
        idx_path = sorted(idx_dir.glob("*_dataset_index.json"))[0]

    actual_split = idx_path.stem.replace("_dataset_index", "")
    ds = HBMambaDataset(
        dataset_index_path = str(idx_path),
        norm_stats_path    = NS_PATH,
        split              = actual_split,
        gap_config         = GapMaskConfig(min_gap_frac=0.25, max_gap_frac=0.45),
        cache_macro        = True,
        cache_micro        = False,
    )

    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(ds), size=min(pool, len(ds)), replace=False)

    scored = []
    for raw_idx in pool_idx:
        item  = ds[int(raw_idx)]
        n_day = item["n_day"]
        if n_day < min_pings:
            continue
        lat  = denorm_lat(item["micro_tokens"][:n_day, 0].numpy())
        lon  = denorm_lon(item["micro_tokens"][:n_day, 1].numpy())
        dist = path_distance_km(lat, lon)
        n_gap = int(item["mask"][:n_day].sum().item())
        if dist >= min_dist_km and n_gap >= 8:
            scored.append((dist, item))
        if len(scored) >= n_samples * 10:
            break

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:n_samples]]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, item, device, sample_id=0, ckpt_label="ckpt"):
    """
    Run the model and return a rich result dict with every raw value needed
    for both plotting and diagnostics export.
    """
    micro_cpu = item["micro_tokens"]       # [N_day, 11]
    mask_bool = item["mask"].bool()
    n_day     = item["n_day"]
    gap_mask  = mask_bool[:n_day].numpy()
    vis_mask  = ~gap_mask

    actual_lat = denorm_lat(micro_cpu[:n_day, 0].numpy())
    actual_lon = denorm_lon(micro_cpu[:n_day, 1].numpy())

    # model forward
    macro_features = item["macro_features"].unsqueeze(0).to(device)
    macro_lat_idx  = item["macro_lat_idx"].unsqueeze(0).to(device)
    macro_lon_idx  = item["macro_lon_idx"].unsqueeze(0).to(device)
    micro_input    = item["micro_tokens_masked"].unsqueeze(0).to(device)
    mask_input     = mask_bool.unsqueeze(0).to(device)

    padding_input = torch.zeros_like(mask_input)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out      = model.predict(macro_features, macro_lat_idx, macro_lon_idx,
                                 micro_input, mask=mask_input)
        # v2.3 delta head: pass per-position linear-interp baseline.
        from model.loss_heads import compute_linear_interp_baseline
        baseline = compute_linear_interp_baseline(
            micro_input, mask_input, padding_input
        )
        pred_raw = model.loss_heads.recon_head(out["h_fwd"], out["h_bwd"], baseline)

    pred = pred_raw[0].float().cpu().clamp(0.0, 1.0)   # [N_day, 8]
    pred_np = pred[:n_day].numpy()

    pred_lat = denorm_lat(pred_np[:, 0])
    pred_lon = denorm_lon(pred_np[:, 1])
    pred_sog = denorm_sog(pred_np[:, 2])

    # linear interpolation baseline
    gap_indices  = np.where(gap_mask)[0]
    vis_indices  = np.where(vis_mask)[0]
    gap_start_i  = int(gap_indices[0])
    gap_end_i    = int(gap_indices[-1])
    n_gap        = len(gap_indices)

    before_gap = vis_indices[vis_indices < gap_start_i]
    after_gap  = vis_indices[vis_indices > gap_end_i]
    has_before = len(before_gap) > 0
    has_after  = len(after_gap)  > 0

    prev_lat = actual_lat[int(before_gap[-1])] if has_before else actual_lat[0]
    prev_lon = actual_lon[int(before_gap[-1])] if has_before else actual_lon[0]
    next_lat = actual_lat[int(after_gap[0])]   if has_after  else actual_lat[-1]
    next_lon = actual_lon[int(after_gap[0])]   if has_after  else actual_lon[-1]

    li_lat, li_lon = linear_interp(prev_lat, prev_lon, next_lat, next_lon, n_gap)

    # per-ping errors
    gap_actual_lat = actual_lat[gap_mask]
    gap_actual_lon = actual_lon[gap_mask]
    gap_pred_lat   = pred_lat[gap_mask]
    gap_pred_lon   = pred_lon[gap_mask]

    model_errs  = haversine_km(gap_actual_lat, gap_actual_lon,
                               gap_pred_lat,   gap_pred_lon)
    linear_errs = haversine_km(gap_actual_lat, gap_actual_lon, li_lat, li_lon)

    # stitch predicted path: actual at visible, predicted at gap
    stitched_lat = actual_lat.copy()
    stitched_lon = actual_lon.copy()
    stitched_lat[gap_mask] = gap_pred_lat
    stitched_lon[gap_mask] = gap_pred_lon

    total_dist = path_distance_km(actual_lat, actual_lon)
    gap_chord  = haversine_km(prev_lat, prev_lon, next_lat, next_lon)

    # raw normalized features for diagnostics
    act_norm = micro_cpu[:n_day].numpy()   # [n_day, 11]

    return {
        # metadata
        "sample_id"    : sample_id,
        "ckpt_label"   : ckpt_label,
        "mmsi"         : item["mmsi"],
        "date"         : item["date"],
        "n_day"        : n_day,
        "gap_type"     : item["gap_type"],
        "n_gap"        : n_gap,
        "gap_start_i"  : gap_start_i,
        "gap_end_i"    : gap_end_i,
        "total_dist_km": float(total_dist),
        "gap_chord_km" : float(gap_chord),
        "gap_frac"     : float(n_gap / n_day),
        # full arrays (for plotting)
        "actual_lat"   : actual_lat,
        "actual_lon"   : actual_lon,
        "stitched_lat" : stitched_lat,
        "stitched_lon" : stitched_lon,
        "vis_lat"      : actual_lat[vis_mask],
        "vis_lon"      : actual_lon[vis_mask],
        # gap arrays (for diagnostics)
        "gap_indices"       : gap_indices,
        "gap_actual_lat"    : gap_actual_lat,
        "gap_actual_lon"    : gap_actual_lon,
        "gap_pred_lat"      : gap_pred_lat,
        "gap_pred_lon"      : gap_pred_lon,
        "gap_linear_lat"    : li_lat,
        "gap_linear_lon"    : li_lon,
        "model_errs_km"     : model_errs,
        "linear_errs_km"    : linear_errs,
        # anchor pings
        "prev_vis_lat"      : prev_lat,
        "prev_vis_lon"      : prev_lon,
        "next_vis_lat"      : next_lat,
        "next_vis_lon"      : next_lon,
        # raw normalized predictions (all n_day positions, columns 0..7)
        "pred_norm"         : pred_np,
        "actual_norm"       : act_norm,
        # aggregate stats
        "mean_km"   : float(model_errs.mean()),
        "median_km" : float(np.median(model_errs)),
        "max_km"    : float(model_errs.max()),
        "std_km"    : float(model_errs.std()),
        "linear_mean_km"   : float(linear_errs.mean()),
        "linear_median_km" : float(np.median(linear_errs)),
    }


# ---------------------------------------------------------------------------
# Diagnostics export
# ---------------------------------------------------------------------------

def save_diagnostics(results, out_dir: Path, ckpt_label="ckpt"):
    """
    Saves three files:
      gap_ping_details_{ckpt_label}.csv  — one row per gap ping, all raw values
      sample_summary_{ckpt_label}.csv    — one row per sample, aggregate stats
      diagnostics_report_{ckpt_label}.txt — structured text for AI-assisted tracing
    """

    # ── 1. gap_ping_details.csv ────────────────────────────────────────────
    ping_csv = out_dir / f"gap_ping_details_{ckpt_label}.csv"
    ping_fields = [
        "sample_id", "mmsi", "date", "n_day", "gap_type",
        "gap_start_i", "gap_end_i", "n_gap", "gap_frac",
        "total_dist_km", "gap_chord_km",
        # per-ping positional fields
        "ping_idx",          # absolute index within the n_day sequence
        "gap_ping_idx",      # 0-based index within the gap
        "frac_through_gap",  # 0.0 = start of gap, 1.0 = end of gap
        # actual position (denormalised)
        "actual_lat", "actual_lon",
        # model prediction (denormalised)
        "pred_lat", "pred_lon",
        # linear interpolation baseline (denormalised)
        "linear_lat", "linear_lon",
        # errors
        "model_err_km", "linear_err_km",
        "model_beats_linear",         # 1 if model < linear, 0 otherwise
        "model_err_vs_linear_ratio",  # model_err / linear_err  (<1 = model wins)
        # anchors (same for all pings in a gap)
        "prev_vis_lat", "prev_vis_lon",
        "next_vis_lat", "next_vis_lon",
        # raw normalised features (what model saw and predicted, for debugging)
        "actual_lat_norm", "actual_lon_norm",
        "actual_sog_norm",
        "actual_cog_sin_norm", "actual_cog_cos_norm",
        "actual_hdg_sin_norm", "actual_hdg_cos_norm",
        "actual_delta_t_norm",
        "pred_lat_norm", "pred_lon_norm",
        "pred_sog_norm",
        "pred_cog_sin_norm", "pred_cog_cos_norm",
        "pred_hdg_sin_norm", "pred_hdg_cos_norm",
        "pred_delta_t_norm",
        # derived
        "pred_sog_knots",
        "lat_err_deg", "lon_err_deg",
    ]

    with open(ping_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ping_fields)
        writer.writeheader()
        for r in results:
            ng = r["n_gap"]
            for k in range(ng):
                ping_abs = int(r["gap_indices"][k])
                me  = float(r["model_errs_km"][k])
                le  = float(r["linear_errs_km"][k])
                ratio = me / (le + 1e-6)
                beats = 1 if me < le else 0

                an = r["actual_norm"][ping_abs]   # [11]
                pn = r["pred_norm"][ping_abs]      # [8]

                row = {
                    "sample_id"     : r["sample_id"],
                    "mmsi"          : r["mmsi"],
                    "date"          : r["date"],
                    "n_day"         : r["n_day"],
                    "gap_type"      : r["gap_type"],
                    "gap_start_i"   : r["gap_start_i"],
                    "gap_end_i"     : r["gap_end_i"],
                    "n_gap"         : r["n_gap"],
                    "gap_frac"      : f"{r['gap_frac']:.3f}",
                    "total_dist_km" : f"{r['total_dist_km']:.2f}",
                    "gap_chord_km"  : f"{r['gap_chord_km']:.2f}",
                    "ping_idx"      : ping_abs,
                    "gap_ping_idx"  : k,
                    "frac_through_gap": f"{k / max(ng - 1, 1):.3f}",
                    "actual_lat"    : f"{r['gap_actual_lat'][k]:.5f}",
                    "actual_lon"    : f"{r['gap_actual_lon'][k]:.5f}",
                    "pred_lat"      : f"{r['gap_pred_lat'][k]:.5f}",
                    "pred_lon"      : f"{r['gap_pred_lon'][k]:.5f}",
                    "linear_lat"    : f"{r['gap_linear_lat'][k]:.5f}",
                    "linear_lon"    : f"{r['gap_linear_lon'][k]:.5f}",
                    "model_err_km"  : f"{me:.3f}",
                    "linear_err_km" : f"{le:.3f}",
                    "model_beats_linear"        : beats,
                    "model_err_vs_linear_ratio" : f"{ratio:.3f}",
                    "prev_vis_lat"  : f"{r['prev_vis_lat']:.5f}",
                    "prev_vis_lon"  : f"{r['prev_vis_lon']:.5f}",
                    "next_vis_lat"  : f"{r['next_vis_lat']:.5f}",
                    "next_vis_lon"  : f"{r['next_vis_lon']:.5f}",
                    "actual_lat_norm"     : f"{float(an[0]):.5f}",
                    "actual_lon_norm"     : f"{float(an[1]):.5f}",
                    "actual_sog_norm"     : f"{float(an[2]):.5f}",
                    "actual_cog_sin_norm" : f"{float(an[3]):.5f}",
                    "actual_cog_cos_norm" : f"{float(an[4]):.5f}",
                    "actual_hdg_sin_norm" : f"{float(an[5]):.5f}",
                    "actual_hdg_cos_norm" : f"{float(an[6]):.5f}",
                    "actual_delta_t_norm" : f"{float(an[7]):.5f}",
                    "pred_lat_norm"       : f"{float(pn[0]):.5f}",
                    "pred_lon_norm"       : f"{float(pn[1]):.5f}",
                    "pred_sog_norm"       : f"{float(pn[2]):.5f}",
                    "pred_cog_sin_norm"   : f"{float(pn[3]):.5f}",
                    "pred_cog_cos_norm"   : f"{float(pn[4]):.5f}",
                    "pred_hdg_sin_norm"   : f"{float(pn[5]):.5f}",
                    "pred_hdg_cos_norm"   : f"{float(pn[6]):.5f}",
                    "pred_delta_t_norm"   : f"{float(pn[7]):.5f}",
                    "pred_sog_knots"      : f"{float(denorm_sog(pn[2])):.2f}",
                    "lat_err_deg"         : f"{abs(r['gap_actual_lat'][k] - r['gap_pred_lat'][k]):.5f}",
                    "lon_err_deg"         : f"{abs(r['gap_actual_lon'][k] - r['gap_pred_lon'][k]):.5f}",
                }
                writer.writerow(row)

    # ── 2. sample_summary.csv ──────────────────────────────────────────────
    summ_csv = out_dir / f"sample_summary_{ckpt_label}.csv"
    summ_fields = [
        "sample_id", "mmsi", "date", "n_day", "gap_type",
        "n_gap", "gap_frac", "gap_start_i", "gap_end_i",
        "total_dist_km", "gap_chord_km",
        "mean_err_km", "median_err_km", "max_err_km", "std_err_km",
        "linear_mean_err_km", "linear_median_err_km",
        "model_beats_linear_pct",    # % of gap pings where model < linear
        "improvement_ratio",         # model_mean / linear_mean  (<1 = model wins)
        "worst_ping_idx",            # ping index with the highest model error
        "worst_err_km",
        "worst_frac_through_gap",    # 0=start 1=end of gap (was error at start/middle/end?)
    ]
    with open(summ_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summ_fields)
        writer.writeheader()
        for r in results:
            beats_pct = float((r["model_errs_km"] < r["linear_errs_km"]).mean() * 100)
            imp_ratio = r["mean_km"] / (r["linear_mean_km"] + 1e-6)
            worst_k   = int(np.argmax(r["model_errs_km"]))
            ng        = r["n_gap"]
            writer.writerow({
                "sample_id"              : r["sample_id"],
                "mmsi"                   : r["mmsi"],
                "date"                   : r["date"],
                "n_day"                  : r["n_day"],
                "gap_type"               : r["gap_type"],
                "n_gap"                  : ng,
                "gap_frac"               : f"{r['gap_frac']:.3f}",
                "gap_start_i"            : r["gap_start_i"],
                "gap_end_i"              : r["gap_end_i"],
                "total_dist_km"          : f"{r['total_dist_km']:.2f}",
                "gap_chord_km"           : f"{r['gap_chord_km']:.2f}",
                "mean_err_km"            : f"{r['mean_km']:.2f}",
                "median_err_km"          : f"{r['median_km']:.2f}",
                "max_err_km"             : f"{r['max_km']:.2f}",
                "std_err_km"             : f"{r['std_km']:.2f}",
                "linear_mean_err_km"     : f"{r['linear_mean_km']:.2f}",
                "linear_median_err_km"   : f"{r['linear_median_km']:.2f}",
                "model_beats_linear_pct" : f"{beats_pct:.1f}",
                "improvement_ratio"      : f"{imp_ratio:.3f}",
                "worst_ping_idx"         : int(r["gap_indices"][worst_k]),
                "worst_err_km"           : f"{r['max_km']:.2f}",
                "worst_frac_through_gap" : f"{worst_k / max(ng - 1, 1):.3f}",
            })

    # ── 3. diagnostics_report.txt ──────────────────────────────────────────
    rpt_path = out_dir / f"diagnostics_report_{ckpt_label}.txt"
    all_model  = np.concatenate([r["model_errs_km"]  for r in results])
    all_linear = np.concatenate([r["linear_errs_km"] for r in results])
    beats_all  = (all_model < all_linear).mean() * 100

    with open(rpt_path, "w") as f:
        W = f.write

        W("=" * 72 + "\n")
        W(f"HB-MAMBA GAP PREDICTION DIAGNOSTICS  —  {ckpt_label}\n")
        W("=" * 72 + "\n\n")

        W("HOW TO READ THIS REPORT\n")
        W("-" * 40 + "\n")
        W("Each sample = one vessel × one 24-hour AIS window.\n")
        W("Model predicts all gap pings simultaneously (BERT-style, not autoregressive).\n")
        W("Baseline = linear interpolation between last visible before gap and\n")
        W("           first visible after gap (dumb straight-line fill).\n")
        W("improvement_ratio < 1.0  means model beats the baseline.\n")
        W("improvement_ratio > 1.0  means baseline beats the model (bad).\n\n")

        W("GLOBAL SUMMARY\n")
        W("-" * 40 + "\n")
        W(f"  samples evaluated       : {len(results)}\n")
        W(f"  total gap pings         : {len(all_model)}\n")
        W(f"  model  mean  err km     : {all_model.mean():.2f}\n")
        W(f"  model  median err km    : {float(np.median(all_model)):.2f}\n")
        W(f"  model  max   err km     : {all_model.max():.2f}\n")
        W(f"  model  std   err km     : {all_model.std():.2f}\n")
        W(f"  linear mean  err km     : {all_linear.mean():.2f}  (baseline)\n")
        W(f"  linear median err km    : {float(np.median(all_linear)):.2f}  (baseline)\n")
        W(f"  model beats linear      : {beats_all:.1f}% of pings\n")
        W(f"  overall improvement     : {all_model.mean() / all_linear.mean():.3f}x  "
          f"(< 1.0 = model better)\n\n")

        W("ERROR DISTRIBUTION\n")
        W("-" * 40 + "\n")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            W(f"  p{p:02d}  model={np.percentile(all_model, p):7.2f} km   "
              f"linear={np.percentile(all_linear, p):7.2f} km\n")
        W("\n")

        W("PER-SAMPLE BREAKDOWN\n")
        W("-" * 40 + "\n")
        header = (f"{'#':>3}  {'MMSI':<12} {'date':<12} {'n_day':>6} "
                  f"{'n_gap':>6} {'gap%':>5} {'type':<6} "
                  f"{'model_mean':>11} {'model_med':>10} {'model_max':>10} "
                  f"{'linear_mean':>12} {'improv':>7} {'beats%':>7}\n")
        W(header)
        W("-" * len(header) + "\n")
        for r in results:
            beats_pct = float((r["model_errs_km"] < r["linear_errs_km"]).mean() * 100)
            imp = r["mean_km"] / (r["linear_mean_km"] + 1e-6)
            W(f"{r['sample_id']:>3}  {str(r['mmsi']):<12} {r['date']:<12} "
              f"{r['n_day']:>6} {r['n_gap']:>6} "
              f"{r['gap_frac']*100:>4.0f}% {r['gap_type'][:5]:<6} "
              f"{r['mean_km']:>11.2f} {r['median_km']:>10.2f} {r['max_km']:>10.2f} "
              f"{r['linear_mean_km']:>12.2f} {imp:>7.3f} {beats_pct:>6.1f}%\n")
        W("\n")

        W("WORST SAMPLES (sorted by model_mean_km descending)\n")
        W("-" * 40 + "\n")
        sorted_r = sorted(results, key=lambda x: x["mean_km"], reverse=True)
        for r in sorted_r[:5]:
            beats_pct = float((r["model_errs_km"] < r["linear_errs_km"]).mean() * 100)
            worst_k = int(np.argmax(r["model_errs_km"]))
            W(f"\n  sample_id={r['sample_id']}  MMSI={r['mmsi']}  date={r['date']}\n")
            W(f"    n_day={r['n_day']}  n_gap={r['n_gap']}  "
              f"gap_frac={r['gap_frac']:.2f}  type={r['gap_type']}\n")
            W(f"    total_path={r['total_dist_km']:.1f} km  "
              f"gap_chord={r['gap_chord_km']:.1f} km\n")
            W(f"    model  : mean={r['mean_km']:.2f} km  "
              f"median={r['median_km']:.2f} km  max={r['max_km']:.2f} km\n")
            W(f"    linear : mean={r['linear_mean_km']:.2f} km  "
              f"median={r['linear_median_km']:.2f} km\n")
            W(f"    beats linear {beats_pct:.1f}% of gap pings\n")
            W(f"    worst ping: idx={r['gap_indices'][worst_k]}  "
              f"frac_through_gap={worst_k / max(r['n_gap']-1,1):.2f}  "
              f"err={r['max_km']:.2f} km\n")
            W(f"    anchor before: ({r['prev_vis_lat']:.4f}, {r['prev_vis_lon']:.4f})\n")
            W(f"    anchor after : ({r['next_vis_lat']:.4f}, {r['next_vis_lon']:.4f})\n")
            W(f"    first 5 gap pings (actual | predicted | linear | model_err | linear_err):\n")
            for k in range(min(5, r["n_gap"])):
                W(f"      [{k}] actual=({r['gap_actual_lat'][k]:.4f},{r['gap_actual_lon'][k]:.4f}) "
                  f"pred=({r['gap_pred_lat'][k]:.4f},{r['gap_pred_lon'][k]:.4f}) "
                  f"linear=({r['gap_linear_lat'][k]:.4f},{r['gap_linear_lon'][k]:.4f}) "
                  f"model={r['model_errs_km'][k]:.2f}km  linear={r['linear_errs_km'][k]:.2f}km\n")

        W("\n\nFEATURE-LEVEL ANALYSIS (averaged over all gap pings)\n")
        W("-" * 40 + "\n")
        feat_names = ["lat", "lon", "sog", "cog_sin", "cog_cos",
                      "hdg_sin", "hdg_cos", "delta_t"]
        for fi, fname in enumerate(feat_names):
            act_vals  = np.concatenate([r["actual_norm"][r["gap_indices"], fi]
                                        for r in results])
            pred_vals = np.concatenate([r["pred_norm"][r["gap_indices"], fi]
                                        for r in results])
            mae = float(np.abs(act_vals - pred_vals).mean())
            bias = float((pred_vals - act_vals).mean())
            W(f"  {fname:<12}  MAE={mae:.5f}  bias={bias:+.5f}  "
              f"act_mean={act_vals.mean():.3f}  pred_mean={pred_vals.mean():.3f}\n")

        W("\n\nERROR VS GAP POSITION (does error grow toward gap center?)\n")
        W("-" * 40 + "\n")
        # bin by frac_through_gap into 5 buckets
        buckets = 5
        bucket_errs = [[] for _ in range(buckets)]
        for r in results:
            ng = r["n_gap"]
            for k in range(ng):
                b = min(int(k / ng * buckets), buckets - 1)
                bucket_errs[b].append(r["model_errs_km"][k])
        for b, errs in enumerate(bucket_errs):
            if errs:
                lo = b / buckets; hi = (b + 1) / buckets
                W(f"  gap frac [{lo:.1f}-{hi:.1f}]  n={len(errs):>5}  "
                  f"mean={np.mean(errs):.2f} km  median={np.median(errs):.2f} km\n")

        W("\n\nGUIDELINES FOR NEXT STEPS\n")
        W("-" * 40 + "\n")
        imp_overall = all_model.mean() / all_linear.mean()
        if imp_overall > 1.5:
            W("  !! Model is WORSE than linear interpolation (ratio > 1.5).\n")
            W("     Likely causes: too few epochs, wrong masking, cheat-through leak.\n")
        elif imp_overall > 1.0:
            W("  !  Model is marginally worse or equal to linear interpolation.\n")
            W("     Try: more epochs, feature-weighted Huber, displacement prediction.\n")
        else:
            W("  OK Model beats linear interpolation overall.\n")
            err_at_center = np.mean(bucket_errs[buckets // 2]) if bucket_errs[buckets // 2] else 0
            err_at_edge   = np.mean(bucket_errs[0]) if bucket_errs[0] else 0
            if err_at_center > err_at_edge * 1.5:
                W("     Gap center errors >> edge errors → model loses temporal context\n")
                W("     mid-gap. Consider: longer context window, larger model, or\n")
                W("     autoregressive decoding for long gaps.\n")

        W("\n" + "=" * 72 + "\n")

    print(f"  gap_ping_details → {ping_csv.name}")
    print(f"  sample_summary   → {summ_csv.name}")
    print(f"  diagnostics_report → {rpt_path.name}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_panel(ax, r, color_pred="darkorange", ckpt_label="Model"):
    lat_a = r["actual_lat"];   lon_a = r["actual_lon"]
    lat_s = r["stitched_lat"]; lon_s = r["stitched_lon"]
    gs    = r["gap_start_i"];  ge    = r["gap_end_i"]

    # gap region shading
    ax.axvspan(min(lon_a[gs], lon_a[ge]), max(lon_a[gs], lon_a[ge]),
               alpha=0.08, color="grey", zorder=0)

    # actual path (blue)
    ax.plot(lon_a, lat_a, color="steelblue", lw=2.0, alpha=0.85, zorder=2)
    ax.scatter(r["vis_lon"], r["vis_lat"], c="steelblue", s=6, zorder=3, alpha=0.5)
    ax.scatter(r["gap_actual_lon"], r["gap_actual_lat"],
               c="red", s=22, zorder=4, alpha=0.9, marker="o")

    # predicted / stitched path
    ax.plot(lon_s, lat_s, color=color_pred, lw=2.0, alpha=0.85,
            linestyle="--", zorder=5)
    ax.scatter(r["gap_pred_lon"], r["gap_pred_lat"],
               c=color_pred, s=35, zorder=6, marker="*",
               edgecolors="black", linewidths=0.3)

    # error lines
    for k in range(r["n_gap"]):
        ax.plot([r["gap_actual_lon"][k], r["gap_pred_lon"][k]],
                [r["gap_actual_lat"][k], r["gap_pred_lat"][k]],
                color="grey", lw=0.6, alpha=0.5, zorder=1)

    # gap boundary markers
    ax.scatter([lon_a[gs], lon_a[ge]], [lat_a[gs], lat_a[ge]],
               marker="^", c="black", s=60, zorder=7)

    imp = r["mean_km"] / (r["linear_mean_km"] + 1e-6)
    ax.set_title(
        f"MMSI {r['mmsi']}  {r['date']}  n={r['n_day']}  "
        f"gap={r['n_gap']} ({r['gap_type'][:5]})\n"
        f"path={r['total_dist_km']:.0f}km  chord={r['gap_chord_km']:.0f}km  "
        f"mean={r['mean_km']:.1f}km  med={r['median_km']:.1f}km  "
        f"vs_linear={imp:.2f}x",
        fontsize=7.5,
    )
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)

    all_lon = np.concatenate([lon_a, lon_s])
    all_lat = np.concatenate([lat_a, lat_s])
    plon = max((all_lon.max() - all_lon.min()) * 0.06, 0.1)
    plat = max((all_lat.max() - all_lat.min()) * 0.06, 0.1)
    ax.set_xlim(all_lon.min() - plon, all_lon.max() + plon)
    ax.set_ylim(all_lat.min() - plat, all_lat.max() + plat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    def _abs(p): return ROOT / p if not Path(p).is_absolute() else Path(p)

    ckpt1_path = _abs(args.checkpoint)
    ckpt2_path = _abs(args.checkpoint2) if args.checkpoint2 else None

    out_dir = _abs(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cols = 2 if ckpt2_path else 1

    print("=" * 65)
    print("HB-Mamba — Trajectory Comparison + Diagnostics Export")
    print(f"  checkpoint  : {ckpt1_path.name}")
    if ckpt2_path:
        print(f"  checkpoint2 : {ckpt2_path.name}")
    print(f"  split       : {args.split}   n_samples={args.n_samples}")
    print(f"  min_dist_km : {args.min_dist_km}   min_pings={args.min_pings}")
    print(f"  output      : {out_dir}")
    print("=" * 65)

    model1, epoch1, vloss1 = load_model(str(ckpt1_path), device)
    print(f"\n  ckpt1  epoch={epoch1}  val_loss={vloss1:.6f}")
    model2 = epoch2 = vloss2 = None
    if ckpt2_path:
        model2, epoch2, vloss2 = load_model(str(ckpt2_path), device)
        print(f"  ckpt2  epoch={epoch2}  val_loss={vloss2:.6f}")

    print(f"\nSearching for {args.n_samples} long paths ...")
    samples = get_long_samples(args.split, args.n_samples, args.seed,
                               args.min_dist_km, args.min_pings, args.pool)
    if not samples:
        print("No qualifying samples found — lower --min_dist_km or --min_pings.")
        return
    print(f"  found {len(samples)} samples\n")

    n = len(samples)
    fig, axes = plt.subplots(n, n_cols,
                             figsize=(9.5 * n_cols, 6.0 * n),
                             constrained_layout=True)
    axes = np.atleast_2d(axes) if n_cols > 1 else np.atleast_1d(axes).reshape(n, 1)

    results1 = []
    results2 = []

    print(f"{'#':>3}  {'MMSI':<12} {'pings':>6} {'dist_km':>9} "
          f"{'gap':>5} {'mean_km':>9} {'vs_linear':>10}")
    print("-" * 60)

    for i, item in enumerate(samples):
        r1 = run_inference(model1, item, device, sample_id=i+1,
                           ckpt_label=ckpt1_path.stem)
        results1.append(r1)

        imp = r1["mean_km"] / (r1["linear_mean_km"] + 1e-6)
        print(f"{i+1:>3}  {str(r1['mmsi']):<12} {r1['n_day']:>6} "
              f"{r1['total_dist_km']:>9.1f} {r1['n_gap']:>5} "
              f"{r1['mean_km']:>9.2f} {imp:>10.3f}x")

        draw_panel(axes[i, 0], r1, color_pred="darkorange",
                   ckpt_label=ckpt1_path.stem)

        if ckpt2_path:
            r2 = run_inference(model2, item, device, sample_id=i+1,
                               ckpt_label=ckpt2_path.stem)
            results2.append(r2)
            draw_panel(axes[i, 1], r2, color_pred="deeppink",
                       ckpt_label=ckpt2_path.stem)

    print("-" * 60)

    legend_handles = [
        Line2D([0], [0], color="steelblue", lw=2,
               label="Actual path (ground truth)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="red", ms=9,
               label="Actual gap pings (hidden from model)"),
        Line2D([0], [0], color="darkorange", lw=2, ls="--",
               label="Model reconstructed path"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="darkorange", ms=12,
               label="Predicted gap pings"),
        Line2D([0], [0], color="grey", lw=1,
               label="Error line (actual → predicted)"),
        mpatches.Patch(color="grey", alpha=0.15, label="Gap region"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, -0.01))

    title = (f"HB-Mamba Gap Prediction  |  {ckpt1_path.name}  "
             f"epoch={epoch1}  val_loss={vloss1:.4f}")
    if ckpt2_path:
        title += (f"\nvs  {ckpt2_path.name}  "
                  f"epoch={epoch2}  val_loss={vloss2:.4f}")
    fig.suptitle(title, fontsize=11, fontweight="bold")

    out_png = out_dir / "trajectory_comparison.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  figure saved: {out_png}")

    print("\nExporting diagnostics ...")
    save_diagnostics(results1, out_dir, ckpt_label=ckpt1_path.stem)
    if results2:
        save_diagnostics(results2, out_dir, ckpt_label=ckpt2_path.stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
