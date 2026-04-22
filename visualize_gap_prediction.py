"""
visualize_gap_prediction.py — HB-Mamba v2.0/v2.1 Gap Prediction Visualizer
==========================================================================

Plots predicted-vs-actual trajectories for N validation samples. Uses the
correct v2.1 inference path:

    - micro_tokens_masked   (dataset zeros features 0..7 at gap positions,
                             preserves static features 8..10, fixes delta_t)
    - mask passed to predict so CrossScaleInjection injects the learned
      [MASK] embedding and MicroEncoder computes H from visible pings only
    - reconstruction output clamped to [0, 1] here (the recon head itself
      no longer clamps)

Usage:
    python visualize_gap_prediction.py \\
        --checkpoint checkpoints/best.pt --split val --n_samples 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_ROOT = ROOT / "data_genration_and_raw_data" / "raw_data"
NS_PATH  = str(RAW_ROOT / "preprocessing" / "norm_stats" / "norm_stats.json")
IDX_DIR  = str(RAW_ROOT / "dataset_index")

LAT_MIN = 17.4068;  LAT_MAX = 31.4648;  LAT_RNG = LAT_MAX - LAT_MIN
LON_MIN = -98.0539; LON_MAX = -80.4330; LON_RNG = LON_MAX - LON_MIN

def denorm_lat(v): return np.asarray(v) * LAT_RNG + LAT_MIN
def denorm_lon(v): return np.asarray(v) * LON_RNG + LON_MIN


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--split",      default="val")
    p.add_argument("--n_samples",  type=int, default=8)
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--output",     default="figs")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--grid_cols",  type=int, default=2,
                   help="Number of columns in the figure grid.")
    return p.parse_args()


def load_model(checkpoint_path, device):
    from model.hb_mamba import HBMamba
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model = HBMamba(**ckpt["model_config"]).to(device)
    # strict=False: tolerates new buffers added after the ckpt was saved
    # (e.g. loss_heads.feature_weights — loss heads aren't used at inference anyway).
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"loss_heads.feature_weights"}
    unknown = [k for k in missing if k not in allowed_missing]
    if unknown or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch.  missing={unknown}  unexpected={unexpected}"
        )
    model.eval()
    return model, ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan"))


def get_samples(split, n_samples, seed):
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
        gap_config         = GapMaskConfig(min_gap_frac=0.20, max_gap_frac=0.40),
        cache_macro        = True,
        cache_micro        = False,
    )

    rng = np.random.default_rng(seed)
    all_idx = rng.choice(len(ds), size=min(n_samples * 20, len(ds)), replace=False)

    samples = []
    for idx in all_idx:
        item = ds[int(idx)]
        n_gap = item["mask"].sum().item()
        if item["n_day"] >= 30 and n_gap >= 5:
            samples.append(item)
        if len(samples) == n_samples:
            break

    return samples


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a.clip(0, 1)))


@torch.no_grad()
def run_inference(model, item, device):
    """Correct v2.1 inference — matches the training-time forward pass."""
    micro_cpu = item["micro_tokens"]
    mask_bool = item["mask"].bool()
    n_day     = item["n_day"]
    gap_mask  = mask_bool[:n_day]
    vis_mask  = ~gap_mask

    macro_features = item["macro_features"].unsqueeze(0).to(device)
    macro_lat_idx  = item["macro_lat_idx"].unsqueeze(0).to(device)
    macro_lon_idx  = item["macro_lon_idx"].unsqueeze(0).to(device)
    micro_input    = item["micro_tokens_masked"].unsqueeze(0).to(device)
    mask_input     = mask_bool.unsqueeze(0).to(device)

    # No padding in single-sample inference; baseline uses only the mask.
    padding_input = torch.zeros_like(mask_input)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out      = model.predict(macro_features, macro_lat_idx, macro_lon_idx,
                                 micro_input, mask=mask_input)
        # v2.3 delta head: supply per-position linear-interp baseline so the
        # recon head only predicts the residual from the chord through the
        # visible anchors.
        from model.loss_heads import compute_linear_interp_baseline
        baseline = compute_linear_interp_baseline(
            micro_input, mask_input, padding_input
        )   # [1, N_day, 2]
        pred_all = model.loss_heads.recon_head(out["h_fwd"], out["h_bwd"], baseline)

    pred_all = pred_all[0].float().cpu().clamp(0.0, 1.0)  # [N_day, 8]

    gap_indices = torch.where(gap_mask)[0]
    vis_indices = torch.where(vis_mask)[0]
    gap_start   = int(gap_indices[0])
    gap_end     = int(gap_indices[-1])
    before_gap  = vis_indices[vis_indices < gap_start]
    after_gap   = vis_indices[vis_indices > gap_end]

    def _at(i, col, denorm):
        if len(i) == 0:
            return None
        return denorm(float(micro_cpu[int(i), col]))

    return {
        "mmsi"                : item["mmsi"],
        "date"                : item["date"],
        "n_day"               : n_day,
        "gap_type"            : item["gap_type"],
        "n_gap"               : int(gap_mask.sum().item()),
        "actual_lat"          : denorm_lat(micro_cpu[:n_day, 0].numpy()),
        "actual_lon"          : denorm_lon(micro_cpu[:n_day, 1].numpy()),
        "vis_lat"             : denorm_lat(micro_cpu[:n_day][vis_mask, 0].numpy()),
        "vis_lon"             : denorm_lon(micro_cpu[:n_day][vis_mask, 1].numpy()),
        "gap_lat_act"         : denorm_lat(micro_cpu[:n_day][gap_mask, 0].numpy()),
        "gap_lon_act"         : denorm_lon(micro_cpu[:n_day][gap_mask, 1].numpy()),
        "gap_lat_pred"        : denorm_lat(pred_all[:n_day][gap_mask, 0].numpy()),
        "gap_lon_pred"        : denorm_lon(pred_all[:n_day][gap_mask, 1].numpy()),
        "last_vis_lat"        : _at(before_gap[-1:], 0, denorm_lat) if len(before_gap) else None,
        "last_vis_lon"        : _at(before_gap[-1:], 1, denorm_lon) if len(before_gap) else None,
        "first_vis_after_lat" : _at(after_gap[:1],  0, denorm_lat) if len(after_gap)  else None,
        "first_vis_after_lon" : _at(after_gap[:1],  1, denorm_lon) if len(after_gap)  else None,
    }


def draw_panel(ax, r):
    actual_lat = r["actual_lat"]; actual_lon = r["actual_lon"]
    vis_lat    = r["vis_lat"];    vis_lon    = r["vis_lon"]
    gap_la     = r["gap_lat_act"]; gap_lo    = r["gap_lon_act"]
    pred_la    = r["gap_lat_pred"]; pred_lo  = r["gap_lon_pred"]

    ax.plot(actual_lon, actual_lat, color="steelblue", lw=0.8, alpha=0.35, zorder=1)
    ax.scatter(vis_lon, vis_lat, c="green", s=10, zorder=3, alpha=0.8, label="Visible")
    ax.scatter(gap_lo, gap_la,   c="red",   s=20, zorder=4, alpha=0.9, marker="o", label="Actual gap")
    ax.scatter(pred_lo, pred_la, c="darkorange", s=32, zorder=5, marker="*",
               alpha=0.95, edgecolors="black", linewidths=0.3, label="Predicted")

    for i in range(len(gap_la)):
        ax.plot([gap_lo[i], pred_lo[i]], [gap_la[i], pred_la[i]],
                color="grey", lw=0.5, alpha=0.5, zorder=2)

    if len(gap_la) > 0:
        errs   = haversine_km(gap_la, gap_lo, pred_la, pred_lo)
        mean_e = float(errs.mean())
        med_e  = float(np.median(errs))
        max_e  = float(errs.max())
    else:
        mean_e = med_e = max_e = float("nan")

    ax.set_title(
        f"MMSI {r['mmsi']}  {r['date']}  "
        f"n_day={r['n_day']}  gap={r['n_gap']} ({r['gap_type'][:5]})\n"
        f"mean={mean_e:.1f} km   median={med_e:.1f} km   max={max_e:.1f} km",
        fontsize=8,
    )
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)

    all_lo = np.concatenate([actual_lon, pred_lo]) if len(pred_lo) else actual_lon
    all_la = np.concatenate([actual_lat, pred_la]) if len(pred_la) else actual_lat
    pad_lo = max((all_lo.max() - all_lo.min()) * 0.08, 0.1)
    pad_la = max((all_la.max() - all_la.min()) * 0.08, 0.1)
    ax.set_xlim(all_lo.min() - pad_lo, all_lo.max() + pad_lo)
    ax.set_ylim(all_la.min() - pad_la, all_la.max() + pad_la)

    return mean_e, med_e, max_e


def main():
    args   = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("HB-Mamba — Gap Prediction Visualization")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  device     : {device}")
    print(f"  split      : {args.split}")
    print(f"  n_samples  : {args.n_samples}")
    print(f"  output dir : {out_dir}")
    print("=" * 65)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, epoch, val_loss = load_model(str(ckpt_path), device)
    print(f"\n  checkpoint epoch={epoch}  val_loss={val_loss:.6f}")

    print(f"\nSelecting up to {args.n_samples} samples ...")
    samples = get_samples(args.split, args.n_samples, args.seed)
    print(f"  got {len(samples)} samples\n")

    if not samples:
        print("No samples matched the filters — nothing to plot.")
        return

    n_cols = max(1, args.grid_cols)
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.0 * n_cols, 5.0 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    print(f"{'#':<4} {'MMSI':<12} {'gap':>5}  "
          f"{'mean km':>10} {'median km':>11} {'max km':>10}")
    print("-" * 55)

    mean_all = []
    med_all  = []
    for i, item in enumerate(samples):
        r = run_inference(model, item, device)
        ax = axes[i // n_cols, i % n_cols]
        me, md, mx = draw_panel(ax, r)
        mean_all.append(me)
        med_all.append(md)
        print(f"[{i+1:<2}] {r['mmsi']:<12} {r['n_gap']:>5}  "
              f"{me:>10.2f} {md:>11.2f} {mx:>10.2f}")

    print("-" * 55)
    print(f"  panel mean km   : {np.mean(mean_all):.2f}")
    print(f"  panel median km : {np.median(med_all):.2f}")

    for i in range(len(samples), n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    handles = [
        mpatches.Patch(color="steelblue", alpha=0.5, label="Full actual trajectory"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="green", ms=8, label="Visible pings (model input)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="red",   ms=8, label="Actual gap (hidden)"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="darkorange", ms=12, label="Model prediction"),
        plt.Line2D([0], [0], color="grey", lw=1, label="Error (actual → predicted)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"HB-Mamba — Gap Prediction  |  "
        f"checkpoint={ckpt_path.name}  epoch={epoch}  val_loss={val_loss:.4f}",
        fontsize=12, fontweight="bold",
    )

    out_path = out_dir / "gap_predictions.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
