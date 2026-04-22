"""
visualize_embeddings.py — HB-Mamba v2.0  Embedding Visualizer
==============================================================

Loads embeddings/embeddings.npz produced by inference.py and generates
a suite of plots that explain what the model has learned.

Usage:
    python visualize_embeddings.py                        # all plots, save to figs/
    python visualize_embeddings.py --show                 # display interactively
    python visualize_embeddings.py --input my_emb.npz    # custom file
    python visualize_embeddings.py --top_vessels 20      # show top-N vessels

What each plot shows — quick reference
---------------------------------------
1. pca_2d.png          2D PCA scatter.  Each dot = one vessel-day.
                       Color = unique MMSI (vessel identity).
                       If the model learned vessel-specific features,
                       dots from the same vessel will cluster together.

2. pca_variance.png    Scree plot: how much variance each PC captures.
                       A fast drop means the embedding space is low-rank
                       (the model found a compact representation).
                       Slow drop = information spread across many dims.

3. norm_dist.png       Histogram of L2 norms of each embedding H.
                       A tight, unimodal distribution = the model is
                       stable and not collapsing/exploding.
                       Outlier norms = unusual trajectories or OOD data.

4. cosine_sim.png      Mean pairwise cosine similarity between vessel-days
                       of the SAME vessel (intra) vs DIFFERENT vessels
                       (inter).  Good embedding: intra >> inter.

5. vessel_paths.png    Top-N vessels by day count plotted in PC1-PC2 space.
                       Each line connects a vessel's days chronologically.
                       Smooth curves = the model captures temporal drift.
                       Scattered = day-to-day variance is high.

6. temporal_drift.png  Mean PC1 and PC2 across all vessels per calendar date.
                       Reveals whether the embedding space shifts seasonally
                       or with fleet-level events.

7. top_vessels_bar.png Which vessels have the most embedded days.
                       Useful for knowing which MMSIs are well-represented.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pca(X: np.ndarray, n_components: int = 50):
    """
    PCA via SVD (no sklearn needed).

    Parameters
    ----------
    X            : [N, D]  float32 array
    n_components : number of components to keep

    Returns
    -------
    scores       : [N, n_components]  projected data
    explained    : [n_components]     fraction of variance explained per PC
    components   : [n_components, D] principal axes
    """
    X_c = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    total_var  = (S ** 2).sum()
    explained  = (S[:n_components] ** 2) / total_var
    scores     = U[:, :n_components] * S[:n_components]
    components = Vt[:n_components]
    return scores.astype(np.float32), explained, components


def cosine_similarity_matrix(A: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix [N, N]."""
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    A_n   = A / norms
    return (A_n @ A_n.T).clip(-1, 1)


def _color_cycle(n: int):
    """Return n visually distinct colors."""
    cmap = cm.get_cmap("tab20" if n <= 20 else "turbo")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_pca_2d(scores, mmsi, out_path, top_n=40, show=False):
    """
    2D PCA scatter coloured by vessel (MMSI).

    Only the top_n most frequent vessels are coloured individually;
    the rest are shown in grey so the plot stays readable.
    """
    unique, counts = np.unique(mmsi, return_counts=True)
    order     = np.argsort(-counts)
    top_mmsi  = set(unique[order[:top_n]])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(
        "PCA of Trajectory Embeddings (PC1 vs PC2)\n"
        "Each point = one vessel-day  |  colour = vessel MMSI",
        fontsize=12
    )

    # Plot non-top vessels first (grey background)
    grey_mask = np.array([m not in top_mmsi for m in mmsi])
    if grey_mask.any():
        ax.scatter(
            scores[grey_mask, 0], scores[grey_mask, 1],
            c="lightgrey", s=6, alpha=0.4, linewidths=0, label="other vessels"
        )

    # Plot top vessels with distinct colours
    colours = _color_cycle(top_n)
    for i, m in enumerate(unique[order[:top_n]]):
        mask = (mmsi == m)
        ax.scatter(
            scores[mask, 0], scores[mask, 1],
            c=[colours[i]], s=18, alpha=0.75, linewidths=0,
            label=f"MMSI {m}"
        )

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_pca_variance(explained, out_path, show=False):
    """
    Scree plot: fraction of variance explained per PC and cumulative.
    """
    n   = len(explained)
    cum = np.cumsum(explained) * 100
    x   = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.bar(x, explained * 100, color="steelblue", alpha=0.7, label="per-PC variance")
    ax2.plot(x, cum, color="darkorange", linewidth=2, marker="o",
             markersize=3, label="cumulative %")

    # Mark 80% and 95% thresholds
    for thr, ls in [(80, "--"), (95, ":")]:
        idx = np.searchsorted(cum, thr)
        ax2.axhline(thr, color="grey", linestyle=ls, alpha=0.6,
                    label=f"{thr}% @ PC{idx+1}")

    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Variance explained (%)", fontsize=11, color="steelblue")
    ax2.set_ylabel("Cumulative variance (%)", fontsize=11, color="darkorange")
    ax1.set_title(
        "PCA Scree Plot\n"
        "Fast drop = compact representation  |  slow = information spread across dims",
        fontsize=11
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_norm_distribution(H, out_path, show=False):
    """
    Histogram of L2 norms.  Tight unimodal = stable model.
    """
    norms = np.linalg.norm(H, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norms, bins=60, color="steelblue", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.axvline(norms.mean(),   color="darkorange", linewidth=2,
               label=f"mean = {norms.mean():.2f}")
    ax.axvline(np.median(norms), color="green", linewidth=2, linestyle="--",
               label=f"median = {np.median(norms):.2f}")

    ax.set_xlabel("L2 Norm of H  (embedding magnitude)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Distribution of Embedding L2 Norms\n"
        "Tight unimodal = model stable  |  long tail = unusual trajectories",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Stats box
    stats = (f"N={len(norms):,}\n"
             f"mean={norms.mean():.2f}\n"
             f"std={norms.std():.2f}\n"
             f"min={norms.min():.2f}\n"
             f"max={norms.max():.2f}")
    ax.text(0.97, 0.97, stats, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_cosine_similarity(H, mmsi, out_path, sample_n=2000, show=False):
    """
    Intra-vessel vs inter-vessel cosine similarity comparison.

    Good embedding: same vessel → high cosine sim, different → low.
    Shows as a dual histogram (overlapping distributions).
    """
    rng = np.random.default_rng(42)

    # Subsample to keep it fast
    if len(H) > sample_n:
        idx = rng.choice(len(H), sample_n, replace=False)
        H_s, mmsi_s = H[idx], mmsi[idx]
    else:
        H_s, mmsi_s = H, mmsi

    sim = cosine_similarity_matrix(H_s.astype(np.float32))

    # Gather intra and inter pairs (upper triangle only, excluding diagonal)
    i_idx, j_idx = np.triu_indices(len(H_s), k=1)
    same    = mmsi_s[i_idx] == mmsi_s[j_idx]
    intra   = sim[i_idx[same],  j_idx[same]]
    inter   = sim[i_idx[~same], j_idx[~same]]

    # Sub-sample inter (usually far more pairs)
    if len(inter) > 50_000:
        inter = rng.choice(inter, 50_000, replace=False)

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(-0.2, 1.0, 80)
    ax.hist(inter, bins=bins, color="steelblue",  alpha=0.6,
            label=f"inter-vessel  (μ={inter.mean():.3f})", density=True)
    ax.hist(intra, bins=bins, color="darkorange", alpha=0.75,
            label=f"intra-vessel  (μ={intra.mean():.3f})", density=True)

    ax.axvline(inter.mean(), color="steelblue",  linewidth=1.5, linestyle="--")
    ax.axvline(intra.mean(), color="darkorange", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Intra-vessel vs Inter-vessel Cosine Similarity\n"
        "Good embedding: orange peak >> blue peak (same vessel = more similar)",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Separation score
    sep = intra.mean() - inter.mean()
    ax.text(0.03, 0.97,
            f"Separation Δ = {sep:.3f}\n"
            f"intra pairs : {len(intra):,}\n"
            f"inter pairs : {len(inter):,}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_vessel_paths(scores, mmsi, date, out_path, top_n=12, show=False):
    """
    Top-N vessels by day count, each plotted as a trajectory in PC1-PC2 space
    with days connected chronologically.

    Smooth path = model captures temporal drift within a vessel's behaviour.
    Scattered    = high day-to-day variance or no temporal structure.
    """
    unique, counts = np.unique(mmsi, return_counts=True)
    order   = np.argsort(-counts)
    top_mmsi = unique[order[:top_n]]

    colours = _color_cycle(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, m in enumerate(top_mmsi):
        mask = (mmsi == m)
        pts  = scores[mask]
        ds   = date[mask]

        # Sort by date
        sort_idx = np.argsort(ds)
        pts = pts[sort_idx]
        ds  = ds[sort_idx]

        c = colours[i]
        ax.plot(pts[:, 0], pts[:, 1], color=c, linewidth=1.2, alpha=0.7)
        ax.scatter(pts[:, 0], pts[:, 1], color=c, s=25, zorder=3,
                   label=f"MMSI {m}  ({counts[order[i]]} days)")
        # Mark first and last day
        ax.scatter(pts[0,  0], pts[0,  1], color=c, s=80,
                   marker="^", zorder=4, edgecolors="black", linewidths=0.5)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=c, s=80,
                   marker="s", zorder=4, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title(
        f"Top-{top_n} Vessels — Trajectory Through Embedding Space\n"
        "▲ = first day  ■ = last day  |  lines connect consecutive days",
        fontsize=11
    )
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_temporal_drift(scores, date, out_path, show=False):
    """
    Mean PC1 and PC2 per calendar date across the whole fleet.

    Reveals fleet-level seasonal or event-driven shifts in embedding space.
    """
    df = pd.DataFrame({
        "date" : date,
        "pc1"  : scores[:, 0],
        "pc2"  : scores[:, 1],
    })
    daily = df.groupby("date")[["pc1", "pc2"]].mean().sort_index()

    if len(daily) < 2:
        print("  [skip] temporal_drift — fewer than 2 unique dates")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax1.plot(daily.index, daily["pc1"], color="steelblue",  linewidth=1.8)
    ax1.fill_between(daily.index, daily["pc1"],
                     alpha=0.15, color="steelblue")
    ax1.set_ylabel("Mean PC 1", fontsize=10)
    ax1.set_title(
        "Fleet-level Temporal Drift in Embedding Space\n"
        "Mean PC1 / PC2 across all vessels per calendar date",
        fontsize=11
    )
    ax1.grid(True, alpha=0.3)

    ax2.plot(daily.index, daily["pc2"], color="darkorange", linewidth=1.8)
    ax2.fill_between(daily.index, daily["pc2"],
                     alpha=0.15, color="darkorange")
    ax2.set_ylabel("Mean PC 2", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Rotate x-axis labels
    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_top_vessels_bar(mmsi, out_path, top_n=30, show=False):
    """
    Horizontal bar chart: vessels with the most embedded days.
    """
    unique, counts = np.unique(mmsi, return_counts=True)
    order   = np.argsort(-counts)
    top_mmsi   = unique[order[:top_n]]
    top_counts = counts[order[:top_n]]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.32)))
    colours = _color_cycle(top_n)
    y = np.arange(top_n)

    bars = ax.barh(y, top_counts, color=colours, edgecolor="white",
                   linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([str(m) for m in top_mmsi], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of embedded days", fontsize=11)
    ax.set_title(
        f"Top-{top_n} Vessels by Day Count\n"
        "Vessels with more days are better represented in the embedding space",
        fontsize=11
    )

    for bar, cnt in zip(bars, top_counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=7)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Visualize HB-Mamba v2.0 embeddings")
    p.add_argument("--input",       default="embeddings/embeddings.npz",
                   help="Path to embeddings.npz  [embeddings/embeddings.npz]")
    p.add_argument("--output",      default="figs",
                   help="Output directory for figures  [figs]")
    p.add_argument("--show",        action="store_true",
                   help="Display plots interactively (requires display)")
    p.add_argument("--top_vessels", type=int, default=15,
                   help="Number of top vessels to highlight  [15]")
    p.add_argument("--pca_dims",    type=int, default=50,
                   help="PCA components to compute  [50]")
    args = p.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    emb_path = Path(args.input)
    if not emb_path.is_absolute():
        emb_path = ROOT / emb_path
    if not emb_path.exists():
        print(f"ERROR: embeddings file not found: {emb_path}")
        sys.exit(1)

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {emb_path} ...")
    data = np.load(emb_path, allow_pickle=True)

    H        = data["H"].astype(np.float32)    # [N, 256]
    mmsi     = data["mmsi"]                     # [N]  int64
    date     = data["date"]                     # [N]  object (str)
    n_day    = data["n_day"]                    # [N]  int64
    ckpt_epoch    = int(data["epoch"])
    ckpt_val_loss = float(data["val_loss"])

    N, D = H.shape
    n_vessels = len(np.unique(mmsi))

    print(f"\n{'='*60}")
    print(f"Embeddings summary")
    print(f"{'='*60}")
    print(f"  samples       : {N:,}")
    print(f"  unique vessels: {n_vessels:,}")
    print(f"  embedding dim : {D}")
    print(f"  date range    : {date.min()} → {date.max()}")
    print(f"  n_day range   : {n_day.min()} – {n_day.max()}  "
          f"(mean {n_day.mean():.1f})")
    print(f"  checkpoint    : epoch {ckpt_epoch}, val_loss={ckpt_val_loss:.6f}")
    print(f"  H L2 norm     : mean={np.linalg.norm(H,axis=1).mean():.3f}  "
          f"std={np.linalg.norm(H,axis=1).std():.3f}")
    print(f"{'='*60}\n")

    # ── PCA ───────────────────────────────────────────────────────────────────
    n_pca = min(args.pca_dims, N, D)
    print(f"Computing PCA ({n_pca} components) ...")
    scores, explained, _ = pca(H, n_components=n_pca)

    cum80  = int(np.searchsorted(np.cumsum(explained), 0.80)) + 1
    cum95  = int(np.searchsorted(np.cumsum(explained), 0.95)) + 1
    print(f"  PC1 explains  : {explained[0]*100:.1f}%")
    print(f"  top-2 explain : {explained[:2].sum()*100:.1f}%")
    print(f"  80% variance  : {cum80} PCs")
    print(f"  95% variance  : {cum95} PCs\n")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("Generating plots ...")

    plot_pca_2d(
        scores, mmsi,
        out_dir / "pca_2d.png",
        top_n = args.top_vessels,
        show  = args.show,
    )
    plot_pca_variance(
        explained,
        out_dir / "pca_variance.png",
        show = args.show,
    )
    plot_norm_distribution(
        H,
        out_dir / "norm_dist.png",
        show = args.show,
    )
    plot_cosine_similarity(
        H, mmsi,
        out_dir / "cosine_sim.png",
        show = args.show,
    )
    plot_vessel_paths(
        scores, mmsi, date,
        out_dir / "vessel_paths.png",
        top_n = args.top_vessels,
        show  = args.show,
    )
    plot_temporal_drift(
        scores, date,
        out_dir / "temporal_drift.png",
        show = args.show,
    )
    plot_top_vessels_bar(
        mmsi,
        out_dir / "top_vessels_bar.png",
        top_n = 30,
        show  = args.show,
    )

    print(f"\nAll figures saved to: {out_dir}/")
    print("\nHow to read the plots")
    print("-" * 60)
    print("pca_2d.png        Dots in 2D: clustered by colour = model")
    print("                  learned vessel-specific patterns.")
    print("pca_variance.png  How many PCs capture 80%/95% variance.")
    print("                  Fewer PCs needed = more compact representation.")
    print("norm_dist.png     Tight bell = stable model. Right tail = unusual")
    print("                  trajectories worth investigating.")
    print("cosine_sim.png    Orange peak > blue peak = model separates")
    print("                  vessels. The gap (Δ) is your key metric.")
    print("vessel_paths.png  ▲=first day, ■=last day. Smooth path through")
    print("                  embedding space = temporal continuity learned.")
    print("temporal_drift.png Fleet-level embedding shift over time.")
    print("                  Trends here = seasonal/event patterns captured.")
    print("top_vessels_bar.png How many days per vessel in this split.")


if __name__ == "__main__":
    main()
