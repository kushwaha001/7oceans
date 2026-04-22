"""
LossHeads — HB-Mamba v2.0, Component 4
========================================

Four MLP heads that fire simultaneously during training.  Each head targets a
different self-supervised objective, and their outputs are combined into a
single scalar loss.

Combined loss (Table 15):
    L = 1.0·L_mask + 0.8·L_next + 0.5·L_contrast + 0.5·L_align

Head 1 — Gap Reconstruction (L_mask)
    Input  : [h_fwd_t, h_bwd_t]  [B, N_day, 2×d_model = 512]
    Output : ŷ_t                  [B, N_day, 8]
    Target : micro_tokens[:, :, :8]  at masked, non-padded positions only
    Loss   : Huber, averaged over masked pings

Head 2 — Next-Ping Prediction (L_next)
    Input  : h_fwd_t              [B, N_day, d_model = 256]   (causal only)
    Output : ŷ_{t+1}             [B, N_day-1, 8]
    Target : micro_tokens[:, 1:, :8]
    Loss   : Huber, averaged over non-padded transitions

Head 3 — Contrastive NT-Xent (L_contrast)
    Input  : H                    [B, d_model = 256]
    Proj   : z = L2_norm(MLP(H)) [B, d_proj = 128]
    Loss   : NT-Xent with τ=0.07, positives = same MMSI in batch

Head 4 — Cross-Scale Alignment (L_align)
    Builds vessel_Z from saved CSI attention weights + macro_output:
        macro_sub = macro_output[topk_idx]                [B, N_day, K, 256]
        ping_ctx  = (attn_weights ⊗ macro_sub).sum(K)    [B, N_day, 256]
        vessel_Z  = mean(ping_ctx, dim=N_day)             [B, 256]
    Then:
        Z' = L2_norm(MLP_z_proj(vessel_Z))               [B, 128]
        H' = L2_norm(MLP_h_proj(H))                      [B, 128]
        L_align = mean(1 - Z' · H')                      scalar

MLP architectures (from spec, GELU throughout — no ReLU):
    Reconstruction : Linear(512, 256) → GELU → Linear(256,  8)
    Next-ping      : Linear(256, 128) → GELU → Linear(128,  8)
    Contrastive    : Linear(256, 256) → GELU → Linear(256, 128) → L2_norm
    Align z_proj   : Linear(256, 128) → GELU → Linear(128, 128) → L2_norm
    Align h_proj   : Linear(256, 128) → GELU → Linear(128, 128) → L2_norm

Target features (first 8 of 11 micro_tokens):
    [lat, lon, sog, cog_sin, cog_cos, hdg_sin, hdg_cos, delta_t]
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_linear_interp_baseline(
    micro_tokens : Tensor,   # [B, N, D]  D>=2 (first 2 = lat, lon)
    mask         : Tensor,   # [B, N]  bool  True = masked (gap to predict)
    padding_mask : Tensor,   # [B, N]  bool  True = padded (not a real ping)
) -> Tensor:
    """
    Per-position lat/lon anchor built from the nearest visible neighbours.

    For every position t we find:
        prev = max index i <= t with visible[i]  (or -1 if none)
        next = min index j >= t with visible[j]  (or INF if none)
    where visible = ~mask & ~padding_mask.

    Baseline at position t:
        both anchors   → linear interp by index fraction
        only prev      → lat/lon of prev (flat forward extrapolation)
        only next      → lat/lon of next (flat backward extrapolation)
        neither        → 0.5 (mid of normalised range, only hit by fully-empty sequences)

    The model predicts lat/lon as `baseline + delta`, so the target delta is
    `(actual - baseline)` which is small for smooth trajectories.  This breaks
    the mean-collapse failure mode where the network was outputting a global
    average regardless of the surrounding context.

    Returns
    -------
    [B, N, 2]  fp32 baseline lat/lon.
    """
    B, N, _ = micro_tokens.shape
    device  = micro_tokens.device

    visible = (~mask) & (~padding_mask)                       # [B, N]

    lat = micro_tokens[..., 0].float()                         # [B, N]
    lon = micro_tokens[..., 1].float()

    pos  = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)   # [B, N]
    INF  = N   # sentinel "no successor"

    # Predecessor: running MAX of (pos if visible else -1) along N
    prev_src      = torch.where(visible, pos, torch.full_like(pos, -1))
    prev_idx, _   = prev_src.cummax(dim=1)                   # [B, N]

    # Successor: running MIN of (pos if visible else INF) from the right.
    # Implement as reverse-cummax of (INF - src), then flip back.
    succ_src      = torch.where(visible, pos, torch.full_like(pos, INF))
    neg           = INF - succ_src
    run_max, _    = neg.flip(dims=[1]).cummax(dim=1)
    run_max       = run_max.flip(dims=[1])
    succ_idx      = INF - run_max                             # [B, N]; INF if no successor

    has_prev = prev_idx >= 0
    has_succ = succ_idx < INF
    both     = has_prev & has_succ
    only_p   = has_prev & (~has_succ)
    only_s   = (~has_prev) & has_succ

    prev_safe = prev_idx.clamp(min=0)
    succ_safe = succ_idx.clamp(max=N - 1)

    lat_prev = torch.gather(lat, 1, prev_safe)
    lat_succ = torch.gather(lat, 1, succ_safe)
    lon_prev = torch.gather(lon, 1, prev_safe)
    lon_succ = torch.gather(lon, 1, succ_safe)

    denom = (succ_safe - prev_safe).clamp(min=1).float()
    frac  = (pos - prev_safe).float() / denom                 # 0 at prev, 1 at succ

    lat_base = torch.full_like(lat, 0.5)
    lon_base = torch.full_like(lon, 0.5)

    lat_interp = lat_prev + (lat_succ - lat_prev) * frac
    lon_interp = lon_prev + (lon_succ - lon_prev) * frac

    lat_base = torch.where(both,   lat_interp, lat_base)
    lat_base = torch.where(only_p, lat_prev,   lat_base)
    lat_base = torch.where(only_s, lat_succ,   lat_base)

    lon_base = torch.where(both,   lon_interp, lon_base)
    lon_base = torch.where(only_p, lon_prev,   lon_base)
    lon_base = torch.where(only_s, lon_succ,   lon_base)

    return torch.stack([lat_base, lon_base], dim=-1)          # [B, N, 2]


# ---------------------------------------------------------------------------
# Individual heads
# ---------------------------------------------------------------------------

class ReconstructionHead(nn.Module):
    """
    Head 1 — Gap reconstruction.

    Concatenates forward + backward hidden states, projects to n_features.
    Used for both interpolation and extrapolation gaps.

    Parameters
    ----------
    d_model    : int   Model dimension (256).
    n_features : int   Number of micro features to predict (8).
    """

    def __init__(self, d_model: int = 256, n_features: int = 8) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

    def forward(
        self,
        h_fwd           : Tensor,
        h_bwd           : Tensor,
        baseline_latlon : Optional[Tensor] = None,   # [B, N_day, 2]
    ) -> Tensor:
        """
        Parameters
        ----------
        h_fwd           : [B, N_day, d_model]
        h_bwd           : [B, N_day, d_model]
        baseline_latlon : [B, N_day, 2]  optional linear-interp baseline.
                          When provided, the first 2 output channels are
                          interpreted as deltas: pred_latlon = baseline + delta.
                          When None, the head returns absolute values (legacy).

        Returns
        -------
        [B, N_day, n_features]   — final (absolute) feature prediction.
        """
        raw = self.mlp(torch.cat([h_fwd, h_bwd], dim=-1))
        if baseline_latlon is None:
            return raw
        # delta head for lat/lon only; other features stay absolute.
        latlon = baseline_latlon.to(raw.dtype) + raw[..., :2]
        return torch.cat([latlon, raw[..., 2:]], dim=-1)


class NextPingHead(nn.Module):
    """
    Head 2 — Next-ping prediction.

    Uses h_fwd only (causal — the forward pass sees only past context).
    Prediction at position t is the target at position t+1.

    v2.1 change: a decoupling adapter (Linear) sits between h_fwd and the
    MLP so that L_next gradients do not force h_fwd to encode t+1 information
    directly (Fix 4).

    Parameters
    ----------
    d_model    : int   Model dimension (256).
    n_features : int   Number of micro features to predict (8).
    """

    def __init__(self, d_model: int = 256, n_features: int = 8) -> None:
        super().__init__()
        self.adapter = nn.Linear(d_model, d_model)   # v2.1 Fix 4: decoupling adapter
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_features),
        )

    def forward(self, h_fwd: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h_fwd : [B, N_day, d_model]

        Returns
        -------
        [B, N_day, n_features]   — shift by 1 and apply mask in loss computation
        """
        return self.mlp(self.adapter(h_fwd))   # adapter absorbs t+1 bias


class ContrastiveHead(nn.Module):
    """
    Head 3 — NT-Xent projection head.

    Maps trajectory embedding H to a unit-normed projection z used for
    contrastive learning.  Positive pairs are same-MMSI samples in the batch
    (different dates, same vessel behaviour pattern).

    Parameters
    ----------
    d_model : int   Input dimension (256).
    d_proj  : int   Output projection dimension (128).
    """

    def __init__(self, d_model: int = 256, d_proj: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_proj),
        )

    def forward(self, H: Tensor) -> Tensor:
        """
        Parameters
        ----------
        H : [B, d_model]

        Returns
        -------
        z : [B, d_proj]  L2-normalised
        """
        return F.normalize(self.mlp(H), dim=-1)


class AlignmentHead(nn.Module):
    """
    Head 4 — Cross-scale alignment.

    Projects both vessel_Z (attention-derived macro summary) and H
    (MicroEncoder trajectory embedding) to a shared 128-dim unit sphere,
    then measures cosine distance.

    Parameters
    ----------
    d_model : int   Input dimension for both vessel_Z and H (256).
    d_proj  : int   Shared projection dimension (128).
    """

    def __init__(self, d_model: int = 256, d_proj: int = 128) -> None:
        super().__init__()
        self.z_proj = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj),
        )
        self.h_proj = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj),
        )

    def forward(self, vessel_Z: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        vessel_Z : [B, d_model]   attention-weighted macro summary
        H        : [B, d_model]   MicroEncoder trajectory embedding

        Returns
        -------
        z_prime : [B, d_proj]  L2-normalised vessel_Z projection
        h_prime : [B, d_proj]  L2-normalised H projection
        """
        z_prime = F.normalize(self.z_proj(vessel_Z), dim=-1)
        h_prime = F.normalize(self.h_proj(H),        dim=-1)
        return z_prime, h_prime


# ---------------------------------------------------------------------------
# Combined loss module
# ---------------------------------------------------------------------------

class LossHeads(nn.Module):
    """
    Combined loss module — wraps all four heads and computes the weighted sum.

    Combined loss (v2.1):
        L = 1.0·L_mask + 0.3·L_next + 0.5·L_contrast + 0.5·L_align

    Parameters
    ----------
    d_model    : int    Model dimension (must match MacroEncoder / MicroEncoder).
    d_proj     : int    Contrastive / alignment projection dimension (128).
    n_features : int    Number of target micro features (8 = lat,lon,sog,...,delta_t).
    tau        : float  NT-Xent temperature (0.07).
    w_mask     : float  Weight for L_mask  (1.0).
    w_next     : float  Weight for L_next  (0.3, was 0.8 in v2.0).
    w_contrast : float  Weight for L_contrast (0.5).
    w_align    : float  Weight for L_align (0.5).
    """

    def __init__(
        self,
        d_model    : int   = 256,
        d_proj     : int   = 128,
        n_features : int   = 8,
        tau        : float = 0.07,
        w_mask     : float = 1.0,
        w_next     : float = 0.3,    # v2.1: reduced from 0.8 to reduce conflict with L_mask
        w_contrast : float = 0.5,
        w_align    : float = 0.5,
    ) -> None:
        super().__init__()

        self.d_model    = d_model
        self.d_proj     = d_proj
        self.n_features = n_features
        self.tau        = tau
        self.w_mask     = w_mask
        self.w_next     = w_next
        self.w_contrast = w_contrast
        self.w_align    = w_align

        self.recon_head    = ReconstructionHead(d_model=d_model, n_features=n_features)
        self.next_head     = NextPingHead(d_model=d_model,       n_features=n_features)
        self.contrast_head = ContrastiveHead(d_model=d_model,    d_proj=d_proj)
        self.align_head    = AlignmentHead(d_model=d_model,      d_proj=d_proj)

        # v2.2: feature-weighted Huber. Features order:
        #   [lat, lon, sog, cog_sin, cog_cos, hdg_sin, hdg_cos, delta_t]
        # lat/lon weighted 5x (what gap-filling actually cares about),
        # sog 1x, cog/hdg/delta_t 0.3x (high aleatoric noise — they cap the loss floor
        # and waste capacity when weighted equally).
        feat_w = torch.tensor([5.0, 5.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3])
        self.register_buffer("feature_weights", feat_w)

    def _weighted_huber(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Per-element Huber, feature-weighted, then mean over (positions × features).

        pred, target : [..., n_features]
        """
        per_elem = F.huber_loss(pred, target, reduction="none")     # [..., n_features]
        weighted = per_elem * self.feature_weights                  # broadcast on last dim
        return weighted.mean()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_vessel_Z(
        self,
        attn_weights : Tensor,   # [B, N_day, K]
        topk_idx     : Tensor,   # [B, N_day, K]  int64
        macro_output : Tensor,   # [B, N_cells, d_model]
        padding_mask : Tensor,   # [B, N_day]  True = padded
    ) -> Tensor:
        """
        Construct vessel_Z — the attention-weighted macro summary per vessel.

        vessel_Z[b] is the mean across real pings of the per-ping macro context
        vector (the weighted sum of the K attended macro cells for that ping).

        Returns [B, d_model].
        """
        B, N_day, K = attn_weights.shape

        # Gather the K selected macro cells for every ping
        flat_idx      = topk_idx.reshape(B, N_day * K)                         # [B, N_day*K]
        flat_expanded = flat_idx.unsqueeze(-1).expand(-1, -1, self.d_model)    # [B, N_day*K, d_model]
        macro_sub     = macro_output.gather(1, flat_expanded).reshape(
                            B, N_day, K, self.d_model)                         # [B, N_day, K, d_model]

        # Per-ping context: weighted sum over K attended cells
        ping_ctx = (attn_weights.unsqueeze(-1) * macro_sub).sum(dim=2)        # [B, N_day, d_model]

        # Mean pool over REAL pings only (exclude padded positions)
        real_mask = (~padding_mask).float().unsqueeze(-1)                      # [B, N_day, 1]
        n_real    = real_mask.sum(dim=1).clamp(min=1)                         # [B, 1]
        vessel_Z  = (ping_ctx * real_mask).sum(dim=1) / n_real                # [B, d_model]

        return vessel_Z

    def _nt_xent(self, z: Tensor, mmsi: List[int]) -> Tensor:
        """
        NT-Xent loss (SimCLR-style).

        For each sample i with at least one positive j (same MMSI, different
        sample index) in the batch:
            L_i = log(sum_{k!=i} exp(sim/tau)) - log(sum_{j in pos(i)} exp(sim/tau))

        If no positive pairs exist in the batch (all MMSIs unique), returns
        zero so training continues — contrastive signal requires deliberate
        pair sampling in the DataLoader during real training runs.

        Parameters
        ----------
        z    : [B, d_proj]  L2-normalised projections
        mmsi : list[int]    MMSI for each sample in batch

        Returns
        -------
        scalar loss tensor
        """
        B      = z.shape[0]
        device = z.device

        # Full similarity matrix
        sim = torch.mm(z, z.T) / self.tau   # [B, B]

        # Positive mask: same MMSI, different sample index
        mmsi_t   = torch.tensor(mmsi, dtype=torch.long, device=device)
        pos_mask = (mmsi_t.unsqueeze(0) == mmsi_t.unsqueeze(1))   # [B, B]
        pos_mask.fill_diagonal_(False)

        if not pos_mask.any():
            return torch.zeros(1, device=device, requires_grad=False).squeeze()

        self_mask = torch.eye(B, dtype=torch.bool, device=device)

        losses = []
        for i in range(B):
            if not pos_mask[i].any():
                continue
            log_denom = torch.logsumexp(sim[i][~self_mask[i]], dim=0)
            log_numer = torch.logsumexp(sim[i][pos_mask[i]],  dim=0)
            losses.append(log_denom - log_numer)

        return torch.stack(losses).mean()

    # ── Main forward ────────────────────────────────────────────────────────

    def forward(
        self,
        h_fwd        : Tensor,       # [B, N_day, d_model]   MicroEncoder fwd states
        h_bwd        : Tensor,       # [B, N_day, d_model]   MicroEncoder bwd states
        H            : Tensor,       # [B, d_model]           trajectory embedding
        macro_output : Tensor,       # [B, N_cells, d_model]  MacroEncoder output
        attn_weights : Tensor,       # [B, N_day, K]          saved from CSI
        topk_idx     : Tensor,       # [B, N_day, K]  int64   saved from CSI
        micro_tokens : Tensor,       # [B, N_day, 11]          original (unmasked) features
        mask         : Tensor,       # [B, N_day]  bool        True = masked ping
        padding_mask : Tensor,       # [B, N_day]  bool        True = padded position
        mmsi         : List[int],    # [B]                     MMSI per sample
    ) -> Dict[str, Tensor]:
        """
        Compute all four losses and the combined weighted sum.

        Returns
        -------
        dict with keys:
            L_mask      scalar — gap reconstruction loss
            L_next      scalar — next-ping prediction loss
            L_contrast  scalar — NT-Xent contrastive loss
            L_align     scalar — cross-scale alignment loss
            L_total     scalar — weighted combination
        """
        target = micro_tokens[:, :, :self.n_features]   # [B, N_day, 8]

        # v2.3 delta head: build a per-position linear-interp anchor from the
        # visible (non-masked, non-padded) lat/lon. The recon head then only
        # has to predict the residual (actual - baseline), which kills the
        # mean-collapse failure mode — the network can't get away with
        # outputting a global average because the baseline already contains
        # the local context.
        baseline_latlon = compute_linear_interp_baseline(
            micro_tokens, mask, padding_mask
        )   # [B, N_day, 2]

        # ── Head 1 — Gap reconstruction ───────────────────────────────────
        y_hat_mask = self.recon_head(h_fwd, h_bwd, baseline_latlon)   # [B, N_day, 8]
        active     = mask & ~padding_mask                # [B, N_day]
        if active.any():
            L_mask = self._weighted_huber(y_hat_mask[active], target[active])
        else:
            L_mask = torch.zeros(1, device=h_fwd.device).squeeze()

        # ── Head 2 — Next-ping prediction ─────────────────────────────────
        y_hat_next  = self.next_head(h_fwd[:, :-1, :])  # [B, N_day-1, 8]
        target_next = target[:, 1:, :]                   # [B, N_day-1, 8]
        # Valid only where both t and t+1 are real pings
        next_active = ~padding_mask[:, :-1] & ~padding_mask[:, 1:]
        if next_active.any():
            L_next = self._weighted_huber(y_hat_next[next_active], target_next[next_active])
        else:
            L_next = torch.zeros(1, device=h_fwd.device).squeeze()

        # ── Head 3 — Contrastive ──────────────────────────────────────────
        z          = self.contrast_head(H)               # [B, 128]
        L_contrast = self._nt_xent(z, mmsi)

        # ── Head 4 — Cross-scale alignment (cross-modal InfoNCE) ─────────
        # v2.2: replaced pure cosine-distance loss (which trivially mode-collapsed
        # to a shared constant direction) with a symmetric InfoNCE between vessel_Z
        # and H. Positive: (z_i, h_i) — same vessel. Negatives: all off-diagonal
        # pairs within the batch. Can't collapse because collapse → every off-diag
        # logit matches the diagonal → cross-entropy blows up.
        vessel_Z         = self._build_vessel_Z(
                               attn_weights, topk_idx, macro_output, padding_mask)
        z_prime, h_prime = self.align_head(vessel_Z, H)  # [B, d_proj] each, L2-normed
        B_a              = z_prime.shape[0]
        if B_a > 1:
            logits_zh = torch.mm(h_prime, z_prime.T) / self.tau   # [B, B]
            labels    = torch.arange(B_a, device=logits_zh.device)
            L_align   = (F.cross_entropy(logits_zh,      labels)
                       + F.cross_entropy(logits_zh.T,    labels)) * 0.5
        else:
            # Single-sample batch — no negatives. Fall back to cosine distance.
            L_align = (1.0 - (z_prime * h_prime).sum(dim=-1)).mean()

        # ── Combined loss ─────────────────────────────────────────────────
        L_total = (self.w_mask     * L_mask
                 + self.w_next     * L_next
                 + self.w_contrast * L_contrast
                 + self.w_align    * L_align)

        return {
            "L_mask"     : L_mask,
            "L_next"     : L_next,
            "L_contrast" : L_contrast,
            "L_align"    : L_align,
            "L_total"    : L_total,
        }


# =============================================================================
# Self-contained test block
# =============================================================================

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import json
    from hb_mamba_dataset import build_dataloaders
    from model.macro_encoder         import MacroEncoder
    from model.cross_scale_injection import CrossScaleInjection
    from model.micro_decoder         import MicroEncoder

    if not torch.cuda.is_available():
        print("No CUDA device — test requires GPU (Mamba2 is CUDA-only).")
        sys.exit(1)

    device = torch.device("cuda")

    _raw     = _root / "data_genration_and_raw_data" / "raw_data"
    _ns_path = _raw / "preprocessing" / "norm_stats" / "norm_stats.json"
    with _ns_path.open() as _f:
        _ns = json.load(_f)
    N_LAT  = int(_ns["N_LAT_STEPS"])
    N_LON  = int(_ns["N_LON_STEPS"])
    N_CELL = int(_ns["N_TOTAL_CELLS"])

    print("=" * 65)
    print(f"LossHeads — HB-Mamba v2.0  self-test  (device: {device})")
    print("=" * 65)

    all_passed = True

    def _check(name: str, cond: bool) -> None:
        global all_passed
        s = "PASS" if cond else "FAIL"
        if not cond:
            all_passed = False
        print(f"  [{s}]  {name}")

    # ── Build full pipeline ────────────────────────────────────────────────
    loaders = build_dataloaders(
        dataset_index_dir = str(_raw / "dataset_index"),
        norm_stats_path   = str(_ns_path),
        batch_size        = 8,
        num_workers       = 0,
        pin_memory        = False,
    )
    batch = next(iter(loaders["train"]))

    macro_features = batch["macro_features"].to(device)
    macro_lat_idx  = batch["macro_lat_idx"].to(device)
    macro_lon_idx  = batch["macro_lon_idx"].to(device)
    micro_tokens   = batch["micro_tokens"].to(device)
    mask           = batch["mask"].to(device)
    padding_mask   = batch["padding_mask"].to(device)
    mmsi_list      = batch["mmsi"]

    B = macro_features.shape[0]

    macro_enc = MacroEncoder(n_lat_steps=N_LAT, n_lon_steps=N_LON).to(device).eval()
    csi       = CrossScaleInjection(d_micro=11, d_model=256, K=32).to(device).eval()
    micro_enc = MicroEncoder(d_model=256, n_layers=4).to(device).eval()
    heads     = LossHeads(d_model=256, d_proj=128).to(device)

    with torch.no_grad():
        macro_out, _          = macro_enc(macro_features, macro_lat_idx, macro_lon_idx)
        x_cond, attn_w, tidx  = csi(micro_tokens, macro_out)
        h_fwd, h_bwd, H       = micro_enc(x_cond)

    # ── Check 1 — Individual head output shapes ────────────────────────────
    print("\nCheck 1 — Individual head output shapes")

    max_n = micro_tokens.shape[1]
    with torch.no_grad():
        y_recon = heads.recon_head(h_fwd, h_bwd)
        y_next  = heads.next_head(h_fwd)
        z_cont  = heads.contrast_head(H)
        vZ      = heads._build_vessel_Z(attn_w, tidx, macro_out, padding_mask)
        zp, hp  = heads.align_head(vZ, H)

    _check(f"recon output shape == ({B}, {max_n}, 8)",  y_recon.shape == (B, max_n, 8))
    _check(f"next  output shape == ({B}, {max_n}, 8)",  y_next.shape  == (B, max_n, 8))
    _check(f"z_contrast shape   == ({B}, 128)",          z_cont.shape  == (B, 128))
    _check(f"vessel_Z shape     == ({B}, 256)",           vZ.shape      == (B, 256))
    _check(f"z_prime shape      == ({B}, 128)",           zp.shape      == (B, 128))
    _check(f"h_prime shape      == ({B}, 128)",           hp.shape      == (B, 128))

    # ── Check 2 — Projections are L2-normed ───────────────────────────────
    print("\nCheck 2 — Contrast and alignment projections are L2-normed")

    _check("z_contrast norms == 1.0",
           torch.allclose(z_cont.norm(dim=-1),
                          torch.ones(B, device=device), atol=1e-5))
    _check("z_prime norms == 1.0",
           torch.allclose(zp.norm(dim=-1),
                          torch.ones(B, device=device), atol=1e-5))
    _check("h_prime norms == 1.0",
           torch.allclose(hp.norm(dim=-1),
                          torch.ones(B, device=device), atol=1e-5))

    # ── Check 3 — No NaNs in any head output ──────────────────────────────
    print("\nCheck 3 — No NaNs in any head output")

    _check("recon output: no NaN", not torch.isnan(y_recon).any())
    _check("next  output: no NaN", not torch.isnan(y_next).any())
    _check("z_contrast:   no NaN", not torch.isnan(z_cont).any())
    _check("vessel_Z:     no NaN", not torch.isnan(vZ).any())

    # ── Check 4 — Forward pass: all four losses ───────────────────────────
    print("\nCheck 4 — All four losses (forward only)")

    heads.eval()
    with torch.no_grad():
        losses = heads(
            h_fwd, h_bwd, H,
            macro_out, attn_w, tidx,
            micro_tokens, mask, padding_mask, mmsi_list,
        )

    for name, val in losses.items():
        _check(f"{name} is finite  (={val.item():.4f})", torch.isfinite(val).item())

    _check("L_mask  >= 0", losses["L_mask"].item()  >= 0)
    _check("L_next  >= 0", losses["L_next"].item()  >= 0)
    _check("L_align >= 0", losses["L_align"].item() >= 0)

    # ── Check 5 — NT-Xent with injected positive pairs ────────────────────
    print("\nCheck 5 — NT-Xent loss with forced positive pairs")

    B_half        = B // 2
    fake_mmsi     = mmsi_list[:B_half] + mmsi_list[:B_half]
    z_dup         = z_cont[:B_half].repeat(2, 1)

    L_ct_forced = heads._nt_xent(z_dup, fake_mmsi)
    _check("NT-Xent with positives: finite",      torch.isfinite(L_ct_forced).item())
    _check("NT-Xent with positives: non-negative", L_ct_forced.item() >= 0)
    print(f"  NT-Xent (forced positives) = {L_ct_forced.item():.4f}")

    L_ct_nopos = heads._nt_xent(z_cont, mmsi_list)
    _check("NT-Xent with no positives: returns 0.0",
           abs(L_ct_nopos.item()) < 1e-9)
    print(f"  NT-Xent (no positives)     = {L_ct_nopos.item():.4f}")

    # ── Check 6 — Backward through all four losses ────────────────────────
    print("\nCheck 6 — Backward through all four losses")

    macro_enc_t = MacroEncoder(n_lat_steps=N_LAT, n_lon_steps=N_LON).to(device).train()
    csi_t       = CrossScaleInjection(d_micro=11, d_model=256, K=32).to(device).train()
    micro_enc_t = MicroEncoder(d_model=256, n_layers=4).to(device).train()
    heads_t     = LossHeads(d_model=256, d_proj=128).to(device).train()

    B2  = 4
    mf2 = batch["macro_features"][:B2].to(device)
    la2 = batch["macro_lat_idx"][:B2].to(device)
    lo2 = batch["macro_lon_idx"][:B2].to(device)
    mt2 = batch["micro_tokens"][:B2].to(device)
    mk2 = batch["mask"][:B2].to(device)
    pm2 = batch["padding_mask"][:B2].to(device)
    ms2 = batch["mmsi"][:B2 // 2] + batch["mmsi"][:B2 // 2]   # force positive pairs

    mo2, _         = macro_enc_t(mf2, la2, lo2)
    xc2, aw2, ti2  = csi_t(mt2, mo2)
    hf2, hb2, H2   = micro_enc_t(xc2)
    losses2        = heads_t(hf2, hb2, H2, mo2, aw2, ti2, mt2, mk2, pm2, ms2)
    losses2["L_total"].backward()

    _check("LossHeads grads exist",
           all(p.grad is not None for p in heads_t.parameters()))
    _check("MicroEncoder grads exist",
           all(p.grad is not None for p in micro_enc_t.parameters()))
    _check("CSI grads exist",
           all(p.grad is not None for p in csi_t.parameters()))
    _check("MacroEncoder grads exist",
           all(p.grad is not None
               for p in macro_enc_t.parameters() if p.requires_grad))
    _check("W_Q non-zero (selection path alive)",
           csi_t.W_Q.weight.grad is not None and
           csi_t.W_Q.weight.grad.abs().max().item() > 1e-10)

    # ── Check 7 — Parameter count ──────────────────────────────────────────
    print("\nCheck 7 — Parameter count")

    def _c(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

    p_recon = _c(heads.recon_head)
    p_next  = _c(heads.next_head)
    p_cont  = _c(heads.contrast_head)
    p_align = _c(heads.align_head)
    p_total = _c(heads)

    print(f"  recon_head     : {p_recon:>10,}  (512->256->8)")
    print(f"  next_head      : {p_next:>10,}  (256->128->8)")
    print(f"  contrast_head  : {p_cont:>10,}  (256->256->128)")
    print(f"  align_head     : {p_align:>10,}  (z:256->128->128, h:256->128->128)")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  TOTAL          : {p_total:>10,}")

    _check("recon_head    > 0 params", p_recon > 0)
    _check("next_head     > 0 params", p_next  > 0)
    _check("contrast_head > 0 params", p_cont  > 0)
    _check("align_head    > 0 params", p_align > 0)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_passed:
        print("All checks passed — LossHeads v2.0 OK")
    else:
        print("One or more checks FAILED — see [FAIL] lines above")
    print("=" * 65)
