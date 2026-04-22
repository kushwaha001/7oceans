"""
HBMamba — HB-Mamba v2.0  Top-Level Model
==========================================

Wires all four components into a single nn.Module with one forward() call.

Data flow:
    batch (from DataLoader)
        macro_features  [B, N_cells, 10]
        macro_lat_idx   [B, N_cells]
        macro_lon_idx   [B, N_cells]
        micro_tokens    [B, N_day, 11]   zero-padded
        mask            [B, N_day]       True = masked ping to predict
        padding_mask    [B, N_day]       True = padded position
        mmsi            list[int]

    ── MacroEncoder ─────────────────────────────────────────────────────────
        macro_output  [B, N_cells, 256]
        Z_global      [B, 256]           (fallback, not used in loss)

    ── CrossScaleInjection ──────────────────────────────────────────────────
        x_conditioned [B, N_day, 256]
        attn_weights  [B, N_day, 32]     SAVED for L_align
        topk_idx      [B, N_day, 32]     SAVED for L_align

    ── MicroEncoder ─────────────────────────────────────────────────────────
        h_fwd  [B, N_day, 256]
        h_bwd  [B, N_day, 256]
        H      [B, 256]

    ── LossHeads ────────────────────────────────────────────────────────────
        L_mask, L_next, L_contrast, L_align, L_total

Training mode  (forward):  returns loss dict
Inference mode (predict):  returns hidden states + embeddings, no loss
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Relative imports work both as a module and when run directly as __main__
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model.macro_encoder         import MacroEncoder
from model.cross_scale_injection import CrossScaleInjection
from model.micro_decoder         import MicroEncoder
from model.loss_heads            import LossHeads


class HBMamba(nn.Module):
    """
    HB-Mamba v2.0 — Hierarchical Bidirectional Mamba for AIS trajectory learning.

    Parameters
    ----------
    n_lat_steps : int   Number of latitude grid steps  (read from norm_stats).
    n_lon_steps : int   Number of longitude grid steps (read from norm_stats).
    d_macro     : int   Macro feature dimension (10).
    d_micro     : int   Micro feature dimension (11).
    d_model     : int   Internal model dimension (256).
    n_layers    : int   Bidirectional Mamba2 blocks in each encoder (4).
    d_state     : int   Mamba2 SSM state dimension (64).
    K           : int   Top-K cells per ping in CrossScaleInjection (32).
    d_proj      : int   Projection head dimension for contrastive / alignment (128).
    n_features  : int   Target micro features predicted by loss heads (8).
    tau         : float NT-Xent temperature (0.07).
    w_mask      : float Loss weight for L_mask (1.0).
    w_next      : float Loss weight for L_next (0.8).
    w_contrast  : float Loss weight for L_contrast (0.5).
    w_align     : float Loss weight for L_align (0.5).
    """

    def __init__(
        self,
        n_lat_steps : int,
        n_lon_steps : int,
        d_macro     : int   = 10,
        d_micro     : int   = 11,
        d_model     : int   = 256,
        n_layers    : int   = 4,
        d_state     : int   = 64,
        K           : int   = 32,
        d_proj      : int   = 128,
        n_features  : int   = 8,
        tau         : float = 0.07,
        w_mask      : float = 1.0,
        w_next      : float = 0.8,
        w_contrast  : float = 0.5,
        w_align     : float = 0.5,
    ) -> None:
        super().__init__()

        # Store config for repr / checkpointing
        self.config = dict(
            n_lat_steps = n_lat_steps,
            n_lon_steps = n_lon_steps,
            d_macro     = d_macro,
            d_micro     = d_micro,
            d_model     = d_model,
            n_layers    = n_layers,
            d_state     = d_state,
            K           = K,
            d_proj      = d_proj,
            n_features  = n_features,
            tau         = tau,
            w_mask      = w_mask,
            w_next      = w_next,
            w_contrast  = w_contrast,
            w_align     = w_align,
        )

        self.macro_encoder = MacroEncoder(
            n_lat_steps = n_lat_steps,
            n_lon_steps = n_lon_steps,
            d_macro     = d_macro,
            d_model     = d_model,
            n_layers    = n_layers,
            d_state     = d_state,
        )

        self.cross_scale = CrossScaleInjection(
            d_micro = d_micro,
            d_model = d_model,
            K       = K,
        )

        self.micro_encoder = MicroEncoder(
            d_model  = d_model,
            n_layers = n_layers,
            d_state  = d_state,
        )

        self.loss_heads = LossHeads(
            d_model    = d_model,
            d_proj     = d_proj,
            n_features = n_features,
            tau        = tau,
            w_mask     = w_mask,
            w_next     = w_next,
            w_contrast = w_contrast,
            w_align    = w_align,
        )

    # ── Constructors ────────────────────────────────────────────────────────

    @classmethod
    def from_norm_stats(cls, norm_stats_path: str, **kwargs) -> "HBMamba":
        """
        Build the model by reading grid dimensions from norm_stats.json.

        Parameters
        ----------
        norm_stats_path : str  Path to norm_stats.json.
        **kwargs            Any HBMamba __init__ param to override the default.
        """
        with open(norm_stats_path) as f:
            ns = json.load(f)
        return cls(
            n_lat_steps = int(ns["N_LAT_STEPS"]),
            n_lon_steps = int(ns["N_LON_STEPS"]),
            **kwargs,
        )

    # ── Core forward ────────────────────────────────────────────────────────

    def forward(
        self,
        macro_features      : Tensor,                # [B, N_cells, d_macro]
        macro_lat_idx       : Tensor,                # [B, N_cells]  int
        macro_lon_idx       : Tensor,                # [B, N_cells]  int
        micro_tokens        : Tensor,                # [B, N_day, d_micro]  FULL original (loss target)
        mask                : Tensor,                # [B, N_day]  bool  True=masked ping
        padding_mask        : Tensor,                # [B, N_day]  bool  True=padded position
        mmsi                : List[int],             # [B]  vessel identifiers
        micro_tokens_masked : Optional[Tensor] = None,  # [B, N_day, d_micro]  gap-zeroed (encoder input)
    ) -> Dict[str, Tensor]:
        """
        Full forward pass — training mode.

        v2.1 change: accepts optional micro_tokens_masked. When provided,
        the encoder sees the masked version (gap positions zeroed) while
        the loss heads use the full micro_tokens as reconstruction target.
        When None (v2.0 fallback), micro_tokens is used for both.

        Returns
        -------
        dict with keys:
            L_mask, L_next, L_contrast, L_align, L_total
        """
        # v2.1: use masked version for encoder, full version for loss target
        encoder_input = micro_tokens_masked if micro_tokens_masked is not None else micro_tokens

        # Component 1 — Macro context
        macro_output, _ = self.macro_encoder(
            macro_features, macro_lat_idx, macro_lon_idx
        )   # [B, N_cells, d_model]

        # Component 2 — Cross-scale injection (uses masked input)
        # v2.2: pass mask so CSI adds the [MASK] token embedding at masked
        # positions; combined with sinusoidal PE this makes Q unique per
        # position even when features are zeroed.
        x_conditioned, attn_weights, topk_idx = self.cross_scale(
            encoder_input, macro_output, mask=mask
        )   # [B, N_day, d_model]

        # Component 3 — Micro encoding
        # v2.1: pass visibility_mask so H excludes gap + padded positions
        visibility_mask = ~mask & ~padding_mask   # True = visible (not gap, not padded)
        h_fwd, h_bwd, H = self.micro_encoder(x_conditioned, visibility_mask=visibility_mask)
        # h_fwd, h_bwd : [B, N_day, d_model]
        # H            : [B, d_model]

        # Component 4 — Loss heads (uses FULL micro_tokens as target)
        losses = self.loss_heads(
            h_fwd, h_bwd, H,
            macro_output, attn_weights, topk_idx,
            micro_tokens, mask, padding_mask, mmsi,
        )

        return losses

    # ── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        macro_features : Tensor,
        macro_lat_idx  : Tensor,
        macro_lon_idx  : Tensor,
        micro_tokens   : Tensor,
        padding_mask   : Optional[Tensor] = None,
        mask           : Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Inference-only forward — returns hidden states and embeddings, no loss.

        Parameters
        ----------
        macro_features : [B, N_cells, d_macro]
        macro_lat_idx  : [B, N_cells]
        macro_lon_idx  : [B, N_cells]
        micro_tokens   : [B, N_day, d_micro]  — caller is responsible for zeroing
                         gap positions if using v2.1 (pass micro_tokens_masked here).
        padding_mask   : [B, N_day]  bool, optional
        mask           : [B, N_day]  bool, optional — if provided, used with
                         padding_mask to compute H from visible positions only.

        Returns
        -------
        dict with keys:
            macro_output  [B, N_cells, d_model]
            h_fwd         [B, N_day,   d_model]
            h_bwd         [B, N_day,   d_model]
            H             [B, d_model]            trajectory embedding
            attn_weights  [B, N_day,   K]
            topk_idx      [B, N_day,   K]  int64
        """
        macro_output, _ = self.macro_encoder(
            macro_features, macro_lat_idx, macro_lon_idx
        )
        x_cond, attn_weights, topk_idx = self.cross_scale(
            micro_tokens, macro_output, mask=mask
        )

        # v2.1: compute visibility mask for H if mask is provided
        visibility_mask = None
        if mask is not None:
            if padding_mask is not None:
                visibility_mask = ~mask & ~padding_mask
            else:
                visibility_mask = ~mask

        h_fwd, h_bwd, H = self.micro_encoder(x_cond, visibility_mask=visibility_mask)

        return {
            "macro_output" : macro_output,
            "h_fwd"        : h_fwd,
            "h_bwd"        : h_bwd,
            "H"            : H,
            "attn_weights" : attn_weights,
            "topk_idx"     : topk_idx,
        }

    # ── Utilities ────────────────────────────────────────────────────────────

    def param_groups(self, lr: float = 1e-4, weight_decay: float = 0.01):
        """
        Returns AdamW parameter groups with weight decay disabled for biases
        and 1-D parameters (LayerNorm, embedding scales).

        Usage:
            optimizer = torch.optim.AdamW(model.param_groups(), lr=1e-4)
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay,    "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]

    def n_params(self) -> Dict[str, int]:
        """Return parameter counts per component."""
        def _c(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "macro_encoder" : _c(self.macro_encoder),
            "cross_scale"   : _c(self.cross_scale),
            "micro_encoder" : _c(self.micro_encoder),
            "loss_heads"    : _c(self.loss_heads),
            "total"         : _c(self),
        }

    def __repr__(self) -> str:
        counts = self.n_params()
        lines = [
            "HBMamba v2.0",
            f"  n_lat={self.config['n_lat_steps']}  n_lon={self.config['n_lon_steps']}"
            f"  d_model={self.config['d_model']}  n_layers={self.config['n_layers']}"
            f"  K={self.config['K']}",
            f"  macro_encoder : {counts['macro_encoder']:>10,}",
            f"  cross_scale   : {counts['cross_scale']:>10,}",
            f"  micro_encoder : {counts['micro_encoder']:>10,}",
            f"  loss_heads    : {counts['loss_heads']:>10,}",
            f"  total         : {counts['total']:>10,}",
        ]
        return "\n".join(lines)


# =============================================================================
# Self-contained test block
# =============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA device — test requires GPU (Mamba2 is CUDA-only).")
        sys.exit(1)

    device = torch.device("cuda")

    _raw     = _root / "data_genration_and_raw_data" / "raw_data"
    _ns_path = str(_raw / "preprocessing" / "norm_stats" / "norm_stats.json")

    print("=" * 65)
    print(f"HBMamba — HB-Mamba v2.0  self-test  (device: {device})")
    print("=" * 65)

    all_passed = True

    def _check(name: str, cond: bool) -> None:
        global all_passed
        s = "PASS" if cond else "FAIL"
        if not cond:
            all_passed = False
        print(f"  [{s}]  {name}")

    # ── Build model ────────────────────────────────────────────────────────
    model = HBMamba.from_norm_stats(_ns_path).to(device)
    print(f"\n{model}\n")

    # ── Load a real batch ──────────────────────────────────────────────────
    from hb_mamba_dataset import build_dataloaders

    loaders = build_dataloaders(
        dataset_index_dir = str(_raw / "dataset_index"),
        norm_stats_path   = _ns_path,
        batch_size        = 4,
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

    # ── Check 1 — Forward pass returns all five loss keys ──────────────────
    print("Check 1 — Forward pass (training mode)")

    model.train()
    losses = model(
        macro_features, macro_lat_idx, macro_lon_idx,
        micro_tokens, mask, padding_mask, mmsi_list,
    )

    expected_keys = {"L_mask", "L_next", "L_contrast", "L_align", "L_total"}
    _check("returns all 5 loss keys", set(losses.keys()) == expected_keys)
    for k, v in losses.items():
        _check(f"{k} is finite  (={v.item():.4f})", torch.isfinite(v).item())

    # ── Check 2 — Backward through L_total ────────────────────────────────
    print("\nCheck 2 — Backward through L_total")

    # Force positive MMSI pairs so L_contrast > 0 and contrast_head gets grads
    B = macro_features.shape[0]
    mmsi_pairs = mmsi_list[: B // 2] + mmsi_list[: B // 2]

    model.train()
    model.zero_grad()
    losses_bp = model(
        macro_features, macro_lat_idx, macro_lon_idx,
        micro_tokens, mask, padding_mask, mmsi_pairs,
    )
    losses_bp["L_total"].backward()

    def _all_grad(module):
        return all(p.grad is not None for p in module.parameters() if p.requires_grad)

    _check("macro_encoder grads exist", _all_grad(model.macro_encoder))
    _check("cross_scale grads exist",   _all_grad(model.cross_scale))
    _check("micro_encoder grads exist", _all_grad(model.micro_encoder))
    _check("loss_heads grads exist",    _all_grad(model.loss_heads))
    _check("W_Q non-zero (selection path alive)",
           model.cross_scale.W_Q.weight.grad is not None and
           model.cross_scale.W_Q.weight.grad.abs().max().item() > 1e-10)

    # ── Check 3 — Inference / predict mode ────────────────────────────────
    print("\nCheck 3 — predict() inference mode")

    model.eval()
    out = model.predict(
        macro_features, macro_lat_idx, macro_lon_idx,
        micro_tokens, padding_mask,
    )

    B       = macro_features.shape[0]
    N_day   = micro_tokens.shape[1]
    N_cells = macro_features.shape[1]
    K       = model.config["K"]

    _check(f"macro_output shape == ({B}, {N_cells}, 256)",
           out["macro_output"].shape == (B, N_cells, 256))
    _check(f"h_fwd shape        == ({B}, {N_day}, 256)",
           out["h_fwd"].shape == (B, N_day, 256))
    _check(f"H shape            == ({B}, 256)",
           out["H"].shape == (B, 256))
    _check(f"attn_weights shape == ({B}, {N_day}, {K})",
           out["attn_weights"].shape == (B, N_day, K))
    _check("no NaN in H",         not torch.isnan(out["H"]).any())
    _check("no NaN in h_fwd",     not torch.isnan(out["h_fwd"]).any())

    # ── Check 4 — bfloat16 autocast ───────────────────────────────────────
    print("\nCheck 4 — bfloat16 autocast")

    model.train()
    model.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses_bf = model(
            macro_features, macro_lat_idx, macro_lon_idx,
            micro_tokens, mask, padding_mask, mmsi_pairs,  # use forced pairs
        )
    losses_bf["L_total"].backward()

    _check("bfloat16 forward runs without error", True)
    _check("L_total finite under bfloat16",
           torch.isfinite(losses_bf["L_total"]).item())
    _check("backward under bfloat16 succeeds",
           all(p.grad is not None
               for p in model.parameters() if p.requires_grad))

    # ── Check 5 — state_dict save / load round-trip ───────────────────────
    print("\nCheck 5 — state_dict save / load round-trip")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    torch.save({"model_state": model.state_dict(), "config": model.config}, tmp_path)

    model2 = HBMamba(**model.config).to(device)
    ckpt   = torch.load(tmp_path, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt["model_state"])
    os.unlink(tmp_path)

    model2.eval()
    with torch.no_grad():
        out2 = model2.predict(
            macro_features, macro_lat_idx, macro_lon_idx, micro_tokens
        )

    _check("H identical after load (tol=1e-5)",
           torch.allclose(out["H"], out2["H"], atol=1e-5))

    # ── Check 6 — param_groups for AdamW ──────────────────────────────────
    print("\nCheck 6 — param_groups for AdamW")

    groups   = model.param_groups(lr=1e-4, weight_decay=0.01)
    n_decay  = sum(p.numel() for p in groups[0]["params"])
    n_nodecay = sum(p.numel() for p in groups[1]["params"])
    total    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    _check("two param groups returned",   len(groups) == 2)
    _check("decay + no_decay == total",   n_decay + n_nodecay == total)
    _check("decay group has wd=0.01",     groups[0]["weight_decay"] == 0.01)
    _check("no_decay group has wd=0.0",   groups[1]["weight_decay"] == 0.0)

    # ── Check 7 — parameter count ─────────────────────────────────────────
    print("\nCheck 7 — Parameter count")

    counts = model.n_params()
    for k, v in counts.items():
        print(f"  {k:20s} : {v:>10,}")
    _check("total > 7M", counts["total"] > 7_000_000)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_passed:
        print("All checks passed — HBMamba v2.0 OK")
    else:
        print("One or more checks FAILED — see [FAIL] lines above")
    print("=" * 65)
