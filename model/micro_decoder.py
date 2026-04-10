"""
MicroEncoder — HB-Mamba v2.0, Component 3
==========================================

Encodes a vessel's full-day trajectory (after cross-scale macro context has
been injected) into per-ping bidirectional hidden states and a single
trajectory-level embedding H.

Input:  x_conditioned  [B, N_day, d_model]  — output of CrossScaleInjection
Output:
    h_fwd  [B, N_day, d_model]  forward  hidden states (last layer)
    h_bwd  [B, N_day, d_model]  backward hidden states (last layer)
    H      [B, d_model]         trajectory embedding = mean(h_fwd + h_bwd)

Downstream consumers:
    Loss Head 1 (L_mask)     — concat [h_fwd_t, h_bwd_t] [B, N_day, 2×d_model]
    Loss Head 2 (L_next)     — h_fwd only                 [B, N_day, d_model]
    Loss Head 3 (L_contrast) — H                          [B, d_model]
    Loss Head 4 (L_align)    — H + (attn_weights, topk_idx from Component 2)

v1.0 → v2.0 changes
---------------------
    v1.0: causal Mamba (left-to-right only) — could not use post-gap context,
          making interpolation (70% of training) impossible.
    v2.0: bidirectional Mamba2 — every ping sees both past and future context.
          H is mean pool over all positions (not last hidden state) so every
          ping contributes equally to the trajectory embedding.
          Reconstruction head now uses concatenated [h_fwd, h_bwd] [512] instead
          of h_{t-1} alone [256].
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from torch import Tensor


class MicroEncoder(nn.Module):
    """
    Component 3 of HB-Mamba v2.0 — Micro Encoder.

    Runs n_layers bidirectional Mamba2 blocks over the macro-conditioned
    ping sequence.  Intermediate layers sum forward + backward outputs and
    pass the combined tensor to the next layer.  The final layer keeps
    h_fwd and h_bwd separate so the loss heads can use them independently.

    Parameters
    ----------
    d_model : int
        Model dimension.  Input x_conditioned is already in this space
        (CrossScaleInjection projects 11 → d_model), so no input projection
        is needed here.
    n_layers : int
        Number of stacked bidirectional Mamba2 blocks.
    d_state : int
        Mamba2 SSM state dimension.
    """

    def __init__(
        self,
        d_model  : int = 256,   # must match CrossScaleInjection / MacroEncoder
        n_layers : int = 4,     # architecture spec: 4 bidirectional blocks
        d_state  : int = 64,    # Mamba2 default
    ) -> None:
        super().__init__()

        self.d_model  = d_model
        self.n_layers = n_layers
        self.d_state  = d_state

        # ── Bidirectional Mamba2 stack ────────────────────────────────────────
        # Two parallel ModuleLists (fwd + bwd) — must be nn.ModuleList so
        # PyTorch registers all Mamba2 parameters for optimizer and device moves.
        #
        # All n_layers pairs are created identically.  The difference is in the
        # forward pass: intermediate layers sum fwd + bwd; the final layer
        # keeps them separate for the loss heads.
        #
        # SSM recurrence (reference):
        #   h_t = A_bar(x_t) · h_{t-1} + B_bar(x_t) · x_t
        #   y_t = C(x_t) · h_t
        #
        # v2.0 upgrade: Mamba1 (S6) → Mamba2. Faster on Ada GPUs via
        # reformulated parallel scan that maps better to tensor cores.
        self.mamba_fwd = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        self.mamba_bwd = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])

    # -------------------------------------------------------------------------
    def forward(self, x_conditioned: Tensor):
        """
        Forward pass.

        Parameters
        ----------
        x_conditioned : Tensor[B, N_day, d_model]
            Macro-conditioned ping sequence from CrossScaleInjection.
            N_day varies per sample and per batch — never hardcoded.

        Returns
        -------
        h_fwd : Tensor[B, N_day, d_model]
            Forward hidden states from the final Mamba2 layer.
            Used by Loss Head 1 (concat with h_bwd) and Loss Head 2 (alone).
        h_bwd : Tensor[B, N_day, d_model]
            Backward hidden states from the final Mamba2 layer.
            Used by Loss Head 1 (concat with h_fwd).
        H : Tensor[B, d_model]
            Trajectory embedding — mean pool of (h_fwd + h_bwd) over N_day.
            Mean pool is used (not last token) because the sequence is
            temporal and every ping contributes equally to the vessel summary.
            Used by Loss Head 3 (contrastive) and Loss Head 4 (alignment).
        """
        x = x_conditioned   # [B, N_day, d_model]

        # ── Intermediate layers (0 … n_layers-2) ─────────────────────────────
        # Each layer runs fwd + bwd Mamba2, sums the outputs, and passes the
        # combined tensor forward.  Summing keeps dimension at d_model
        # throughout (no expansion).
        for i in range(self.n_layers - 1):
            # Forward direction: left → right along the ping sequence
            y_fwd = self.mamba_fwd[i](x)                          # [B, N_day, d_model]

            # Backward direction: flip sequence, run Mamba2, flip back
            # Flipping lets a standard causal Mamba2 operate right → left;
            # the second flip restores the original temporal ordering.
            y_bwd = self.mamba_bwd[i](x.flip(dims=[1])).flip(dims=[1])   # [B, N_day, d_model]

            # Sum (not concatenate) to keep d_model constant across layers.
            # The model learns to specialise fwd/bwd channels independently.
            x = y_fwd + y_bwd   # [B, N_day, d_model]

        # ── Final layer (n_layers-1) — keep h_fwd and h_bwd separate ─────────
        # Loss Head 1 needs both independently:
        #   concat([h_fwd_t, h_bwd_t]) → [B, N_day, 2×d_model]   for reconstruction
        # Loss Head 2 needs h_fwd only (causal — predicting future pings):
        #   h_fwd_t → [B, N_day, d_model]                         for next-ping
        last = self.n_layers - 1
        h_fwd = self.mamba_fwd[last](x)                                   # [B, N_day, d_model]
        h_bwd = self.mamba_bwd[last](x.flip(dims=[1])).flip(dims=[1])     # [B, N_day, d_model]

        # ── Trajectory embedding H ────────────────────────────────────────────
        # Mean pool over the N_day dimension of the combined bidirectional output.
        # Every ping position is equally weighted — appropriate for a spatial
        # summary of a full-day trajectory (no positional priority).
        #
        # v2.0 change: v1.0 used the final hidden state h_{N_p}. Mean pool is
        # more robust because early pings are as informative as late ones, and
        # it is unaffected by trajectory length (N_day varies per vessel).
        H = (h_fwd + h_bwd).mean(dim=1)   # [B, d_model]

        return h_fwd, h_bwd, H


# =============================================================================
# Self-contained test block
# =============================================================================

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import json
    from model.macro_encoder import MacroEncoder
    from model.cross_scale_injection import CrossScaleInjection

    if not torch.cuda.is_available():
        print("✗  No CUDA device found — test requires a GPU (Mamba2 is CUDA-only).")
        sys.exit(1)

    device = torch.device("cuda")

    # Load real grid config — no hardcoded dims
    _ns_path = _root / "data_genration_and_raw_data" / "raw_data" / \
               "preprocessing" / "norm_stats" / "norm_stats.json"
    with _ns_path.open() as _f:
        _ns = json.load(_f)
    N_LAT  = int(_ns["N_LAT_STEPS"])
    N_LON  = int(_ns["N_LON_STEPS"])
    N_CELL = int(_ns["N_TOTAL_CELLS"])

    print("=" * 65)
    print(f"MicroEncoder — HB-Mamba v2.0  self-test  (device: {device})")
    print(f"Grid from norm_stats: n_lat={N_LAT}, n_lon={N_LON}, N_cells={N_CELL}")
    print("=" * 65)

    all_passed = True

    def _check(name: str, cond: bool) -> None:
        global all_passed
        status = "PASS" if cond else "FAIL"
        if not cond:
            all_passed = False
        print(f"  [{status}]  {name}")

    # Build the full pipeline up to MicroEncoder
    macro_enc  = MacroEncoder(n_lat_steps=N_LAT, n_lon_steps=N_LON).to(device).eval()
    csi        = CrossScaleInjection(d_micro=11, d_model=256, K=32).to(device).eval()
    micro_enc  = MicroEncoder(d_model=256, n_layers=4).to(device).eval()

    # ── Check 1 — Output shapes (realistic batch) ─────────────────────────────
    print("\nCheck 1 — Output shapes  (B=4, N_day=120)")

    B, N_day = 4, 120
    macro_features = torch.randn(B, N_CELL, 10,           device=device)
    macro_lat_idx  = torch.randint(0, N_LAT, (B, N_CELL), device=device)
    macro_lon_idx  = torch.randint(0, N_LON, (B, N_CELL), device=device)
    micro_tokens   = torch.randn(B, N_day,  11,           device=device)

    with torch.no_grad():
        macro_out, z_global              = macro_enc(macro_features, macro_lat_idx, macro_lon_idx)
        x_cond, attn_w, topk_idx         = csi(micro_tokens, macro_out)
        h_fwd, h_bwd, H                  = micro_enc(x_cond)

    _check(f"h_fwd.shape == ({B}, {N_day}, 256)",  h_fwd.shape == (B, N_day, 256))
    _check(f"h_bwd.shape == ({B}, {N_day}, 256)",  h_bwd.shape == (B, N_day, 256))
    _check(f"H.shape     == ({B}, 256)",            H.shape     == (B, 256))
    _check("no NaN in h_fwd",                       not torch.isnan(h_fwd).any())
    _check("no NaN in h_bwd",                       not torch.isnan(h_bwd).any())
    _check("no NaN in H",                           not torch.isnan(H).any())

    # ── Check 2 — H is mean(h_fwd + h_bwd) over N_day ────────────────────────
    print("\nCheck 2 — H == mean(h_fwd + h_bwd, dim=1)")

    H_expected = (h_fwd + h_bwd).mean(dim=1)
    _check("H == mean(h_fwd + h_bwd) (tol=1e-5)",
           torch.allclose(H, H_expected, atol=1e-5))

    # ── Check 3 — Variable N_day (no hardcoding) ──────────────────────────────
    print("\nCheck 3 — Variable N_day  (N_day=473, mimics a real vessel)")

    N_day2       = 473
    micro_tokens2 = torch.randn(B, N_day2, 11, device=device)

    with torch.no_grad():
        x_cond2, _, _ = csi(micro_tokens2, macro_out)
        h_fwd2, h_bwd2, H2 = micro_enc(x_cond2)

    _check(f"h_fwd.shape == ({B}, {N_day2}, 256)", h_fwd2.shape == (B, N_day2, 256))
    _check(f"h_bwd.shape == ({B}, {N_day2}, 256)", h_bwd2.shape == (B, N_day2, 256))
    _check(f"H.shape     == ({B}, 256)",            H2.shape     == (B, 256))

    # ── Check 4 — Loss head input shapes ──────────────────────────────────────
    print("\nCheck 4 — Loss head input shapes")

    # Loss Head 1 input: concat [h_fwd, h_bwd] per ping → [B, N_day, 512]
    recon_input = torch.cat([h_fwd, h_bwd], dim=-1)
    _check(f"concat([h_fwd, h_bwd]).shape == ({B}, {N_day}, 512)",
           recon_input.shape == (B, N_day, 512))

    # Loss Head 2 input: h_fwd only → [B, N_day, 256]
    _check(f"h_fwd (next-ping input).shape == ({B}, {N_day}, 256)",
           h_fwd.shape == (B, N_day, 256))

    # Loss Head 3 input: H → [B, 256]
    _check(f"H (contrastive input).shape == ({B}, 256)",
           H.shape == (B, 256))

    # ── Check 5 — bfloat16 compatibility ──────────────────────────────────────
    print("\nCheck 5 — bfloat16 compatibility (torch.autocast)")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        macro_bf, _         = macro_enc(macro_features, macro_lat_idx, macro_lon_idx)
        x_cond_bf, _, _     = csi(micro_tokens, macro_bf)
        h_fwd_bf, h_bwd_bf, H_bf = micro_enc(x_cond_bf)

    _check("runs under bfloat16 autocast without error", True)
    _check("h_fwd no NaN under bfloat16", not torch.isnan(h_fwd_bf.float()).any())
    _check("h_bwd no NaN under bfloat16", not torch.isnan(h_bwd_bf.float()).any())
    _check("H     no NaN under bfloat16", not torch.isnan(H_bf.float()).any())

    # ── Check 6 — Gradient flow through all three components ──────────────────
    print("\nCheck 6 — Gradient flow (MacroEncoder → CSI → MicroEncoder)")

    macro_enc_g = MacroEncoder(n_lat_steps=N_LAT, n_lon_steps=N_LON).to(device).train()
    csi_g       = CrossScaleInjection().to(device).train()
    micro_enc_g = MicroEncoder().to(device).train()

    mf = torch.randn(2, N_CELL, 10,           device=device)
    la = torch.randint(0, N_LAT, (2, N_CELL), device=device)
    lo = torch.randint(0, N_LON, (2, N_CELL), device=device)
    mi = torch.randn(2, 60, 11,               device=device)

    mo_g, _          = macro_enc_g(mf, la, lo)
    xc_g, _, _       = csi_g(mi, mo_g)
    hf_g, hb_g, H_g  = micro_enc_g(xc_g)

    # Simulate all three loss head inputs in one scalar
    loss = hf_g.mean() + hb_g.mean() + H_g.mean()
    loss.backward()

    _check("gradients flow through MicroEncoder",
           all(p.grad is not None for p in micro_enc_g.parameters()))
    _check("gradients flow back into CrossScaleInjection",
           all(p.grad is not None for p in csi_g.parameters()))
    _check("gradients flow back into MacroEncoder",
           all(p.grad is not None for p in macro_enc_g.parameters() if p.requires_grad))

    # ── Check 7 — Parameter count ──────────────────────────────────────────────
    print("\nCheck 7 — Parameter count")

    total      = sum(p.numel() for p in micro_enc.parameters() if p.requires_grad)
    fwd_params = sum(p.numel() for p in micro_enc.mamba_fwd.parameters())
    bwd_params = sum(p.numel() for p in micro_enc.mamba_bwd.parameters())

    print(f"  mamba_fwd (4 layers) : {fwd_params:>10,}")
    print(f"  mamba_bwd (4 layers) : {bwd_params:>10,}")
    print(f"  ────────────────────────────────────────")
    print(f"  TOTAL                : {total:>10,}")

    _check("mamba_fwd and mamba_bwd have identical param counts", fwd_params == bwd_params)
    _check("total params > 0",                                    total > 0)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_passed:
        print("✓  All checks passed — MicroEncoder v2.0 OK")
    else:
        print("✗  One or more checks FAILED — see [FAIL] lines above")
    print("=" * 65)
