"""
MacroEncoder — HB-Mamba v2.0, Component 1
==========================================

Reads the daily fleet-level traffic state as a sequence of [N_cells, 10]
grid-cell tokens and produces:
  - macro_output : [B, N_cells, d_model]  — full spatial sequence for cross-attention
  - Z_global     : [B, d_model]           — mean-pooled summary for alignment loss

v1.0 → v2.0 changes in this component
---------------------------------------
  v1.0: no geographic embedding; output was a single mean-pooled vector Z [d_model]
        used for BOTH cross-attention and alignment loss.
  v2.0: geographic embedding (Embed_lat + Embed_lon) added; full sequence
        [N_cells, d_model] returned for cross-attention; Z_global (mean pool)
        returned separately for alignment loss only.
        Mamba 1 (S6) replaced with Mamba 2 (faster on Ada GPUs, d_state=64 default).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class MacroEncoder(nn.Module):
    """
    Component 1 of HB-Mamba v2.0 — Macro Encoder.

    Encodes a daily AIS grid snapshot into a spatially-rich sequence that
    downstream cross-scale injection can attend to.

    Pipeline (strict order):
        A. Geographic embedding  : lat/lon indices → learned spatial identity
        B. Input projection      : raw 10-dim features → d_model
           Fusion                : traffic_proj + geo_embed → cell_input
        C. Bidirectional Mamba 2 : n_layers of forward + backward SSM stacks
        D. Output                : full sequence + mean-pooled global vector

    Parameters
    ----------
    n_lat_steps : int
        Number of latitude  bin steps — set by preprocessing config.
        No default: must always match the data the model is trained on.
    n_lon_steps : int
        Number of longitude bin steps — set by preprocessing config.
        No default: must always match the data the model is trained on.
    d_macro : int
        Raw input feature dimension. Matches the 10 features produced by the
        macro preprocessor (vessel count, ping count, type distribution, etc.).
    d_model : int
        Internal model dimension. Also the geographic embedding dimension so
        that additive fusion (geo_embed + traffic_proj) is dimension-consistent.
    n_layers : int
        Number of stacked bidirectional Mamba 2 blocks.
    d_state : int
        Mamba 2 SSM state dimension (library default = 64).
    """

    def __init__(
        self,
        n_lat_steps : int,          # no default — caller must set from preprocessing config
        n_lon_steps : int,          # no default — caller must set from preprocessing config
        d_macro     : int = 10,     # raw input feature dim
        d_model     : int = 256,    # internal model / embedding dimension
        n_layers    : int = 4,      # number of bidirectional Mamba 2 blocks
        d_state     : int = 64,     # Mamba 2 SSM state size
    ) -> None:
        super().__init__()

        # Store config so the caller can inspect it after construction
        self.n_lat_steps = n_lat_steps
        self.n_lon_steps = n_lon_steps
        self.d_macro     = d_macro
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.d_state     = d_state

        # ── Step A — Geographic Embedding ────────────────────────────────────
        # Two independent embedding tables: one for latitude bins, one for
        # longitude bins.  Their outputs are summed (not concatenated) so the
        # combined geo embedding lives in the same d_model space as the traffic
        # features — enabling direct additive fusion in Step B.
        #
        # v2.0 addition: v1.0 had no geographic embedding, meaning the model
        # had no stable spatial reference and had to infer location from the
        # traffic feature values alone.
        #
        # Parameter cost: (n_lat_steps + n_lon_steps) × d_model
        #   e.g. (29 + 36) × 256 = 16,640 params  — negligible.
        self.Embed_lat = nn.Embedding(n_lat_steps, d_model)
        self.Embed_lon = nn.Embedding(n_lon_steps, d_model)

        # ── Step B — Input Projection ─────────────────────────────────────────
        # Project raw traffic features from d_macro (10) to d_model (256).
        # This brings traffic features into the same embedding space as the
        # geographic embeddings so they can be fused with simple addition.
        self.input_proj = nn.Linear(d_macro, d_model)

        # ── Step C — Bidirectional Mamba 2 Stack ──────────────────────────────
        # Two parallel ModuleLists: one for the forward pass (left→right over
        # the cell sequence), one for the backward pass (right→left).
        #
        # IMPORTANT: must be nn.ModuleList — NOT plain Python lists.
        # nn.ModuleList registers every Mamba2 instance as a proper sub-module
        # so PyTorch can track its parameters, move it to device, serialise it,
        # etc.  A plain list makes all parameters invisible to the optimizer.
        #
        # SSM recurrence (for reference):
        #   h_t = A_bar(x_t) · h_{t-1} + B_bar(x_t) · x_t
        #   y_t = C(x_t) · h_t
        #
        # v2.0 upgrade: Mamba 1 (S6) → Mamba 2.  Mamba 2 is ~2–3× faster on
        # Ada GPUs due to a reformulated parallel scan that maps better to
        # tensor-core workloads.
        self.mamba_fwd = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        self.mamba_bwd = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])

    # -------------------------------------------------------------------------
    def forward(
        self,
        macro_features : torch.Tensor,   # [B, N_cells, d_macro]  float32
        macro_lat_idx  : torch.Tensor,   # [B, N_cells]            int64
        macro_lon_idx  : torch.Tensor,   # [B, N_cells]            int64
    ):
        """
        Forward pass of the MacroEncoder.

        N_cells is derived at runtime from the incoming tensor shape and is
        NEVER hardcoded.  The model handles any grid configuration without
        code changes.

        Parameters
        ----------
        macro_features : Tensor[B, N_cells, d_macro]
            Normalised traffic features for every grid cell.
            Empty cells (no vessels that day) carry zero feature vectors.
        macro_lat_idx : Tensor[B, N_cells]  int64
            Latitude  bin index per cell, in [0, n_lat_steps).
        macro_lon_idx : Tensor[B, N_cells]  int64
            Longitude bin index per cell, in [0, n_lon_steps).

        Returns
        -------
        macro_output : Tensor[B, N_cells, d_model]
            Full processed spatial sequence.
            Used as Keys and Values in Top-K=32 cross-attention (Component 2).
            Must NOT be pooled before returning — the full sequence is required.
        Z_global : Tensor[B, d_model]
            Global summary vector (mean pool of macro_output over cells).
            Used ONLY as input to the alignment loss (Loss Head 4).
            NOT used for cross-attention in v2.0.
        """

        # ── Step A — Geographic Embedding ────────────────────────────────────
        # Look up both embedding tables and sum.
        # Summing (not concatenating) keeps the output in d_model space, which
        # is required for the additive fusion with traffic features in Step B.
        #
        #   Embed_lat : [B, N_cells] int64 → [B, N_cells, d_model]
        #   Embed_lon : [B, N_cells] int64 → [B, N_cells, d_model]
        #   geo_embed : [B, N_cells, d_model]  (element-wise sum)
        geo_embed = self.Embed_lat(macro_lat_idx) + self.Embed_lon(macro_lon_idx)

        # ── Step B — Input Projection & Fusion ───────────────────────────────
        # Project raw traffic features from d_macro → d_model.
        #   macro_features : [B, N_cells, d_macro]
        #   traffic_proj   : [B, N_cells, d_model]
        traffic_proj = self.input_proj(macro_features)

        # Additive fusion: combine geographic identity with traffic content.
        # Both tensors are in d_model space so element-wise addition is valid.
        # This lets the model learn to route spatial patterns through the SSM
        # without losing track of which cell each token belongs to.
        #   cell_input : [B, N_cells, d_model]
        cell_input = traffic_proj + geo_embed

        # ── Step C — Bidirectional Mamba 2 Stack ──────────────────────────────
        # x accumulates the contextualised cell representations across layers.
        # After each layer, every cell token has attended to its spatial
        # neighbours in both directions (forward = south→north along sorted
        # cells; backward = north→south).
        x = cell_input   # [B, N_cells, d_model]

        for i in range(self.n_layers):

            # Forward direction — process cell sequence left to right.
            # Each cell's hidden state is informed by all cells to its left.
            #   y_fwd : [B, N_cells, d_model]
            y_fwd = self.mamba_fwd[i](x)

            # Backward direction — flip sequence, run Mamba, flip back.
            # Flipping lets a standard causal Mamba2 see the sequence in
            # reverse; the second flip restores the original cell ordering.
            # Each cell's hidden state is now also informed by all cells to
            # its right.
            #   y_bwd : [B, N_cells, d_model]
            y_bwd = self.mamba_bwd[i](x.flip(dims=[1])).flip(dims=[1])

            # Sum (NOT concatenate) forward and backward outputs.
            # Summation keeps the dimension at d_model throughout the stack,
            # avoiding the dimension explosion that concatenation would cause.
            # The model can learn to specialise fwd/bwd channels independently.
            x = y_fwd + y_bwd   # [B, N_cells, d_model]

        # ── Step D — Output ───────────────────────────────────────────────────
        # The final x IS macro_output — no further projection needed.
        # Every cell token now encodes bidirectional context from all other cells.
        macro_output = x   # [B, N_cells, d_model]

        # Z_global: collapse the spatial sequence to a single summary vector.
        # Mean pool is used rather than last-token or CLS because the cell
        # sequence is spatial (not temporal) — every cell contributes equally
        # to the global fleet state.
        #
        # v2.0 change: in v1.0 this was the ONLY output and was used for both
        # cross-attention and alignment loss.  In v2.0, cross-attention uses
        # the full macro_output sequence (Top-K=32 attention); Z_global is
        # reserved for the alignment loss (L_align) only.
        Z_global = macro_output.mean(dim=1)   # [B, d_model]

        return macro_output, Z_global


# =============================================================================
# Self-contained test block
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path so imports resolve when running this file directly
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    # Mamba2 (mamba_ssm) requires CUDA — abort early with a clear message if
    # no GPU is available rather than failing deep inside a Triton kernel.
    if not torch.cuda.is_available():
        print("✗  No CUDA device found — MacroEncoder test requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    print("=" * 65)
    print(f"MacroEncoder — HB-Mamba v2.0  self-test  (device: {device})")
    print("=" * 65)

    all_passed = True

    def _check(name: str, cond: bool) -> None:
        global all_passed
        status = "PASS" if cond else "FAIL"
        if not cond:
            all_passed = False
        print(f"  [{status}]  {name}")

    # ── Check 1 — Standard production config ─────────────────────────────────
    print("\nCheck 1 — Standard production config")
    print(f"  n_lat_steps=28  n_lon_steps=36  →  N_cells = {28*36}")

    encoder_1 = MacroEncoder(n_lat_steps=28, n_lon_steps=36).to(device)
    encoder_1.eval()

    B1, N1 = 2, 28 * 36   # N_cells derived from grid dims — never hardcoded
    features_1  = torch.randn(B1, N1, 10, device=device)
    lat_idx_1   = torch.randint(0, 28, (B1, N1), device=device)
    lon_idx_1   = torch.randint(0, 36, (B1, N1), device=device)

    with torch.no_grad():
        macro_out_1, z_global_1 = encoder_1(features_1, lat_idx_1, lon_idx_1)

    _check(f"macro_output.shape == ({B1}, {N1}, 256)",  macro_out_1.shape == (B1, N1, 256))
    _check(f"Z_global.shape     == ({B1}, 256)",         z_global_1.shape == (B1, 256))
    _check("no NaN in macro_output",                     not torch.isnan(macro_out_1).any())
    _check("no NaN in Z_global",                         not torch.isnan(z_global_1).any())
    _check("Z_global == macro_output.mean(dim=1)",
           torch.allclose(z_global_1, macro_out_1.mean(dim=1), atol=1e-4))

    # ── Check 2 — Different bounding box / resolution (proves no hardcoding) ──
    print("\nCheck 2 — Different bbox/resolution  (no hardcoded integers)")
    print(f"  n_lat_steps=24  n_lon_steps=39  →  N_cells = {24*39}")

    encoder_2 = MacroEncoder(n_lat_steps=24, n_lon_steps=39).to(device)
    encoder_2.eval()

    B2, N2 = 4, 24 * 39
    features_2  = torch.randn(B2, N2, 10, device=device)
    lat_idx_2   = torch.randint(0, 24, (B2, N2), device=device)
    lon_idx_2   = torch.randint(0, 39, (B2, N2), device=device)

    with torch.no_grad():
        macro_out_2, z_global_2 = encoder_2(features_2, lat_idx_2, lon_idx_2)

    _check(f"macro_output.shape == ({B2}, {N2}, 256)",  macro_out_2.shape == (B2, N2, 256))
    _check(f"Z_global.shape     == ({B2}, 256)",         z_global_2.shape == (B2, 256))
    _check("no NaN in macro_output",                     not torch.isnan(macro_out_2).any())
    _check("no NaN in Z_global",                         not torch.isnan(z_global_2).any())

    # ── Check 3 — bfloat16 compatibility ──────────────────────────────────────
    print("\nCheck 3 — bfloat16 compatibility (torch.autocast)")

    encoder_bf = MacroEncoder(n_lat_steps=28, n_lon_steps=36).to(device)
    encoder_bf.eval()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        macro_out_bf, z_global_bf = encoder_bf(features_1, lat_idx_1, lon_idx_1)

    _check("runs under bfloat16 autocast without error", True)
    _check("macro_output has no NaN under bfloat16",     not torch.isnan(macro_out_bf.float()).any())

    # ── Check 4 — Parameter count ─────────────────────────────────────────────
    print("\nCheck 4 — Parameter count  (encoder_1: n_lat=28, n_lon=36, d_model=256)")

    total_params = sum(p.numel() for p in encoder_1.parameters() if p.requires_grad)
    embed_lat_p  = sum(p.numel() for p in encoder_1.Embed_lat.parameters())
    embed_lon_p  = sum(p.numel() for p in encoder_1.Embed_lon.parameters())
    input_proj_p = sum(p.numel() for p in encoder_1.input_proj.parameters())
    mamba_fwd_p  = sum(p.numel() for p in encoder_1.mamba_fwd.parameters())
    mamba_bwd_p  = sum(p.numel() for p in encoder_1.mamba_bwd.parameters())

    print(f"  Embed_lat   : {embed_lat_p:>10,}  (expected ≈ {28*256:,})")
    print(f"  Embed_lon   : {embed_lon_p:>10,}  (expected ≈ {36*256:,})")
    print(f"  input_proj  : {input_proj_p:>10,}  (expected ≈ {10*256+256:,})")
    print(f"  mamba_fwd   : {mamba_fwd_p:>10,}")
    print(f"  mamba_bwd   : {mamba_bwd_p:>10,}")
    print(f"  ─────────────────────────────────────────")
    print(f"  TOTAL       : {total_params:>10,}")

    _check("Embed_lat params == 28 × 256",              embed_lat_p == 28 * 256)
    _check("Embed_lon params == 36 × 256",              embed_lon_p == 36 * 256)
    _check("input_proj params == 10×256 + 256",         input_proj_p == 10 * 256 + 256)
    _check("mamba_fwd and mamba_bwd same param count",  mamba_fwd_p == mamba_bwd_p)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_passed:
        print("✓  All checks passed — MacroEncoder v2.0 OK")
    else:
        print("✗  One or more checks FAILED — see [FAIL] lines above")
    print("=" * 65)
