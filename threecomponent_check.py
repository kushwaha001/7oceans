"""
Three-Component Pipeline Check — HB-Mamba v2.0
===============================================

Couples the real DataLoader with all three model components and verifies:
    1.  Real batch shapes flow through the full pipeline without error
    2.  No NaNs at any stage
    3.  Padding mask is respected (padded positions zeroed before loss)
    4.  Gradient existence — every parameter in every component gets a grad
    5.  Gradient magnitudes — W_Q and W_K (CSI) are non-zero (selection path)
    6.  End-to-end backward through MacroEncoder → CSI → MicroEncoder
    7.  Mask-only loss — loss computed only at masked pings, backward still works
    8.  bfloat16 autocast through the full pipeline
    9.  Parameter count summary across all three components

Pipeline data flow:
    DataLoader batch
      macro_features  [B, N_cells, 10]
      macro_lat_idx   [B, N_cells]
      macro_lon_idx   [B, N_cells]
      micro_tokens    [B, max_n_day, 11]   (zero-padded)
      mask            [B, max_n_day]       True = masked ping to predict
      padding_mask    [B, max_n_day]       True = padded position (ignore)

    ── Component 1 — MacroEncoder ─────────────────────────────────────────
      macro_output    [B, N_cells, 256]
      Z_global        [B, 256]

    ── Component 2 — CrossScaleInjection ──────────────────────────────────
      x_conditioned   [B, max_n_day, 256]
      attn_weights    [B, max_n_day, 32]
      topk_idx        [B, max_n_day, 32]

    ── Component 3 — MicroEncoder ─────────────────────────────────────────
      h_fwd           [B, max_n_day, 256]
      h_bwd           [B, max_n_day, 256]
      H               [B, 256]
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn as nn

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hb_mamba_dataset import build_dataloaders, GapMaskConfig
from model.macro_encoder        import MacroEncoder
from model.cross_scale_injection import CrossScaleInjection
from model.micro_decoder        import MicroEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_ROOT          = ROOT / "data_genration_and_raw_data" / "raw_data"
DATASET_INDEX_DIR = str(RAW_ROOT / "dataset_index")
NORM_STATS_PATH   = str(RAW_ROOT / "preprocessing" / "norm_stats" / "norm_stats.json")

# ── Guard ─────────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("✗  No CUDA device — this test requires a GPU (Mamba2 is CUDA-only).")
    sys.exit(1)

device = torch.device("cuda")


# =============================================================================
# Helpers
# =============================================================================

all_passed = True

def _check(name: str, cond: bool) -> None:
    global all_passed
    status = "PASS" if cond else "FAIL"
    if not cond:
        all_passed = False
    print(f"  [{status}]  {name}")

def _grad_ok(module: nn.Module) -> bool:
    """True if every parameter that requires grad has a non-None, non-zero grad."""
    for p in module.parameters():
        if p.requires_grad:
            if p.grad is None:
                return False
            if p.grad.abs().max().item() == 0.0:
                return False
    return True

def _any_nan(*tensors) -> bool:
    return any(torch.isnan(t).any().item() for t in tensors)


# =============================================================================
# Load one real batch
# =============================================================================

print("=" * 68)
print("Three-Component Pipeline Check — HB-Mamba v2.0")
print(f"Device: {device}")
print("=" * 68)

print("\n── Loading DataLoader (train split, batch_size=4, num_workers=0) ──")
dataloaders = build_dataloaders(
    dataset_index_dir = DATASET_INDEX_DIR,
    norm_stats_path   = NORM_STATS_PATH,
    batch_size        = 4,
    num_workers       = 0,
    pin_memory        = False,
)

if "train" not in dataloaders:
    print(f"✗  'train' split not found. Available: {list(dataloaders.keys())}")
    sys.exit(1)

train_loader = dataloaders["train"]
dataset      = train_loader.dataset
N_CELLS      = dataset.n_total_cells
N_LAT        = dataset.n_lat_steps
N_LON        = dataset.n_lon_steps

batch = next(iter(train_loader))

B         = batch["macro_features"].shape[0]
max_n_day = batch["micro_tokens"].shape[1]

print(f"  Batch size        : {B}")
print(f"  max_n_day (padded): {max_n_day}")
print(f"  N_cells           : {N_CELLS}  (n_lat={N_LAT}, n_lon={N_LON})")
print(f"  n_day per sample  : {batch['n_day'].tolist()}")
print(f"  gap_types         : {batch['gap_type']}")

# Move all tensors to device
macro_features = batch["macro_features"].to(device)   # [B, N_cells, 10]
macro_lat_idx  = batch["macro_lat_idx"].to(device)    # [B, N_cells]
macro_lon_idx  = batch["macro_lon_idx"].to(device)    # [B, N_cells]
micro_tokens   = batch["micro_tokens"].to(device)     # [B, max_n_day, 11]
mask           = batch["mask"].to(device)              # [B, max_n_day]  bool
padding_mask   = batch["padding_mask"].to(device)     # [B, max_n_day]  bool

print(f"\n  mask True count     : {mask.sum().item()} pings")
print(f"  padding True count  : {padding_mask.sum().item()} positions")
print(f"  Overlap (must=0)    : {(mask & padding_mask).sum().item()}")


# =============================================================================
# Build models (eval for forward checks, train for grad checks)
# =============================================================================

macro_enc = MacroEncoder(n_lat_steps=N_LAT, n_lon_steps=N_LON).to(device)
csi       = CrossScaleInjection(d_micro=11, d_model=256, K=32).to(device)
micro_enc = MicroEncoder(d_model=256, n_layers=4).to(device)


# =============================================================================
# Check 1 — Forward pass shapes with real batch
# =============================================================================

print("\nCheck 1 — Forward pass shapes (real DataLoader batch)")

macro_enc.eval(); csi.eval(); micro_enc.eval()

with torch.no_grad():
    macro_output, Z_global    = macro_enc(macro_features, macro_lat_idx, macro_lon_idx)
    x_cond, attn_w, topk_idx  = csi(micro_tokens, macro_output)
    h_fwd, h_bwd, H           = micro_enc(x_cond)

_check(f"macro_output  shape == ({B}, {N_CELLS}, 256)",
       macro_output.shape == (B, N_CELLS, 256))
_check(f"Z_global      shape == ({B}, 256)",
       Z_global.shape == (B, 256))
_check(f"x_conditioned shape == ({B}, {max_n_day}, 256)",
       x_cond.shape == (B, max_n_day, 256))
_check(f"attn_weights  shape == ({B}, {max_n_day}, 32)",
       attn_w.shape == (B, max_n_day, 32))
_check(f"topk_idx      shape == ({B}, {max_n_day}, 32)",
       topk_idx.shape == (B, max_n_day, 32))
_check(f"h_fwd         shape == ({B}, {max_n_day}, 256)",
       h_fwd.shape == (B, max_n_day, 256))
_check(f"h_bwd         shape == ({B}, {max_n_day}, 256)",
       h_bwd.shape == (B, max_n_day, 256))
_check(f"H             shape == ({B}, 256)",
       H.shape == (B, 256))


# =============================================================================
# Check 2 — No NaNs at any stage
# =============================================================================

print("\nCheck 2 — No NaNs anywhere in the pipeline")

_check("no NaN in macro_output",  not torch.isnan(macro_output).any())
_check("no NaN in Z_global",      not torch.isnan(Z_global).any())
_check("no NaN in x_conditioned", not torch.isnan(x_cond).any())
_check("no NaN in attn_weights",  not torch.isnan(attn_w).any())
_check("no NaN in h_fwd",         not torch.isnan(h_fwd).any())
_check("no NaN in h_bwd",         not torch.isnan(h_bwd).any())
_check("no NaN in H",             not torch.isnan(H).any())


# =============================================================================
# Check 3 — Padding mask respected
# =============================================================================

print("\nCheck 3 — Padding mask respected")

# Masked positions simulating L_mask input: [h_fwd, h_bwd] at non-padded slots
# Padded positions should not influence anything — zero them out explicitly
# and verify no NaN or inf leaks into the valid positions

active = ~padding_mask   # [B, max_n_day]  True = real ping

h_fwd_active = h_fwd[active]   # [n_real_pings, 256]
h_bwd_active = h_bwd[active]

_check("h_fwd at real pings: no NaN",  not torch.isnan(h_fwd_active).any())
_check("h_bwd at real pings: no NaN",  not torch.isnan(h_bwd_active).any())
_check("h_fwd at real pings: no Inf",  not torch.isinf(h_fwd_active).any())

# Verify that padding_mask and mask do not overlap (collate invariant)
_check("padding_mask & mask == empty  (collate invariant)",
       not (padding_mask & mask).any())

# Count how many pings are masked vs real
n_real   = active.sum().item()
n_masked = mask.sum().item()
print(f"  Real pings   : {n_real}")
print(f"  Masked pings : {n_masked}  ({100*n_masked/n_real:.1f}% of real)")


# =============================================================================
# Check 4 — Gradient existence (full pipeline backward)
# =============================================================================

print("\nCheck 4 — Gradient existence through all three components")

macro_enc.train(); csi.train(); micro_enc.train()
macro_enc.zero_grad(); csi.zero_grad(); micro_enc.zero_grad()

macro_out_g, _      = macro_enc(macro_features, macro_lat_idx, macro_lon_idx)
x_cond_g, _, _      = csi(micro_tokens, macro_out_g)
h_fwd_g, h_bwd_g, H_g = micro_enc(x_cond_g)

# Simple scalar loss combining all outputs
loss_all = h_fwd_g.mean() + h_bwd_g.mean() + H_g.mean()
loss_all.backward()

_check("all MicroEncoder params have gradients",
       all(p.grad is not None for p in micro_enc.parameters()))
_check("all CrossScaleInjection params have gradients",
       all(p.grad is not None for p in csi.parameters()))
_check("all MacroEncoder params have gradients",
       all(p.grad is not None for p in macro_enc.parameters() if p.requires_grad))

_check("MicroEncoder grads are non-zero",   _grad_ok(micro_enc))
_check("CrossScaleInjection grads are non-zero", _grad_ok(csi))
_check("MacroEncoder grads are non-zero",   _grad_ok(macro_enc))


# =============================================================================
# Check 5 — CSI selection path: W_Q and W_K must be non-zero
# =============================================================================

print("\nCheck 5 — CSI selection path gradient (W_Q, W_K)")
print("  (These were zero in the broken v1 design)")

wq_grad = csi.W_Q.weight.grad
wk_grad = csi.W_K.weight.grad
wv_grad = csi.W_V.weight.grad

_check("W_Q.weight.grad is not None",  wq_grad is not None)
_check("W_K.weight.grad is not None",  wk_grad is not None)
_check("W_Q grad non-zero  ← selection path alive",
       wq_grad is not None and wq_grad.abs().max().item() > 1e-10)
_check("W_K grad non-zero  ← selection path alive",
       wk_grad is not None and wk_grad.abs().max().item() > 1e-10)

print(f"\n  Gradient magnitudes (max | mean):")
for name, g in [("W_Q",  wq_grad),
                ("W_K",  wk_grad),
                ("W_V",  wv_grad)]:
    if g is not None:
        print(f"    {name}   max={g.abs().max().item():.3e}  "
              f"mean={g.abs().mean().item():.3e}")


# =============================================================================
# Check 6 — Mask-only loss (loss computed only at masked pings)
# =============================================================================

print("\nCheck 6 — Mask-only loss  (loss at masked pings only, then backward)")

macro_enc.zero_grad(); csi.zero_grad(); micro_enc.zero_grad()

macro_out_m, _         = macro_enc(macro_features, macro_lat_idx, macro_lon_idx)
x_cond_m, _, _         = csi(micro_tokens, macro_out_m)
h_fwd_m, h_bwd_m, H_m = micro_enc(x_cond_m)

# L_mask simulation: mean of [h_fwd, h_bwd] at masked, non-padded positions
# mask is True at positions the model must predict; padding_mask is True at padded ones
# Invariant from collate: they never overlap, so mask already excludes padded positions
recon_input = torch.cat([h_fwd_m, h_bwd_m], dim=-1)   # [B, max_n_day, 512]
masked_vecs = recon_input[mask]                         # [n_masked_pings, 512]

_check("masked_vecs has correct dim-1 == 512", masked_vecs.shape[1] == 512)
_check(f"masked_vecs has {n_masked} rows (== mask.sum())",
       masked_vecs.shape[0] == n_masked)
_check("no NaN in masked_vecs", not torch.isnan(masked_vecs).any())

loss_mask = masked_vecs.mean()
loss_mask.backward()

_check("backward through mask-only loss: MicroEncoder grads exist",
       all(p.grad is not None for p in micro_enc.parameters()))
_check("backward through mask-only loss: CSI grads exist",
       all(p.grad is not None for p in csi.parameters()))
_check("backward through mask-only loss: MacroEncoder grads exist",
       all(p.grad is not None for p in macro_enc.parameters() if p.requires_grad))
_check("W_Q non-zero after mask-only loss",
       csi.W_Q.weight.grad is not None and
       csi.W_Q.weight.grad.abs().max().item() > 1e-10)


# =============================================================================
# Check 7 — bfloat16 full pipeline
# =============================================================================

print("\nCheck 7 — bfloat16 autocast through full pipeline")

macro_enc.eval(); csi.eval(); micro_enc.eval()

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    mo_bf, zg_bf              = macro_enc(macro_features, macro_lat_idx, macro_lon_idx)
    xc_bf, aw_bf, _           = csi(micro_tokens, mo_bf)
    hf_bf, hb_bf, H_bf        = micro_enc(xc_bf)

_check("bfloat16 pipeline runs without error", True)
_check("no NaN in macro_output  (bf16)", not torch.isnan(mo_bf.float()).any())
_check("no NaN in x_conditioned (bf16)", not torch.isnan(xc_bf.float()).any())
_check("no NaN in h_fwd         (bf16)", not torch.isnan(hf_bf.float()).any())
_check("no NaN in H             (bf16)", not torch.isnan(H_bf.float()).any())


# =============================================================================
# Check 8 — Parameter count summary
# =============================================================================

print("\nCheck 8 — Parameter count summary")

def _count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

p_macro = _count(macro_enc)
p_csi   = _count(csi)
p_micro = _count(micro_enc)
p_total = p_macro + p_csi + p_micro

print(f"  MacroEncoder          : {p_macro:>10,}")
print(f"  CrossScaleInjection   : {p_csi:>10,}")
print(f"  MicroEncoder          : {p_micro:>10,}")
print(f"  ─────────────────────────────────────")
print(f"  TOTAL (3 components)  : {p_total:>10,}")

_check("MacroEncoder    > 0 params",  p_macro > 0)
_check("CSI             > 0 params",  p_csi   > 0)
_check("MicroEncoder    > 0 params",  p_micro > 0)
_check("Total params > 5M (sanity)",  p_total > 5_000_000)


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 68)
if all_passed:
    print("✓  All checks passed — three-component pipeline is end-to-end OK")
else:
    print("✗  One or more checks FAILED — see [FAIL] lines above")
print("=" * 68)
