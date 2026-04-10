"""
Single-Batch Overfit Check — HB-Mamba v2.0
===========================================

Trains on exactly one batch for N_STEPS steps.
A correctly implemented model should memorise one batch — if it can't,
there is a bug in the forward pass, loss computation, or gradient flow.

Expected behaviour
------------------
    L_mask    : 0.20 → near 0   (model memorises masked ping values)
    L_next    : 0.20 → near 0   (model memorises next-ping values)
    L_align   : 1.00 → < 0.10   (macro/micro representations align)
    L_total   : ~0.80 → < 0.05  (all supervised signals collapse)
    L_contrast: 0.00             (no positive MMSI pairs in one batch — expected)

Pass criteria
-------------
    L_total drops by > 90% of its initial value within N_STEPS steps.
    No NaN or Inf at any step.
    No gradient explosion (norm stays < 100 after warmup).
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hb_mamba_dataset import build_dataloaders
from model.hb_mamba   import HBMamba

# ── Config ────────────────────────────────────────────────────────────────────
N_STEPS    = 300      # optimiser steps
LR         = 3e-3     # higher than training LR to overfit faster
WARMUP     = 20       # linear warmup steps
LOG_EVERY  = 20       # print interval
PASS_DROP  = 0.90     # L_total must drop by at least 90%

RAW_ROOT   = ROOT / "data_genration_and_raw_data" / "raw_data"
NS_PATH    = str(RAW_ROOT / "preprocessing" / "norm_stats" / "norm_stats.json")
IDX_DIR    = str(RAW_ROOT / "dataset_index")

# ─────────────────────────────────────────────────────────────────────────────

if not torch.cuda.is_available():
    print("No CUDA device — requires GPU (Mamba2 is CUDA-only).")
    sys.exit(1)

device = torch.device("cuda")

print("=" * 65)
print("Single-Batch Overfit Check — HB-Mamba v2.0")
print(f"  steps={N_STEPS}  lr={LR}  warmup={WARMUP}")
print("=" * 65)

# ── One fixed batch ───────────────────────────────────────────────────────────
loaders = build_dataloaders(
    dataset_index_dir = IDX_DIR,
    norm_stats_path   = NS_PATH,
    batch_size        = 8,
    num_workers       = 0,
    pin_memory        = False,
)

raw_batch = next(iter(loaders["train"]))

macro_features = raw_batch["macro_features"].to(device)
macro_lat_idx  = raw_batch["macro_lat_idx"].to(device)
macro_lon_idx  = raw_batch["macro_lon_idx"].to(device)
micro_tokens   = raw_batch["micro_tokens"].to(device)
mask           = raw_batch["mask"].to(device)
padding_mask   = raw_batch["padding_mask"].to(device)
mmsi           = raw_batch["mmsi"]

B       = macro_features.shape[0]
n_day   = raw_batch["n_day"].tolist()
n_mask  = int(mask.sum())
n_real  = int((~padding_mask).sum())

print(f"\nBatch info:")
print(f"  B={B}  n_day={n_day}  masked_pings={n_mask}  real_pings={n_real}")

# ── Model + optimiser ─────────────────────────────────────────────────────────
model = HBMamba.from_norm_stats(NS_PATH).to(device).train()

optimizer = torch.optim.AdamW(
    model.param_groups(lr=LR, weight_decay=0.01)
)

def _lr_lambda(step: int) -> float:
    if step < WARMUP:
        return (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, N_STEPS - WARMUP)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'step':>6}  {'L_total':>8}  {'L_mask':>8}  {'L_next':>8}"
      f"  {'L_align':>8}  {'grad_norm':>9}  {'lr':>10}")
print("-" * 72)

loss_at_step0  = None
loss_at_stepN  = None
exploded       = False
any_nan        = False

for step in range(N_STEPS):
    optimizer.zero_grad()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model(
            macro_features, macro_lat_idx, macro_lon_idx,
            micro_tokens, mask, padding_mask, mmsi,
        )

    L = losses["L_total"]

    # NaN / Inf guard
    if not torch.isfinite(L):
        print(f"\n  ABORT at step {step}: L_total = {L.item()} (NaN or Inf)")
        any_nan = True
        break

    L.backward()

    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item()

    if grad_norm > 100:
        print(f"\n  WARNING step {step}: grad_norm={grad_norm:.1f} — possible explosion")
        exploded = True

    optimizer.step()
    scheduler.step()

    l_total   = losses["L_total"].item()
    l_mask    = losses["L_mask"].item()
    l_next    = losses["L_next"].item()
    l_align   = losses["L_align"].item()
    lr_now    = scheduler.get_last_lr()[0]

    if step == 0:
        loss_at_step0 = l_total

    loss_at_stepN = l_total

    if step % LOG_EVERY == 0 or step == N_STEPS - 1:
        print(f"  {step:4d}  {l_total:8.4f}  {l_mask:8.4f}  {l_next:8.4f}"
              f"  {l_align:8.4f}  {grad_norm:9.4f}  {lr_now:10.2e}")

# ── Final assessment ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Results")
print("=" * 65)

if any_nan:
    print("  [FAIL]  NaN/Inf detected — training diverged")
    sys.exit(1)

drop_frac = (loss_at_step0 - loss_at_stepN) / (loss_at_step0 + 1e-9)

print(f"  L_total step   0 : {loss_at_step0:.4f}")
print(f"  L_total step {N_STEPS-1:3d} : {loss_at_stepN:.4f}")
print(f"  Drop            : {drop_frac*100:.1f}%  (pass threshold: {PASS_DROP*100:.0f}%)")

passed = (drop_frac >= PASS_DROP) and (not exploded) and (not any_nan)

if passed:
    print(f"\n  [PASS]  Model overfits one batch — architecture and gradients OK")
else:
    if drop_frac < PASS_DROP:
        print(f"\n  [FAIL]  Loss only dropped {drop_frac*100:.1f}% — expected >= {PASS_DROP*100:.0f}%")
    if exploded:
        print(f"  [WARN]  Gradient explosion detected at some step")

print("=" * 65)
sys.exit(0 if passed else 1)
