# HB-Mamba v2.1 — Training & Architecture Fix Specification
### For Claude Opus — Full Implementation Prompt

---

## 0. Who You Are and What This Project Is

You are implementing fixes to **HB-Mamba v2.0**, a self-supervised vessel trajectory learning model
for the Gulf of Mexico AIS dataset. The model learns to embed vessel trajectories into dense 256-dim
vectors usable for anomaly detection, vessel re-identification, and SAR-AIS correlation.

**Working directory:** `/home/hpc25/AIS/SAR_AIS_analysis`
**Python environment:** conda env `hb_mamba`
**Run all commands as:** `conda run -n hb_mamba python <script>`
**GPU setup:** 2× NVIDIA RTX 4500 Ada (24 GB each). Training uses GPU 0 (or both via gloo DDP).
Use `CUDA_VISIBLE_DEVICES` is NOT set — always address GPUs by index directly.
Before any mamba_ssm import, call `torch.cuda.set_device(device)`.

---

## 1. Current File Structure (DO NOT RENAME EXISTING FILES)

```
SAR_AIS_analysis/
├── train.py                          ← training entry point (DO NOT MODIFY)
├── inference.py                      ← embedding extraction (DO NOT MODIFY)
├── hb_mamba_dataset.py               ← Dataset/DataLoader — MODIFY THIS
├── model/
│   ├── hb_mamba.py                   ← top-level model — MODIFY THIS
│   ├── macro_encoder.py              ← DO NOT MODIFY
│   ├── cross_scale_injection.py      ← DO NOT MODIFY
│   ├── micro_decoder.py              ← DO NOT MODIFY
│   └── loss_heads.py                 ← MODIFY THIS
├── training/
│   └── trainer.py                    ← DO NOT MODIFY
├── checkpoints/
│   ├── best.pt                       ← epoch 21, val_total=0.0023 (current best)
│   └── epoch_024.pt                  ← latest periodic checkpoint
└── visualize_gap_prediction.py       ← MODIFY THIS (update for new interface)
```

**New files you must create:**
```
SAR_AIS_analysis/
├── train_v2_1.py                     ← NEW: v2.1 training entry point
├── checkpoints_v2_1/                 ← NEW: separate checkpoint dir for v2.1
└── figs/gap_predictions_v2_1.png     ← NEW: produced by updated visualizer
```

Do NOT overwrite `train.py`, `checkpoints/best.pt`, or any existing files.
The v2.0 training run must remain reproducible.

---

## 2. Background: What v2.0 Does (Read This Carefully)

### Data pipeline
- **Macro features** `[B, N_cells=1044, 10]` — daily grid statistics (vessel counts, speed stats per 0.5° cell)
- **Micro tokens** `[B, N_day, 11]` — per-ping AIS trajectory features, variable length
- **Mask** `[B, N_day]` bool — True = gap positions (model must predict these)
- **Padding mask** `[B, N_day]` bool — True = padded zeros at end of shorter sequences

### 11 micro token features (in order, indices 0–10):
```
0  lat_norm       normalised latitude   → (lat - 17.4068) / 14.058   clipped [0,1]
1  lon_norm       normalised longitude  → (lon - (-98.0539)) / 17.621 clipped [0,1]
2  sog_norm       speed over ground (normalised)
3  cog_sin        sin(course over ground)
4  cog_cos        cos(course over ground)
5  hdg_sin        sin(heading)
6  hdg_cos        cos(heading)
7  delta_t_norm   normalised time to next ping (forward-looking)
8  length_norm    vessel length  ← STATIC (same for all pings of same vessel)
9  draft_norm     vessel draft   ← STATIC
10 type_norm      vessel type    ← STATIC
```
Features 0–7 are dynamic (change per ping). Features 8–10 are static (constant per vessel-day).

### Forward pass (v2.0)
```
micro_tokens [B,N_day,11]  →  CrossScaleInjection  →  x_conditioned [B,N_day,256]
macro_output [B,1044,256]  ↗                                ↓
                                                      MicroEncoder (4× Bidirectional Mamba2)
                                                            ↓
                                              h_fwd [B,N_day,256]  h_bwd [B,N_day,256]
                                              H = mean(h_fwd + h_bwd) [B,256]
                                                            ↓
                                                       LossHeads
                                            L_mask  L_next  L_contrast  L_align
```

### Current combined loss:
```
L = 1.0·L_mask + 0.8·L_next + 0.5·L_contrast + 0.5·L_align
```

---

## 3. The 7 Problems Found in v2.0

### Problem 1 — CRITICAL: Model sees gap tokens (training objective is wrong)
**File:** `hb_mamba_dataset.py` and `model/hb_mamba.py`

`hb_mamba_dataset.py` returns `micro_tokens` = the FULL unmasked trajectory.
`model/hb_mamba.py` `forward()` passes this full tensor into CrossScaleInjection and MicroEncoder.
The `mask` is only used inside `LossHeads` to select which positions contribute to the loss.

**Consequence:** The model learns to *reconstruct* features it can already see, not *predict* from incomplete input. At inference time, when you zero the gap tokens, errors explode from ~100 km to ~1400 km because the model has never seen zeroed inputs.

**Proof from experiment:** Running inference with zeroed gap tokens gave 6–18× worse results than passing full tokens. The model defaults to predicting lat≈31.4°N, lon≈-80.4° (the northeast corner of the Gulf bounding box) for all vessels — a clear distribution shift collapse.

### Problem 2 — SIGNIFICANT: Bidirectional intermediate layers contaminate h_fwd
**File:** `model/micro_decoder.py` lines 128–139

```python
# Current code (PROBLEMATIC):
for i in range(self.n_layers - 1):
    y_fwd = self.mamba_fwd[i](x)
    y_bwd = self.mamba_bwd[i](x.flip(dims=[1])).flip(dims=[1])
    x = y_fwd + y_bwd   # ← h_fwd and h_bwd are merged here
```

After the first intermediate layer, `x` at every position contains both forward AND backward
information. By the time we reach the final layer (where h_fwd and h_bwd are kept separate),
even `h_fwd` already "knows" the full sequence in both directions.

For gap prediction: even with zeroed gap tokens, the forward stream at gap position t has already
been contaminated by backward information from later positions in intermediate layers.

### Problem 3 — CRITICAL: No output clamping on reconstruction head
**File:** `model/loss_heads.py` — `ReconstructionHead.forward()`

The MLP output is raw logits. Predicted `lat_norm` or `lon_norm` can be > 1.0 or < 0.0,
mapping to coordinates outside the Gulf of Mexico. No clamp or sigmoid is applied.

### Problem 4 — SIGNIFICANT: h_fwd trained for two conflicting objectives
**File:** `model/loss_heads.py`

`L_mask` (reconstruction) trains `recon_head(cat[h_fwd[t], h_bwd[t]])` to predict features at position **t**.
`L_next` (next-ping) trains `next_head(h_fwd[t])` to predict features at position **t+1**.

Both losses backpropagate into `h_fwd[t]` with conflicting gradients:
- L_mask wants h_fwd[t] to encode the current state at t
- L_next wants h_fwd[t] to encode a predictor of state at t+1

This is why gap predictions look like trajectory extrapolation — h_fwd has a forward-prediction bias baked in.

### Problem 5 — SIGNIFICANT: Static vessel features zeroed at gap boundary
**File:** `hb_mamba_dataset.py` (affects inference when gap tokens are zeroed)

Features 8–10 (vessel length, draft, type) are identical for all pings of the same vessel.
When we zero gap positions, these static features become 0 — the model loses vessel identity.
The fix is to preserve features 8–10 while only zeroing features 0–7 at gap positions.

### Problem 6 — MODERATE: delta_t is incorrect at gap boundaries
**File:** `hb_mamba_dataset.py` / data preprocessing

Feature 7 (`delta_t_norm`) at the last visible ping before the gap encodes the time to the
*next actual ping* — not the time to the start of the gap. The model has no signal that a
large temporal gap is about to start.

The fix: when the gap mask is applied, set `delta_t` at the last pre-gap ping to
`(gap_start_time - last_visible_ping_time) / normalisation_constant`. Since exact timestamps
are not stored in the tensor (they were normalised away), a practical approximation is:
`delta_t_at_last_visible_ping = gap_length_pings × mean_delta_t_of_visible_pings`.

### Problem 7 — MODERATE: H includes gap positions in mean pool
**File:** `model/micro_decoder.py` line 158

```python
H = (h_fwd + h_bwd).mean(dim=1)   # includes ALL positions, even gap positions
```

When gap tokens are properly zeroed, the hidden states at gap positions will reflect zeros,
not real vessel state. Including them in H dilutes the trajectory embedding.

Fix: compute H as the mean over VISIBLE (non-gap, non-padded) positions only.
This requires passing a combined visibility mask into `MicroEncoder.forward()`.

---

## 4. What v2.1 Must Do — Complete Fix Specification

### 4.1 Dataset change: `hb_mamba_dataset.py`

**Change `__getitem__` to return two versions of micro_tokens:**

```python
return {
    # ... existing keys unchanged ...
    "micro_tokens":         micro_tokens,           # FULL original — used as loss TARGET
    "micro_tokens_masked":  micro_tokens_masked,    # Gap zeroed   — used as model INPUT
    # ... rest of keys ...
}
```

**How to build `micro_tokens_masked`:**
```python
micro_tokens_masked = micro_tokens.clone()

# Step 1: Zero dynamic features (0-7) at gap positions
micro_tokens_masked[mask, :8] = 0.0

# Step 2: Preserve static features (8-10) at gap positions
# Find the last visible ping before the gap start
gap_positions   = mask.nonzero(as_tuple=True)[0]
gap_start       = int(gap_positions[0])
visible_before  = (~mask[:gap_start]).nonzero(as_tuple=True)[0]
if len(visible_before) > 0:
    ref_idx = int(visible_before[-1])
else:
    # No visible ping before gap — use first visible ping anywhere
    ref_idx = int((~mask).nonzero(as_tuple=True)[0][0])
micro_tokens_masked[mask, 8:] = micro_tokens[ref_idx, 8:]

# Step 3: Fix delta_t at the last visible ping before the gap
# Set it to the mean inter-ping interval × gap_length so the model
# knows a long temporal gap is starting
if len(visible_before) > 0:
    last_vis = int(visible_before[-1])
    gap_len  = int(gap_positions[-1]) - int(gap_positions[0]) + 1
    # mean delta_t of all other visible pings
    vis_mask_bool = ~mask
    vis_delta_t   = micro_tokens[vis_mask_bool, 7]
    mean_delta_t  = vis_delta_t.mean().item() if len(vis_delta_t) > 0 else 0.0
    micro_tokens_masked[last_vis, 7] = min(mean_delta_t * gap_len, 1.0)
```

**Also update `hb_mamba_collate_fn`** to pad and stack `micro_tokens_masked` exactly the
same way as `micro_tokens`. The batch dict must contain both keys.

### 4.2 Model forward change: `model/hb_mamba.py`

**Change `forward()` signature to accept both tensors:**

```python
def forward(
    self,
    macro_features      : Tensor,        # [B, N_cells, d_macro]
    macro_lat_idx       : Tensor,        # [B, N_cells]
    macro_lon_idx       : Tensor,        # [B, N_cells]
    micro_tokens_masked : Tensor,        # [B, N_day, d_micro]  ← ENCODER INPUT (gap zeroed)
    micro_tokens        : Tensor,        # [B, N_day, d_micro]  ← LOSS TARGET  (full original)
    mask                : Tensor,        # [B, N_day] bool
    padding_mask        : Tensor,        # [B, N_day] bool
    mmsi                : List[int],
) -> Dict[str, Tensor]:
```

- Pass `micro_tokens_masked` to `CrossScaleInjection` and `MicroEncoder`
- Pass `micro_tokens` (original) to `LossHeads` as the reconstruction target
- Pass `mask` and `padding_mask` also into `MicroEncoder` for Fix 7 (H from visible only)

**Change `predict()` signature** to accept an optional `mask` parameter:
```python
def predict(
    self,
    macro_features      : Tensor,
    macro_lat_idx       : Tensor,
    macro_lon_idx       : Tensor,
    micro_tokens_masked : Tensor,    # caller is responsible for zeroing gap positions
    padding_mask        : Optional[Tensor] = None,
    mask                : Optional[Tensor] = None,   # if provided, use for H computation
) -> Dict[str, Tensor]:
```

### 4.3 MicroEncoder change: `model/micro_decoder.py`

**Fix 2 — Separate intermediate streams (do not mix fwd+bwd until final layer):**

```python
def forward(self, x_conditioned: Tensor, visibility_mask: Optional[Tensor] = None):
    """
    visibility_mask : [B, N_day] bool — True = position is VISIBLE (not gap, not padded)
                      If provided, H is computed only over visible positions (Fix 7).
                      If None, H is mean over all positions (v2.0 behaviour).
    """
    x_fwd = x_conditioned   # [B, N_day, d_model]
    x_bwd = x_conditioned   # separate streams — DO NOT MIX in intermediate layers

    # Intermediate layers: run fwd and bwd INDEPENDENTLY
    for i in range(self.n_layers - 1):
        x_fwd = self.mamba_fwd[i](x_fwd)
        x_bwd = self.mamba_bwd[i](x_bwd.flip(dims=[1])).flip(dims=[1])
        # DO NOT sum here — keep streams separate

    # Final layer
    last  = self.n_layers - 1
    h_fwd = self.mamba_fwd[last](x_fwd)
    h_bwd = self.mamba_bwd[last](x_bwd.flip(dims=[1])).flip(dims=[1])

    # Fix 7: H from visible positions only
    if visibility_mask is not None:
        # visibility_mask: True = visible, [B, N_day]
        vis = visibility_mask.unsqueeze(-1).float()          # [B, N_day, 1]
        combined = (h_fwd + h_bwd) * vis                    # zero out gap/pad positions
        n_vis = vis.sum(dim=1).clamp(min=1)                  # [B, 1]
        H = combined.sum(dim=1) / n_vis                      # [B, d_model]
    else:
        H = (h_fwd + h_bwd).mean(dim=1)                     # v2.0 fallback

    return h_fwd, h_bwd, H
```

**Important:** The intermediate layers now run independently. The final layer's h_fwd still
receives the full context from intermediate fwd layers (which processed zeroed gap tokens
cleanly without backward contamination). This allows the forward stream to model
"trajectory up to this point" and the backward stream to model "trajectory f  Summary table

  ┌─────┬──────────────────────────────────────────────────────────────────────┬──────────────────────┬────────────────────────────────────────────────┐
  │  #  │                                Issue                                 │       Affects        │                   Fix needed                   │
  ├─────┼──────────────────────────────────────────────────────────────────────┼──────────────────────┼────────────────────────────────────────────────┤
  │ 1   │ CSI uses gap lat/lon to build Q → picks right macro cells            │ Gap prediction       │ Zero gap in micro_tokens before CSI            │
  ├─────┼──────────────────────────────────────────────────────────────────────┼──────────────────────┼────────────────────────────────────────────────┤
  │ 2   │ Intermediate bidirectional mixing contaminates h_fwd                 │ Gap prediction       │ Architectural — hard to fix without retraining │
  ├─────┼──────────────────────────────────────────────────────────────────────┼──────────────────────┼────────────────────────────────────────────────┤
  │ 3   │ No clamp on reconstruction output                                    │ Visualization errors │ pred.clamp(0, 1)                               │
  ├─────┼──────────────────────────────────────────────────────────────────────┼──────────────────────┼────────────────────────────────────────────────┤
  │ 4   │ h_fwd trained to predict t+1 (L_next) but used to predict t (L_mask) │ Both losses conflict │ Reduce w_next or decouple heads                │
  ├─────┼──────────────────────────────────────────────────────────────────────┼──────────────────────┼────────────────────────────────────────────────┤
  │ 5   │ Static features (8-10) zeroed at gap                                 │ Inference            │ Copy from visible pings                        │
  ├─────┼──────────────────────────────────────────────────────────────────────┼──────────────────────┼────────────────────────────────────────────────┤
  │ 6   │ delta_t wrong at gap boundary                                        │ Temporal signal      │ Fix in preprocessing or mask handling          │
  ├─────┼──────────────────────────────────────────────────────────────────────┼──────────────────────┼────────────────────────────────────────────────┤
  │ 7   │ H averages over gap positions                                        │ Embedding quality    │ Exclude gap positions from H mean pool         │
  └─────┴──────────────────────────────────────────────────────────────────────┴──────────────────────┴────────────────────────────────────────────────┘

  ---
  Issues 1, 3, 5, 7 can be fixed now without retraining. Issues 2 and 4 require architectural changes and a retrain. Issue 6 requires a preprocessing change. Want me to fix what's fixable now (1, 3, 5,
  7) and outline the retrain changes needed for 2 and 4?

rom this point
onward" without cross-contamination in earlier layers.

### 4.4 LossHeads change: `model/loss_heads.py`

**Fix 3 — Add output clamping in `ReconstructionHead`:**

```python
class ReconstructionHead(nn.Module):
    def forward(self, h_fwd: Tensor, h_bwd: Tensor) -> Tensor:
        out = self.mlp(torch.cat([h_fwd, h_bwd], dim=-1))
        return out.clamp(0.0, 1.0)    # ← ADD THIS LINE
```

**Fix 4 — Decouple h_fwd from the next-ping objective** by adding a small adapter layer
so L_next does not force h_fwd to encode t+1 information:

```python
class NextPingHead(nn.Module):
    def __init__(self, d_model: int = 256, n_features: int = 8) -> None:
        super().__init__()
        self.adapter = nn.Linear(d_model, d_model)    # ← ADD: decoupling adapter
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_features),
        )

    def forward(self, h_fwd: Tensor) -> Tensor:
        return self.mlp(self.adapter(h_fwd))           # adapter absorbs t+1 bias
```

**Also reduce w_next** in `LossHeads.__init__` default from `0.8` → `0.3`:
```python
w_next : float = 0.3,    # was 0.8 — reduced because it conflicted with L_mask
```
Keep backward-compatible: `w_next` is still a constructor argument so v2.0 checkpoints
can be loaded with `w_next=0.8` if needed.

### 4.5 New training entry point: `train_v2_1.py`

Create `train_v2_1.py` as a copy of `train.py` with these changes:

1. **Default checkpoint dir:** `--checkpoint_dir checkpoints_v2_1`
2. **Default w_next:** pass `w_next=0.3` when constructing `TrainerConfig` / model
3. **Import HBMamba** with same deferred-import pattern as `train.py`
4. **Model construction:** `HBMamba.from_norm_stats(NS_PATH, w_next=0.3)`
5. **DO NOT resume from old checkpoints** by default — start fresh (v2.1 has different
   forward signature, old checkpoints are incompatible)

Header docstring must say:
```
train_v2_1.py — HB-Mamba v2.1  Training Entry Point
Changes from v2.0:
  - micro_tokens_masked passed to encoder (gap zeroed, static features preserved)
  - micro_tokens (full) passed as loss target only
  - delta_t corrected at gap boundary
  - Intermediate Mamba streams kept separate (no bidirectional mixing until final layer)
  - ReconstructionHead output clamped to [0, 1]
  - NextPingHead has decoupling adapter, w_next reduced 0.8 → 0.3
  - H computed from visible positions only
```

### 4.6 Update `visualize_gap_prediction.py`

The `run_fixed()` function currently performs zeroing manually at inference time.
After the v2.1 changes, the model's `predict()` method expects the CALLER to pass
`micro_tokens_masked` (already zeroed). Update `run_fixed()` to use the dataset's
`micro_tokens_masked` key directly (provided by the updated `__getitem__`).

```python
def run_fixed_v21(model, item, device):
    """
    Uses micro_tokens_masked from the dataset (properly prepared by __getitem__).
    No manual zeroing needed here — the dataset handles it.
    """
    micro_tokens_masked = item["micro_tokens_masked"].unsqueeze(0).to(device)
    # ... rest identical to run_fixed() but using micro_tokens_masked ...
```

---

## 5. Backward Compatibility Requirements

| Requirement | Detail |
|---|---|
| v2.0 checkpoints loadable | `HBMamba.forward()` new signature must have default `micro_tokens_masked=None`; when None, fall back to using `micro_tokens` for encoder input (v2.0 behaviour) |
| `train.py` unchanged | Original training script must still run correctly |
| `inference.py` unchanged | Embedding extraction still works against v2.0 checkpoints |
| `checkpoints/` untouched | All existing checkpoints stay as-is |

**How to implement the backward-compatible fallback in `forward()`:**
```python
def forward(self, macro_features, macro_lat_idx, macro_lon_idx,
            micro_tokens,                          # always required
            mask, padding_mask, mmsi,
            micro_tokens_masked=None):             # NEW optional arg
    encoder_input = micro_tokens_masked if micro_tokens_masked is not None else micro_tokens
    # use encoder_input for CSI + MicroEncoder
    # use micro_tokens for LossHeads target
```

---

## 6. Verification Steps After Implementation

After all changes are implemented, run these checks in order:

### Step 1 — Dataset produces both keys
```python
conda run -n hb_mamba python -c "
from hb_mamba_dataset import build_dataloaders, GapMaskConfig
import sys
RAW = 'data_genration_and_raw_data/raw_data'
loaders = build_dataloaders(RAW+'/dataset_index', RAW+'/preprocessing/norm_stats/norm_stats.json',
                            batch_size=4, num_workers=0, pin_memory=False)
batch = next(iter(loaders['train']))
assert 'micro_tokens_masked' in batch, 'MISSING micro_tokens_masked'
assert 'micro_tokens' in batch, 'MISSING micro_tokens'
# Verify gap positions are zeroed in masked version
mask = batch['mask']   # [B, N_day]
mt   = batch['micro_tokens']
mtm  = batch['micro_tokens_masked']
for b in range(4):
    gap_pos = mask[b].nonzero(as_tuple=True)[0]
    if len(gap_pos) > 0:
        assert (mtm[b][gap_pos, :8] == 0).all(), 'Gap features not zeroed'
        assert (mtm[b][gap_pos, 8:] != 0).any(), 'Static features should not be zeroed'
print('Dataset OK — both keys present, gap zeroed correctly')
"
```

### Step 2 — Model forward pass runs with new signature
```python
conda run -n hb_mamba python -c "
import torch, sys
sys.path.insert(0, '.')
torch.cuda.set_device(0)
from model.hb_mamba import HBMamba
model = HBMamba.from_norm_stats(
    'data_genration_and_raw_data/raw_data/preprocessing/norm_stats/norm_stats.json',
    w_next=0.3
).cuda()
# test with micro_tokens_masked
B, N, D = 2, 80, 11
mf  = torch.randn(B, 1044, 10).cuda()
la  = torch.randint(0, 29, (B, 1044)).cuda()
lo  = torch.randint(0, 36, (B, 1044)).cuda()
mt  = torch.randn(B, N, D).cuda()
mtm = mt.clone(); mtm[:, 20:40, :8] = 0  # zero a gap
mask = torch.zeros(B, N, dtype=torch.bool).cuda(); mask[:, 20:40] = True
pm   = torch.zeros(B, N, dtype=torch.bool).cuda()
losses = model(mf, la, lo, mt, mask, pm, [123]*B, micro_tokens_masked=mtm)
assert 'L_total' in losses
losses['L_total'].backward()
print('v2.1 forward pass OK, loss:', {k: f'{v.item():.4f}' for k,v in losses.items()})
"
```

### Step 3 — Verify h_fwd is no longer contaminated (streams stay separate)
```python
# After implementing separate intermediate streams, verify that h_fwd at a
# gap position (all-zero input) differs between separate-stream and mixed-stream
# by checking that the reconstruction head output clamps correctly.
conda run -n hb_mamba python -c "
import torch, sys
sys.path.insert(0, '.')
torch.cuda.set_device(0)
from model.hb_mamba import HBMamba
model = HBMamba.from_norm_stats(
    'data_genration_and_raw_data/raw_data/preprocessing/norm_stats/norm_stats.json',
    w_next=0.3
).cuda().eval()
B, N = 1, 50
mf = torch.randn(B,1044,10).cuda()
la = torch.randint(0,29,(B,1044)).cuda()
lo = torch.randint(0,36,(B,1044)).cuda()
mt = torch.rand(B,N,11).cuda()    # all real features [0,1]
mtm = mt.clone(); mtm[:,20:35,:8] = 0   # zero the gap
with torch.no_grad():
    out = model.predict(mf,la,lo,mtm)
    pred = model.loss_heads.recon_head(out['h_fwd'], out['h_bwd'])
assert pred.min() >= 0.0, f'Clamp failed: min={pred.min()}'
assert pred.max() <= 1.0, f'Clamp failed: max={pred.max()}'
print('Clamp OK: pred range [{pred.min():.4f}, {pred.max():.4f}]')
"
```

### Step 4 — Sanity check gap prediction vs v2.0
Run `python visualize_gap_prediction.py` and check that:
- The "fixed" (right column) errors are LOWER than original, not higher
- Predictions do not cluster at the Gulf northeast corner
- At least some samples show mean error < 100 km

### Step 5 — Start v2.1 training
```bash
# Single GPU (verify one epoch completes before launching 2-GPU)
conda run -n hb_mamba python train_v2_1.py --epochs 1 --batch_size 64

# If OK, launch 2-GPU training
torchrun --nproc_per_node=2 train_v2_1.py --epochs 40 --batch_size 64
```

---

## 7. What NOT to Do

- Do NOT modify `train.py` — it must stay as v2.0
- Do NOT delete or overwrite `checkpoints/best.pt`
- Do NOT change `macro_encoder.py` or `cross_scale_injection.py`
- Do NOT use `CUDA_VISIBLE_DEVICES` remapping — use `--device cuda:0` or `cuda:1`
- Do NOT use NCCL backend for DDP — use gloo (system has driver/NCCL version mismatch)
- Do NOT install new packages — only `matplotlib`, `numpy`, `pandas`, `torch`, `mamba_ssm` are available
- Do NOT add BatchNorm anywhere — model uses LayerNorm/RMSNorm throughout
- Do NOT cache or save intermediate tensors across batches — memory is tight at batch_size=64

---

## 8. Key Numbers and Constants

| Constant | Value | Source |
|---|---|---|
| Gulf lat min/max | 17.4068 / 31.4648 | norm_stats.json |
| Gulf lon min/max | -98.0539 / -80.4330 | norm_stats.json |
| N_LAT_STEPS | 29 | norm_stats.json |
| N_LON_STEPS | 36 | norm_stats.json |
| N_TOTAL_CELLS | 1044 | norm_stats.json |
| BIN_SIZE | 0.5° | norm_stats.json |
| d_model | 256 | model config |
| n_layers | 4 | model config |
| K (top-k macro cells) | 32 | model config |
| d_proj | 128 | model config |
| n_features (predicted) | 8 | features 0–7 |
| tau (NT-Xent temperature) | 0.07 | model config |
| norm_stats path | `data_genration_and_raw_data/raw_data/preprocessing/norm_stats/norm_stats.json` | |
| dataset index dir | `data_genration_and_raw_data/raw_data/dataset_index` | |

---

## 9. Summary of All Changes by File

| File | Change type | What changes |
|---|---|---|
| `hb_mamba_dataset.py` | Modify | `__getitem__` returns `micro_tokens_masked`; `collate_fn` pads it; delta_t at gap boundary corrected |
| `model/hb_mamba.py` | Modify | `forward()` accepts `micro_tokens_masked` kwarg; passes it to encoder; passes `mask` to MicroEncoder |
| `model/micro_decoder.py` | Modify | Intermediate layers keep fwd/bwd separate; `forward()` accepts `visibility_mask` for H computation |
| `model/loss_heads.py` | Modify | `ReconstructionHead` output clamped to [0,1]; `NextPingHead` gets decoupling adapter; default `w_next=0.3` |
| `train_v2_1.py` | Create new | Copy of `train.py` with `w_next=0.3` and `checkpoint_dir=checkpoints_v2_1` |
| `visualize_gap_prediction.py` | Modify | `run_fixed()` uses `micro_tokens_masked` from dataset instead of manual zeroing |

---

## 10. Expected Outcome After v2.1 Training

After ~21 epochs of v2.1 training (comparable to the v2.0 best checkpoint), expect:

- **Gap prediction errors:** Should be meaningfully lower than v2.0's 74–168 km on the same samples, because the model is now trained to predict from genuinely incomplete input
- **Cosine similarity (intra vs inter vessel):** Should remain above 0.9 intra / 0.3 inter (the embedding task is unchanged)
- **H quality:** Slightly cleaner because gap positions no longer pollute the mean pool
- **Convergence:** May be slightly slower initially (harder task), but should reach similar val_total ≈ 0.002–0.003 range

The key test: run `visualize_gap_prediction.py` on v2.1 checkpoint and confirm the right-column (fixed) errors are LOWER than the left-column (original), which is the opposite of what v2.0 showed.

---

*Document created: 2026-04-12*
*Based on v2.0 training run: 24 epochs complete, best at epoch 21 (val_total=0.0023)*
*Current training status: resumed at epoch 25, running on 2× RTX 4500 Ada via gloo DDP*
