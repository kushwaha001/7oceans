# HB-Mamba v2.3 — How It Works

End-to-end reference for what the model does, how we train it, and how we use it at inference time to fill AIS gaps.

---

## 1. The Problem

AIS (Automatic Identification System) transponders on vessels broadcast position pings roughly every few seconds. In practice those streams are full of **gaps** — equipment outages, coverage holes, intentional transponder shutoffs. The task:

> Given a vessel's partial 24-hour AIS track (some pings visible, some missing), predict the missing lat/lon (and motion features) at every gap position.

Baseline: **linear interpolation** between the last visible ping before the gap and the first after. This is a surprisingly strong baseline — ships move in smooth curves, and a straight chord is already a decent approximation. The model must learn to beat this by learning *curvature*.

Evaluation: **haversine distance in km** between predicted and actual lat/lon at each gap ping. Lower is better. Linear's mean is ~22 km on our Gulf-of-Mexico test set.

---

## 2. Data Pipeline

### 2.1 Inputs

Each sample is one (vessel × 24-hour window):

| Field | Shape | Description |
|---|---|---|
| `macro_features` | `[N_cells, 10]` | Grid-cell-level traffic summary features |
| `macro_lat_idx` | `[N_cells]` | Lat-bin index of each non-empty cell |
| `macro_lon_idx` | `[N_cells]` | Lon-bin index of each non-empty cell |
| `micro_tokens` | `[N_day, 11]` | Per-ping features (full, unmasked) |
| `mask` | `[N_day]` bool | `True` at positions we want the model to predict |
| `padding_mask` | `[N_day]` bool | `True` at padded positions |
| `mmsi` | int | Vessel identity |

`micro_tokens` channels: `[lat, lon, sog, cog_sin, cog_cos, hdg_sin, hdg_cos, delta_t, vessel_type_embed_1, vessel_type_embed_2, vessel_type_embed_3]`.

Only the first 8 are predicted by the heads — the last 3 are static identity features.

### 2.2 Normalisation

Lat and lon are min-max normalised to `[0, 1]` using the Gulf-of-Mexico bounding box stored in `norm_stats.json`:

```
LAT_MIN=17.4068  LAT_MAX=31.4648
LON_MIN=-98.0539 LON_MAX=-80.4330
```

Angles (`cog`, `hdg`) are encoded as `(sin, cos)` pairs so the model can regress them without wrap-around discontinuity at 0/360°.

### 2.3 Gap masking (dataset side)

For every sample the `HBMambaDataset`:

1. Picks a random gap fraction in `[0.20, 0.40]` of the window, and a gap type (interpolation — gap lies in the interior; or extrapolation — gap lies at one end).
2. Produces a `mask` indicating gap positions.
3. Creates `micro_tokens_masked` — a copy with the **dynamic** features (0–6: lat, lon, sog, cog, hdg) zeroed at masked positions, but:
   - `delta_t` is preserved so every gap ping still has its temporal identity (prevents mid-gap collapse).
   - Static features (8–10: vessel type) are copied from the nearest visible ping.

**Validation masks are deterministic** (seeded by sample index) so every epoch evaluates on exactly the same gaps → losses are comparable epoch-to-epoch.

---

## 3. Architecture — 4 Components

```
 batch ──► MacroEncoder ──► macro_output          [B, N_cells, 256]
                             │
                             ▼
 micro_tokens_masked ──► CrossScaleInjection ──► x_conditioned   [B, N_day, 256]
                                            ├── attn_weights    [B, N_day, K=32]
                                            └── topk_idx        [B, N_day, K=32]
                             │
                             ▼
                         MicroEncoder (bi-Mamba2) ──► h_fwd, h_bwd, H
                             │
                             ▼
                         LossHeads (4 heads) ──► L_mask, L_next, L_contrast, L_align
```

### 3.1 MacroEncoder

A Mamba2 stack over a **spatial grid**. Builds global context: "what are vessels generally doing in each grid cell today?".

- Input: `[B, N_cells, 10]` macro features + their 2-D grid indices.
- Output: `[B, N_cells, 256]` per-cell representations.
- The grid is sparse (only non-empty cells are kept), so `N_cells` varies per batch.

### 3.2 CrossScaleInjection (CSI)

Bridges macro (cell-level) and micro (ping-level) scales. For every ping:

1. Project the micro token into Q-space: `Q = W_Q(micro_token)`.
2. Compute attention scores against all macro cells: `scores = Q · macro_output^T`.
3. Pick top-K=32 cells per ping, softmax-normalise the scores.
4. Aggregate: `ping_ctx = Σₖ attn_weightsₖ · macro_cellₖ`.
5. Combine: `x_conditioned = micro_token + ping_ctx + PE(pos)`.
6. At masked positions, add a learned `[MASK]` embedding so the encoder knows "the ground-truth features here are hidden".

Saves `attn_weights` and `topk_idx` for reuse in L_align (Head 4).

### 3.3 MicroEncoder

**Bidirectional Mamba2** stack over the time sequence.

- Forward Mamba2 pass → `h_fwd[t]` = representation using past context.
- Backward Mamba2 pass (over reversed sequence) → `h_bwd[t]` = representation using future context.
- Trajectory embedding `H`: mean-pooled `h_fwd + h_bwd` over the **visibility mask** (not gap, not padded) so the embedding represents what the model actually *observed*.

This is the piece that makes **non-autoregressive gap filling** possible — see §5.

### 3.4 LossHeads — 4 heads fired simultaneously

| Head | Input | Target | Purpose |
|---|---|---|---|
| **ReconstructionHead** (L_mask) | `[h_fwd, h_bwd]` at position t | `micro_tokens[t]` at masked t | Gap filling — the main task |
| **NextPingHead** (L_next) | `h_fwd[t]` | `micro_tokens[t+1]` | Auxiliary: learn forward motion dynamics |
| **ContrastiveHead** (L_contrast) | `H` (trajectory embedding) | Same-MMSI pairs in batch | Identity-aware representation (NT-Xent) |
| **AlignmentHead** (L_align) | `vessel_Z` (macro summary) + `H` | Symmetric InfoNCE | Force macro and micro representations to match per-vessel |

Combined loss:

```
L_total = 1.0·L_mask + 0.3·L_next + 0.5·L_contrast + 0.5·L_align
```

### 3.5 The ReconstructionHead in v2.3 (delta mode)

This is the key change from v2.2. The head now predicts a **residual** from a linear-interpolation baseline:

```
baseline[t] = linear_interp(lat,lon at nearest visible neighbours of t)
pred_latlon[t] = baseline[t] + mlp_out[t, :2]
pred_other[t]  = mlp_out[t, 2:]     # sog, cog, hdg, delta_t stay absolute
```

The baseline is built inside `compute_linear_interp_baseline()`:

- For each position `t`, find the nearest **visible** (not masked, not padded) neighbour on the left (`prev`) and right (`succ`).
- Both anchors present → linear interpolation by index fraction.
- Only one anchor (edge extrapolation) → flat extrapolation using that anchor.
- Neither (empty window) → 0.5 fallback (normalized midpoint).

Why this matters → see the companion doc `HB_Mamba_v2_3_Delta_Head_Fix.md`.

---

## 4. Training

### 4.1 Forward pass (one step)

```
encoder_input = micro_tokens_masked     # gap features zeroed — NO cheat

macro_output = MacroEncoder(macro_features, macro_lat_idx, macro_lon_idx)

x_conditioned, attn_weights, topk_idx = CrossScaleInjection(
    encoder_input, macro_output, mask=mask     # [MASK] embedding at gap pos
)

visibility_mask = ~mask & ~padding_mask        # used to compute H correctly
h_fwd, h_bwd, H = MicroEncoder(x_conditioned, visibility_mask=visibility_mask)

# loss heads use FULL micro_tokens as target (not the masked version)
losses = LossHeads(h_fwd, h_bwd, H, macro_output, attn_weights, topk_idx,
                   micro_tokens, mask, padding_mask, mmsi)
```

### 4.2 Optimiser and schedule

- **AdamW**, `weight_decay=0.01`, applied only to 2-D+ parameters.
- **Peak LR**: `1e-4`.
- **Warmup**: linear, 5% of total steps.
- **Decay**: cosine to 0 after warmup.
- **Gradient clip**: `max_norm=2.0`.
- **Spike guard**: if pre-clip grad norm > 100, skip this step entirely (don't corrupt Adam's moments). The scheduler still advances.

### 4.3 Other details

- **Mixed precision**: `bfloat16` autocast on the forward/backward pass. No GradScaler needed on Ada (Hopper-class GPUs handle bf16 natively).
- **Gradient accumulation**: configurable (default 4 steps) → effective batch ≈ 64 on a single GPU.
- **Checkpointing**: saves `best.pt` on new lowest `val_L_total`, plus periodic `epoch_NNN.pt` every `save_every` epochs.
- **DDP**: the trainer is DDP-agnostic — wrap the model in `DistributedDataParallel` before instantiating the `Trainer` and it "just works".

### 4.4 Loss expectations

| Loss | Typical range at init | Typical range converged | Notes |
|---|---|---|---|
| `L_mask` | 0.04–0.05 | 0.014 | Dominated by cog/hdg noise once lat/lon is learnt |
| `L_next` | 0.30–0.40 | 0.007 | Easier than L_mask because the encoder sees all past context |
| `L_contrast` | 0.0 (no pos pairs in batch) | 0.0 | Requires same-MMSI pairs — rarely fires in default loaders |
| `L_align` | 1.5–2.0 | 0.01–0.05 | Symmetric InfoNCE; prone to transient spikes |
| `L_total` | ~1.0 | ~0.024 | Weighted sum |

A converged run on the full pipeline: `val_L_total ≈ 0.024`, `val_L_mask ≈ 0.014`.

### 4.5 Sanity: the single-batch overfit check

`overfit_check.py` trains 300 steps on a single fixed batch. A healthy model should:

- Drive `L_mask` down to roughly the irreducible feature-noise floor (~0.02 for a B=8 batch in v2.3).
- Drive `L_next` below 0.05.
- Produce no NaN/Inf.

With the v2.3 delta head, `L_mask` starts **low** at init (~0.04) — because `baseline ≈ target` already — so the drop during overfit looks smaller (0.04→0.02) than in pre-v2.3 runs (0.32→0.0001 via cheat-path). That smaller drop is **not a bug**; it's the direct consequence of starting closer to the solution.

---

## 5. Inference — how we actually fill a gap

### 5.1 Is this autoregressive?

No. The model predicts **every gap ping in a single forward pass**, not one-at-a-time. This surprises people because trajectory prediction "feels" sequential.

What makes it possible: the **bidirectional MicroEncoder** gives each gap position `t` a hidden state built from:

- `h_fwd[t]` — forward Mamba2 pass over positions `0..t`, so it sees **everything visible before** the gap.
- `h_bwd[t]` — backward Mamba2 pass over positions `T..t`, so it sees **everything visible after** the gap.

The gap positions themselves are fed as `[MASK]` embeddings (their dynamic features are zeroed), so the forward pass flows *through* the gap carrying left-context, and the backward pass flows through the gap carrying right-context. At every gap position the concatenation `[h_fwd[t], h_bwd[t]]` already contains enough context to reconstruct `t` — **no need to fill `t-1` before predicting `t`**.

Think of it as curve-fitting through two known endpoints: given you know the start and end of the gap, you can produce a smooth curve through the middle *all at once*. The model's job is to learn what a vessel's motion curve between two observed endpoints typically looks like.

Limitation: predictions don't feed back into each other. If sample correlation matters (e.g. the error at ping 5 should inform the prediction at ping 6), we would need iterative refinement — not currently implemented.

### 5.2 Inference pipeline

```python
# 1. Load checkpoint
ckpt = torch.load("checkpoints_v2_3_delta/best.pt", weights_only=False)
model = HBMamba(**ckpt["model_config"]).to(device)
model.load_state_dict(ckpt["model_state"], strict=False)    # tolerate new buffers
model.eval()

# 2. Build the input tensors from a single vessel's 24-hour window
#    (see hb_mamba_dataset.py — same collation as training)
#    You need: macro_features, macro_lat_idx, macro_lon_idx,
#              micro_tokens_masked (gap positions zeroed),
#              mask, padding_mask

# 3. Run the encoder (no loss heads yet)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.predict(macro_features, macro_lat_idx, macro_lon_idx,
                        micro_tokens_masked,
                        padding_mask=padding_mask,
                        mask=mask)
h_fwd = out["h_fwd"]   # [B, N_day, 256]
h_bwd = out["h_bwd"]   # [B, N_day, 256]

# 4. Build the linear-interpolation baseline (v2.3 delta head requirement)
from model.loss_heads import compute_linear_interp_baseline
baseline = compute_linear_interp_baseline(micro_tokens_masked, mask, padding_mask)
# baseline: [B, N_day, 2]

# 5. Run the reconstruction head with the baseline
pred = model.loss_heads.recon_head(h_fwd, h_bwd, baseline)
# pred: [B, N_day, 8]  — (lat, lon, sog, cog_sin, cog_cos, hdg_sin, hdg_cos, delta_t)

# 6. Clamp to [0, 1] and denormalise lat/lon back to degrees
pred = pred.clamp(0.0, 1.0)
pred_lat_deg = LAT_MIN + pred[..., 0] * (LAT_MAX - LAT_MIN)
pred_lon_deg = LON_MIN + pred[..., 1] * (LON_MAX - LON_MIN)

# 7. Use pred_lat/lon_deg ONLY at gap positions — visible positions are already known
final_lat = torch.where(mask, pred_lat_deg, actual_lat)
final_lon = torch.where(mask, pred_lon_deg, actual_lon)
```

### 5.3 Output semantics in v2.3

`recon_head` now returns **final absolute predictions** (not deltas):

- Channels 0,1 (lat, lon): `baseline + mlp_delta` — the head internally adds the baseline.
- Channels 2–7 (sog, cog, hdg, delta_t): plain MLP output (no delta).

So inference code downstream of `recon_head` treats the output exactly like in v2.2 — the delta math is internal to the head.

### 5.4 Utility scripts

- `predict_path.py` — batch inference over the val/test set; emits per-gap predictions and overall error stats.
- `visualize_gap_prediction.py` — plots small multi-panel figures (model path vs actual vs linear baseline) for selected samples.
- `visualize_trajectory_comparison.py` — plots long continuous trajectories (one figure = one vessel) and exports three diagnostic files:
  - `gap_ping_details_*.csv` — one row per gap ping with 40+ columns (actual/pred/linear positions, errors, bias).
  - `sample_summary_*.csv` — per-sample aggregates.
  - `diagnostics_report_*.txt` — human-readable report with global summary, error distribution, per-sample breakdown, feature-level analysis, error-vs-gap-position buckets.
- `inference.py` — lightweight single-window inference for ad-hoc checks.

All three inference scripts now compute and pass the baseline to `recon_head`.

---

## 6. File Map

| Path | Role |
|---|---|
| `model/macro_encoder.py` | MacroEncoder (Component 1) |
| `model/cross_scale_injection.py` | CSI with `[MASK]` embedding (Component 2) |
| `model/micro_decoder.py` | Bidirectional MicroEncoder (Component 3) |
| `model/loss_heads.py` | 4 heads + `compute_linear_interp_baseline` helper (Component 4) |
| `model/hb_mamba.py` | Top-level `HBMamba` wrapper + `forward` / `predict` |
| `hb_mamba_dataset.py` | Dataset, gap masking, collation |
| `training/trainer.py` | AdamW + warmup-cosine + spike-skip + checkpointing |
| `train.py` | CLI entry point for training |
| `overfit_check.py` | 300-step single-batch sanity test |
| `predict_path.py` | Batch inference with error stats |
| `visualize_gap_prediction.py` | Per-sample grid figures |
| `visualize_trajectory_comparison.py` | Long-trajectory comparison + diagnostics export |
| `inference.py` | Single-window utility inference |

---

## 7. Training Results (v2.3)

| Epoch | val_L_mask | val_L_next | val_L_align | val_L_total |
|---|---|---|---|---|
| 17 | 0.0143 | 0.0072 | 0.0078 | **0.0240** (best so far) |

On the held-out `val` set with 6 long-trajectory samples (see `figs/new_model_v2_3_delta/`):

| Metric | v2.2 | v2.3 | Linear baseline |
|---|---|---|---|
| Mean km error | 143.7 | **~13** | 22.4 |
| Worst sample | 448 km | 27 km | — |
| Samples that beat linear | 0/6 | 1/6 (and others close) | — |

The delta head closes the gap to linear baseline performance and, on sufficiently long trajectories, begins to exceed it.
