# HB-Mamba v2.0 / v2.1 — Architecture & Key Improvements

---

## 1. Overview

HB-Mamba (**H**ierarchical **B**idirectional **Mamba**) is a 57M-parameter model for AIS vessel trajectory gap prediction in the Gulf of Mexico. Given a 24-hour ping sequence with a masked gap, the model reconstructs the missing positions and predicts the next ping after the sequence.

**Task:** Given visible AIS pings (lat, lon, sog, cog, hdg, delta_t) with one contiguous gap zeroed out, predict the actual positions inside that gap.

**Dataset split:** ~99K train samples / ~18.6K val samples — one sample = one vessel-day.

---

## 2. Architecture

### 2.1 Pipeline (four components in series)

```
macro_features ──────────────────────────────────────────────────────────────────►
                 [ MacroEncoder ]                                                  │
                       │  macro_output [B, N_cells, 256]                          │
                       ▼                                                           │
micro_tokens_masked ──►[ CrossScaleInjection ]──────────────────────────────────►│
                            x_cond [B, N_day, 256]  attn_weights  topk_idx        │
                               │                                                   │
                               ▼                                                   │
                       [ MicroEncoder (Bidirectional Mamba2) ]                    │
                            h_fwd [B, N_day, 256]                                 │
                            h_bwd [B, N_day, 256]                                 │
                            H     [B, 256]  (trajectory embedding)                │
                               │                                                   │
                               ▼                                                   │
                       [ LossHeads ]◄─────────────────────────────────────────────┘
                            L_mask / L_next / L_contrast / L_align
```

---

### 2.2 Component 1 — MacroEncoder

Encodes the 29×36 spatial grid of historical AIS cell statistics into per-cell embeddings.

| Property | Value |
|---|---|
| Grid size | 29 lat × 36 lon = 1044 cells |
| Input features per cell | macro AIS statistics (vessel density, mean speed, etc.) |
| Lat/lon embeddings | Learned, 256-dim each |
| Output | `[B, 1044, 256]` macro_output |
| Parameters | 27,552,512 |

---

### 2.3 Component 2 — CrossScaleInjection (CSI)

Bridges the spatial macro context with the temporal micro ping sequence using **Top-K=32 sparse cross-attention**. At masked gap positions, a learned `[MASK]` embedding is injected so the model never sees real positional information at those positions.

| Property | Value |
|---|---|
| Input (query) | micro_tokens_masked `[B, N_day, 11]` |
| Input (key/value) | macro_output `[B, 1044, 256]` |
| Top-K | 32 nearest macro cells per ping |
| Mask token | Learned `nn.Parameter([256])` |
| Positional encoding | Sinusoidal (fixed), added to queries |
| Output | `x_cond [B, N_day, 256]` |
| Parameters | 1,055,744 |

**Mask injection logic:**
```python
# At masked positions: replace projected micro_tokens with learned mask_token
# cross_scale_injection.py lines ~154-170
q = self.W_Q(micro_proj + PE)          # query from micro features
q[mask] = self.mask_token              # inject learned [MASK] at gap positions
# → TopK attention proceeds with mask_token as query for gap positions
```

---

### 2.4 Component 3 — MicroEncoder (Bidirectional Mamba2)

Processes `x_cond` in both forward and backward directions using **Mamba2 SSM** (Triton-backed). The two streams are kept separate — no early fusion.

| Property | Value |
|---|---|
| d_model | 512 (each direction = 256) |
| n_layers | 8 (4 fwd + 4 bwd, interleaved) |
| d_state | 128 (SSM state size) |
| d_proj | 256 |
| Trajectory embedding H | Mean of `h_fwd + h_bwd` over **visible-only** positions |
| Output | `h_fwd [B, N_day, 256]`, `h_bwd [B, N_day, 256]`, `H [B, 256]` |
| Parameters | 27,513,600 |

---

### 2.5 Component 4 — LossHeads (four self-supervised objectives)

| Head | Input | Target | Loss type |
|---|---|---|---|
| **L_mask** Gap reconstruction | `[h_fwd, h_bwd]` → `Linear(512→256)→GELU→Linear(256→8)` | `micro_tokens[:, mask, :8]` | Weighted Huber |
| **L_next** Next-ping prediction | `h_fwd[:, :-1]` → adapter → `Linear(256→128)→GELU→Linear(128→8)` | `micro_tokens[:, 1:, :8]` | Weighted Huber |
| **L_contrast** Contrastive | `H → Linear(256→256)→GELU→Linear(256→128)→L2_norm` | Same MMSI in batch | NT-Xent (τ=0.07) |
| **L_align** Cross-scale | vessel_Z (attn-weighted macro) vs H | Diagonal (same vessel) | Symmetric InfoNCE |

**Combined loss:**
```
L_total = 1.0·L_mask + 0.3·L_next + 0.5·L_contrast + 0.5·L_align
```

**Target features (first 8 of 11 micro_tokens):**
```
[lat, lon, sog, cog_sin, cog_cos, hdg_sin, hdg_cos, delta_t]
  0    1    2      3        4        5        6        7
```

**Feature weights for Huber loss (v2.2 change):**
```
[5.0, 5.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3]
 lat  lon  sog  cog_sin cog_cos hdg_sin hdg_cos delta_t
```

---

### 2.6 Full parameter summary

| Component | Parameters |
|---|---|
| MacroEncoder | 27,552,512 |
| CrossScaleInjection | 1,055,744 |
| MicroEncoder | 27,513,600 |
| LossHeads | 1,713,168 |
| **Total** | **57,835,024** |

---

## 3. Key Changes That Caused Loss Improvement

---

### Change 1 — THE CHEAT BUG FIX ⭐ (7× km error improvement)

**File:** `training/trainer.py` — `_forward()` method

**Root cause:** During real training, `_forward()` passed the **full unmasked** `micro_tokens` to the model. The model saw the actual lat/lon at masked gap positions at every training step. It learned to copy them through. At inference, it saw **zeroed** gap positions for the first time → predictions were random.

Meanwhile the overfit loop in `train.py` correctly passed `micro_tokens_masked`, so the overfit check passed — masking the bug.

**Symptom:** Training L_mask collapsed to 0.003 suspiciously fast. Inference correlation between predicted and actual lat = **−0.07** (worse than random). Mean km error = **877 km**.

**Before:**
```python
# training/trainer.py — _forward()
def _forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.use_amp):
        return self.model(
            batch["macro_features"],
            batch["macro_lat_idx"],
            batch["macro_lon_idx"],
            batch["micro_tokens"],      # ← WRONG: full unmasked tokens
            batch["mask"],
            batch["padding_mask"],
            batch["mmsi"],
            # micro_tokens_masked was never passed → fell back to micro_tokens inside model
        )
```

**After:**
```python
# training/trainer.py — _forward()
def _forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.use_amp):
        return self.model(
            batch["macro_features"],
            batch["macro_lat_idx"],
            batch["macro_lon_idx"],
            batch["micro_tokens"],
            batch["mask"],
            batch["padding_mask"],
            batch["mmsi"],
            batch.get("micro_tokens_masked"),  # ← FIXED: pass masked input
        )
```

**Result:** Mean km error dropped **877 km → 124 km** (7× improvement) after 2 epochs of clean training.

---

### Change 2 — Optimizer Stability (LR warmup + grad clipping + skip-spike guard)

**File:** `training/trainer.py`, `train.py`

The original trainer had no warmup, fixed grad clip of 1.0, and no protection against gradient spikes. Early training showed unstable loss spikes.

**Before:**
```python
# TrainerConfig
grad_clip: float = 1.0   # no skip-spike guard

# _train_epoch — no spike protection
nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
self.optimizer.step()
self.optimizer.zero_grad()
self.scheduler.step()
```

**After:**
```python
# TrainerConfig
grad_clip: float = 2.0
skip_grad_threshold: float = 100.0   # skip step if norm blows up

# _train_epoch — skip spike guard
gn = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
gn_val = gn.item() if torch.is_tensor(gn) else float(gn)
if (not math.isfinite(gn_val)) or gn_val > self.cfg.skip_grad_threshold:
    self.optimizer.zero_grad()
    skipped += 1
else:
    self.optimizer.step()
    self.optimizer.zero_grad()
self.scheduler.step()   # scheduler always advances, even on skipped steps
```

**LR schedule added:**
```python
# Linear warmup (5% of total steps) → cosine decay to 0
self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
    self.optimizer,
    max_lr       = self.cfg.lr,
    total_steps  = total_steps,
    pct_start    = self.cfg.warmup_frac,
    anneal_strategy = "cos",
)
```

---

### Change 3 — Feature-Weighted Huber Loss (v2.2, ongoing)

**File:** `model/loss_heads.py`

**Root cause of plateau:** The plain Huber was averaged equally across all 8 features. `cog_sin/cog_cos/hdg_sin/hdg_cos/delta_t` have irreducible aleatoric noise (vessels turn unpredictably). Their per-feature Huber floors at ~0.08. When averaged with lat/lon (~0.01 each), the loss is capped at ~0.047 regardless of how well position is predicted. The model was wasting capacity trying to fit unfittable noise.

**Diagnostic evidence:**
```
mae_lat_deg : 1.13°   (≈ 126 km)
mae_lon_deg : 1.25°   (≈ 125 km)
mae_cog_deg : 34.9°   ← model learned almost nothing about course
```

**Before:**
```python
# loss_heads.py — LossHeads.__init__
self.huber = nn.HuberLoss(reduction="mean")  # equal weight across all 8 features

# forward
L_mask = self.huber(y_hat_mask[active], target[active])
L_next = self.huber(y_hat_next[next_active], target_next[next_active])
```

**After:**
```python
# loss_heads.py — LossHeads.__init__
feat_w = torch.tensor([5.0, 5.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3])
#                     lat  lon  sog  cog_sin cog_cos hdg_sin hdg_cos delta_t
self.register_buffer("feature_weights", feat_w)

# loss_heads.py — new helper method
def _weighted_huber(self, pred: Tensor, target: Tensor) -> Tensor:
    per_elem = F.huber_loss(pred, target, reduction="none")  # [..., 8]
    weighted = per_elem * self.feature_weights               # broadcast
    return weighted.mean()

# forward
L_mask = self._weighted_huber(y_hat_mask[active], target[active])
L_next = self._weighted_huber(y_hat_next[next_active], target_next[next_active])
```

**Expected result:** km error to drop from ~195 km to <80 km (training ongoing).

---

### Change 4 — Cross-Scale Alignment: Cosine Distance → Symmetric InfoNCE

**File:** `model/loss_heads.py`

**Root cause:** The original `L_align` used a pure cosine distance loss:
```
L_align = mean(1 - z' · h')
```
This **mode-collapsed** — both projections converged to the same constant direction (loss trivially minimized). The model stopped learning cross-scale structure.

**Before:**
```python
# L_align — collapses to constant direction
L_align = (1.0 - (z_prime * h_prime).sum(dim=-1)).mean()
```

**After:**
```python
# Symmetric InfoNCE — positives = (z_i, h_i) same vessel, negatives = all off-diagonal
logits_zh = torch.mm(h_prime, z_prime.T) / self.tau   # [B, B]
labels    = torch.arange(B_a, device=logits_zh.device)
L_align   = (F.cross_entropy(logits_zh,   labels)
           + F.cross_entropy(logits_zh.T, labels)) * 0.5
# Can't collapse: collapse → every off-diag logit matches diagonal → loss blows up
```

---

### Change 5 — NextPingHead Decoupling Adapter (v2.1)

**File:** `model/loss_heads.py`

Without a decoupling adapter, `L_next` gradients forced `h_fwd` to encode `t+1` information directly — interfering with `L_mask`'s requirement for `h_fwd` to encode context for gap reconstruction.

**Before:**
```python
class NextPingHead(nn.Module):
    def __init__(self, d_model=256, n_features=8):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_features),
        )
    def forward(self, h_fwd):
        return self.mlp(h_fwd)   # L_next grads flow directly into backbone
```

**After:**
```python
class NextPingHead(nn.Module):
    def __init__(self, d_model=256, n_features=8):
        self.adapter = nn.Linear(d_model, d_model)   # absorbs t+1 bias
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_features),
        )
    def forward(self, h_fwd):
        return self.mlp(self.adapter(h_fwd))   # adapter isolates gradient path
```

Also: `w_next` reduced from **0.8 → 0.3** to prevent `L_next` from overpowering `L_mask`.

---

## 4. Training Results Summary

| Version | Epochs | Training setup | Mean km error | Median km error |
|---|---|---|---|---|
| Pre-fix (cheat bug) | 6 | micro_tokens passed (unmasked) | ~877 km | ~830 km |
| v2.1 clean | 2 | micro_tokens_masked (fixed) | 124 km | 97 km |
| v2.1 + feat-weighted | 6 | plateau — loss stopped dropping | 195 km* | — |
| v2.2 (ongoing) | — | feat-weighted Huber, 15 epochs | TBD | TBD |

*195 km is on a harder sample subset (mean gap = 49.6 pings, some up to 244 pings)

---

## 5. Inference Pipeline (Correct v2.1 Path)

```python
# 1. Dataset provides micro_tokens_masked:
#    - features 0..7 zeroed at gap positions
#    - features 8..10 (vessel length, draft, type) preserved from nearest visible ping
#    - delta_t boundary corrected

# 2. Pass masked tokens + mask tensor to predict()
out = model.predict(
    macro_features,
    macro_lat_idx,
    macro_lon_idx,
    micro_tokens_masked,   # NOT micro_tokens
    mask=mask,             # → CSI injects [MASK] token, H uses visible-only mean
)

# 3. Reconstruct gap positions
pred = model.loss_heads.recon_head(out["h_fwd"], out["h_bwd"])
pred = pred.clamp(0.0, 1.0)   # normalize to Gulf of Mexico bounds

# 4. Denormalize
lat_pred = pred[..., 0] * (LAT_MAX - LAT_MIN) + LAT_MIN
lon_pred = pred[..., 1] * (LON_MAX - LON_MIN) + LON_MIN
```

---

## 6. File Map

| File | Role |
|---|---|
| `model/hb_mamba.py` | Full model: forward + predict |
| `model/macro_encoder.py` | Component 1: spatial cell encoder |
| `model/cross_scale_injection.py` | Component 2: Top-K cross-attention bridge |
| `model/micro_decoder.py` | Component 3: Bidirectional Mamba2 |
| `model/loss_heads.py` | Component 4: four self-supervised heads |
| `hb_mamba_dataset.py` | Dataset: masking, normalization, collation |
| `training/trainer.py` | Training loop, optimizer, scheduler |
| `train.py` | CLI entry point, overfit sanity check |
| `predict_path.py` | Inference: km error evaluation |
| `visualize_gap_prediction.py` | Visual plot: predicted vs actual trajectories |
| `inference.py` | Embedding extraction |
