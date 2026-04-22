"""
Trainer — HB-Mamba v2.0
========================

Training loop with:
    - AdamW + cosine-warmup LR schedule
    - bfloat16 autocast  (no GradScaler needed on Ada GPUs)
    - Gradient accumulation
    - Gradient clipping
    - Per-loss logging
    - Best-val checkpoint saving

Training config (from architecture spec):
    Optimiser       AdamW  weight_decay=0.01
    Peak LR         1e-4
    Warmup          5% of total steps  (linear)
    Decay           cosine to 0 after warmup
    Batch size      64 per GPU  (128 effective across 2 GPUs)
    Grad accum      4 steps  → effective batch 256 / GPU
    Mixed precision bfloat16
    Grad clip       2.0   (with skip-step guard at |grad| > 100)
    Epochs          30–40

DDP note:
    Wrap the model in DistributedDataParallel BEFORE passing it to Trainer.
    Trainer itself is DDP-agnostic — it calls model.forward() and
    model.param_groups() through whatever wrapper is given.

Validation determinism note:
    During validation the DataLoader should use a seeded HBMambaDataset so
    that every epoch evaluates on the same gap positions (architecture doc,
    Section 9.2).  This is a dataset concern, not a trainer concern.
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """
    All training hyperparameters in one place.

    Parameters
    ----------
    checkpoint_dir : str   Where to save .pt checkpoints.
    lr             : float Peak learning rate.
    weight_decay   : float AdamW weight decay (applied to 2-D+ params only).
    max_epochs     : int   Total training epochs.
    warmup_frac    : float Fraction of total steps used for linear warmup.
    grad_clip      : float Max gradient norm (global).
    grad_accum     : int   Gradient accumulation steps.
    use_amp        : bool  Use bfloat16 autocast.
    log_every      : int   Print train metrics every N optimizer steps.
    save_every     : int   Always save a checkpoint every N epochs (on top of best-val).
    """
    checkpoint_dir      : str   = "checkpoints"
    lr                  : float = 1e-4
    weight_decay        : float = 0.01
    max_epochs          : int   = 40
    warmup_frac         : float = 0.05
    grad_clip           : float = 2.0
    skip_grad_threshold : float = 100.0
    grad_accum          : int   = 4
    use_amp             : bool  = True
    log_every           : int   = 10
    save_every          : int   = 5


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training and validation loop for HBMamba.

    Parameters
    ----------
    model        : nn.Module       HBMamba (or DDP-wrapped HBMamba).
    train_loader : DataLoader      Training DataLoader.
    val_loader   : DataLoader      Validation DataLoader.
    config       : TrainerConfig   Hyperparameters.
    device       : torch.device    Target device.
    """

    def __init__(
        self,
        model        : nn.Module,
        train_loader : DataLoader,
        val_loader   : DataLoader,
        config       : TrainerConfig,
        device       : torch.device,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = config
        self.device       = device

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── Optimizer ──────────────────────────────────────────────────────
        # Use param_groups() if available (HBMamba exposes it); fall back to
        # all parameters with a single weight-decay setting.
        if hasattr(model, "param_groups"):
            pg = model.param_groups(lr=config.lr, weight_decay=config.weight_decay)
        else:
            # DDP-wrapped model — access inner module
            inner = model.module if hasattr(model, "module") else model
            pg    = inner.param_groups(lr=config.lr, weight_decay=config.weight_decay)

        self.optimizer = torch.optim.AdamW(pg, lr=config.lr)

        # ── LR scheduler — linear warmup then cosine decay ─────────────────
        # total_steps is estimated from loader length; updated if loader changes.
        steps_per_epoch  = math.ceil(len(train_loader) / config.grad_accum)
        self.total_steps = steps_per_epoch * config.max_epochs
        warmup_steps     = max(1, int(self.total_steps * config.warmup_frac))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps                         # linear warmup
            progress = (step - warmup_steps) / max(1, self.total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))    # cosine decay

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_lr_lambda
        )

        # ── State ──────────────────────────────────────────────────────────
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epoch = 0

    # ── Internal helpers ────────────────────────────────────────────────────

    def _move_batch(self, batch: Dict) -> Dict:
        """Move all tensor values in a batch dict to the training device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v   # mmsi (list), date (list), gap_type (list)
        return out

    def _forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single forward pass; returns loss dict."""
        # CRITICAL: pass micro_tokens_masked so the encoder sees zeros at gap
        # positions (as at inference). Omitting this lets the model cheat by
        # reading ground-truth at "masked" positions during training → training
        # loss collapses to ~0 via copy-through but inference is pure noise.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.use_amp):
            return self.model(
                batch["macro_features"],
                batch["macro_lat_idx"],
                batch["macro_lon_idx"],
                batch["micro_tokens"],
                batch["mask"],
                batch["padding_mask"],
                batch["mmsi"],
                batch.get("micro_tokens_masked"),
            )

    @staticmethod
    def _fmt_losses(d: Dict[str, float], prefix: str = "") -> str:
        parts = [f"{prefix}{k}={v:.4f}" for k, v in d.items()]
        return "  ".join(parts)

    # ── Core loops ──────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        accum    = defaultdict(float)   # loss accumulator for logging
        n_log    = 0
        skipped  = 0                    # steps skipped this log window
        t0       = time.time()

        self.optimizer.zero_grad()

        for batch_idx, raw_batch in enumerate(self.train_loader):
            batch  = self._move_batch(raw_batch)
            losses = self._forward(batch)

            # Scale by accum steps so the effective gradient magnitude is
            # independent of grad_accum setting.
            scaled = losses["L_total"] / self.cfg.grad_accum
            scaled.backward()

            # Accumulate loss values for logging (use unscaled values)
            for k, v in losses.items():
                accum[k] += v.item()
            n_log += 1

            # Optimizer step after every grad_accum batches
            is_accum_step = ((batch_idx + 1) % self.cfg.grad_accum == 0)
            is_last_batch = (batch_idx + 1 == len(self.train_loader))

            if is_accum_step or is_last_batch:
                # clip_grad_norm_ returns the pre-clip total norm — use it to
                # detect and reject pathological gradient spikes that would
                # otherwise corrupt Adam's running moments.
                gn = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
                gn_val = gn.item() if torch.is_tensor(gn) else float(gn)

                if (not math.isfinite(gn_val)) or gn_val > self.cfg.skip_grad_threshold:
                    # Spike or NaN — drop this update entirely.
                    self.optimizer.zero_grad()
                    skipped += 1
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Advance the schedule even on skipped steps so warmup/cosine
                # shape stays tied to wall-clock progress rather than to how
                # many updates were accepted.
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.cfg.log_every == 0:
                    avg   = {k: v / n_log for k, v in accum.items()}
                    lr    = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(
                        f"  epoch {epoch:3d}  step {self.global_step:6d}"
                        f"  lr={lr:.2e}  gn={gn_val:.2f}  skip={skipped}"
                        f"  {self._fmt_losses(avg, '')}  ({elapsed:.1f}s)"
                    )
                    accum   = defaultdict(float)
                    n_log   = 0
                    skipped = 0
                    t0      = time.time()

        return {k: v / max(1, n_log) for k, v in accum.items()}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        totals  = defaultdict(float)
        n       = 0

        for raw_batch in self.val_loader:
            batch  = self._move_batch(raw_batch)
            losses = self._forward(batch)
            for k, v in losses.items():
                totals[k] += v.item()
            n += 1

        return {k: v / max(1, n) for k, v in totals.items()}

    # ── Checkpoint helpers ──────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str) -> Path:
        """Save model + optimizer + scheduler + trainer state."""
        inner  = self.model.module if hasattr(self.model, "module") else self.model
        config = inner.config if hasattr(inner, "config") else {}

        ckpt_path = self.checkpoint_dir / f"{tag}.pt"
        torch.save({
            "epoch"       : epoch,
            "global_step" : self.global_step,
            "val_loss"    : val_loss,
            "model_config": config,
            "model_state" : inner.state_dict(),
            "optimizer"   : self.optimizer.state_dict(),
            "scheduler"   : self.scheduler.state_dict(),
        }, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str) -> Dict:
        """
        Restore model, optimizer, and scheduler from a checkpoint.

        Returns the checkpoint dict (contains epoch, val_loss, etc.).
        """
        ckpt  = torch.load(path, map_location=self.device, weights_only=False)
        inner = self.model.module if hasattr(self.model, "module") else self.model
        inner.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step  = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        self.epoch        = ckpt.get("epoch", 0)
        return ckpt

    # ── Main entry point ────────────────────────────────────────────────────

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Run the full training loop.

        Parameters
        ----------
        resume_from : str | None   Path to a checkpoint to resume from.
        """
        start_epoch = 0
        if resume_from:
            ckpt = self.load_checkpoint(resume_from)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"Resumed from {resume_from}  (epoch {start_epoch})")

        print(
            f"\nTraining HBMamba v2.0"
            f"\n  epochs={self.cfg.max_epochs}  lr={self.cfg.lr}"
            f"  grad_accum={self.cfg.grad_accum}  clip={self.cfg.grad_clip}"
            f"  amp={self.cfg.use_amp}"
            f"\n  total_steps={self.total_steps}"
            f"  warmup={int(self.total_steps * self.cfg.warmup_frac)} steps\n"
        )

        for epoch in range(start_epoch, self.cfg.max_epochs):
            self.epoch = epoch
            t_ep = time.time()

            # Training
            train_metrics = self._train_epoch(epoch)

            # Validation
            val_metrics = self._val_epoch(epoch)
            val_loss    = val_metrics["L_total"]

            elapsed = time.time() - t_ep
            print(
                f"Epoch {epoch:3d}/{self.cfg.max_epochs}  "
                f"{self._fmt_losses(val_metrics, 'val_')}  "
                f"({elapsed:.0f}s)"
            )

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                path = self._save_checkpoint(epoch, val_loss, "best")
                print(f"  -> new best val loss {val_loss:.4f}  saved {path}")

            # Periodic checkpoint
            if (epoch + 1) % self.cfg.save_every == 0:
                path = self._save_checkpoint(epoch, val_loss, f"epoch_{epoch:03d}")
                print(f"  -> periodic checkpoint saved {path}")

        print(f"\nTraining complete.  Best val loss: {self.best_val_loss:.4f}")


# =============================================================================
# Self-contained test block
# =============================================================================

if __name__ == "__main__":
    import sys
    import tempfile
    import itertools
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from hb_mamba_dataset import build_dataloaders
    from model.hb_mamba import HBMamba

    if not torch.cuda.is_available():
        print("No CUDA device — test requires GPU (Mamba2 is CUDA-only).")
        sys.exit(1)

    device = torch.device("cuda")

    _raw     = _root / "data_genration_and_raw_data" / "raw_data"
    _ns_path = str(_raw / "preprocessing" / "norm_stats" / "norm_stats.json")

    print("=" * 65)
    print(f"Trainer — HB-Mamba v2.0  self-test  (device: {device})")
    print("=" * 65)

    all_passed = True

    def _check(name: str, cond: bool) -> None:
        global all_passed
        s = "PASS" if cond else "FAIL"
        if not cond:
            all_passed = False
        print(f"  [{s}]  {name}")

    # ── Single model instantiation for the whole test ──────────────────────
    # Each HBMamba creation triggers Mamba2 CUDA kernel compilation — expensive.
    # All checks share this one model; we reset gradients between checks.
    loaders = build_dataloaders(
        dataset_index_dir = str(_raw / "dataset_index"),
        norm_stats_path   = _ns_path,
        batch_size        = 4,
        num_workers       = 0,
        pin_memory        = False,
    )
    train_iter = iter(loaders["train"])
    batch      = trainer_batch = next(train_iter)

    model = HBMamba.from_norm_stats(_ns_path).to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainerConfig(
            checkpoint_dir = tmpdir,
            lr             = 1e-4,
            weight_decay   = 0.01,
            max_epochs     = 2,
            warmup_frac    = 0.10,
            grad_clip      = 1.0,
            grad_accum     = 2,
            use_amp        = True,
            log_every      = 5,
            save_every     = 1,
        )

        trainer = Trainer(
            model        = model,
            train_loader = loaders["train"],
            val_loader   = loaders.get("val", loaders["train"]),
            config       = cfg,
            device       = device,
        )

        # ── Check 1 — Trainer construction ────────────────────────────────
        print("\nCheck 1 — Trainer construction")

        _check("optimizer is AdamW",
               isinstance(trainer.optimizer, torch.optim.AdamW))
        _check("scheduler is LambdaLR",
               isinstance(trainer.scheduler,
                          torch.optim.lr_scheduler.LambdaLR))
        _check("total_steps > 0",        trainer.total_steps > 0)
        _check("checkpoint_dir exists",  trainer.checkpoint_dir.exists())
        print(f"  total_steps  = {trainer.total_steps}")
        print(f"  warmup_steps = {int(trainer.total_steps * cfg.warmup_frac)}")

        # ── Check 2 — Single forward + backward + optimizer step ──────────
        print("\nCheck 2 — Single forward + backward + optimizer step")

        b = trainer._move_batch(batch)
        model.train()
        trainer.optimizer.zero_grad()
        losses = trainer._forward(b)

        _check("L_total is finite", torch.isfinite(losses["L_total"]).item())

        losses["L_total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        _check("grad norm is finite", math.isfinite(total_norm))
        _check("grad norm > 0",       total_norm > 0)
        print(f"  grad norm before clip = {total_norm:.4f}")

        trainer.optimizer.step()
        trainer.scheduler.step()
        lr_after = trainer.scheduler.get_last_lr()[0]
        _check("LR after 1 step > 0", lr_after > 0)
        print(f"  LR after step 1 = {lr_after:.6f}")

        # ── Check 3 — LR schedule shape (pure arithmetic, no model) ───────
        print("\nCheck 3 — LR schedule shape (warmup then cosine decay)")

        # Test the schedule formula directly on a tiny dummy optimizer —
        # no model instantiation needed.
        _dummy_p = torch.nn.Parameter(torch.zeros(1))
        _dummy_opt = torch.optim.AdamW([_dummy_p], lr=cfg.lr)

        T_test      = 200                         # synthetic total_steps
        wup         = max(1, int(T_test * cfg.warmup_frac))

        def _lr_lambda_test(step: int) -> float:
            if step < wup:
                return step / wup
            progress = (step - wup) / max(1, T_test - wup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        _sched = torch.optim.lr_scheduler.LambdaLR(_dummy_opt, _lr_lambda_test)
        lrs = []
        for _ in range(T_test):
            _dummy_opt.step()
            _sched.step()
            lrs.append(_sched.get_last_lr()[0])

        lr_peak  = max(lrs)
        lr_final = lrs[-1]

        _check("LR peaks within warmup zone",  lrs.index(lr_peak) <= wup + 2)
        _check("LR at end < LR at peak",       lr_final < lr_peak)
        _check("LR at end >= 0",               lr_final >= 0)
        print(f"  peak LR  = {lr_peak:.6f}  at step {lrs.index(lr_peak)}")
        print(f"  final LR = {lr_final:.6f}")

        # ── Check 4 — Validation (capped at 3 batches) ────────────────────
        print("\nCheck 4 — Validation epoch (3 batches)")

        # Monkey-patch val_loader to yield only 3 batches for the test
        val_loader_orig = trainer.val_loader
        trainer.val_loader = list(itertools.islice(loaders["train"], 3))

        val_metrics = trainer._val_epoch(epoch=0)
        trainer.val_loader = val_loader_orig   # restore

        expected = {"L_mask", "L_next", "L_contrast", "L_align", "L_total"}
        _check("val returns all 5 loss keys", set(val_metrics.keys()) == expected)
        for k, v in val_metrics.items():
            _check(f"val {k} is finite  (={v:.4f})", math.isfinite(v))

        # ── Check 5 — Checkpoint save / load ──────────────────────────────
        print("\nCheck 5 — Checkpoint save / load")

        ckpt_path = trainer._save_checkpoint(epoch=0, val_loss=0.5, tag="test")
        _check("checkpoint file created", ckpt_path.exists())

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        _check("has model_state",  "model_state"  in ckpt)
        _check("has optimizer",    "optimizer"    in ckpt)
        _check("has scheduler",    "scheduler"    in ckpt)
        _check("has model_config", "model_config" in ckpt)
        _check("epoch == 0",       ckpt["epoch"]  == 0)
        _check("val_loss == 0.5",  abs(ckpt["val_loss"] - 0.5) < 1e-6)

        # Verify round-trip: save state → scramble weights → load → same output
        mf = b["macro_features"]
        la = b["macro_lat_idx"]
        lo = b["macro_lon_idx"]
        mt = b["micro_tokens"]

        with torch.no_grad():
            H_before = model.predict(mf, la, lo, mt)["H"].clone()

        # Scramble all parameters
        with torch.no_grad():
            for p in model.parameters():
                p.data.uniform_(-1, 1)

        # Restore from checkpoint
        model.load_state_dict(ckpt["model_state"])

        with torch.no_grad():
            H_after = model.predict(mf, la, lo, mt)["H"]

        _check("H identical after save-scramble-load (tol=1e-5)",
               torch.allclose(H_before, H_after, atol=1e-5))

        # ── Check 6 — Gradient accumulation ──────────────────────────────
        print("\nCheck 6 — Gradient accumulation  (accum×2 == single scaled step)")

        # One backward with full loss
        model.train()
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l1 = model(mf, la, lo, mt,
                       b["mask"], b["padding_mask"], b["mmsi"])["L_total"]
        l1.backward()
        norm_single = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        # Two backwards with loss / 2  (gradient accumulation)
        model.zero_grad()
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l2 = model(mf, la, lo, mt,
                           b["mask"], b["padding_mask"], b["mmsi"])["L_total"]
            (l2 / 2).backward()
        norm_accum = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        _check("single-step grad norm finite and > 0",
               math.isfinite(norm_single) and norm_single > 0)
        _check("accum grad norm finite and > 0",
               math.isfinite(norm_accum) and norm_accum > 0)
        # Both runs use the same batch and same model weights, so norms should
        # be close (bfloat16 rounding may cause small differences).
        ratio = norm_accum / (norm_single + 1e-9)
        _check("accum norm ≈ single norm  (ratio 0.9–1.1)",  0.9 < ratio < 1.1)
        print(f"  single norm = {norm_single:.4f}")
        print(f"  accum  norm = {norm_accum:.4f}  (ratio={ratio:.3f})")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_passed:
        print("All checks passed — Trainer v2.0 OK")
    else:
        print("One or more checks FAILED — see [FAIL] lines above")
    print("=" * 65)
