"""
train.py — HB-Mamba v2.0  Training Entry Point
================================================

Single-GPU usage:
    python train.py

Multi-GPU DDP via gloo (2 GPUs, no NCCL — works on any CUDA driver):
    torchrun --nproc_per_node=2 train.py

Multi-GPU DDP via nccl (requires matching CUDA driver):
    torchrun --nproc_per_node=2 train.py --nccl

Key flags:
    --checkpoint_dir   DIR     Where to save checkpoints  [checkpoints]
    --resume           PATH    Resume from a checkpoint
    --epochs           N       Max epochs                  [40]
    --batch_size       N       Per-GPU batch size          [64]
    --lr               F       Peak learning rate          [1e-4]
    --grad_accum       N       Gradient accumulation steps [4]
    --no_amp                   Disable bfloat16 autocast
    --num_workers      N       DataLoader workers          [4]
    --log_every        N       Log every N optimizer steps [20]
    --save_every       N       Periodic checkpoint every N epochs [5]
    --compile                  torch.compile the model (20-30% speedup)
    --nccl                     Use NCCL backend for DDP (default: gloo)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hb_mamba_dataset  import build_dataloaders, GapMaskConfig
from training.trainer  import Trainer, TrainerConfig
# NOTE: HBMamba is imported inside main() after torch.cuda.set_device() so that
# mamba_ssm's Triton kernels initialise against the correct CUDA context.


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_ROOT = ROOT / "data_genration_and_raw_data" / "raw_data"
NS_PATH  = str(RAW_ROOT / "preprocessing" / "norm_stats" / "norm_stats.json")
IDX_DIR  = str(RAW_ROOT / "dataset_index")


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _is_ddp() -> bool:
    return int(os.environ.get("WORLD_SIZE", 1)) > 1

def _rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))

def _is_main() -> bool:
    return _rank() == 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train HB-Mamba v2.0")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume",         default=None,  help="Path to checkpoint to resume from")
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--batch_size",     type=int,   default=64,  help="Per-GPU batch size")
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--grad_accum",     type=int,   default=4)
    p.add_argument("--grad_clip",      type=float, default=2.0)
    p.add_argument("--skip_grad_threshold", type=float, default=100.0,
                   help="Skip optimizer step if pre-clip |grad| exceeds this")
    p.add_argument("--warmup_frac",    type=float, default=0.05)
    p.add_argument("--no_amp",         action="store_true", help="Disable bfloat16 autocast")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--log_every",      type=int,   default=20)
    p.add_argument("--save_every",     type=int,   default=5)
    p.add_argument("--compile",        action="store_true", help="torch.compile the model")
    p.add_argument("--nccl",           action="store_true", help="Use NCCL backend (default: gloo)")
    p.add_argument("--overfit",        action="store_true", help="Overfit on a single batch (sanity check)")
    p.add_argument("--overfit_steps",  type=int, default=500, help="Steps for --overfit mode")
    p.add_argument("--overfit_batch",  type=int, default=8,   help="Batch size for --overfit mode")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DataLoader builder (DDP-aware)
# ---------------------------------------------------------------------------

def _build_loaders(args, rank: int, world_size: int):
    """
    Build train / val DataLoaders.

    In DDP mode each rank gets a DistributedSampler so every GPU sees a
    non-overlapping shard of the dataset each epoch.
    """
    from hb_mamba_dataset import HBMambaDataset, hb_mamba_collate_fn

    gap_cfg = GapMaskConfig()

    loaders = {}
    splits  = {}

    # Discover available splits
    idx_dir = Path(IDX_DIR)
    for fp in sorted(idx_dir.glob("*_dataset_index.json")):
        split = fp.stem.replace("_dataset_index", "")
        splits[split] = fp

    for split, fp in splits.items():
        is_train = (split == "train")

        dataset = HBMambaDataset(
            dataset_index_path = str(fp),
            norm_stats_path    = NS_PATH,
            split              = split,
            gap_config         = gap_cfg,
            cache_macro        = True,
            cache_micro        = False,
        )

        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas = world_size,
                rank         = rank,
                shuffle      = is_train,
                drop_last    = is_train,
            )
            shuffle   = False
            drop_last = False
        else:
            sampler   = None
            shuffle   = is_train
            drop_last = is_train

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size  = args.batch_size,
            shuffle     = shuffle,
            drop_last   = drop_last,
            sampler     = sampler,
            collate_fn  = hb_mamba_collate_fn,
            num_workers = args.num_workers,
            pin_memory  = True,
        )
        loaders[split] = loader

    return loaders


# ---------------------------------------------------------------------------
# Overfit sanity check
# ---------------------------------------------------------------------------

def _overfit_loop(model, batch, device, steps: int, lr: float = 3e-4) -> None:
    """
    Train on one fixed batch for `steps` gradient steps.

    Overfit diagnoses the regression heads (L_mask, L_next) — the actual
    prediction signal.  L_align and L_contrast both need batch diversity
    (InfoNCE negatives / same-MMSI positives) and are pathological at B=8,
    so they are disabled here.  On real training with B=64 they re-enable.
    """
    # Move batch to device once
    fixed = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    # Unwrap DDP/compile wrappers if present
    inner = getattr(model, "module", model)
    loss_heads = inner.loss_heads

    # Save original weights, temporarily disable contrastive-style losses
    orig_w_align    = loss_heads.w_align
    orig_w_contrast = loss_heads.w_contrast
    loss_heads.w_align    = 0.0
    loss_heads.w_contrast = 0.0

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # Warmup: linear 0 → lr over first 10% of steps.
    # Deep Mamba stacks are prone to early-step gradient explosions without warmup.
    warmup_steps = max(20, steps // 10)
    def _lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        return lr

    model.train()

    print(f"\n{'='*78}")
    print(f"Overfit sanity check  ({steps} steps, batch={fixed['micro_tokens'].shape[0]}, "
          f"peak_lr={lr}, warmup={warmup_steps})")
    print(f"  L_align and L_contrast disabled (need B > ~32 to be meaningful)")
    print(f"{'='*78}")
    print(f"{'step':>6}  {'L_total':>9}  {'L_mask':>9}  {'L_next':>9}  "
          f"{'L_align*':>9}  {'|grad|':>9}")
    print(f"{'-'*68}")

    try:
        for step in range(1, steps + 1):
            # Apply LR schedule
            for pg in opt.param_groups:
                pg["lr"] = _lr_at(step - 1)

            opt.zero_grad()
            losses = model(
                fixed["macro_features"],
                fixed["macro_lat_idx"],
                fixed["macro_lon_idx"],
                fixed["micro_tokens"],
                fixed["mask"],
                fixed["padding_mask"],
                fixed["mmsi"],
                fixed.get("micro_tokens_masked"),
            )
            losses["L_total"].backward()
            # Moderate clip: prevents the rare Mamba gradient spike
            # from nuking the weights, without throttling normal updates.
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            # Skip the step if the pre-clip norm is still pathologically large
            if gn.item() > 100.0:
                opt.zero_grad()
                continue
            opt.step()

            if step % 20 == 0 or step == 1:
                print(
                    f"{step:>6}  "
                    f"{losses['L_total'].item():>9.5f}  "
                    f"{losses['L_mask'].item():>9.5f}  "
                    f"{losses['L_next'].item():>9.5f}  "
                    f"{losses['L_align'].item():>9.5f}  "
                    f"{gn.item():>9.3f}"
                )

        print(f"\nFinal L_mask+L_next = {losses['L_mask'].item() + 0.3*losses['L_next'].item():.6f}")
        if losses["L_mask"].item() < 0.005 and losses["L_next"].item() < 0.01:
            print("PASS — regression heads collapsed, model can overfit.")
        else:
            print("WARN — regression heads didn't collapse; see L_mask/L_next columns.")
        print("=" * 78)
    finally:
        # Restore original weights
        loss_heads.w_align    = orig_w_align
        loss_heads.w_contrast = orig_w_contrast


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────
    if _is_ddp():
        backend = "nccl" if args.nccl else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(_rank())
        if _is_main():
            print(f"  DDP backend : {backend}")

    device = torch.device(f"cuda:{_rank()}" if torch.cuda.is_available()
                          else "cpu")

    if _is_main():
        print("=" * 65)
        print("HB-Mamba v2.0 — Training")
        print(f"  GPUs        : {_world_size()}")
        print(f"  batch/GPU   : {args.batch_size}")
        print(f"  grad_accum  : {args.grad_accum}")
        print(f"  eff. batch  : {args.batch_size * args.grad_accum * _world_size()}")
        print(f"  epochs      : {args.epochs}")
        print(f"  lr          : {args.lr}")
        print(f"  amp         : {not args.no_amp}")
        print("=" * 65)

    # ── DataLoaders ───────────────────────────────────────────────────────
    loaders = _build_loaders(args, rank=_rank(), world_size=_world_size())

    if "train" not in loaders:
        print("ERROR: no train split found in dataset index dir.")
        sys.exit(1)

    train_loader = loaders["train"]
    val_loader   = loaders.get("val", loaders.get("test", loaders["train"]))

    if _is_main():
        print(f"\n  train batches : {len(train_loader)}")
        print(f"  val   batches : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    from model.hb_mamba import HBMamba   # deferred: Triton needs CUDA context first
    model = HBMamba.from_norm_stats(
        NS_PATH,
        d_model  = 512,   # up from 384 — break plateau
        n_layers = 8,     # up from 6
        d_state  = 128,   # up from 64 — more SSM capacity
        d_proj   = 256,   # up from 192 (scales with d_model: 512/2)
    ).to(device)

    if _is_main():
        print(f"\n{model}\n")

    if args.compile:
        if _is_main():
            print("  torch.compile enabled — first epoch will be slow (compilation).\n")
        model = torch.compile(model)

    if _is_ddp():
        model = DDP(model, device_ids=[_rank()], find_unused_parameters=False)

    # ── Overfit sanity check ──────────────────────────────────────────────
    if args.overfit:
        from hb_mamba_dataset import HBMambaDataset, hb_mamba_collate_fn
        import torch.utils.data as tud
        idx_dir = Path(IDX_DIR)
        train_fp = next(iter(sorted(idx_dir.glob("train_dataset_index.json"))), None) \
                   or next(iter(sorted(idx_dir.glob("*_dataset_index.json"))))
        ov_ds = HBMambaDataset(
            dataset_index_path = str(train_fp),
            norm_stats_path    = NS_PATH,
            split              = "train",
            gap_config         = GapMaskConfig(),
            cache_macro        = True,
            cache_micro        = False,
        )
        ov_loader = tud.DataLoader(
            ov_ds,
            batch_size  = args.overfit_batch,
            shuffle     = True,
            collate_fn  = hb_mamba_collate_fn,
            num_workers = 0,
        )
        ov_batch = next(iter(ov_loader))
        _overfit_loop(model, ov_batch, device, steps=args.overfit_steps, lr=args.lr)
        if _is_ddp():
            dist.destroy_process_group()
        return

    # ── Trainer ───────────────────────────────────────────────────────────
    cfg = TrainerConfig(
        checkpoint_dir = args.checkpoint_dir,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        max_epochs     = args.epochs,
        warmup_frac         = args.warmup_frac,
        grad_clip           = args.grad_clip,
        skip_grad_threshold = args.skip_grad_threshold,
        grad_accum          = args.grad_accum,
        use_amp        = not args.no_amp,
        log_every      = args.log_every,
        save_every     = args.save_every,
    )

    # Only rank 0 saves checkpoints — pass a dummy dir to other ranks
    if not _is_main():
        cfg.checkpoint_dir = "/tmp/hb_mamba_rank_discard"

    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = cfg,
        device       = device,
    )

    # ── Run ───────────────────────────────────────────────────────────────
    trainer.train(resume_from=args.resume)

    if _is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
