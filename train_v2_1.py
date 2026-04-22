"""
train_v2_1.py — HB-Mamba v2.1  Training Entry Point
====================================================

Changes from v2.0:
  - micro_tokens_masked passed to encoder (gap zeroed, static features preserved)
  - micro_tokens (full) passed as loss target only
  - delta_t corrected at gap boundary
  - Intermediate Mamba streams kept separate (no bidirectional mixing until final layer)
  - ReconstructionHead output clamped to [0, 1]
  - NextPingHead has decoupling adapter, w_next reduced 0.8 -> 0.3
  - H computed from visible positions only

Single-GPU usage:
    conda run -n hb_mamba python train_v2_1.py

Multi-GPU DDP via gloo (2 GPUs):
    torchrun --nproc_per_node=2 train_v2_1.py

Key flags:
    --checkpoint_dir   DIR     Where to save checkpoints  [checkpoints_v2_1]
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
from typing import Dict

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
    p = argparse.ArgumentParser(description="Train HB-Mamba v2.1")
    p.add_argument("--checkpoint_dir", default="checkpoints_v2_1")
    p.add_argument("--resume",         default=None,  help="Path to checkpoint to resume from")
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--batch_size",     type=int,   default=64,  help="Per-GPU batch size")
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--grad_accum",     type=int,   default=4)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--warmup_frac",    type=float, default=0.05)
    p.add_argument("--no_amp",         action="store_true", help="Disable bfloat16 autocast")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--log_every",      type=int,   default=20)
    p.add_argument("--save_every",     type=int,   default=5)
    p.add_argument("--compile",        action="store_true", help="torch.compile the model")
    p.add_argument("--nccl",           action="store_true", help="Use NCCL backend (default: gloo)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# v2.1 Trainer subclass — passes micro_tokens_masked to model.forward()
# ---------------------------------------------------------------------------

class TrainerV21(Trainer):
    """
    Subclass of Trainer that passes micro_tokens_masked to the model.

    The base Trainer._forward() only passes micro_tokens (v2.0 interface).
    This override also passes micro_tokens_masked so the encoder sees
    the gap-zeroed input while the loss heads use the full original.
    """

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
                micro_tokens_masked=batch["micro_tokens_masked"],
            )


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
        print("HB-Mamba v2.1 — Training")
        print(f"  GPUs        : {_world_size()}")
        print(f"  batch/GPU   : {args.batch_size}")
        print(f"  grad_accum  : {args.grad_accum}")
        print(f"  eff. batch  : {args.batch_size * args.grad_accum * _world_size()}")
        print(f"  epochs      : {args.epochs}")
        print(f"  lr          : {args.lr}")
        print(f"  amp         : {not args.no_amp}")
        print(f"  w_next      : 0.3  (v2.1)")
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
    model = HBMamba.from_norm_stats(NS_PATH, w_next=0.3).to(device)

    if _is_main():
        print(f"\n{model}\n")

    if args.compile:
        if _is_main():
            print("  torch.compile enabled — first epoch will be slow (compilation).\n")
        model = torch.compile(model)

    if _is_ddp():
        model = DDP(model, device_ids=[_rank()], find_unused_parameters=False)

    # ── Trainer (v2.1 subclass) ───────────────────────────────────────────
    cfg = TrainerConfig(
        checkpoint_dir = args.checkpoint_dir,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        max_epochs     = args.epochs,
        warmup_frac    = args.warmup_frac,
        grad_clip      = args.grad_clip,
        grad_accum     = args.grad_accum,
        use_amp        = not args.no_amp,
        log_every      = args.log_every,
        save_every     = args.save_every,
    )

    # Only rank 0 saves checkpoints — pass a dummy dir to other ranks
    if not _is_main():
        cfg.checkpoint_dir = "/tmp/hb_mamba_rank_discard"

    trainer = TrainerV21(
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
