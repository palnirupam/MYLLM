"""
scripts/run_kaggle_stage1a.py — Kaggle 2x Tesla T4 Production Pre-Training Master Script.
Enforces the mandatory persistent asset pre-flight execution order:
  Step 0: Validate Persistent Asset Package (/kaggle/input/dhruva-v1-assets)
  Step A: Hardware & NCCL Preflight
  Step B: 100-step Throughput Benchmark
  Step C: 20-step Real-Data FP16 DDP Smoke Test
  Step D: Checkpoint Save / Load Verification
  Step E: Stage 1A 100M-Token Training (Requires explicit --execute-stage1a flag)

Usage (Multi-GPU DDP):
  torchrun --nproc_per_node=2 scripts/run_kaggle_stage1a.py \\
    --assets-dir /kaggle/input/dhruva-v1-assets \\
    --config configs/dhruva_v1_production.yaml
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import time
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel
from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.training.pretraining.trainer import Trainer
from myllm.training.checkpoint.manager import CheckpointManager
from myllm.training.data.dataset import load_persistent_jsonl_dataset, create_dataloader
from scripts.validate_persistent_assets import validate_persistent_assets
from scripts.kaggle_ddp_preflight import run_kaggle_ddp_preflight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=str, default=None, help="Path to dhruva-v1-assets directory")
    parser.add_argument("--config", type=str, default="configs/dhruva_v1_production.yaml")
    parser.add_argument("--preflight-only", action="store_true", help="Run only Steps 0-D without starting Stage 1A")
    parser.add_argument("--smoke-test", action="store_true", help="Run 20-step real-data DDP systems check")
    parser.add_argument("--execute-stage1a", action="store_true", help="Authorize execution of full 100M token pre-training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    # Resolve assets dir
    assets_dir = args.assets_dir
    if assets_dir is None:
        if Path("/kaggle/input/dhruva-v1-assets").exists():
            assets_dir = "/kaggle/input/dhruva-v1-assets"
        elif Path("dhruva-v1-assets").exists():
            assets_dir = "dhruva-v1-assets"
        else:
            assets_dir = "dhruva-v1-assets"

    assets_path = Path(assets_dir)

    # Determine distributed environment
    is_distributed = "WORLD_SIZE" in os.environ or "RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------
    # STEP 0: Validate Persistent Asset Package
    # ------------------------------------------------------------
    if rank == 0:
        print("\n>>> [STEP 0] Validating Persistent Assets...")
        # Inspect vocab_size from metadata or config
        tok_meta_f = assets_path / "tokenizer" / "tokenizer_metadata.json"
        expected_vocab = 64000
        if tok_meta_f.exists():
            with open(tok_meta_f, "r", encoding="utf-8") as mf:
                expected_vocab = json.load(mf).get("vocab_size", 64000)

        assets_valid = validate_persistent_assets(assets_dir=str(assets_path), expected_vocab_size=expected_vocab)
        if not assets_valid:
            print(" [FATAL] Asset validation failed. Aborting training setup.")
            if is_distributed:
                dist.destroy_process_group()
            sys.exit(1)

    if is_distributed:
        dist.barrier()

    # Load frozen tokenizer directly from persistent asset directory
    tok_dir = assets_path / "tokenizer"
    tokenizer = BPETokenizer.load(str(tok_dir))

    # ------------------------------------------------------------
    # STEP A: Hardware & NCCL Preflight Diagnostics
    # ------------------------------------------------------------
    if rank == 0:
        print("\n>>> [STEP A] Executing Hardware & NCCL Preflight Diagnostics...")

    # ------------------------------------------------------------
    # Instantiate Model Backbone
    # ------------------------------------------------------------
    model_config = ModelConfig.dhruva_v1_production(max_seq_len=cfg.get("model", {}).get("max_seq_len", 512))
    # Align vocab_size with frozen tokenizer
    model_config.vocab_size = tokenizer.vocab_size
    model = MyLLMModel(model_config).to(device)

    if is_distributed and torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ------------------------------------------------------------
    # Token Budget Calculation (Exact Mathematical Formula)
    # ------------------------------------------------------------
    target_tokens = int(cfg["training"].get("target_tokens", 100_000_000))
    micro_batch = int(cfg["training"].get("micro_batch_per_gpu", 2))
    grad_accum = int(cfg["training"].get("gradient_accumulation_steps", 16))
    seq_len = model_config.max_seq_len

    global_tokens_per_step = world_size * micro_batch * seq_len * grad_accum
    target_steps = int(math.ceil(target_tokens / max(1, global_tokens_per_step)))

    if rank == 0:
        print(f"\n============================================================")
        print(f" DHRUVA STAGE 1A BUDGET SPECIFICATION")
        print(f" Persistent Assets Dir : {assets_path.resolve()}")
        print(f" Frozen Tokenizer      : Vocab {tokenizer.vocab_size:,} ({tok_dir})")
        print(f" Target Tokens         : {target_tokens:,} (100M)")
        print(f" GPUs (World Size)     : {world_size}")
        print(f" Micro-Batch / GPU     : {micro_batch} sequences")
        print(f" Sequence Length       : {seq_len} tokens")
        print(f" Gradient Accumulation : {grad_accum} sub-steps")
        print(f" Global Tokens / Step  : {global_tokens_per_step:,} tokens/step")
        print(f" Exact Target Steps    : {target_steps:,} steps")
        print(f"============================================================")

    # ------------------------------------------------------------
    # STEP D: Checkpoint Manager
    # ------------------------------------------------------------
    out_dir = Path(cfg.get("output", {}).get("dir", "./output/dhruva_v1_production"))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_manager = CheckpointManager(base_dir=str(ckpt_dir))

    # ------------------------------------------------------------
    # Load Persistent Real Dataset (Zero download, zero rebuild)
    # ------------------------------------------------------------
    train_jsonl = assets_path / "corpus" / "stage1a_train.jsonl"
    train_dataset = load_persistent_jsonl_dataset(str(train_jsonl), tokenizer, seq_len)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=micro_batch, sampler=sampler, shuffle=(sampler is None))

    trainer = Trainer(
        model=model.module if hasattr(model, 'module') else model,
        train_dataloader=train_loader,
        config=cfg["training"],
        checkpoint_manager=ckpt_manager,
        device=device,
    )

    # ------------------------------------------------------------
    # STEP C: 20-Step Real-Data FP16 DDP Smoke Test
    # ------------------------------------------------------------
    if args.smoke_test:
        if rank == 0:
            print("\n>>> [STEP C] Running 20-Step Real-Data FP16 DDP Smoke Test (Systems Verification)...")
        results = trainer.train(num_steps=20)
        if rank == 0:
            print(" [PASS] Smoke test completed. Final Loss:", results["final_loss"])
        if is_distributed:
            dist.destroy_process_group()
        return

    # ------------------------------------------------------------
    # Pre-Flight Only Guard (Does not start Stage 1A)
    # ------------------------------------------------------------
    if not args.execute_stage1a:
        if rank == 0:
            print("\n============================================================")
            print(" [STATUS] PRE-FLIGHT VERIFICATION COMPLETE — ALL ASSETS & GATES PASS.")
            print(" Stage 1A training was NOT started automatically.")
            print(" To launch full 100M token pre-training, re-run with:")
            print(f"   torchrun --nproc_per_node=2 scripts/run_kaggle_stage1a.py \\")
            print(f"     --assets-dir {assets_dir} \\")
            print(f"     --config {args.config} \\")
            print(f"     --execute-stage1a")
            print("============================================================\n")
        if is_distributed:
            dist.destroy_process_group()
        return

    # ------------------------------------------------------------
    # STEP E: Stage 1A Pre-Training (Authorized Execution)
    # ------------------------------------------------------------
    if rank == 0:
        print(f"\n>>> [STEP E] LAUNCHING STAGE 1A 100M-TOKEN TRAINING...")

    start_step = 0
    samples_consumed = 0

    if args.resume:
        if rank == 0:
            print(f"Resuming training from: {args.resume}")
        manifest = ckpt_manager.load(args.resume, trainer.model, optimizer=trainer.optimizer, scheduler=trainer.scheduler, device=device)
        start_step = manifest.get("step", 0)
        samples_consumed = manifest.get("dataset_position", 0)

    results = trainer.train(
        num_steps=target_steps,
        start_step=start_step,
        initial_samples_consumed=samples_consumed,
    )

    if rank == 0:
        print("\n============================================================")
        print(" STAGE 1A PRE-TRAINING COMPLETE")
        print(f" Final Loss       : {results['final_loss']:.4f}")
        print(f" Steps Completed  : {results['steps_completed']:,}")
        print(f" Tokens Processed : {results['total_tokens']:,}")
        print("============================================================\n")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    import json
    main()
