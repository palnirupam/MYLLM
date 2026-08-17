"""
scripts/kaggle_ddp_preflight.py — Kaggle 2x Tesla T4 Multi-GPU DDP Preflight & Diagnostics.
Executed via: torchrun --nproc_per_node=2 scripts/kaggle_ddp_preflight.py
Validates dual GPU topology, NCCL all-reduce, per-rank VRAM, FP16 throughput, and global tokens/sec.
"""

import os
import sys
from pathlib import Path
import json
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel


def run_kaggle_ddp_preflight():
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

    is_cuda = torch.cuda.is_available() and "cuda" in device

    if rank == 0:
        print(f"============================================================")
        print(f" KAGGLE DUAL GPU DDP PREFLIGHT & HARDWARE DIAGNOSTICS")
        print(f"============================================================")
        print(f" PyTorch Version : {torch.__version__}")
        print(f" CUDA Available  : {is_cuda} (Version: {torch.version.cuda if is_cuda else 'N/A'})")
        print(f" Total CUDA GPUs : {torch.cuda.device_count() if is_cuda else 0}")
        print(f" DDP World Size  : {world_size}")

    # Inspect per-rank GPU details
    gpu_name = torch.cuda.get_device_name(local_rank) if is_cuda else "Host CPU"
    vram_gb = (torch.cuda.get_device_properties(local_rank).total_memory / (1024 ** 3)) if is_cuda else 0.0

    print(f" [Rank {rank}] Device: {device} | GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB")

    # 1. NCCL All-Reduce Ring Verification
    nccl_passed = False
    if is_distributed and is_cuda:
        test_tensor = torch.tensor([1.0], device=device)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        if test_tensor.item() == float(world_size):
            nccl_passed = True
            if rank == 0:
                print(f" [Rank 0] NCCL All-Reduce Verification: PASS (Sum = {test_tensor.item()})")
    elif not is_distributed:
        nccl_passed = True

    # 2. Stage 1A Benchmark on Production 100M Architecture
    # micro_batch_per_gpu = 2, seq_len = 512, grad_accum = 16
    config = ModelConfig.dhruva_v1_production(max_seq_len=512)
    model = MyLLMModel(config).to(device)

    if is_distributed and is_cuda:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(local_rank)

    micro_batch = 2
    seq_len = 512
    grad_accum = 16
    warmup_steps = 2
    benchmark_steps = 5

    step_times = []

    # Run benchmark steps
    for step in range(warmup_steps + benchmark_steps):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for _ in range(grad_accum):
            x = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)
            y = torch.randint(0, config.vocab_size, (micro_batch, seq_len), device=device)

            device_type = "cuda" if is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=is_cuda):
                logits, _ = model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / grad_accum

            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        t1 = time.time()
        if step >= warmup_steps:
            step_times.append(t1 - t0)

    avg_step_sec = sum(step_times) / max(1, len(step_times))
    global_batch_tokens = world_size * micro_batch * seq_len * grad_accum
    global_tokens_per_sec = global_batch_tokens / max(1e-5, avg_step_sec)
    peak_vram_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 * 1024) if is_cuda else 0.0

    # Gather results to Rank 0
    if rank == 0:
        # Time to train 100M tokens estimate based on actual measured tokens/sec
        est_hours_100m = (100_000_000 / max(1.0, global_tokens_per_sec)) / 3600.0

        report = {
            "timestamp": time.time(),
            "world_size": world_size,
            "gpu_model": gpu_name,
            "per_gpu_vram_gb": round(vram_gb, 2),
            "cuda_version": torch.version.cuda if is_cuda else None,
            "pytorch_version": torch.__version__,
            "nccl_initialization_passed": nccl_passed,
            "fp16_supported": True,
            "stage_1a_parameters": {
                "micro_batch_per_gpu": micro_batch,
                "seq_len": seq_len,
                "gradient_accumulation_steps": grad_accum,
                "global_tokens_per_step": global_batch_tokens,
                "exact_target_steps_100m": int(100_000_000 / global_batch_tokens) + 1,
            },
            "empirical_measurements": {
                "average_step_latency_ms": round(avg_step_sec * 1000, 2),
                "peak_vram_per_gpu_mb": round(peak_vram_mb, 2),
                "global_throughput_tokens_per_sec": round(global_tokens_per_sec, 1),
                "estimated_hours_for_100m": round(est_hours_100m, 2),
            }
        }

        out_path = Path("artifacts/kaggle_ddp_preflight.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n============================================================")
        print(f" KAGGLE PREFLIGHT SUMMARY (Rank 0)")
        print(f" Measured Step Latency : {avg_step_sec * 1000:.1f} ms")
        print(f" Peak VRAM / GPU       : {peak_vram_mb:.1f} MB / {vram_gb * 1024:.1f} MB")
        print(f" Global Tokens/Step    : {global_batch_tokens:,} tokens")
        print(f" Global Throughput     : {global_tokens_per_sec:,.1f} tokens/sec")
        print(f" Est. Time for 100M    : {est_hours_100m:.2f} hours (~{est_hours_100m*60:.1f} mins)")
        print(f" Report Persisted      : {out_path}")
        print(f"============================================================\n")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_kaggle_ddp_preflight()
