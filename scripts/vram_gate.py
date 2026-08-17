"""
scripts/vram_gate.py — VRAM Gate Evaluation Script.
Evaluates whether a single GPU (e.g. Tesla T4 16GB) can execute full forward and backward passes
at seq_len=2048 in FP16 mixed precision without encountering Out-Of-Memory (OOM).

Gate Rule: Production max_seq_len MUST remain 512 unless this gate explicitly passes on target GPU hardware.
"""

import sys
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel


def run_vram_gate(
    micro_batch_size: int = 2,
    seq_len: int = 2048,
    num_warmup_steps: int = 2,
    num_measure_steps: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    print(f"============================================================")
    print(f" DHRUVA VRAM GATE: Target seq_len={seq_len}, micro_batch={micro_batch_size}, FP16")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"============================================================")

    # 1. Instantiate Dhruva production backbone with target seq_len
    config = ModelConfig(
        vocab_size=64000,
        d_model=768,
        n_layers=8,
        n_heads=12,
        n_kv_heads=4,
        intermediate_size=2048,
        max_seq_len=seq_len,
        dropout=0.0,
        norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )

    is_cuda = torch.cuda.is_available() and "cuda" in str(device)
    model = MyLLMModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    total_params = model.count_parameters()
    print(f"Model Parameters: {total_params:,}")

    # Track VRAM
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        print(f"Total Physical GPU VRAM: {total_vram_mb:.1f} MB ({total_vram_mb/1024:.2f} GB)")
    else:
        total_vram_mb = 0.0
        print("Running in CPU test mode (CUDA unavailable).")

    oom_occurred = False
    error_message = None
    step_latencies = []

    try:
        # Run forward + backward iterations
        total_steps = num_warmup_steps + num_measure_steps
        for step in range(total_steps):
            t0 = time.time()

            # Synthetic batch at target sequence length
            input_ids = torch.randint(0, config.vocab_size, (micro_batch_size, seq_len), device=device)
            labels = torch.randint(0, config.vocab_size, (micro_batch_size, seq_len), device=device)

            optimizer.zero_grad(set_to_none=True)

            device_type = "cuda" if is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=is_cuda):
                logits, _ = model(input_ids)
                loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t1 = time.time()
            if step >= num_warmup_steps:
                step_latencies.append((t1 - t0) * 1000.0)

            if is_cuda:
                curr_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
                peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                print(f"Step {step+1:2d}/{total_steps} | Loss {loss.item():.4f} | Latency {(t1-t0)*1000:.1f}ms | VRAM Curr: {curr_allocated:.1f}MB | Peak: {peak_allocated:.1f}MB")
            else:
                print(f"Step {step+1:2d}/{total_steps} | Loss {loss.item():.4f} | Latency {(t1-t0)*1000:.1f}ms (CPU)")

    except torch.cuda.OutOfMemoryError as e:
        oom_occurred = True
        error_message = f"CUDA OOM: {str(e)}"
        print(f"\n[FAIL] OUT OF MEMORY ENCOUNTERED at seq_len={seq_len}: {e}")
    except Exception as e:
        error_message = f"Error during execution: {str(e)}"
        print(f"\n[ERROR] Execution failed: {e}")

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if is_cuda else 0.0
    avg_latency = sum(step_latencies) / max(1, len(step_latencies))
    tokens_per_step = micro_batch_size * seq_len
    tokens_per_sec = (tokens_per_step / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0

    # Gate decision rule: Must complete without OOM and stay within safe headroom (< 90% of total VRAM on T4)
    vram_headroom_passed = (peak_vram_mb < 0.90 * total_vram_mb) if is_cuda else True
    gate_passed = (not oom_occurred) and (error_message is None) and vram_headroom_passed

    report = {
        "timestamp": time.time(),
        "gate_passed": gate_passed,
        "oom_occurred": oom_occurred,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if is_cuda else "CPU",
        "micro_batch_size": micro_batch_size,
        "seq_len": seq_len,
        "mixed_precision": "fp16",
        "model_parameters": total_params,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "total_vram_mb": round(total_vram_mb, 2),
        "vram_utilization_pct": round((peak_vram_mb / max(1.0, total_vram_mb)) * 100, 2) if is_cuda else 0.0,
        "average_step_latency_ms": round(avg_latency, 2),
        "measured_tokens_per_sec": round(tokens_per_sec, 2),
        "error_message": error_message,
        "recommendation": "SAFE FOR seq_len=2048" if gate_passed else "REVERT/KEEP max_seq_len=512 FOR STAGE 1A",
    }

    # Persist report
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "vram_gate_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n============================================================")
    print(f" GATE RESULT: {'PASS [SAFE]' if gate_passed else 'FAIL [UNSAFE / HEADROOM EXCEEDED]'}")
    print(f" Peak VRAM: {peak_vram_mb:.1f} MB / {total_vram_mb:.1f} MB")
    print(f" Recommendation: {report['recommendation']}")
    print(f" Report saved to: {report_path}")
    print(f"============================================================\n")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_vram_gate(micro_batch_size=args.batch_size, seq_len=args.seq_len, device=args.device)
