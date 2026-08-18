"""
scripts/run_sft_pilot.py — Stage 2A SFT Pilot Training Runner for Dhruva V1.
Universal compatibility: runs on Kaggle GPU (T4/P100), Google Colab, Linux, and Windows.
Strictly preserves the base model checkpoint intact.
Saves the new instruct checkpoint to the specified output directory.
"""

import sys
import math
import time
import json
import random
import shutil
import hashlib
import argparse
import torch
from myllm.utils.runtime_guard import assert_training_environment
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from safetensors.torch import save_file

# Resolve repository root dynamically relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.runtime.local.inference import LocalInferenceRuntime


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_seq_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        system_content = "You are Dhruva, a helpful and concise multilingual AI assistant."
        user_content = ""
        assistant_content = ""
        for turn in item["conversations"]:
            if turn["role"] == "user":
                user_content = turn["content"]
            elif turn["role"] == "assistant":
                assistant_content = turn["content"]

        prompt_text = f"<bos>[SYSTEM]\n{system_content}\n\n[USER]\n{user_content}\n\n[ASSISTANT]\n"
        full_text = f"{prompt_text}{assistant_content}<eos>"

        p_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        f_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(f_ids) > self.max_seq_len:
            f_ids = f_ids[:self.max_seq_len]

        p_len = min(len(p_ids), len(f_ids))
        inp = torch.tensor(f_ids, dtype=torch.long)
        lbl = inp.clone()
        lbl[:p_len] = -100  # Mask prompt tokens

        return {"input_ids": inp, "labels": lbl}


def sft_collate_fn(batch, pad_token_id=0):
    max_len = max(len(x["input_ids"]) for x in batch)
    b_size = len(batch)

    padded_inputs = torch.full((b_size, max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((b_size, max_len), -100, dtype=torch.long)

    for i, x in enumerate(batch):
        seq_len = len(x["input_ids"])
        padded_inputs[i, :seq_len] = x["input_ids"]
        padded_labels[i, :seq_len] = x["labels"]

    return {"input_ids": padded_inputs, "labels": padded_labels}


def parse_args():
    parser = argparse.ArgumentParser(description="Dhruva V1 Stage 2A SFT Pilot Training Runner")
    parser.add_argument("--base_model_path", type=str, default=str(REPO_ROOT / "releases/dhruva-v1-100m/inference_model"),
                        help="Path to the frozen base model directory")
    parser.add_argument("--data_path", type=str, default=str(REPO_ROOT / "dhruva-v1-assets/sft/stage2a_pilot_5k.jsonl"),
                        help="Path to the 5K SFT dataset JSONL file")
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "releases/dhruva-v1-100m-instruct-pilot-sft5k"),
                        help="Path to save the trained SFT model checkpoint")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=2.5e-6, help="Minimum learning rate for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of linear warmup steps")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro-batch size per GPU step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=20260818, help="Random seed for reproducibility")
    parser.add_argument("--val_interval_steps", type=int, default=50, help="Validation logging step interval")
    return parser.parse_args()


def main():
    assert_training_environment()
    args = parse_args()

    print("================================================================================")
    print(" DHRUVA V1 — STAGE 2A SFT PILOT TRAINING LAUNCH")
    print("================================================================================\n")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_dir = Path(args.base_model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_file = Path(args.data_path).resolve()

    if not base_dir.exists():
        print(f"[!] ERROR: Base model directory does not exist: {base_dir}")
        sys.exit(1)

    if not dataset_file.exists():
        print(f"[!] ERROR: Dataset file does not exist: {dataset_file}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base Model Hash Check (Before Training)
    base_model_file = base_dir / "model.safetensors"
    base_hash_before = compute_sha256(base_model_file)
    print(f"[*] Base Checkpoint SHA256 (Before Training): {base_hash_before}")

    # 2. Print Configuration
    print("\n--- FINAL TRAINING CONFIGURATION ---")
    print(f"  {'learning_rate':<28}: {args.learning_rate}")
    print(f"  {'min_lr':<28}: {args.min_lr}")
    print(f"  {'weight_decay':<28}: {args.weight_decay}")
    print(f"  {'betas':<28}: (0.9, 0.95)")
    print(f"  {'eps':<28}: 1e-08")
    print(f"  {'warmup_steps':<28}: {args.warmup_steps}")
    print(f"  {'epochs':<28}: {args.epochs}")
    print(f"  {'micro_batch_size':<28}: {args.micro_batch_size}")
    print(f"  {'gradient_accumulation_steps':<28}: {args.gradient_accumulation_steps}")
    print(f"  {'effective_batch_size':<28}: {args.micro_batch_size * args.gradient_accumulation_steps} samples")
    print(f"  {'max_seq_len':<28}: {args.max_seq_len}")
    print(f"  {'grad_clip':<28}: {args.grad_clip}")
    print(f"  {'seed':<28}: {args.seed}")
    print(f"  {'mixed_precision':<28}: fp16 (AMP)")
    print(f"  {'base_model_source':<28}: {base_dir}")
    print(f"  {'checkpoint_output':<28}: {output_dir}")
    print("------------------------------------\n")

    # 3. Load Tokenizer & Dataset
    tokenizer = BPETokenizer.load(base_dir / "tokenizer")
    with open(dataset_file, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    random.shuffle(all_samples)
    val_size = int(len(all_samples) * 0.05)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]

    print(f"[*] Dataset Loaded: {len(all_samples):,} total samples")
    print(f"[*] Train Split   : {len(train_samples):,} samples")
    print(f"[*] Val Split     : {len(val_samples):,} samples")

    train_ds = SFTDataset(train_samples, tokenizer, args.max_seq_len)
    val_ds = SFTDataset(val_samples, tokenizer, args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True, collate_fn=sft_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False, collate_fn=sft_collate_fn)

    # 4. Load Base Model
    runtime = LocalInferenceRuntime(model_path=str(base_dir))
    model = runtime.model
    device = runtime.device
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    use_cuda = (str(device) == "cuda" or (hasattr(device, "type") and device.type == "cuda"))
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    total_micro_batches = len(train_loader) * args.epochs
    total_opt_steps = math.ceil(total_micro_batches / args.gradient_accumulation_steps)

    def get_lr(step):
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
        decay_ratio = (step - args.warmup_steps) / max(1, total_opt_steps - args.warmup_steps)
        decay_ratio = min(1.0, max(0.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    # 5. Training Loop
    print("\n" + "=" * 80)
    print(f" STARTING SFT TRAINING (Total Steps: {total_opt_steps}, Epochs: {args.epochs})")
    print(f" Execution Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    start_time = time.time()
    global_step = 0
    micro_step = 0
    tokens_processed = 0
    running_loss = 0.0

    model.train()
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        print(f"\n>>> Epoch {epoch} / {args.epochs} Started <<<")
        for batch_idx, batch in enumerate(train_loader, 1):
            micro_step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            tok_count = (labels != -100).sum().item()
            tokens_processed += tok_count

            with torch.amp.autocast('cuda', enabled=use_cuda):
                out = model(input_ids)
                logits = out[0] if isinstance(out, tuple) else out
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, runtime.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss_scaled = loss / args.gradient_accumulation_steps

            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f"[!] CRITICAL ABORT: Non-finite loss ({loss_val}) at micro-step {micro_step}!")
                sys.exit(1)

            scaler.scale(loss_scaled).backward()
            running_loss += loss_val

            if micro_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()

                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    print(f"[!] CRITICAL ABORT: Non-finite grad norm ({grad_norm}) at step {global_step}!")
                    sys.exit(1)

                curr_lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # Periodic Logging
                if global_step % 20 == 0 or global_step == total_opt_steps:
                    elapsed = time.time() - start_time
                    tok_per_sec = tokens_processed / max(0.001, elapsed)
                    vram_mb = torch.cuda.memory_allocated() / (1024**2) if use_cuda else 0
                    print(f"  Step {global_step:>4}/{total_opt_steps:<4} | Epoch {epoch} | Loss: {loss_val:.4f} | LR: {curr_lr:.2e} | GradNorm: {grad_norm:.3f} | {tok_per_sec:.0f} tok/s | VRAM: {vram_mb:.0f} MB")
                    running_loss = 0.0

                # Validation Loop
                if global_step % args.val_interval_steps == 0 or global_step == total_opt_steps:
                    model.eval()
                    val_loss_sum = 0.0
                    val_tok_count = 0
                    with torch.no_grad():
                        for v_batch in val_loader:
                            v_inp = v_batch["input_ids"].to(device)
                            v_lbl = v_batch["labels"].to(device)
                            with torch.amp.autocast('cuda', enabled=use_cuda):
                                v_out = model(v_inp)
                                v_logits = v_out[0] if isinstance(v_out, tuple) else v_out
                                v_s_logits = v_logits[:, :-1, :].contiguous()
                                v_s_lbl = v_lbl[:, 1:].contiguous()
                                v_loss = F.cross_entropy(
                                    v_s_logits.view(-1, runtime.config.vocab_size),
                                    v_s_lbl.view(-1),
                                    ignore_index=-100,
                                    reduction="sum"
                                )
                            val_loss_sum += v_loss.item()
                            val_tok_count += (v_s_lbl != -100).sum().item()

                    mean_val_loss = val_loss_sum / max(1, val_tok_count)
                    val_ppl = math.exp(min(20.0, mean_val_loss))
                    print(f"  >>> [EVALUATION @ Step {global_step}] Validation Loss: {mean_val_loss:.4f} | Val PPL: {val_ppl:.2f} <<<")
                    model.train()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" TRAINING COMPLETE in {total_time:.2f}s ({total_time/60.0:.2f} mins)")
    print(f" Total Tokens Processed: {tokens_processed:,}")
    print("=" * 80)

    # 6. Save SFT Checkpoint
    print(f"\n[*] Saving SFT Model to: {output_dir}")
    model.eval()

    state_dict = model.state_dict()
    save_file(state_dict, str(output_dir / "model.safetensors"))

    shutil.copy2(base_dir / "config.json", output_dir / "config.json")

    out_tok_dir = output_dir / "tokenizer"
    if out_tok_dir.exists():
        shutil.rmtree(out_tok_dir)
    shutil.copytree(base_dir / "tokenizer", out_tok_dir)

    # 7. Post-Training Base Checkpoint Invariance Verification
    base_hash_after = compute_sha256(base_model_file)
    sft_hash = compute_sha256(output_dir / "model.safetensors")

    print(f"[*] Base Checkpoint SHA256 (After Training) : {base_hash_after}")
    print(f"[*] Base Checkpoint Unchanged Check        : {base_hash_before == base_hash_after} (MUST BE TRUE)")
    print(f"[*] New SFT Checkpoint SHA256              : {sft_hash}")
    print(f"[*] SFT Checkpoint Size                    : {(output_dir / 'model.safetensors').stat().st_size:,} bytes")
    print("\n[+] SFT Checkpoint Saved Successfully.")


if __name__ == "__main__":
    main()
