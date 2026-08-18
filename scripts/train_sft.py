#!/usr/bin/env python3
"""MyLLM V0 SFT Training Script"""

import os
import yaml
import argparse
import time
import torch
from myllm.utils.runtime_guard import assert_training_environment
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
import shutil

from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel
from myllm.core.tokenizer.bpe import BPETokenizer
from safetensors.torch import load_model, save_model
from myllm.training.sft.dataset import get_sft_dataloaders
from myllm.training.checkpoint.manager import CheckpointManager

def evaluate_validation(model, val_loader, device, vocab_size, max_batches=20):
    """Compute average validation loss on response tokens."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.inference_mode():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids)
            # labels are unshifted in dataset.py, so we shift logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, vocab_size), 
                shift_labels.view(-1), 
                ignore_index=-100
            )
            
            total_loss += loss.item()
            total_batches += 1
            
    model.train()
    return total_loss / max(1, total_batches)

def main():
    assert_training_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    base_model_path = config_dict["training"]["base_model_path"]
    output_dir = Path(config_dict["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = BPETokenizer.load(os.path.join(base_model_path, "tokenizer"))
    # Save a copy to SFT output
    tokenizer.save(str(output_dir / "tokenizer"))
    
    # Setup data
    batch_size = config_dict["training"].get("batch_size", 2)
    max_seq_len = config_dict["model"].get("max_seq_len", 512)
    train_loader, val_loader = get_sft_dataloaders(tokenizer, max_seq_len, batch_size)
    
    # Load model
    model_config = ModelConfig.load(os.path.join(base_model_path, "config.json"))
    model_config.save(str(output_dir / "config.json"))
    
    model = MyLLMModel(model_config)
    print("Loading pretrained weights...")
    load_model(model, os.path.join(base_model_path, "model.safetensors"))
    model.to(device)
    
    # Setup Optimizer
    learning_rate = float(config_dict["training"].get("learning_rate", 2e-5))
    weight_decay = float(config_dict["training"].get("weight_decay", 0.01))
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    max_steps = config_dict["training"].get("max_steps", 500)
    warmup_steps = config_dict["training"].get("warmup_steps", 50)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress) * 3.1415926535)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    mixed_precision = config_dict["training"].get("mixed_precision", "fp16")
    dtype = torch.bfloat16 if mixed_precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    grad_accum_steps = config_dict["training"].get("gradient_accumulation_steps", 16)
    max_grad_norm = float(config_dict["training"].get("max_grad_norm", 1.0))
    eval_steps = config_dict["training"].get("eval_steps", 100)
    
    print("\nStarting SFT Training...")
    model.train()
    step = 0
    best_val_loss = float('inf')
    total_tokens = 0
    start_time = time.time()
    
    dataloader_iter = iter(train_loader)
    vocab_size = model_config.vocab_size
    
    while step < max_steps:
        for i in range(grad_accum_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_loader)
                batch = next(dataloader_iter)
                
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast('cuda', dtype=dtype):
                logits, _ = model(input_ids)
                
                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab_size), 
                    shift_labels.view(-1), 
                    ignore_index=-100
                )
                loss = loss / grad_accum_steps
                
            scaler.scale(loss).backward()
            total_tokens += (shift_labels != -100).sum().item()
            final_loss = loss.item() * grad_accum_steps
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        
        step += 1
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            lr = scheduler.get_last_lr()[0]
            mem = torch.cuda.memory_allocated(device) / 1024**3
            print(f"Step {step:4d} | Train Loss {final_loss:.4f} | LR {lr:.2e} | Tok/s {tokens_per_sec:.0f} | Mem {mem:.1f}GB")
            
        if step % eval_steps == 0:
            val_loss = evaluate_validation(model, val_loader, device, vocab_size)
            print(f">>> Validation Loss at step {step}: {val_loss:.4f}")
            
            # Save if best
            if val_loss < best_val_loss:
                print(f"    New best validation loss! Saving to best_model...")
                best_val_loss = val_loss
                
                best_model_dir = output_dir / "best_model"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                
                save_model(model, str(best_model_dir / "model.safetensors"))
                model_config.save(str(best_model_dir / "config.json"))
                tokenizer.save(str(best_model_dir / "tokenizer"))

    print("\n--- SFT Training Complete ---")
    print(f"Final Train Loss: {final_loss:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    
    # Ensure final best model is copied to final_model
    best_model_dir = output_dir / "best_model"
    final_model_dir = output_dir / "final_model"
    if best_model_dir.exists():
        if final_model_dir.exists():
            shutil.rmtree(final_model_dir)
        shutil.copytree(best_model_dir, final_model_dir)
        print(f"Best model copied to {final_model_dir}")

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
