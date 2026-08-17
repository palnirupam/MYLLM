"""
myllm.training.pretraining.trainer — Production Pre-Training Engine.
Supports FP16 AMP + GradScaler on Tesla T4, token-budget stopping, gradient accumulation,
SafeTensors checkpointing, and DDP distributed data parallelism.
"""

import math
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW


class Trainer:
    def __init__(self, model, train_dataloader, config: dict, checkpoint_manager, device='cuda'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.device = device

        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 3e-4),
            weight_decay=self.config.get("weight_decay", 0.1),
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # 1. Exact Token Budget & Target Step Computation (Requirement 2 & 3)
        self.target_tokens = int(self.config.get("target_tokens", 100_000_000))
        self.micro_batch_per_gpu = int(self.config.get("micro_batch_per_gpu", self.config.get("batch_size", 2)))
        self.grad_accum_steps = int(self.config.get("gradient_accumulation_steps", 16))
        self.seq_len = int(self.model.config.max_seq_len if hasattr(self.model, "config") else 512)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = int(self.config.get("world_size", 1))
            self.rank = 0

        self.global_tokens_per_step = (
            self.world_size
            * self.micro_batch_per_gpu
            * self.seq_len
            * self.grad_accum_steps
        )
        self.target_steps = int(math.ceil(self.target_tokens / max(1, self.global_tokens_per_step)))
        self.warmup_steps = int(self.config.get("warmup_steps", min(200, max(5, int(0.065 * self.target_steps)))))

        if self.rank == 0:
            print(f"============================================================")
            print(f" DHRUVA TRAINING ENGINE: Stage 1A Pre-Training")
            print(f" Target Token Budget : {self.target_tokens:,} tokens (100M)")
            print(f" DDP World Size      : {self.world_size} GPU(s)")
            print(f" Micro-Batch / GPU   : {self.micro_batch_per_gpu} sequences")
            print(f" Max Sequence Length : {self.seq_len} tokens")
            print(f" Grad Accumulation   : {self.grad_accum_steps} sub-steps")
            print(f" Global Batch Tokens : {self.global_tokens_per_step:,} tokens/step")
            print(f" Exact Target Steps  : {self.target_steps:,} steps")
            print(f" Warmup Steps        : {self.warmup_steps} steps")
            print(f"============================================================")

        # 2. Calibrated LR Scheduler over exact target steps
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.target_steps - self.warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return max(0.1, 0.5 * (1.0 + math.cos(progress * math.pi)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # 3. Precision & GradScaler
        mixed_precision = self.config.get("mixed_precision", "fp16")
        self.is_cuda = torch.cuda.is_available() and "cuda" in str(self.device)
        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16

        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.dtype == torch.float16 and self.is_cuda))

        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.log_interval = int(self.config.get("log_interval", 10))
        self.checkpoint_interval = int(self.config.get("checkpoint_interval", 250))

    def train(
        self,
        num_steps: int = None,
        start_step: int = 0,
        initial_samples_consumed: int = 0,
        initial_tokens: int = 0,
    ) -> dict:
        self.model.train()
        step = start_step
        total_tokens = initial_tokens
        final_loss = 0.0
        samples_consumed = initial_samples_consumed

        # If num_steps not explicitly overridden (e.g. for smoke tests), use exact target_steps
        max_run_steps = num_steps if num_steps is not None else self.target_steps

        dataloader_iter = iter(self.train_dataloader)

        if not hasattr(self.model, 'config') or not hasattr(self.model.config, 'vocab_size'):
            raise ValueError("Model must have a .config attribute with .vocab_size for loss computation.")
        vocab_size = self.model.config.vocab_size

        start_time = time.time()

        while step < max_run_steps and total_tokens < self.target_tokens:
            step_loss = 0.0

            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(self.train_dataloader)
                    batch = next(dataloader_iter)

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                device_type = "cuda" if self.is_cuda else "cpu"
                with torch.amp.autocast(device_type=device_type, dtype=self.dtype, enabled=self.is_cuda):
                    logits, _ = self.model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, vocab_size),
                        labels.view(-1),
                        ignore_index=-100
                    )
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                tokens_in_substep = (labels != -100).sum().item()
                # Aggregate tokens across world size
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    t_tensor = torch.tensor(tokens_in_substep, device=self.device)
                    torch.distributed.all_reduce(t_tensor, op=torch.distributed.ReduceOp.SUM)
                    tokens_in_substep = int(t_tensor.item())

                total_tokens += tokens_in_substep
                samples_consumed += input_ids.shape[0] * self.world_size
                step_loss += loss.item() * self.grad_accum_steps

            final_loss = step_loss

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            step += 1

            if step % self.log_interval == 0 and self.rank == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / max(0.001, elapsed)
                lr = self.scheduler.get_last_lr()[0]
                mem = torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0.0
                pct = (total_tokens / self.target_tokens) * 100.0

                print(
                    f"Step {step:4d}/{self.target_steps:4d} ({pct:5.1f}%) | "
                    f"Loss {final_loss:.4f} | LR {lr:.2e} | "
                    f"Tok/s {tokens_per_sec:6.0f} | "
                    f"Tokens {total_tokens:,}/{self.target_tokens:,} | "
                    f"Mem {mem:.1f}GB",
                    flush=True,
                )

                if not math.isfinite(final_loss):
                    raise RuntimeError(f"Training aborted: loss={final_loss} is non-finite at step {step}.")

                # Write live progress artifact
                live_path = Path("artifacts/training_diagnostics/live_progress.json")
                live_path.parent.mkdir(parents=True, exist_ok=True)
                live_data = {
                    "step": step,
                    "target_steps": self.target_steps,
                    "target_tokens": self.target_tokens,
                    "total_tokens": total_tokens,
                    "percent_complete": round(pct, 2),
                    "loss": round(float(final_loss), 6),
                    "lr": round(float(lr), 8),
                    "tokens_per_sec": round(float(tokens_per_sec), 1),
                    "elapsed_sec": round(elapsed, 1),
                    "gpu_mem_gb": round(mem, 3),
                }
                live_path.write_text(json.dumps(live_data, indent=2))

            if step % self.checkpoint_interval == 0 and self.checkpoint_manager is not None:
                rng_states = {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                }
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=step,
                    epoch=0,
                    loss=final_loss,
                    config=self.model.config if hasattr(self.model, 'config') else None,
                    tokenizer_path="tokenizer",
                    rng_states=rng_states,
                    dataset_position=samples_consumed,
                )

        return {
            "final_loss": final_loss,
            "steps_completed": step,
            "total_tokens": total_tokens,
            "target_tokens": self.target_tokens,
            "target_steps": self.target_steps,
            "global_tokens_per_step": self.global_tokens_per_step,
        }
