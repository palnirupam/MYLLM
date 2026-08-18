"""FP16/DDP pretraining loop with explicit loss-token accounting."""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW


class Trainer:
    # The scientific budget is cumulative non-ignored next-token labels.
    CANONICAL_TOKEN_COUNTER = "cumulative_loss_tokens"

    def __init__(self, model, train_dataloader, config: dict, checkpoint_manager, device="cuda", validation_dataloader=None):
        self.model = model
        self.base_model = model.module if hasattr(model, "module") else model
        self.train_dataloader = train_dataloader
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 3e-4),
            weight_decay=self.config.get("weight_decay", 0.1),
            betas=(0.9, 0.95), eps=1e-8,
        )
        self.target_tokens = int(self.config.get("target_tokens", 100_000_000))
        self.micro_batch_per_gpu = int(self.config.get("micro_batch_per_gpu", self.config.get("batch_size", 2)))
        self.grad_accum_steps = int(self.config.get("gradient_accumulation_steps", 16))
        model_config = getattr(self.base_model, "config", None)
        self.seq_len = int(self.config.get("_seq_len_override", model_config.max_seq_len if model_config is not None else 512))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = int(self.config.get("world_size", 1))
            self.rank = 0
        self.global_tokens_per_step = self.world_size * self.micro_batch_per_gpu * self.seq_len * self.grad_accum_steps
        self.target_steps = int(math.ceil(self.target_tokens / max(1, self.global_tokens_per_step)))
        self.warmup_steps = int(self.config.get("warmup_steps", min(200, max(5, int(0.065 * self.target_steps)))))

        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = (current_step - self.warmup_steps) / float(max(1, self.target_steps - self.warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress)))))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        mixed_precision = self.config.get("mixed_precision", "fp16")
        self.is_cuda = torch.cuda.is_available() and "cuda" in str(self.device)
        self.dtype = torch.bfloat16 if mixed_precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.dtype == torch.float16 and self.is_cuda))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.log_interval = int(self.config.get("log_interval", 10))
        self.checkpoint_interval = int(self.config.get("checkpoint_interval", 250))
        self.validation_interval = int(self.config.get("validation_interval", 250))

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> dict:
        if self.validation_dataloader is None:
            raise RuntimeError("validation dataloader is not configured")
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_tokens = torch.tensor(0, device=self.device, dtype=torch.long)
        vocab_size = self.base_model.config.vocab_size
        for index, batch in enumerate(self.validation_dataloader):
            if max_batches is not None and index >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda" if self.is_cuda else "cpu", dtype=self.dtype, enabled=self.is_cuda):
                logits, _ = self.model(input_ids)
                loss_sum = nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100, reduction="sum"
                )
            total_loss += loss_sum.float()
            total_tokens += (labels != -100).sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(total_loss)
            torch.distributed.all_reduce(total_tokens)
        self.model.train()
        if int(total_tokens.item()) == 0:
            raise RuntimeError("validation stream has no loss-bearing tokens")
        return {"validation_loss": float((total_loss / total_tokens).item()), "validation_tokens": int(total_tokens.item())}

    def _global_label_count(self, labels: torch.Tensor) -> int:
        count = int((labels != -100).sum().item())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor = torch.tensor(count, device=self.device, dtype=torch.long)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            count = int(tensor.item())
        return count

    def _rank_rng_state(self) -> dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if self.is_cuda else None,
        }

    def _all_rank_rng_states(self) -> dict[str, dict]:
        local = self._rank_rng_state()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(self.world_size)]
            torch.distributed.all_gather_object(gathered, local)
            return {str(index): state for index, state in enumerate(gathered)}
        return {"0": local}

    def _save_checkpoint(self, step, loss, source_tokens, packed_tokens, total_tokens, samples_consumed, identities):
        rank_rng = self._all_rank_rng_states()
        if self.rank == 0:
            training_state = {
                "global_step": step, "optimizer_step": step,
                "micro_step": step * self.grad_accum_steps, "accumulation_step": 0,
                "cumulative_source_tokens": source_tokens,
                "cumulative_packed_tokens": packed_tokens,
                "cumulative_loss_tokens": total_tokens,
                "dataset_cursor": samples_consumed, "samples_consumed": samples_consumed,
                "sampler_epoch": 0, "sampler_cursor": samples_consumed,
            }
            self.checkpoint_manager.save(
                model=self.base_model, optimizer=self.optimizer, scheduler=self.scheduler,
                scaler=self.scaler, step=step, epoch=0, loss=loss,
                config=self.base_model.config,
                tokenizer_path=self.config.get("tokenizer_path", "tokenizer"),
                training_config=self.config,
                training_state=training_state, identities=identities,
                runtime=self.config.get("runtime_metadata", {}), rank_rng_states=rank_rng,
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def train(
        self,
        num_steps: int | None = None,
        start_step: int = 0,
        initial_samples_consumed: int = 0,
        initial_tokens: int = 0,
        initial_state: dict | None = None,
        identities: dict | None = None,
    ) -> dict:
        self.model.train()
        state = dict(initial_state or {})
        step = int(state.get("optimizer_step", start_step))
        total_tokens = int(state.get("cumulative_loss_tokens", initial_tokens))
        samples_consumed = int(state.get("samples_consumed", initial_samples_consumed))
        source_tokens = int(state.get("cumulative_source_tokens", total_tokens))
        packed_tokens = int(state.get("cumulative_packed_tokens", total_tokens))
        final_loss = 0.0
        max_run_steps = int(num_steps if num_steps is not None else self.target_steps)
        dataloader_iter = iter(self.train_dataloader)
        model_config = getattr(self.base_model, "config", None)
        if model_config is None or not hasattr(model_config, "vocab_size"):
            raise ValueError("model must expose config.vocab_size")
        vocab_size = model_config.vocab_size

        if initial_samples_consumed > 0 or samples_consumed > 0:
            samples_per_global_batch = self.micro_batch_per_gpu * self.world_size
            for _ in range(samples_consumed // max(1, samples_per_global_batch)):
                try:
                    next(dataloader_iter)
                except StopIteration as exc:
                    raise RuntimeError("checkpoint dataset cursor exceeds available corpus") from exc

        start_time = time.time()
        stop = False
        while step < max_run_steps and total_tokens < self.target_tokens and not stop:
            self.optimizer.zero_grad(set_to_none=True)
            accepted = 0
            step_loss = 0.0
            for accumulation_index in range(self.grad_accum_steps):
                try:
                    batch = next(dataloader_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"packed corpus exhausted at {total_tokens:,} loss tokens before target {self.target_tokens:,}"
                    ) from exc
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                batch_tokens = self._global_label_count(labels)
                if total_tokens + batch_tokens > self.target_tokens:
                    stop = True
                    break
                sync_context = nullcontext()
                if hasattr(self.model, "no_sync") and accumulation_index < self.grad_accum_steps - 1:
                    sync_context = self.model.no_sync()
                with sync_context:
                    with torch.amp.autocast(device_type="cuda" if self.is_cuda else "cpu", dtype=self.dtype, enabled=self.is_cuda):
                        logits, _ = self.model(input_ids)
                        loss = nn.functional.cross_entropy(
                            logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100
                        )
                        loss = loss / self.grad_accum_steps
                    if not torch.isfinite(loss.detach()):
                        raise RuntimeError(f"non-finite loss at optimizer step {step}")
                    self.scaler.scale(loss).backward()
                total_tokens += batch_tokens
                source_tokens += batch_tokens
                packed_tokens += batch_tokens
                samples_consumed += int(input_ids.shape[0]) * self.world_size
                accepted += 1
                step_loss += float(loss.item()) * self.grad_accum_steps

            if stop:
                self.optimizer.zero_grad(set_to_none=True)
                break
            if accepted != self.grad_accum_steps:
                raise RuntimeError("incomplete gradient accumulation reached optimizer step")
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at optimizer step {step}")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            step += 1
            final_loss = step_loss / max(1, accepted)

            if step % self.log_interval == 0 and self.rank == 0:
                elapsed = max(0.001, time.time() - start_time)
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Step {step}/{self.target_steps} | Loss {final_loss:.4f} | LR {lr:.2e} | "
                    f"Tokens {total_tokens:,}/{self.target_tokens:,} | Tok/s {total_tokens / elapsed:.0f}", flush=True
                )

            if self.checkpoint_manager is not None and step % self.checkpoint_interval == 0:
                self._save_checkpoint(step, final_loss, source_tokens, packed_tokens, total_tokens, samples_consumed, identities)

            if self.validation_dataloader is not None and step % self.validation_interval == 0:
                metrics = self.evaluate(max_batches=self.config.get("validation_max_batches"))
                if self.rank == 0:
                    print(f"Validation | loss={metrics['validation_loss']:.4f} | tokens={metrics['validation_tokens']:,}", flush=True)

        if self.checkpoint_manager is not None and step > 0 and step % self.checkpoint_interval != 0:
            self._save_checkpoint(step, final_loss, source_tokens, packed_tokens, total_tokens, samples_consumed, identities)

        return {
            "final_loss": final_loss,
            "steps_completed": step,
            "total_tokens": total_tokens,
            "canonical_token_counter": self.CANONICAL_TOKEN_COUNTER,
            "target_tokens": self.target_tokens,
            "target_steps": self.target_steps,
            "global_tokens_per_step": self.global_tokens_per_step,
            "samples_consumed": samples_consumed,
        }
