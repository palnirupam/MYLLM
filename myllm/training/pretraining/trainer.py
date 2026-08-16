import torch
import torch.nn as nn
from torch.optim import AdamW
import time

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
            weight_decay=self.config.get("weight_decay", 0.1)
        )
        
        self.max_steps = self.config.get("max_steps", 1000)
        self.warmup_steps = self.config.get("warmup_steps", 100)
        
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress) * 3.1415926535)))
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        mixed_precision = self.config.get("mixed_precision", "fp16")
        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
            
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == torch.float16))

        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.log_interval = self.config.get("log_interval", 10)
        self.checkpoint_interval = self.config.get("checkpoint_interval", 200)

    def train(self, num_steps: int, start_step: int = 0, initial_samples_consumed: int = 0) -> dict:
        self.model.train()
        step = start_step
        total_tokens = 0
        final_loss = 0.0
        samples_consumed = initial_samples_consumed  # B4: Track for checkpoint resume

        dataloader_iter = iter(self.train_dataloader)

        # B5 FIX: Never silently fall back to 32000. The model MUST have a config.
        if not hasattr(self.model, 'config') or not hasattr(self.model.config, 'vocab_size'):
            raise ValueError(
                "Model must have a .config attribute with .vocab_size. "
                "Found model without config — cannot determine vocab_size for loss computation. "
                "This would silently use the wrong vocabulary size and corrupt training."
            )
        vocab_size = self.model.config.vocab_size

        start_time = time.time()

        while step < num_steps:
            for i in range(self.grad_accum_steps):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(self.train_dataloader)
                    batch = next(dataloader_iter)
                    
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    logits, _ = self.model(input_ids)
                    
                    # Dataset already provides shifted labels (labels[i] = input_ids[i+1])
                    # so we use logits and labels directly without additional shifting
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, vocab_size), 
                        labels.view(-1), 
                        ignore_index=-100
                    )
                    loss = loss / self.grad_accum_steps
                    
                self.scaler.scale(loss).backward()

                total_tokens += (labels != -100).sum().item()
                samples_consumed += input_ids.shape[0]  # B4: Track batch count
                final_loss = loss.item() * self.grad_accum_steps

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            
            step += 1
            
            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                lr = self.scheduler.get_last_lr()[0]
                mem = torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0
                print(f"Step {step:4d} | Loss {final_loss:.4f} | LR {lr:.2e} | Tok/s {tokens_per_sec:.0f} | Mem {mem:.1f}GB", flush=True)

                # NaN/Inf abort (B24 fix)
                import math
                if not math.isfinite(final_loss):
                    raise RuntimeError(
                        f"Training aborted at step {step}: loss={final_loss} is not finite. "
                        "This indicates gradient explosion or a bug in the loss computation."
                    )

                # Write live progress artifact (future instrumentation requirement)
                import json, pathlib
                live_path = pathlib.Path("artifacts/training_diagnostics/live_progress.json")
                live_path.parent.mkdir(parents=True, exist_ok=True)
                live_data = {
                    "step": step,
                    "total_steps": num_steps,
                    "loss": round(float(final_loss), 6),
                    "lr": round(float(lr), 8),
                    "tokens_per_sec": round(float(tokens_per_sec), 1),
                    "total_tokens": total_tokens,
                    "elapsed_sec": round(elapsed, 1),
                    "gpu_mem_gb": round(mem, 3),
                }
                live_path.write_text(json.dumps(live_data, indent=2))

            if step % self.checkpoint_interval == 0:
                rng_states = {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
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
                    # B4 FIX: Pass number of samples consumed for correct DataLoader resume
                    dataset_position=samples_consumed,
                )

        return {
            "final_loss": final_loss,
            "steps_completed": step,
            "total_tokens": total_tokens
        }
