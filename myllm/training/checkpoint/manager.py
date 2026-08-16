import os
import json
import time
import torch
import safetensors.torch
from pathlib import Path

class CheckpointManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, scheduler, step, epoch, loss, config, tokenizer_path,
             rng_states=None, overwrite: bool = False,
             dataset_position: int = None) -> str:
        """
        Save a training checkpoint.

        Args:
            model: The model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            step: Current global training step
            epoch: Current epoch
            loss: Most recent training loss
            config: ModelConfig instance
            tokenizer_path: Path to the tokenizer directory
            rng_states: Dict with 'cpu' and 'cuda' RNG states
            overwrite: If True, allow overwriting an existing checkpoint at this step
            dataset_position: B4 FIX — Number of samples consumed in the current epoch.
                Used to skip forward in the DataLoader on resume, preventing
                re-training on already-seen data.
        """
        checkpoint_dir = self.base_dir / f"checkpoint-{step}"

        # B10 FIX: Overwrite protection.
        if checkpoint_dir.exists() and not overwrite:
            raise ValueError(
                f"Checkpoint at step {step} already exists: {checkpoint_dir}. "
                "Pass overwrite=True explicitly if you intend to replace it."
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        safetensors.torch.save_model(model, checkpoint_dir / "model.safetensors")

        # Save optimizer, scheduler, rng_states
        if optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        if rng_states is not None:
            torch.save(rng_states, checkpoint_dir / "rng_states.pt")

        # Save config
        if config is not None:
            config.save(str(checkpoint_dir / "config.json"))

        # Save manifest
        manifest = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "config_path": "config.json",
            "tokenizer_path": str(tokenizer_path),
            "timestamp": time.time(),
            # B4 FIX: Record how many samples have been consumed in the current epoch.
            # On resume, the DataLoader can skip this many samples to avoid re-training.
            "dataset_position": dataset_position,
        }
        with open(checkpoint_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        return str(checkpoint_dir)

    def load(self, checkpoint_path: str, model, optimizer=None, scheduler=None,
             device='cuda', restore_rng: bool = True) -> dict:
        """
        Load a checkpoint and return the manifest.

        Args:
            checkpoint_path: Path to the checkpoint directory
            model: Model to restore weights into
            optimizer: Optimizer to restore state into (optional)
            scheduler: LR scheduler to restore (optional)
            device: Device to load tensors onto
            restore_rng: If True, restore CPU/CUDA RNG states for deterministic resume

        Returns:
            Manifest dict (includes step, epoch, loss, dataset_position, etc.)
        """
        checkpoint_dir = Path(checkpoint_path)

        # Load weights
        safetensors.torch.load_model(model, checkpoint_dir / "model.safetensors")
        model.to(device)

        if optimizer is not None:
            opt_path = checkpoint_dir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=device, weights_only=True))

        if scheduler is not None:
            sched_path = checkpoint_dir / "scheduler.pt"
            if sched_path.exists():
                scheduler.load_state_dict(
                    torch.load(sched_path, map_location=device, weights_only=True))

        # Restore RNG states for fully deterministic resume
        if restore_rng:
            rng_path = checkpoint_dir / "rng_states.pt"
            if rng_path.exists():
                rng_states = torch.load(rng_path, map_location='cpu', weights_only=True)
                if rng_states.get('cpu') is not None:
                    torch.set_rng_state(rng_states['cpu'])
                if torch.cuda.is_available() and rng_states.get('cuda') is not None:
                    torch.cuda.set_rng_state(rng_states['cuda'])

        manifest_path = checkpoint_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                return json.load(f)
        return {}

    def list_checkpoints(self) -> list[dict]:
        checkpoints = []
        if not self.base_dir.exists():
            return checkpoints
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith("checkpoint-"):
                manifest_path = d / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, "r") as f:
                        data = json.load(f)
                        data["path"] = str(d)
                        checkpoints.append(data)
        return sorted(checkpoints, key=lambda x: x.get("step", 0))
