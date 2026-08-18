"""Crash-safe, content-verified training checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from myllm.training.artifacts import fsync_file, sha256_file


REQUIRED_TRAINING_STATE = {
    "global_step",
    "optimizer_step",
    "micro_step",
    "accumulation_step",
    "cumulative_source_tokens",
    "cumulative_packed_tokens",
    "cumulative_loss_tokens",
    "dataset_cursor",
    "samples_consumed",
    "sampler_epoch",
    "sampler_cursor",
}


class CheckpointManager:
    def __init__(self, base_dir: str, max_retained: int = 3):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_retained = int(max_retained)
        if self.max_retained < 1:
            raise ValueError("max_retained must be at least 1")

    def save(
        self,
        model,
        optimizer,
        scheduler,
        step,
        epoch,
        loss,
        config,
        tokenizer_path,
        rng_states=None,
        overwrite: bool = False,
        dataset_position: int | None = None,
        *,
        scaler=None,
        training_state: dict[str, Any] | None = None,
        identities: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        rank_rng_states: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
    ) -> str:
        checkpoint_dir = self.base_dir / f"checkpoint-{int(step)}"
        if checkpoint_dir.exists() and not overwrite:
            raise ValueError(f"checkpoint already exists: {checkpoint_dir}")
        if overwrite:
            raise ValueError("atomic checkpoints do not support in-place overwrite")

        state = dict(training_state or {})
        if training_state is not None:
            missing = REQUIRED_TRAINING_STATE - set(state)
            if missing:
                raise ValueError(f"training state missing required fields: {sorted(missing)}")
        else:
            state = {
                "global_step": int(step),
                "optimizer_step": int(step),
                "micro_step": 0,
                "accumulation_step": 0,
                "cumulative_source_tokens": 0,
                "cumulative_packed_tokens": 0,
                "cumulative_loss_tokens": 0,
                "dataset_cursor": int(dataset_position or 0),
                "samples_consumed": int(dataset_position or 0),
                "sampler_epoch": int(epoch),
                "sampler_cursor": int(dataset_position or 0),
            }

        temp_dir = Path(tempfile.mkdtemp(prefix=f".{checkpoint_dir.name}.", dir=str(self.base_dir)))
        try:
            safetensors.torch.save_model(model, str(temp_dir / "model.safetensors"))
            if optimizer is not None:
                torch.save(optimizer.state_dict(), temp_dir / "optimizer.pt")
            if scheduler is not None:
                torch.save(scheduler.state_dict(), temp_dir / "scheduler.pt")
            if scaler is not None:
                torch.save(scaler.state_dict(), temp_dir / "scaler.pt")
            rng_payload = rank_rng_states if rank_rng_states is not None else rng_states
            if rng_payload is not None:
                torch.save(rng_payload, temp_dir / "rng_states.pt")
            if config is not None:
                config.save(str(temp_dir / "config.json"))
            if training_config is not None:
                (temp_dir / "training_config.json").write_text(
                    json.dumps(training_config, indent=2, sort_keys=True, default=str), encoding="utf-8"
                )

            artifact_files = sorted(p.name for p in temp_dir.iterdir() if p.is_file())
            file_hashes = {name: sha256_file(temp_dir / name) for name in artifact_files}
            manifest = {
                "schema_version": 2,
                "status": "COMPLETE",
                "step": int(step),
                "epoch": int(epoch),
                "loss": float(loss),
                "timestamp": time.time(),
                "config_path": "config.json" if config is not None else None,
                "tokenizer_path": str(tokenizer_path),
                "dataset_position": state["samples_consumed"],
                "training_state": state,
                "identities": dict(identities or {}),
                "runtime": dict(runtime or {}),
                "files": file_hashes,
            }
            manifest_path = temp_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    fsync_file(file_path)
            marker = temp_dir / "COMPLETE"
            marker.write_text(sha256_file(manifest_path) + "\n", encoding="ascii")
            fsync_file(marker)
            self.validate_checkpoint(temp_dir, require_training_state=training_state is not None)
            temp_dir.rename(checkpoint_dir)
            self.prune()
            return str(checkpoint_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def validate_checkpoint(self, checkpoint_path: str | Path, require_training_state: bool = True) -> dict:
        checkpoint_dir = Path(checkpoint_path)
        manifest_path = checkpoint_dir / "manifest.json"
        marker_path = checkpoint_dir / "COMPLETE"
        if not manifest_path.is_file() or not marker_path.is_file():
            raise ValueError(f"incomplete checkpoint: {checkpoint_dir}")
        if marker_path.read_text(encoding="ascii").strip() != sha256_file(manifest_path):
            raise ValueError("checkpoint completion marker does not match manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE" or manifest.get("schema_version") != 2:
            raise ValueError("unsupported or incomplete checkpoint manifest")
        required_files = {"model.safetensors", "config.json"}
        if require_training_state:
            required_files.update({"optimizer.pt", "scheduler.pt", "scaler.pt", "rng_states.pt"})
        if not required_files.issubset(set(manifest.get("files", {}))):
            raise ValueError(f"checkpoint is missing required artifacts: {sorted(required_files - set(manifest.get('files', {})))}")
        for name, expected_hash in manifest.get("files", {}).items():
            path = checkpoint_dir / name
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"checkpoint artifact hash mismatch: {name}")
        if require_training_state:
            state = manifest.get("training_state")
            if not isinstance(state, dict):
                raise ValueError("checkpoint has no training_state")
            missing = REQUIRED_TRAINING_STATE - set(state)
            if missing:
                raise ValueError(f"checkpoint training_state missing fields: {sorted(missing)}")
            if int(state["accumulation_step"]) != 0:
                raise ValueError("resume from a partial gradient accumulation is not supported")
        return manifest

    def load(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        scheduler=None,
        device="cuda",
        restore_rng: bool = True,
        *,
        scaler=None,
        rank: int = 0,
        expected_identities: dict[str, Any] | None = None,
        require_training_state: bool = False,
    ) -> dict:
        checkpoint_dir = Path(checkpoint_path)
        manifest = self.validate_checkpoint(checkpoint_dir, require_training_state=require_training_state)
        if expected_identities is not None:
            actual = manifest.get("identities", {})
            mismatches = {
                key: (actual.get(key), expected)
                for key, expected in expected_identities.items()
                if actual.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"checkpoint resume identity mismatch: {mismatches}")

        safetensors.torch.load_model(model, str(checkpoint_dir / "model.safetensors"))
        model.to(device)
        if optimizer is not None:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            if not optimizer_path.is_file():
                raise ValueError("checkpoint is missing optimizer state")
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))
        if scheduler is not None:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            if not scheduler_path.is_file():
                raise ValueError("checkpoint is missing scheduler state")
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=True))
        if scaler is not None:
            scaler_path = checkpoint_dir / "scaler.pt"
            if not scaler_path.is_file():
                raise ValueError("checkpoint is missing GradScaler state")
            scaler.load_state_dict(torch.load(scaler_path, map_location="cpu", weights_only=True))
        if restore_rng:
            rng_path = checkpoint_dir / "rng_states.pt"
            if not rng_path.is_file():
                if require_training_state:
                    raise ValueError("checkpoint is missing rank-specific RNG state")
                return manifest
            all_states = torch.load(rng_path, map_location="cpu", weights_only=False)
            rank_state = all_states.get(str(rank)) if isinstance(all_states, dict) else None
            if rank_state is None:
                raise ValueError(f"checkpoint has no RNG state for rank {rank}")
            import random
            import numpy as np
            random.setstate(rank_state["python"])
            np.random.set_state(rank_state["numpy"])
            torch.set_rng_state(rank_state["torch_cpu"])
            if torch.cuda.is_available():
                if rank_state.get("torch_cuda") is None:
                    raise ValueError(f"checkpoint has no CUDA RNG state for rank {rank}")
                torch.cuda.set_rng_state(rank_state["torch_cuda"], device=device)
        return manifest

    def list_checkpoints(self) -> list[dict]:
        checkpoints = []
        if not self.base_dir.exists():
            return checkpoints
        for directory in self.base_dir.iterdir():
            if not directory.is_dir() or not directory.name.startswith("checkpoint-"):
                continue
            try:
                data = self.validate_checkpoint(directory, require_training_state=False)
            except Exception:
                continue
            data["path"] = str(directory)
            checkpoints.append(data)
        return sorted(checkpoints, key=lambda item: item.get("step", 0))

    def prune(self) -> None:
        checkpoints = self.list_checkpoints()
        for checkpoint in checkpoints[:-self.max_retained]:
            shutil.rmtree(Path(checkpoint["path"]))

    def projected_storage_bytes(self, checkpoint_bytes: int, total_steps: int, interval: int) -> int:
        if interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        planned = max(1, total_steps // interval)
        return int(checkpoint_bytes) * min(planned, self.max_retained)
