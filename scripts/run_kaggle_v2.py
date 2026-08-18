#!/usr/bin/env python3
"""Fail-closed Dhruva V2 Kaggle DDP runner and smoke-test harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import yaml
from safetensors.torch import save_model

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel
from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.training.artifacts import runtime_metadata, sha256_file, sha256_json, source_revision
from myllm.training.checkpoint.manager import CheckpointManager
from myllm.training.data.packed import PackedTokenDataset
from myllm.training.pretraining.trainer import Trainer


def set_seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)


def init_ddp() -> tuple[int, int, int, str]:
    if os.environ.get("DHRUVA_KAGGLE_RUNNER") != "1" and not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise RuntimeError("V2 training requires the approved Kaggle execution contract")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("V2 training must be launched with torchrun and LOCAL_RANK")
    if not torch.cuda.is_available():
        raise RuntimeError("V2 Kaggle runner requires CUDA")
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2 or torch.cuda.device_count() < 2 or not 0 <= local_rank < torch.cuda.device_count():
        dist.destroy_process_group()
        raise RuntimeError("V2 requires exactly two visible GPUs and valid local rank")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, f"cuda:{local_rank}"


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
        raise ValueError("invalid V2 configuration")
    return data


def _validate_identity(args, config_data: dict, tokenizer, packed_manifest: dict, quality_manifest: dict) -> dict:
    model_config = ModelConfig(**config_data["model"])
    expected = int(config_data.get("validation", {}).get("expected_parameters", 112382208))
    required_arch = {
        "vocab_size": 48000, "d_model": 768, "n_layers": 12, "n_heads": 12,
        "n_kv_heads": 4, "intermediate_size": 2048, "max_seq_len": 1024,
        "dropout": 0.0, "qk_norm": True, "tie_word_embeddings": True,
    }
    for field, expected_value in required_arch.items():
        if getattr(model_config, field) != expected_value:
            raise RuntimeError(f"primary architecture invariant failed: {field}={getattr(model_config, field)!r}")
    if model_config.expected_parameter_count() != expected:
        raise RuntimeError(f"formula parameter count {model_config.expected_parameter_count()} != configured {expected}")
    if tokenizer.vocab_size != model_config.vocab_size:
        raise RuntimeError("tokenizer/model vocabulary mismatch")
    if model_config.vocab_size != int(config_data["tokenizer"]["vocab_size"]):
        raise RuntimeError("config tokenizer vocabulary mismatch")
    if quality_manifest.get("status") != "PASS":
        raise RuntimeError("quality manifest is not PASS")
    if packed_manifest.get("source_corpus_sha256") != quality_manifest.get("cleaned_corpus_sha256") and packed_manifest.get("source_corpus_sha256") != quality_manifest.get("corpus_sha256"):
        raise RuntimeError("packed corpus source hash does not match quality manifest")
    if packed_manifest.get("tokenizer_vocab_size") != tokenizer.vocab_size:
        raise RuntimeError("packed corpus tokenizer vocabulary mismatch")
    tokenizer_json = Path(args.tokenizer) / "tokenizer.json"
    if not (Path(args.tokenizer) / "tokenizer_metadata.json").is_file():
        raise RuntimeError("tokenizer metadata is required")
    if packed_manifest.get("quality_manifest_sha256") != sha256_file(args.quality_manifest):
        raise RuntimeError("packed corpus was not built from the supplied quality manifest")
    if packed_manifest.get("tokenizer_sha256") != sha256_file(tokenizer_json):
        raise RuntimeError("packed corpus was not built from the supplied tokenizer")
    validation_manifest = json.loads((Path(args.validation_packed_corpus) / "manifest.json").read_text(encoding="utf-8"))
    validation_quality = json.loads(Path(args.validation_quality_manifest).read_text(encoding="utf-8"))
    if validation_quality.get("status") != "PASS":
        raise RuntimeError("validation quality manifest is not PASS")
    if validation_manifest.get("quality_manifest_sha256") != sha256_file(args.validation_quality_manifest):
        raise RuntimeError("validation packed corpus quality identity mismatch")
    if validation_manifest.get("tokenizer_sha256") != sha256_file(tokenizer_json):
        raise RuntimeError("validation corpus tokenizer mismatch")
    if validation_manifest.get("tokens_sha256") == packed_manifest.get("tokens_sha256"):
        raise RuntimeError("training and validation packed corpora are identical")
    repo_root = Path(__file__).resolve().parent.parent
    relative_code_paths = [
        Path(__file__),
        Path("myllm/core/model/config.py"), Path("myllm/core/model/attention.py"),
        Path("myllm/core/model/transformer.py"), Path("myllm/training/pretraining/trainer.py"),
        Path("myllm/training/checkpoint/manager.py"), Path("myllm/training/data/packed.py"),
    ]
    code_paths = [path if path.is_absolute() else repo_root / path for path in relative_code_paths]
    code_identity = {str(path.relative_to(repo_root)).replace("\\", "/"): sha256_file(path) for path in code_paths}
    return {
        "config_sha256": sha256_json(config_data),
        "architecture_sha256": sha256_json(config_data["model"]),
        "training_config_sha256": sha256_json(config_data["training"]),
        "tokenizer_sha256": sha256_file(tokenizer_json),
        "packed_manifest_sha256": sha256_json(packed_manifest),
        "packed_tokens_sha256": packed_manifest["tokens_sha256"],
        "quality_manifest_sha256": sha256_file(args.quality_manifest),
        "validation_manifest_sha256": sha256_json(validation_manifest),
        "validation_tokens_sha256": validation_manifest["tokens_sha256"],
        "validation_quality_manifest_sha256": sha256_file(args.validation_quality_manifest),
        "source_revision": source_revision(),
        "source_tree_sha256": sha256_json(code_identity),
        "world_size": int(config_data["training"]["world_size"]),
    }


def _make_loader(args, model_config, train_config, world_size, rank, seq_len):
    dataset = PackedTokenDataset(args.packed_corpus, seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["micro_batch_per_gpu"]),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    available = len(loader) * int(train_config["micro_batch_per_gpu"]) * world_size * seq_len
    if not args.smoke_test and available < int(train_config["target_tokens"]):
        raise RuntimeError(f"packed corpus has {available:,} usable global-batch tokens, below target {train_config['target_tokens']:,}")
    return dataset, loader


def _make_validation_loader(path, train_config, world_size, rank, seq_len):
    dataset = PackedTokenDataset(path, seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    loader = DataLoader(
        dataset, batch_size=int(train_config["micro_batch_per_gpu"]), sampler=sampler,
        num_workers=1, pin_memory=True, drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("validation corpus is too short for distributed evaluation")
    return dataset, loader


def _parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dhruva_v2_t4.yaml")
    parser.add_argument("--packed-corpus", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--quality-manifest", required=True)
    parser.add_argument("--validation-packed-corpus", required=True)
    parser.add_argument("--validation-quality-manifest", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    rank, world_size, local_rank, device = init_ddp()
    try:
        config_path = Path(args.config)
        config_data = _load_yaml(config_path)
        model_config = ModelConfig(**config_data["model"])
        train_config = dict(config_data["training"])
        if int(config_data.get("data", {}).get("max_seq_len", model_config.max_seq_len)) != model_config.max_seq_len:
            raise RuntimeError("data max_seq_len does not match model max_seq_len")
        if int(train_config.get("world_size", 2)) != world_size:
            raise RuntimeError("config world_size does not match torchrun world size")
        set_seed(int(train_config.get("seed", 20260818)), rank)
        tokenizer = BPETokenizer.load(args.tokenizer)
        packed_manifest = json.loads((Path(args.packed_corpus) / "manifest.json").read_text(encoding="utf-8"))
        quality_manifest = json.loads(Path(args.quality_manifest).read_text(encoding="utf-8"))
        identities = _validate_identity(args, config_data, tokenizer, packed_manifest, quality_manifest)
        model = MyLLMModel(model_config).to(device)
        instantiated = model.count_parameters()
        expected = int(config_data.get("validation", {}).get("expected_parameters", 112382208))
        if instantiated != expected:
            raise RuntimeError(f"instantiated parameter count {instantiated} != expected {expected}")
        if model_config.tie_word_embeddings and model.output_proj.weight.data_ptr() != model.token_embedding.weight.data_ptr():
            raise RuntimeError("tied embedding invariant failed")

        checkpoint_manager = CheckpointManager(
            config_data["output"]["dir"] + "/checkpoints",
            max_retained=int(config_data["output"].get("max_retained_checkpoints", 3)),
        )
        if rank == 0:
            estimate = instantiated * 12
            global_tokens_per_step = world_size * int(train_config["micro_batch_per_gpu"]) * model_config.max_seq_len * int(train_config["gradient_accumulation_steps"])
            target_steps = int(np.ceil(int(train_config["target_tokens"]) / max(1, global_tokens_per_step)))
            projected = checkpoint_manager.projected_storage_bytes(
                estimate, target_steps,
                int(train_config.get("checkpoint_interval", 250)),
            )
            print(f"DHRUVA V2 PREFLIGHT PASS | params={instantiated:,} | projected_retained_checkpoint_bytes={projected:,}")
        seq_len = 512 if args.smoke_test else model_config.max_seq_len
        dataset, loader = _make_loader(args, model_config, train_config, world_size, rank, seq_len)
        validation_dataset, validation_loader = _make_validation_loader(args.validation_packed_corpus, train_config, world_size, rank, seq_len)
        model.gradient_checkpointing = bool(train_config.get("gradient_checkpointing", True))
        model = DistributedDataParallel(model, device_ids=[local_rank])
        run_config = dict(train_config)
        run_config["_seq_len_override"] = seq_len
        run_config["tokenizer_path"] = args.tokenizer
        run_config["runtime_metadata"] = runtime_metadata()
        if args.smoke_test:
            run_config["target_tokens"] = int(world_size * run_config["micro_batch_per_gpu"] * seq_len * run_config["gradient_accumulation_steps"] * 4)
            run_config["checkpoint_interval"] = 2
            run_config["validation_interval"] = 2
            run_config["validation_max_batches"] = 2
        trainer = Trainer(model, loader, run_config, checkpoint_manager, device=device, validation_dataloader=validation_loader)
        identities = dict(identities)
        identities.update({
            "optimizer_class": type(trainer.optimizer).__name__,
            "scheduler_class": type(trainer.scheduler).__name__,
            "scaler_enabled": bool(trainer.scaler.is_enabled()),
        })
        if args.preflight_only:
            dist.barrier()
            return
        if not args.smoke_test and not args.execute:
            raise RuntimeError("training requires explicit --smoke-test or --execute")
        initial_state = None
        if args.resume:
            manifest = checkpoint_manager.load(
                args.resume, trainer.base_model, optimizer=trainer.optimizer, scheduler=trainer.scheduler,
                scaler=trainer.scaler, device=device, rank=rank,
                expected_identities=identities, require_training_state=True,
            )
            initial_state = manifest["training_state"]
        if args.smoke_test and not args.resume:
            # Stage A: real 512-token construction/backward/checkpoint path.
            stage_a = trainer.train(num_steps=2, initial_state=None, identities=identities)
            dist.barrier()
            checkpoint_path = str(Path(checkpoint_manager.base_dir) / "checkpoint-2")
            if not Path(checkpoint_path).is_dir():
                raise RuntimeError("smoke Stage A did not publish checkpoint-2")
            # Stage C: uninterrupted continuation versus save/reload continuation.
            continuous_state = {
                "optimizer_step": 2, "cumulative_source_tokens": stage_a["total_tokens"],
                "cumulative_packed_tokens": stage_a["total_tokens"],
                "cumulative_loss_tokens": stage_a["total_tokens"],
                "samples_consumed": stage_a["samples_consumed"],
            }
            continuous_result = trainer.train(num_steps=4, initial_state=continuous_state, identities=identities)
            continuous_digest = _parameter_digest(trainer.base_model)
            del trainer, model, loader, dataset, validation_loader, validation_dataset
            torch.cuda.empty_cache()
            dataset, loader = _make_loader(args, model_config, train_config, world_size, rank, 512)
            resume_model = DistributedDataParallel(MyLLMModel(model_config).to(device), device_ids=[local_rank])
            resume_config = dict(train_config, _seq_len_override=512, tokenizer_path=args.tokenizer, runtime_metadata=runtime_metadata(), target_tokens=world_size * int(train_config["micro_batch_per_gpu"]) * 512 * int(train_config["gradient_accumulation_steps"]) * 4, checkpoint_interval=99)
            resume_trainer = Trainer(resume_model, loader, resume_config, None, device=device)
            resume_manifest = checkpoint_manager.load(
                checkpoint_path, resume_trainer.base_model, optimizer=resume_trainer.optimizer,
                scheduler=resume_trainer.scheduler, scaler=resume_trainer.scaler, device=device,
                rank=rank, expected_identities=identities, require_training_state=True,
            )
            resumed_result = resume_trainer.train(num_steps=4, initial_state=resume_manifest["training_state"], identities=identities)
            resumed_digest = _parameter_digest(resume_trainer.base_model)
            if rank == 0:
                if continuous_result["total_tokens"] != resumed_result["total_tokens"]:
                    raise RuntimeError("smoke resume token counters diverged")
                if continuous_result["samples_consumed"] != resumed_result["samples_consumed"]:
                    raise RuntimeError("smoke resume dataset cursors diverged")
                if continuous_digest != resumed_digest:
                    raise RuntimeError("smoke resume parameter digest diverged")
            stage_c = {"continuous": continuous_result, "resumed": resumed_result, "parameter_digest": resumed_digest}
            # Stage B: fresh 1024-token forward/backward and gradient-checkpoint path.
            del resume_trainer, resume_model, loader, dataset
            torch.cuda.empty_cache()
            dataset, loader = _make_loader(args, model_config, train_config, world_size, rank, 1024)
            stage_model = DistributedDataParallel(MyLLMModel(model_config).to(device), device_ids=[local_rank])
            stage_config = dict(train_config, _seq_len_override=1024, tokenizer_path=args.tokenizer, runtime_metadata=runtime_metadata(), target_tokens=world_size * int(train_config["micro_batch_per_gpu"]) * 1024 * int(train_config["gradient_accumulation_steps"]))
            stage_trainer = Trainer(stage_model, loader, stage_config, None, device=device)
            stage_b = stage_trainer.train(num_steps=1, identities=identities)
            trainer = stage_trainer
            results = {"stage_512": stage_a, "stage_resume": stage_c, "stage_1024": stage_b}
        else:
            results = trainer.train(
                num_steps=4 if args.smoke_test else None,
                initial_state=initial_state,
                identities=identities,
            )
        if rank == 0 and args.execute:
            final_dir = Path(config_data["output"]["dir"]) / "final_model"
            final_dir.mkdir(parents=True, exist_ok=True)
            save_model(trainer.base_model, str(final_dir / "model.safetensors"))
            model_config.save(str(final_dir / "config.json"))
            shutil.copytree(args.tokenizer, final_dir / "tokenizer", dirs_exist_ok=True)
            (final_dir / "training_result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
