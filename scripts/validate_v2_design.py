#!/usr/bin/env python3
"""Static Dhruva V2 architecture validation without importing PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - used only in minimal local envs
    yaml = None


def _parse_scalar(value: str):
    value = value.split(" #", 1)[0].strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("\"'")


def _load_minimal_yaml(path: Path) -> dict:
    """Parse the scalar, two-level config used by the V2 preflight file."""
    result: dict = {}
    section = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not raw_line.startswith(" ") and line.endswith(":"):
            section = line[:-1]
            result[section] = {}
            continue
        if section is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[section][key.strip()] = _parse_scalar(value)
    return result


def estimate_parameters(model: dict) -> dict[str, int]:
    vocab_size = int(model["vocab_size"])
    d_model = int(model["d_model"])
    n_layers = int(model["n_layers"])
    n_heads = int(model["n_heads"])
    n_kv_heads = int(model["n_kv_heads"])
    intermediate_size = int(model["intermediate_size"])
    tied = bool(model.get("tie_word_embeddings", True))
    qk_norm = bool(model.get("qk_norm", False))

    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads for GQA")

    head_dim = d_model // n_heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    embedding = vocab_size * d_model
    kv_width = n_kv_heads * head_dim
    attention_per_layer = (
        d_model * d_model
        + d_model * kv_width
        + d_model * kv_width
        + d_model * d_model
    )
    ffn_per_layer = 3 * d_model * intermediate_size
    norms_per_layer = 2 * d_model
    qk_norm_per_layer = 2 * head_dim if qk_norm else 0
    transformer = n_layers * (
        attention_per_layer + ffn_per_layer + norms_per_layer + qk_norm_per_layer
    )
    final_norm = d_model
    output = 0 if tied else vocab_size * d_model
    total = embedding + transformer + final_norm + output

    return {
        "embedding": embedding,
        "transformer": transformer,
        "final_norm": final_norm,
        "output": output,
        "total": total,
        "head_dim": head_dim,
        "gqa_ratio": n_heads // n_kv_heads,
        "qk_norm_parameters": n_layers * qk_norm_per_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/dhruva_v2_t4.yaml",
        help="Dhruva V2 YAML configuration",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if yaml else _load_minimal_yaml(config_path)
    model = config["model"]
    training = config["training"]
    tokenizer = config["tokenizer"]

    stats = estimate_parameters(model)

    if int(tokenizer["vocab_size"]) != int(model["vocab_size"]):
        raise ValueError("tokenizer and model vocab sizes do not match")
    if int(config["data"]["max_seq_len"]) != int(model["max_seq_len"]):
        raise ValueError("data and model max_seq_len values do not match")
    if int(training.get("world_size", 1)) != 2:
        raise ValueError("Dhruva V2 Kaggle config must use two T4 GPUs")
    if not bool(training.get("gradient_checkpointing", False)):
        raise ValueError("gradient checkpointing must be enabled for the T4 run")
    expected_parameters = int(config["validation"]["expected_parameters"])
    if stats["total"] != expected_parameters:
        raise ValueError(
            f"parameter count changed: {stats['total']:,} != "
            f"{expected_parameters:,}"
        )

    fp16_weight_gib = stats["total"] * 2 / 1024**3
    print("DHRUVA V2 DESIGN PREFLIGHT: PASS")
    print(f"Config             : {config_path}")
    print(f"Parameters         : {stats['total']:,}")
    print(f"Embedding          : {stats['embedding']:,}")
    print(f"Transformer body   : {stats['transformer']:,}")
    print(f"Head dimension     : {stats['head_dim']}")
    print(f"GQA ratio          : {stats['gqa_ratio']}:1")
    print(f"FP16 weight memory : {fp16_weight_gib:.3f} GiB")
    effective_sequences = (
        int(training["world_size"])
        * int(training["micro_batch_per_gpu"])
        * int(training["gradient_accumulation_steps"])
    )
    effective_tokens = effective_sequences * int(model["max_seq_len"])
    print(f"Effective batch    : {effective_sequences} sequences")
    print(f"Tokens/step        : {effective_tokens:,} maximum packed tokens")
    print("Training target    : Kaggle 2 x Tesla T4 with DDP")


if __name__ == "__main__":
    main()
