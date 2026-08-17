"""
myllm.evaluation.loader — Real checkpoint loading and orchestrator initialization.
Ensures zero mock substitution when evaluating real model weights.
"""

from pathlib import Path
import torch
from typing import Optional, Tuple
from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel
from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.runtime.local.inference import LocalInferenceRuntime
from myllm.intelligence.orchestrator import DhruvaOrchestrator
from myllm.intelligence.retrieval.bm25 import InMemoryBM25Retriever
from safetensors.torch import load_model


def load_real_checkpoint(
    checkpoint_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[MyLLMModel, BPETokenizer, ModelConfig]:
    """
    Loads real model weights from a checkpoint directory containing:
    - config.json
    - tokenizer/ (or tokenizer.model)
    - model.safetensors
    """
    p = Path(checkpoint_dir)
    config_file = p / "config.json"
    safetensors_file = p / "model.safetensors"
    tokenizer_dir = p / "tokenizer"

    if not config_file.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint directory: {checkpoint_dir}")
    if not safetensors_file.exists():
        raise FileNotFoundError(f"Missing model.safetensors in checkpoint directory: {checkpoint_dir}")

    config = ModelConfig.load(str(config_file))
    tokenizer = BPETokenizer.load(str(tokenizer_dir))

    model = MyLLMModel(config)
    load_model(model, str(safetensors_file))
    model.to(device)
    model.eval()

    return model, tokenizer, config


def build_production_orchestrator(
    checkpoint_dir: Optional[str] = None,
    runtime: Optional[LocalInferenceRuntime] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DhruvaOrchestrator:
    """
    Builds a fully wired Dhruva orchestrator backed by a real model runtime.
    """
    if runtime is None:
        if checkpoint_dir is None:
            raise ValueError("Must provide either checkpoint_dir or an instantiated runtime.")
        runtime = LocalInferenceRuntime(model_path=checkpoint_dir, device=device)

    retriever = InMemoryBM25Retriever()
    orchestrator = DhruvaOrchestrator(runtime=runtime, retriever=retriever)
    return orchestrator
