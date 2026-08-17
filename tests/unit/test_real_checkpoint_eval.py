"""
tests/unit/test_real_checkpoint_eval.py
Verifies the production evaluation harness, real checkpoint loading,
and multi-domain benchmark evaluation.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import torch
from safetensors.torch import save_model

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel
from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.runtime.local.inference import LocalInferenceRuntime
from myllm.evaluation.loader import load_real_checkpoint
from myllm.evaluation.eval_harness import ProductionEvaluationHarness
from myllm.evaluation.datasets.benchmark_v1 import get_benchmark_dataset


def _create_real_checkpoint_dir(temp_dir: str) -> str:
    """Helper creating a valid real model checkpoint on disk for evaluation tests."""
    p = Path(temp_dir)

    # 1. Config (Test scale of Dhruva architecture: d=128, L=2, H=4, KV=2, vocab=1000)
    config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=256,
        max_seq_len=512,
        dropout=0.0,
        norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    config.save(str(p / "config.json"))

    # 2. Real PyTorch model weights
    model = MyLLMModel(config)
    save_model(model, str(p / "model.safetensors"))

    # 3. Real Tokenizer
    tok_dir = p / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.train_from_texts(
        ["Hello world! Protein synthesis happens in ribosomes. 144 * 12 = 1728. বাংলা সাহিত্য। हिंदी।"],
        vocab_size=1000,
    )
    tok.save(str(tok_dir))

    return temp_dir


# 1. Real Checkpoint Loading Test
def test_real_checkpoint_loading():
    temp_dir = tempfile.mkdtemp()
    try:
        ckpt_dir = _create_real_checkpoint_dir(temp_dir)
        model, tokenizer, config = load_real_checkpoint(ckpt_dir, device="cpu")

        assert isinstance(model, MyLLMModel)
        assert isinstance(tokenizer, BPETokenizer)
        assert config.d_model == 128
        assert config.n_layers == 2
        assert model.token_embedding.weight.shape == (1000, 128)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# 2. Real Model Inference Runtime with KV Cache
def test_real_model_inference_runtime():
    temp_dir = tempfile.mkdtemp()
    try:
        ckpt_dir = _create_real_checkpoint_dir(temp_dir)
        runtime = LocalInferenceRuntime(model_path=ckpt_dir, device="cpu")

        output = runtime.generate("Hello world", max_new_tokens=10, temperature=0.7)
        assert isinstance(output, str)
        assert len(output) > 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# 3. Production Evaluation Battery Execution
def test_production_eval_harness_battery():
    temp_dir = tempfile.mkdtemp()
    try:
        ckpt_dir = _create_real_checkpoint_dir(temp_dir)
        runtime = LocalInferenceRuntime(model_path=ckpt_dir, device="cpu")

        harness = ProductionEvaluationHarness(runtime=runtime, model_name="Dhruva-100M-Test")
        dataset = get_benchmark_dataset()

        summary = harness.evaluate_battery(dataset, mode="adaptive")

        assert summary.total_samples == len(dataset)
        assert 0.0 <= summary.overall_accuracy_or_pass_rate <= 1.0
        assert summary.average_latency_ms >= 0.0
        assert summary.total_generated_tokens > 0
        assert "english_qa" in summary.category_metrics
        assert "mathematics" in summary.category_metrics
        assert "unanswerable_qa" in summary.category_metrics
        assert len(summary.sample_results) == len(dataset)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# 4. A/B Comparison Mode (Fast-only vs Adaptive)
def test_eval_harness_ab_comparison():
    temp_dir = tempfile.mkdtemp()
    try:
        ckpt_dir = _create_real_checkpoint_dir(temp_dir)
        runtime = LocalInferenceRuntime(model_path=ckpt_dir, device="cpu")

        harness = ProductionEvaluationHarness(runtime=runtime, model_name="Dhruva-100M-Test")
        dataset = get_benchmark_dataset()[:4]  # Subset for fast test

        ab_results = harness.run_ab_comparison(dataset)

        assert "variant_a_fast_only" in ab_results
        assert "variant_b_adaptive" in ab_results
        assert "delta_comparison" in ab_results
        assert "accuracy_improvement" in ab_results["delta_comparison"]
        assert "latency_overhead_ms" in ab_results["delta_comparison"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# 5. Dhruva V1 Production Config Validation
def test_dhruva_v1_production_config():
    prod_config = ModelConfig.dhruva_v1_production()
    assert prod_config.d_model == 768
    assert prod_config.n_layers == 8
    assert prod_config.n_heads == 12
    assert prod_config.n_kv_heads == 4
    assert prod_config.intermediate_size == 2048
    assert prod_config.vocab_size == 64000
    assert prod_config.max_seq_len == 512


if __name__ == "__main__":
    tests = [
        test_real_checkpoint_loading,
        test_real_model_inference_runtime,
        test_production_eval_harness_battery,
        test_eval_harness_ab_comparison,
        test_dhruva_v1_production_config,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL REAL CHECKPOINT EVALUATION TESTS PASSED")
