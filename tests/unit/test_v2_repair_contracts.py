import json
from pathlib import Path

from myllm.core.model.config import ModelConfig
from myllm.training.artifacts import sha256_file, sha256_json


def test_primary_parameter_count_and_qk_norm_delta():
    primary = ModelConfig(
        vocab_size=48000, d_model=768, n_layers=12, n_heads=12,
        n_kv_heads=4, intermediate_size=2048, max_seq_len=1024,
        qk_norm=True, tie_word_embeddings=True,
    )
    without_qk = ModelConfig(**{**primary.__dict__, "qk_norm": False})
    assert primary.expected_parameter_count() == 112_382_208
    assert without_qk.expected_parameter_count() == 112_380_672


def test_json_identity_is_order_independent():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_file_hash_is_content_hash(tmp_path: Path):
    path = tmp_path / "artifact"
    path.write_bytes(b"dhruva-v2")
    assert sha256_file(path) == "8dea42ff996fe77819d3a04490ab803046e028ee154958bd7ae0eee4d5203fff"
