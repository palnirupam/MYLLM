from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    intermediate_size: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    # Disabled by default so existing V1 checkpoints keep the same state dict.
    qk_norm: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("vocab_size, d_model, and n_layers must be positive")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.intermediate_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("intermediate_size and max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def expected_parameter_count(self) -> int:
        """Exact unique trainable parameter count for this bias-free architecture."""
        embedding = self.vocab_size * self.d_model
        attention = (
            self.d_model * (self.n_heads * self.head_dim)
            + 2 * self.d_model * (self.n_kv_heads * self.head_dim)
            + (self.n_heads * self.head_dim) * self.d_model
        )
        swiglu = 3 * self.d_model * self.intermediate_size
        block_norms = 2 * self.d_model
        qk_norms = 2 * self.head_dim if self.qk_norm else 0
        final_norm = self.d_model
        output = 0 if self.tie_word_embeddings else self.vocab_size * self.d_model
        return embedding + self.n_layers * (attention + swiglu + block_norms + qk_norms) + final_norm + output

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def dhruva_v1_production(cls, max_seq_len: int = 512) -> 'ModelConfig':
        """
        Dhruva V1 Production Backbone (Stage 1A):
        d_model=768, n_layers=8, n_heads=12, n_kv_heads=4, intermediate_size=2048, vocab_size=64000, max_seq_len=512.
        """
        return cls(
            vocab_size=64000,
            d_model=768,
            n_layers=8,
            n_heads=12,
            n_kv_heads=4,
            intermediate_size=2048,
            max_seq_len=max_seq_len,
            dropout=0.0,
            norm_eps=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
