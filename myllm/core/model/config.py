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

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

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
