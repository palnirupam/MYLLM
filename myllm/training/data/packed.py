"""Memory-mapped packed token dataset for large Kaggle training corpora."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from myllm.training.artifacts import sha256_file


class PackedTokenDataset(Dataset):
    def __init__(self, directory: str, seq_len: int):
        self.directory = Path(directory)
        if not (self.directory / "COMPLETE").is_file():
            raise ValueError("packed corpus has no COMPLETE marker")
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Packed corpus manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.seq_len = int(seq_len)
        required = {"schema_version", "token_count", "dtype", "tokens_file", "tokens_sha256", "tokens_bytes"}
        missing = required - set(self.manifest)
        if missing:
            raise ValueError(f"packed manifest missing required fields: {sorted(missing)}")
        if self.manifest["dtype"] != "uint32":
            raise ValueError("packed corpus dtype must be uint32")
        self.token_count = int(self.manifest["token_count"])
        token_path = self.directory / self.manifest["tokens_file"]
        if not token_path.is_file():
            raise FileNotFoundError(token_path)
        expected_bytes = self.token_count * np.dtype(np.uint32).itemsize
        if int(self.manifest["tokens_bytes"]) != expected_bytes or token_path.stat().st_size != expected_bytes:
            raise ValueError("packed token byte length does not match manifest token_count")
        if sha256_file(token_path) != self.manifest["tokens_sha256"]:
            raise ValueError("packed token file hash mismatch")
        if self.token_count <= self.seq_len:
            raise ValueError("Packed corpus is shorter than one training sequence")
        self.tokens = np.memmap(
            token_path,
            dtype=np.uint32,
            mode="r",
            shape=(self.token_count,),
        )
        if self.token_count and int(self.tokens.min()) < 0:
            raise ValueError("packed token IDs cannot be negative")
        vocab_size = self.manifest.get("tokenizer_vocab_size")
        if vocab_size is not None and self.token_count:
            for start in range(0, self.token_count, 1_048_576):
                block = np.asarray(self.tokens[start : min(self.token_count, start + 1_048_576)], dtype=np.uint64)
                if block.size and int(block.max()) >= int(vocab_size):
                    raise ValueError("packed token ID exceeds tokenizer vocabulary")

    def __len__(self) -> int:
        return (self.token_count - 1) // self.seq_len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.seq_len
        values = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        if values.shape[0] != self.seq_len + 1:
            raise IndexError(index)
        return {
            "input_ids": torch.from_numpy(values[:-1].copy()),
            "labels": torch.from_numpy(values[1:].copy()),
        }
