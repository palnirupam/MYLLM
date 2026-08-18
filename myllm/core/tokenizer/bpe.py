import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Iterable
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from .base import TokenizerBase

class BPETokenizer(TokenizerBase):
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self.byte_fallback = bool(getattr(tokenizer.model, "byte_fallback", True))
        if self._tokenizer.decoder is None:
            self._tokenizer.decoder = ByteLevelDecoder()
        self._pad_token = "<pad>"
        self._unk_token = "<unk>"
        self._bos_token = "<bos>"
        self._eos_token = "<eos>"

    @classmethod
    def train_from_texts(cls, texts: Iterable[str], vocab_size: int = 32000, byte_fallback: bool = True) -> 'BPETokenizer':
        tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=byte_fallback))
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        )
        
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[list[int]]:
        """Fast multi-threaded Rust batch encoding using HuggingFace Tokenizers."""
        encodings = self._tokenizer.encode_batch(texts)
        if add_special_tokens:
            bos, eos = self.bos_token_id, self.eos_token_id
            return [[bos] + enc.ids + [eos] for enc in encodings]
        return [enc.ids for enc in encodings]

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Fast multi-threaded Rust batch token counting."""
        encodings = self._tokenizer.encode_batch(texts)
        return [len(enc.ids) for enc in encodings]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.token_to_id(self._bos_token)

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.token_to_id(self._eos_token)

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.token_to_id(self._pad_token)

    @property
    def unk_token_id(self) -> int:
        return self._tokenizer.token_to_id(self._unk_token)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path / "tokenizer.json"))
        metadata = {
            "schema_version": 1,
            "tokenizer_sha256": hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest(),
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad": self.pad_token_id,
                "unk": self.unk_token_id,
                "bos": self.bos_token_id,
                "eos": self.eos_token_id,
            },
            "normalization": "tokenizers-default",
            "byte_fallback": self.byte_fallback,
        }
        (path / "tokenizer_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> 'BPETokenizer':
        path = Path(directory) / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.decoder is None:
            tokenizer.decoder = ByteLevelDecoder()
        instance = cls(tokenizer)
        metadata_path = path.parent / "tokenizer_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if metadata.get("tokenizer_sha256") != actual:
                raise ValueError("tokenizer metadata hash does not match tokenizer.json")
            if int(metadata.get("vocab_size", -1)) != instance.vocab_size:
                raise ValueError("tokenizer metadata vocab size mismatch")
            expected_special = metadata.get("special_tokens", {})
            actual_special = {
                "pad": instance.pad_token_id,
                "unk": instance.unk_token_id,
                "bos": instance.bos_token_id,
                "eos": instance.eos_token_id,
            }
            if expected_special != actual_special:
                raise ValueError("tokenizer special-token IDs do not match metadata")
        return instance
