import os
import sys
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
        if self._tokenizer.decoder is None:
            self._tokenizer.decoder = ByteLevelDecoder()
        self._pad_token = "<pad>"
        self._unk_token = "<unk>"
        self._bos_token = "<bos>"
        self._eos_token = "<eos>"

    @classmethod
    def train_from_texts(cls, texts: Iterable[str], vocab_size: int = 32000) -> 'BPETokenizer':
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
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

    @classmethod
    def load(cls, directory: str) -> 'BPETokenizer':
        path = Path(directory) / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.decoder is None:
            tokenizer.decoder = ByteLevelDecoder()
        return cls(tokenizer)
