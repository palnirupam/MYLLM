import os
import torch
import torch.nn.functional as F
from typing import Generator
from pathlib import Path
from myllm.runtime.interfaces.base import InferenceRuntime
from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel
from myllm.core.tokenizer.bpe import BPETokenizer
from safetensors.torch import load_model

class LocalInferenceRuntime(InferenceRuntime):
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        p = Path(model_path)
        config_path = p / "config.json"
        weights_path = p / "model.safetensors"
        tokenizer_path = p / "tokenizer"

        self.config = ModelConfig.load(str(config_path))
        self.model = MyLLMModel(self.config)
        
        load_model(self.model, str(weights_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = BPETokenizer.load(str(tokenizer_path))

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
        logits = logits / max(temperature, 1e-5)
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        raw_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = ([self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []) + raw_ids
        if not input_ids:
            input_ids = [self.tokenizer.bos_token_id or 0]

        # Guard against prompt exceeding max_seq_len
        if len(input_ids) >= self.config.max_seq_len:
            input_ids = input_ids[-(self.config.max_seq_len - 1):]

        max_generate = min(max_new_tokens, self.config.max_seq_len - len(input_ids))
        if max_generate <= 0:
            return prompt

        input_tensor = torch.tensor([input_ids], device=self.device)
        kv_cache = None
        generated_ids = []

        for _ in range(max_generate):
            logits, kv_cache = self.model(input_tensor, kv_cache=kv_cache, use_cache=True)
            next_logits = logits[:, -1, :]
            
            next_token_id = self._sample(next_logits, temperature, top_k, top_p)
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]], device=self.device)
            
        # SentencePiece BPE decode always prepends a leading space to the first token.
        # Strip it to avoid generated text starting with an unwanted space.
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).lstrip(' ')
        return prompt + generated_text

    @torch.inference_mode()
    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        raw_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = ([self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []) + raw_ids
        if not input_ids:
            input_ids = [self.tokenizer.bos_token_id or 0]
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        kv_cache = None
        first_token = True

        for _ in range(max_new_tokens):
            logits, kv_cache = self.model(input_tensor, kv_cache=kv_cache, use_cache=True)
            next_logits = logits[:, -1, :]
            
            next_token_id = self._sample(next_logits, temperature, top_k, top_p)
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            token_str = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            # SentencePiece prepends a leading space on the first decoded token — strip it.
            if first_token:
                token_str = token_str.lstrip(' ')
                first_token = False
            yield token_str
            
            input_tensor = torch.tensor([[next_token_id]], device=self.device)
