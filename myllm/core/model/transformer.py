import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .config import ModelConfig
from .position import precompute_freqs
from .layers import RMSNorm, SwiGLUFFN
from .attention import Attention

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config.d_model, config.intermediate_size)

    def forward(
        self,
        x: Tensor,
        freqs: tuple[Tensor, Tensor],
        mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        attn_out, new_kv_cache = self.attention(self.attention_norm(x), freqs, mask, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv_cache


class MyLLMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.output_proj.weight = self.token_embedding.weight
            
        cos, sin = precompute_freqs(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

        # Initialize weights with correct scale.
        # nn.Embedding defaults to N(0,1) which causes logit explosion at init
        # (observed: init loss 255 nats vs uniform baseline 10 nats = 24.7× too high).
        # LLaMA-style init: N(0, 1/sqrt(d_model)) for embedding, N(0, 0.02) for linear.
        self._init_weights()

    def _init_weights(self):
        """
        Apply weight initialization following LLaMA/GPT-NeoX conventions.
        - Embeddings: N(0, 1/sqrt(d_model))
        - Linear layers: N(0, 0.02)
        - RMSNorm: already initialized to 1 by default (correct)
        - Output projection: handled via weight tying (same as embedding if tied)
        """
        import math
        std = 1.0 / math.sqrt(self.config.d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)

        for layer in self.layers:
            # Attention projections
            for proj in [layer.attention.q_proj, layer.attention.k_proj,
                         layer.attention.v_proj, layer.attention.o_proj]:
                if hasattr(proj, 'weight'):
                    nn.init.normal_(proj.weight, mean=0.0, std=0.02)

            # FFN projections
            for proj in [layer.ffn.gate_proj, layer.ffn.up_proj, layer.ffn.down_proj]:
                if hasattr(proj, 'weight'):
                    nn.init.normal_(proj.weight, mean=0.0, std=0.02)

        # If not tied, init output_proj separately
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.output_proj.weight, mean=0.0, std=std)


    def forward(
        self, 
        input_ids: Tensor, 
        attention_mask: Optional[Tensor] = None, 
        kv_cache: Optional[list] = None, 
        use_cache: bool = False
    ) -> tuple[Tensor, Optional[list]]:
        bsz, seqlen = input_ids.shape
        h = self.token_embedding(input_ids)
        
        # Calculate position offset for KV cache (past tokens already processed)
        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].shape[1]  # (bsz, past_seq_len, n_kv_heads, head_dim)
        
        freqs = (
            self.freqs_cos[:, past_len:past_len + seqlen, :, :],
            self.freqs_sin[:, past_len:past_len + seqlen, :, :]
        )
        
        new_kv_cache = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            h, layer_new_kv_cache = layer(h, freqs, attention_mask, layer_kv_cache)
            if use_cache:
                new_kv_cache.append(layer_new_kv_cache)
                
        h = self.final_norm(h)
        logits = self.output_proj(h)
        
        return logits, new_kv_cache

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
