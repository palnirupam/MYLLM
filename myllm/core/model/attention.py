import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig
from .position import apply_rotary_emb
from .layers import RMSNorm

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else None

    def forward(
        self,
        x: Tensor,
        freqs: tuple[Tensor, Tensor],
        mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        bsz, seqlen, _ = x.shape
        
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = freqs
        
        # RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = (k, v)

        # Repeat KV heads for GQA
        # k shape: (bsz, seqlen_kv, n_kv_heads, head_dim) -> (bsz, seqlen_kv, n_kv_heads, n_rep, head_dim) -> (bsz, seqlen_kv, n_heads, head_dim)
        k = k[:, :, :, None, :].expand(bsz, k.shape[1], self.n_kv_heads, self.n_rep, self.head_dim).reshape(bsz, k.shape[1], self.n_heads, self.head_dim)
        v = v[:, :, :, None, :].expand(bsz, v.shape[1], self.n_kv_heads, self.n_rep, self.head_dim).reshape(bsz, v.shape[1], self.n_heads, self.head_dim)

        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        k = k.transpose(1, 2)  # (bsz, n_heads, seqlen_kv, head_dim)
        v = v.transpose(1, 2)  # (bsz, n_heads, seqlen_kv, head_dim)

        # RMSNorm and RoPE use FP32 internally, then explicitly return to the
        # projection dtype. SDPA must receive Q/K/V with one common dtype.
        sdpa_dtype = v.dtype
        q = q.to(sdpa_dtype)
        k = k.to(sdpa_dtype)

        is_causal = mask is None and seqlen > 1
        
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.o_proj(output)
        
        return output, new_kv_cache
