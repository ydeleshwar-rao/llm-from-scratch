"""
Grouped Query Attention (GQA).

GQA = fewer KV heads than Q heads → smaller KV cache, faster inference.
  n_kv_heads == n_heads  → standard Multi-Head Attention (MHA)
  n_kv_heads == 1        → Multi-Query Attention (MQA)
  1 < n_kv_heads < n_heads → Grouped Query Attention (used by Claude / LLaMA-2)

Includes:
  - RoPE applied to Q and K
  - Causal (or no) mask support
  - KV cache for single-token generation steps
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryEmbedding
from config.model_config import ModelConfig


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups   = config.n_groups
        self.head_dim   = config.head_dim
        self.scale      = 1.0 / math.sqrt(config.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.n_heads    * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim,  config.d_model,   bias=False)

        self.rope    = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,                         # (B, T, d_model)
        mask: Optional[torch.Tensor] = None,     # (1, 1, T, T) causal mask or None
        kv_cache: Optional[dict]     = None,     # {"k": ..., "v": ...} from previous step
        use_cache: bool              = False,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE — offset by past tokens when cache is present
        past_len = kv_cache["k"].shape[2] if (kv_cache and "k" in kv_cache) else 0
        q, k = self.rope(q, k, offset=past_len)

        # Extend KV cache
        if kv_cache and "k" in kv_cache:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)

        new_cache = {"k": k.detach(), "v": v.detach()} if use_cache else None

        # Expand KV heads to match Q heads (GQA broadcast)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, H, T_q, T_k)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                                 # (B, H, T_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)              # (B, T, d_model)

        return self.o_proj(out), new_cache
