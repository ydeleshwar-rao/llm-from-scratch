"""
Transformer Block — Pre-Norm variant (Claude / LLaMA style).

Pre-norm applies RMSNorm *before* attention/FFN rather than after.
This improves training stability at large scale compared to post-norm.

Structure:
  x = x + Attention(RMSNorm(x))
  x = x + FFN(RMSNorm(x))
"""

from typing import Optional

import torch
import torch.nn as nn

from .normalization import RMSNorm
from .attention import GroupedQueryAttention
from .ffn import SwiGLU
from config.model_config import ModelConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model)
        self.attn      = GroupedQueryAttention(config)
        self.norm_ffn  = RMSNorm(config.d_model)
        self.ffn       = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict]     = None,
        use_cache: bool              = False,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        attn_out, new_cache = self.attn(
            self.norm_attn(x), mask=mask, kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x, new_cache
