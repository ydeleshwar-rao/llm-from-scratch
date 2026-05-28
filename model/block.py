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
import torch.utils.checkpoint as checkpoint_util

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
        self.use_checkpoint = False  # trainer sets this True for large models

    def _forward(
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict]     = None,
        use_cache: bool              = False,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        # Gradient checkpointing: recompute activations in backward pass
        # Saves ~3x activation memory at cost of ~20% slower training
        if self.use_checkpoint and self.training and kv_cache is None:
            def ckpt_fn(x_):
                out, _ = self._forward(x_, mask=mask, use_cache=False)
                return out
            x = checkpoint_util.checkpoint(ckpt_fn, x, use_reentrant=False)
            return x, None
        return self._forward(x, mask=mask, kv_cache=kv_cache, use_cache=use_cache)
