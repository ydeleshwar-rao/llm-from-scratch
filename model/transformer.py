"""
Decoder-only Transformer — the full model.

Architecture (Claude / LLaMA-2 style):
  TokenEmbedding
  → N × TransformerBlock (Pre-Norm, GQA + RoPE, SwiGLU)
  → RMSNorm
  → Linear lm_head  (weight-tied to embedding)

Weight tying: lm_head shares weights with the embedding matrix.
This saves ~10% parameters and improves perplexity at small scale.
"""

from typing import Optional

import torch
import torch.nn as nn

from .normalization import RMSNorm
from .embeddings import TokenEmbedding
from .block import TransformerBlock
from config.model_config import ModelConfig


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config    = config
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.blocks    = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm      = RMSNorm(config.d_model)
        self.lm_head   = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.embedding.weight

        self.apply(self._init_weights)

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,                       # (B, T)
        mask: Optional[torch.Tensor]   = None,         # explicit causal mask or None
        kv_caches: Optional[list]      = None,         # list of per-block caches
        use_cache: bool                = False,
    ) -> tuple[torch.Tensor, Optional[list]]:
        B, T = input_ids.shape

        # Auto-generate causal mask only during the first/full-sequence pass
        if mask is None and kv_caches is None:
            mask = (
                torch.tril(torch.ones(T, T, device=input_ids.device))
                .unsqueeze(0)
                .unsqueeze(0)
            )

        x = self.embedding(input_ids)

        new_kv_caches: list = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, mask=mask, kv_cache=layer_cache, use_cache=use_cache)
            new_kv_caches.append(new_cache)

        x      = self.norm(x)
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        return logits, (new_kv_caches if use_cache else None)

    # ── Utility ───────────────────────────────────────────────────────────────
    def param_count(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        if total >= 1_000_000_000:
            return f"{total / 1e9:.2f}B"
        if total >= 1_000_000:
            return f"{total / 1e6:.2f}M"
        return f"{total / 1_000:.2f}K"
