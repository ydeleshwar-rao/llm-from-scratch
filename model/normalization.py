"""
RMSNorm — Root Mean Square Layer Normalization.

Used by Claude, LLaMA, Mistral instead of standard LayerNorm.
Faster because it skips mean subtraction; equally stable.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability, cast back
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).type_as(x) * self.weight
