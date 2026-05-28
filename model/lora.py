"""
LoRA (Low-Rank Adaptation) — efficient fine-tuning with frozen base weights.

Instead of updating W (d_out × d_in), we add a small ΔW = B × A:
  - A: (rank × d_in)  — small random init
  - B: (d_out × rank) — zero init  →  ΔW = 0 at start, so training begins
                                        from the same point as the base model

Output: W(x) + (B × A)(x) × (alpha / rank)

Why: rank << min(d_in, d_out) so adapter has far fewer parameters.
     rank=8 on a 256×256 layer: 65,536 → 4,096 params (~6% of original).
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a frozen base + trainable LoRA adapter."""

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        d_out, d_in = base_linear.weight.shape
        self.base  = base_linear
        self.rank  = rank
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scale

    def extra_repr(self) -> str:
        d_out, d_in = self.base.weight.shape
        return f"in={d_in}, out={d_out}, rank={self.rank}, scale={self.scale:.2f}"
