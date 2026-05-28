"""
SwiGLU Feed-Forward Network.

SwiGLU(x) = SiLU(gate(x)) * up(x)  → down(result)

Outperforms standard GELU/ReLU FFN at equal parameter counts.
Used by Claude, PaLM, LLaMA. Hidden dim is scaled to 2/3 of d_ff
so parameter count matches a standard 2-layer FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # 2/3 scaling keeps params equal to a plain 2-layer FFN
        hidden = int(2 * config.d_ff / 3)
        # Round up to nearest multiple of 64 for hardware alignment
        hidden = 64 * ((hidden + 63) // 64)

        self.gate_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.up_proj   = nn.Linear(config.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))     # SiLU ≈ smooth ReLU
        up   = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))
