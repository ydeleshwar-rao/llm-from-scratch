"""
Cosine Annealing with Linear Warmup.

Standard LR schedule for transformer pre-training (used by GPT, LLaMA, Claude).
  - Linear ramp from 0 → peak_lr over warmup_steps
  - Cosine decay from peak_lr → min_lr over remaining steps
"""

import math
import torch


class CosineWithWarmup:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-5,
    ):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.min_lr       = min_lr
        self.peak_lr      = optimizer.param_groups[0]["lr"]
        self._step        = 0

    def step(self):
        self._step += 1
        lr = self._compute_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _compute_lr(self) -> float:
        if self._step <= self.warmup_steps:
            return self.peak_lr * self._step / max(self.warmup_steps, 1)
        progress = (self._step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
