"""
Training loop for the Transformer.

Features:
  - AdamW optimizer (standard for transformers)
  - Gradient clipping to prevent exploding gradients
  - Cosine LR schedule with warmup
  - Optional checkpoint save/load
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.transformer import Transformer
from .scheduler import CosineWithWarmup

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: Transformer,
        train_loader: DataLoader,
        lr: float           = 3e-4,
        warmup_steps: int   = 100,
        max_steps: int      = 1000,
        grad_clip: float    = 1.0,
        device: str         = "cpu",
    ):
        self.model        = model.to(device)
        self.device       = device
        self.grad_clip    = grad_clip
        self.train_loader = train_loader

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
        )
        self.scheduler = CosineWithWarmup(self.optimizer, warmup_steps, max_steps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # ── Single gradient step ──────────────────────────────────────────────────
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        logits, _ = self.model(x)                                # use_cache=False by default
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    # ── Full training run ─────────────────────────────────────────────────────
    def train(
        self,
        epochs: int = 1,
        log_every: int = 10,
        checkpoint_path: Optional[str] = None,
        save_every: int = 500,
    ) -> List[float]:
        self.model.train()
        losses: List[float] = []
        best_loss = float("inf")
        step = 0

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for x, y in self.train_loader:
                loss = self.train_step(x, y)
                losses.append(loss)
                epoch_loss += loss
                step += 1

                if step % log_every == 0:
                    logger.info(
                        f"step={step:5d}  loss={loss:.4f}  lr={self.scheduler.current_lr:.2e}"
                    )

                if checkpoint_path and step % save_every == 0:
                    self.save(checkpoint_path)
                    logger.info(f"  ↳ checkpoint saved at step {step}")

            avg = epoch_loss / max(len(self.train_loader), 1)
            logger.info(f"── Epoch {epoch}/{epochs}  avg_loss={avg:.4f}")

            if checkpoint_path and avg < best_loss:
                best_loss = avg
                self.save(checkpoint_path)
                logger.info(f"  ↳ best checkpoint saved (avg_loss={avg:.4f})")

        return losses

    # ── Checkpointing ─────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config":          self.model.config.model_dump(),
            },
            path,
        )
        logger.info(f"Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        logger.info(f"Loaded checkpoint ← {path}")
