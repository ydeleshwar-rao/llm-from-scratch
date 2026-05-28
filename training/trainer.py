"""
Training loop for the Transformer.

Features:
  - AdamW optimizer (standard for transformers)
  - Gradient clipping to prevent exploding gradients
  - Cosine LR schedule with warmup
  - fp16 mixed precision (GPU memory half ho jaati hai)
  - Gradient checkpointing (activation memory kam hoti hai)
  - Optional checkpoint save/load
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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
        fp16: bool          = True,    # mixed precision (GPU only)
        grad_checkpoint: bool = True,  # gradient checkpointing (saves activation memory)
    ):
        self.device    = device
        self.grad_clip = grad_clip
        self.train_loader = train_loader
        self.use_fp16  = fp16 and device == "cuda"

        # Gradient checkpointing — activations recompute during backward
        # Saves ~3-4x activation memory, ~20% slower but fits larger models
        if grad_checkpoint and device == "cuda":
            for block in model.blocks:
                block.use_checkpoint = True
            logger.info("Gradient checkpointing: ON")

        self.model = model.to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
        )
        self.scheduler = CosineWithWarmup(self.optimizer, warmup_steps, max_steps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # fp16 scaler — prevents underflow during mixed precision training
        self.scaler = GradScaler() if self.use_fp16 else None
        if self.use_fp16:
            logger.info("Mixed precision (fp16): ON  — GPU memory ~50% kam hogi")

    # ── Single gradient step ──────────────────────────────────────────────────
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()

        if self.use_fp16:
            with autocast():
                logits, _ = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, _ = self.model(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
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
