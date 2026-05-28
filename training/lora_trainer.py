"""
LoRA Trainer — fine-tunes only the LoRA adapter parameters.

Same loop as Trainer but the optimizer only touches lora_A / lora_B.
All base model weights stay frozen throughout.
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.transformer import Transformer
from model.lora_manager import LoRAManager
from .scheduler import CosineWithWarmup

logger = logging.getLogger(__name__)


class LoRATrainer:
    def __init__(
        self,
        model: Transformer,
        train_loader: DataLoader,
        lora_manager: LoRAManager,
        lr: float         = 1e-4,
        warmup_steps: int = 50,
        max_steps: int    = 500,
        grad_clip: float  = 1.0,
        device: str       = "cpu",
    ):
        self.model        = model.to(device)
        self.device       = device
        self.grad_clip    = grad_clip
        self.train_loader = train_loader
        self.manager      = lora_manager

        trainable = lora_manager.trainable_params(model)
        if not trainable:
            raise ValueError(
                "No trainable params found. "
                "Call manager.inject(model) then manager.freeze_base(model) first."
            )

        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        self.scheduler = CosineWithWarmup(self.optimizer, warmup_steps, max_steps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        total       = sum(p.numel() for p in model.parameters())
        trainable_n = sum(p.numel() for p in trainable)
        logger.info(f"LoRATrainer ready: {trainable_n:,} / {total:,} params trainable")

    # ── Single step ───────────────────────────────────────────────────────────

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        logits, _ = self.model(x)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.manager.trainable_params(self.model), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    # ── Full training run ─────────────────────────────────────────────────────

    def train(
        self,
        epochs: int = 1,
        log_every: int = 10,
        adapter_path: Optional[str] = None,
        save_every: int = 200,
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

                if adapter_path and step % save_every == 0:
                    self.manager.save_adapter(self.model, adapter_path)

            avg = epoch_loss / max(len(self.train_loader), 1)
            logger.info(f"── Epoch {epoch}/{epochs}  avg_loss={avg:.4f}")

            if adapter_path and avg < best_loss:
                best_loss = avg
                self.manager.save_adapter(self.model, adapter_path)
                logger.info(f"  ↳ best adapter saved (avg_loss={avg:.4f})")

        return losses
