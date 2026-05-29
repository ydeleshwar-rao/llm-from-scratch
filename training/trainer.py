"""
Training loop for the Transformer.

Features:
  - AdamW optimizer
  - Gradient clipping
  - Cosine LR schedule with warmup
  - fp16 mixed precision (GPU memory half)
  - Gradient checkpointing (activation memory kam)
  - HuggingFace Hub auto-push every N steps (session expire safe)
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from model.transformer import Transformer
from .scheduler import CosineWithWarmup

logger = logging.getLogger(__name__)


def _push_to_hub(local_path: str, hf_repo: str, hf_token: str, step: int):
    """Push checkpoint file to HuggingFace Hub. Auto-creates repo if missing."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Auto-create repo if it doesn't exist
        api.create_repo(
            repo_id=hf_repo,
            repo_type="model",
            token=hf_token,
            exist_ok=True,   # no error if already exists
            private=False,
        )
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo="model.pt",
            repo_id=hf_repo,
            repo_type="model",
            token=hf_token,
            commit_message=f"checkpoint step={step}",
        )
        logger.info(f"  ↳ HuggingFace Hub pe push kiya → {hf_repo}  (step {step})")
    except Exception as e:
        logger.warning(f"  HF Hub push failed: {e}")


class Trainer:
    def __init__(
        self,
        model: Transformer,
        train_loader: DataLoader,
        lr: float             = 3e-4,
        warmup_steps: int     = 100,
        max_steps: int        = 1000,
        grad_clip: float      = 1.0,
        device: str           = "cpu",
        fp16: bool            = True,
        grad_checkpoint: bool = True,
        hf_repo: str          = None,   # "username/model-name" — HF Hub repo
        hf_token: str         = None,   # HuggingFace API token
    ):
        self.device       = device
        self.grad_clip    = grad_clip
        self.train_loader = train_loader
        self.use_fp16     = fp16 and device == "cuda"
        self.hf_repo      = hf_repo
        self.hf_token     = hf_token

        if grad_checkpoint and device == "cuda":
            for block in model.blocks:
                block.use_checkpoint = True
            logger.info("Gradient checkpointing: ON")

        self.model = model.to(device)

        # Use 8-bit Adam on CUDA (4x less optimizer memory: 4GB → 1GB for 500M model)
        if device == "cuda":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
                )
                logger.info("8-bit AdamW: ON — optimizer memory ~4x kam hogi")
            except ImportError:
                self.optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
                )
                logger.info("bitsandbytes nahi mila — standard AdamW use ho raha hai")
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
            )
        self.scheduler = CosineWithWarmup(self.optimizer, warmup_steps, max_steps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.scaler = GradScaler("cuda") if self.use_fp16 else None
        if self.use_fp16:
            logger.info("Mixed precision (fp16): ON")

        if hf_repo:
            logger.info(f"HF Hub auto-save: ON → {hf_repo}")

    # ── Single gradient step ──────────────────────────────────────────────────
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()

        if self.use_fp16:
            with autocast("cuda"):
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

                # Save locally every save_every steps
                if checkpoint_path and step % save_every == 0:
                    self.save(checkpoint_path)
                    logger.info(f"  ↳ checkpoint saved at step {step}")
                    # Push to HF Hub if configured
                    if self.hf_repo and self.hf_token:
                        _push_to_hub(checkpoint_path, self.hf_repo, self.hf_token, step)

            avg = epoch_loss / max(len(self.train_loader), 1)
            logger.info(f"── Epoch {epoch}/{epochs}  avg_loss={avg:.4f}")

            if checkpoint_path and avg < best_loss:
                best_loss = avg
                self.save(checkpoint_path)
                logger.info(f"  ↳ best checkpoint saved (avg_loss={avg:.4f})")
                if self.hf_repo and self.hf_token:
                    _push_to_hub(checkpoint_path, self.hf_repo, self.hf_token, step)

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
