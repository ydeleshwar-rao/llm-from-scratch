"""
LoRA Manager — inject, freeze, save, and swap LoRA adapters.

Typical flow:
  1. manager = LoRAManager(rank=8, alpha=16)
  2. manager.inject(model)
     → replaces q_proj / v_proj Linear layers with LoRALinear
  3. manager.freeze_base(model)
     → freezes everything except lora_A / lora_B
  4. Train with LoRATrainer (only adapter params update)
  5. manager.save_adapter(model, "checkpoints/adapters/customer_service.pt")

Later, to swap adapters on the same base model:
  6. manager.load_adapter(model, "checkpoints/adapters/customer_service.pt")
"""

import logging
import os
from typing import Iterable

import torch
import torch.nn as nn

from .lora import LoRALinear

logger = logging.getLogger(__name__)

_DEFAULT_TARGETS = {"q_proj", "v_proj"}


class LoRAManager:
    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Iterable[str] = _DEFAULT_TARGETS,
    ):
        self.rank    = rank
        self.alpha   = alpha
        self.targets = set(target_modules)

    # ── Injection ─────────────────────────────────────────────────────────────

    def inject(self, model: nn.Module) -> nn.Module:
        """Replace target Linear layers with LoRALinear in-place. Returns model."""
        count = 0
        for name, module in list(model.named_modules()):
            for target in self.targets:
                if hasattr(module, target):
                    child = getattr(module, target)
                    if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                        lora_layer = LoRALinear(child, rank=self.rank, alpha=self.alpha)
                        setattr(module, target, lora_layer)
                        count += 1
                        logger.debug(f"LoRA injected: {name}.{target}")
        logger.info(f"LoRA injected into {count} layers (rank={self.rank}, alpha={self.alpha})")
        return model

    # ── Freeze ────────────────────────────────────────────────────────────────

    def freeze_base(self, model: nn.Module):
        """Freeze all parameters except LoRA A/B matrices."""
        frozen = trained = 0
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                trained += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()
        pct = 100 * trained / max(frozen + trained, 1)
        logger.info(
            f"Frozen {frozen:,} base params | "
            f"Trainable {trained:,} LoRA params ({pct:.2f}%)"
        )

    def trainable_params(self, model: nn.Module) -> list:
        """Return list of parameters with requires_grad=True."""
        return [p for p in model.parameters() if p.requires_grad]

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save_adapter(self, model: nn.Module, path: str):
        """Save only the LoRA adapter weights (A and B matrices)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        adapter_state = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        }
        torch.save(
            {"adapter_state": adapter_state, "rank": self.rank, "alpha": self.alpha},
            path,
        )
        logger.info(f"Adapter saved → {path}  ({len(adapter_state)} tensors)")

    def load_adapter(self, model: nn.Module, path: str):
        """Load LoRA adapter weights into an already-injected model."""
        ckpt          = torch.load(path, map_location="cpu")
        adapter_state = ckpt["adapter_state"]
        model_state   = dict(model.named_parameters())
        loaded = missing = 0
        for name, data in adapter_state.items():
            if name in model_state:
                model_state[name].data.copy_(data)
                loaded += 1
            else:
                logger.warning(f"Adapter key not found in model: {name}")
                missing += 1
        logger.info(
            f"Adapter loaded ← {path}  "
            f"({loaded}/{len(adapter_state)} tensors matched, {missing} missing)"
        )
