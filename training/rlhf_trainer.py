"""
RLHF via DPO (Direct Preference Optimization).

DPO trains the model to prefer chosen responses over rejected ones
without needing a separate reward model.

DPO loss:
  L = -E[ log σ( β * ( (log π_θ(y_w|x) - log π_ref(y_w|x))
                      - (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ) ]

Where:
  π_θ   = policy model (being trained)
  π_ref = reference model (frozen copy of base weights)
  y_w   = chosen (preferred / safe) response
  y_l   = rejected (harmful / unsafe) response
  β     = KL penalty strength (0.1 is a good default)

Dataset format (list of dicts):
  [{"prompt": "...", "chosen": "...", "rejected": "..."}, ...]

Also exports:
  safety_score(text) → float  0.0 (safe) … 1.0 (very unsafe)
  is_safe(text)      → bool
"""

import copy
import logging
import os
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.transformer import Transformer
from tokenizer.bpe_tokenizer import BPETokenizer
from .scheduler import CosineWithWarmup

logger = logging.getLogger(__name__)


# ── Rule-based safety filter ──────────────────────────────────────────────────

_HARMFUL_RE = re.compile(
    r"\b(kill|murder|suicide|bomb|weapon|explosive|poison|harm\b|hurt\b|"
    r"attack|rape|abuse|torture|assault|shoot|stab|violence)\b",
    re.IGNORECASE,
)
_SEXUAL_RE = re.compile(
    r"\b(porn|nude|naked|sexual|erotic|xxx|nsfw)\b",
    re.IGNORECASE,
)
_HATE_RE = re.compile(
    r"\b(nigger|faggot|retard|slut|whore|cunt)\b",
    re.IGNORECASE,
)


def safety_score(text: str) -> float:
    """
    Rule-based safety score.
    Returns 0.0 (clean) to 1.0 (very unsafe).
    """
    score = 0.0
    if _HARMFUL_RE.search(text):
        score += 0.5
    if _SEXUAL_RE.search(text):
        score += 0.5
    if _HATE_RE.search(text):
        score += 0.6
    return min(score, 1.0)


def is_safe(text: str, threshold: float = 0.5) -> bool:
    """Return True if text is below the safety threshold."""
    return safety_score(text) < threshold


# ── Preference dataset ────────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    """
    Tokenized preference-pair dataset for DPO training.
    Each dict must have: {"prompt": str, "chosen": str, "rejected": str}
    """

    def __init__(
        self,
        pairs: List[dict],
        tokenizer: BPETokenizer,
        max_len: int = 128,
    ):
        self.items: list = []
        for pair in pairs:
            prompt   = pair["prompt"]
            chosen   = pair["chosen"]
            rejected = pair["rejected"]

            prompt_ids   = tokenizer.encode(prompt,           add_bos=True, add_eos=False)
            chosen_ids   = tokenizer.encode(prompt + chosen,  add_bos=True, add_eos=True)[:max_len]
            rejected_ids = tokenizer.encode(prompt + rejected, add_bos=True, add_eos=True)[:max_len]

            self.items.append({
                "chosen_ids":   chosen_ids,
                "rejected_ids": rejected_ids,
                "prompt_len":   len(prompt_ids),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _pad_collate(batch: list, pad_id: int = 0):
    def pad(seqs):
        L = max(len(s) for s in seqs)
        return torch.tensor([s + [pad_id] * (L - len(s)) for s in seqs], dtype=torch.long)

    return (
        pad([b["chosen_ids"]   for b in batch]),
        pad([b["rejected_ids"] for b in batch]),
        torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
    )


# ── DPO Trainer ───────────────────────────────────────────────────────────────

class DPOTrainer:
    """Direct Preference Optimization trainer."""

    def __init__(
        self,
        model: Transformer,
        tokenizer: BPETokenizer,
        preference_data: List[dict],
        beta: float       = 0.1,
        lr: float         = 5e-5,
        warmup_steps: int = 20,
        max_steps: int    = 200,
        batch_size: int   = 2,
        max_len: int      = 128,
        device: str       = "cpu",
    ):
        self.device = device
        self.beta   = beta
        self.model  = model.to(device)

        # Frozen reference model — snapshot of weights at the start of RLHF
        self.ref_model = copy.deepcopy(model).to(device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        dataset = PreferenceDataset(preference_data, tokenizer, max_len=max_len)
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: _pad_collate(b, pad_id=0),
        )

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = CosineWithWarmup(self.optimizer, warmup_steps, max_steps)

        logger.info(
            f"DPOTrainer ready: {len(dataset)} pairs, "
            f"β={beta}, lr={lr}, max_steps={max_steps}"
        )

    # ── Log-prob computation ──────────────────────────────────────────────────

    def _response_log_prob(
        self,
        model: Transformer,
        input_ids: torch.Tensor,  # (B, T)
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Sum of log probs for response tokens only (tokens at index >= prompt_len).
        Returns shape (B,).
        """
        training = model is self.model
        with torch.set_grad_enabled(training):
            logits, _ = model(input_ids)   # (B, T, V)

        log_probs    = F.log_softmax(logits[:, :-1, :], dim=-1)   # (B, T-1, V)
        targets      = input_ids[:, 1:]                             # (B, T-1)
        token_lp     = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)   # (B, T-1)

        # Mask: only response tokens, ignore padding (id=0)
        mask = torch.zeros_like(token_lp, dtype=torch.bool)
        if prompt_len < token_lp.shape[1]:
            mask[:, prompt_len:] = True
        mask &= targets != 0

        return (token_lp * mask.float()).sum(dim=-1)   # (B,)

    # ── DPO loss ──────────────────────────────────────────────────────────────

    def _dpo_loss(
        self,
        chosen_ids:   torch.Tensor,
        rejected_ids: torch.Tensor,
        prompt_lens:  torch.Tensor,
    ) -> torch.Tensor:
        min_p = int(prompt_lens.min().item())

        pi_w  = self._response_log_prob(self.model,     chosen_ids,   min_p)
        pi_l  = self._response_log_prob(self.model,     rejected_ids, min_p)
        ref_w = self._response_log_prob(self.ref_model, chosen_ids,   min_p)
        ref_l = self._response_log_prob(self.ref_model, rejected_ids, min_p)

        log_ratio = (pi_w - ref_w) - (pi_l - ref_l)
        return -F.logsigmoid(self.beta * log_ratio).mean()

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        epochs: int = 3,
        log_every: int = 10,
        checkpoint_path: Optional[str] = None,
    ) -> List[float]:
        self.model.train()
        losses: List[float] = []
        best_loss = float("inf")
        step = 0

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for chosen_ids, rejected_ids, prompt_lens in self.loader:
                chosen_ids   = chosen_ids.to(self.device)
                rejected_ids = rejected_ids.to(self.device)
                prompt_lens  = prompt_lens.to(self.device)

                loss = self._dpo_loss(chosen_ids, rejected_ids, prompt_lens)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                losses.append(loss.item())
                epoch_loss += loss.item()
                step += 1

                if step % log_every == 0:
                    logger.info(
                        f"DPO step={step:4d}  loss={loss.item():.4f}  "
                        f"lr={self.scheduler.current_lr:.2e}"
                    )

            avg = epoch_loss / max(len(self.loader), 1)
            logger.info(f"── DPO Epoch {epoch}/{epochs}  avg_loss={avg:.4f}")

            if checkpoint_path and avg < best_loss:
                best_loss = avg
                os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "config":      self.model.config.model_dump(),
                    },
                    checkpoint_path,
                )
                logger.info(f"  ↳ RLHF checkpoint saved (avg_loss={avg:.4f})")

        return losses
