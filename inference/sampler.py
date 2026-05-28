"""
Sampling strategies for text generation.

greedy   — always pick the most probable token (deterministic, repetitive)
top_k    — sample from the top-K most probable tokens (fast, slightly random)
top_p    — nucleus sampling: sample from the smallest set covering p probability mass
           (better quality than top_k, used by Claude and GPT-4 by default)
"""

import torch
import torch.nn.functional as F


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """(B, V) → (B,)"""
    return logits.argmax(dim=-1)


def top_k_sample(
    logits: torch.Tensor, k: int = 50, temperature: float = 1.0
) -> torch.Tensor:
    """(B, V) → (B,)"""
    logits = logits / max(temperature, 1e-8)
    k = min(k, logits.size(-1))
    threshold = torch.topk(logits, k).values[..., -1, None]
    logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs  = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_p_sample(
    logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0
) -> torch.Tensor:
    """Nucleus sampling. (B, V) → (B,)"""
    logits = logits / max(temperature, 1e-8)
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    sorted_probs  = F.softmax(sorted_logits, dim=-1)
    cumprobs      = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens beyond the nucleus (cumulative prob > p)
    # Shift by one so the token that *crosses* the threshold is still included
    remove_mask = cumprobs - sorted_probs > p
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

    # Scatter back to original vocabulary order
    logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    probs  = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
