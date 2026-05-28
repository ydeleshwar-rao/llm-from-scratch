"""
Autoregressive text generator with KV cache.

Two-phase generation:
  1. Prefill  — process the full prompt in one forward pass, build KV cache
  2. Decode   — generate one token at a time, extending the KV cache each step

KV cache avoids recomputing attention for past tokens on every step,
cutting decode latency from O(n²) to O(n).
"""

import logging
from typing import Literal, Optional

import torch

from model.transformer import Transformer
from tokenizer.bpe_tokenizer import BPETokenizer
from .sampler import greedy, top_k_sample, top_p_sample
from .postprocessor import clean

logger = logging.getLogger(__name__)

Strategy = Literal["greedy", "top_k", "top_p"]


class Generator:
    def __init__(
        self,
        model: Transformer,
        tokenizer: BPETokenizer,
        device: str = "cpu",
    ):
        self.model     = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device    = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int         = 200,
        strategy: Strategy          = "top_p",
        temperature: float          = 0.8,
        top_k: int                  = 50,
        top_p: float                = 0.9,
        truncate_sentence: bool     = False,
    ) -> str:
        input_ids  = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x          = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        kv_caches: Optional[list] = None

        for _ in range(max_new_tokens):
            logits, kv_caches = self.model(x, kv_caches=kv_caches, use_cache=True)
            next_logits = logits[:, -1, :]

            if strategy == "greedy":
                next_id = greedy(next_logits)
            elif strategy == "top_k":
                next_id = top_k_sample(next_logits, k=top_k, temperature=temperature)
            else:
                next_id = top_p_sample(next_logits, p=top_p, temperature=temperature)

            token = next_id.item()
            if token == self.tokenizer.eos_id:
                break
            # skip any other special token the model might emit
            if token in self.tokenizer._special_ids:
                continue

            input_ids.append(token)
            x = torch.tensor([[token]], dtype=torch.long, device=self.device)

        raw = self.tokenizer.decode(input_ids)
        return clean(raw, prompt=prompt, truncate_sentence=truncate_sentence)
