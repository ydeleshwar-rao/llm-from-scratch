"""
BPE Tokenizer wrapper around tiktoken.

tiktoken is already a dependency of the ai-service in this project.
cl100k_base = same encoding used by GPT-4 / Claude family (100k vocab).
"""

from typing import List
import tiktoken


class BPETokenizer:
    # cl100k_base: 100,277 tokens, used by GPT-4 and Claude models
    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc        = tiktoken.get_encoding(encoding_name)
        self.vocab_size  = self._enc.n_vocab
        # <|endoftext|> serves as both BOS and EOS (same convention as GPT-2)
        self.bos_id      = self._enc.eot_token
        self.eos_id      = self._enc.eot_token
        self._special_ids = set(self._enc._special_tokens.values())

    # ── Single-sequence ───────────────────────────────────────────────────────
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        ids = self._enc.encode(text, allowed_special="all")
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Special tokens (BOS/EOS/etc.) live above the regular vocab range.
        # tiktoken raises KeyError if they reach decode(), so strip them out.
        special_ids = set(self._enc._special_tokens.values())
        valid = [i for i in ids if 0 <= i < self.vocab_size and i not in special_ids]
        return self._enc.decode(valid)

    # ── Batch ─────────────────────────────────────────────────────────────────
    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    def decode_batch(self, batch: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch]
