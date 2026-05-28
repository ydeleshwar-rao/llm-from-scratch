"""
Text dataset for next-token-prediction pre-training.

Tokenizes a list of strings, concatenates them with EOS separators,
then slices into fixed-length (max_seq_len) training chunks.
Each chunk produces (input[:-1], target[1:]) — standard causal LM setup.
"""

from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer.bpe_tokenizer import BPETokenizer


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BPETokenizer, max_seq_len: int):
        self.max_seq_len = max_seq_len
        all_ids: List[int] = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))

        # Slice into overlapping chunks of length max_seq_len + 1
        self.chunks: List[torch.Tensor] = []
        for start in range(0, len(all_ids) - max_seq_len, max_seq_len):
            chunk = all_ids[start : start + max_seq_len + 1]
            if len(chunk) == max_seq_len + 1:
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]   # (input_ids, target_ids)


def make_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
