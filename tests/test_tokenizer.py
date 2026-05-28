"""
Unit tests for BPETokenizer.
"""

import pytest
from tokenizer.bpe_tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer()


def test_vocab_size(tok):
    assert tok.vocab_size == 100277          # cl100k_base


def test_encode_returns_ints(tok):
    ids = tok.encode("Hello world")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_encode_adds_bos(tok):
    with_bos    = tok.encode("hello", add_bos=True)
    without_bos = tok.encode("hello", add_bos=False)
    assert len(with_bos) == len(without_bos) + 1
    assert with_bos[0] == tok.bos_id


def test_encode_adds_eos(tok):
    with_eos    = tok.encode("hello", add_bos=False, add_eos=True)
    without_eos = tok.encode("hello", add_bos=False, add_eos=False)
    assert len(with_eos) == len(without_eos) + 1
    assert with_eos[-1] == tok.eos_id


def test_decode_roundtrip(tok):
    text    = "The quick brown fox jumps over the lazy dog."
    ids     = tok.encode(text, add_bos=False, add_eos=False)
    decoded = tok.decode(ids)
    assert decoded == text


def test_batch_encode(tok):
    texts = ["Hello world", "Transformer model", "LLM from scratch"]
    batch = tok.encode_batch(texts, add_bos=False)
    assert len(batch) == len(texts)
    assert all(isinstance(ids, list) for ids in batch)


def test_batch_decode(tok):
    texts    = ["Hello", "World"]
    encoded  = tok.encode_batch(texts, add_bos=False)
    decoded  = tok.decode_batch(encoded)
    assert decoded == texts


def test_decode_ignores_out_of_range(tok):
    # Should not raise, just silently skip
    result = tok.decode([0, 999999999, 42])
    assert isinstance(result, str)
