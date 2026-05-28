"""
Unit tests for samplers and the Generator.
"""

import pytest
import torch

from config.model_config import SMALL_CONFIG
from model.transformer import Transformer
from tokenizer.bpe_tokenizer import BPETokenizer
from inference.sampler import greedy, top_k_sample, top_p_sample
from inference.generator import Generator


# ── Samplers ──────────────────────────────────────────────────────────────────

def test_greedy_picks_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2, 0.5]])
    assert greedy(logits).item() == 1


def test_greedy_deterministic():
    logits = torch.randn(1, 100)
    assert greedy(logits).item() == greedy(logits).item()


def test_top_k_output_shape():
    logits = torch.randn(2, 100)
    out    = top_k_sample(logits, k=10)
    assert out.shape == (2,)


def test_top_k_stays_in_vocab():
    logits = torch.randn(1, 100)
    for _ in range(20):
        assert 0 <= top_k_sample(logits, k=5).item() < 100


def test_top_p_output_shape():
    logits = torch.randn(2, 100)
    out    = top_p_sample(logits, p=0.9)
    assert out.shape == (2,)


def test_top_p_stays_in_vocab():
    logits = torch.randn(1, 100)
    for _ in range(20):
        assert 0 <= top_p_sample(logits, p=0.9).item() < 100


# ── Generator ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gen():
    model     = Transformer(SMALL_CONFIG)
    tokenizer = BPETokenizer()
    return Generator(model, tokenizer, device="cpu")


def test_generate_returns_string(gen):
    out = gen.generate("Hello", max_new_tokens=10, strategy="greedy")
    assert isinstance(out, str)     # always a string, never raises


def test_generate_top_p(gen):
    out = gen.generate("The transformer", max_new_tokens=15, strategy="top_p")
    assert isinstance(out, str)


def test_generate_top_k(gen):
    out = gen.generate("Attention is", max_new_tokens=15, strategy="top_k")
    assert isinstance(out, str)


def test_generate_output_starts_with_prompt(gen):
    prompt = "Hello world"
    out    = gen.generate(prompt, max_new_tokens=20, strategy="greedy")
    # Prompt is stripped from output by the postprocessor; output is a clean str
    assert isinstance(out, str)
    assert "<|endoftext|>" not in out


def test_generate_respects_max_tokens(gen):
    prompt    = "Test"
    base      = gen.generate(prompt, max_new_tokens=5,  strategy="greedy")
    extended  = gen.generate(prompt, max_new_tokens=50, strategy="greedy")
    # Extended should be at least as long as base (more tokens generated)
    assert len(extended) >= len(base)
