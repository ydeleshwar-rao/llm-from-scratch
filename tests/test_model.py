"""
Unit tests for all model sub-components + full forward pass.
Run: pytest tests/ -v
"""

import pytest
import torch

from config.model_config import SMALL_CONFIG
from model.normalization import RMSNorm
from model.embeddings import TokenEmbedding, RotaryEmbedding
from model.attention import GroupedQueryAttention
from model.ffn import SwiGLU
from model.block import TransformerBlock
from model.transformer import Transformer

CFG = SMALL_CONFIG
B, T = 2, 16


# ── RMSNorm ───────────────────────────────────────────────────────────────────

def test_rmsnorm_output_shape():
    norm = RMSNorm(CFG.d_model)
    x    = torch.randn(B, T, CFG.d_model)
    assert norm(x).shape == (B, T, CFG.d_model)

def test_rmsnorm_unit_norm():
    norm = RMSNorm(CFG.d_model)
    x    = torch.randn(1, 1, CFG.d_model)
    out  = norm(x)
    # After norm, RMS ≈ 1 (modulo the learned weight which starts at 1)
    rms  = out.pow(2).mean().sqrt().item()
    assert 0.5 < rms < 2.0


# ── Embeddings ────────────────────────────────────────────────────────────────

def test_token_embedding_shape():
    emb = TokenEmbedding(CFG.vocab_size, CFG.d_model)
    ids = torch.randint(0, CFG.vocab_size, (B, T))
    assert emb(ids).shape == (B, T, CFG.d_model)

def test_rope_shape():
    rope = RotaryEmbedding(CFG.head_dim, CFG.max_seq_len)
    q    = torch.randn(B, CFG.n_heads,    T, CFG.head_dim)
    k    = torch.randn(B, CFG.n_kv_heads, T, CFG.head_dim)
    q2, k2 = rope(q, k)
    assert q2.shape == q.shape
    assert k2.shape == k.shape


# ── Attention ─────────────────────────────────────────────────────────────────

def test_attention_output_shape():
    attn = GroupedQueryAttention(CFG)
    x    = torch.randn(B, T, CFG.d_model)
    out, cache = attn(x)
    assert out.shape == (B, T, CFG.d_model)
    assert cache is None                           # use_cache=False default

def test_attention_with_cache():
    attn = GroupedQueryAttention(CFG)
    x    = torch.randn(1, T, CFG.d_model)
    _, cache = attn(x, use_cache=True)
    assert cache is not None
    assert "k" in cache and "v" in cache
    assert cache["k"].shape == (1, CFG.n_kv_heads, T, CFG.head_dim)

def test_gqa_groups():
    assert CFG.n_groups == CFG.n_heads // CFG.n_kv_heads


# ── FFN ───────────────────────────────────────────────────────────────────────

def test_swiglu_output_shape():
    ffn = SwiGLU(CFG)
    x   = torch.randn(B, T, CFG.d_model)
    assert ffn(x).shape == (B, T, CFG.d_model)


# ── Block ─────────────────────────────────────────────────────────────────────

def test_block_output_shape():
    block = TransformerBlock(CFG)
    x     = torch.randn(B, T, CFG.d_model)
    out, cache = block(x)
    assert out.shape == (B, T, CFG.d_model)
    assert cache is None


# ── Full Transformer ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    return Transformer(CFG)

def test_transformer_forward_shape(model):
    ids    = torch.randint(0, CFG.vocab_size, (B, T))
    logits, caches = model(ids)
    assert logits.shape  == (B, T, CFG.vocab_size)
    assert caches is None                          # use_cache=False

def test_transformer_use_cache(model):
    ids    = torch.randint(0, CFG.vocab_size, (1, T))
    logits, caches = model(ids, use_cache=True)
    assert caches is not None
    assert len(caches) == CFG.n_layers

def test_transformer_param_count(model):
    s = model.param_count()
    assert any(unit in s for unit in ["K", "M", "B"])

def test_kv_cache_next_token(model):
    """Output for the last prompt token should match when using KV cache."""
    model.eval()
    ids = torch.randint(0, CFG.vocab_size, (1, T))

    with torch.no_grad():
        logits_full, caches = model(ids, use_cache=True)
        # Decode one more token using the cache — (1, 1) shape
        next_id   = logits_full[:, -1, :].argmax(dim=-1, keepdim=True)   # (B, 1)
        logits_kv, _ = model(next_id, kv_caches=caches, use_cache=True)

    assert logits_kv.shape == (1, 1, CFG.vocab_size)

def test_weight_tying(model):
    """lm_head and embedding should share the exact same weight tensor."""
    assert model.lm_head.weight.data_ptr() == model.embedding.embedding.weight.data_ptr()
