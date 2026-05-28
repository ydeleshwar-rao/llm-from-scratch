"""
Tests for LoRA adapter system and RLHF safety filter.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from model.lora import LoRALinear
from model.lora_manager import LoRAManager
from training.rlhf_trainer import safety_score, is_safe


# ── LoRALinear ────────────────────────────────────────────────────────────────

def test_lora_linear_output_shape():
    base = nn.Linear(64, 32, bias=False)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x   = torch.randn(2, 10, 64)
    out = lora(x)
    assert out.shape == (2, 10, 32)


def test_lora_linear_zero_delta_at_init():
    """lora_B is initialized to zero → ΔW = B×A = 0 at init."""
    base = nn.Linear(64, 32, bias=False)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 64)
    with torch.no_grad():
        base_out = base(x)
        lora_out = lora(x)
    assert torch.allclose(base_out, lora_out, atol=1e-6), "LoRA delta must be zero at init"


def test_lora_linear_trains_only_adapter():
    """Gradient flows only through lora_A and lora_B, not the frozen base."""
    base = nn.Linear(32, 16, bias=False)
    base.weight.requires_grad = False
    lora = LoRALinear(base, rank=4, alpha=8.0)

    x    = torch.randn(2, 32)
    loss = lora(x).sum()
    loss.backward()

    assert base.weight.grad is None, "Base weight must have no gradient"
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


def test_lora_linear_scale():
    """scale = alpha / rank."""
    base = nn.Linear(8, 4, bias=False)
    lora = LoRALinear(base, rank=4, alpha=16.0)
    assert lora.scale == 4.0


# ── LoRAManager ──────────────────────────────────────────────────────────────

@pytest.fixture
def small_model():
    from config.model_config import SMALL_CONFIG
    from model.transformer import Transformer
    return Transformer(SMALL_CONFIG)


def test_lora_manager_inject(small_model):
    manager = LoRAManager(rank=4, alpha=8.0)
    manager.inject(small_model)

    lora_layers = [
        m for m in small_model.modules()
        if isinstance(m, LoRALinear)
    ]
    assert len(lora_layers) > 0, "Expected LoRA layers after injection"


def test_lora_manager_freeze_base(small_model):
    manager = LoRAManager(rank=4, alpha=8.0)
    manager.inject(small_model)
    manager.freeze_base(small_model)

    for name, param in small_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_lora_manager_trainable_params(small_model):
    manager = LoRAManager(rank=4, alpha=8.0)
    manager.inject(small_model)
    manager.freeze_base(small_model)

    trainable = manager.trainable_params(small_model)
    total     = list(small_model.parameters())
    assert len(trainable) < len(total), "Should have fewer trainable than total"
    assert len(trainable) > 0


def test_lora_manager_save_load_adapter(small_model):
    manager = LoRAManager(rank=4, alpha=8.0)
    manager.inject(small_model)
    manager.freeze_base(small_model)

    # Modify adapter weights
    for name, param in small_model.named_parameters():
        if "lora_B" in name:
            param.data.fill_(0.5)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        manager.save_adapter(small_model, path)

        # Reset adapter weights to zero
        for name, param in small_model.named_parameters():
            if "lora_B" in name:
                param.data.zero_()

        # Reload
        manager.load_adapter(small_model, path)

        for name, param in small_model.named_parameters():
            if "lora_B" in name:
                assert torch.all(param.data == 0.5), f"{name} should be 0.5 after reload"
    finally:
        os.unlink(path)


def test_lora_manager_save_adapter_only_lora_keys(small_model):
    """Saved adapter file must contain only lora_A / lora_B keys."""
    import torch as _torch
    manager = LoRAManager(rank=4, alpha=8.0)
    manager.inject(small_model)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        manager.save_adapter(small_model, path)
        ckpt = _torch.load(path, map_location="cpu")
        for key in ckpt["adapter_state"]:
            assert "lora_A" in key or "lora_B" in key, f"Unexpected key: {key}"
    finally:
        os.unlink(path)


# ── Safety scoring ────────────────────────────────────────────────────────────

def test_safety_score_clean_text():
    assert safety_score("Hello, how are you today?") == 0.0


def test_safety_score_harmful():
    score = safety_score("I want to kill all enemies with a bomb.")
    assert score >= 0.5


def test_safety_score_sexual():
    score = safety_score("This is explicit porn content.")
    assert score >= 0.5


def test_safety_score_hate():
    score = safety_score("I hate that nigger.")
    assert score >= 0.5


def test_safety_score_capped_at_one():
    score = safety_score("kill murder rape bomb porn nigger")
    assert score <= 1.0


def test_is_safe_clean():
    assert is_safe("The weather is nice today.")


def test_is_safe_harmful():
    assert not is_safe("I will shoot you with a weapon.")


def test_is_safe_threshold():
    assert is_safe("Hello world", threshold=0.5)
    assert not is_safe("kill everyone with bombs", threshold=0.3)
