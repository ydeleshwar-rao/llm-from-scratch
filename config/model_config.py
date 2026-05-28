"""
Model configuration — all hyperparameters in one place.
Three preset sizes: SMALL (for testing), MEDIUM, LARGE.
"""

from pydantic import BaseModel, computed_field


class ModelConfig(BaseModel):
    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size: int = 100277          # cl100k_base tiktoken vocab size

    # ── Sequence ──────────────────────────────────────────────────────────────
    max_seq_len: int = 2048

    # ── Transformer dimensions ────────────────────────────────────────────────
    d_model: int = 512                # embedding / hidden dimension
    n_layers: int = 6                 # number of transformer blocks
    n_heads: int = 8                  # query attention heads
    n_kv_heads: int = 4               # key/value heads (GQA: n_kv_heads <= n_heads)
    d_ff: int = 2048                  # FFN intermediate dimension

    # ── Regularization ────────────────────────────────────────────────────────
    dropout: float = 0.0              # 0 for inference, ~0.1 for training

    # ── RoPE position encoding ────────────────────────────────────────────────
    rope_theta: float = 10000.0       # base frequency for rotary embeddings

    @computed_field
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @computed_field
    @property
    def n_groups(self) -> int:
        """How many Q heads share one KV head."""
        return self.n_heads // self.n_kv_heads


# ── Preset sizes (inspired by Claude-style scaling) ───────────────────────────

SMALL_CONFIG = ModelConfig(
    vocab_size=100277,
    max_seq_len=512,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    d_ff=512,
)

MEDIUM_CONFIG = ModelConfig(
    vocab_size=100277,
    max_seq_len=2048,
    d_model=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=4,
    d_ff=2048,
)

LARGE_CONFIG = ModelConfig(
    vocab_size=100277,
    max_seq_len=4096,
    d_model=1024,
    n_layers=16,
    n_heads=16,
    n_kv_heads=8,
    d_ff=4096,
)

# ~500M parameters — needs serious GPU (A100 / H100 recommended)
XLARGE_CONFIG = ModelConfig(
    vocab_size=100277,
    max_seq_len=4096,
    d_model=1024,
    n_layers=36,
    n_heads=16,
    n_kv_heads=8,
    d_ff=4096,
)
