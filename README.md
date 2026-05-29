# LLM From Scratch

Claude-style decoder-only Transformer — 500M parameters, built from scratch in PyTorch.

> **Full guide:** See [GUIDE.md](GUIDE.md) for all commands.
> **Attention explained:** See [ATTENTION.md](ATTENTION.md)
> **Hardware guide:** See [HARDWARE.md](HARDWARE.md)

---

## Quick Start

```bash
# Local (CPU — small model)
python main.py train complete-mix 5000 --size small
python main.py generate --size small

# Kaggle T4 GPU (500M model)
python main.py train complete-mix 20000 --size xlarge --hf-repo username/my-llm-xlarge
python main.py generate --size xlarge
```

---

## Model Sizes

| Size | Parameters | RAM needed | Use case |
|------|-----------|-----------|---------|
| small | 13.6M | 2 GB | Local CPU testing |
| medium | 80M | 6 GB | Mid-range GPU |
| large | 288M | 12 GB | High-end GPU |
| xlarge | 520M | 8 GB (with fixes) | Kaggle T4 |

---

## Training Datasets

| Dataset | Topic | Size |
|---------|-------|------|
| `wikitext103` | English grammar | 1M+ |
| `emotion` | Emotions (joy/anger/sadness) | 400K |
| `social_iqa` | Human behaviour | 37K |
| `customer_support` | Customer service | 10K+ |
| `sales` | Sales conversations | 100K |
| `daily_dialog` | Daily conversations | 11K |
| `hate_speech` | Abuse recognition (English) | 24K |
| `hate_speech_hindi` | Abuse recognition (Hindi) | 5K |
| `sexual_health` | Health education (medical) | 100K |
| `prosocial` | Non-violent behaviour | 58K |
| `hindi` | Hindi language | 1.6M |
| `spell_correction` | Spelling typo pairs | 49K |
| `grammar_correction` | Grammar error pairs | 125K |
| `noisy_english` | Noisy Twitter text | 1.6M |

**Recommended mix:** `complete-mix` — trains on all of the above together.

---

## Kaggle Setup (Free 500M Training)

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")

os.chdir("/kaggle/working")
if os.path.exists("llm-from-scratch"):
    os.chdir("llm-from-scratch")
    os.system("git pull")
else:
    os.system("git clone https://github.com/ydeleshwar-rao/llm-from-scratch.git")
    os.chdir("llm-from-scratch")

os.system("pip install tiktoken datasets pydantic huggingface_hub bitsandbytes -q")

os.system("python main.py train complete-mix 20000 --size xlarge --hf-repo username/my-llm-xlarge")
```

**Requirements:**
- Kaggle account (free)
- GPU T4 x2 enabled in Settings
- Internet ON (phone verification needed once)
- HuggingFace token saved as Kaggle Secret `HF_TOKEN`

---

## Problems Faced & How We Fixed Them

### Problem 1 — CUDA Out of Memory (Attention)

**Error:**
```
torch.OutOfMemoryError: Tried to allocate 2.00 GiB
at model/attention.py line 73: torch.matmul(q, k.transpose(-2,-1))
```

**Root cause:**
```
seq_len=4096, batch=2, heads=16
Attention matrix = 2 × 16 × 4096 × 4096 × 4 bytes = 2 GB
T4 only had 1.96 GB free → crash
```

**Fix:**
```python
# config/model_config.py
XLARGE_CONFIG = ModelConfig(
    max_seq_len=1024,   # 4096 → 1024 (16x less attention memory)
    ...
)

# main.py
batch_size = 1  # xlarge ke liye batch=1
```

---

### Problem 2 — CUDA Out of Memory (Backward Pass)

**Error:**
```
torch.OutOfMemoryError: Tried to allocate 392 MiB
at training/trainer.py: loss.backward()
```

**Root cause:**
```
fp32 training pe activations bahut zyada memory lete hain
500M model × fp32 = 14GB — T4 full ho jaata hai
```

**Fix — fp16 Mixed Precision:**
```python
# training/trainer.py
from torch.amp import GradScaler, autocast

self.scaler = GradScaler("cuda")

with autocast("cuda"):
    logits, _ = self.model(x)
    loss = self.criterion(...)

self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

**Fix — Gradient Checkpointing:**
```python
# model/block.py
# Activations recompute during backward instead of storing
if self.use_checkpoint and self.training:
    x = checkpoint_util.checkpoint(ckpt_fn, x, use_reentrant=False)
```

**Memory saved:**
```
fp16:                 forward memory 50% kam
gradient checkpoint:  activation memory 3x kam
Together:             8 GB total (14.5 GB mein fit)
```

---

### Problem 3 — CUDA Out of Memory (Optimizer Init)

**Error:**
```
torch.OutOfMemoryError: Tried to allocate 12.00 MiB
at torch/optim/adam.py: state["exp_avg"] = torch.zeros_like(...)
```

**Root cause:**
```
Standard Adam optimizer = 2 tensors per parameter
520M params × 2 × 4 bytes = 4.16 GB just for optimizer
Model (2GB) + gradients (2GB) + Adam (4GB) = 8GB → crash
```

**Fix — 8-bit Adam:**
```python
# training/trainer.py
import bitsandbytes as bnb
self.optimizer = bnb.optim.AdamW8bit(
    model.parameters(), lr=lr, weight_decay=0.1
)
# Optimizer memory: 4 GB → 1 GB (4x reduction)
```

**Final memory breakdown:**
```
Model weights (fp32):  2.0 GB
fp16 copy:             1.0 GB
Gradients:             2.0 GB
8-bit Adam:            1.0 GB   ← was 4.0 GB
Activations:           2.0 GB
──────────────────────────────
Total:                 8.0 GB   ← fits in T4 (14.5 GB) ✅
```

---

### Problem 4 — Kaggle Session Expire → Data Loss

**Problem:**
```
Kaggle free session = 12 hour limit
Session expire → /kaggle/working/ wipe → model.pt gone
Hours of training lost
```

**Fix — HuggingFace Hub auto-save:**
```python
# training/trainer.py
def _push_to_hub(local_path, hf_repo, hf_token, step):
    api = HfApi()
    api.create_repo(repo_id=hf_repo, exist_ok=True, token=hf_token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo="model.pt",
        repo_id=hf_repo,
        token=hf_token,
        commit_message=f"checkpoint step={step}",
    )

# Every 500 steps → auto push to HF Hub
if step % save_every == 0:
    self.save(checkpoint_path)
    _push_to_hub(checkpoint_path, self.hf_repo, self.hf_token, step)
```

**Resume from HF Hub:**
```python
# main.py — on new session, auto-download from HF
def _download_from_hub(hf_repo, hf_token, local_path):
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=hf_repo, filename="model.pt", ...)
```

**Now:**
```
Session expire → koi problem nahi ✅
New session → HF Hub se resume ✅
```

---

### Problem 5 — HuggingFace Repo Not Found (404)

**Error:**
```
httpx — HTTP Request: POST .../ydeleshwar-rao/my-llm-xlarge/preupload/main
"HTTP/1.1 404 Not Found"
Repository Not Found
```

**Root causes:**
1. Repo exist nahi karta tha
2. Username mismatch: `ydeleshwar-rao` vs `ydeleshwarrao`

**Fix:**
```python
# Auto-create repo if not exists
api.create_repo(
    repo_id=hf_repo,
    repo_type="model",
    token=hf_token,
    exist_ok=True,   # no error if already exists
)
```

---

### Problem 6 — Catastrophic Forgetting

**Problem:**
```
Pehle tinystories train kiya (5000 examples)
Phir daily_dialog train kiya
→ Model tinystories bhool gaya completely
```

**Fix — --mix flag:**
```bash
# Sab datasets ek saath mix karke train karo
python main.py train --mix tinystories:3000 daily_dialog:2000 wikitext2:2000
python main.py train complete-mix 20000
```

**Fix — Incremental training:**
```python
# Existing checkpoint load karo, toot ke shuru mat karo
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path)["model_state"])
    # Training jaari rakhta hai jahan choda tha
```

---

## Architecture

```
Input tokens
     ↓
TokenEmbedding (100,277 vocab × d_model)
     ↓
TransformerBlock × N
  ├── RMSNorm
  ├── GroupedQueryAttention + RoPE
  ├── RMSNorm
  └── SwiGLU FFN
     ↓
RMSNorm
     ↓
lm_head → 100,277 vocab logits
     ↓
Next token probabilities
```

**Key features:**
- RMSNorm (faster than LayerNorm)
- RoPE position encoding
- Grouped Query Attention (smaller KV cache)
- SwiGLU activation
- Weight tying (embedding = lm_head)
- KV Cache for fast inference

---

## Tests

```bash
python main.py test   # 81 unit tests
```
