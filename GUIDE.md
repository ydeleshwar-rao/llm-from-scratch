# LLM From Scratch — Complete Guide

> Claude-style decoder-only Transformer, built from scratch in PyTorch.
> LoRA adapter system + RLHF safety alignment included.

---

## Project Structure

```
llm-from-scratch/
│
├── main.py                      ← Sab commands yahan se chalte hain
│
├── config/
│   └── model_config.py          ← Model sizes (SMALL / MEDIUM / LARGE)
│
├── model/
│   ├── transformer.py           ← Pura model
│   ├── attention.py             ← Grouped Query Attention + RoPE
│   ├── embeddings.py            ← Token + Rotary Position embeddings
│   ├── normalization.py         ← RMSNorm
│   ├── ffn.py                   ← SwiGLU Feed Forward Network
│   ├── block.py                 ← Ek transformer layer
│   ├── lora.py                  ← LoRALinear (trainable A×B adapter)
│   └── lora_manager.py          ← Inject / freeze / save / load adapters
│
├── tokenizer/
│   └── bpe_tokenizer.py         ← Text ↔ Numbers (tiktoken cl100k_base)
│
├── training/
│   ├── dataset.py               ← Text → training chunks
│   ├── trainer.py               ← Base model training loop
│   ├── lora_trainer.py          ← Adapter-only training loop
│   ├── rlhf_trainer.py          ← DPO safety alignment + safety_score()
│   └── scheduler.py             ← Cosine LR with warmup
│
├── inference/
│   ├── generator.py             ← Autoregressive text generation + KV cache
│   ├── sampler.py               ← greedy / top_k / top_p
│   └── postprocessor.py         ← Output clean karna (garbage remove)
│
├── serving/
│   └── api.py                   ← FastAPI server (port 8001)
│
├── tests/                       ← 81 unit tests
│
└── checkpoints/
    ├── base/       model.pt      ← Stage 1: base language model
    ├── rlhf/       model.pt      ← Stage 2: safety-aligned model
    └── adapters/   NAME.pt       ← Stage 3: LoRA adapter (one per use case)
```

---

## Sab Commands

```bash
python main.py info                                  # Model sizes dekhna

python main.py train [dataset] [n]                   # Base model train karo
python main.py train-rlhf [pairs.json]               # Safety alignment (DPO)
python main.py train-adapter NAME [dataset] [n]      # LoRA adapter train karo

python main.py generate [--adapter NAME]             # Text generate karo
python main.py serve    [--adapter NAME]             # API server start karo

python main.py test                                  # 81 tests chalao
```

---

## 3-Stage Training Pipeline

### Stage 1 — Base Model (Language Foundation)

Yahan model grammar, sentence structure, aur natural language seekhta hai.

```bash
python main.py train tinystories 5000    # Recommended: children's stories
python main.py train wikitext2 3000      # Wikipedia (formal English)
python main.py train wikitext103 10000   # Bada Wikipedia dump
python main.py train grammar-mix 6000    # PTB + WikiText2 + Gutenberg (best)
python main.py train                     # Sirf 15 built-in sentences (testing only)
```

**Saves to:** `checkpoints/base/model.pt`

---

### Stage 2 — RLHF Safety Alignment (DPO)

Model ko sikhaao ki harmful/sexual responses mat do.
DPO (Direct Preference Optimization) use hota hai — reward model ki zarurat nahi.

```bash
python main.py train-rlhf                # Built-in 8 sample pairs (demo)
python main.py train-rlhf my_pairs.json  # Apna data dalo (recommended)
```

**JSON format:**
```json
[
  {
    "prompt": "How do I solve a problem?",
    "chosen": "Think calmly and break it into small steps.",
    "rejected": "Blame others and give up."
  },
  {
    "prompt": "How do I handle anger?",
    "chosen": "Take a breath and talk to someone you trust.",
    "rejected": "Hurt the person who made you angry."
  }
]
```

**Reads from:** `checkpoints/base/model.pt`
**Saves to:**   `checkpoints/rlhf/model.pt`

---

### Stage 3 — LoRA Adapter (Use-Case Specialization)

Base model freeze rakhke sirf ek chota adapter train karo.
Har use case ka alag adapter hoga — base model kabhi nahi badlega.

```bash
python main.py train-adapter customer_service tinystories 2000
python main.py train-adapter qa_bot wikitext2 3000
python main.py train-adapter my_chatbot tinystories 1000
```

**Reads from:** `checkpoints/rlhf/model.pt` (ya `base/model.pt` agar RLHF nahi hai)
**Saves to:**   `checkpoints/adapters/customer_service.pt`

**LoRA kyun use karte hain?**

```
Base model:  13.6M parameters  → ~55 MB file
LoRA adapter:   65K parameters  → ~0.5 MB file

Full model retrain karo → 55 MB + lambi training
LoRA adapter train karo → 0.5 MB + bahut fast training

Aur ek base model pe 10 adapters rakh sakte ho — ek hi GPU mein!
```

---

## Generate Karna

```bash
python main.py generate                            # RLHF / base model
python main.py generate --adapter customer_service # Adapter ke saath
python main.py generate --adapter qa_bot
```

---

## API Server

```bash
python main.py serve                               # Plain server
python main.py serve --adapter customer_service    # Adapter ke saath
```

Browser mein kholo: `http://localhost:8001/docs`

**API request example:**
```json
POST http://localhost:8001/generate
{
  "prompt": "Once upon a time",
  "max_new_tokens": 100,
  "strategy": "top_p",
  "temperature": 0.8
}
```

---

## Checkpoint Delete Kaise Karo

### Purana checkpoint delete karna (Windows PowerShell):

```powershell
# Base model delete (fresh retrain ke liye):
Remove-Item checkpoints\base\model.pt

# RLHF checkpoint delete:
Remove-Item checkpoints\rlhf\model.pt

# Ek specific adapter delete:
Remove-Item checkpoints\adapters\customer_service.pt

# Saare adapters delete:
Remove-Item checkpoints\adapters\*

# Sab kuch delete (full reset):
Remove-Item checkpoints\base\*, checkpoints\rlhf\*, checkpoints\adapters\*
```

### File Explorer se delete:

```
checkpoints/
├── base/       → model.pt yahan hai → select → Delete
├── rlhf/       → model.pt yahan hai → select → Delete
└── adapters/   → NAME.pt yahan hai  → select → Delete
```

### Zaruri baat — delete karne ke baad:

```
Agar base/model.pt delete kiya:
  → Dobara train karna hoga: python main.py train tinystories 5000
  → Phir RLHF:               python main.py train-rlhf
  → Phir adapters:           python main.py train-adapter NAME

Agar sirf rlhf/model.pt delete kiya:
  → Base model safe hai
  → Dobara RLHF:             python main.py train-rlhf

Agar sirf adapter delete kiya:
  → Base + RLHF safe hain
  → Dobara adapter:          python main.py train-adapter NAME dataset n
```

> **Note:** TinyStories test checkpoint (purana wala) already delete ho chuka hai.
> Naya training agar karna hai toh `python main.py train tinystories 5000` chalao.

---

## Safety Filter — Har jagah Active

`safety_score()` function automatically run hota hai generated text pe.

```python
from training.rlhf_trainer import safety_score, is_safe

score = safety_score("I will kill you")   # → 0.5 (harmful detected)
score = safety_score("Hello how are you") # → 0.0 (clean)

safe = is_safe("Hello world")             # → True
safe = is_safe("bomb attack weapon")      # → False
```

**Kya detect karta hai:**
- Harmful content: violence, weapons, self-harm keywords
- Sexual content: explicit / NSFW keywords
- Hate speech: slurs and derogatory terms

---

## Available Datasets (HuggingFace)

| Dataset | Content | Best for |
|---------|---------|----------|
| `tinystories` | Children's stories (simple, clean English) | Base model (shuru karo yahan se) |
| `wikitext2` | Wikipedia articles | Grammar + factual knowledge |
| `wikitext103` | Larger Wikipedia dump | Better grammar |
| `grammar-mix` | PTB + WikiText-2 + Gutenberg | Sabse strong grammar base |
| `ptb` | Penn Treebank sentences | Grammar research |
| `gutenberg` | Classic English literature | Rich vocabulary |

Pehli baar download hoga (few minutes). Baad mein cache mein hoga.

---

## Model Sizes

| Config | Parameters | Use case |
|--------|-----------|----------|
| SMALL  | 13.6M | Fast training, local testing (default) |
| MEDIUM | 74.9M | Better quality (needs more data + GPU) |
| LARGE  | 288M  | Production-level (needs serious GPU + large dataset) |

---

## Loss aur LR — Quick Reference

```
Loss kya matlab hai:
  ~11.5 = Random model (training shuru)
  ~7-9  = Kuch seekha
  ~4-6  = Decent (TinyStories ke baad)
  ~2-3  = Good model
  ~<1.5 = Excellent

Learning Rate (3e-4 = 0.0003):
  Warmup phase: LR 0 se peak tak badhta hai (training stable hoti hai)
  Cosine decay: LR dhire dhire garta hai (fine-tuning)
  Min LR 1e-5:  Training end
```

---

## Architecture (Claude-style, Decoder-Only)

```
Input token IDs  (e.g. [100257, 7454, 2402])
       ↓
TokenEmbedding   (token IDs → d_model=128 vectors, scaled by √128)
       ↓
TransformerBlock × 4  (n_layers=4)
  ├── RMSNorm
  ├── GroupedQueryAttention + RoPE  (4 Q heads, 2 KV heads)
  ├── RMSNorm
  └── SwiGLU FFN  (128 → 512 → 128)
       ↓
RMSNorm
       ↓
lm_head  (128 → 100,277 vocab logits)  [weight-tied to embedding]
       ↓
Next token probabilities
```

**Key features:**
- **RMSNorm** — LayerNorm se faster (mean subtraction skip)
- **RoPE** — Position info directly Q/K vectors mein encode
- **GQA** — Fewer KV heads → smaller KV cache → faster inference
- **SwiGLU** — Better activation than GELU or ReLU
- **Weight tying** — lm_head aur embedding ek hi matrix share karte hain
- **KV Cache** — Decode phase mein O(n) instead of O(n²)

---

## Tests

```bash
python main.py test    # Ya directly:
pytest tests/ -v
```

81 tests: model architecture, tokenizer, inference, postprocessor, LoRA, safety scoring.
