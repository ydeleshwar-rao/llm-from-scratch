# Packages Guide — Kya Install Hai aur Kahan Use Hota Hai

> Har package ka kaam, kyun zaroorat hai, aur project mein exactly kahan use hota hai.

---

## Quick Overview

```
Package          Kaam
────────────────────────────────────────────────
torch            Model ka dil — matrix math, GPU, training
numpy            Numbers aur arrays ka helper
tiktoken         Text ↔ Numbers (tokenizer)
pydantic         Settings aur configs validate karna
fastapi          HTTP API server banana
uvicorn          FastAPI ko actually chalana (web server)
datasets         HuggingFace se training data download karna
python-dotenv    .env file se secrets load karna
tenacity         Retry logic (network errors pe)
pytest           Tests chalana
pytest-asyncio   Async functions test karna
```

---

## 1. `torch` — PyTorch

**Yeh kya hai?**
Machine learning ka sabse popular framework. Model ke andar jo bhi math hota hai —
matrix multiplication, gradients, backpropagation — sab torch karta hai.

**Kyun zaroorat hai?**
Bina torch ke model exist nahi kar sakta. Yeh foundation hai poore project ka.

**Kahan use hota hai:**

| File | Kya karta hai wahan |
|---|---|
| `model/transformer.py` | `nn.Module`, `nn.Linear`, `nn.ModuleList` — model define karna |
| `model/attention.py` | `torch.matmul`, `F.softmax` — attention math |
| `model/embeddings.py` | `nn.Embedding` — words ko vectors banana |
| `model/normalization.py` | `torch.rsqrt` — RMSNorm calculation |
| `training/trainer.py` | `torch.optim.AdamW` — weights update karna |
| `inference/generator.py` | `torch.no_grad()` — inference mein gradients off |
| `main.py` | `torch.cuda.is_available()` — GPU check |

**Example:**
```python
import torch

# Matrix multiplication (attention mein yahi hota hai)
q = torch.randn(2, 4, 512, 32)   # Query
k = torch.randn(2, 4, 512, 32)   # Key
scores = torch.matmul(q, k.transpose(-2, -1))  # Attention scores

# Gradient automatically calculate hota hai
loss.backward()   # ← yeh torch ka magic hai
```

**CPU vs GPU:**
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# cuda  → NVIDIA GPU pe chalao (fast)
# cpu   → normal processor pe chalao (slow but works)
```

---

## 2. `numpy` — Numerical Python

**Yeh kya hai?**
Fast array operations ka library. Scientific computing ka building block.

**Kyun zaroorat hai?**
Torch ke saath compatible hai. Data processing aur math operations mein helper.

**Kahan use hota hai:**
Mostly torch ke background mein use hota hai. Direct use kam hai is project mein,
lekin `datasets` aur `tiktoken` dono internally numpy use karte hain.

**Example:**
```python
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
arr * 2        # → [2.0, 4.0, 6.0]
np.mean(arr)   # → 2.0
```

---

## 3. `tiktoken` — Tokenizer

**Yeh kya hai?**
OpenAI ne banaya BPE (Byte Pair Encoding) tokenizer.
Wahi tokenizer jo GPT-4 mein use hota hai — `cl100k_base` encoding.

**Kyun zaroorat hai?**
Model text nahi samajhta, sirf numbers. Tiktoken text → numbers → text karta hai.

**Kahan use hota hai:**

| File | Kya karta hai |
|---|---|
| `tokenizer/bpe_tokenizer.py` | Poori tokenizer class yahi wrap karti hai |
| `training/dataset.py` | Training texts ko token IDs mein convert |
| `inference/generator.py` | Prompt encode, output decode |
| `serving/api.py` | Token count calculate karna |

**Example:**
```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# Encode: text → numbers
enc.encode("Hello world")     # → [9906, 1917]
enc.encode("Once upon a time") # → [7454, 2402, 264, 892]

# Decode: numbers → text
enc.decode([9906, 1917])       # → "Hello world"

# Vocab size
enc.n_vocab   # → 100,277 total tokens
enc.eot_token # → 100257  (End of Text special token)
```

**BPE kaise kaam karta hai:**
```
"unhappiness"
  → un + happi + ness    (common subwords mein toot ta hai)
  → [84, 71829, 2136]

Faida: 100,277 tokens se poori language cover hoti hai
       naye words bhi handle ho jaate hain (subwords se bana ke)
```

---

## 4. `pydantic` — Data Validation

**Yeh kya hai?**
Python objects ko validate karne ka library. Type checking aur default values
automatically handle karta hai.

**Kyun zaroorat hai?**
Model config aur API request/response ko safe aur validated rakhna.

**Kahan use hota hai:**

| File | Kya karta hai |
|---|---|
| `config/model_config.py` | `ModelConfig` class — model settings validate |
| `serving/api.py` | `GenerateRequest`, `GenerateResponse` — API schemas |

**Example:**
```python
from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str           = Field(..., min_length=1)    # required, min 1 char
    max_new_tokens: int   = Field(default=200, ge=1, le=2048)  # 1-2048 range
    temperature: float    = Field(default=0.8, ge=0.01, le=2.0)

# Automatic validation:
req = GenerateRequest(prompt="Hello", max_new_tokens=5000)
# → Error! max_new_tokens must be ≤ 2048

req = GenerateRequest(prompt="")
# → Error! prompt min_length is 1

req = GenerateRequest(prompt="Hello")
# → OK! max_new_tokens=200 (default use hua)
```

**ModelConfig mein:**
```python
class ModelConfig(BaseModel):
    d_model: int = 512
    n_layers: int = 6

cfg = ModelConfig(d_model=128, n_layers=4)
cfg.d_model   # → 128
cfg.n_layers  # → 4

# model_copy — ek cheez change karke naya config banao
cfg2 = cfg.model_copy(update={"max_seq_len": 32})
```

---

## 5. `fastapi` — Web Framework

**Yeh kya hai?**
Modern, fast Python web framework. REST APIs banane ka sabse popular tool abhi.

**Kyun zaroorat hai?**
Model ko HTTP API ke through accessible banana — taaki koi bhi
application (frontend, mobile, other services) model use kar sake.

**Kahan use hota hai:**

| File | Kya karta hai |
|---|---|
| `serving/api.py` | Poori API yahan define hai |
| `main.py` | `cmd_serve()` — API start karna |

**Example (api.py se):**
```python
from fastapi import FastAPI

app = FastAPI(title="LLM From Scratch")

@app.post("/generate")       # POST request
async def generate(req: GenerateRequest):
    output = generator.generate(req.prompt)
    return {"text": output}

@app.get("/health")          # GET request
def health():
    return {"status": "ok"}
```

**Automatic Swagger UI:**
```
Server start karo: python main.py serve
Browser mein kholo: http://localhost:8001/docs
→ Automatically sab endpoints ka documentation aur test UI mil jaata hai
```

**CORS Middleware:**
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"])
# Matlab: kisi bhi domain se API call ho sakti hai
# (Frontend alag port pe ho toh bhi kaam karega)
```

---

## 6. `uvicorn` — ASGI Web Server

**Yeh kya hai?**
FastAPI ko actually internet pe chalane wala server.
FastAPI sirf "rules" define karta hai — Uvicorn actually HTTP connections handle karta hai.

**Kyun zaroorat hai?**
FastAPI bina uvicorn ke chal nahi sakta. Yeh engine hai, FastAPI blueprint hai.

**Kahan use hota hai:**

| File | Kya karta hai |
|---|---|
| `main.py` | `uvicorn.run(app, host="0.0.0.0", port=8001)` |

**Example:**
```python
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

# host="0.0.0.0" → kisi bhi IP se accessible (local network bhi)
# host="127.0.0.1" → sirf apne computer se accessible
# port=8001       → is port pe suno
```

**`uvicorn[standard]` kya hai?**
```
uvicorn           → basic version
uvicorn[standard] → extra packages ke saath:
                    - websockets support
                    - httptools (faster HTTP parsing)
                    - watchfiles (auto-reload on code change)
```

---

## 7. `datasets` — HuggingFace Datasets

**Yeh kya hai?**
HuggingFace ka library jo thousands of public datasets easily download
aur use karne deta hai.

**Kyun zaroorat hai?**
Training ke liye data chahiye. Yeh library internet se directly data stream kar sakta hai
— poora download kiye bina bhi kaam karta hai (streaming=True).

**Kahan use hota hai:**

| File | Kya karta hai |
|---|---|
| `main.py` | `_load_hf_texts()` — datasets download karna |

**Example:**
```python
from datasets import load_dataset

# Streaming mode — download kiye bina use karo
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

for row in ds:
    print(row["text"])   # → "Once upon a time there was..."
    break

# Poora download karna (badi datasets ke liye streaming better)
ds = load_dataset("Salesforce/wikitext", name="wikitext-2-raw-v1", split="train")
```

**Is project mein available datasets:**
```
tinystories   → roneneldan/TinyStories       (simple stories)
wikitext2     → Salesforce/wikitext          (Wikipedia, chhota)
wikitext103   → Salesforce/wikitext          (Wikipedia, bara)
openwebtext   → Skylion007/openwebtext       (web text)
ptb           → ptb_text_only                (grammar, WSJ articles)
gutenberg     → sedthh/gutenberg_english     (classic books)
grammar-mix   → PTB + WikiText2 + Gutenberg  (mixed grammar)
```

---

## 8. `python-dotenv` — Environment Variables

**Yeh kya hai?**
`.env` file se environment variables load karta hai.

**Kyun zaroorat hai?**
Secrets (API keys, passwords) ko code mein hardcode nahi karna chahiye.
`.env` file mein rakhte hain, git mein push nahi hoti.

**Kahan use hota hai:**
Is project mein abhi direct use nahi hai, lekin ai-service ke saath integrate hone ke liye hai.

**Example:**
```
# .env file:
HF_TOKEN=hf_xxxxxxxxxxxx
OPENAI_KEY=sk-xxxxxxxxxxxx
```

```python
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")  # → "hf_xxxxxxxxxxxx"
```

**HF_TOKEN kyun chahiye?**
```
Bina token:   Rate limited (slow downloads, sometimes blocked)
Token ke saath: Higher rate limits, private datasets bhi accessible
```

---

## 9. `tenacity` — Retry Logic

**Yeh kya hai?**
Network errors ya temporary failures pe automatically retry karne ka library.

**Kyun zaroorat hai?**
HuggingFace se download karte waqt ya AI APIs call karte waqt
network temporarily fail ho sakta hai — tenacity automatically retry karta hai.

**Kahan use hota hai:**
Is project mein future use ke liye include hai (ai-service se common dependency).

**Example:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),           # 3 baar try karo
    wait=wait_exponential(min=1, max=10)  # 1s, 2s, 4s wait
)
def download_data():
    # Agar fail ho → automatically retry
    response = requests.get("https://huggingface.co/...")
    return response.json()
```

---

## 10. `pytest` — Testing Framework

**Yeh kya hai?**
Python ka standard testing library. Code sahi kaam kar raha hai yeh verify karta hai.

**Kyun zaroorat hai?**
Code change karne ke baad automatically check karo ki kuch toot toh nahi gaya.

**Kahan use hota hai:**

| File | Kya test karta hai |
|---|---|
| `tests/test_model.py` | Transformer forward pass sahi hai |
| `tests/test_tokenizer.py` | Encode/decode sahi kaam karta hai |
| `tests/test_inference.py` | Generator text produce karta hai |
| `tests/test_postprocessor.py` | Text cleaning sahi hai |

**Chalana:**
```bash
python main.py test
# Ya directly:
pytest tests/ -v
```

**Example test:**
```python
def test_tokenizer_roundtrip():
    tok = BPETokenizer()
    text = "Hello world"
    ids = tok.encode(text, add_bos=False, add_eos=False)
    decoded = tok.decode(ids)
    assert decoded == text   # ← agar yeh fail ho toh tokenizer mein bug hai
```

---

## 11. `pytest-asyncio` — Async Testing

**Yeh kya hai?**
pytest ka extension jo `async` functions test karne deta hai.

**Kyun zaroorat hai?**
FastAPI endpoints `async def` hain — unhe test karne ke liye yeh chahiye.

**Kahan use hota hai:**
`tests/` folder mein API tests ke liye.

**Example:**
```python
import pytest

@pytest.mark.asyncio
async def test_generate_endpoint():
    response = await client.post("/generate", json={"prompt": "Hello"})
    assert response.status_code == 200
```

---

## Packages aur Files ka Map

```
Package          →  Files jahan use hota hai
─────────────────────────────────────────────────────────
torch            →  model/*, training/trainer.py,
                    inference/generator.py, main.py

numpy            →  (background mein torch/datasets ke saath)

tiktoken         →  tokenizer/bpe_tokenizer.py

pydantic         →  config/model_config.py, serving/api.py

fastapi          →  serving/api.py

uvicorn          →  main.py (cmd_serve)

datasets         →  main.py (_load_hf_texts)

python-dotenv    →  future use / .env loading

tenacity         →  future use / retry logic

pytest           →  tests/*

pytest-asyncio   →  tests/* (async tests)
```

---

## Install Commands

```bash
# Sab ek saath install karo:
pip install -r requirements.txt

# Sirf training ke liye (datasets alag install karna padta hai):
pip install datasets

# Check karo kya install hai:
pip list | grep -E "torch|tiktoken|fastapi|pydantic|datasets"
```

---

## Package Sizes (approximately)

```
torch        ~  800MB  ← sabse bara (neural network engine)
numpy        ~   25MB
tiktoken     ~    5MB
pydantic     ~    2MB
fastapi      ~    1MB
uvicorn      ~    1MB
datasets     ~  528KB  (+ pyarrow ~27MB, pandas ~10MB)
pytest       ~    3MB
```
