"""
FastAPI serving layer for the LLM.

POST /generate  — run inference
GET  /health    — liveness probe
GET  /model     — architecture info
"""

import logging
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM From Scratch", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated by setup() before uvicorn starts
_generator = None
_model     = None


# ── Request / Response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str                = Field(..., min_length=1)
    max_new_tokens: int        = Field(default=200,   ge=1, le=2048)
    strategy: Literal["greedy", "top_k", "top_p"] = "top_p"
    temperature: float         = Field(default=0.8,   ge=0.01, le=2.0)
    top_k: int                 = Field(default=50,    ge=1, le=1000)
    top_p: float               = Field(default=0.9,   ge=0.01, le=1.0)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int


class ModelInfo(BaseModel):
    param_count: str
    config: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(req: GenerateRequest):
    if _generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        prompt_tokens = len(_generator.tokenizer.encode(req.prompt, add_bos=True))
        output = _generator.generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            strategy=req.strategy,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        output_tokens = len(_generator.tokenizer.encode(output, add_bos=False))
        return GenerateResponse(
            text=output,
            prompt_tokens=prompt_tokens,
            generated_tokens=max(output_tokens - prompt_tokens, 0),
        )
    except Exception as exc:
        logger.exception("Generation error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "model_loaded": _generator is not None}


@app.get("/model", response_model=ModelInfo, tags=["System"])
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return ModelInfo(
        param_count=_model.param_count(),
        config=_model.config.model_dump(),
    )


# ── Called from main.py before uvicorn.run() ─────────────────────────────────

def setup(model, generator):
    global _model, _generator
    _model     = model
    _generator = generator
