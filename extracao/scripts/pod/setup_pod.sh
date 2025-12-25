#!/bin/bash
# Script para configurar GPU Server no POD RunPod
# Executar no Web Terminal do RunPod: bash setup_pod.sh

set -e

echo "======================================"
echo "SETUP GPU SERVER - RUNPOD"
echo "======================================"

# Cria diretorio
mkdir -p /workspace/rag-pipeline

# Cria gpu_server.py
cat > /workspace/rag-pipeline/gpu_server.py << 'ENDOFFILE'
"""
GPU Server unificado para POD.

Combina:
1. Embeddings (BGE-M3) - dense + sparse
2. LLM proxy para vLLM local

Roda na porta 8080 (exposta externamente via RunPod).

Uso:
    python gpu_server.py --port 8080
"""

import os
import time
import asyncio
import logging
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# vLLM local (no mesmo POD)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")

# Modelo de embeddings (lazy load)
_embed_model = None
_embed_lock = asyncio.Lock()


def get_embed_model():
    """Carrega BGE-M3 (lazy loading)."""
    global _embed_model
    if _embed_model is None:
        logger.info("Carregando BGE-M3...")
        start = time.time()
        from FlagEmbedding import BGEM3FlagModel
        _embed_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=True,
            device="cuda",
        )
        logger.info(f"BGE-M3 carregado em {time.time() - start:.2f}s")
    return _embed_model


# HTTP client para vLLM
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=VLLM_BASE_URL,
            timeout=httpx.Timeout(180.0, connect=30.0),
        )
    return _http_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle do app."""
    logger.info("Iniciando GPU Server...")
    logger.info(f"vLLM URL: {VLLM_BASE_URL}")
    yield
    global _http_client
    if _http_client:
        await _http_client.aclose()
    logger.info("GPU Server encerrado")


app = FastAPI(
    title="GPU Server - RAG Pipeline",
    description="Embeddings (BGE-M3) + LLM proxy (vLLM)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Modelos Pydantic
# ============================================================================

class EmbedRequest(BaseModel):
    texts: list[str]
    batch_size: int = 8


class EmbedResponse(BaseModel):
    dense: list[list[float]]
    sparse: list[dict]
    count: int
    time_ms: float


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "Qwen/Qwen3-8B-AWQ"
    temperature: float = 0.0
    max_tokens: int = 512
    response_format: Optional[dict] = None


class ChatResponse(BaseModel):
    content: str
    model: str
    usage: dict
    time_ms: float


class HealthResponse(BaseModel):
    status: str
    embedding_model_loaded: bool
    vllm_available: bool
    vllm_url: str
    gpu_available: bool
    timestamp: str


# ============================================================================
# Endpoints - Health
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Status do servidor."""
    embed_loaded = _embed_model is not None

    # Verifica vLLM
    vllm_ok = False
    try:
        client = await get_http_client()
        resp = await client.get("/health", timeout=5.0)
        vllm_ok = resp.status_code == 200
    except Exception:
        pass

    # Verifica GPU
    gpu_ok = False
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        embedding_model_loaded=embed_loaded,
        vllm_available=vllm_ok,
        vllm_url=VLLM_BASE_URL,
        gpu_available=gpu_ok,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/")
async def root():
    return {
        "service": "GPU Server - RAG Pipeline",
        "endpoints": ["/health", "/embed/hybrid", "/llm/chat", "/llm/models"],
    }


# ============================================================================
# Endpoints - Embeddings
# ============================================================================

@app.post("/embed/hybrid", response_model=EmbedResponse)
async def embed_hybrid(request: EmbedRequest):
    """Gera embeddings dense + sparse com BGE-M3."""
    start = time.time()

    try:
        model = get_embed_model()

        result = model.encode(
            request.texts,
            batch_size=request.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = result['dense_vecs'].tolist()
        sparse = [
            {int(k): float(v) for k, v in s.items()}
            for s in result['lexical_weights']
        ]

        elapsed = (time.time() - start) * 1000

        return EmbedResponse(
            dense=dense,
            sparse=sparse,
            count=len(dense),
            time_ms=elapsed,
        )

    except Exception as e:
        logger.exception(f"Erro em embed_hybrid: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/dense")
async def embed_dense(request: EmbedRequest):
    """Gera apenas embeddings dense."""
    start = time.time()

    try:
        model = get_embed_model()

        result = model.encode(
            request.texts,
            batch_size=request.batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense = result['dense_vecs'].tolist()
        elapsed = (time.time() - start) * 1000

        return {
            "dense": dense,
            "count": len(dense),
            "time_ms": elapsed,
        }

    except Exception as e:
        logger.exception(f"Erro em embed_dense: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoints - LLM (proxy para vLLM)
# ============================================================================

@app.post("/llm/chat", response_model=ChatResponse)
async def llm_chat(request: ChatRequest):
    """Chat completion via vLLM."""
    start = time.time()

    try:
        client = await get_http_client()

        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        if request.response_format:
            payload["response_format"] = request.response_format

        resp = await client.post("/v1/chat/completions", json=payload)

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"vLLM error: {resp.text}"
            )

        data = resp.json()
        elapsed = (time.time() - start) * 1000

        return ChatResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", request.model),
            usage=data.get("usage", {}),
            time_ms=elapsed,
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="vLLM timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"vLLM offline: {VLLM_BASE_URL}")
    except Exception as e:
        logger.exception(f"Erro em llm_chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/models")
async def list_models():
    """Lista modelos disponiveis no vLLM."""
    try:
        client = await get_http_client()
        resp = await client.get("/v1/models", timeout=10.0)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="vLLM error")

        data = resp.json()
        return {
            "models": [m["id"] for m in data.get("data", [])],
            "vllm_url": VLLM_BASE_URL,
        }

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"vLLM offline: {VLLM_BASE_URL}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="GPU Server")
    parser.add_argument("--port", type=int, default=8080, help="Porta")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--vllm-url", type=str, default=None, help="URL do vLLM")
    args = parser.parse_args()

    if args.vllm_url:
        VLLM_BASE_URL = args.vllm_url

    logger.info(f"Iniciando GPU Server em {args.host}:{args.port}")
    logger.info(f"vLLM: {VLLM_BASE_URL}")

    uvicorn.run(
        "gpu_server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
ENDOFFILE

echo "gpu_server.py criado"

# Instala dependencias
echo ""
echo "Instalando dependencias..."
pip install -q fastapi uvicorn httpx pydantic FlagEmbedding

echo ""
echo "======================================"
echo "SETUP COMPLETO!"
echo "======================================"
echo ""
echo "Para iniciar o GPU Server:"
echo "  cd /workspace/rag-pipeline"
echo "  python gpu_server.py --port 8080"
echo ""
echo "O servidor ficara disponivel em:"
echo "  http://localhost:8080 (interno)"
echo "  http://195.26.233.70:55278 (externo)"
echo ""
