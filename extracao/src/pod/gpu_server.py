"""
Servidor FastAPI unificado para GPU no POD.

Expoe endpoints para:
1. Embeddings (BGE-M3) - /embed/*
2. LLM (proxy para vLLM) - /llm/*

Roda no POD e recebe requisicoes HTTP diretas (sem tunel SSH).

Uso (no POD):
    cd /workspace/pipeline/extracao
    python -m src.pod.gpu_server --port 8080

    # Ou com uvicorn
    uvicorn src.pod.gpu_server:app --host 0.0.0.0 --port 8080
"""

import logging
import time
from typing import Optional, Any
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuracao
# ============================================================================

VLLM_BASE_URL = "http://localhost:8000"  # vLLM roda local no POD


# ============================================================================
# Modelos Pydantic
# ============================================================================

# --- Embeddings ---

class EmbedRequest(BaseModel):
    texts: list[str]
    batch_size: int = 8


class EmbedResponse(BaseModel):
    dense: list[list[float]]
    sparse: list[dict]
    count: int
    time_ms: float


# --- LLM ---

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
    gpu_available: bool


# ============================================================================
# Modelo de Embeddings (Singleton)
# ============================================================================

_embed_model = None


def get_embed_model():
    """Carrega modelo BGE-M3 (singleton)."""
    global _embed_model
    if _embed_model is None:
        logger.info("Carregando BGE-M3...")
        start = time.time()
        from FlagEmbedding import BGEM3FlagModel
        _embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        logger.info(f"BGE-M3 carregado em {time.time() - start:.2f}s")
    return _embed_model


# ============================================================================
# Cliente HTTP para vLLM
# ============================================================================

_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Retorna cliente HTTP para vLLM."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            base_url=VLLM_BASE_URL,
            timeout=180.0,
        )
    return _http_client


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa recursos no startup."""
    logger.info("Iniciando GPU Server...")

    # Pre-carrega modelo de embeddings
    get_embed_model()

    # Verifica vLLM
    client = get_http_client()
    try:
        resp = await client.get("/v1/models")
        if resp.status_code == 200:
            models = resp.json()
            logger.info(f"vLLM disponivel: {models['data'][0]['id']}")
    except Exception as e:
        logger.warning(f"vLLM nao disponivel: {e}")

    logger.info("GPU Server pronto!")
    yield

    # Cleanup
    if _http_client:
        await _http_client.aclose()
    logger.info("GPU Server encerrado.")


# ============================================================================
# App FastAPI
# ============================================================================

app = FastAPI(
    title="GPU Server - POD",
    description="API unificada para Embeddings (BGE-M3) e LLM (vLLM)",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Endpoints - Health
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica saude do servidor."""
    import torch

    # Verifica vLLM
    vllm_ok = False
    try:
        client = get_http_client()
        resp = await client.get("/v1/models", timeout=5.0)
        vllm_ok = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        embedding_model_loaded=_embed_model is not None,
        vllm_available=vllm_ok,
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/")
async def root():
    """Pagina inicial."""
    return {
        "service": "GPU Server - POD",
        "endpoints": {
            "health": "/health",
            "embeddings": "/embed/hybrid",
            "llm": "/llm/chat",
        },
        "models": {
            "embedding": "BAAI/bge-m3",
            "llm": "Qwen/Qwen3-8B-AWQ",
        },
    }


# ============================================================================
# Endpoints - Embeddings
# ============================================================================

@app.post("/embed/hybrid", response_model=EmbedResponse)
async def embed_hybrid(request: EmbedRequest):
    """Gera embeddings hibridos (dense + sparse)."""
    try:
        model = get_embed_model()
        start = time.time()

        result = model.encode(
            request.texts,
            batch_size=request.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = result['dense_vecs'].tolist()

        sparse = []
        for s in result['lexical_weights']:
            sparse.append({int(k): float(v) for k, v in s.items()})

        elapsed = (time.time() - start) * 1000

        return EmbedResponse(
            dense=dense,
            sparse=sparse,
            count=len(dense),
            time_ms=elapsed,
        )
    except Exception as e:
        logger.exception(f"Erro ao gerar embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/dense")
async def embed_dense(request: EmbedRequest):
    """Gera apenas embeddings densos."""
    try:
        model = get_embed_model()
        start = time.time()

        result = model.encode(
            request.texts,
            batch_size=request.batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        embeddings = result['dense_vecs'].tolist()
        elapsed = (time.time() - start) * 1000

        return {
            "embeddings": embeddings,
            "count": len(embeddings),
            "time_ms": elapsed,
        }
    except Exception as e:
        logger.exception(f"Erro ao gerar embeddings densos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoints - LLM
# ============================================================================

@app.post("/llm/chat", response_model=ChatResponse)
async def llm_chat(request: ChatRequest):
    """Envia requisicao de chat para vLLM."""
    try:
        client = get_http_client()
        start = time.time()

        # Monta payload para vLLM
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        if request.response_format:
            payload["response_format"] = request.response_format

        # Envia para vLLM
        resp = await client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()

        data = resp.json()
        elapsed = (time.time() - start) * 1000

        # Extrai conteudo
        content = data["choices"][0]["message"]["content"]

        return ChatResponse(
            content=content,
            model=data["model"],
            usage=data.get("usage", {}),
            time_ms=elapsed,
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Erro HTTP do vLLM: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.exception(f"Erro ao chamar vLLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/models")
async def llm_models():
    """Lista modelos disponiveis no vLLM."""
    try:
        client = get_http_client()
        resp = await client.get("/v1/models")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.exception(f"Erro ao listar modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="GPU Server - POD")
    parser.add_argument("--port", type=int, default=8080, help="Porta do servidor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    args = parser.parse_args()

    uvicorn.run(
        "src.pod.gpu_server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
