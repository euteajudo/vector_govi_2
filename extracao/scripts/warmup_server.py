#!/usr/bin/env python3
"""
Servidor de warmup que mantém BGE-M3 e Reranker carregados na GPU.

Expõe uma API simples na porta 8100 para:
- GET /health - Health check
- GET /status - Status dos modelos
- POST /embed - Gerar embeddings
- POST /rerank - Reranking

Uso:
    python scripts/warmup_server.py

    # Em background:
    nohup python scripts/warmup_server.py > /workspace/warmup.log 2>&1 &
"""

import sys
import time
import logging
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS (carregados no startup)
# =============================================================================

app = FastAPI(title="RAG Warmup Server", version="1.0.0")

# Globais para os modelos
_embedder = None
_reranker = None
_load_time = None


def load_models():
    """Carrega modelos na GPU."""
    global _embedder, _reranker, _load_time

    start = time.time()
    logger.info("=" * 50)
    logger.info("Iniciando warmup dos modelos...")
    logger.info("=" * 50)

    # Carrega BGE-M3
    logger.info("Carregando BGE-M3...")
    from model_pool import get_embedder
    _embedder = get_embedder()

    # Teste de sanidade
    test_emb = _embedder.encode(["teste de warmup"])
    logger.info(f"BGE-M3 OK - embedding dim: {len(test_emb[0])}")

    # Carrega Reranker
    logger.info("Carregando BGE-Reranker...")
    from model_pool import get_reranker
    _reranker = get_reranker()

    # Teste de sanidade
    test_score = _reranker.compute_scores("query", ["documento teste"])
    logger.info(f"Reranker OK - test score: {test_score[0]:.4f}")

    _load_time = time.time() - start
    logger.info("=" * 50)
    logger.info(f"Warmup concluído em {_load_time:.2f}s")
    logger.info("Modelos prontos na GPU!")
    logger.info("=" * 50)


# =============================================================================
# API ENDPOINTS
# =============================================================================

class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    count: int


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


class RerankResponse(BaseModel):
    scores: list[float]
    count: int


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "models_loaded": _embedder is not None}


@app.get("/status")
def status():
    """Status detalhado dos modelos."""
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        }

    return {
        "embedder_loaded": _embedder is not None,
        "reranker_loaded": _reranker is not None,
        "load_time_seconds": round(_load_time, 2) if _load_time else None,
        "gpu": gpu_info,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    """Gera embeddings para textos."""
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Embedder não carregado")

    embeddings = _embedder.encode(request.texts)
    return EmbedResponse(
        embeddings=embeddings,
        count=len(embeddings)
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    """Rerank documentos."""
    if _reranker is None:
        raise HTTPException(status_code=503, detail="Reranker não carregado")

    scores = _reranker.compute_scores(request.query, request.documents)
    return RerankResponse(
        scores=scores,
        count=len(scores)
    )


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Carrega modelos no startup."""
    load_models()


if __name__ == "__main__":
    print("=" * 50)
    print("RAG Warmup Server")
    print("Mantém BGE-M3 e Reranker carregados na GPU")
    print("=" * 50)
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )
