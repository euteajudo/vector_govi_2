"""
Servidor FastAPI para embeddings BGE-M3.

Roda no POD e expoe API para gerar embeddings via HTTP.
Mantem o modelo BGE-M3 carregado na GPU para baixa latencia.

Uso (no POD):
    cd /workspace/pipeline/extracao
    python -m src.pod.embedding_server

    # Ou com uvicorn
    uvicorn src.pod.embedding_server:app --host 0.0.0.0 --port 8100
"""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelo global (singleton)
_model = None


def get_model():
    """Carrega modelo BGE-M3 (singleton)."""
    global _model
    if _model is None:
        logger.info("Carregando BGE-M3...")
        start = time.time()
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        logger.info(f"BGE-M3 carregado em {time.time() - start:.2f}s")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo no startup."""
    logger.info("Iniciando servidor de embeddings...")
    get_model()  # Pre-carrega
    logger.info("Servidor pronto!")
    yield
    logger.info("Encerrando servidor...")


app = FastAPI(
    title="BGE-M3 Embedding Server",
    description="API para gerar embeddings BGE-M3 (dense + sparse)",
    version="1.0.0",
    lifespan=lifespan,
)


# === Request/Response Models ===

class EncodeRequest(BaseModel):
    texts: list[str]
    batch_size: int = 8


class DenseResponse(BaseModel):
    embeddings: list[list[float]]
    count: int
    time_ms: float


class SparseResponse(BaseModel):
    embeddings: list[dict]
    count: int
    time_ms: float


class HybridResponse(BaseModel):
    dense: list[list[float]]
    sparse: list[dict]
    count: int
    time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool


# === Endpoints ===

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica saude do servidor."""
    import torch
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/encode/dense", response_model=DenseResponse)
async def encode_dense(request: EncodeRequest):
    """Gera embeddings densos."""
    try:
        model = get_model()
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

        return DenseResponse(
            embeddings=embeddings,
            count=len(embeddings),
            time_ms=elapsed,
        )
    except Exception as e:
        logger.exception(f"Erro ao gerar embeddings densos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode/sparse", response_model=SparseResponse)
async def encode_sparse(request: EncodeRequest):
    """Gera embeddings esparsos (lexical weights)."""
    try:
        model = get_model()
        start = time.time()

        result = model.encode(
            request.texts,
            batch_size=request.batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        # Converte para formato serializavel
        embeddings = []
        for sparse in result['lexical_weights']:
            # sparse e um dict {token_id: weight}
            embeddings.append({int(k): float(v) for k, v in sparse.items()})

        elapsed = (time.time() - start) * 1000

        return SparseResponse(
            embeddings=embeddings,
            count=len(embeddings),
            time_ms=elapsed,
        )
    except Exception as e:
        logger.exception(f"Erro ao gerar embeddings esparsos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode/hybrid", response_model=HybridResponse)
async def encode_hybrid(request: EncodeRequest):
    """Gera embeddings hibridos (dense + sparse)."""
    try:
        model = get_model()
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

        return HybridResponse(
            dense=dense,
            sparse=sparse,
            count=len(dense),
            time_ms=elapsed,
        )
    except Exception as e:
        logger.exception(f"Erro ao gerar embeddings hibridos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Pagina inicial."""
    return {
        "service": "BGE-M3 Embedding Server",
        "endpoints": [
            "/health",
            "/encode/dense",
            "/encode/sparse",
            "/encode/hybrid",
        ],
        "model": "BAAI/bge-m3",
        "dimension": 1024,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.pod.embedding_server:app",
        host="0.0.0.0",
        port=8100,
        log_level="info",
    )
