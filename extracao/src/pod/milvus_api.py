"""
API REST para Milvus - Roda no POD.

Conecta ao Milvus remoto (Windows) via IP publico.
Permite que workers Celery no POD acessem o Milvus.

Uso (no POD):
    # Com Milvus exposto via ngrok ou IP publico
    python -m src.pod.milvus_api --milvus-host <IP_WINDOWS> --milvus-port 19530 --port 8200

    # Ou via variavel de ambiente
    export MILVUS_HOST=<IP_WINDOWS>
    export MILVUS_PORT=19530
    python -m src.pod.milvus_api --port 8200

Endpoints:
    GET  /health              - Status do servidor
    POST /insert              - Insere chunks
    POST /search              - Busca vetorial
    POST /hybrid_search       - Busca hibrida
    POST /query               - Query por filtro
    POST /upsert              - Atualiza ou insere
    DELETE /delete            - Deleta por filtro
    GET  /collections         - Lista collections
    GET  /collection/{name}   - Info de uma collection
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymilvus import connections, Collection, utility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuracao (via env ou argumentos)
# ============================================================================

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
DEFAULT_COLLECTION = "leis_v3"


def set_milvus_config(host: str, port: int):
    """Atualiza configuracao do Milvus."""
    global MILVUS_HOST, MILVUS_PORT
    MILVUS_HOST = host
    MILVUS_PORT = port
    logger.info(f"Milvus configurado: {host}:{port}")


# ============================================================================
# Modelos Pydantic
# ============================================================================

class InsertRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    data: list[dict]


class SearchRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    vector: list[float]
    vector_field: str = "dense_vector"
    top_k: int = 10
    output_fields: list[str] = ["chunk_id", "text", "article_number"]
    filter: Optional[str] = None


class HybridSearchRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    dense_vector: list[float]
    sparse_vector: dict
    top_k: int = 10
    output_fields: list[str] = ["chunk_id", "text", "article_number"]
    filter: Optional[str] = None
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


class QueryRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    filter: str
    output_fields: list[str] = ["chunk_id", "text"]
    limit: int = 100


class DeleteRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    filter: str


class UpsertRequest(BaseModel):
    collection: str = DEFAULT_COLLECTION
    data: list[dict]
    pk_field: str = "chunk_id"


class HealthResponse(BaseModel):
    status: str
    milvus_host: str
    milvus_port: int
    milvus_connected: bool
    collections: list[str]
    timestamp: str


# ============================================================================
# Conexao Milvus
# ============================================================================

_connected = False


def get_connection():
    """Garante conexao com Milvus remoto."""
    global _connected
    try:
        if _connected and connections.has_connection("default"):
            return True

        # Desconecta se existir conexao antiga
        try:
            connections.disconnect("default")
        except Exception:
            pass

        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            timeout=30,
        )
        _connected = True
        logger.info(f"Conectado ao Milvus em {MILVUS_HOST}:{MILVUS_PORT}")
        return True
    except Exception as e:
        _connected = False
        logger.error(f"Erro conectando ao Milvus {MILVUS_HOST}:{MILVUS_PORT}: {e}")
        return False


def get_collection(name: str) -> Collection:
    """Retorna collection pelo nome."""
    if not get_connection():
        raise HTTPException(status_code=503, detail=f"Milvus nao disponivel em {MILVUS_HOST}:{MILVUS_PORT}")
    return Collection(name)


# ============================================================================
# App FastAPI
# ============================================================================

app = FastAPI(
    title="Milvus API - POD",
    description="API REST para acessar Milvus remoto (Windows)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints - Health
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica saude do servidor."""
    connected = get_connection()
    collections = []

    if connected:
        try:
            collections = utility.list_collections()
        except Exception:
            pass

    return HealthResponse(
        status="ok" if connected else "milvus_offline",
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        milvus_connected=connected,
        collections=collections,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/")
async def root():
    """Pagina inicial."""
    return {
        "service": "Milvus API - POD",
        "milvus_target": f"{MILVUS_HOST}:{MILVUS_PORT}",
        "endpoints": [
            "/health",
            "/insert",
            "/search",
            "/hybrid_search",
            "/query",
            "/upsert",
            "/delete",
            "/collections",
        ],
    }


# ============================================================================
# Endpoints - Collections
# ============================================================================

@app.get("/collections")
async def list_collections():
    """Lista todas as collections."""
    if not get_connection():
        raise HTTPException(status_code=503, detail="Milvus offline")
    try:
        collections = utility.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection/{name}")
async def collection_info(name: str):
    """Informacoes de uma collection."""
    try:
        col = get_collection(name)
        return {
            "name": name,
            "num_entities": col.num_entities,
            "schema": str(col.schema),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoints - CRUD
# ============================================================================

@app.post("/insert")
async def insert_data(request: InsertRequest):
    """Insere dados na collection."""
    try:
        col = get_collection(request.collection)

        if not request.data:
            raise HTTPException(status_code=400, detail="Data vazio")

        fields = list(request.data[0].keys())
        data = {field: [] for field in fields}
        for record in request.data:
            for field in fields:
                data[field].append(record.get(field))

        result = col.insert([data[f] for f in fields])
        col.flush()

        return {
            "success": True,
            "insert_count": result.insert_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erro no insert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_vectors(request: SearchRequest):
    """Busca vetorial simples."""
    try:
        col = get_collection(request.collection)
        col.load()

        results = col.search(
            data=[request.vector],
            anns_field=request.vector_field,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=request.top_k,
            output_fields=request.output_fields,
            expr=request.filter,
        )

        hits = []
        for hit in results[0]:
            entity_dict = {}
            for field in request.output_fields:
                if hasattr(hit.entity, field):
                    entity_dict[field] = getattr(hit.entity, field)
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "entity": entity_dict,
            })

        return {"hits": hits, "count": len(hits)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erro no search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hybrid_search")
async def hybrid_search(request: HybridSearchRequest):
    """Busca hibrida (dense + sparse)."""
    try:
        from pymilvus import AnnSearchRequest, WeightedRanker

        col = get_collection(request.collection)
        col.load()

        dense_req = AnnSearchRequest(
            data=[request.dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=request.top_k * 2,
        )

        sparse_dict = {int(k): float(v) for k, v in request.sparse_vector.items()}

        sparse_req = AnnSearchRequest(
            data=[sparse_dict],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_build": 0.2}},
            limit=request.top_k * 2,
        )

        ranker = WeightedRanker(request.dense_weight, request.sparse_weight)

        results = col.hybrid_search(
            [dense_req, sparse_req],
            ranker,
            limit=request.top_k,
            output_fields=request.output_fields,
        )

        hits = []
        for hit in results[0]:
            entity_dict = {}
            for field in request.output_fields:
                if hasattr(hit.entity, field):
                    entity_dict[field] = getattr(hit.entity, field)
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "entity": entity_dict,
            })

        return {"hits": hits, "count": len(hits)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erro no hybrid_search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_data(request: QueryRequest):
    """Query por filtro (sem vetores)."""
    try:
        col = get_collection(request.collection)
        col.load()

        results = col.query(
            expr=request.filter,
            output_fields=request.output_fields,
            limit=request.limit,
        )

        return {"results": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erro no query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upsert")
async def upsert_data(request: UpsertRequest):
    """Atualiza ou insere dados (delete + insert)."""
    try:
        col = get_collection(request.collection)

        pks = [record[request.pk_field] for record in request.data]
        pk_filter = f'{request.pk_field} in {json.dumps(pks)}'

        col.delete(pk_filter)

        fields = list(request.data[0].keys())
        data = {field: [] for field in fields}
        for record in request.data:
            for field in fields:
                data[field].append(record.get(field))

        result = col.insert([data[f] for f in fields])
        col.flush()

        return {
            "success": True,
            "upsert_count": result.insert_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erro no upsert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete")
async def delete_data(request: DeleteRequest):
    """Deleta dados por filtro."""
    try:
        col = get_collection(request.collection)
        result = col.delete(request.filter)
        col.flush()

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erro no delete: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Milvus API Server (POD)")
    parser.add_argument("--port", type=int, default=8200, help="Porta do servidor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--milvus-host", type=str, default=None, help="Host do Milvus")
    parser.add_argument("--milvus-port", type=int, default=None, help="Porta do Milvus")
    args = parser.parse_args()

    if args.milvus_host:
        MILVUS_HOST = args.milvus_host
    if args.milvus_port:
        MILVUS_PORT = args.milvus_port

    logger.info(f"Milvus target: {MILVUS_HOST}:{MILVUS_PORT}")

    uvicorn.run(
        "milvus_api:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
