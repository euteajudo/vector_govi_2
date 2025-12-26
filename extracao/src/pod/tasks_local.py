"""
Celery tasks para enriquecimento LOCAL no POD.

Roda no POD com vLLM e BGE-M3 locais.
Envia resultados para Milvus na VPS.

Uso no POD:
    cd /workspace/extracao
    celery -A src.pod.celery_app worker --loglevel=info --concurrency=10
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from .celery_app import app

logger = logging.getLogger(__name__)

# Configuracao
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "qwen3-8b"
MILVUS_HOST = "77.37.43.160"
MILVUS_PORT = 19530
COLLECTION_NAME = "leis_v3"


def get_vllm_client():
    """Retorna cliente OpenAI para vLLM local."""
    from openai import OpenAI
    return OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed",
    )


def get_embedder():
    """Retorna embedder BGE-M3 local."""
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


# Cache dos modelos (singleton)
_vllm_client = None
_embedder = None


def init_models():
    """Inicializa modelos (uma vez por worker)."""
    global _vllm_client, _embedder
    if _vllm_client is None:
        _vllm_client = get_vllm_client()
        logger.info("vLLM client inicializado")
    if _embedder is None:
        _embedder = get_embedder()
        logger.info("BGE-M3 carregado")
    return _vllm_client, _embedder


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def enrich_chunk_local(
    self,
    chunk_id: str,
    text: str,
    device_type: str,
    article_number: str,
    document_id: str,
    document_type: str,
    number: str,
    year: int,
) -> dict:
    """
    Enriquece um chunk usando vLLM e BGE-M3 locais no POD.

    Args:
        chunk_id: ID do chunk
        text: Texto original
        device_type: article, paragraph, inciso
        article_number: Numero do artigo
        document_id: ID do documento
        document_type: LEI, IN, DECRETO
        number: Numero do documento
        year: Ano

    Returns:
        Dict com resultado
    """
    from pymilvus import connections, Collection

    start_time = time.time()
    logger.info(f"[POD] Enriquecendo: {chunk_id}")

    try:
        vllm, embedder = init_models()

        # 1. Gera context_header via LLM
        context_prompt = f"""Analise o dispositivo legal abaixo e gere UMA FRASE de contexto.

Documento: {document_type} {number}/{year}
Artigo: {article_number}
Texto:
{text[:3000]}

Responda APENAS com uma frase curta (max 150 chars) contextualizando este dispositivo.
Formato: "Este artigo da [documento] [verbo] [assunto principal]"
"""

        context_response = vllm.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": context_prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        context_header = context_response.choices[0].message.content.strip()

        # 2. Gera thesis via LLM
        thesis_prompt = f"""Analise o dispositivo legal e classifique seu tipo.

Texto:
{text[:3000]}

Tipos possiveis: definicao, requisito, procedimento, prazo, vedacao, sancao, excecao, disposicao

Responda no formato:
TIPO: resumo do que o dispositivo determina
"""

        thesis_response = vllm.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": thesis_prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        thesis_raw = thesis_response.choices[0].message.content.strip()

        # Parse thesis
        thesis_type = "disposicao"
        thesis_text = thesis_raw
        if ":" in thesis_raw:
            parts = thesis_raw.split(":", 1)
            thesis_type = parts[0].strip().lower()
            thesis_text = parts[1].strip() if len(parts) > 1 else thesis_raw

        # 3. Gera perguntas sinteticas
        questions_prompt = f"""Liste 3 perguntas que este dispositivo legal responde.

Texto:
{text[:2000]}

Responda APENAS com as perguntas, uma por linha.
"""

        questions_response = vllm.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": questions_prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        questions_raw = questions_response.choices[0].message.content.strip()
        synthetic_questions = [
            q.strip().lstrip("0123456789.-) ")
            for q in questions_raw.split("\n")
            if q.strip() and "?" in q
        ][:5]

        # 4. Monta enriched_text
        parts = []
        if context_header:
            parts.append(f"[CONTEXTO: {context_header}]")
        parts.append(text)
        if synthetic_questions:
            parts.append(f"[PERGUNTAS RELACIONADAS:\n" + "\n".join(f"- {q}" for q in synthetic_questions) + "]")
        enriched_text = "\n\n".join(parts)

        # 5. Gera embeddings localmente
        embed_result = embedder.encode(
            [enriched_text, thesis_text],
            return_dense=True,
            return_sparse=True,
        )

        dense_vector = embed_result["dense"][0].tolist()
        thesis_vector = embed_result["dense"][1].tolist()
        sparse_dict = embed_result["lexical_weights"][0]

        # Converte sparse para formato Milvus
        sparse_vector = {int(k): float(v) for k, v in sparse_dict.items()}

        # 6. Atualiza no Milvus (VPS)
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()

        # Busca chunk existente
        results = collection.query(
            expr=f'chunk_id == "{chunk_id}"',
            output_fields=["*"],
            limit=1,
        )

        if not results:
            logger.warning(f"[POD] Chunk nao encontrado: {chunk_id}")
            connections.disconnect("default")
            return {"success": False, "chunk_id": chunk_id, "error": "Not found"}

        chunk = results[0]

        # Prepara upsert
        row = {
            "chunk_id": chunk["chunk_id"],
            "parent_chunk_id": chunk.get("parent_chunk_id", ""),
            "span_id": chunk.get("span_id", ""),
            "device_type": chunk.get("device_type", device_type),
            "chunk_level": chunk.get("chunk_level", ""),
            "text": chunk["text"],
            "enriched_text": enriched_text[:65535],
            "dense_vector": dense_vector,
            "thesis_vector": thesis_vector,
            "sparse_vector": sparse_vector,
            "context_header": context_header[:2000],
            "thesis_text": thesis_text[:5000],
            "thesis_type": thesis_type[:100],
            "synthetic_questions": "\n".join(synthetic_questions)[:10000],
            "citations": chunk.get("citations", ""),
            "document_id": chunk.get("document_id", document_id),
            "tipo_documento": chunk.get("tipo_documento", document_type),
            "numero": chunk.get("numero", number),
            "ano": chunk.get("ano", year),
            "article_number": chunk.get("article_number", article_number),
            "schema_version": chunk.get("schema_version", "1.0.0"),
            "extractor_version": chunk.get("extractor_version", "1.0.0"),
            "ingestion_timestamp": chunk.get("ingestion_timestamp", ""),
            "document_hash": chunk.get("document_hash", ""),
        }

        # Delete + insert
        collection.delete(expr=f'chunk_id == "{chunk_id}"')
        collection.insert([row])
        collection.flush()

        connections.disconnect("default")

        elapsed = time.time() - start_time
        logger.info(f"[POD] OK em {elapsed:.1f}s: {chunk_id} - {thesis_type}")

        return {
            "success": True,
            "chunk_id": chunk_id,
            "context_header": context_header[:50],
            "thesis_type": thesis_type,
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.error(f"[POD] ERRO {chunk_id}: {e}")
        import traceback
        traceback.print_exc()
        raise self.retry(exc=e, countdown=30 * (self.request.retries + 1))


@app.task
def health_check_local():
    """Verifica saude dos servicos locais."""
    import requests

    results = {}

    # vLLM
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
        results["vllm"] = resp.status_code == 200
    except:
        results["vllm"] = False

    # Milvus
    try:
        from pymilvus import connections
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        connections.disconnect("default")
        results["milvus"] = True
    except:
        results["milvus"] = False

    return results
