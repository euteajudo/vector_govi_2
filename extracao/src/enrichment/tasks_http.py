"""
Celery tasks para enriquecimento de chunks via GPU Server HTTP.

Cada task enriquece um chunk individualmente usando o GPU Server remoto
e atualiza no Milvus local (VPS).

Uso na VPS:
    # Iniciar worker
    cd /root/vector_govi_2/extracao
    source venv/bin/activate
    celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=4

    # Disparar enriquecimento
    python -c "from src.enrichment.tasks_http import enrich_all_chunks_http; enrich_all_chunks_http.delay()"
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .celery_app import app

logger = logging.getLogger(__name__)

# Configuracao via env
import os
GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "http://195.26.233.70:55278")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-8b")  # Modelo 8B
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")  # localhost na VPS
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def enrich_chunk_http(
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
    Enriquece um chunk via GPU Server HTTP e atualiza no Milvus local.

    Args:
        chunk_id: ID do chunk (ex: LEI-14133-2021#ART-005)
        text: Texto do chunk
        device_type: Tipo (article, paragraph, inciso)
        article_number: Numero do artigo
        document_id: ID do documento
        document_type: Tipo (LEI, IN, etc)
        number: Numero do documento
        year: Ano

    Returns:
        Dict com resultado do enriquecimento
    """
    from pymilvus import connections, Collection
    from src.pod.gpu_client import GPUClient
    from src.chunking.enrichment_prompts import (
        CONTEXT_PROMPT_TEMPLATE,
        THESIS_PROMPT_TEMPLATE,
        QUESTIONS_PROMPT_TEMPLATE,
        build_enriched_text,
    )

    start_time = time.time()
    logger.info(f"[TASK] Enriquecendo chunk: {chunk_id}")

    try:
        # Inicializa GPU Client
        gpu_client = GPUClient(GPU_SERVER_URL, timeout=300)

        # 1. Gera context_header
        context_prompt = CONTEXT_PROMPT_TEMPLATE.format(
            document_type=document_type,
            number=number,
            year=year,
            article_number=article_number,
            text=text[:2000],  # Limita texto
        )

        context_result = gpu_client.chat(
            messages=[{"role": "user", "content": context_prompt}],
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=200,
        )
        context_header = context_result.get("content", "").strip()

        # 2. Gera thesis
        thesis_prompt = THESIS_PROMPT_TEMPLATE.format(
            document_type=document_type,
            article_number=article_number,
            text=text[:2000],
        )

        thesis_result = gpu_client.chat(
            messages=[{"role": "user", "content": thesis_prompt}],
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=300,
        )
        thesis_response = thesis_result.get("content", "").strip()

        # Parse thesis response (formato: TIPO: texto)
        thesis_type = "disposicao"
        thesis_text = thesis_response
        if ":" in thesis_response:
            parts = thesis_response.split(":", 1)
            if len(parts) == 2:
                thesis_type = parts[0].strip().lower()
                thesis_text = parts[1].strip()

        # 3. Gera perguntas sinteticas
        questions_prompt = QUESTIONS_PROMPT_TEMPLATE.format(
            document_type=document_type,
            article_number=article_number,
            text=text[:2000],
        )

        questions_result = gpu_client.chat(
            messages=[{"role": "user", "content": questions_prompt}],
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=300,
        )
        questions_raw = questions_result.get("content", "").strip()

        # Parse perguntas (uma por linha)
        synthetic_questions = [
            q.strip().lstrip("0123456789.-) ")
            for q in questions_raw.split("\n")
            if q.strip() and "?" in q
        ][:5]  # Max 5 perguntas

        # 4. Monta enriched_text
        enriched_text = build_enriched_text(
            text=text,
            context_header=context_header,
            synthetic_questions=synthetic_questions,
        )

        # 5. Gera embeddings via GPU Server
        embed_result = gpu_client.embed_hybrid([enriched_text])
        dense_vector = embed_result["dense"][0]
        sparse_vector = embed_result["sparse"][0]

        # Thesis vector
        if thesis_text:
            thesis_embed = gpu_client.embed_hybrid([thesis_text])
            thesis_vector = thesis_embed["dense"][0]
        else:
            thesis_vector = dense_vector

        gpu_client.close()

        # 6. Atualiza no Milvus (localhost na VPS)
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection("leis_v3")
        collection.load()

        # Busca chunk existente
        results = collection.query(
            expr=f'chunk_id == "{chunk_id}"',
            output_fields=["*"],
            limit=1,
        )

        if not results:
            logger.warning(f"[TASK] Chunk nao encontrado no Milvus: {chunk_id}")
            connections.disconnect("default")
            return {"success": False, "chunk_id": chunk_id, "error": "Chunk not found"}

        chunk = results[0]

        # Prepara dados para upsert (row-oriented format)
        row = {
            "chunk_id": chunk["chunk_id"],
            "parent_chunk_id": chunk["parent_chunk_id"],
            "span_id": chunk["span_id"],
            "device_type": chunk["device_type"],
            "chunk_level": chunk["chunk_level"],
            "text": chunk["text"],
            "enriched_text": enriched_text[:65535],
            "dense_vector": dense_vector,
            "thesis_vector": thesis_vector,
            "sparse_vector": sparse_vector,
            "context_header": context_header[:2000],
            "thesis_text": thesis_text[:5000],
            "thesis_type": thesis_type[:100],
            "synthetic_questions": "\n".join(synthetic_questions)[:10000],
            "citations": chunk["citations"],
            "document_id": chunk["document_id"],
            "tipo_documento": chunk["tipo_documento"],
            "numero": chunk["numero"],
            "ano": chunk["ano"],
            "article_number": chunk["article_number"],
            "schema_version": chunk["schema_version"],
            "extractor_version": chunk["extractor_version"],
            "ingestion_timestamp": chunk["ingestion_timestamp"],
            "document_hash": chunk["document_hash"],
            "page": chunk["page"],
            "bbox_left": chunk["bbox_left"],
            "bbox_top": chunk["bbox_top"],
            "bbox_right": chunk["bbox_right"],
            "bbox_bottom": chunk["bbox_bottom"],
        }

        # Delete e insert (upsert)
        collection.delete(expr=f'chunk_id == "{chunk_id}"')
        collection.insert([row])
        collection.flush()

        connections.disconnect("default")

        elapsed = time.time() - start_time
        logger.info(f"[TASK] Chunk enriquecido em {elapsed:.1f}s: {chunk_id}")

        return {
            "success": True,
            "chunk_id": chunk_id,
            "context_header": context_header[:100],
            "thesis_type": thesis_type,
            "questions_count": len(synthetic_questions),
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.error(f"[TASK] Erro em {chunk_id}: {e}")
        import traceback
        traceback.print_exc()
        # Retry com backoff
        raise self.retry(exc=e, countdown=30 * (self.request.retries + 1))


@app.task
def enrich_all_chunks_http() -> dict:
    """
    Dispara enriquecimento para todos os chunks nao enriquecidos.

    Returns:
        Dict com contagem de tasks disparadas
    """
    from pymilvus import connections, Collection
    from celery import group

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection("leis_v3")
    collection.load()

    # Busca chunks sem enriched_text ou com enriched_text = text
    results = collection.query(
        expr='enriched_text == "" or context_header == ""',
        output_fields=["chunk_id", "text", "device_type", "article_number",
                      "document_id", "tipo_documento", "numero", "ano"],
        limit=5000,
    )

    logger.info(f"[BATCH] Encontrados {len(results)} chunks para enriquecer")

    tasks = []
    for chunk in results:
        task = enrich_chunk_http.s(
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            device_type=chunk["device_type"],
            article_number=chunk["article_number"],
            document_id=chunk["document_id"],
            document_type=chunk["tipo_documento"],
            number=chunk["numero"],
            year=chunk["ano"],
        )
        tasks.append(task)

    connections.disconnect("default")

    if tasks:
        # Executa em paralelo
        job = group(tasks)
        result = job.apply_async()

        return {
            "group_id": str(result.id),
            "task_count": len(tasks),
        }

    return {"task_count": 0, "message": "Nenhum chunk para enriquecer"}
