"""
Celery tasks para enriquecimento de chunks.

Cada task enriquece um chunk individualmente e atualiza no Milvus.
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


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def enrich_chunk_task(
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
    Enriquece um chunk individual e atualiza no Milvus.

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
    from src.llm.vllm_client import VLLMClient, LLMConfig
    from src.enrichment.chunk_enricher import ChunkEnricher, DocumentMetadata
    from src.chunking.enrichment_prompts import build_enriched_text
    from src.embeddings.bge_m3 import BGEM3Embedder, EmbeddingConfig

    start_time = time.time()
    logger.info(f"[TASK] Enriquecendo chunk: {chunk_id}")

    try:
        # Inicializa componentes
        llm_config = LLMConfig(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen3-8B-AWQ",
            temperature=0.0,
            max_tokens=1024,
            timeout=300.0,
        )
        llm_client = VLLMClient(config=llm_config)
        enricher = ChunkEnricher(llm_client=llm_client)

        doc_meta = DocumentMetadata(
            document_id=document_id,
            document_type=document_type,
            number=number,
            year=year,
            issuing_body="PRESIDENCIA DA REPUBLICA" if document_type == "LEI" else "SEGES/ME",
        )

        # Enriquece
        enrichment = enricher.enrich_single(
            text=text,
            device_type=device_type,
            article_number=article_number,
            doc_meta=doc_meta,
        )

        if not enrichment:
            logger.warning(f"[TASK] Falha ao enriquecer: {chunk_id}")
            enricher.close()
            return {"success": False, "chunk_id": chunk_id, "error": "Enrichment failed"}

        # Monta enriched_text
        enriched_text = build_enriched_text(
            text=text,
            context_header=enrichment.get("context_header", ""),
            synthetic_questions=enrichment.get("synthetic_questions", []),
        )

        # Gera novos embeddings
        embedder = BGEM3Embedder(EmbeddingConfig(use_fp16=True))
        embed_result = embedder.encode_hybrid([enriched_text])
        dense_vector = embed_result["dense"][0]
        sparse_vector = embed_result["sparse"][0]

        # Thesis vector
        thesis_text = enrichment.get("thesis_text", "")
        if thesis_text:
            thesis_result = embedder.encode_hybrid([thesis_text])
            thesis_vector = thesis_result["dense"][0]
        else:
            thesis_vector = dense_vector

        # Atualiza no Milvus
        connections.connect(host="localhost", port="19530")
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
            enricher.close()
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
            "enriched_text": enriched_text,
            "dense_vector": dense_vector,
            "thesis_vector": thesis_vector,
            "sparse_vector": sparse_vector,
            "context_header": enrichment.get("context_header", ""),
            "thesis_text": thesis_text,
            "thesis_type": enrichment.get("thesis_type", "disposicao"),
            "synthetic_questions": "\n".join(enrichment.get("synthetic_questions", [])),
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
        }

        # Delete e insert (upsert)
        collection.delete(expr=f'chunk_id == "{chunk_id}"')
        collection.insert([row])  # Lista de rows
        collection.flush()

        connections.disconnect("default")
        enricher.close()

        elapsed = time.time() - start_time
        logger.info(f"[TASK] Chunk enriquecido em {elapsed:.1f}s: {chunk_id}")

        return {
            "success": True,
            "chunk_id": chunk_id,
            "context_header": enrichment.get("context_header", "")[:100],
            "thesis_type": enrichment.get("thesis_type", ""),
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.error(f"[TASK] Erro em {chunk_id}: {e}")
        # Retry com backoff
        raise self.retry(exc=e, countdown=30 * (self.request.retries + 1))


@app.task
def enrich_batch_task(chunk_ids: list[str]) -> dict:
    """
    Enriquece um batch de chunks.

    Cria subtasks para cada chunk.

    Args:
        chunk_ids: Lista de chunk_ids para enriquecer

    Returns:
        Dict com IDs das subtasks criadas
    """
    from pymilvus import connections, Collection
    from celery import group

    # Busca dados dos chunks no Milvus
    connections.connect(host="localhost", port="19530")
    collection = Collection("leis_v3")
    collection.load()

    tasks = []
    for chunk_id in chunk_ids:
        results = collection.query(
            expr=f'chunk_id == "{chunk_id}"',
            output_fields=["text", "device_type", "article_number", "document_id", "tipo_documento", "numero", "ano"],
            limit=1,
        )

        if results:
            chunk = results[0]
            task = enrich_chunk_task.s(
                chunk_id=chunk_id,
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

    # Executa em paralelo
    job = group(tasks)
    result = job.apply_async()

    return {
        "group_id": str(result.id),
        "task_count": len(tasks),
    }
