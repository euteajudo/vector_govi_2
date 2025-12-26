"""
Celery tasks para enriquecimento - RODANDO NO POD.

GPU Server LOCAL (localhost:8000) + Milvus VPS (77.37.43.160:19530)

Uso no POD:
    # Iniciar Redis
    redis-server --daemonize yes

    # Iniciar worker
    cd /workspace/rag-pipeline/extracao
    export PYTHONPATH=/workspace/rag-pipeline/extracao/src:/workspace/pip-packages
    celery -A src.enrichment.tasks_pod worker --loglevel=info --concurrency=4

    # Iniciar Flower
    celery -A src.enrichment.tasks_pod flower --port=5555 --address=0.0.0.0
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Celery config
from celery import Celery

app = Celery(
    "enrichment_pod",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/Sao_Paulo",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    task_soft_time_limit=300,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)

logger = logging.getLogger(__name__)

# Configuracao POD - GPU local, Milvus remoto
GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "http://localhost:8080")  # GPU Server (8080)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-8b")
MILVUS_HOST = os.getenv("MILVUS_HOST", "77.37.43.160")  # VPS
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# System prompt para desabilitar thinking e forçar português
SYSTEM_PROMPT = "Você é um assistente especializado em direito administrativo brasileiro. Responda sempre em português do Brasil, de forma direta e concisa. /no_think"


def clean_llm_response(text: str) -> str:
    """Remove tags <think>...</think> e limpa a resposta do LLM."""
    import re
    # Remove <think>...</think> incluindo conteúdo
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove <think> sozinho (caso não fechado)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    # Remove espaços extras
    text = text.strip()
    return text


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def enrich_chunk_pod(
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
    Enriquece um chunk usando GPU LOCAL e atualiza no Milvus VPS.
    """
    import requests
    from pymilvus import connections, Collection

    start_time = time.time()
    logger.info(f"[POD] Enriquecendo chunk: {chunk_id}")

    try:
        # 1. Gera context_header via vLLM LOCAL
        context_prompt = f"""Dado o seguinte trecho de {document_type} {number}/{year}, Art. {article_number}:

{text[:2000]}

Escreva UMA FRASE curta (max 50 palavras) que contextualize este artigo no documento.
Responda APENAS com a frase, sem explicacoes."""

        context_resp = requests.post(
            f"{GPU_SERVER_URL}/llm/chat",
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 100,
            },
            timeout=60,
        )
        context_resp.raise_for_status()
        context_header = clean_llm_response(context_resp.json()["content"])

        # 2. Gera thesis via vLLM LOCAL
        thesis_prompt = f"""Analise este dispositivo legal:

{text[:2000]}

Classifique e resuma:
1. TIPO: definicao, procedimento, obrigacao, proibicao, excecao, ou prazo
2. RESUMO: Uma frase com a essencia do dispositivo

Formato: TIPO: resumo
Exemplo: procedimento: Define os passos para elaboracao do ETP"""

        thesis_resp = requests.post(
            f"{GPU_SERVER_URL}/llm/chat",
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": thesis_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 150,
            },
            timeout=60,
        )
        thesis_resp.raise_for_status()
        thesis_response = clean_llm_response(thesis_resp.json()["content"])

        # Parse thesis
        thesis_type = "disposicao"
        thesis_text = thesis_response
        if ":" in thesis_response:
            parts = thesis_response.split(":", 1)
            if len(parts) == 2:
                thesis_type = parts[0].strip().lower()
                thesis_text = parts[1].strip()

        # 3. Gera perguntas sinteticas
        questions_prompt = f"""Liste 3 perguntas em português que este artigo responde:

{text[:1500]}

Formato: uma pergunta por linha, terminando com ?"""

        questions_resp = requests.post(
            f"{GPU_SERVER_URL}/llm/chat",
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": questions_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=60,
        )
        questions_resp.raise_for_status()
        questions_raw = clean_llm_response(questions_resp.json()["content"])

        synthetic_questions = [
            q.strip().lstrip("0123456789.-) ")
            for q in questions_raw.split("\n")
            if q.strip() and "?" in q
        ][:5]

        # 4. Monta enriched_text
        enriched_text = f"[CONTEXTO: {context_header}]\n\n{text}"
        if synthetic_questions:
            enriched_text += "\n\n[PERGUNTAS RELACIONADAS:\n"
            enriched_text += "\n".join(f"- {q}" for q in synthetic_questions)
            enriched_text += "]"

        # 5. Gera embeddings via GPU Server LOCAL (endpoint /embed/hybrid)
        embed_resp = requests.post(
            f"{GPU_SERVER_URL}/embed/hybrid",
            json={"texts": [enriched_text]},
            timeout=120,
        )
        embed_resp.raise_for_status()
        embed_data = embed_resp.json()
        dense_vector = embed_data["dense"][0]
        sparse_vector = embed_data["sparse"][0]

        # Thesis vector
        if thesis_text:
            thesis_embed_resp = requests.post(
                f"{GPU_SERVER_URL}/embed/hybrid",
                json={"texts": [thesis_text]},
                timeout=60,
            )
            thesis_embed_resp.raise_for_status()
            thesis_vector = thesis_embed_resp.json()["dense"][0]
        else:
            thesis_vector = dense_vector

        # 6. Atualiza no Milvus VPS
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
            logger.warning(f"[POD] Chunk nao encontrado: {chunk_id}")
            connections.disconnect("default")
            return {"success": False, "chunk_id": chunk_id, "error": "Chunk not found"}

        chunk = results[0]

        # Prepara dados para upsert
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
        logger.info(f"[POD] Chunk enriquecido em {elapsed:.1f}s: {chunk_id}")

        return {
            "success": True,
            "chunk_id": chunk_id,
            "context_header": context_header[:100],
            "thesis_type": thesis_type,
            "questions_count": len(synthetic_questions),
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.error(f"[POD] Erro em {chunk_id}: {e}")
        import traceback
        traceback.print_exc()
        raise self.retry(exc=e, countdown=30 * (self.request.retries + 1))


@app.task
def enrich_all_chunks_pod(document_id: str = None) -> dict:
    """
    Dispara enriquecimento para todos os chunks pendentes.

    Args:
        document_id: Se especificado, enriquece apenas chunks deste documento
    """
    from pymilvus import connections, Collection
    from celery import group

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection("leis_v3")
    collection.load()

    # Busca chunks sem context_header (nao enriquecidos)
    if document_id:
        expr = f'document_id == "{document_id}" and context_header == ""'
    else:
        expr = 'context_header == ""'

    results = collection.query(
        expr=expr,
        output_fields=["chunk_id", "text", "device_type", "article_number",
                      "document_id", "tipo_documento", "numero", "ano"],
        limit=10000,
    )

    logger.info(f"[POD] Encontrados {len(results)} chunks para enriquecer")

    tasks = []
    for chunk in results:
        task = enrich_chunk_pod.s(
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
        job = group(tasks)
        result = job.apply_async()

        return {
            "group_id": str(result.id),
            "task_count": len(tasks),
            "document_id": document_id,
        }

    return {"task_count": 0, "message": "Nenhum chunk para enriquecer"}


@app.task
def check_enrichment_progress() -> dict:
    """Verifica progresso do enriquecimento."""
    from pymilvus import connections, Collection

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection("leis_v3")
    collection.load()

    total = collection.num_entities

    enriched = collection.query(
        expr='context_header != ""',
        output_fields=["chunk_id"],
        limit=10000,
    )

    connections.disconnect("default")

    enriched_count = len(enriched)
    pending = total - enriched_count
    progress = (enriched_count / total * 100) if total > 0 else 0

    return {
        "total": total,
        "enriched": enriched_count,
        "pending": pending,
        "progress_pct": round(progress, 1),
    }
