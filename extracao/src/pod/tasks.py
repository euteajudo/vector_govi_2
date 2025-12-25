"""
Tasks Celery para enriquecimento de chunks no POD.

Usa vLLM local (POD) para enriquecimento e
RemoteEmbedder para gerar novos embeddings.
Milvus é acessado via túnel reverso SSH.
"""

import logging
import time
from typing import Optional

from .celery_app import app
from .config import get_pod_config
from .remote_embedder import RemoteEmbedder

logger = logging.getLogger(__name__)


def _get_vllm_client():
    """Inicializa cliente vLLM (lazy loading)."""
    from src.llm.vllm_client import VLLMClient, LLMConfig

    config = get_pod_config()
    llm_config = LLMConfig(
        model=config.vllm_model,
        base_url=config.vllm_base_url,
        temperature=0.3,
        max_tokens=1024,
    )
    return VLLMClient(config=llm_config)


def _get_enricher():
    """Inicializa enricher (lazy loading)."""
    from src.enrichment.chunk_enricher import ChunkEnricher
    return ChunkEnricher(llm_client=_get_vllm_client())


def _get_milvus_client():
    """Inicializa cliente Milvus (via túnel reverso)."""
    from pymilvus import MilvusClient

    config = get_pod_config()
    return MilvusClient(
        uri=f"http://{config.milvus_host}:{config.milvus_port}"
    )


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def enrich_chunk_task(
    self,
    chunk_id: str,
    text: str,
    device_type: str,
    document_id: str,
    document_context: str,
    collection_name: str = "leis_v3",
):
    """
    Enriquece um chunk e atualiza no Milvus.

    Args:
        chunk_id: ID único do chunk
        text: Texto do chunk
        device_type: Tipo (article, paragraph, inciso)
        document_id: ID do documento
        document_context: Contexto do documento (titulo, ementa)
        collection_name: Nome da collection no Milvus
    """
    try:
        start = time.time()
        logger.info(f"[POD] Enriquecendo chunk {chunk_id}...")

        # 1. Enriquece com LLM (vLLM local no POD)
        enricher = _get_enricher()
        result = enricher.enrich(
            text=text,
            device_type=device_type,
            document_context=document_context,
        )

        if not result.success:
            logger.warning(f"[POD] Falha no enriquecimento de {chunk_id}: {result.error}")
            raise Exception(result.error)

        # 2. Monta enriched_text
        enriched_text = _build_enriched_text(
            text=text,
            context_header=result.context_header,
            synthetic_questions=result.synthetic_questions,
        )

        # 3. Gera novos embeddings (BGE-M3 local no POD)
        embedder = RemoteEmbedder()
        emb_result = embedder.encode_hybrid([enriched_text, result.thesis_text])
        embedder.close()

        dense_vector = emb_result["dense"][0]
        thesis_vector = emb_result["dense"][1]
        sparse_vector = emb_result["sparse"][0]

        # 4. Atualiza no Milvus (via túnel reverso)
        client = _get_milvus_client()

        # Delete chunk antigo
        client.delete(
            collection_name=collection_name,
            filter=f'chunk_id == "{chunk_id}"',
        )

        # Busca dados originais para manter campos
        # (simplificado: assume que chunk já existe)

        # Insert atualizado
        client.insert(
            collection_name=collection_name,
            data=[{
                "chunk_id": chunk_id,
                "text": text,
                "enriched_text": enriched_text,
                "context_header": result.context_header or "",
                "thesis_text": result.thesis_text or "",
                "thesis_type": result.thesis_type or "",
                "synthetic_questions": result.synthetic_questions or "",
                "dense_vector": dense_vector,
                "thesis_vector": thesis_vector,
                "sparse_vector": _sparse_to_milvus(sparse_vector),
                # Outros campos serão preenchidos pelo caller
            }],
        )

        elapsed = time.time() - start
        logger.info(f"[POD] Chunk {chunk_id} enriquecido em {elapsed:.2f}s")

        return {
            "chunk_id": chunk_id,
            "success": True,
            "time_seconds": elapsed,
        }

    except Exception as e:
        logger.exception(f"[POD] Erro ao enriquecer {chunk_id}: {e}")
        raise self.retry(exc=e)


def _build_enriched_text(
    text: str,
    context_header: Optional[str],
    synthetic_questions: Optional[str],
) -> str:
    """Monta texto enriquecido para embedding."""
    parts = []

    if context_header:
        parts.append(f"[CONTEXTO: {context_header}]")

    parts.append(text)

    if synthetic_questions:
        parts.append(f"[PERGUNTAS RELACIONADAS:\n{synthetic_questions}]")

    return "\n\n".join(parts)


def _sparse_to_milvus(sparse_dict: dict) -> dict:
    """Converte sparse dict para formato Milvus."""
    # Milvus espera {indices: [...], values: [...]}
    if not sparse_dict:
        return {"indices": [], "values": []}

    indices = [int(k) for k in sparse_dict.keys()]
    values = [float(v) for v in sparse_dict.values()]

    return {"indices": indices, "values": values}


@app.task
def health_check():
    """Verifica saúde dos serviços."""
    config = get_pod_config()

    # Verifica vLLM
    import requests
    try:
        resp = requests.get(f"{config.vllm_base_url}/models", timeout=5)
        vllm_ok = resp.status_code == 200
    except Exception:
        vllm_ok = False

    # Verifica Embedding Server
    embedder = RemoteEmbedder()
    embed_ok = embedder.health_check()
    embedder.close()

    # Verifica Milvus
    try:
        client = _get_milvus_client()
        collections = client.list_collections()
        milvus_ok = True
    except Exception:
        milvus_ok = False

    return {
        "vllm": vllm_ok,
        "embedding_server": embed_ok,
        "milvus": milvus_ok,
    }
