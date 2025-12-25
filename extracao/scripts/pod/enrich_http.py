"""
Enriquecimento de chunks via GPU Server HTTP (sem Celery).

Executa na VPS, usa GPU Server no POD para LLM e embeddings.

Uso:
    python scripts/pod/enrich_http.py --document-id IN-65-2021
    python scripts/pod/enrich_http.py --document-id IN-65-2021 --batch-size 5
"""

import sys
import time
import json
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pymilvus import connections, Collection
from pod.gpu_client import GPUClient
from chunking.enrichment_prompts import (
    build_enrichment_prompt,
    build_enriched_text,
    parse_enrichment_response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuracao
GPU_SERVER_URL = "http://195.26.233.70:55278"
LLM_MODEL = "qwen3-8b"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "leis_v3"


def enrich_chunk(
    gpu_client: GPUClient,
    chunk: dict,
) -> dict:
    """Enriquece um chunk usando GPU Server."""

    chunk_id = chunk["chunk_id"]
    text = chunk["text"]
    document_type = chunk.get("tipo_documento", "LEI")
    number = chunk.get("numero", "")
    year = chunk.get("ano", 2021)
    article_number = chunk.get("article_number", "")

    logger.info(f"Enriquecendo: {chunk_id}")
    start_time = time.time()

    try:
        # Monta prompt de enriquecimento
        system_prompt, user_prompt = build_enrichment_prompt(
            text=text[:3000],  # Limita texto
            document_type=document_type,
            number=number,
            year=year,
            issuing_body="Governo Federal",
            chapter_number="",
            chapter_title="",
            article_number=article_number,
            article_title=None,
        )

        # Chama LLM
        result = gpu_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=800,
        )

        response_text = result.get("content", "")

        # Parse resposta
        try:
            enrichment_data = parse_enrichment_response(response_text)
        except Exception as e:
            logger.warning(f"  Erro no parse: {e}")
            # Fallback simples
            enrichment_data = {
                "context_header": f"Artigo {article_number} da {document_type} {number}/{year}",
                "thesis_text": text[:200],
                "thesis_type": "disposicao",
                "synthetic_questions": "",
            }

        context_header = enrichment_data["context_header"]
        thesis_text = enrichment_data["thesis_text"]
        thesis_type = enrichment_data["thesis_type"]
        synthetic_questions = enrichment_data.get("synthetic_questions", "")

        # Converte questions para lista se for string
        if isinstance(synthetic_questions, str):
            questions_list = [q.strip() for q in synthetic_questions.split("\n") if q.strip()]
        else:
            questions_list = synthetic_questions

        # Monta enriched_text
        enriched_text = build_enriched_text(
            text=text,
            context_header=context_header,
            synthetic_questions=questions_list,
        )

        # Gera embeddings
        embed_result = gpu_client.embed_hybrid([enriched_text])
        dense_vector = embed_result["dense"][0]
        sparse_vector = embed_result["sparse"][0]

        # Thesis vector
        if thesis_text:
            thesis_embed = gpu_client.embed_hybrid([thesis_text])
            thesis_vector = thesis_embed["dense"][0]
        else:
            thesis_vector = dense_vector

        elapsed = time.time() - start_time
        logger.info(f"  OK em {elapsed:.1f}s - context: {len(context_header)} chars, type: {thesis_type}")

        return {
            "success": True,
            "chunk_id": chunk_id,
            "context_header": context_header[:2000],
            "thesis_text": thesis_text[:5000],
            "thesis_type": thesis_type[:100],
            "synthetic_questions": "\n".join(questions_list) if isinstance(questions_list, list) else synthetic_questions,
            "enriched_text": enriched_text[:65535],
            "dense_vector": dense_vector,
            "thesis_vector": thesis_vector,
            "sparse_vector": sparse_vector,
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.error(f"  ERRO: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "chunk_id": chunk_id, "error": str(e)}


def update_chunk_in_milvus(collection: Collection, chunk: dict, enrichment: dict):
    """Atualiza chunk no Milvus com dados de enriquecimento."""

    # Prepara dados para upsert
    row = {
        "chunk_id": chunk["chunk_id"],
        "parent_chunk_id": chunk.get("parent_chunk_id", ""),
        "span_id": chunk.get("span_id", ""),
        "device_type": chunk.get("device_type", "article"),
        "chunk_level": chunk.get("chunk_level", "article"),
        "text": chunk["text"],
        "enriched_text": enrichment["enriched_text"],
        "dense_vector": enrichment["dense_vector"],
        "thesis_vector": enrichment["thesis_vector"],
        "sparse_vector": enrichment["sparse_vector"],
        "context_header": enrichment["context_header"],
        "thesis_text": enrichment["thesis_text"],
        "thesis_type": enrichment["thesis_type"],
        "synthetic_questions": enrichment["synthetic_questions"][:10000],
        "citations": chunk.get("citations", ""),
        "document_id": chunk.get("document_id", ""),
        "tipo_documento": chunk.get("tipo_documento", ""),
        "numero": chunk.get("numero", ""),
        "ano": chunk.get("ano", 2021),
        "article_number": chunk.get("article_number", ""),
        "schema_version": chunk.get("schema_version", "1.0.0"),
        "extractor_version": chunk.get("extractor_version", "1.0.0"),
        "ingestion_timestamp": chunk.get("ingestion_timestamp", ""),
        "document_hash": chunk.get("document_hash", ""),
        "page": chunk.get("page", 0),
        "bbox_left": chunk.get("bbox_left", 0.0),
        "bbox_top": chunk.get("bbox_top", 0.0),
        "bbox_right": chunk.get("bbox_right", 0.0),
        "bbox_bottom": chunk.get("bbox_bottom", 0.0),
    }

    # Delete e insert
    collection.delete(expr=f'chunk_id == "{chunk["chunk_id"]}"')
    collection.insert([row])


def main(
    document_id: str,
    batch_size: int = 5,
    start_from: int = 0,
):
    """Enriquece todos os chunks de um documento."""

    print("=" * 70)
    print("ENRIQUECIMENTO VIA GPU SERVER HTTP")
    print("=" * 70)
    print(f"GPU Server: {GPU_SERVER_URL}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"Document: {document_id}")
    print("=" * 70)

    # Conecta GPU Server
    gpu_client = GPUClient(GPU_SERVER_URL, timeout=300)
    if not gpu_client.is_healthy():
        print("ERRO: GPU Server nao disponivel!")
        return
    print("GPU Server: OK")

    # Conecta Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()
    print(f"Milvus: OK ({collection.num_entities} chunks)")

    # Busca chunks do documento
    chunks = collection.query(
        expr=f'document_id == "{document_id}"',
        output_fields=["*"],
        limit=10000,
    )

    print(f"Chunks encontrados: {len(chunks)}")

    if start_from > 0:
        chunks = chunks[start_from:]
        print(f"Iniciando do chunk {start_from}")

    # Filtra chunks nao enriquecidos
    chunks_to_enrich = [
        c for c in chunks
        if not c.get("context_header") or c.get("context_header") == ""
    ]

    print(f"Chunks para enriquecer: {len(chunks_to_enrich)}")

    if not chunks_to_enrich:
        print("Todos os chunks ja estao enriquecidos!")
        return

    # Processa
    total = len(chunks_to_enrich)
    success = 0
    errors = 0
    total_time = 0

    for i, chunk in enumerate(chunks_to_enrich):
        print(f"\n[{i+1}/{total}] {chunk['chunk_id']}")

        enrichment = enrich_chunk(gpu_client, chunk)

        if enrichment["success"]:
            update_chunk_in_milvus(collection, chunk, enrichment)
            success += 1
            total_time += enrichment["elapsed"]
        else:
            errors += 1

        # Flush a cada batch
        if (i + 1) % batch_size == 0:
            collection.flush()
            avg_time = total_time / success if success > 0 else 0
            print(f"\n--- Batch {(i+1)//batch_size}: {success} OK, {errors} erros, {avg_time:.1f}s/chunk ---")

    # Flush final
    collection.flush()

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Total processados: {total}")
    print(f"Sucesso: {success}")
    print(f"Erros: {errors}")
    print(f"Tempo total: {total_time:.1f}s")
    if success > 0:
        print(f"Tempo medio: {total_time/success:.1f}s/chunk")
    print("=" * 70)

    gpu_client.close()
    connections.disconnect("default")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enriquecimento HTTP")
    parser.add_argument("--document-id", "-d", required=True, help="ID do documento")
    parser.add_argument("--batch-size", "-b", type=int, default=5, help="Tamanho do batch")
    parser.add_argument("--start-from", "-s", type=int, default=0, help="Iniciar do chunk N")

    args = parser.parse_args()

    main(
        document_id=args.document_id,
        batch_size=args.batch_size,
        start_from=args.start_from,
    )
