"""
Script para enriquecer chunks já indexados no Milvus.

Processa chunk a chunk para evitar timeout do LLM.
Atualiza campos de enriquecimento e regenera embeddings.

Uso:
    python scripts/enrich_existing_chunks.py --document-id LEI-14133-2021
    python scripts/enrich_existing_chunks.py --document-id LEI-14133-2021 --batch-size 1 --start-from 100
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymilvus import connections, Collection
from llm.vllm_client import VLLMClient, LLMConfig
from enrichment import ChunkEnricher
from enrichment.chunk_enricher import DocumentMetadata
from embeddings.bge_m3 import BGEM3Embedder, EmbeddingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_enrichment_client() -> VLLMClient:
    """Cria cliente vLLM otimizado para enriquecimento."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-8B-AWQ",
        temperature=0.0,
        max_tokens=2048,
        timeout=600.0,  # 10 minutos por chunk
        max_retries=5,
        retry_delay=2.0,
    )
    return VLLMClient(config=config)


def get_chunks_to_enrich(
    collection: Collection,
    document_id: str,
    start_from: int = 0,
    limit: int = 2000,
) -> list[dict]:
    """Busca chunks sem enriquecimento."""

    # Busca chunks do documento que não têm context_header
    expr = f'document_id == "{document_id}" and context_header == ""'

    results = collection.query(
        expr=expr,
        output_fields=[
            "id", "chunk_id", "span_id", "device_type", "text",
            "article_number", "tipo_documento", "numero", "ano",
            "parent_chunk_id", "citations",
        ],
        limit=limit,
        offset=start_from,
    )

    return results


def update_chunk_in_milvus(
    collection: Collection,
    chunk_id: str,
    context_header: str,
    thesis_text: str,
    thesis_type: str,
    synthetic_questions: str,
    enriched_text: str,
    dense_vector: list,
    thesis_vector: list,
    sparse_vector: dict,
):
    """Atualiza chunk enriquecido no Milvus via upsert."""

    # Busca o chunk existente para pegar todos os campos
    results = collection.query(
        expr=f'chunk_id == "{chunk_id}"',
        output_fields=["*"],
        limit=1,
    )

    if not results:
        logger.warning(f"Chunk não encontrado: {chunk_id}")
        return False

    chunk = results[0]

    # Prepara dados para upsert
    data = {
        "chunk_id": [chunk["chunk_id"]],
        "parent_chunk_id": [chunk["parent_chunk_id"]],
        "span_id": [chunk["span_id"]],
        "device_type": [chunk["device_type"]],
        "chunk_level": [chunk["chunk_level"]],
        "text": [chunk["text"]],
        "enriched_text": [enriched_text],
        "dense_vector": [dense_vector],
        "thesis_vector": [thesis_vector],
        "sparse_vector": [sparse_vector],
        "context_header": [context_header],
        "thesis_text": [thesis_text],
        "thesis_type": [thesis_type],
        "synthetic_questions": [synthetic_questions],
        "citations": [chunk["citations"]],
        "document_id": [chunk["document_id"]],
        "tipo_documento": [chunk["tipo_documento"]],
        "numero": [chunk["numero"]],
        "ano": [chunk["ano"]],
        "article_number": [chunk["article_number"]],
        "schema_version": [chunk["schema_version"]],
        "extractor_version": [chunk["extractor_version"]],
        "ingestion_timestamp": [chunk["ingestion_timestamp"]],
        "document_hash": [chunk["document_hash"]],
        "page": [chunk["page"]],
        "bbox_left": [chunk["bbox_left"]],
        "bbox_top": [chunk["bbox_top"]],
        "bbox_right": [chunk["bbox_right"]],
        "bbox_bottom": [chunk["bbox_bottom"]],
    }

    # Deleta o chunk antigo
    collection.delete(expr=f'chunk_id == "{chunk_id}"')

    # Insere o chunk atualizado
    collection.insert(data)

    return True


def enrich_single_chunk(
    enricher: ChunkEnricher,
    embedder: BGEM3Embedder,
    chunk: dict,
    doc_meta: DocumentMetadata,
) -> Optional[dict]:
    """Enriquece um único chunk."""

    try:
        # Gera enriquecimento
        enrichment = enricher.enrich_single(
            text=chunk["text"],
            device_type=chunk["device_type"],
            article_number=chunk["article_number"],
            doc_meta=doc_meta,
        )

        if not enrichment:
            return None

        # Monta enriched_text
        enriched_text = enricher._build_enriched_text(
            original_text=chunk["text"],
            context_header=enrichment.get("context_header", ""),
            synthetic_questions=enrichment.get("synthetic_questions", []),
        )

        # Gera novos embeddings
        embed_result = embedder.encode_hybrid([enriched_text])
        dense_vector = embed_result["dense"][0]
        sparse_vector = embed_result["sparse"][0]

        # Gera thesis_vector se tiver thesis_text
        thesis_text = enrichment.get("thesis_text", "")
        if thesis_text:
            thesis_result = embedder.encode_hybrid([thesis_text])
            thesis_vector = thesis_result["dense"][0]
        else:
            thesis_vector = dense_vector

        return {
            "context_header": enrichment.get("context_header", ""),
            "thesis_text": thesis_text,
            "thesis_type": enrichment.get("thesis_type", "disposicao"),
            "synthetic_questions": "\n".join(enrichment.get("synthetic_questions", [])),
            "enriched_text": enriched_text,
            "dense_vector": dense_vector,
            "thesis_vector": thesis_vector,
            "sparse_vector": sparse_vector,
        }

    except Exception as e:
        logger.error(f"Erro ao enriquecer chunk: {e}")
        return None


def run_enrichment(
    document_id: str,
    batch_size: int = 1,
    start_from: int = 0,
    max_chunks: int = 0,
    dry_run: bool = False,
):
    """Executa enriquecimento em segunda etapa."""

    print("=" * 70)
    print("ENRIQUECIMENTO EM SEGUNDA ETAPA")
    print("=" * 70)
    print(f"Documento: {document_id}")
    print(f"Batch size: {batch_size}")
    print(f"Start from: {start_from}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)

    # Conecta ao Milvus
    print("\n[1/5] Conectando ao Milvus...")
    connections.connect(host="localhost", port="19530")
    collection = Collection("leis_v3")
    collection.load()

    # Busca chunks sem enriquecimento
    print("\n[2/5] Buscando chunks sem enriquecimento...")
    chunks = get_chunks_to_enrich(collection, document_id, start_from)

    if max_chunks > 0:
        chunks = chunks[:max_chunks]

    print(f"      Chunks a enriquecer: {len(chunks)}")

    if not chunks:
        print("      Nenhum chunk para enriquecer!")
        return

    if dry_run:
        print("\n[DRY RUN] Não fará alterações.")
        for chunk in chunks[:5]:
            print(f"  - {chunk['chunk_id']}: {chunk['text'][:50]}...")
        return

    # Inicializa componentes
    print("\n[3/5] Inicializando LLM e embedder...")
    llm_client = create_enrichment_client()
    enricher = ChunkEnricher(llm_client=llm_client)
    embedder = BGEM3Embedder(EmbeddingConfig(use_fp16=True))

    # Detecta tipo de documento
    first_chunk = chunks[0]
    doc_meta = DocumentMetadata(
        document_id=document_id,
        document_type=first_chunk["tipo_documento"],
        number=first_chunk["numero"],
        year=first_chunk["ano"],
        issuing_body="PRESIDENCIA DA REPUBLICA" if first_chunk["tipo_documento"] == "LEI" else "SEGES/ME",
    )

    print(f"      Documento: {doc_meta.document_type} {doc_meta.number}/{doc_meta.year}")

    # Processa chunks
    print("\n[4/5] Enriquecendo chunks...")

    success_count = 0
    error_count = 0
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]

        try:
            # Enriquece
            result = enrich_single_chunk(enricher, embedder, chunk, doc_meta)

            if result:
                # Atualiza no Milvus
                update_chunk_in_milvus(
                    collection=collection,
                    chunk_id=chunk_id,
                    **result,
                )
                success_count += 1

                # Log a cada 10 chunks
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(chunks) - i - 1) / rate if rate > 0 else 0
                    print(f"      [{i+1}/{len(chunks)}] {success_count} OK, {error_count} erros | "
                          f"{elapsed:.0f}s decorridos, ~{remaining:.0f}s restantes")
            else:
                error_count += 1
                logger.warning(f"Falha ao enriquecer: {chunk_id}")

        except Exception as e:
            error_count += 1
            logger.error(f"Erro em {chunk_id}: {e}")

            # Pausa em caso de erro para não sobrecarregar
            time.sleep(2)

    # Flush
    collection.flush()

    # Relatório
    print("\n[5/5] Relatório final")
    print("=" * 70)
    total_time = time.time() - start_time
    print(f"Tempo total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Chunks processados: {len(chunks)}")
    print(f"Sucesso: {success_count}")
    print(f"Erros: {error_count}")
    print(f"Taxa: {success_count/total_time*60:.1f} chunks/min")
    print("=" * 70)

    # Cleanup
    enricher.close()
    connections.disconnect("default")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enriquece chunks existentes no Milvus")
    parser.add_argument("--document-id", required=True, help="ID do documento (ex: LEI-14133-2021)")
    parser.add_argument("--batch-size", type=int, default=1, help="Chunks por batch (default: 1)")
    parser.add_argument("--start-from", type=int, default=0, help="Offset inicial")
    parser.add_argument("--max-chunks", type=int, default=0, help="Máximo de chunks (0 = todos)")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra o que seria feito")

    args = parser.parse_args()

    run_enrichment(
        document_id=args.document_id,
        batch_size=args.batch_size,
        start_from=args.start_from,
        max_chunks=args.max_chunks,
        dry_run=args.dry_run,
    )
