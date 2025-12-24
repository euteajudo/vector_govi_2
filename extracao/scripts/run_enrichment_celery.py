"""
Script para disparar enriquecimento de chunks via Celery.

Busca chunks no Milvus que ainda nao foram enriquecidos (context_header vazio)
e envia tasks para o Celery processar em paralelo.

Pre-requisitos:
    1. Redis rodando:
       docker run -d --name redis -p 6379:6379 redis:alpine

    2. Worker Celery rodando:
       cd extracao
       celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=2

    3. (Opcional) Flower para monitoramento:
       celery -A src.enrichment.celery_app flower

Uso:
    # Enriquecer todos os chunks pendentes
    python scripts/run_enrichment_celery.py

    # Enriquecer apenas chunks de um documento
    python scripts/run_enrichment_celery.py --document-id LEI-14133-2021

    # Limitar numero de chunks
    python scripts/run_enrichment_celery.py --limit 100

    # Dry run (mostra o que seria feito)
    python scripts/run_enrichment_celery.py --dry-run
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add extracao to path (for src.* imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymilvus import connections, Collection


def get_pending_chunks(
    collection_name: str = "leis_v3",
    document_id: str = None,
    limit: int = None,
) -> list[dict]:
    """
    Busca chunks que ainda nao foram enriquecidos.

    Args:
        collection_name: Nome da collection no Milvus
        document_id: Filtrar por document_id (opcional)
        limit: Limitar numero de chunks (opcional)

    Returns:
        Lista de chunks pendentes
    """
    connections.connect(host="localhost", port="19530")
    collection = Collection(collection_name)
    collection.load()

    # Filtro: context_header vazio (nao enriquecido)
    expr = 'context_header == ""'
    if document_id:
        expr = f'{expr} and document_id == "{document_id}"'

    # Busca chunks pendentes
    results = collection.query(
        expr=expr,
        output_fields=[
            "chunk_id",
            "text",
            "device_type",
            "article_number",
            "document_id",
            "tipo_documento",
            "numero",
            "ano",
        ],
        limit=limit or 10000,
    )

    connections.disconnect("default")
    return results


def dispatch_enrichment_tasks(
    chunks: list[dict],
    batch_size: int = 10,
    dry_run: bool = False,
) -> dict:
    """
    Dispara tasks de enriquecimento para o Celery.

    Args:
        chunks: Lista de chunks para enriquecer
        batch_size: Tamanho do batch (chunks por grupo)
        dry_run: Se True, apenas mostra o que seria feito

    Returns:
        Dict com estatisticas
    """
    from src.enrichment.tasks import enrich_chunk_task

    stats = {
        "total_chunks": len(chunks),
        "tasks_dispatched": 0,
        "task_ids": [],
        "start_time": datetime.now().isoformat(),
    }

    if dry_run:
        print(f"\n[DRY RUN] Seriam disparadas {len(chunks)} tasks")
        for i, chunk in enumerate(chunks[:5]):
            print(f"  - {chunk['chunk_id']}")
        if len(chunks) > 5:
            print(f"  ... e mais {len(chunks) - 5} chunks")
        return stats

    print(f"\n[DISPATCH] Enviando {len(chunks)} tasks para o Celery...")

    for i, chunk in enumerate(chunks):
        # Dispara task assincrona
        result = enrich_chunk_task.delay(
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            device_type=chunk["device_type"],
            article_number=chunk["article_number"],
            document_id=chunk["document_id"],
            document_type=chunk["tipo_documento"],
            number=chunk["numero"],
            year=chunk["ano"],
        )

        stats["task_ids"].append(result.id)
        stats["tasks_dispatched"] += 1

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Dispatched: {i + 1}/{len(chunks)}")

    stats["end_time"] = datetime.now().isoformat()
    return stats


def check_celery_status() -> bool:
    """Verifica se o Celery esta disponivel."""
    try:
        from src.enrichment.celery_app import app

        # Tenta pingar o broker
        inspect = app.control.inspect()
        active = inspect.active()

        if active:
            workers = list(active.keys())
            print(f"[OK] Celery conectado. Workers ativos: {workers}")
            return True
        else:
            print("[WARN] Celery conectado mas nenhum worker ativo")
            print("       Inicie um worker com:")
            print("       celery -A src.enrichment.celery_app worker --loglevel=info")
            return False

    except Exception as e:
        print(f"[ERROR] Celery nao disponivel: {e}")
        print("        Verifique se o Redis esta rodando:")
        print("        docker run -d --name redis -p 6379:6379 redis:alpine")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Dispara enriquecimento de chunks via Celery"
    )
    parser.add_argument(
        "--document-id",
        help="Filtrar por document_id (ex: LEI-14133-2021)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limitar numero de chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Tamanho do batch (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas mostra o que seria feito",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Pular verificacao do Celery",
    )
    parser.add_argument(
        "--collection",
        default="leis_v3",
        help="Nome da collection (default: leis_v3)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ENRIQUECIMENTO VIA CELERY")
    print("=" * 70)

    # Verifica Celery (a menos que seja dry-run ou skip-check)
    if not args.dry_run and not args.skip_check:
        if not check_celery_status():
            print("\n[ABORTED] Celery nao esta pronto. Use --skip-check para forcar.")
            return

    # Busca chunks pendentes
    print(f"\n[SEARCH] Buscando chunks pendentes...")
    if args.document_id:
        print(f"         Filtro: document_id = {args.document_id}")
    if args.limit:
        print(f"         Limite: {args.limit}")

    chunks = get_pending_chunks(
        collection_name=args.collection,
        document_id=args.document_id,
        limit=args.limit,
    )

    if not chunks:
        print("\n[INFO] Nenhum chunk pendente de enriquecimento!")
        return

    print(f"\n[FOUND] {len(chunks)} chunks pendentes")

    # Agrupa por documento
    by_doc = {}
    for chunk in chunks:
        doc_id = chunk["document_id"]
        by_doc[doc_id] = by_doc.get(doc_id, 0) + 1

    print("\nChunks por documento:")
    for doc_id, count in sorted(by_doc.items()):
        print(f"  {doc_id}: {count}")

    # Dispara tasks
    stats = dispatch_enrichment_tasks(
        chunks=chunks,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"Total de chunks: {stats['total_chunks']}")
    print(f"Tasks disparadas: {stats['tasks_dispatched']}")

    if not args.dry_run and stats["task_ids"]:
        print(f"\nPrimeiras 5 task IDs:")
        for tid in stats["task_ids"][:5]:
            print(f"  {tid}")

        print("\n[INFO] Acompanhe o progresso com:")
        print("       celery -A src.enrichment.celery_app flower")
        print("       Acesse: http://localhost:5555")


if __name__ == "__main__":
    main()
