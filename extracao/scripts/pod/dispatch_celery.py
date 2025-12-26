"""
Dispara tasks de enriquecimento para Celery no POD.

Este script roda LOCAL e envia tasks para o Redis do POD.
Requer que o Redis do POD esteja acessível.

Uso:
    # Com Redis local (Windows)
    python scripts/pod/dispatch_celery.py --redis-host localhost

    # Com Redis no POD (via tunnel)
    ssh -L 6379:localhost:6379 user@pod
    python scripts/pod/dispatch_celery.py --redis-host localhost
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pymilvus import connections, Collection


def get_pending_chunks(
    document_id: str = "LEI-14133-2021",
    milvus_host: str = "77.37.43.160",
    limit: int = 5000,
) -> list[dict]:
    """Busca chunks não enriquecidos."""
    connections.connect(host=milvus_host, port=19530)
    collection = Collection("leis_v3")
    collection.load()

    chunks = collection.query(
        expr=f'document_id == "{document_id}" and context_header == ""',
        output_fields=[
            "chunk_id", "text", "device_type", "article_number",
            "document_id", "tipo_documento", "numero", "ano",
        ],
        limit=limit,
    )

    connections.disconnect("default")
    return chunks


def dispatch_tasks(
    chunks: list[dict],
    redis_host: str = "localhost",
    redis_port: int = 6379,
    dry_run: bool = False,
) -> dict:
    """Envia tasks para Celery."""
    import os
    os.environ["REDIS_HOST"] = redis_host
    os.environ["REDIS_PORT"] = str(redis_port)

    # Import após setar env
    from enrichment.tasks_http import enrich_chunk_http

    stats = {
        "total": len(chunks),
        "dispatched": 0,
        "task_ids": [],
        "start_time": datetime.now().isoformat(),
    }

    if dry_run:
        print(f"\n[DRY RUN] Seriam enviadas {len(chunks)} tasks")
        for chunk in chunks[:5]:
            print(f"  - {chunk['chunk_id']}")
        if len(chunks) > 5:
            print(f"  ... e mais {len(chunks) - 5} chunks")
        return stats

    print(f"\n[DISPATCH] Enviando {len(chunks)} tasks para Celery...")

    for i, chunk in enumerate(chunks):
        result = enrich_chunk_http.delay(
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
        stats["dispatched"] += 1

        if (i + 1) % 100 == 0:
            print(f"  Enviadas: {i + 1}/{len(chunks)}")

    stats["end_time"] = datetime.now().isoformat()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Dispara tasks Celery para POD")
    parser.add_argument(
        "--document-id", "-d", default="LEI-14133-2021",
        help="Document ID"
    )
    parser.add_argument(
        "--redis-host", "-r", default="localhost",
        help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--redis-port", "-p", type=int, default=6379,
        help="Redis port (default: 6379)"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=5000,
        help="Max chunks"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Apenas mostra o que seria feito"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DISPATCHER CELERY PARA POD")
    print("=" * 60)
    print(f"Document: {args.document_id}")
    print(f"Redis: {args.redis_host}:{args.redis_port}")
    print("=" * 60)

    # Busca chunks
    print("\n[1] Buscando chunks pendentes...")
    chunks = get_pending_chunks(
        document_id=args.document_id,
        limit=args.limit,
    )
    print(f"    Encontrados: {len(chunks)}")

    if not chunks:
        print("\n[INFO] Nenhum chunk pendente!")
        return

    # Dispara tasks
    print("\n[2] Enviando tasks...")
    stats = dispatch_tasks(
        chunks=chunks,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Total: {stats['total']}")
    print(f"Enviadas: {stats['dispatched']}")

    if stats['task_ids']:
        print(f"\nPrimeiras 5 task IDs:")
        for tid in stats['task_ids'][:5]:
            print(f"  {tid}")


if __name__ == "__main__":
    main()
