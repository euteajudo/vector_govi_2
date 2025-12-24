"""
Script para monitorar progresso do enriquecimento.

Uso:
    python scripts/check_progress.py
    python scripts/check_progress.py --watch  # Atualiza a cada 30s
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymilvus import connections, Collection


def check_progress():
    """Verifica progresso do enriquecimento."""
    try:
        connections.connect(host="localhost", port="19530")
        collection = Collection("leis_v3")
        collection.load()

        # Total por documento
        docs = {}

        # Lei 14133
        lei_enriched = collection.query(
            expr='context_header != "" and document_id == "LEI-14133-2021"',
            output_fields=["chunk_id"],
            limit=10000,
        )
        lei_pending = collection.query(
            expr='context_header == "" and document_id == "LEI-14133-2021"',
            output_fields=["chunk_id"],
            limit=10000,
        )
        docs["LEI-14133-2021"] = {
            "enriched": len(lei_enriched),
            "pending": len(lei_pending),
            "total": len(lei_enriched) + len(lei_pending),
        }

        # IN 65
        in_enriched = collection.query(
            expr='context_header != "" and document_id == "IN-65-2021"',
            output_fields=["chunk_id"],
            limit=10000,
        )
        in_pending = collection.query(
            expr='context_header == "" and document_id == "IN-65-2021"',
            output_fields=["chunk_id"],
            limit=10000,
        )
        docs["IN-65-2021"] = {
            "enriched": len(in_enriched),
            "pending": len(in_pending),
            "total": len(in_enriched) + len(in_pending),
        }

        connections.disconnect("default")
        return docs

    except Exception as e:
        print(f"Erro: {e}")
        return None


def print_progress(docs):
    """Imprime progresso formatado."""
    print("\n" + "=" * 50)
    print(f"PROGRESSO DO ENRIQUECIMENTO - {time.strftime('%H:%M:%S')}")
    print("=" * 50)

    total_enriched = 0
    total_pending = 0

    for doc_id, stats in docs.items():
        pct = 100 * stats["enriched"] / stats["total"] if stats["total"] > 0 else 0
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)

        print(f"\n{doc_id}:")
        print(f"  [{bar}] {pct:.1f}%")
        print(f"  Enriquecidos: {stats['enriched']}/{stats['total']}")
        print(f"  Pendentes: {stats['pending']}")

        total_enriched += stats["enriched"]
        total_pending += stats["pending"]

    total = total_enriched + total_pending
    total_pct = 100 * total_enriched / total if total > 0 else 0

    print("\n" + "-" * 50)
    print(f"TOTAL: {total_enriched}/{total} ({total_pct:.1f}%)")
    print(f"Pendentes: {total_pending}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Monitora progresso do enriquecimento")
    parser.add_argument("--watch", "-w", action="store_true", help="Atualiza a cada 30s")
    args = parser.parse_args()

    if args.watch:
        print("Monitorando progresso (Ctrl+C para sair)...")
        try:
            while True:
                docs = check_progress()
                if docs:
                    # Limpa tela (Windows)
                    print("\033[H\033[J", end="")
                    print_progress(docs)
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitoramento encerrado.")
    else:
        docs = check_progress()
        if docs:
            print_progress(docs)


if __name__ == "__main__":
    main()
