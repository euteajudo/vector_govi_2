"""
Monitor de progresso do enriquecimento Lei 14.133.

Uso:
    python scripts/pod/monitor_enrichment.py
    python scripts/pod/monitor_enrichment.py --watch  # Atualiza a cada 30s
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pymilvus import connections, Collection


def check_progress(document_id: str = "LEI-14133-2021") -> dict:
    """Verifica progresso do enriquecimento."""
    connections.connect(host="77.37.43.160", port=19530)
    collection = Collection("leis_v3")
    collection.load()

    # Total
    total = collection.query(
        expr=f'document_id == "{document_id}"',
        output_fields=["chunk_id"],
        limit=10000,
    )

    # Sem enriquecimento
    not_enriched = collection.query(
        expr=f'document_id == "{document_id}" and context_header == ""',
        output_fields=["chunk_id"],
        limit=10000,
    )

    connections.disconnect("default")

    enriched = len(total) - len(not_enriched)
    pct = enriched / len(total) * 100 if total else 0

    return {
        "total": len(total),
        "enriched": enriched,
        "pending": len(not_enriched),
        "percentage": pct,
    }


def format_time(seconds: float) -> str:
    """Formata segundos em hh:mm:ss."""
    return str(timedelta(seconds=int(seconds)))


def display_progress(progress: dict, rate: float = 4.5):
    """Exibe progresso formatado."""
    remaining_time = progress["pending"] * rate
    eta = datetime.now() + timedelta(seconds=remaining_time)

    print("=" * 60)
    print(f"ENRIQUECIMENTO Lei 14.133/2021 - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    print(f"Total chunks:    {progress['total']}")
    print(f"Enriquecidos:    {progress['enriched']} ({progress['percentage']:.1f}%)")
    print(f"Pendentes:       {progress['pending']}")
    print("-" * 60)
    print(f"Taxa estimada:   {rate}s/chunk")
    print(f"Tempo restante:  {format_time(remaining_time)}")
    print(f"ETA conclusão:   {eta.strftime('%H:%M:%S')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Monitor enriquecimento")
    parser.add_argument(
        "--watch", "-w", action="store_true",
        help="Atualiza a cada 30 segundos"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=30,
        help="Intervalo de atualização em segundos"
    )
    args = parser.parse_args()

    if args.watch:
        print("Monitorando... (Ctrl+C para sair)")
        last_enriched = 0
        last_time = time.time()

        while True:
            try:
                progress = check_progress()

                # Calcula taxa real
                current_time = time.time()
                elapsed = current_time - last_time
                new_chunks = progress["enriched"] - last_enriched

                if new_chunks > 0 and elapsed > 0:
                    rate = elapsed / new_chunks
                else:
                    rate = 4.5  # Estimativa padrão

                display_progress(progress, rate)

                if progress["pending"] == 0:
                    print("\n✓ ENRIQUECIMENTO CONCLUÍDO!")
                    break

                last_enriched = progress["enriched"]
                last_time = current_time

                time.sleep(args.interval)
                print("\n" * 2)

            except KeyboardInterrupt:
                print("\nMonitoramento interrompido.")
                break
    else:
        progress = check_progress()
        display_progress(progress)


if __name__ == "__main__":
    main()
