"""
Teste do AnswerGenerator - Geracao de respostas RAG.

Uso:
    python scripts/test_answer_generator.py
    python scripts/test_answer_generator.py --query "Quando o ETP pode ser dispensado?"
    python scripts/test_answer_generator.py --fast  # Sem HyDE e sem reranker
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag import AnswerGenerator, GenerationConfig


def test_answer_generator(query: str, use_fast: bool = False):
    """Testa geracao de resposta."""
    print("=" * 70)
    print("TESTE DO ANSWER GENERATOR")
    print("=" * 70)

    # Configuracao
    if use_fast:
        print("\n[CONFIG] Modo rapido (sem HyDE, sem reranker)")
        config = GenerationConfig.fast()
    else:
        print("\n[CONFIG] Modo padrao (com HyDE, com reranker)")
        config = GenerationConfig.default()

    print(f"  Model: {config.model}")
    print(f"  Top-K: {config.top_k}")
    print(f"  HyDE: {config.use_hyde}")
    print(f"  Reranker: {config.use_reranker}")

    # Query
    print(f"\n[QUERY] {query}")
    print("-" * 70)

    # Gera resposta
    with AnswerGenerator(config=config) as generator:
        response = generator.generate(query)

    # Resultados
    print("\n[RESPOSTA]")
    print(response.answer)

    print(f"\n[CONFIANCA] {response.confidence:.1%}")

    print(f"\n[CITACOES] ({len(response.citations)} encontradas)")
    for i, citation in enumerate(response.citations, 1):
        print(f"\n  [{i}] {citation.text}")
        print(f"      Artigo: {citation.article}")
        if citation.device:
            print(f"      Dispositivo: {citation.device} {citation.device_number}")

    print(f"\n[FONTES] ({len(response.sources)} documentos)")
    for source in response.sources:
        print(f"  - {source['document_id']}: {source['tipo_documento']} {source['numero']}/{source['ano']}")

    print(f"\n[METRICAS]")
    print(f"  Latencia total: {response.metadata.latency_ms}ms")
    print(f"  Retrieval: {response.metadata.retrieval_ms}ms")
    print(f"  Generation: {response.metadata.generation_ms}ms")
    print(f"  Chunks recuperados: {response.metadata.chunks_retrieved}")
    print(f"  Chunks usados: {response.metadata.chunks_used}")

    print("\n" + "=" * 70)

    # Retorna resposta para uso programatico
    return response


def main():
    parser = argparse.ArgumentParser(description="Testa o AnswerGenerator")
    parser.add_argument(
        "--query", "-q",
        default="Quando o ETP pode ser dispensado?",
        help="Pergunta para testar"
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Modo rapido (sem HyDE, sem reranker)"
    )
    args = parser.parse_args()

    test_answer_generator(args.query, use_fast=args.fast)


if __name__ == "__main__":
    main()
