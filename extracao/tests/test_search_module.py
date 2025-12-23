"""
Teste do modulo de busca hibrida.

Demonstra uso do HybridSearcher com diferentes configuracoes.

Uso:
    python tests/test_search_module.py
"""

import sys
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from search import (
    HybridSearcher,
    SearchConfig,
    SearchFilter,
    SearchMode,
    RerankMode,
)


def test_search_basic():
    """Teste basico de busca."""
    print("=" * 70)
    print("TESTE 1: Busca Basica")
    print("=" * 70)

    # Configuracao padrao (3-way hybrid + reranker)
    config = SearchConfig.default()
    print(f"\nConfig: {config.search_mode.value} + {config.rerank_mode.value}")
    print(f"Pesos: dense={config.weight_dense}, sparse={config.weight_sparse}, thesis={config.weight_thesis}")

    with HybridSearcher(config) as searcher:
        print(f"\n{searcher}")

        query = "O que e ETP e qual sua finalidade?"
        print(f"\nQuery: {query}")

        result = searcher.search(query, top_k=5)

        print(f"\nResultados: {len(result.hits)} hits")
        print(f"Tempo: Stage1={result.stage1_time_ms:.1f}ms, Stage2={result.stage2_time_ms:.1f}ms")
        print(f"Total: {result.total_time_ms:.1f}ms")

        print("\nTop 5:")
        for i, hit in enumerate(result.hits, 1):
            print(f"\n  {i}. Art. {hit.article_number}")
            print(f"     Score: {hit.score:.4f} -> Rerank: {hit.rerank_score:.4f}")
            print(f"     Tipo: {hit.thesis_type}")
            if hit.context_header:
                print(f"     Contexto: {hit.context_header[:60]}...")


def test_search_fast():
    """Teste de busca rapida (sem reranking)."""
    print("\n" + "=" * 70)
    print("TESTE 2: Busca Rapida (sem reranking)")
    print("=" * 70)

    config = SearchConfig.fast()
    print(f"\nConfig: {config.search_mode.value} + {config.rerank_mode.value}")

    with HybridSearcher(config) as searcher:
        query = "Quando o ETP e dispensado?"
        print(f"\nQuery: {query}")

        result = searcher.search(query, top_k=3)

        print(f"\nResultados: {len(result.hits)} hits em {result.total_time_ms:.1f}ms")

        for i, hit in enumerate(result.hits, 1):
            print(f"  {i}. Art. {hit.article_number} | Score: {hit.score:.4f} | {hit.thesis_type}")


def test_search_with_filters():
    """Teste de busca com filtros."""
    print("\n" + "=" * 70)
    print("TESTE 3: Busca com Filtros")
    print("=" * 70)

    config = SearchConfig.default()

    with HybridSearcher(config) as searcher:
        # Filtro por tipo de documento
        filters = SearchFilter(
            document_type="IN",
            # thesis_types=["definicao", "procedimento"],
        )

        print(f"\nFiltros: {filters}")
        print(f"Expr Milvus: {filters.to_milvus_expr()}")

        query = "definicoes basicas"
        print(f"Query: {query}")

        result = searcher.search(query, top_k=5, filters=filters)

        print(f"\nResultados: {len(result.hits)} hits")
        for i, hit in enumerate(result.hits, 1):
            print(f"  {i}. Art. {hit.article_number} | {hit.document_type} | {hit.thesis_type}")


def test_search_compare_modes():
    """Compara diferentes modos de busca."""
    print("\n" + "=" * 70)
    print("TESTE 4: Comparacao de Modos")
    print("=" * 70)

    query = "contratacoes correlatas e interdependentes"
    print(f"\nQuery: {query}")

    modes = [
        ("Dense Only", SearchConfig.dense_only()),
        ("Hybrid (2-way)", SearchConfig.fast()),
        ("Hybrid 3-way", SearchConfig.default()),
        ("Precise", SearchConfig.precise()),
    ]

    for name, config in modes:
        print(f"\n--- {name} ---")
        print(f"    Mode: {config.search_mode.value}, Rerank: {config.rerank_mode.value}")

        with HybridSearcher(config) as searcher:
            result = searcher.search(query, top_k=3)

            print(f"    Tempo: {result.total_time_ms:.1f}ms")
            top_arts = [f"Art.{h.article_number}" for h in result.hits]
            print(f"    Top 3: {', '.join(top_arts)}")


def test_search_simple_api():
    """Teste da API simplificada."""
    print("\n" + "=" * 70)
    print("TESTE 5: API Simplificada")
    print("=" * 70)

    with HybridSearcher() as searcher:
        # Retorna lista de dicts
        results = searcher.search_simple("responsaveis pela elaboracao do ETP", top_k=3)

        print(f"\nResultados (dicts): {len(results)}")
        for r in results:
            print(f"  - Art. {r['article_number']}: {r['final_score']:.4f}")


def test_search_helper():
    """Teste da funcao helper."""
    print("\n" + "=" * 70)
    print("TESTE 6: Funcao Helper")
    print("=" * 70)

    from search import search

    result = search("sistema ETP digital", top_k=3)

    print(f"\nQuery: {result.query}")
    print(f"Resultados: {len(result.hits)} hits em {result.total_time_ms:.1f}ms")

    for hit in result.hits:
        print(f"  - Art. {hit.article_number}: {hit.final_score:.4f}")


def main():
    """Executa todos os testes."""
    print("\n" + "=" * 70)
    print("TESTE DO MODULO DE BUSCA HIBRIDA")
    print("=" * 70)

    # Verifica conexao com Milvus
    try:
        from pymilvus import connections
        connections.connect(host="localhost", port="19530")
        connections.disconnect("default")
        print("\n[OK] Milvus conectado")
    except Exception as e:
        print(f"\n[ERRO] Erro ao conectar no Milvus: {e}")
        print("  Verifique se o Milvus esta rodando (docker-compose up -d)")
        return

    # Executa testes
    try:
        test_search_basic()
        test_search_fast()
        test_search_with_filters()
        test_search_compare_modes()
        test_search_simple_api()
        test_search_helper()

        print("\n" + "=" * 70)
        print("TODOS OS TESTES CONCLUIDOS")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERRO] Erro durante testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
