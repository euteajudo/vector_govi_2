"""
Modulo de busca hibrida para documentos legais.

Implementa 2-stage retrieval:
1. Stage 1: BGE-M3 (dense + sparse + thesis) no Milvus
2. Stage 2: BGE-Reranker (cross-encoder) para reordenacao

Uso basico:
    from search import HybridSearcher

    searcher = HybridSearcher()
    result = searcher.search("O que e ETP?")

    for hit in result.hits:
        print(f"Art. {hit.article_number}: {hit.final_score:.2f}")
        print(f"  {hit.context_header}")

Uso com configuracao:
    from search import HybridSearcher, SearchConfig

    config = SearchConfig.precise()  # Mais candidatos, mais preciso
    searcher = HybridSearcher(config)

    result = searcher.search("Quando ETP e dispensado?", top_k=10)

Uso com filtros:
    from search import HybridSearcher, SearchFilter

    filters = SearchFilter(
        document_type="IN",
        year=2022,
    )
    result = searcher.search("definicao de ETP", filters=filters)

Busca rapida (helper):
    from search import search

    result = search("O que e ETP?", top_k=5)
"""

from .config import SearchConfig, SearchMode, RerankMode
from .models import SearchHit, SearchResult, SearchFilter
from .hybrid_searcher import HybridSearcher, search

__all__ = [
    # Principal
    "HybridSearcher",
    "search",
    # Configuracao
    "SearchConfig",
    "SearchMode",
    "RerankMode",
    # Modelos
    "SearchHit",
    "SearchResult",
    "SearchFilter",
]
