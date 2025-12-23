"""
Modulo de embeddings para o pipeline de documentos legais.

Componentes:
- BGE-M3: Embeddings dense (1024d) + sparse (learned)
- BGE-Reranker-v2-m3: Cross-encoder para reranking

Arquitetura 2-stage retrieval:
1. BGE-M3 busca top 100 candidatos (rapido, bi-encoder)
2. BGE-Reranker reordena para top 10 (preciso, cross-encoder)

Ambos rodam localmente via FlagEmbedding (nao em Docker).
"""

from .bge_m3 import (
    BGEM3Embedder,
    EmbeddingConfig,
    MockEmbedder,
    get_embedder,
    HybridEmbeddings,
    SingleHybridEmbedding,
)

from .bge_reranker import (
    BGEReranker,
    RerankerConfig,
    MockReranker,
    get_reranker,
)

__all__ = [
    # Embeddings
    "BGEM3Embedder",
    "EmbeddingConfig",
    "MockEmbedder",
    "get_embedder",
    "HybridEmbeddings",
    "SingleHybridEmbedding",
    # Reranker
    "BGEReranker",
    "RerankerConfig",
    "MockReranker",
    "get_reranker",
]
