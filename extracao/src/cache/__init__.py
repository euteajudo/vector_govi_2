"""
Módulo de cache para o sistema RAG.

Inclui:
- SemanticCache: Cache baseado em similaridade de embeddings
"""

from .semantic_cache import SemanticCache, CacheConfig, cached_response

__all__ = ["SemanticCache", "CacheConfig", "cached_response"]
