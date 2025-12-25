"""
Singleton pool para modelos compartilhados.

Mantém BGE-M3 e Reranker em memória para evitar recarregamento.

## Modos de Operação

- **development** (padrão): Lazy loading, cada instância cria seus objetos
- **production**: Singleton, modelos mantidos na GPU permanentemente

Configure via variável de ambiente:
    export RAG_MODE=development  # GPU 12GB (padrão)
    export RAG_MODE=production   # GPU 24GB+

## Uso em Produção

O modo production mantém modelos na GPU, reduzindo latência de ~40s para ~10s.
Requer GPU com 24GB+ de VRAM (vLLM + BGE-M3 + Reranker = ~8GB).

## Uso em Desenvolvimento

O modo development não mantém modelos na GPU, liberando VRAM após cada uso.
Ideal para GPUs com 12GB onde não cabe tudo simultaneamente.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Modo de operação: 'development' (padrão) ou 'production'
RAG_MODE = os.getenv("RAG_MODE", "development").lower()

# Singletons (apenas usados em modo production)
_embedder = None
_reranker = None
_raw_embedder = None


def is_production_mode() -> bool:
    """Verifica se está em modo produção."""
    return RAG_MODE == "production"


def get_embedder():
    """
    Retorna wrapper BGEM3Embedder.

    Em modo production: retorna singleton (mantém na GPU).
    Em modo development: retorna None (força criar nova instância).
    """
    if not is_production_mode():
        logger.debug("Modo development: não usando singleton para embedder")
        return None

    global _embedder
    if _embedder is None:
        logger.info("Carregando BGEM3Embedder (singleton - modo production)...")
        from embeddings import BGEM3Embedder, EmbeddingConfig
        _embedder = BGEM3Embedder(EmbeddingConfig(use_fp16=True))
        _embedder._ensure_initialized()
        logger.info("BGEM3Embedder carregado!")
    return _embedder


def get_raw_embedder():
    """
    Retorna FlagEmbedding model diretamente.

    Em modo production: retorna singleton (mantém na GPU).
    Em modo development: retorna None (força criar nova instância).
    """
    if not is_production_mode():
        logger.debug("Modo development: não usando singleton para raw_embedder")
        return None

    global _raw_embedder
    if _raw_embedder is None:
        logger.info("Carregando BGEM3FlagModel (singleton - modo production)...")
        from FlagEmbedding import BGEM3FlagModel
        _raw_embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        logger.info("BGEM3FlagModel carregado!")
    return _raw_embedder


def get_reranker():
    """
    Retorna wrapper BGEReranker.

    Em modo production: retorna singleton (mantém na GPU).
    Em modo development: retorna None (força criar nova instância).
    """
    if not is_production_mode():
        logger.debug("Modo development: não usando singleton para reranker")
        return None

    global _reranker
    if _reranker is None:
        logger.info("Carregando BGEReranker (singleton - modo production)...")
        from embeddings import BGEReranker, RerankerConfig
        _reranker = BGEReranker(RerankerConfig(use_fp16=True))
        _reranker._ensure_initialized()
        logger.info("BGEReranker carregado!")
    return _reranker


def preload_models():
    """
    Pré-carrega todos os modelos.

    Só funciona em modo production. Em development, não faz nada.
    """
    if not is_production_mode():
        logger.warning(
            "preload_models() chamado em modo development - ignorando. "
            "Use RAG_MODE=production para habilitar singletons."
        )
        return

    logger.info("Pré-carregando modelos (modo production)...")
    get_embedder()
    get_reranker()
    logger.info("Modelos prontos na GPU!")


def get_mode_info() -> dict:
    """Retorna informações sobre o modo atual."""
    return {
        "mode": RAG_MODE,
        "is_production": is_production_mode(),
        "embedder_loaded": _embedder is not None,
        "reranker_loaded": _reranker is not None,
        "raw_embedder_loaded": _raw_embedder is not None,
    }
