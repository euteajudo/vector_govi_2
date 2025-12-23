"""
Configuracoes para o modulo de busca hibrida.

Define parametros para busca vetorial, reranking e fusao de resultados.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SearchMode(Enum):
    """Modos de busca disponiveis."""

    # Busca apenas com dense vector
    DENSE_ONLY = "dense_only"

    # Busca hibrida dense + sparse
    HYBRID = "hybrid"

    # Busca hibrida com 3 vetores (dense + sparse + thesis)
    HYBRID_3WAY = "hybrid_3way"


class RerankMode(Enum):
    """Modos de reranking."""

    # Sem reranking (apenas Stage 1)
    NONE = "none"

    # Cross-encoder (BGE-Reranker)
    CROSS_ENCODER = "cross_encoder"

    # ColBERT (late interaction)
    COLBERT = "colbert"


@dataclass
class SearchConfig:
    """
    Configuracao completa para busca hibrida.

    Attributes:
        # Conexao Milvus
        milvus_host: Host do Milvus
        milvus_port: Porta do Milvus
        collection_name: Nome da collection

        # Modo de busca
        search_mode: Modo de busca (dense, hybrid, hybrid_3way)
        rerank_mode: Modo de reranking (none, cross_encoder, colbert)

        # Pesos para fusao (dense, sparse, thesis)
        weight_dense: Peso do vetor dense (0-1)
        weight_sparse: Peso do vetor sparse (0-1)
        weight_thesis: Peso do vetor thesis (0-1)

        # Limites
        stage1_limit: Candidatos do Stage 1 (antes do rerank)
        top_k: Resultados finais retornados

        # Parametros HNSW
        nprobe: Numero de clusters a buscar (HNSW)

        # Campos de saida
        output_fields: Campos a retornar do Milvus

        # Performance
        use_fp16: Usar FP16 para embeddings
        batch_size: Tamanho do batch para reranking
    """

    # Conexao
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "leis_v2"

    # Modos
    search_mode: SearchMode = SearchMode.HYBRID_3WAY
    rerank_mode: RerankMode = RerankMode.CROSS_ENCODER

    # Pesos (devem somar 1.0 para hybrid_3way)
    weight_dense: float = 0.5
    weight_sparse: float = 0.3
    weight_thesis: float = 0.2

    # Limites
    stage1_limit: int = 20
    top_k: int = 5

    # HNSW
    nprobe: int = 10

    # Campos de saida
    output_fields: Optional[list[str]] = None

    # Performance
    use_fp16: bool = True
    batch_size: int = 32

    def __post_init__(self):
        """Valida configuracao."""
        # Define campos padrao se nao especificados
        if self.output_fields is None:
            self.output_fields = [
                "chunk_id",
                "text",
                "enriched_text",
                "article_number",
                "context_header",
                "thesis_text",
                "thesis_type",
                "synthetic_questions",
                "tipo_documento",
                "numero_documento",
                "chapter_number",
                "chapter_title",
            ]

        # Valida pesos
        if self.search_mode == SearchMode.HYBRID_3WAY:
            total = self.weight_dense + self.weight_sparse + self.weight_thesis
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Pesos devem somar 1.0 para HYBRID_3WAY. "
                    f"Atual: {total:.2f} ({self.weight_dense} + {self.weight_sparse} + {self.weight_thesis})"
                )

    @property
    def weights(self) -> tuple[float, float, float]:
        """Retorna tupla de pesos (dense, sparse, thesis)."""
        return (self.weight_dense, self.weight_sparse, self.weight_thesis)

    @classmethod
    def default(cls) -> "SearchConfig":
        """Configuracao padrao (hybrid 3-way + cross-encoder)."""
        return cls()

    @classmethod
    def fast(cls) -> "SearchConfig":
        """Configuracao rapida (sem reranking)."""
        return cls(
            search_mode=SearchMode.HYBRID,
            rerank_mode=RerankMode.NONE,
            stage1_limit=10,
            top_k=5,
        )

    @classmethod
    def precise(cls) -> "SearchConfig":
        """Configuracao precisa (mais candidatos, reranking)."""
        return cls(
            search_mode=SearchMode.HYBRID_3WAY,
            rerank_mode=RerankMode.CROSS_ENCODER,
            stage1_limit=50,
            top_k=10,
            nprobe=20,
        )

    @classmethod
    def dense_only(cls) -> "SearchConfig":
        """Apenas busca densa (debug/comparacao)."""
        return cls(
            search_mode=SearchMode.DENSE_ONLY,
            rerank_mode=RerankMode.NONE,
            weight_dense=1.0,
            weight_sparse=0.0,
            weight_thesis=0.0,
        )

    @classmethod
    def for_legal_search(cls) -> "SearchConfig":
        """
        Configuracao otimizada para busca em documentos legais.

        Usa 3 vetores com pesos balanceados e reranking cross-encoder.
        """
        return cls(
            search_mode=SearchMode.HYBRID_3WAY,
            rerank_mode=RerankMode.CROSS_ENCODER,
            weight_dense=0.5,   # Semantica geral
            weight_sparse=0.3,  # Termos especificos
            weight_thesis=0.2,  # Essencia do artigo
            stage1_limit=20,
            top_k=5,
        )
