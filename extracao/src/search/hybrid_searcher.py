"""
Buscador Hibrido para documentos legais.

Implementa busca 2-stage:
1. Stage 1: Busca hibrida no Milvus (dense + sparse + thesis)
2. Stage 2: Reranking com cross-encoder (BGE-Reranker)

Uso:
    from search import HybridSearcher, SearchConfig

    searcher = HybridSearcher()
    result = searcher.search("O que e ETP?")

    for hit in result.hits:
        print(f"{hit.article_number}: {hit.final_score:.4f}")
"""

import logging
import time
from typing import Optional

from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker

from .config import SearchConfig, SearchMode, RerankMode
from .models import SearchHit, SearchResult, SearchFilter

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Buscador hibrido para documentos legais no Milvus.

    Combina busca vetorial (dense + sparse + thesis) com reranking
    para maximizar precisao em buscas juridicas.

    Attributes:
        config: Configuracao de busca
        embedder: Modelo BGE-M3 para embeddings
        reranker: Modelo BGE-Reranker para reranking
        collection: Collection Milvus carregada
    """

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        embedder=None,
        reranker=None,
    ):
        """
        Inicializa o buscador.

        Args:
            config: Configuracao de busca. Usa default se nao fornecido.
            embedder: Instancia de BGEM3Embedder. Cria novo se nao fornecido.
            reranker: Instancia de BGEReranker. Cria novo se nao fornecido.
        """
        self.config = config or SearchConfig.default()
        self._embedder = embedder
        self._reranker = reranker
        self._hyde_expander = None
        self._collection: Optional[Collection] = None
        self._connected = False

    # =========================================================================
    # LAZY LOADING
    # =========================================================================

    @property
    def embedder(self):
        """Carrega embedder sob demanda."""
        if self._embedder is None:
            from embeddings import BGEM3Embedder, EmbeddingConfig

            logger.info("Carregando BGE-M3 embedder...")
            self._embedder = BGEM3Embedder(
                EmbeddingConfig(use_fp16=self.config.use_fp16)
            )
        return self._embedder

    @property
    def reranker(self):
        """Carrega reranker sob demanda."""
        if self._reranker is None and self.config.rerank_mode == RerankMode.CROSS_ENCODER:
            from embeddings import BGEReranker, RerankerConfig

            logger.info("Carregando BGE-Reranker...")
            self._reranker = BGEReranker(
                RerankerConfig(use_fp16=self.config.use_fp16)
            )
        return self._reranker

    @property
    def hyde_expander(self):
        """Carrega HyDE expander sob demanda (apenas se use_hyde=True)."""
        if self._hyde_expander is None and self.config.use_hyde:
            from .hyde_expander import HyDEExpander
            from llm.vllm_client import VLLMClient, LLMConfig

            logger.info("Carregando HyDE expander...")
            llm = VLLMClient(LLMConfig.for_enrichment())
            self._hyde_expander = HyDEExpander(
                llm_client=llm,
                embedder=self.embedder,
                n_hypothetical=self.config.hyde_n_hypothetical,
                query_weight=self.config.hyde_query_weight,
                doc_weight=self.config.hyde_doc_weight,
            )
        return self._hyde_expander

    @property
    def collection(self) -> Collection:
        """Conecta ao Milvus e carrega collection sob demanda."""
        if self._collection is None:
            self._connect()
        return self._collection

    def _connect(self):
        """Conecta ao Milvus e carrega a collection."""
        if self._connected:
            return

        logger.info(
            f"Conectando ao Milvus ({self.config.milvus_host}:{self.config.milvus_port})..."
        )

        connections.connect(
            alias="default",
            host=self.config.milvus_host,
            port=self.config.milvus_port,
        )

        self._collection = Collection(self.config.collection_name)
        self._collection.load()

        self._connected = True
        logger.info(
            f"Collection '{self.config.collection_name}' carregada "
            f"({self._collection.num_entities} entidades)"
        )

    def disconnect(self):
        """Desconecta do Milvus."""
        if self._connected:
            connections.disconnect("default")
            self._collection = None
            self._connected = False
            logger.info("Desconectado do Milvus")

    # =========================================================================
    # BUSCA PRINCIPAL
    # =========================================================================

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilter] = None,
        use_reranker: Optional[bool] = None,
    ) -> SearchResult:
        """
        Executa busca hibrida com reranking opcional.

        Args:
            query: Texto da consulta
            top_k: Numero de resultados (default: config.top_k)
            filters: Filtros opcionais (tipo_documento, ano, etc)
            use_reranker: Forcar uso/nao-uso de reranker (default: config)

        Returns:
            SearchResult com hits ordenados por relevancia
        """
        top_k = top_k or self.config.top_k
        use_reranker = use_reranker if use_reranker is not None else (
            self.config.rerank_mode != RerankMode.NONE
        )

        # Stage 1: Busca hibrida no Milvus
        stage1_start = time.perf_counter()
        stage1_hits = self._search_stage1(query, filters)
        stage1_time = (time.perf_counter() - stage1_start) * 1000

        logger.debug(f"Stage 1: {len(stage1_hits)} hits em {stage1_time:.1f}ms")

        # Stage 2: Reranking (opcional)
        stage2_time = 0.0
        if use_reranker and stage1_hits:
            stage2_start = time.perf_counter()
            stage1_hits = self._rerank_stage2(query, stage1_hits)
            stage2_time = (time.perf_counter() - stage2_start) * 1000
            logger.debug(f"Stage 2: reranked em {stage2_time:.1f}ms")

        # Aplica top_k
        final_hits = stage1_hits[:top_k]

        return SearchResult(
            query=query,
            hits=final_hits,
            total_found=len(stage1_hits),
            stage1_time_ms=stage1_time,
            stage2_time_ms=stage2_time,
            top_k=top_k,
            use_reranker=use_reranker,
            weights=self.config.weights,
        )

    # =========================================================================
    # STAGE 1: BUSCA HIBRIDA NO MILVUS
    # =========================================================================

    def _search_stage1(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
    ) -> list[SearchHit]:
        """
        Executa busca hibrida no Milvus (Stage 1).

        Combina dense, sparse e thesis vectors com pesos configurados.

        Args:
            query: Texto da consulta
            filters: Filtros opcionais

        Returns:
            Lista de SearchHit ordenados por score combinado
        """
        # Gera embedding da query (com HyDE se habilitado)
        if self.config.use_hyde and self.hyde_expander:
            logger.debug("Usando HyDE para query expansion")
            hyde_result = self.hyde_expander.expand(query)
            query_embedding = {
                "dense": hyde_result.combined_dense,
                "sparse": hyde_result.combined_sparse,
            }
        else:
            query_embedding = self.embedder.encode_hybrid_single(query)

        # Prepara expressao de filtro
        filter_expr = filters.to_milvus_expr() if filters else ""

        # Monta requests de busca baseado no modo
        search_requests = []

        if self.config.search_mode in [SearchMode.DENSE_ONLY, SearchMode.HYBRID, SearchMode.HYBRID_3WAY]:
            # Dense search
            dense_req = AnnSearchRequest(
                data=[query_embedding["dense"]],
                anns_field="dense_vector",
                param={
                    "metric_type": "COSINE",
                    "params": {"nprobe": self.config.nprobe},
                },
                limit=self.config.stage1_limit,
                expr=filter_expr if filter_expr else None,
            )
            search_requests.append(dense_req)

        if self.config.search_mode in [SearchMode.HYBRID, SearchMode.HYBRID_3WAY]:
            # Sparse search
            sparse_req = AnnSearchRequest(
                data=[query_embedding["sparse"]],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=self.config.stage1_limit,
                expr=filter_expr if filter_expr else None,
            )
            search_requests.append(sparse_req)

        if self.config.search_mode == SearchMode.HYBRID_3WAY:
            # Thesis search
            thesis_req = AnnSearchRequest(
                data=[query_embedding["dense"]],
                anns_field="thesis_vector",
                param={
                    "metric_type": "COSINE",
                    "params": {"nprobe": self.config.nprobe},
                },
                limit=self.config.stage1_limit,
                expr=filter_expr if filter_expr else None,
            )
            search_requests.append(thesis_req)

        # Determina pesos para o ranker
        weights = self._get_weights_for_mode()

        # Executa hybrid search
        results = self.collection.hybrid_search(
            reqs=search_requests,
            rerank=WeightedRanker(*weights),
            limit=self.config.stage1_limit,
            output_fields=self.config.output_fields,
        )

        # Converte para SearchHit
        hits = []
        for milvus_hit in results[0]:
            hit = SearchHit.from_milvus_hit(milvus_hit)
            hits.append(hit)

        return hits

    def _get_weights_for_mode(self) -> tuple:
        """Retorna pesos apropriados para o modo de busca."""
        if self.config.search_mode == SearchMode.DENSE_ONLY:
            return (1.0,)
        elif self.config.search_mode == SearchMode.HYBRID:
            # Normaliza dense + sparse para somar 1
            total = self.config.weight_dense + self.config.weight_sparse
            return (
                self.config.weight_dense / total,
                self.config.weight_sparse / total,
            )
        else:  # HYBRID_3WAY
            return self.config.weights

    # =========================================================================
    # STAGE 2: RERANKING
    # =========================================================================

    def _rerank_stage2(
        self,
        query: str,
        hits: list[SearchHit],
    ) -> list[SearchHit]:
        """
        Reordena resultados com cross-encoder (Stage 2).

        IMPORTANTE: Usa texto ORIGINAL para reranking, não enriched_text.
        O enriched_text tem prefixo [CONTEXTO: ...] que dilui a relevância
        para o cross-encoder. Testes mostraram:
        - Texto original: score 0.55
        - Enriched_text: score 0.27

        Args:
            query: Texto da consulta
            hits: Resultados do Stage 1

        Returns:
            Lista reordenada por rerank_score
        """
        if not hits or not self.reranker:
            return hits

        # Prepara documentos para reranking
        # USA TEXTO ORIGINAL - melhor para cross-encoder
        documents = [
            {
                "hit": hit,
                "text": hit.text,  # Texto original, sem prefixo [CONTEXTO]
            }
            for hit in hits
        ]

        # Executa reranking
        reranked = self.reranker.rerank(
            query=query,
            documents=documents,
            text_key="text",  # Usa texto original
            return_scores=True,
        )

        # Atualiza hits com scores do reranker
        result_hits = []
        for doc in reranked:
            hit = doc["hit"]
            hit.rerank_score = doc.get("rerank_score", 0.0)
            result_hits.append(hit)

        return result_hits

    # =========================================================================
    # METODOS AUXILIARES
    # =========================================================================

    def search_simple(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Busca simplificada que retorna dicts.

        Util para integracao com APIs.

        Args:
            query: Texto da consulta
            top_k: Numero de resultados

        Returns:
            Lista de dicts com resultados
        """
        result = self.search(query, top_k=top_k)
        return [hit.to_dict() for hit in result.hits]

    def search_articles(
        self,
        query: str,
        document_type: Optional[str] = None,
        year: Optional[int] = None,
        top_k: int = 5,
    ) -> SearchResult:
        """
        Busca artigos com filtros comuns.

        Args:
            query: Texto da consulta
            document_type: Filtrar por tipo (LEI, DECRETO, IN)
            year: Filtrar por ano
            top_k: Numero de resultados

        Returns:
            SearchResult filtrado
        """
        filters = SearchFilter(
            document_type=document_type,
            year=year,
        )
        return self.search(query, top_k=top_k, filters=filters)

    def get_article_by_number(
        self,
        article_number: str,
        document_number: Optional[str] = None,
    ) -> Optional[SearchHit]:
        """
        Busca artigo especifico por numero.

        Args:
            article_number: Numero do artigo (ex: "3", "14")
            document_number: Numero do documento (opcional)

        Returns:
            SearchHit ou None se nao encontrado
        """
        filters = SearchFilter(
            article_numbers=[article_number],
            document_number=document_number,
        )

        # Usa busca simples (dense only)
        old_mode = self.config.search_mode
        self.config.search_mode = SearchMode.DENSE_ONLY

        # Query generica para encontrar o artigo
        result = self.search(
            query=f"Artigo {article_number}",
            top_k=1,
            filters=filters,
            use_reranker=False,
        )

        self.config.search_mode = old_mode

        return result.hits[0] if result.hits else None

    def __enter__(self):
        """Suporte a context manager."""
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Desconecta ao sair do context."""
        self.disconnect()

    def __repr__(self) -> str:
        return (
            f"HybridSearcher("
            f"collection='{self.config.collection_name}', "
            f"mode={self.config.search_mode.value}, "
            f"rerank={self.config.rerank_mode.value}, "
            f"weights={self.config.weights})"
        )


# =============================================================================
# FUNCAO HELPER
# =============================================================================

def search(
    query: str,
    top_k: int = 5,
    config: Optional[SearchConfig] = None,
) -> SearchResult:
    """
    Funcao helper para busca rapida.

    Cria um searcher temporario e executa a busca.
    Para multiplas buscas, use HybridSearcher diretamente.

    Args:
        query: Texto da consulta
        top_k: Numero de resultados
        config: Configuracao opcional

    Returns:
        SearchResult com hits
    """
    with HybridSearcher(config) as searcher:
        return searcher.search(query, top_k=top_k)
