"""
Retrieval Contextual com MMR e Query Router.

Implementa:
1. Busca híbrida (Weighted ou RRF baseado na query)
2. Expansão parent-child com cap
3. MMR (Maximal Marginal Relevance) para diversidade de irmãos
4. Validação de citações

Uso:
    from search import ContextualRetriever

    retriever = ContextualRetriever()
    result = retriever.retrieve("Como fazer pesquisa de preços?")

    # Resultado inclui:
    # - chunks: lista de chunks recuperados
    # - context: texto concatenado para o LLM
    # - citations: span_ids usados
    # - strategy: 'weighted' ou 'rrf'
"""

import re
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker, WeightedRanker


class RetrievalStrategy(str, Enum):
    """Estratégia de retrieval."""
    WEIGHTED = "weighted"  # 0.7 dense + 0.3 sparse (padrão)
    RRF = "rrf"            # Reciprocal Rank Fusion (dispositivo específico)


@dataclass
class RetrievedChunk:
    """Chunk recuperado com metadados."""
    chunk_id: str
    span_id: str
    device_type: str
    text: str
    score: float
    parent_chunk_id: str = ""
    article_number: str = ""
    is_parent: bool = False
    is_sibling: bool = False


@dataclass
class RetrievalResult:
    """Resultado do retrieval contextual."""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    context: str = ""
    citations: list[str] = field(default_factory=list)
    strategy: RetrievalStrategy = RetrievalStrategy.WEIGHTED
    query: str = ""

    # Estatísticas
    top_k_retrieved: int = 0
    parents_expanded: int = 0
    siblings_added: int = 0
    mmr_filtered: int = 0


@dataclass
class RetrieverConfig:
    """Configuração do retriever."""
    # Busca inicial
    top_k: int = 5

    # Pesos para Weighted
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    # Expansão
    max_parents: int = 1
    max_siblings: int = 4

    # MMR
    mmr_lambda: float = 0.7  # 0=diversidade total, 1=relevância total
    mmr_threshold: float = 0.85  # Filtra irmãos muito similares

    # Milvus
    collection_name: str = "leis_v3"
    ef_search: int = 64


class ContextualRetriever:
    """
    Retriever contextual com expansão parent-child e MMR.

    Fluxo:
    1. Detecta estratégia (Weighted ou RRF) baseado na query
    2. Busca híbrida inicial (top_k)
    3. Expande para pais (artigos)
    4. Seleciona irmãos relevantes com MMR
    5. Monta contexto ordenado hierarquicamente
    """

    # Padrões para detectar query de dispositivo específico
    DEVICE_PATTERNS = [
        r'\bart\.?\s*\d+',      # art. 5, art 10
        r'\bartigo\s*\d+',      # artigo 5
        r'§\s*\d+',             # § 1º, § 2
        r'\binciso\b',          # inciso
        r'\bal[ií]nea\b',       # alínea
        r'\b[IVX]+\s*[-–]',     # I -, II -, III -
        r'\b[a-d]\)',           # a), b), c)
    ]

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        embedder=None,
    ):
        """
        Inicializa o retriever.

        Args:
            config: Configuração do retriever
            embedder: Embedder BGE-M3 (lazy load se não fornecido)
        """
        self.config = config or RetrieverConfig()
        self._embedder = embedder
        self._collection = None
        self._connected = False

    def _ensure_connected(self):
        """Conecta ao Milvus se necessário."""
        if not self._connected:
            connections.connect(host="localhost", port="19530")
            self._collection = Collection(self.config.collection_name)
            self._collection.load()
            self._connected = True

    def _ensure_embedder(self):
        """Carrega embedder se necessário.
        
        Em modo production (RAG_MODE=production): usa singleton do model_pool.
        Em modo development (padrao): cria nova instancia local.
        """
        if self._embedder is None:
            from model_pool import get_raw_embedder
            self._embedder = get_raw_embedder()
            
            # Se retornou None (modo development), cria instancia local
            if self._embedder is None:
                from FlagEmbedding import BGEM3FlagModel
                self._embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def detect_strategy(self, query: str) -> RetrievalStrategy:
        """
        Detecta estratégia baseado na query.

        - RRF: Query contém referência a dispositivo específico
        - Weighted: Query conceitual/ampla
        """
        query_lower = query.lower()

        for pattern in self.DEVICE_PATTERNS:
            if re.search(pattern, query_lower):
                return RetrievalStrategy.RRF

        return RetrievalStrategy.WEIGHTED

    def retrieve(
        self,
        query: str,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> RetrievalResult:
        """
        Executa retrieval contextual completo.

        Args:
            query: Pergunta do usuário
            strategy: Força estratégia (auto-detecta se None)

        Returns:
            RetrievalResult com chunks, contexto e citações
        """
        self._ensure_connected()
        self._ensure_embedder()

        # 1. Detecta estratégia
        if strategy is None:
            strategy = self.detect_strategy(query)

        result = RetrievalResult(query=query, strategy=strategy)

        # 2. Gera embeddings
        enc = self._embedder.encode([query], return_dense=True, return_sparse=True)
        dense_vec = enc['dense_vecs'][0].tolist()
        sparse_dict = enc['lexical_weights'][0]
        sparse_vec = {int(k): float(v) for k, v in sparse_dict.items() if abs(v) > 1e-6}

        # 3. Busca híbrida
        chunks = self._hybrid_search(dense_vec, sparse_vec, strategy)
        result.top_k_retrieved = len(chunks)

        if not chunks:
            return result

        # 4. Expande para pais
        expanded = self._expand_to_parents(chunks)
        result.parents_expanded = sum(1 for c in expanded if c.is_parent)

        # 5. Adiciona irmãos com MMR
        with_siblings = self._add_siblings_mmr(expanded, dense_vec)
        result.siblings_added = sum(1 for c in with_siblings if c.is_sibling)

        # 6. Ordena hierarquicamente e monta contexto
        result.chunks = self._order_hierarchically(with_siblings)
        result.context = self._build_context(result.chunks)
        result.citations = [c.span_id for c in result.chunks]

        return result

    def _hybrid_search(
        self,
        dense_vec: list[float],
        sparse_vec: dict[int, float],
        strategy: RetrievalStrategy,
    ) -> list[RetrievedChunk]:
        """Executa busca híbrida."""
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field='dense_vector',
            param={'metric_type': 'COSINE', 'params': {'ef': self.config.ef_search}},
            limit=self.config.top_k * 2  # Busca mais para compensar filtros
        )

        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field='sparse_vector',
            param={'metric_type': 'IP', 'params': {}},
            limit=self.config.top_k * 2
        )

        if strategy == RetrievalStrategy.RRF:
            ranker = RRFRanker(k=60)
        else:
            ranker = WeightedRanker(
                self.config.dense_weight,
                self.config.sparse_weight
            )

        results = self._collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=self.config.top_k,
            output_fields=[
                'chunk_id', 'span_id', 'device_type', 'text',
                'parent_chunk_id', 'article_number'
            ]
        )

        chunks = []
        for hit in results[0]:
            chunk = RetrievedChunk(
                chunk_id=hit.entity.get('chunk_id', ''),
                span_id=hit.entity.get('span_id', ''),
                device_type=hit.entity.get('device_type', ''),
                text=hit.entity.get('text', ''),
                score=hit.score,
                parent_chunk_id=hit.entity.get('parent_chunk_id', ''),
                article_number=hit.entity.get('article_number', ''),
            )
            chunks.append(chunk)

        return chunks

    def _expand_to_parents(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Expande chunks para incluir pais (artigos)."""
        result = list(chunks)
        seen_ids = {c.chunk_id for c in chunks}
        parents_added = 0

        for chunk in chunks:
            if parents_added >= self.config.max_parents:
                break

            if chunk.parent_chunk_id and chunk.parent_chunk_id not in seen_ids:
                # Busca o pai
                parent_results = self._collection.query(
                    expr=f'chunk_id == "{chunk.parent_chunk_id}"',
                    output_fields=[
                        'chunk_id', 'span_id', 'device_type', 'text',
                        'parent_chunk_id', 'article_number'
                    ],
                    limit=1
                )

                if parent_results:
                    p = parent_results[0]
                    parent_chunk = RetrievedChunk(
                        chunk_id=p['chunk_id'],
                        span_id=p['span_id'],
                        device_type=p['device_type'],
                        text=p['text'],
                        score=0.0,  # Pai não tem score de busca
                        parent_chunk_id=p.get('parent_chunk_id', ''),
                        article_number=p.get('article_number', ''),
                        is_parent=True,
                    )
                    result.append(parent_chunk)
                    seen_ids.add(parent_chunk.chunk_id)
                    parents_added += 1

        return result

    def _add_siblings_mmr(
        self,
        chunks: list[RetrievedChunk],
        query_vec: list[float],
    ) -> list[RetrievedChunk]:
        """
        Adiciona irmãos relevantes usando MMR para diversidade.

        MMR = λ * sim(doc, query) - (1-λ) * max(sim(doc, selected))
        """
        result = list(chunks)
        seen_ids = {c.chunk_id for c in chunks}

        # Encontra pais únicos
        parent_ids = set()
        for chunk in chunks:
            if chunk.parent_chunk_id:
                parent_ids.add(chunk.parent_chunk_id)
            elif chunk.device_type == 'article':
                parent_ids.add(chunk.chunk_id)

        if not parent_ids:
            return result

        # Busca todos os irmãos potenciais
        for parent_id in list(parent_ids)[:self.config.max_parents]:
            siblings = self._collection.query(
                expr=f'parent_chunk_id == "{parent_id}"',
                output_fields=[
                    'chunk_id', 'span_id', 'device_type', 'text',
                    'parent_chunk_id', 'article_number', 'dense_vector'
                ],
                limit=20
            )

            # Filtra irmãos já incluídos
            new_siblings = [s for s in siblings if s['chunk_id'] not in seen_ids]

            if not new_siblings:
                continue

            # Aplica MMR para selecionar os melhores
            selected = self._mmr_select(
                new_siblings,
                query_vec,
                [c for c in chunks if c.chunk_id in seen_ids],
                max_select=self.config.max_siblings
            )

            for s in selected:
                sibling_chunk = RetrievedChunk(
                    chunk_id=s['chunk_id'],
                    span_id=s['span_id'],
                    device_type=s['device_type'],
                    text=s['text'],
                    score=0.0,
                    parent_chunk_id=s.get('parent_chunk_id', ''),
                    article_number=s.get('article_number', ''),
                    is_sibling=True,
                )
                result.append(sibling_chunk)
                seen_ids.add(sibling_chunk.chunk_id)

        return result

    def _mmr_select(
        self,
        candidates: list[dict],
        query_vec: list[float],
        selected: list[RetrievedChunk],
        max_select: int,
    ) -> list[dict]:
        """
        Seleciona candidatos usando MMR.

        MMR = λ * sim(doc, query) - (1-λ) * max(sim(doc, selected))
        """
        if not candidates:
            return []

        query_vec = np.array(query_vec)

        # Calcula similaridade com query para cada candidato
        candidate_vecs = []
        for c in candidates:
            vec = c.get('dense_vector', [])
            if vec:
                candidate_vecs.append(np.array(vec))
            else:
                candidate_vecs.append(np.zeros(1024))

        query_sims = []
        for vec in candidate_vecs:
            sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
            query_sims.append(sim)

        # Vetores já selecionados (para calcular diversidade)
        selected_vecs = []
        for chunk in selected:
            # Busca vetor do chunk selecionado
            res = self._collection.query(
                expr=f'chunk_id == "{chunk.chunk_id}"',
                output_fields=['dense_vector'],
                limit=1
            )
            if res and res[0].get('dense_vector'):
                selected_vecs.append(np.array(res[0]['dense_vector']))

        result = []
        remaining = list(range(len(candidates)))

        while len(result) < max_select and remaining:
            best_idx = None
            best_score = float('-inf')

            for idx in remaining:
                # Similaridade com query
                rel_score = query_sims[idx]

                # Máxima similaridade com já selecionados
                if selected_vecs:
                    div_scores = []
                    for sel_vec in selected_vecs:
                        sim = np.dot(candidate_vecs[idx], sel_vec) / (
                            np.linalg.norm(candidate_vecs[idx]) * np.linalg.norm(sel_vec) + 1e-8
                        )
                        div_scores.append(sim)
                    max_sim = max(div_scores)
                else:
                    max_sim = 0

                # MMR score
                mmr_score = (
                    self.config.mmr_lambda * rel_score -
                    (1 - self.config.mmr_lambda) * max_sim
                )

                # Filtra muito similares
                if max_sim > self.config.mmr_threshold:
                    continue

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                result.append(candidates[best_idx])
                selected_vecs.append(candidate_vecs[best_idx])
                remaining.remove(best_idx)
            else:
                break

        return result

    def _order_hierarchically(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Ordena chunks hierarquicamente (pai → filhos)."""
        # Agrupa por artigo
        by_article = {}
        for chunk in chunks:
            art_num = chunk.article_number or 'unknown'
            if art_num not in by_article:
                by_article[art_num] = {'parent': None, 'children': []}

            if chunk.device_type == 'article' or chunk.is_parent:
                by_article[art_num]['parent'] = chunk
            else:
                by_article[art_num]['children'].append(chunk)

        # Ordena cada grupo
        result = []
        for art_num in sorted(by_article.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            group = by_article[art_num]
            if group['parent']:
                result.append(group['parent'])

            # Ordena filhos por span_id
            children = sorted(group['children'], key=lambda c: c.span_id)
            result.extend(children)

        return result

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Monta contexto para o LLM."""
        parts = []

        for chunk in chunks:
            # Formata hierarquicamente
            if chunk.device_type == 'article':
                prefix = f"\n[ARTIGO {chunk.article_number}]\n"
            elif chunk.device_type == 'paragraph':
                prefix = f"\n  [{chunk.span_id}] "
            elif chunk.device_type == 'inciso':
                prefix = f"\n    [{chunk.span_id}] "
            else:
                prefix = f"\n      [{chunk.span_id}] "

            parts.append(f"{prefix}{chunk.text}")

        return "".join(parts).strip()

    def close(self):
        """Desconecta do Milvus."""
        if self._connected:
            connections.disconnect("default")
            self._connected = False


# =============================================================================
# VALIDADOR DE CITAÇÕES
# =============================================================================

@dataclass
class AnswerValidation:
    """Resultado da validação de resposta."""
    valid: bool = True
    grounded: bool = True
    missing_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    confidence: float = 1.0
    message: str = ""


class CitationValidator:
    """Valida citações em respostas do LLM."""

    def __init__(self, collection_name: str = "leis_v3"):
        self.collection_name = collection_name
        self._collection = None
        self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            connections.connect(host="localhost", port="19530")
            self._collection = Collection(self.collection_name)
            self._collection.load()
            self._connected = True

    def validate(
        self,
        citations: list[str],
        context_used: list[str],
    ) -> AnswerValidation:
        """
        Valida citações da resposta.

        Regras:
        1. citations ⊆ context_used (citações devem estar no contexto)
        2. Todos os span_ids devem existir no Milvus
        """
        self._ensure_connected()

        result = AnswerValidation()

        # Verifica se citações estão no contexto
        context_set = set(context_used)
        for cit in citations:
            if cit not in context_set:
                result.missing_citations.append(cit)

        # Verifica se span_ids existem
        all_spans = set(citations) | set(context_used)
        for span_id in all_spans:
            exists = self._collection.query(
                expr=f'span_id == "{span_id}"',
                output_fields=['span_id'],
                limit=1
            )
            if not exists:
                result.invalid_citations.append(span_id)

        # Calcula resultado
        if result.missing_citations or result.invalid_citations:
            result.valid = False
            result.grounded = False
            result.confidence = 0.3

            issues = []
            if result.missing_citations:
                issues.append(f"Citações não encontradas no contexto: {result.missing_citations}")
            if result.invalid_citations:
                issues.append(f"Span IDs inválidos: {result.invalid_citations}")

            result.message = "; ".join(issues)
        else:
            result.message = "Todas as citações válidas e fundamentadas"

        return result

    def close(self):
        if self._connected:
            connections.disconnect("default")
            self._connected = False


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE: ContextualRetriever com MMR")
    print("=" * 70)

    retriever = ContextualRetriever()

    # Teste 1: Query ampla (deve usar Weighted)
    query1 = "Como fazer pesquisa de preços?"
    print(f"\nQuery 1: {query1}")
    result1 = retriever.retrieve(query1)
    print(f"  Estratégia: {result1.strategy.value}")
    print(f"  Top-K: {result1.top_k_retrieved}")
    print(f"  Pais expandidos: {result1.parents_expanded}")
    print(f"  Irmãos MMR: {result1.siblings_added}")
    print(f"  Citações: {result1.citations[:5]}...")

    # Teste 2: Query de dispositivo (deve usar RRF)
    query2 = "O que diz o inciso IV do art. 5?"
    print(f"\nQuery 2: {query2}")
    result2 = retriever.retrieve(query2)
    print(f"  Estratégia: {result2.strategy.value}")
    print(f"  Top-K: {result2.top_k_retrieved}")
    print(f"  Citações: {result2.citations[:5]}...")

    retriever.close()

    print("\n" + "=" * 70)
    print("TESTE: CitationValidator")
    print("=" * 70)

    validator = CitationValidator()

    # Teste validação
    validation = validator.validate(
        citations=["ART-005", "INC-005-IV"],
        context_used=["ART-005", "INC-005-IV", "PAR-005-1"]
    )
    print(f"\nValidação: {validation.valid}")
    print(f"Grounded: {validation.grounded}")
    print(f"Mensagem: {validation.message}")

    validator.close()
