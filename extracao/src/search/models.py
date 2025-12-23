"""
Modelos de dados para o modulo de busca.

Define dataclasses para resultados de busca, facilitando
tipagem e manipulacao dos dados retornados.
"""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class SearchHit:
    """
    Resultado individual de uma busca.

    Attributes:
        chunk_id: ID unico do chunk no Milvus
        text: Texto original do artigo
        enriched_text: Texto enriquecido (contexto + texto + perguntas)
        article_number: Numero do artigo (ex: "3", "14")
        score: Score combinado da busca hibrida (Stage 1)
        rerank_score: Score do reranker (Stage 2), se aplicavel

        # Campos de enriquecimento
        context_header: Frase contextualizando o artigo
        thesis_text: Resumo objetivo do artigo
        thesis_type: Tipo: definicao, procedimento, excecao, etc.
        synthetic_questions: Perguntas que o artigo responde

        # Hierarquia legal
        document_type: Tipo do documento (LEI, DECRETO, IN)
        document_number: Numero do documento
        chapter_number: Numero do capitulo
        chapter_title: Titulo do capitulo

        # Metadados extras
        metadata: Campos adicionais do Milvus
    """

    # Identificacao
    chunk_id: str
    text: str
    enriched_text: str = ""
    article_number: str = ""

    # Campos v3 (parent-child)
    span_id: str = ""
    parent_chunk_id: str = ""
    device_type: str = ""  # article, paragraph, inciso, alinea
    document_id: str = ""

    # Scores
    score: float = 0.0
    milvus_score: float = 0.0  # Score original do Milvus
    rerank_score: Optional[float] = None

    # Enriquecimento
    context_header: str = ""
    thesis_text: str = ""
    thesis_type: str = ""
    synthetic_questions: str = ""

    # Hierarquia
    document_type: str = ""  # LEI, DECRETO, IN
    document_number: str = ""
    chapter_number: str = ""
    chapter_title: str = ""
    tipo_documento: str = ""  # Alias para document_type (compatibilidade Milvus)

    # Extras
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """Retorna o score final (rerank se disponivel, senao stage1)."""
        return self.rerank_score if self.rerank_score is not None else self.score

    @property
    def display_text(self) -> str:
        """Retorna texto para exibicao (enriched se disponivel)."""
        return self.enriched_text or self.text

    def to_dict(self) -> dict[str, Any]:
        """Converte para dicionario."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "enriched_text": self.enriched_text,
            "article_number": self.article_number,
            # Campos v3
            "span_id": self.span_id,
            "parent_chunk_id": self.parent_chunk_id,
            "device_type": self.device_type,
            "document_id": self.document_id,
            # Scores
            "score": self.score,
            "milvus_score": self.milvus_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            # Enriquecimento
            "context_header": self.context_header,
            "thesis_text": self.thesis_text,
            "thesis_type": self.thesis_type,
            "synthetic_questions": self.synthetic_questions,
            # Hierarquia
            "document_type": self.document_type,
            "document_number": self.document_number,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "tipo_documento": self.tipo_documento,
            "metadata": self.metadata,
        }

    @classmethod
    def from_milvus_hit(cls, hit: Any) -> "SearchHit":
        """
        Cria SearchHit a partir de resultado do Milvus.

        Args:
            hit: Resultado do Milvus (hit object)

        Returns:
            SearchHit populado
        """
        entity = hit.entity if hasattr(hit, "entity") else hit

        # Funcao helper para extrair campo
        def get_field(name: str, default: Any = "") -> Any:
            if hasattr(entity, "get"):
                return entity.get(name, default)
            return getattr(entity, name, default)

        milvus_score = hit.distance if hasattr(hit, "distance") else 0.0
        tipo_doc = get_field("tipo_documento", "")

        return cls(
            chunk_id=get_field("chunk_id", ""),
            text=get_field("text", ""),
            enriched_text=get_field("enriched_text", ""),
            article_number=get_field("article_number", ""),
            # Campos v3
            span_id=get_field("span_id", ""),
            parent_chunk_id=get_field("parent_chunk_id", ""),
            device_type=get_field("device_type", ""),
            document_id=get_field("document_id", ""),
            # Scores
            score=milvus_score,
            milvus_score=milvus_score,
            # Enriquecimento
            context_header=get_field("context_header", ""),
            thesis_text=get_field("thesis_text", ""),
            thesis_type=get_field("thesis_type", ""),
            synthetic_questions=get_field("synthetic_questions", ""),
            # Hierarquia
            document_type=tipo_doc,
            document_number=get_field("numero", ""),
            chapter_number=get_field("chapter_number", ""),
            chapter_title=get_field("chapter_title", ""),
            tipo_documento=tipo_doc,
        )


@dataclass
class SearchResult:
    """
    Resultado completo de uma busca.

    Attributes:
        query: Query original
        hits: Lista de resultados ordenados por relevancia
        total_found: Total de resultados encontrados (antes do limit)
        stage1_time_ms: Tempo da busca hibrida (ms)
        stage2_time_ms: Tempo do reranking (ms), se aplicavel

        # Configuracao usada
        top_k: Numero de resultados solicitados
        use_reranker: Se reranking foi aplicado
        weights: Pesos usados na busca hibrida
    """

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    total_found: int = 0

    # Timing
    stage1_time_ms: float = 0.0
    stage2_time_ms: float = 0.0

    # Config
    top_k: int = 10
    use_reranker: bool = True
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)

    @property
    def total_time_ms(self) -> float:
        """Tempo total da busca em ms."""
        return self.stage1_time_ms + self.stage2_time_ms

    @property
    def is_empty(self) -> bool:
        """Verifica se nao ha resultados."""
        return len(self.hits) == 0

    def top(self, n: int = 1) -> list[SearchHit]:
        """Retorna os top N resultados."""
        return self.hits[:n]

    def to_dict(self) -> dict[str, Any]:
        """Converte para dicionario."""
        return {
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "total_found": self.total_found,
            "total_time_ms": self.total_time_ms,
            "stage1_time_ms": self.stage1_time_ms,
            "stage2_time_ms": self.stage2_time_ms,
            "top_k": self.top_k,
            "use_reranker": self.use_reranker,
            "weights": self.weights,
        }


@dataclass
class SearchFilter:
    """
    Filtros para busca no Milvus.

    Permite filtrar por campos especificos antes da busca vetorial.

    Attributes:
        document_type: Filtrar por tipo (LEI, DECRETO, IN)
        document_number: Filtrar por numero do documento
        article_numbers: Filtrar por numeros de artigos
        thesis_types: Filtrar por tipos de tese
        year: Filtrar por ano
        chapter_numbers: Filtrar por capitulos
    """

    document_type: Optional[str] = None
    document_number: Optional[str] = None
    article_numbers: Optional[list[str]] = None
    thesis_types: Optional[list[str]] = None
    year: Optional[int] = None
    chapter_numbers: Optional[list[str]] = None

    def to_milvus_expr(self) -> str:
        """
        Converte filtros para expressao Milvus.

        Returns:
            String de expressao para o parametro 'expr' do Milvus
        """
        conditions = []

        if self.document_type:
            conditions.append(f'tipo_documento == "{self.document_type}"')

        if self.document_number:
            conditions.append(f'numero_documento == "{self.document_number}"')

        if self.article_numbers:
            articles = ", ".join(f'"{a}"' for a in self.article_numbers)
            conditions.append(f"article_number in [{articles}]")

        if self.thesis_types:
            types = ", ".join(f'"{t}"' for t in self.thesis_types)
            conditions.append(f"thesis_type in [{types}]")

        if self.year:
            conditions.append(f"ano == {self.year}")

        if self.chapter_numbers:
            chapters = ", ".join(f'"{c}"' for c in self.chapter_numbers)
            conditions.append(f"chapter_number in [{chapters}]")

        return " and ".join(conditions) if conditions else ""

    @property
    def is_empty(self) -> bool:
        """Verifica se nenhum filtro foi definido."""
        return not any([
            self.document_type,
            self.document_number,
            self.article_numbers,
            self.thesis_types,
            self.year,
            self.chapter_numbers,
        ])
