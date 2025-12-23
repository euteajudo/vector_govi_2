"""
Span Models - Data structures for parsed legal document spans.

Each span represents an atomic unit of the document that can be
referenced by ID. The LLM will only select span IDs, never generate text.

Span ID Format:
    - Artigo:    ART-{numero}           (ex: ART-001, ART-012)
    - Paragrafo: PAR-{art}-{id}         (ex: PAR-001-1, PAR-001-UNICO)
    - Inciso:    INC-{art}-{romano}     (ex: INC-001-I, INC-001-IV)
    - Alinea:    ALI-{art}-{inc}-{let}  (ex: ALI-001-I-a, ALI-001-II-b)
    - Cabecalho: HDR-{seq}              (ex: HDR-001, HDR-002)
    - Capitulo:  CAP-{romano}           (ex: CAP-I, CAP-II)

Note: Incisos dentro de parágrafos mantêm o mesmo formato INC-{art}-{romano}.
      O vínculo ao parágrafo fica em parent_id (ex: parent_id="PAR-001-2").
      Quando há conflito (ex: inciso I no caput e no §2º), adiciona-se sufixo:
      INC-001-I (primeiro), INC-001-I_2 (segundo com mesmo romano).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SpanType(str, Enum):
    """Tipos de spans em documentos legais brasileiros."""

    # Estrutura principal
    HEADER = "header"           # Cabeçalho do documento (ementa, órgão, etc.)
    CAPITULO = "capitulo"       # CAPÍTULO I, II, III...
    SECAO = "secao"             # Seção I, II, III...
    SUBSECAO = "subsecao"       # Subseção

    # Artigos e subdivisões
    ARTIGO = "artigo"           # Art. 1º, Art. 2º...
    PARAGRAFO = "paragrafo"     # § 1º, § 2º, Parágrafo único
    INCISO = "inciso"           # I, II, III, IV...
    ALINEA = "alinea"           # a), b), c)...
    ITEM = "item"               # 1), 2), 3)... (raro, mas existe)

    # Outros
    TITULO = "titulo"           # Título de artigo/seção
    TEXTO = "texto"             # Texto livre (entre estruturas)
    ASSINATURA = "assinatura"   # Assinatura do documento


@dataclass
class Span:
    """
    Representa um trecho atômico do documento.

    Attributes:
        span_id: Identificador único (ex: ART-001, INC-001-I)
        span_type: Tipo do span (artigo, inciso, etc.)
        text: Texto original do span
        identifier: Identificador legal (ex: "1º", "I", "a")
        parent_id: ID do span pai (ex: inciso aponta para artigo)
        start_pos: Posição inicial no texto original
        end_pos: Posição final no texto original
        metadata: Dados extras (título, contexto, etc.)
    """

    span_id: str
    span_type: SpanType
    text: str
    identifier: Optional[str] = None
    parent_id: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    order: int = 0  # Ordem de inserção (para ordenação estável)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Normaliza o texto."""
        self.text = self.text.strip()

    @property
    def is_article(self) -> bool:
        return self.span_type == SpanType.ARTIGO

    @property
    def is_paragraph(self) -> bool:
        return self.span_type == SpanType.PARAGRAFO

    @property
    def is_inciso(self) -> bool:
        return self.span_type == SpanType.INCISO

    @property
    def is_alinea(self) -> bool:
        return self.span_type == SpanType.ALINEA

    @property
    def article_number(self) -> Optional[str]:
        """Extrai número do artigo do span_id."""
        if self.span_id.startswith("ART-"):
            return self.span_id.split("-")[1]
        elif "-" in self.span_id:
            # PAR-001-1, INC-001-I, ALI-001-I-a
            parts = self.span_id.split("-")
            if len(parts) >= 2:
                return parts[1]
        return None

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "span_id": self.span_id,
            "span_type": self.span_type.value,
            "text": self.text,
            "identifier": self.identifier,
            "parent_id": self.parent_id,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "order": self.order,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Span({self.span_id}, {self.span_type.value}, '{text_preview}')"


@dataclass
class ParsedDocument:
    """
    Documento parseado com todos os spans identificados.

    Attributes:
        spans: Lista de todos os spans do documento
        span_index: Índice span_id -> Span para lookup rápido
        source_text: Texto original do documento
        metadata: Metadados do documento (tipo, número, data, etc.)
    """

    spans: list[Span] = field(default_factory=list)
    source_text: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Constrói índice de spans."""
        self._span_index: dict[str, Span] = {}
        self._rebuild_index()

    def _rebuild_index(self):
        """Reconstrói o índice de spans."""
        self._span_index = {span.span_id: span for span in self.spans}

    def add_span(self, span: Span):
        """Adiciona um span ao documento."""
        span.order = len(self.spans)  # Define ordem de inserção
        self.spans.append(span)
        self._span_index[span.span_id] = span

    def get_span(self, span_id: str) -> Optional[Span]:
        """Busca span por ID."""
        return self._span_index.get(span_id)

    def get_spans_by_type(self, span_type: SpanType) -> list[Span]:
        """Retorna todos os spans de um tipo, ordenados por ordem de inserção."""
        spans = [s for s in self.spans if s.span_type == span_type]
        return sorted(spans, key=lambda s: s.order)

    def get_children(self, parent_id: str) -> list[Span]:
        """Retorna filhos de um span, ordenados por ordem no documento."""
        children = [s for s in self.spans if s.parent_id == parent_id]
        return sorted(children, key=lambda s: s.order)

    def get_article_spans(self, article_number: str) -> list[Span]:
        """Retorna todos os spans de um artigo (incluindo filhos)."""
        art_id = f"ART-{article_number.zfill(3)}"
        result = []

        # Artigo principal
        if art_id in self._span_index:
            result.append(self._span_index[art_id])

        # Filhos diretos e indiretos
        for span in self.spans:
            if span.parent_id == art_id:
                result.append(span)
            elif span.parent_id and span.parent_id.startswith(f"INC-{article_number.zfill(3)}"):
                result.append(span)
            elif span.parent_id and span.parent_id.startswith(f"PAR-{article_number.zfill(3)}"):
                result.append(span)

        return result

    def reconstruct_text(self, span_ids: list[str]) -> str:
        """
        Reconstrói texto a partir de lista de span_ids.

        Esta é a função chave: o LLM retorna IDs, e o código
        reconstrói o texto de forma determinística.
        """
        texts = []
        for span_id in span_ids:
            span = self.get_span(span_id)
            if span:
                texts.append(span.text)
        return "\n".join(texts)

    def validate_span_ids(self, span_ids: list[str]) -> tuple[bool, list[str]]:
        """
        Valida se todos os span_ids existem.

        Returns:
            (válido, lista de IDs inválidos)
        """
        invalid = [sid for sid in span_ids if sid not in self._span_index]
        return len(invalid) == 0, invalid

    @property
    def articles(self) -> list[Span]:
        """Retorna todos os artigos."""
        return self.get_spans_by_type(SpanType.ARTIGO)

    @property
    def capitulos(self) -> list[Span]:
        """Retorna todos os capítulos."""
        return self.get_spans_by_type(SpanType.CAPITULO)

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "metadata": self.metadata,
            "span_count": len(self.spans),
            "article_count": len(self.articles),
            "spans": [s.to_dict() for s in self.spans],
        }

    def to_annotated_markdown(self) -> str:
        """
        Gera markdown anotado com span_ids.

        Este é o formato que será enviado ao LLM.
        """
        lines = []
        for span in self.spans:
            lines.append(f"[{span.span_id}] {span.text}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ParsedDocument("
            f"spans={len(self.spans)}, "
            f"articles={len(self.articles)}, "
            f"capitulos={len(self.capitulos)})"
        )
