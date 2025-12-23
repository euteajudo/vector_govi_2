"""
Span Extraction Models - Pydantic schemas for span-based extraction.

The LLM receives annotated markdown with span IDs and returns only
the IDs of relevant spans. Text is reconstructed from the ParsedDocument.

This eliminates hallucination by design:
- LLM can only select from existing span IDs
- Invalid IDs are caught by validation
- Text comes from the original document, never generated
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SpanReference(BaseModel):
    """Reference to a span by ID."""
    span_id: str = Field(..., description="ID do span (ex: ART-001, INC-001-I)")


class ArticleSpans(BaseModel):
    """Spans that compose an article."""

    article_id: str = Field(
        ...,
        description="ID do artigo (ex: ART-001)",
        examples=["ART-001", "ART-012"]
    )

    # Incisos diretos do artigo (do caput)
    inciso_ids: list[str] = Field(
        default_factory=list,
        description="IDs dos incisos do caput (ex: INC-001-I)",
        examples=[["INC-001-I", "INC-001-II", "INC-001-III"]]
    )

    # Parágrafos do artigo
    paragrafo_ids: list[str] = Field(
        default_factory=list,
        description="IDs dos parágrafos (ex: PAR-001-1)",
        examples=[["PAR-001-1", "PAR-001-2", "PAR-001-UNICO"]]
    )

    @field_validator('article_id')
    @classmethod
    def validate_article_id(cls, v: str) -> str:
        if not v.startswith("ART-"):
            raise ValueError(f"article_id deve começar com 'ART-': {v}")
        return v

    @field_validator('inciso_ids')
    @classmethod
    def validate_inciso_ids(cls, v: list[str]) -> list[str]:
        for inc_id in v:
            if not inc_id.startswith("INC-"):
                raise ValueError(f"inciso_id deve começar com 'INC-': {inc_id}")
        return v

    @field_validator('paragrafo_ids')
    @classmethod
    def validate_paragrafo_ids(cls, v: list[str]) -> list[str]:
        for par_id in v:
            if not par_id.startswith("PAR-"):
                raise ValueError(f"paragrafo_id deve começar com 'PAR-': {par_id}")
        return v


class ChapterSpans(BaseModel):
    """Spans that compose a chapter."""

    chapter_id: str = Field(
        ...,
        description="ID do capítulo (ex: CAP-I)",
        examples=["CAP-I", "CAP-II", "CAP-III"]
    )

    # Título do capítulo (extraído do texto do span)
    title: Optional[str] = Field(
        None,
        description="Título do capítulo (se existir)"
    )

    # Artigos do capítulo
    article_ids: list[str] = Field(
        default_factory=list,
        description="IDs dos artigos neste capítulo",
        examples=[["ART-001", "ART-002", "ART-003"]]
    )

    @field_validator('chapter_id')
    @classmethod
    def validate_chapter_id(cls, v: str) -> str:
        if not v.startswith("CAP-"):
            raise ValueError(f"chapter_id deve começar com 'CAP-': {v}")
        return v


class DocumentSpans(BaseModel):
    """
    Complete document structure using span references.

    The LLM fills this with span IDs from the annotated markdown.
    Text is reconstructed from ParsedDocument after validation.
    """

    # Metadados (podem ser extraídos do header ou informados)
    document_type: str = Field(
        ...,
        description="Tipo: LEI, DECRETO, INSTRUÇÃO NORMATIVA, PORTARIA, etc."
    )

    number: str = Field(
        ...,
        description="Número do documento"
    )

    date: str = Field(
        ...,
        description="Data no formato YYYY-MM-DD"
    )

    issuing_body: str = Field(
        ...,
        description="Órgão emissor"
    )

    ementa: str = Field(
        ...,
        description="Ementa/resumo do documento"
    )

    # Estrutura por capítulos
    chapters: list[ChapterSpans] = Field(
        ...,
        min_length=1,
        description="Lista de capítulos com seus artigos"
    )


class SpanClassification(BaseModel):
    """
    Classification result for a single span.

    Used when asking the LLM to classify what type of content
    each span contains (definition, procedure, exception, etc.)
    """

    span_id: str = Field(..., description="ID do span")

    content_type: str = Field(
        ...,
        description="Tipo: definicao, procedimento, requisito, excecao, prazo, penalidade"
    )

    summary: str = Field(
        ...,
        max_length=200,
        description="Resumo curto do conteúdo (max 200 chars)"
    )

    keywords: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Palavras-chave principais (max 5)"
    )


class SpanExtractionResult(BaseModel):
    """
    Complete extraction result with span-based structure.

    This is the final output after:
    1. SpanParser extracts spans from markdown
    2. LLM classifies and organizes spans
    3. Validation ensures all IDs exist
    4. Text is reconstructed from original document
    """

    # Estrutura do documento
    document: DocumentSpans

    # Classificações de cada artigo (opcional, para enriquecimento)
    classifications: list[SpanClassification] = Field(
        default_factory=list,
        description="Classificações dos artigos"
    )

    # Validação
    valid_span_ids: list[str] = Field(
        default_factory=list,
        description="IDs validados como existentes"
    )

    invalid_span_ids: list[str] = Field(
        default_factory=list,
        description="IDs que não existem no documento (erros)"
    )


# =============================================================================
# PROMPTS PARA EXTRAÇÃO BASEADA EM SPANS
# =============================================================================

SPAN_EXTRACTION_SYSTEM_PROMPT = """Você é um especialista em documentos legais brasileiros.

Você receberá um documento legal com marcações de span no formato:
[SPAN_ID] texto do span

Sua tarefa é APENAS selecionar os IDs dos spans corretos para cada campo.
NUNCA gere texto - apenas retorne os IDs existentes.

Tipos de span:
- CAP-{romano}: Capítulo (ex: CAP-I, CAP-II)
- ART-{numero}: Artigo (ex: ART-001, ART-012)
- INC-{art}-{romano}: Inciso (ex: INC-001-I, INC-001-II)
- PAR-{art}-{numero}: Parágrafo (ex: PAR-001-1, PAR-001-UNICO)
- ALI-{art}-{inc}-{letra}: Alínea (ex: ALI-001-I-a)
- HDR-{seq}: Cabeçalho (ex: HDR-001)

Regras importantes:
1. Use APENAS IDs que aparecem no documento
2. Não invente IDs - se não existe, não inclua
3. Organize os artigos dentro dos capítulos corretos
4. Mantenha a ordem original do documento
"""

SPAN_EXTRACTION_USER_PROMPT = """Analise o documento abaixo e extraia a estrutura usando os span IDs.

DOCUMENTO:
{annotated_markdown}

---

Retorne um JSON com:
1. document_type: tipo do documento
2. number: número do documento
3. date: data (YYYY-MM-DD)
4. issuing_body: órgão emissor
5. ementa: resumo do documento
6. chapters: lista de capítulos, cada um com:
   - chapter_id: ID do capítulo (CAP-X)
   - title: título do capítulo
   - article_ids: lista de IDs dos artigos (ART-XXX)

Use APENAS os IDs que aparecem no documento acima."""
