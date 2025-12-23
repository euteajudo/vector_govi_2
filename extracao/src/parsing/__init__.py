"""
Parsing module - Regex-first extraction for Brazilian legal documents.

This module provides deterministic parsing of legal document structure
using regex patterns, eliminating LLM hallucination by design.

Usage:
    from parsing import SpanParser, SpanType

    # Parse markdown to spans
    parser = SpanParser()
    result = parser.parse(markdown_text)

    for span in result.spans:
        print(f"{span.span_id}: {span.text[:50]}...")

    # Extract with LLM (span-based, no hallucination)
    from parsing import SpanExtractor

    extractor = SpanExtractor(llm_client)
    result = extractor.extract(markdown_text)
"""

from .span_models import (
    SpanType,
    Span,
    ParsedDocument,
)
from .span_parser import SpanParser, ParserConfig
from .span_extraction_models import (
    DocumentSpans,
    ChapterSpans,
    ArticleSpans,
    SpanClassification,
    SpanExtractionResult,
)
from .span_extractor import (
    SpanExtractor,
    SpanExtractorConfig,
    ExtractionResult,
    extract_with_spans,
)
from .article_orchestrator import (
    ArticleOrchestrator,
    OrchestratorConfig,
    ArticleChunk,
    ArticleExtractionResult,
    ValidationStatus,
    extract_articles_with_hierarchy,
)
from .page_spans import (
    PageSpanExtractor,
    BoundingBox,
    TextLocation,
    SpanLocation,
    extract_page_spans_from_pdf,
)

__all__ = [
    # Core models
    "SpanType",
    "Span",
    "ParsedDocument",
    # Parser
    "SpanParser",
    "ParserConfig",
    # Extraction models
    "DocumentSpans",
    "ChapterSpans",
    "ArticleSpans",
    "SpanClassification",
    "SpanExtractionResult",
    # Extractor (Fase 2)
    "SpanExtractor",
    "SpanExtractorConfig",
    "ExtractionResult",
    "extract_with_spans",
    # Orchestrator (Fase 3)
    "ArticleOrchestrator",
    "OrchestratorConfig",
    "ArticleChunk",
    "ArticleExtractionResult",
    "ValidationStatus",
    "extract_articles_with_hierarchy",
    # Page spans (Fase 4)
    "PageSpanExtractor",
    "BoundingBox",
    "TextLocation",
    "SpanLocation",
    "extract_page_spans_from_pdf",
]
