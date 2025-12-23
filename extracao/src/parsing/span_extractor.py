"""
SpanExtractor - LLM-based extraction using span references.

This extractor eliminates hallucination by design:
1. SpanParser extracts spans deterministically using regex
2. LLM receives annotated markdown with span IDs
3. LLM can only select from existing span IDs
4. Invalid IDs are caught by validation
5. Text is reconstructed from original document

Usage:
    from parsing import SpanExtractor

    extractor = SpanExtractor(llm_client)
    result = extractor.extract(markdown_text)

    # Access validated structure
    for chapter in result.document.chapters:
        print(f"Chapter: {chapter.chapter_id}")
        for art_id in chapter.article_ids:
            article = result.get_article_text(art_id)
            print(f"  {art_id}: {article[:50]}...")
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Any

from .span_parser import SpanParser, ParserConfig
from .span_models import ParsedDocument, Span, SpanType
from .span_extraction_models import (
    DocumentSpans,
    ChapterSpans,
    SpanExtractionResult,
    SPAN_EXTRACTION_SYSTEM_PROMPT,
    SPAN_EXTRACTION_USER_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class SpanExtractorConfig:
    """Configuration for SpanExtractor."""

    # Parser config
    parser_config: ParserConfig = field(default_factory=ParserConfig)

    # LLM config
    model: str = "Qwen/Qwen3-8B-AWQ"
    temperature: float = 0.0
    max_tokens: int = 4096

    # Validation
    strict_validation: bool = True  # Fail if any invalid span ID
    auto_fix_ids: bool = True  # Try to fix common ID errors


@dataclass
class ExtractionResult:
    """Result of span-based extraction."""

    # Parsed document with all spans
    parsed_doc: ParsedDocument

    # LLM response (validated)
    document: DocumentSpans

    # Validation results
    valid_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    fixed_ids: dict[str, str] = field(default_factory=dict)  # old -> new

    # Raw LLM response for debugging
    raw_response: Optional[str] = None

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get span by ID."""
        return self.parsed_doc.get_span(span_id)

    def get_article_text(self, article_id: str) -> str:
        """Get full article text including children."""
        span = self.parsed_doc.get_span(article_id)
        if not span:
            return ""

        # Get article text
        texts = [span.text]

        # Add children (incisos, paragrafos)
        for child in self.parsed_doc.get_children(article_id):
            texts.append(f"  {child.text}")

            # Add grandchildren (alineas)
            for grandchild in self.parsed_doc.get_children(child.span_id):
                texts.append(f"    {grandchild.text}")

        return "\n".join(texts)

    def get_chapter_text(self, chapter_id: str) -> str:
        """Get full chapter text including all articles."""
        span = self.parsed_doc.get_span(chapter_id)
        if not span:
            return ""

        texts = [span.text]

        # Find articles in this chapter
        for chapter in self.document.chapters:
            if chapter.chapter_id == chapter_id:
                for art_id in chapter.article_ids:
                    texts.append(self.get_article_text(art_id))
                break

        return "\n\n".join(texts)

    @property
    def is_valid(self) -> bool:
        """Check if extraction is valid (no invalid IDs)."""
        return len(self.invalid_ids) == 0


class SpanExtractor:
    """
    Extractor that uses spans to eliminate LLM hallucination.

    The LLM can only select from existing span IDs, never generate text.
    This makes the extraction deterministic and verifiable.
    """

    def __init__(
        self,
        llm_client: Any,  # VLLMClient or compatible
        config: Optional[SpanExtractorConfig] = None
    ):
        """
        Initialize SpanExtractor.

        Args:
            llm_client: LLM client with chat() or chat_with_schema() method
            config: Extractor configuration
        """
        self.llm = llm_client
        self.config = config or SpanExtractorConfig()
        self.parser = SpanParser(self.config.parser_config)

    def extract(self, markdown: str) -> ExtractionResult:
        """
        Extract document structure using span-based approach.

        Args:
            markdown: Document markdown (from Docling)

        Returns:
            ExtractionResult with validated structure
        """
        # 1. Parse markdown to get spans
        logger.info("Parsing markdown to extract spans...")
        parsed_doc = self.parser.parse(markdown)
        logger.info(
            f"Extracted {len(parsed_doc.spans)} spans: "
            f"{len(parsed_doc.articles)} articles, "
            f"{len(parsed_doc.capitulos)} chapters"
        )

        # 2. Generate annotated markdown for LLM
        annotated = parsed_doc.to_annotated_markdown()

        # 3. Call LLM to classify spans
        logger.info("Calling LLM to classify spans...")
        raw_response = self._call_llm(annotated, parsed_doc)

        # 4. Parse and validate response
        logger.info("Validating LLM response...")
        document, valid_ids, invalid_ids, fixed_ids = self._validate_response(
            raw_response, parsed_doc
        )

        if invalid_ids:
            logger.warning(f"Found {len(invalid_ids)} invalid span IDs: {invalid_ids}")

        if fixed_ids:
            logger.info(f"Auto-fixed {len(fixed_ids)} span IDs: {fixed_ids}")

        return ExtractionResult(
            parsed_doc=parsed_doc,
            document=document,
            valid_ids=valid_ids,
            invalid_ids=invalid_ids,
            fixed_ids=fixed_ids,
            raw_response=raw_response,
        )

    def _call_llm(self, annotated_markdown: str, parsed_doc: ParsedDocument) -> str:
        """Call LLM to classify spans."""
        # Build prompt
        user_prompt = SPAN_EXTRACTION_USER_PROMPT.format(
            annotated_markdown=annotated_markdown
        )

        messages = [
            {"role": "system", "content": SPAN_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Try guided JSON if available
        if hasattr(self.llm, 'chat_with_schema'):
            response = self.llm.chat_with_schema(
                messages=messages,
                schema=DocumentSpans,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            # Fallback to regular chat
            response = self.llm.chat(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        return response

    def _validate_response(
        self,
        response: str,
        parsed_doc: ParsedDocument
    ) -> tuple[DocumentSpans, list[str], list[str], dict[str, str]]:
        """
        Validate LLM response and fix common errors.

        Returns:
            (document, valid_ids, invalid_ids, fixed_ids)
        """
        # Parse JSON response
        try:
            if isinstance(response, str):
                # Handle /think tags from Qwen3
                response = self._clean_response(response)
                data = json.loads(response)
            else:
                data = response
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # Return minimal valid structure
            return self._create_fallback_document(parsed_doc), [], [], {}

        # Track validation
        valid_ids = []
        invalid_ids = []
        fixed_ids = {}

        # Validate and fix chapter IDs
        if "chapters" in data:
            for chapter in data["chapters"]:
                # Validate chapter_id
                chapter_id = chapter.get("chapter_id", "")
                validated_id = self._validate_span_id(
                    chapter_id, "CAP", parsed_doc, fixed_ids
                )
                if validated_id:
                    chapter["chapter_id"] = validated_id
                    valid_ids.append(validated_id)
                else:
                    invalid_ids.append(chapter_id)

                # Validate article_ids
                validated_articles = []
                for art_id in chapter.get("article_ids", []):
                    validated_id = self._validate_span_id(
                        art_id, "ART", parsed_doc, fixed_ids
                    )
                    if validated_id:
                        validated_articles.append(validated_id)
                        valid_ids.append(validated_id)
                    else:
                        invalid_ids.append(art_id)

                chapter["article_ids"] = validated_articles

        # Create validated document
        try:
            document = DocumentSpans(**data)
        except Exception as e:
            logger.error(f"Failed to create DocumentSpans: {e}")
            document = self._create_fallback_document(parsed_doc)

        return document, valid_ids, invalid_ids, fixed_ids

    def _validate_span_id(
        self,
        span_id: str,
        expected_prefix: str,
        parsed_doc: ParsedDocument,
        fixed_ids: dict[str, str]
    ) -> Optional[str]:
        """
        Validate a span ID exists, with auto-fix for common errors.

        Returns:
            Validated ID or None if invalid
        """
        if not span_id:
            return None

        # Check if exists
        if parsed_doc.get_span(span_id):
            return span_id

        # Try auto-fix if enabled
        if not self.config.auto_fix_ids:
            return None

        # Common fixes
        original_id = span_id

        # Fix: ART-1 -> ART-001
        if re.match(rf'^{expected_prefix}-(\d+)$', span_id):
            num = re.search(r'\d+', span_id).group()
            fixed = f"{expected_prefix}-{num.zfill(3)}"
            if parsed_doc.get_span(fixed):
                fixed_ids[original_id] = fixed
                return fixed

        # Fix: ART-01 -> ART-001
        if re.match(rf'^{expected_prefix}-(\d{{2}})$', span_id):
            num = re.search(r'\d+', span_id).group()
            fixed = f"{expected_prefix}-{num.zfill(3)}"
            if parsed_doc.get_span(fixed):
                fixed_ids[original_id] = fixed
                return fixed

        # Fix: CAP-1 -> CAP-I (roman numeral)
        if expected_prefix == "CAP" and re.match(r'^CAP-\d+$', span_id):
            num = int(re.search(r'\d+', span_id).group())
            roman = self._int_to_roman(num)
            fixed = f"CAP-{roman}"
            if parsed_doc.get_span(fixed):
                fixed_ids[original_id] = fixed
                return fixed

        return None

    def _clean_response(self, response: str) -> str:
        """Clean LLM response, removing think tags etc."""
        # Remove <think>...</think> tags (Qwen3)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # Find JSON object
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            return response[start:end]

        return response

    def _create_fallback_document(self, parsed_doc: ParsedDocument) -> DocumentSpans:
        """Create fallback document from parsed spans when LLM fails."""
        # Extract metadata from header
        doc_type = parsed_doc.metadata.get("document_type", "INSTRUÇÃO NORMATIVA")
        number = parsed_doc.metadata.get("number", "")
        date = parsed_doc.metadata.get("date_raw", "")

        # Build chapters from spans
        chapters = []
        current_chapter = None

        for span in parsed_doc.spans:
            if span.span_type == SpanType.CAPITULO:
                if current_chapter:
                    chapters.append(current_chapter)
                current_chapter = ChapterSpans(
                    chapter_id=span.span_id,
                    title=span.text.split('\n')[0] if '\n' in span.text else span.text,
                    article_ids=[]
                )
            elif span.span_type == SpanType.ARTIGO:
                if current_chapter:
                    current_chapter.article_ids.append(span.span_id)
                else:
                    # Article before first chapter - create default chapter
                    current_chapter = ChapterSpans(
                        chapter_id="CAP-DEFAULT",
                        title="Disposições Gerais",
                        article_ids=[span.span_id]
                    )

        if current_chapter:
            chapters.append(current_chapter)

        # If no chapters found, create one with all articles
        if not chapters:
            chapters = [ChapterSpans(
                chapter_id="CAP-I",
                title="Disposições Gerais",
                article_ids=[a.span_id for a in parsed_doc.articles]
            )]

        return DocumentSpans(
            document_type=doc_type,
            number=number,
            date=date,
            issuing_body="",
            ementa="",
            chapters=chapters
        )

    def _int_to_roman(self, num: int) -> str:
        """Convert integer to roman numeral."""
        val = [100, 90, 50, 40, 10, 9, 5, 4, 1]
        syms = ['C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        result = ''
        for i, v in enumerate(val):
            while num >= v:
                result += syms[i]
                num -= v
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_with_spans(
    markdown: str,
    llm_client: Any,
    config: Optional[SpanExtractorConfig] = None
) -> ExtractionResult:
    """
    Convenience function for span-based extraction.

    Args:
        markdown: Document markdown
        llm_client: LLM client
        config: Optional configuration

    Returns:
        ExtractionResult
    """
    extractor = SpanExtractor(llm_client, config)
    return extractor.extract(markdown)
