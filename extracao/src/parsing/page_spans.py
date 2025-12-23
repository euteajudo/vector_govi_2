"""
Page Spans - Extração de coordenadas PDF para citações visuais.

Este módulo extrai bounding boxes do Docling e mapeia para os spans
do SpanParser, permitindo navegação visual no PDF original.

Estrutura de PageSpan:
{
    "span_id": "ART-005",
    "page": 2,
    "bbox": {
        "l": 108.0,   # left
        "t": 405.0,   # top
        "r": 504.0,   # right
        "b": 330.0    # bottom
    },
    "coord_origin": "TOPLEFT"
}

Uso:
    from docling.document_converter import DocumentConverter
    from parsing import SpanParser, PageSpanExtractor

    # Converte PDF
    converter = DocumentConverter()
    result = converter.convert("documento.pdf")

    # Extrai page spans
    extractor = PageSpanExtractor()
    page_spans = extractor.extract_from_docling(result.document)

    # Parseia markdown
    parser = SpanParser()
    parsed_doc = parser.parse(result.document.export_to_markdown())

    # Mapeia spans para coordenadas
    span_locations = extractor.map_spans_to_locations(parsed_doc, page_spans)
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box com coordenadas normalizadas."""

    left: float
    top: float
    right: float
    bottom: float
    page: int
    coord_origin: str = "TOPLEFT"

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return abs(self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "l": round(self.left, 2),
            "t": round(self.top, 2),
            "r": round(self.right, 2),
            "b": round(self.bottom, 2),
            "coord_origin": self.coord_origin,
        }

    @classmethod
    def from_docling(cls, prov: Any, page_no: int) -> Optional["BoundingBox"]:
        """Cria BoundingBox a partir de ProvenanceItem do Docling."""
        try:
            if not hasattr(prov, 'bbox') or prov.bbox is None:
                return None

            bbox = prov.bbox

            # Docling usa BOTTOMLEFT, convertemos para TOPLEFT
            # Precisamos da altura da página para converter
            return cls(
                left=bbox.l,
                top=bbox.t,
                right=bbox.r,
                bottom=bbox.b,
                page=page_no,
                coord_origin=str(bbox.coord_origin) if hasattr(bbox, 'coord_origin') else "BOTTOMLEFT",
            )
        except Exception as e:
            logger.warning(f"Erro ao criar BoundingBox: {e}")
            return None


@dataclass
class TextLocation:
    """Localização de um trecho de texto no PDF."""

    text: str
    page: int
    bbox: BoundingBox
    char_start: int = 0
    char_end: int = 0


@dataclass
class SpanLocation:
    """Localização de um span no PDF."""

    span_id: str
    page: int
    bbox: BoundingBox
    confidence: float = 1.0  # Confiança do mapeamento

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "page": self.page,
            "bbox": self.bbox.to_dict(),
            "confidence": round(self.confidence, 3),
        }


class PageSpanExtractor:
    """
    Extrai coordenadas de página do Docling e mapeia para spans.

    O Docling fornece:
    - `doc.texts`: lista de TextItems com `text` e `prov`
    - `prov`: lista de ProvenanceItem com `bbox` e `page_no`

    Este extrator:
    1. Coleta todos os TextLocations do Docling
    2. Para cada span do SpanParser, encontra a localização correspondente
    """

    def __init__(self, fuzzy_match_threshold: float = 0.8):
        """
        Args:
            fuzzy_match_threshold: Threshold para matching fuzzy de texto
        """
        self.fuzzy_match_threshold = fuzzy_match_threshold

    def extract_from_docling(self, docling_doc: Any) -> list[TextLocation]:
        """
        Extrai localizações de texto do DoclingDocument.

        Args:
            docling_doc: DoclingDocument do Docling

        Returns:
            Lista de TextLocation com texto e coordenadas
        """
        locations = []

        try:
            # Itera sobre todos os textos do documento
            if hasattr(docling_doc, 'texts'):
                for text_item in docling_doc.texts:
                    if not hasattr(text_item, 'text') or not text_item.text:
                        continue

                    # Cada texto pode ter múltiplas proveniências (páginas)
                    if hasattr(text_item, 'prov') and text_item.prov:
                        for prov in text_item.prov:
                            page_no = getattr(prov, 'page_no', 1)
                            bbox = BoundingBox.from_docling(prov, page_no)

                            if bbox:
                                char_start = prov.charspan[0] if hasattr(prov, 'charspan') else 0
                                char_end = prov.charspan[1] if hasattr(prov, 'charspan') else len(text_item.text)

                                locations.append(TextLocation(
                                    text=text_item.text,
                                    page=page_no,
                                    bbox=bbox,
                                    char_start=char_start,
                                    char_end=char_end,
                                ))

            logger.info(f"Extraídas {len(locations)} localizações de texto do Docling")

        except Exception as e:
            logger.error(f"Erro ao extrair localizações do Docling: {e}")

        return locations

    def map_spans_to_locations(
        self,
        parsed_doc: Any,  # ParsedDocument
        text_locations: list[TextLocation]
    ) -> dict[str, SpanLocation]:
        """
        Mapeia spans do ParsedDocument para suas localizações no PDF.

        Estratégia de matching:
        1. Exact match: texto do span == texto da localização
        2. Contains match: localização contém o início do texto do span
        3. Fuzzy match: similaridade > threshold

        Args:
            parsed_doc: ParsedDocument do SpanParser
            text_locations: Localizações extraídas do Docling

        Returns:
            Dict de span_id -> SpanLocation
        """
        span_locations = {}

        for span in parsed_doc.spans:
            location = self._find_location_for_span(span, text_locations)
            if location:
                span_locations[span.span_id] = location

        logger.info(
            f"Mapeados {len(span_locations)}/{len(parsed_doc.spans)} spans para localizações"
        )

        return span_locations

    def _find_location_for_span(
        self,
        span: Any,  # Span
        text_locations: list[TextLocation]
    ) -> Optional[SpanLocation]:
        """Encontra a melhor localização para um span."""

        # Normaliza texto do span para comparação
        span_text = self._normalize_text(span.text)
        span_start = span_text[:50]  # Primeiros 50 chars para matching

        best_match = None
        best_confidence = 0.0

        for loc in text_locations:
            loc_text = self._normalize_text(loc.text)

            # 1. Exact match
            if loc_text == span_text:
                return SpanLocation(
                    span_id=span.span_id,
                    page=loc.page,
                    bbox=loc.bbox,
                    confidence=1.0,
                )

            # 2. Contains match (localização contém o início do span)
            if span_start and span_start in loc_text:
                confidence = len(span_start) / max(len(span_text), len(loc_text))
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = loc

            # 3. Span começa com texto da localização
            if loc_text and span_text.startswith(loc_text[:30]):
                confidence = 0.9
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = loc

        # Retorna melhor match se acima do threshold
        if best_match and best_confidence >= self.fuzzy_match_threshold:
            return SpanLocation(
                span_id=span.span_id,
                page=best_match.page,
                bbox=best_match.bbox,
                confidence=best_confidence,
            )

        return None

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparação."""
        if not text:
            return ""

        # Remove espaços extras e normaliza
        text = " ".join(text.split())
        # Remove caracteres especiais mas mantém acentos
        text = text.strip().lower()

        return text

    def merge_bboxes(self, locations: list[SpanLocation]) -> Optional[BoundingBox]:
        """
        Merge múltiplas bounding boxes em uma só.

        Útil para spans que atravessam múltiplas linhas.
        """
        if not locations:
            return None

        if len(locations) == 1:
            return locations[0].bbox

        # Pega a página mais comum
        pages = [loc.page for loc in locations]
        main_page = max(set(pages), key=pages.count)

        # Filtra para mesma página
        same_page = [loc for loc in locations if loc.page == main_page]

        if not same_page:
            return locations[0].bbox

        # Merge das bboxes
        left = min(loc.bbox.left for loc in same_page)
        top = min(loc.bbox.top for loc in same_page)
        right = max(loc.bbox.right for loc in same_page)
        bottom = max(loc.bbox.bottom for loc in same_page)

        return BoundingBox(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            page=main_page,
            coord_origin=same_page[0].bbox.coord_origin,
        )


def extract_page_spans_from_pdf(
    pdf_path: str,
    markdown: Optional[str] = None
) -> tuple[str, dict[str, SpanLocation]]:
    """
    Função de conveniência para extrair page spans de um PDF.

    Args:
        pdf_path: Caminho para o PDF
        markdown: Markdown pré-processado (opcional)

    Returns:
        Tuple de (markdown, span_locations)
    """
    from docling.document_converter import DocumentConverter
    from .span_parser import SpanParser

    # Converte PDF
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    # Exporta markdown se não fornecido
    if markdown is None:
        markdown = result.document.export_to_markdown()

    # Extrai localizações
    extractor = PageSpanExtractor()
    text_locations = extractor.extract_from_docling(result.document)

    # Parseia spans
    parser = SpanParser()
    parsed_doc = parser.parse(markdown)

    # Mapeia
    span_locations = extractor.map_spans_to_locations(parsed_doc, text_locations)

    return markdown, span_locations
