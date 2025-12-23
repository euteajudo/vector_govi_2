"""
Teste do PageSpanExtractor - Extração de coordenadas PDF.

Testa:
1. Criação de BoundingBox
2. Mapeamento de spans para localizações
3. Merge de bounding boxes
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsing import SpanParser, SpanType
from parsing.page_spans import (
    PageSpanExtractor,
    BoundingBox,
    TextLocation,
    SpanLocation,
)


# Mock de ProvenanceItem do Docling
@dataclass
class MockBBox:
    l: float
    t: float
    r: float
    b: float
    coord_origin: str = "BOTTOMLEFT"


@dataclass
class MockProvenance:
    bbox: MockBBox
    page_no: int
    charspan: tuple[int, int] = (0, 0)


@dataclass
class MockTextItem:
    text: str
    prov: list[MockProvenance]


@dataclass
class MockDoclingDoc:
    texts: list[MockTextItem]


def test_bounding_box_creation():
    """Testa criação de BoundingBox."""
    print("=" * 60)
    print("TESTE: BoundingBox Creation")
    print("=" * 60)

    # 1. Criação direta
    bbox = BoundingBox(
        left=100.0,
        top=200.0,
        right=400.0,
        bottom=250.0,
        page=1,
    )

    print(f"\n[1] BoundingBox criada:")
    print(f"    left: {bbox.left}")
    print(f"    top: {bbox.top}")
    print(f"    right: {bbox.right}")
    print(f"    bottom: {bbox.bottom}")
    print(f"    width: {bbox.width}")
    print(f"    height: {bbox.height}")
    print(f"    center: ({bbox.center_x}, {bbox.center_y})")

    assert bbox.width == 300.0
    assert bbox.height == 50.0
    print("    [OK] Dimensões corretas")

    # 2. Criação a partir de mock Docling
    mock_prov = MockProvenance(
        bbox=MockBBox(l=50.0, t=100.0, r=200.0, b=150.0),
        page_no=2,
    )

    bbox2 = BoundingBox.from_docling(mock_prov, page_no=2)

    print(f"\n[2] BoundingBox from Docling:")
    print(f"    page: {bbox2.page}")
    print(f"    coords: l={bbox2.left}, t={bbox2.top}, r={bbox2.right}, b={bbox2.bottom}")

    assert bbox2.page == 2
    assert bbox2.left == 50.0
    print("    [OK] Conversão do Docling correta")

    # 3. to_dict
    bbox_dict = bbox2.to_dict()
    print(f"\n[3] to_dict:")
    print(f"    {bbox_dict}")

    assert "page" in bbox_dict
    assert "l" in bbox_dict
    print("    [OK] Serialização correta")

    return True


def test_page_span_extractor():
    """Testa extração de localizações do Docling mock."""
    print("\n" + "=" * 60)
    print("TESTE: PageSpanExtractor")
    print("=" * 60)

    # Cria mock de DoclingDocument
    mock_doc = MockDoclingDoc(texts=[
        MockTextItem(
            text="Art. 5º O estudo técnico preliminar deverá conter:",
            prov=[MockProvenance(
                bbox=MockBBox(l=100, t=200, r=500, b=220),
                page_no=2,
            )]
        ),
        MockTextItem(
            text="I - descrição da necessidade;",
            prov=[MockProvenance(
                bbox=MockBBox(l=120, t=240, r=480, b=260),
                page_no=2,
            )]
        ),
        MockTextItem(
            text="II - área requisitante e área técnica;",
            prov=[MockProvenance(
                bbox=MockBBox(l=120, t=280, r=480, b=300),
                page_no=2,
            )]
        ),
        MockTextItem(
            text="§ 1º A priorização será definida...",
            prov=[MockProvenance(
                bbox=MockBBox(l=100, t=400, r=500, b=420),
                page_no=3,
            )]
        ),
    ])

    print(f"\n[1] Mock Docling Document criado:")
    print(f"    Textos: {len(mock_doc.texts)}")

    # Extrai localizações
    extractor = PageSpanExtractor()
    text_locations = extractor.extract_from_docling(mock_doc)

    print(f"\n[2] Localizações extraídas: {len(text_locations)}")
    for loc in text_locations:
        print(f"    - Pág {loc.page}: '{loc.text[:40]}...'")
        print(f"      bbox: l={loc.bbox.left}, t={loc.bbox.top}")

    assert len(text_locations) == 4
    print("    [OK] Todas as localizações extraídas")

    return text_locations


def test_span_mapping():
    """Testa mapeamento de spans para localizações."""
    print("\n" + "=" * 60)
    print("TESTE: Mapeamento Spans -> Localizações")
    print("=" * 60)

    # Markdown de teste
    markdown = """## CAPÍTULO I - DO ESTUDO TÉCNICO

Art. 5º O estudo técnico preliminar deverá conter:

I - descrição da necessidade;

II - área requisitante e área técnica;

§ 1º A priorização será definida pelo órgão competente.
"""

    # Parseia markdown
    parser = SpanParser()
    parsed_doc = parser.parse(markdown)

    print(f"\n[1] Spans parseados: {len(parsed_doc.spans)}")
    for span in parsed_doc.spans:
        print(f"    - {span.span_id}: '{span.text[:40]}...'")

    # Cria localizações mock
    text_locations = [
        TextLocation(
            text="Art. 5º O estudo técnico preliminar deverá conter:",
            page=2,
            bbox=BoundingBox(left=100, top=200, right=500, bottom=220, page=2),
        ),
        TextLocation(
            text="I - descrição da necessidade;",
            page=2,
            bbox=BoundingBox(left=120, top=240, right=480, bottom=260, page=2),
        ),
        TextLocation(
            text="II - área requisitante e área técnica;",
            page=2,
            bbox=BoundingBox(left=120, top=280, right=480, bottom=300, page=2),
        ),
        TextLocation(
            text="§ 1º A priorização será definida pelo órgão competente.",
            page=3,
            bbox=BoundingBox(left=100, top=400, right=500, bottom=420, page=3),
        ),
    ]

    print(f"\n[2] Localizações mock: {len(text_locations)}")

    # Mapeia spans para localizações
    extractor = PageSpanExtractor(fuzzy_match_threshold=0.5)
    span_locations = extractor.map_spans_to_locations(parsed_doc, text_locations)

    print(f"\n[3] Spans mapeados: {len(span_locations)}")
    for span_id, loc in span_locations.items():
        print(f"    - {span_id} -> Pág {loc.page} (conf: {loc.confidence:.2f})")
        print(f"      bbox: l={loc.bbox.left:.0f}, t={loc.bbox.top:.0f}")

    # Verifica mapeamentos
    assert "ART-005" in span_locations, "ART-005 deveria estar mapeado"
    assert span_locations["ART-005"].page == 2
    print("\n    [OK] ART-005 mapeado para página 2")

    return span_locations


def test_merge_bboxes():
    """Testa merge de múltiplas bounding boxes."""
    print("\n" + "=" * 60)
    print("TESTE: Merge de BoundingBoxes")
    print("=" * 60)

    extractor = PageSpanExtractor()

    # Múltiplas localizações (span que atravessa linhas)
    locations = [
        SpanLocation(
            span_id="ART-005",
            page=2,
            bbox=BoundingBox(left=100, top=200, right=500, bottom=220, page=2),
        ),
        SpanLocation(
            span_id="ART-005",
            page=2,
            bbox=BoundingBox(left=100, top=230, right=480, bottom=250, page=2),
        ),
    ]

    merged = extractor.merge_bboxes(locations)

    print(f"\n[1] Merge de {len(locations)} bboxes:")
    print(f"    Resultado: l={merged.left}, t={merged.top}, r={merged.right}, b={merged.bottom}")

    assert merged.left == 100
    assert merged.top == 200
    assert merged.right == 500
    assert merged.bottom == 250
    print("    [OK] Merge correto (min/max)")

    return True


def test_integration_with_chunk_materializer():
    """Testa integração com ChunkMaterializer."""
    print("\n" + "=" * 60)
    print("TESTE: Integração com ChunkMaterializer")
    print("=" * 60)

    from chunking import ChunkMaterializer, ChunkMetadata

    # Cria metadados com page_spans
    page_spans = {
        "ART-005": {
            "page": 2,
            "l": 100.0,
            "t": 200.0,
            "r": 500.0,
            "b": 220.0,
        },
        "PAR-005-1": {
            "page": 3,
            "l": 100.0,
            "t": 400.0,
            "r": 500.0,
            "b": 420.0,
        },
    }

    metadata = ChunkMetadata(
        schema_version="1.0.0",
        extractor_version="1.0.0",
        document_hash="abc123",
        page_spans=page_spans,
    )

    print(f"\n[1] ChunkMetadata com page_spans:")
    print(f"    page_spans: {len(metadata.page_spans)} spans")

    # Converte para dict
    meta_dict = metadata.to_dict()
    print(f"\n[2] to_dict():")
    print(f"    page_spans presente: {'page_spans' in meta_dict}")
    print(f"    ART-005 página: {meta_dict['page_spans'].get('ART-005', {}).get('page')}")

    assert "page_spans" in meta_dict
    assert meta_dict["page_spans"]["ART-005"]["page"] == 2
    print("    [OK] page_spans serializado corretamente")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTES DO PAGE SPAN EXTRACTOR")
    print("=" * 70)

    test_bounding_box_creation()
    test_page_span_extractor()
    test_span_mapping()
    test_merge_bboxes()
    test_integration_with_chunk_materializer()

    print("\n" + "=" * 70)
    print("TODOS OS TESTES PASSARAM!")
    print("=" * 70)
