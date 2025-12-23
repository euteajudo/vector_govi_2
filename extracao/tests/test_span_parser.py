"""Teste do SpanParser com IN 65/2021."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docling.document_converter import DocumentConverter
from parsing import SpanParser, SpanType


def test_span_parser():
    """Testa SpanParser com IN 65/2021."""
    pdf_path = Path(__file__).parent.parent / "data" / "INSTRUÇÃO NORMATIVA SEGES _ME Nº 65, DE 7 DE JULHO DE 2021.pdf"

    print("=" * 60)
    print("TESTE DO SPAN PARSER")
    print("=" * 60)

    # 1. Extrai markdown com Docling
    print("\n[1] Extraindo markdown com Docling...")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    print(f"    Markdown: {len(markdown)} caracteres")

    # Salva markdown para debug
    output_path = Path(__file__).parent.parent / "data" / "output" / "in_65_markdown.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(f"    Salvo em: {output_path}")

    # 2. Parseia com SpanParser
    print("\n[2] Parseando com SpanParser...")
    parser = SpanParser()
    doc = parser.parse(markdown)

    # 3. Estatísticas
    print("\n[3] Estatísticas:")
    print(f"    Total spans: {len(doc.spans)}")
    print(f"    Capítulos:   {len(doc.capitulos)}")
    print(f"    Artigos:     {len(doc.articles)}")

    # Conta por tipo
    by_type = {}
    for span in doc.spans:
        by_type[span.span_type] = by_type.get(span.span_type, 0) + 1

    print("\n    Por tipo:")
    for stype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"      {stype.value:12}: {count}")

    # 4. Mostra artigos e seus filhos
    print("\n[4] Artigos e subdivisões:")
    for art in doc.articles[:5]:  # Primeiros 5 artigos
        children = doc.get_children(art.span_id)
        incisos = [c for c in children if c.span_type == SpanType.INCISO]
        paragrafos = [c for c in children if c.span_type == SpanType.PARAGRAFO]

        print(f"\n    {art.span_id}: {art.text[:60]}...")
        print(f"      Incisos: {len(incisos)}, Parágrafos: {len(paragrafos)}")

        for inc in incisos[:3]:  # Primeiros 3 incisos
            print(f"        {inc.span_id}: {inc.text[:50]}...")

    # 5. Mostra um trecho do markdown anotado
    print("\n[5] Markdown anotado (primeiras 20 linhas):")
    annotated = doc.to_annotated_markdown()
    for line in annotated.split('\n')[:20]:
        print(f"    {line[:80]}")

    # 6. Validação de incisos esperados
    print("\n[6] Validação de incisos:")
    art_5 = doc.get_span("ART-005")
    if art_5:
        children = doc.get_children("ART-005")
        incisos = [c for c in children if c.span_type == SpanType.INCISO]
        print(f"    Art. 5 tem {len(incisos)} incisos")
        for inc in incisos:
            print(f"      {inc.span_id}: {inc.text[:60]}...")
    else:
        print("    Art. 5 não encontrado!")

    # 7. Validação de alíneas
    print("\n[7] Validação de alíneas:")
    alineas = doc.get_spans_by_type(SpanType.ALINEA)
    print(f"    Total alíneas: {len(alineas)}")
    for ali in alineas[:10]:
        print(f"      {ali.span_id}: {ali.text[:50]}...")

    # 8. Estrutura hierarquica completa do Art. 5
    print("\n[8] Hierarquia completa do Art. 5:")
    if art_5:
        print(f"    {art_5.span_id}")
        for child in doc.get_children("ART-005"):
            print(f"      +-- {child.span_id}: {child.text[:40]}...")
            for grandchild in doc.get_children(child.span_id):
                print(f"          +-- {grandchild.span_id}: {grandchild.text[:35]}...")
                for great_grandchild in doc.get_children(grandchild.span_id):
                    print(f"              +-- {great_grandchild.span_id}: {great_grandchild.text[:30]}...")

    # 9. Debug: verificar paragrafos do Art. 5 e seus incisos
    print("\n[9] Debug paragrafos do Art. 5:")
    for span in doc.spans:
        if span.span_id.startswith("PAR-005"):
            print(f"    {span.span_id}: {span.text[:50]}...")
            # Verificar incisos dentro do paragrafo
            for inc in doc.get_children(span.span_id):
                print(f"      -> {inc.span_id}: {inc.text[:45]}...")
                # Debug: mostrar full_match do inciso
                full_match = inc.metadata.get("full_match", "")
                print(f"         [full_match: {len(full_match)} chars]")
                if "a)" in full_match:
                    print(f"         >> CONTEM ALINEAS!")
                    print(f"         >> {full_match[:200]}...")
                # Verificar alineas dentro do inciso
                for ali in doc.get_children(inc.span_id):
                    print(f"          -> {ali.span_id}: {ali.text[:40]}...")

    return doc


if __name__ == "__main__":
    test_span_parser()
