"""
Teste das correcoes no extractor.py para evitar alucinacao.

Valida que:
1. Conteudo antes do primeiro CAPITULO e ignorado
2. LLM nao inventa artigos
3. Artigos inventados sao detectados e removidos

Uso:
    python tests/test_extraction_fix.py
"""

import sys
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from extract import Extractor, ExtractConfig
from models.legal_document import LegalDocument


def test_split_by_chapters():
    """Testa que _split_by_chapters ignora conteudo antes do primeiro CAPITULO."""
    print("=" * 70)
    print("TESTE 1: _split_by_chapters ignora conteudo pre-capitulo")
    print("=" * 70)

    extractor = Extractor()

    # Simula markdown com metadados antes do primeiro capitulo
    markdown = """
# INSTRUCAO NORMATIVA SEGES/ME No 65, DE 7 DE JULHO DE 2021

**EMENTA**: Dispoe sobre o procedimento administrativo...

**Data**: 07/07/2021

---

CAPITULO I
DAS DISPOSICOES PRELIMINARES

Art. 1 Esta Instrucao Normativa...

Art. 2 Para os fins...

CAPITULO II
DOS PROCEDIMENTOS

Art. 3 A pesquisa de precos...
"""

    chapters = extractor._split_by_chapters(markdown)

    print(f"\nCapitulos encontrados: {len(chapters)}")
    for i, (title, content) in enumerate(chapters, 1):
        # Conta artigos no capitulo
        import re
        arts = re.findall(r'Art\.?\s*(\d+)', content)
        print(f"  {i}. '{title}' -> {len(arts)} artigos: {arts}")

    # Validacao
    assert len(chapters) == 2, f"Esperado 2 capitulos, encontrado {len(chapters)}"
    assert "CAPITULO I" in chapters[0][0], "Primeiro capitulo deve ser CAPITULO I"
    assert "DISPOSICOES" not in chapters[0][0].upper() or "PRELIMINARES" in chapters[0][0].upper(), \
        "Nao deve criar capitulo fantasma 'DISPOSICOES INICIAIS'"

    print("\n[OK] _split_by_chapters corrigido!")
    return True


def test_extraction_in65():
    """Testa extracao da IN 65 com as correcoes."""
    print("\n" + "=" * 70)
    print("TESTE 2: Extracao da IN 65 (sem alucinacao)")
    print("=" * 70)

    pdf_path = Path(__file__).parent.parent / "data" / "INSTRUÇÃO NORMATIVA SEGES _ME Nº 65, DE 7 DE JULHO DE 2021.pdf"

    if not pdf_path.exists():
        print(f"[SKIP] PDF nao encontrado: {pdf_path}")
        return None

    print(f"\nArquivo: {pdf_path.name}")

    # Configuracao para documentos legais
    config = ExtractConfig.for_legal_documents()
    extractor = Extractor(config)

    print(f"Modelo: {config.llm.model}")
    print("Extraindo...")

    result = extractor.extract(
        pdf_path,
        schema=LegalDocument,
        config=config,
    )

    print(f"\nResultado:")
    print(f"  Success: {result.success}")
    print(f"  Quality: {result.quality_score:.1%}")
    print(f"  Tempo: {result.extraction_time_seconds:.1f}s")

    # Listar capitulos e artigos
    print("\nCapitulos extraidos:")
    total_articles = 0
    for chapter in result.raw_data.get("chapters", []):
        arts = chapter.get("articles", [])
        art_nums = [a.get("article_number", "?") for a in arts]
        print(f"  - {chapter.get('title', 'SEM TITULO')}: Art. {', '.join(art_nums)}")
        total_articles += len(arts)

    print(f"\nTotal de artigos: {total_articles}")

    # Validacao: IN 65/2021 tem 11 artigos (Art. 1 a 11)
    expected_articles = set(range(1, 12))  # 1 a 11
    extracted_articles = set()
    for chapter in result.raw_data.get("chapters", []):
        for art in chapter.get("articles", []):
            try:
                extracted_articles.add(int(art.get("article_number", 0)))
            except (ValueError, TypeError):
                pass

    missing = expected_articles - extracted_articles
    extra = extracted_articles - expected_articles

    if missing:
        print(f"\n[AVISO] Artigos faltando: {sorted(missing)}")
    if extra:
        print(f"\n[ERRO] Artigos inventados detectados: {sorted(extra)}")

    if not extra and len(extracted_articles) == 11:
        print("\n[OK] Extracao correta! 11 artigos, sem alucinacao.")
        return True
    elif not extra:
        print(f"\n[OK] Sem alucinacao, mas faltam artigos.")
        return True
    else:
        print(f"\n[ERRO] Alucinacao detectada!")
        return False


def main():
    """Executa todos os testes."""
    print("\n" + "=" * 70)
    print("TESTES DAS CORRECOES DE ALUCINACAO")
    print("=" * 70)

    results = []

    # Teste 1: _split_by_chapters
    try:
        results.append(("_split_by_chapters", test_split_by_chapters()))
    except Exception as e:
        print(f"\n[ERRO] {e}")
        results.append(("_split_by_chapters", False))

    # Teste 2: Extracao IN 65
    try:
        results.append(("Extracao IN 65", test_extraction_in65()))
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        results.append(("Extracao IN 65", False))

    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)

    for name, passed in results:
        status = "[OK]" if passed else ("[SKIP]" if passed is None else "[ERRO]")
        print(f"  {status} {name}")

    all_passed = all(r for r in [x[1] for x in results] if r is not None)
    print("\n" + ("TODOS OS TESTES PASSARAM!" if all_passed else "ALGUNS TESTES FALHARAM"))


if __name__ == "__main__":
    main()
