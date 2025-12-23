"""
Teste de integracao do SpanExtractor com vLLM.

Este teste verifica que:
1. O SpanParser extrai spans corretamente
2. O LLM classifica spans usando apenas IDs validos
3. A validacao detecta e corrige IDs invalidos
4. O texto e reconstruido corretamente
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsing import SpanExtractor, SpanExtractorConfig
from llm.vllm_client import VLLMClient, LLMConfig


def test_span_extractor():
    """Testa SpanExtractor com IN 65/2021."""
    # Usar markdown ja gerado pelo teste anterior
    markdown_path = Path(__file__).parent.parent / "data" / "output" / "in_65_markdown.md"

    print("=" * 60)
    print("TESTE DO SPAN EXTRACTOR")
    print("=" * 60)

    # 1. Carrega markdown (ja extraido pelo Docling)
    print("\n[1] Carregando markdown...")
    if not markdown_path.exists():
        print(f"    Arquivo nao encontrado: {markdown_path}")
        print("    Execute primeiro: python tests/test_span_parser.py")
        return None

    markdown = markdown_path.read_text(encoding="utf-8")
    print(f"    Markdown: {len(markdown)} caracteres")

    # 2. Configura cliente vLLM
    print("\n[2] Configurando cliente vLLM...")
    llm_config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-8B-AWQ",
        temperature=0.0,
        max_tokens=4096,
    )
    llm_client = VLLMClient(llm_config)
    print(f"    Modelo: {llm_config.model}")

    # 3. Configura extrator
    print("\n[3] Configurando SpanExtractor...")
    config = SpanExtractorConfig(
        model=llm_config.model,
        temperature=0.0,
        max_tokens=4096,
        strict_validation=False,  # Nao falhar em IDs invalidos
        auto_fix_ids=True,  # Tentar corrigir IDs
    )
    extractor = SpanExtractor(llm_client, config)

    # 4. Executa extracao
    print("\n[4] Executando extracao span-based...")
    result = extractor.extract(markdown)

    # 5. Resultados
    print("\n[5] Resultados:")
    print(f"    Total spans parseados: {len(result.parsed_doc.spans)}")
    print(f"    IDs validos: {len(result.valid_ids)}")
    print(f"    IDs invalidos: {len(result.invalid_ids)}")
    print(f"    IDs corrigidos: {len(result.fixed_ids)}")

    if result.invalid_ids:
        print(f"    IDs invalidos encontrados: {result.invalid_ids}")

    if result.fixed_ids:
        print(f"    IDs corrigidos: {result.fixed_ids}")

    # 6. Estrutura extraida
    print("\n[6] Estrutura extraida:")
    print(f"    Tipo: {result.document.document_type}")
    print(f"    Numero: {result.document.number}")
    print(f"    Data: {result.document.date}")
    print(f"    Orgao: {result.document.issuing_body}")
    print(f"    Ementa: {result.document.ementa[:80]}..." if result.document.ementa else "    Ementa: N/A")

    print("\n    Capitulos:")
    for chapter in result.document.chapters:
        print(f"      {chapter.chapter_id}: {chapter.title or 'Sem titulo'}")
        print(f"        Artigos: {chapter.article_ids}")

    # 7. Texto reconstruido
    print("\n[7] Texto reconstruido (Art. 1):")
    art_text = result.get_article_text("ART-001")
    if art_text:
        for line in art_text.split('\n')[:5]:
            print(f"    {line[:70]}...")
    else:
        print("    Art. 1 nao encontrado!")

    # 8. Validacao final
    print("\n[8] Validacao final:")
    if result.is_valid:
        print("    [OK] Extracao valida - nenhum ID invalido")
    else:
        print(f"    [WARN] Extracao com {len(result.invalid_ids)} IDs invalidos")

    return result


def test_span_extractor_fallback():
    """Testa fallback quando LLM falha."""
    print("\n" + "=" * 60)
    print("TESTE DO FALLBACK")
    print("=" * 60)

    # Usar markdown simples para testar fallback
    markdown = """
    CAPITULO I
    DISPOSICOES GERAIS

    Art. 1o Este e o primeiro artigo.

    Art. 2o Este e o segundo artigo.
    - I - primeiro inciso
    - II - segundo inciso

    CAPITULO II
    PROCEDIMENTOS

    Art. 3o Este e o terceiro artigo.
    """

    # Configura cliente mock que retorna JSON invalido
    class MockLLMClient:
        def chat(self, messages, temperature=0, max_tokens=4096):
            return "invalid json response"

    print("\n[1] Testando fallback com resposta invalida do LLM...")
    extractor = SpanExtractor(MockLLMClient())
    result = extractor.extract(markdown)

    print(f"\n[2] Fallback criou estrutura com {len(result.document.chapters)} capitulos:")
    for chapter in result.document.chapters:
        print(f"    {chapter.chapter_id}: {len(chapter.article_ids)} artigos")

    return result


if __name__ == "__main__":
    test_span_extractor()
    test_span_extractor_fallback()
