"""
Teste do ArticleOrchestrator - Fase 3.

Testa:
1. Extracao por artigo com hierarquia completa
2. Validacao de cobertura (parser vs LLM)
3. Deteccao de duplicatas
4. Materializacao de chunks com citations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsing import (
    SpanParser,
    ArticleOrchestrator,
    OrchestratorConfig,
    ValidationStatus,
)
from llm.vllm_client import VLLMClient, LLMConfig


def test_article_orchestrator():
    """Testa ArticleOrchestrator com IN 65/2021."""
    markdown_path = Path(__file__).parent.parent / "data" / "output" / "in_65_markdown.md"

    print("=" * 70)
    print("TESTE DO ARTICLE ORCHESTRATOR - FASE 3")
    print("=" * 70)

    # 1. Carrega markdown
    print("\n[1] Carregando markdown...")
    if not markdown_path.exists():
        print(f"    Arquivo nao encontrado: {markdown_path}")
        print("    Execute primeiro: python tests/test_span_parser.py")
        return None

    markdown = markdown_path.read_text(encoding="utf-8")
    print(f"    Markdown: {len(markdown)} caracteres")

    # 2. Parseia com SpanParser
    print("\n[2] Parseando com SpanParser...")
    parser = SpanParser()
    parsed_doc = parser.parse(markdown)
    print(f"    Total spans: {len(parsed_doc.spans)}")
    print(f"    Artigos: {len(parsed_doc.articles)}")

    # 3. Configura cliente vLLM
    print("\n[3] Configurando cliente vLLM...")
    llm_config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-8B-AWQ",
        temperature=0.0,
        max_tokens=2048,
    )
    llm_client = VLLMClient(llm_config)
    print(f"    Modelo: {llm_config.model}")

    # 4. Configura orquestrador
    print("\n[4] Configurando ArticleOrchestrator...")
    config = OrchestratorConfig(
        temperature=0.0,
        max_tokens=512,  # Suficiente para JSON de IDs
        strict_validation=False,  # Nao falhar em IDs invalidos
        coverage_threshold=0.8,
        auto_fix_article_ids=True,
        auto_fix_child_ids=False,  # Nao adivinhar PAR/INC/ALI
    )
    orchestrator = ArticleOrchestrator(llm_client, config)

    # 5. Executa extracao por artigo
    print("\n[5] Executando extracao por artigo...")
    result = orchestrator.extract_all_articles(parsed_doc)

    # 6. Resultados gerais
    print("\n[6] Resultados:")
    print(f"    Total artigos: {result.total_articles}")
    print(f"    Validos: {result.valid_articles}")
    print(f"    Suspeitos: {result.suspect_articles}")
    print(f"    Invalidos: {result.invalid_articles}")
    print(f"    Taxa de sucesso: {result.success_rate:.0%}")

    # 7. Detalhes por artigo
    print("\n[7] Detalhes por artigo:")
    for chunk in result.chunks[:5]:  # Primeiros 5
        status_icon = {
            ValidationStatus.VALID: "[OK]",
            ValidationStatus.SUSPECT: "[??]",
            ValidationStatus.INVALID: "[XX]",
        }.get(chunk.status, "[--]")

        print(f"\n    {status_icon} {chunk.article_id} (Art. {chunk.article_number})")
        print(f"        Paragrafos: {chunk.paragrafo_ids}")
        print(f"        Incisos: {chunk.inciso_ids[:5]}{'...' if len(chunk.inciso_ids) > 5 else ''}")
        print(f"        Citations: {len(chunk.citations)} spans")
        print(f"        Cobertura PAR: {chunk.llm_paragrafos_count}/{chunk.parser_paragrafos_count} ({chunk.coverage_paragrafos:.0%})")
        print(f"        Cobertura INC: {chunk.llm_incisos_count}/{chunk.parser_incisos_count} ({chunk.coverage_incisos:.0%})")
        if chunk.retry_count > 0:
            print(f"        Retries: {chunk.retry_count}")

        if chunk.validation_notes:
            print(f"        Notas: {chunk.validation_notes}")

    # 8. Teste especifico do Art. 5 (tem paragrafos, incisos e alineas)
    print("\n[8] Validacao detalhada do Art. 5:")
    art5_chunk = next((c for c in result.chunks if c.article_id == "ART-005"), None)

    if art5_chunk:
        print(f"    Status: {art5_chunk.status.value}")
        print(f"    Paragrafos encontrados: {art5_chunk.paragrafo_ids}")
        print(f"    Incisos encontrados: {art5_chunk.inciso_ids}")
        print(f"    Total citations: {len(art5_chunk.citations)}")

        # Verifica cobertura esperada
        expected_pars = ["PAR-005-1", "PAR-005-2"]
        expected_incs = ["INC-005-I", "INC-005-II", "INC-005-III", "INC-005-IV", "INC-005-V"]

        found_pars = set(art5_chunk.paragrafo_ids)
        found_incs = set(art5_chunk.inciso_ids)

        missing_pars = set(expected_pars) - found_pars
        missing_incs = set(expected_incs) - found_incs

        if missing_pars:
            print(f"    [WARN] Paragrafos faltando: {missing_pars}")
        else:
            print(f"    [OK] Todos os paragrafos esperados encontrados")

        if missing_incs:
            print(f"    [WARN] Incisos faltando: {missing_incs}")
        else:
            print(f"    [OK] Todos os incisos do caput encontrados")

        # Verifica incisos do paragrafo 2
        par2_incs = ["INC-005-I_2", "INC-005-II_2"]
        found_par2_incs = [i for i in art5_chunk.inciso_ids if "_2" in i]
        print(f"    Incisos do PAR-005-2: {found_par2_incs}")

    else:
        print("    Art. 5 nao encontrado!")

    # 9. Texto reconstruido (amostra)
    print("\n[9] Texto reconstruido do Art. 5 (primeiras 10 linhas):")
    if art5_chunk:
        for i, line in enumerate(art5_chunk.text.split('\n')[:10]):
            print(f"    {line[:70]}...")

    # 10. Teste de duplicatas
    print("\n[10] Verificacao de duplicatas:")
    has_duplicates = False
    for chunk in result.chunks:
        all_ids = chunk.inciso_ids + chunk.paragrafo_ids
        seen = set()
        dups = []
        for id_ in all_ids:
            if id_ in seen:
                dups.append(id_)
            seen.add(id_)
        if dups:
            print(f"    [WARN] {chunk.article_id} tem duplicatas: {dups}")
            has_duplicates = True

    if not has_duplicates:
        print("    [OK] Nenhuma duplicata encontrada")

    return result


def test_coverage_validation():
    """Testa validacao de cobertura especificamente."""
    print("\n" + "=" * 70)
    print("TESTE DE COBERTURA")
    print("=" * 70)

    # Markdown com incisos para testar cobertura
    markdown = """
Art. 1o Este artigo tem varios incisos.

I - primeiro inciso

II - segundo inciso

III - terceiro inciso

IV - quarto inciso

V - quinto inciso
"""

    # Parseia
    parser = SpanParser()
    parsed_doc = parser.parse(markdown)

    print(f"\n[1] Parser encontrou:")
    for span in parsed_doc.spans:
        print(f"    {span.span_id}: {span.text[:40]}...")

    # Conta incisos para debug
    incisos = [s for s in parsed_doc.spans if s.span_type.value == "inciso"]
    print(f"\n    Total incisos detectados: {len(incisos)}")

    # Simula LLM que retorna apenas alguns IDs via chat_with_schema
    class MockLLMClient:
        def chat_with_schema(self, messages, schema, temperature=0, max_tokens=512):
            # Simula LLM que "esquece" alguns incisos (retorna 2 de 5)
            return {
                "article_id": "ART-001",
                "inciso_ids": ["INC-001-I", "INC-001-II"],
                "paragrafo_ids": []
            }

    print("\n[2] Testando com LLM mock que retorna cobertura parcial (2/5 incisos)...")
    config = OrchestratorConfig(coverage_threshold=0.8, strict_validation=False)
    orchestrator = ArticleOrchestrator(MockLLMClient(), config)
    result = orchestrator.extract_all_articles(parsed_doc)

    if result.chunks:
        chunk = result.chunks[0]
        print(f"\n[3] Resultado:")
        print(f"    Status: {chunk.status.value}")
        print(f"    Cobertura PAR: {chunk.llm_paragrafos_count}/{chunk.parser_paragrafos_count} ({chunk.coverage_paragrafos:.0%})")
        print(f"    Cobertura INC: {chunk.llm_incisos_count}/{chunk.parser_incisos_count} ({chunk.coverage_incisos:.0%})")
        print(f"    Notas: {chunk.validation_notes}")

        if chunk.status == ValidationStatus.SUSPECT:
            print("    [OK] Cobertura baixa detectada corretamente")
        else:
            print("    [WARN] Deveria ser SUSPECT")

    return result


if __name__ == "__main__":
    # Teste principal com vLLM
    test_article_orchestrator()

    # Teste de cobertura com mock
    test_coverage_validation()
