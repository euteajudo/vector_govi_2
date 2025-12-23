"""
Teste do Dashboard de Ingestão.

Testa:
1. Coleta de métricas por fase
2. Métricas de artigos individuais
3. Agregação de métricas de documento
4. Geração de relatório
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dashboard import (
    MetricsCollector,
    ArticleMetrics,
    DocumentMetrics,
    PhaseMetrics,
    IngestionMetrics,
    generate_dashboard_report,
)


def test_phase_metrics():
    """Testa métricas de fase."""
    print("=" * 60)
    print("TESTE: Phase Metrics")
    print("=" * 60)

    collector = MetricsCollector(ingestion_id="test-001")

    # Simula fase de parsing
    collector.start_phase("parsing")
    time.sleep(0.1)  # Simula processamento
    collector.end_phase("parsing", items_processed=100, errors=2)

    # Simula fase de extraction
    collector.start_phase("extraction")
    time.sleep(0.05)
    collector.end_phase("extraction", items_processed=11, errors=0)

    report = collector.generate_report()

    print(f"\n[1] Fases registradas: {len(report.phases)}")
    for phase in report.phases:
        print(f"    - {phase.name}: {phase.duration_seconds:.2f}s, {phase.items_processed} items")

    assert len(report.phases) == 2
    assert report.phases[0].name == "parsing"
    assert report.phases[0].items_processed == 100
    print("    [OK] Fases registradas corretamente")

    return True


def test_article_metrics():
    """Testa métricas de artigos."""
    print("\n" + "=" * 60)
    print("TESTE: Article Metrics")
    print("=" * 60)

    collector = MetricsCollector(ingestion_id="test-002")
    collector.set_document_info(
        document_id="IN-65-2021",
        tipo_documento="INSTRUCAO NORMATIVA",
        numero="65",
        ano=2021,
    )

    # Registra artigos
    collector.record_article_metrics(
        article_id="ART-001",
        article_number="1",
        parser_paragrafos=2,
        llm_paragrafos=2,
        parser_incisos=5,
        llm_incisos=5,
        status="valid",
        tokens_prompt=500,
        tokens_completion=100,
        extraction_time_ms=1500,
    )

    collector.record_article_metrics(
        article_id="ART-005",
        article_number="5",
        parser_paragrafos=3,
        llm_paragrafos=2,  # Faltou 1
        parser_incisos=4,
        llm_incisos=4,
        status="suspect",
        validation_notes=["Cobertura parágrafos: 2/3 (67%)"],
        tokens_prompt=600,
        tokens_completion=120,
        extraction_time_ms=1800,
        retry_count=1,
    )

    collector.record_article_metrics(
        article_id="ART-010",
        article_number="10",
        parser_paragrafos=0,
        llm_paragrafos=0,
        parser_incisos=0,
        llm_incisos=0,
        status="valid",  # Artigo sem filhos
        tokens_prompt=0,
        tokens_completion=0,
        extraction_time_ms=0,  # Curto-circuito
    )

    report = collector.generate_report()

    print(f"\n[1] Artigos registrados: {len(report.articles)}")
    for article in report.articles:
        print(f"    - {article.article_id}: {article.status}")
        print(f"      Cobertura PAR: {article.coverage_paragrafos:.0%}")
        print(f"      Cobertura INC: {article.coverage_incisos:.0%}")

    assert len(report.articles) == 3
    assert report.articles[0].coverage_paragrafos == 1.0
    assert report.articles[1].coverage_paragrafos < 1.0
    print("    [OK] Métricas de artigos corretas")

    print(f"\n[2] Agregação do documento:")
    doc = report.document
    print(f"    Total: {doc.total_articles}")
    print(f"    Válidos: {doc.valid_articles}")
    print(f"    Suspeitos: {doc.suspect_articles}")
    print(f"    Taxa sucesso: {doc.success_rate:.0%}")

    assert doc.total_articles == 3
    assert doc.valid_articles == 2
    assert doc.suspect_articles == 1
    print("    [OK] Agregação correta")

    return report


def test_coverage_calculation():
    """Testa cálculo de cobertura."""
    print("\n" + "=" * 60)
    print("TESTE: Coverage Calculation")
    print("=" * 60)

    article = ArticleMetrics(
        article_id="ART-005",
        parser_paragrafos=4,
        llm_paragrafos=3,
        parser_incisos=10,
        llm_incisos=8,
    )

    print(f"\n[1] Artigo com cobertura parcial:")
    print(f"    Parágrafos: {article.llm_paragrafos}/{article.parser_paragrafos}")
    print(f"    Cobertura PAR: {article.coverage_paragrafos:.2%}")
    print(f"    Incisos: {article.llm_incisos}/{article.parser_incisos}")
    print(f"    Cobertura INC: {article.coverage_incisos:.2%}")
    print(f"    Cobertura Total: {article.coverage_total:.2%}")

    assert article.coverage_paragrafos == 0.75
    assert article.coverage_incisos == 0.80
    assert article.coverage_total == 0.75  # min(0.75, 0.80)
    print("    [OK] Cálculo de cobertura correto")

    # Artigo sem filhos
    article_empty = ArticleMetrics(
        article_id="ART-001",
        parser_paragrafos=0,
        llm_paragrafos=0,
        parser_incisos=0,
        llm_incisos=0,
    )

    print(f"\n[2] Artigo sem filhos:")
    print(f"    Cobertura PAR: {article_empty.coverage_paragrafos:.2%}")
    print(f"    Cobertura INC: {article_empty.coverage_incisos:.2%}")

    assert article_empty.coverage_paragrafos == 1.0
    assert article_empty.coverage_incisos == 1.0
    print("    [OK] Artigo vazio tem 100% cobertura")

    return True


def test_dashboard_report():
    """Testa geração do relatório do dashboard."""
    print("\n" + "=" * 60)
    print("TESTE: Dashboard Report")
    print("=" * 60)

    collector = MetricsCollector(ingestion_id="IN-65-2021-001")
    collector.set_document_info(
        document_id="IN-65-2021",
        tipo_documento="IN",
        numero="65",
        ano=2021,
    )

    # Simula fases
    collector.start_phase("parsing")
    time.sleep(0.05)
    collector.end_phase("parsing", items_processed=1)

    collector.start_phase("extraction")
    time.sleep(0.1)
    collector.end_phase("extraction", items_processed=11)

    # Registra artigos
    for i in range(1, 12):
        status = "valid"
        if i == 5:
            status = "suspect"
        elif i == 10:
            status = "invalid"

        collector.record_article_metrics(
            article_id=f"ART-{i:03d}",
            article_number=str(i),
            parser_paragrafos=2,
            llm_paragrafos=2 if status == "valid" else 1,
            parser_incisos=3,
            llm_incisos=3 if status == "valid" else 2,
            status=status,
            tokens_prompt=500,
            tokens_completion=100,
            extraction_time_ms=1500,
            retry_count=1 if i == 5 else 0,
        )

    collector.set_chunk_counts(
        total=47,
        articles=11,
        paragraphs=19,
        incisos=17,
    )

    report = collector.generate_report()

    # Gera relatório formatado
    dashboard_text = generate_dashboard_report(report)

    print("\n" + dashboard_text)

    # Verifica conteúdo
    assert "IN-65-2021" in dashboard_text
    assert "47" in dashboard_text  # Total chunks
    assert "SUSPECT" in dashboard_text or "Suspeitos" in dashboard_text
    print("\n    [OK] Relatório gerado com sucesso")

    return report


def test_json_serialization():
    """Testa serialização JSON."""
    print("\n" + "=" * 60)
    print("TESTE: JSON Serialization")
    print("=" * 60)

    collector = MetricsCollector(ingestion_id="test-json")
    collector.set_document_info(document_id="TEST-001")
    collector.record_article_metrics(
        article_id="ART-001",
        status="valid",
        tokens_prompt=100,
    )

    report = collector.generate_report()
    json_str = report.to_json()

    print(f"\n[1] JSON gerado ({len(json_str)} chars):")
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)

    # Verifica se é JSON válido
    import json
    data = json.loads(json_str)

    assert "ingestion_id" in data
    assert "document" in data
    assert "articles" in data
    print("\n    [OK] JSON válido e estruturado")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTES DO DASHBOARD DE INGESTÃO")
    print("=" * 70)

    test_phase_metrics()
    test_article_metrics()
    test_coverage_calculation()
    test_dashboard_report()
    test_json_serialization()

    print("\n" + "=" * 70)
    print("TODOS OS TESTES PASSARAM!")
    print("=" * 70)
