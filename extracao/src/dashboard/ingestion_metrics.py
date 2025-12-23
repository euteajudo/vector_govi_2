"""
Ingestion Metrics - Métricas de ingestão para o dashboard.

Coleta e agrega métricas de:
- Cobertura: parágrafos, incisos, alíneas por artigo
- Latência: tempo de parsing, extração LLM, embedding, indexação
- Custo: tokens prompt/completion, chamadas LLM
- Qualidade: artigos válidos, suspeitos, inválidos

Uso:
    from dashboard import MetricsCollector, generate_dashboard_report

    # Durante o pipeline
    collector = MetricsCollector()
    collector.start_phase("parsing")

    # Após parsing
    collector.end_phase("parsing")
    collector.start_phase("extraction")

    # Para cada artigo
    collector.record_article_metrics(
        article_id="ART-005",
        parser_paragrafos=3,
        llm_paragrafos=3,
        parser_incisos=5,
        llm_incisos=5,
        status="valid",
        tokens_prompt=500,
        tokens_completion=100,
    )

    # Gera relatório
    report = collector.generate_report()
    print(generate_dashboard_report(report))
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import time
import json


@dataclass
class PhaseMetrics:
    """Métricas de uma fase do pipeline."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    items_processed: int = 0
    errors: int = 0

    @property
    def duration_seconds(self) -> float:
        if self.end_time <= 0:
            return 0.0
        return self.end_time - self.start_time

    @property
    def items_per_second(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.items_processed / self.duration_seconds

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_seconds": round(self.duration_seconds, 2),
            "items_processed": self.items_processed,
            "errors": self.errors,
            "items_per_second": round(self.items_per_second, 3),
        }


@dataclass
class ArticleMetrics:
    """Métricas de um artigo individual."""

    article_id: str
    article_number: str = ""

    # Cobertura
    parser_paragrafos: int = 0
    llm_paragrafos: int = 0
    parser_incisos: int = 0
    llm_incisos: int = 0
    parser_alineas: int = 0
    llm_alineas: int = 0

    # Status
    status: str = "valid"  # valid, suspect, invalid
    validation_notes: list[str] = field(default_factory=list)

    # Tokens
    tokens_prompt: int = 0
    tokens_completion: int = 0

    # Tempo
    extraction_time_ms: int = 0
    retry_count: int = 0

    @property
    def coverage_paragrafos(self) -> float:
        if self.parser_paragrafos == 0:
            return 1.0
        return self.llm_paragrafos / self.parser_paragrafos

    @property
    def coverage_incisos(self) -> float:
        if self.parser_incisos == 0:
            return 1.0
        return self.llm_incisos / self.parser_incisos

    @property
    def coverage_total(self) -> float:
        return min(self.coverage_paragrafos, self.coverage_incisos)

    @property
    def total_tokens(self) -> int:
        return self.tokens_prompt + self.tokens_completion

    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "article_number": self.article_number,
            "coverage": {
                "paragrafos": f"{self.llm_paragrafos}/{self.parser_paragrafos}",
                "paragrafos_pct": f"{self.coverage_paragrafos:.0%}",
                "incisos": f"{self.llm_incisos}/{self.parser_incisos}",
                "incisos_pct": f"{self.coverage_incisos:.0%}",
            },
            "status": self.status,
            "tokens": {
                "prompt": self.tokens_prompt,
                "completion": self.tokens_completion,
                "total": self.total_tokens,
            },
            "extraction_time_ms": self.extraction_time_ms,
            "retry_count": self.retry_count,
        }


@dataclass
class DocumentMetrics:
    """Métricas agregadas de um documento."""

    document_id: str = ""
    tipo_documento: str = ""
    numero: str = ""
    ano: int = 0

    # Contagens
    total_articles: int = 0
    valid_articles: int = 0
    suspect_articles: int = 0
    invalid_articles: int = 0

    # Cobertura agregada
    total_parser_paragrafos: int = 0
    total_llm_paragrafos: int = 0
    total_parser_incisos: int = 0
    total_llm_incisos: int = 0

    # Tokens
    total_tokens_prompt: int = 0
    total_tokens_completion: int = 0

    # Tempo
    total_extraction_time_ms: int = 0

    # Chunks
    total_chunks: int = 0
    article_chunks: int = 0
    paragraph_chunks: int = 0
    inciso_chunks: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_articles == 0:
            return 0.0
        return self.valid_articles / self.total_articles

    @property
    def coverage_paragrafos(self) -> float:
        if self.total_parser_paragrafos == 0:
            return 1.0
        return self.total_llm_paragrafos / self.total_parser_paragrafos

    @property
    def coverage_incisos(self) -> float:
        if self.total_parser_incisos == 0:
            return 1.0
        return self.total_llm_incisos / self.total_parser_incisos

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_prompt + self.total_tokens_completion

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "tipo_documento": self.tipo_documento,
            "numero": self.numero,
            "ano": self.ano,
            "articles": {
                "total": self.total_articles,
                "valid": self.valid_articles,
                "suspect": self.suspect_articles,
                "invalid": self.invalid_articles,
                "success_rate": f"{self.success_rate:.0%}",
            },
            "coverage": {
                "paragrafos": f"{self.total_llm_paragrafos}/{self.total_parser_paragrafos}",
                "paragrafos_pct": f"{self.coverage_paragrafos:.0%}",
                "incisos": f"{self.total_llm_incisos}/{self.total_parser_incisos}",
                "incisos_pct": f"{self.coverage_incisos:.0%}",
            },
            "tokens": {
                "prompt": self.total_tokens_prompt,
                "completion": self.total_tokens_completion,
                "total": self.total_tokens,
            },
            "extraction_time_ms": self.total_extraction_time_ms,
            "chunks": {
                "total": self.total_chunks,
                "articles": self.article_chunks,
                "paragraphs": self.paragraph_chunks,
                "incisos": self.inciso_chunks,
            },
        }


@dataclass
class IngestionMetrics:
    """Métricas completas de uma ingestão."""

    # Identificação
    ingestion_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Fases
    phases: list[PhaseMetrics] = field(default_factory=list)

    # Documento
    document: DocumentMetrics = field(default_factory=DocumentMetrics)

    # Artigos individuais
    articles: list[ArticleMetrics] = field(default_factory=list)

    # Status final
    status: str = "completed"  # running, completed, failed
    error: Optional[str] = None

    @property
    def total_duration_seconds(self) -> float:
        return sum(p.duration_seconds for p in self.phases)

    def to_dict(self) -> dict:
        return {
            "ingestion_id": self.ingestion_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "error": self.error,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "phases": [p.to_dict() for p in self.phases],
            "document": self.document.to_dict(),
            "articles": [a.to_dict() for a in self.articles],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class MetricsCollector:
    """
    Coletor de métricas de ingestão.

    Uso:
        collector = MetricsCollector(ingestion_id="IN-65-2021-001")

        collector.start_phase("parsing")
        # ... parsing ...
        collector.end_phase("parsing")

        collector.record_article_metrics(...)

        report = collector.generate_report()
    """

    def __init__(self, ingestion_id: str = ""):
        self.metrics = IngestionMetrics(
            ingestion_id=ingestion_id or f"ing-{int(time.time())}"
        )
        self._current_phase: Optional[PhaseMetrics] = None
        self._phase_map: dict[str, PhaseMetrics] = {}

    def start_phase(self, name: str) -> None:
        """Inicia uma nova fase do pipeline."""
        phase = PhaseMetrics(name=name, start_time=time.time())
        self._current_phase = phase
        self._phase_map[name] = phase
        self.metrics.phases.append(phase)

    def end_phase(self, name: str, items_processed: int = 0, errors: int = 0) -> None:
        """Finaliza uma fase do pipeline."""
        phase = self._phase_map.get(name)
        if phase:
            phase.end_time = time.time()
            phase.items_processed = items_processed
            phase.errors = errors

    def record_article_metrics(
        self,
        article_id: str,
        article_number: str = "",
        parser_paragrafos: int = 0,
        llm_paragrafos: int = 0,
        parser_incisos: int = 0,
        llm_incisos: int = 0,
        parser_alineas: int = 0,
        llm_alineas: int = 0,
        status: str = "valid",
        validation_notes: Optional[list[str]] = None,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        extraction_time_ms: int = 0,
        retry_count: int = 0,
    ) -> None:
        """Registra métricas de um artigo."""
        article = ArticleMetrics(
            article_id=article_id,
            article_number=article_number,
            parser_paragrafos=parser_paragrafos,
            llm_paragrafos=llm_paragrafos,
            parser_incisos=parser_incisos,
            llm_incisos=llm_incisos,
            parser_alineas=parser_alineas,
            llm_alineas=llm_alineas,
            status=status,
            validation_notes=validation_notes or [],
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            extraction_time_ms=extraction_time_ms,
            retry_count=retry_count,
        )
        self.metrics.articles.append(article)

        # Atualiza agregados do documento
        doc = self.metrics.document
        doc.total_articles += 1
        if status == "valid":
            doc.valid_articles += 1
        elif status == "suspect":
            doc.suspect_articles += 1
        else:
            doc.invalid_articles += 1

        doc.total_parser_paragrafos += parser_paragrafos
        doc.total_llm_paragrafos += llm_paragrafos
        doc.total_parser_incisos += parser_incisos
        doc.total_llm_incisos += llm_incisos
        doc.total_tokens_prompt += tokens_prompt
        doc.total_tokens_completion += tokens_completion
        doc.total_extraction_time_ms += extraction_time_ms

    def set_document_info(
        self,
        document_id: str = "",
        tipo_documento: str = "",
        numero: str = "",
        ano: int = 0,
    ) -> None:
        """Define informações do documento."""
        self.metrics.document.document_id = document_id
        self.metrics.document.tipo_documento = tipo_documento
        self.metrics.document.numero = numero
        self.metrics.document.ano = ano

    def set_chunk_counts(
        self,
        total: int = 0,
        articles: int = 0,
        paragraphs: int = 0,
        incisos: int = 0,
    ) -> None:
        """Define contagem de chunks."""
        self.metrics.document.total_chunks = total
        self.metrics.document.article_chunks = articles
        self.metrics.document.paragraph_chunks = paragraphs
        self.metrics.document.inciso_chunks = incisos

    def set_error(self, error: str) -> None:
        """Define erro e marca como falha."""
        self.metrics.status = "failed"
        self.metrics.error = error

    def generate_report(self) -> IngestionMetrics:
        """Gera o relatório final de métricas."""
        if self.metrics.status == "running":
            self.metrics.status = "completed"
        return self.metrics


def generate_dashboard_report(metrics: IngestionMetrics) -> str:
    """
    Gera relatório formatado para exibição no terminal.

    Args:
        metrics: IngestionMetrics coletadas

    Returns:
        String formatada para terminal
    """
    lines = []
    doc = metrics.document

    # Header
    lines.append("=" * 70)
    lines.append("DASHBOARD DE INGESTÃO")
    lines.append("=" * 70)
    lines.append(f"Ingestion ID: {metrics.ingestion_id}")
    lines.append(f"Timestamp: {metrics.timestamp}")
    lines.append(f"Status: {metrics.status}")
    if metrics.error:
        lines.append(f"Erro: {metrics.error}")

    # Documento
    lines.append("\n" + "-" * 70)
    lines.append("DOCUMENTO")
    lines.append("-" * 70)
    lines.append(f"ID: {doc.document_id}")
    lines.append(f"Tipo: {doc.tipo_documento} {doc.numero}/{doc.ano}")

    # Artigos
    lines.append("\n" + "-" * 70)
    lines.append("ARTIGOS")
    lines.append("-" * 70)
    lines.append(f"Total: {doc.total_articles}")
    lines.append(f"  [OK] Validos: {doc.valid_articles} ({doc.success_rate:.0%})")
    lines.append(f"  [!!] Suspeitos: {doc.suspect_articles}")
    lines.append(f"  [XX] Invalidos: {doc.invalid_articles}")

    # Cobertura
    lines.append("\n" + "-" * 70)
    lines.append("COBERTURA")
    lines.append("-" * 70)
    lines.append(
        f"Parágrafos: {doc.total_llm_paragrafos}/{doc.total_parser_paragrafos} "
        f"({doc.coverage_paragrafos:.0%})"
    )
    lines.append(
        f"Incisos: {doc.total_llm_incisos}/{doc.total_parser_incisos} "
        f"({doc.coverage_incisos:.0%})"
    )

    # Chunks
    lines.append("\n" + "-" * 70)
    lines.append("CHUNKS GERADOS")
    lines.append("-" * 70)
    lines.append(f"Total: {doc.total_chunks}")
    lines.append(f"  ARTICLE: {doc.article_chunks}")
    lines.append(f"  PARAGRAPH: {doc.paragraph_chunks}")
    lines.append(f"  INCISO: {doc.inciso_chunks}")

    # Tokens
    lines.append("\n" + "-" * 70)
    lines.append("TOKENS LLM")
    lines.append("-" * 70)
    lines.append(f"Prompt: {doc.total_tokens_prompt:,}")
    lines.append(f"Completion: {doc.total_tokens_completion:,}")
    lines.append(f"Total: {doc.total_tokens:,}")

    # Custo estimado (usando preços médios)
    # Preços aproximados para Qwen 8B local (0, mas para referência com API)
    # Usando preços típicos de API: $0.001/1K tokens
    cost_per_1k = 0.001
    estimated_cost = (doc.total_tokens / 1000) * cost_per_1k
    lines.append(f"Custo estimado (API ref): ${estimated_cost:.4f}")

    # Fases
    lines.append("\n" + "-" * 70)
    lines.append("FASES DO PIPELINE")
    lines.append("-" * 70)
    for phase in metrics.phases:
        lines.append(
            f"{phase.name}: {phase.duration_seconds:.2f}s "
            f"({phase.items_processed} items, {phase.errors} erros)"
        )

    lines.append(f"\nTempo Total: {metrics.total_duration_seconds:.2f}s")

    # Artigos problemáticos
    problem_articles = [a for a in metrics.articles if a.status != "valid"]
    if problem_articles:
        lines.append("\n" + "-" * 70)
        lines.append("ARTIGOS PROBLEMÁTICOS")
        lines.append("-" * 70)
        for article in problem_articles:
            lines.append(
                f"  {article.article_id} [{article.status.upper()}]"
            )
            if article.validation_notes:
                for note in article.validation_notes:
                    lines.append(f"    - {note}")

    # Artigos com retry
    retry_articles = [a for a in metrics.articles if a.retry_count > 0]
    if retry_articles:
        lines.append("\n" + "-" * 70)
        lines.append("ARTIGOS COM RETRY")
        lines.append("-" * 70)
        for article in retry_articles:
            lines.append(
                f"  {article.article_id}: {article.retry_count} retry(s)"
            )

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def create_collector_from_extraction_result(
    result,  # ArticleExtractionResult
    document_id: str = "",
    tipo_documento: str = "",
    numero: str = "",
    ano: int = 0,
) -> MetricsCollector:
    """
    Cria um MetricsCollector a partir de um ArticleExtractionResult.

    Args:
        result: ArticleExtractionResult do ArticleOrchestrator
        document_id: ID do documento
        tipo_documento: Tipo (IN, LEI, DECRETO)
        numero: Número do documento
        ano: Ano

    Returns:
        MetricsCollector populado
    """
    collector = MetricsCollector(ingestion_id=document_id)
    collector.set_document_info(
        document_id=document_id,
        tipo_documento=tipo_documento,
        numero=numero,
        ano=ano,
    )

    for chunk in result.chunks:
        status = "valid"
        if hasattr(chunk, 'status'):
            status = chunk.status.value if hasattr(chunk.status, 'value') else str(chunk.status)

        collector.record_article_metrics(
            article_id=chunk.article_id,
            article_number=getattr(chunk, 'article_number', ''),
            parser_paragrafos=getattr(chunk, 'parser_paragrafos_count', 0),
            llm_paragrafos=getattr(chunk, 'llm_paragrafos_count', 0),
            parser_incisos=getattr(chunk, 'parser_incisos_count', 0),
            llm_incisos=getattr(chunk, 'llm_incisos_count', 0),
            status=status,
            validation_notes=getattr(chunk, 'validation_notes', []),
            retry_count=getattr(chunk, 'retry_count', 0),
        )

    return collector
