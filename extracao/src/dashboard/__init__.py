"""
Dashboard de Ingestão - Métricas e monitoramento do pipeline.

Fornece:
- Métricas de cobertura (parágrafos, incisos por artigo)
- Métricas de latência (tempo por fase, por artigo)
- Métricas de custo (tokens prompt, completion)
- Relatórios estruturados para monitoramento
"""

from .ingestion_metrics import (
    IngestionMetrics,
    ArticleMetrics,
    DocumentMetrics,
    PhaseMetrics,
    MetricsCollector,
    generate_dashboard_report,
)

__all__ = [
    "IngestionMetrics",
    "ArticleMetrics",
    "DocumentMetrics",
    "PhaseMetrics",
    "MetricsCollector",
    "generate_dashboard_report",
]
