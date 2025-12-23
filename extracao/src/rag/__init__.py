"""
RAG Module - Retrieval-Augmented Generation para documentos legais.

Expõe modelos e utilitários para:
- Answer-JSON estruturado para front-end
- Cálculo de confiança
- Formatação de citações
"""

from .answer_models import (
    AnswerResponse,
    Citation,
    Source,
    AnswerMetadata,
    QueryRequest,
    calculate_confidence,
    build_answer_response,
)

__all__ = [
    "AnswerResponse",
    "Citation",
    "Source",
    "AnswerMetadata",
    "QueryRequest",
    "calculate_confidence",
    "build_answer_response",
]
