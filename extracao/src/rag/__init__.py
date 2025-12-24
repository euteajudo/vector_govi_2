"""
RAG Module - Retrieval-Augmented Generation para documentos legais.

Expoe modelos e utilitarios para:
- Answer-JSON estruturado para front-end
- Calculo de confianca
- Formatacao de citacoes
- Geracao de respostas RAG

Uso:
    from rag import AnswerGenerator, generate_answer

    # Geracao simples
    response = generate_answer("Quando o ETP pode ser dispensado?")
    print(response.answer)
    for citation in response.citations:
        print(f"  - {citation.text}")

    # Com configuracao
    generator = AnswerGenerator()
    response = generator.generate("Qual o prazo para pesquisa de precos?")
"""

from .answer_models import (
    AnswerResponse,
    AnswerMetadata,
    QueryRequest,
    # Nota: Citation importada de citation_formatter, nao de answer_models
)

from .citation_formatter import (
    Citation,
    CitationFormatter,
    format_citation,
    format_citation_from_hit,
)

from .answer_generator import (
    AnswerGenerator,
    GenerationConfig,
    RAGContext,
    generate_answer,
)

__all__ = [
    # Answer models
    "AnswerResponse",
    "AnswerMetadata",
    "QueryRequest",
    # Citation formatter
    "Citation",
    "CitationFormatter",
    "format_citation",
    "format_citation_from_hit",
    # Answer generator
    "AnswerGenerator",
    "GenerationConfig",
    "RAGContext",
    "generate_answer",
]
