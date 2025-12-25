"""
Answer Models - Modelos de resposta estruturada para o front-end.

Define o formato Answer-JSON que o front-end espera:
{
    "answer": "Texto da resposta...",
    "citations": [
        {"span_id": "ART-005", "text": "...", "relevance": 0.95},
        {"span_id": "INC-005-II", "text": "...", "relevance": 0.87}
    ],
    "confidence": 0.92,
    "sources": [
        {"document_id": "IN-65-2021", "title": "IN SEGES 65/2021"}
    ],
    "metadata": {
        "model": "Qwen/Qwen3-8B-AWQ",
        "latency_ms": 1234,
        "tokens_used": 456
    }
}
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Citation:
    """Uma citação específica na resposta."""

    span_id: str            # Ex: "ART-005", "INC-005-II"
    chunk_id: str           # Ex: "IN-65-2021#ART-005"
    text: str               # Texto do span citado
    relevance: float        # Score de relevância (0-1)

    # Localização visual (opcional)
    page: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None

    def to_dict(self) -> dict:
        result = {
            "span_id": self.span_id,
            "chunk_id": self.chunk_id,
            "text": self.text[:500] + "..." if len(self.text) > 500 else self.text,
            "relevance": round(self.relevance, 3),
        }
        if self.page is not None:
            result["location"] = {
                "page": self.page,
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            }
        return result


@dataclass
class Source:
    """Documento fonte citado."""

    document_id: str        # Ex: "IN-65-2021"
    title: str              # Ex: "IN SEGES Nº 65/2021"
    tipo_documento: str     # Ex: "INSTRUCAO NORMATIVA"

    # Metadados opcionais
    numero: str = ""
    ano: int = 0
    orgao: str = ""
    url: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "tipo_documento": self.tipo_documento,
            "numero": self.numero,
            "ano": self.ano,
            "orgao": self.orgao,
            "url": self.url,
        }


@dataclass
class AnswerMetadata:
    """Metadados da resposta para debugging/monitoramento."""

    model: str = ""
    latency_ms: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    chunks_retrieved: int = 0
    chunks_used: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Cache semântico
    from_cache: bool = False
    cache_similarity: float = 0.0

    def to_dict(self) -> dict:
        result = {
            "model": self.model,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_prompt + self.tokens_completion,
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "chunks_retrieved": self.chunks_retrieved,
            "chunks_used": self.chunks_used,
            "timestamp": self.timestamp,
        }
        if self.from_cache:
            result["from_cache"] = True
            result["cache_similarity"] = round(self.cache_similarity, 4)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "AnswerMetadata":
        return cls(
            model=data.get("model", ""),
            latency_ms=data.get("latency_ms", 0),
            retrieval_ms=data.get("retrieval_ms", 0),
            generation_ms=data.get("generation_ms", 0),
            chunks_retrieved=data.get("chunks_retrieved", 0),
            chunks_used=data.get("chunks_used", 0),
            timestamp=data.get("timestamp", ""),
            from_cache=data.get("from_cache", False),
            cache_similarity=data.get("cache_similarity", 0.0),
        )


@dataclass
class AnswerResponse:
    """
    Resposta estruturada para o front-end.

    Formato JSON-API compativel para facil integracao.
    """

    # Status
    success: bool = True
    error: Optional[str] = None

    # Query original
    query: str = ""

    # Conteudo principal
    answer: str = ""        # Texto da resposta
    confidence: float = 0.0 # Confianca geral (0-1)

    # Citacoes e fontes (aceita qualquer objeto com to_dict())
    citations: list = field(default_factory=list)
    sources: list = field(default_factory=list)

    # Metadados
    metadata: AnswerMetadata = field(default_factory=AnswerMetadata)

    def to_dict(self) -> dict:
        """Converte para dicionario JSON-serializavel."""
        citations_list = []
        for c in self.citations:
            if hasattr(c, 'to_dict'):
                citations_list.append(c.to_dict())
            elif isinstance(c, dict):
                citations_list.append(c)

        sources_list = []
        for s in self.sources:
            if hasattr(s, 'to_dict'):
                sources_list.append(s.to_dict())
            elif isinstance(s, dict):
                sources_list.append(s)

        return {
            "success": self.success,
            "query": self.query,
            "data": {
                "answer": self.answer,
                "confidence": round(self.confidence, 3),
                "citations": citations_list,
                "sources": sources_list,
            },
            "metadata": self.metadata.to_dict(),
            "error": self.error,
        }

    def to_json(self) -> str:
        """Serializa para JSON."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "AnswerResponse":
        """Reconstrói AnswerResponse a partir de dicionário (ex: do cache)."""
        from .citation_formatter import Citation

        # Extrai data aninhado se existir
        inner_data = data.get("data", data)

        # Reconstrói citações
        citations = []
        for c in inner_data.get("citations", []):
            if isinstance(c, dict):
                citations.append(Citation(
                    span_id=c.get("span_id", ""),
                    chunk_id=c.get("chunk_id", ""),
                    text=c.get("text", ""),
                    relevance=c.get("relevance", 0.0),
                    article=c.get("article", ""),
                    document_type=c.get("document_type", ""),
                    document_number=c.get("document_number", ""),
                    year=c.get("year", 0),
                    device=c.get("device", ""),
                    device_number=c.get("device_number", ""),
                ))

        # Reconstrói metadata
        metadata_dict = data.get("metadata", {})
        metadata = AnswerMetadata.from_dict(metadata_dict)

        return cls(
            success=data.get("success", True),
            error=data.get("error"),
            query=data.get("query", ""),
            answer=inner_data.get("answer", ""),
            confidence=inner_data.get("confidence", 0.0),
            citations=citations,
            sources=inner_data.get("sources", []),
            metadata=metadata,
        )


@dataclass
class QueryRequest:
    """Request de query do front-end."""

    query: str                          # Pergunta do usuário
    document_ids: list[str] = field(default_factory=list)  # Filtro por documentos
    top_k: int = 5                      # Quantidade de chunks a recuperar
    rerank: bool = True                 # Se deve usar reranker
    include_context: bool = True        # Se deve incluir contexto pai

    @classmethod
    def from_dict(cls, data: dict) -> "QueryRequest":
        return cls(
            query=data.get("query", ""),
            document_ids=data.get("document_ids", []),
            top_k=data.get("top_k", 5),
            rerank=data.get("rerank", True),
            include_context=data.get("include_context", True),
        )


def calculate_confidence(citations: list[Citation]) -> float:
    """
    Calcula confiança geral baseada nas citações.

    Fórmula:
    - Base: média ponderada das relevâncias das citações
    - Penalidade: se menos de 2 citações, reduz 20%
    - Bonus: se top citação > 0.9, adiciona 5%
    """
    if not citations:
        return 0.0

    # Média ponderada (citações mais relevantes pesam mais)
    weights = [c.relevance ** 2 for c in citations]
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(c.relevance * w for c, w in zip(citations, weights))
    base_confidence = weighted_sum / total_weight

    # Penalidade para poucas citações
    if len(citations) < 2:
        base_confidence *= 0.8

    # Bonus para citação muito relevante
    if citations and citations[0].relevance > 0.9:
        base_confidence = min(1.0, base_confidence * 1.05)

    return min(1.0, max(0.0, base_confidence))


def build_answer_response(
    answer: str,
    retrieved_chunks: list[dict],
    model: str = "",
    latency_ms: int = 0,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
) -> AnswerResponse:
    """
    Constrói AnswerResponse a partir dos chunks recuperados.

    Args:
        answer: Texto da resposta do LLM
        retrieved_chunks: Lista de chunks com score de relevância
        model: Nome do modelo usado
        latency_ms: Latência total em ms

    Returns:
        AnswerResponse formatado para o front-end
    """
    citations = []
    sources_map = {}

    for chunk in retrieved_chunks:
        # Cria citação
        citation = Citation(
            span_id=chunk.get("span_id", ""),
            chunk_id=chunk.get("chunk_id", ""),
            text=chunk.get("text", ""),
            relevance=chunk.get("score", 0.0),
            page=chunk.get("page"),
        )
        citations.append(citation)

        # Agrupa fontes
        doc_id = chunk.get("document_id", "")
        if doc_id and doc_id not in sources_map:
            sources_map[doc_id] = Source(
                document_id=doc_id,
                title=chunk.get("document_title", doc_id),
                tipo_documento=chunk.get("tipo_documento", ""),
                numero=chunk.get("numero", ""),
                ano=chunk.get("ano", 0),
            )

    # Calcula confiança
    confidence = calculate_confidence(citations)

    # Monta metadata
    metadata = AnswerMetadata(
        model=model,
        latency_ms=latency_ms,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        chunks_retrieved=len(retrieved_chunks),
        chunks_used=len(citations),
    )

    return AnswerResponse(
        answer=answer,
        confidence=confidence,
        citations=citations,
        sources=list(sources_map.values()),
        metadata=metadata,
    )
