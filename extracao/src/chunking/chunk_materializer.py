"""
Chunk Materializer - Transforma ArticleChunk em chunks indexáveis com parent-child.

Este módulo materializa a hierarquia extraída pelo ArticleOrchestrator em chunks
prontos para indexação no Milvus, com suporte a parent-child retrieval.

Estrutura de chunks gerados:
- Chunk pai (ARTICLE): contém texto completo do artigo
- Chunks filhos (PARAGRAPH/INCISO): texto do dispositivo com parent_chunk_id

Na busca:
1. Recupera chunks filhos por similaridade
2. Agrega chunks pai para contexto
3. Passa contexto expandido para o LLM
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

from .chunk_models import LegalChunk, ChunkLevel


class DeviceType(str, Enum):
    """Tipo de dispositivo legal."""
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    INCISO = "inciso"
    ALINEA = "alinea"


@dataclass
class ChunkMetadata:
    """Metadados de proveniência e versão."""

    # Versões
    schema_version: str = "1.0.0"
    extractor_version: str = "1.0.0"

    # Timestamps
    ingestion_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Proveniência do documento
    document_hash: str = ""  # SHA-256 do PDF
    pdf_path: Optional[str] = None

    # Versão do documento (vigência)
    valid_from: Optional[str] = None  # Data início vigência
    valid_to: Optional[str] = None    # Data fim vigência (None = vigente)

    # Citações visuais (coordenadas no PDF)
    page_spans: dict = field(default_factory=dict)  # {span_id: {page, x, y, w, h}}

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "extractor_version": self.extractor_version,
            "ingestion_timestamp": self.ingestion_timestamp,
            "document_hash": self.document_hash,
            "pdf_path": self.pdf_path,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "page_spans": self.page_spans,
        }


@dataclass
class MaterializedChunk:
    """Chunk materializado com suporte a parent-child."""

    # IDs
    chunk_id: str           # Ex: "IN-65-2021#ART-005" ou "IN-65-2021#PAR-005-1"
    parent_chunk_id: str    # Ex: "" para artigos, "IN-65-2021#ART-005" para filhos
    span_id: str            # Ex: "ART-005", "PAR-005-1", "INC-005-I"

    # Tipo
    device_type: DeviceType
    chunk_level: ChunkLevel

    # Conteúdo
    text: str
    enriched_text: str = ""

    # Contexto
    context_header: str = ""
    thesis_text: str = ""
    thesis_type: str = "disposicao"
    synthetic_questions: str = ""

    # Hierarquia legal
    document_id: str = ""
    tipo_documento: str = ""
    numero: str = ""
    ano: int = 0
    article_number: str = ""

    # Citations (lista de span_ids que compõem este chunk)
    citations: list[str] = field(default_factory=list)

    # Metadados
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Embeddings (preenchidos depois)
    dense_vector: Optional[list[float]] = None
    sparse_vector: Optional[dict[int, float]] = None

    def to_milvus_dict(self) -> dict:
        """Converte para formato Milvus com campos dinâmicos."""
        return {
            # Campos principais
            "chunk_id": self.chunk_id,
            "text": self.text,
            "enriched_text": self.enriched_text,

            # Embeddings
            "dense_vector": self.dense_vector or [],
            "sparse_vector": self.sparse_vector or {},

            # Campos para filtro
            "document_id": self.document_id,
            "tipo_documento": self.tipo_documento,
            "numero": self.numero,
            "ano": self.ano,
            "article_number": self.article_number,

            # Campos dinâmicos (parent-child)
            "parent_chunk_id": self.parent_chunk_id,
            "span_id": self.span_id,
            "device_type": self.device_type.value,
            "chunk_level": self.chunk_level.value,
            "citations": self.citations,

            # Enriquecimento
            "context_header": self.context_header,
            "thesis_text": self.thesis_text,
            "thesis_type": self.thesis_type,
            "synthetic_questions": self.synthetic_questions,

            # Proveniência
            **self.metadata.to_dict(),
        }


class ChunkMaterializer:
    """
    Materializa ArticleChunk em chunks indexáveis com parent-child.

    Gera:
    - 1 chunk ARTICLE (pai) com texto completo
    - N chunks PARAGRAPH (filhos) para cada parágrafo
    - M chunks INCISO (filhos) para cada inciso
    """

    def __init__(
        self,
        document_id: str,
        tipo_documento: str = "",
        numero: str = "",
        ano: int = 0,
        metadata: Optional[ChunkMetadata] = None
    ):
        self.document_id = document_id
        self.tipo_documento = tipo_documento
        self.numero = numero
        self.ano = ano
        self.metadata = metadata or ChunkMetadata()

    def materialize_article(
        self,
        article_chunk,  # ArticleChunk do article_orchestrator
        parsed_doc,     # ParsedDocument para reconstruir texto
        include_children: bool = True
    ) -> list[MaterializedChunk]:
        """
        Materializa um ArticleChunk em chunks pai e filhos.

        Args:
            article_chunk: ArticleChunk extraído
            parsed_doc: Documento parseado (para reconstruir texto dos filhos)
            include_children: Se True, gera chunks para PAR/INC também

        Returns:
            Lista de MaterializedChunk (1 pai + N filhos)
        """
        chunks = []

        # 1. Chunk pai (ARTICLE)
        parent_chunk_id = f"{self.document_id}#{article_chunk.article_id}"

        parent = MaterializedChunk(
            chunk_id=parent_chunk_id,
            parent_chunk_id="",  # Artigo não tem pai
            span_id=article_chunk.article_id,
            device_type=DeviceType.ARTICLE,
            chunk_level=ChunkLevel.ARTICLE,
            text=article_chunk.text,
            document_id=self.document_id,
            tipo_documento=self.tipo_documento,
            numero=self.numero,
            ano=self.ano,
            article_number=article_chunk.article_number,
            citations=article_chunk.citations,
            metadata=self.metadata,
        )
        chunks.append(parent)

        if not include_children:
            return chunks

        # 2. Chunks filhos (PARAGRAPH)
        for par_id in article_chunk.paragrafo_ids:
            par_span = parsed_doc.get_span(par_id)
            if not par_span:
                continue

            child_chunk_id = f"{self.document_id}#{par_id}"

            child = MaterializedChunk(
                chunk_id=child_chunk_id,
                parent_chunk_id=parent_chunk_id,
                span_id=par_id,
                device_type=DeviceType.PARAGRAPH,
                chunk_level=ChunkLevel.DEVICE,
                text=par_span.text,
                document_id=self.document_id,
                tipo_documento=self.tipo_documento,
                numero=self.numero,
                ano=self.ano,
                article_number=article_chunk.article_number,
                citations=[par_id],
                metadata=self.metadata,
            )
            chunks.append(child)

        # 3. Chunks filhos (INCISO)
        for inc_id in article_chunk.inciso_ids:
            inc_span = parsed_doc.get_span(inc_id)
            if not inc_span:
                continue

            child_chunk_id = f"{self.document_id}#{inc_id}"

            # Reconstrói texto do inciso com alíneas
            inc_text = self._reconstruct_inciso_text(inc_id, parsed_doc)
            inc_citations = self._get_inciso_citations(inc_id, parsed_doc)

            child = MaterializedChunk(
                chunk_id=child_chunk_id,
                parent_chunk_id=parent_chunk_id,
                span_id=inc_id,
                device_type=DeviceType.INCISO,
                chunk_level=ChunkLevel.DEVICE,
                text=inc_text,
                document_id=self.document_id,
                tipo_documento=self.tipo_documento,
                numero=self.numero,
                ano=self.ano,
                article_number=article_chunk.article_number,
                citations=inc_citations,
                metadata=self.metadata,
            )
            chunks.append(child)

        return chunks

    def _reconstruct_inciso_text(self, inc_id: str, parsed_doc) -> str:
        """Reconstrói texto do inciso incluindo alíneas."""
        inc_span = parsed_doc.get_span(inc_id)
        if not inc_span:
            return ""

        lines = [inc_span.text]

        # Adiciona alíneas
        for child in parsed_doc.get_children(inc_id):
            lines.append(f"  {child.text}")

        return "\n".join(lines)

    def _get_inciso_citations(self, inc_id: str, parsed_doc) -> list[str]:
        """Obtém lista de citations para o inciso (inclui alíneas)."""
        citations = [inc_id]

        for child in parsed_doc.get_children(inc_id):
            citations.append(child.span_id)

        return citations

    def materialize_all(
        self,
        article_chunks: list,  # list[ArticleChunk]
        parsed_doc,
        include_children: bool = True
    ) -> list[MaterializedChunk]:
        """
        Materializa todos os ArticleChunks.

        Returns:
            Lista completa de MaterializedChunk
        """
        all_chunks = []

        for article_chunk in article_chunks:
            chunks = self.materialize_article(
                article_chunk, parsed_doc, include_children
            )
            all_chunks.extend(chunks)

        return all_chunks


@dataclass
class MaterializationResult:
    """Resultado da materialização."""

    chunks: list[MaterializedChunk] = field(default_factory=list)

    # Estatísticas
    total_chunks: int = 0
    article_chunks: int = 0
    paragraph_chunks: int = 0
    inciso_chunks: int = 0

    # Metadados
    document_id: str = ""
    ingestion_timestamp: str = ""

    def summary(self) -> dict:
        return {
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "breakdown": {
                "articles": self.article_chunks,
                "paragraphs": self.paragraph_chunks,
                "incisos": self.inciso_chunks,
            },
            "ingestion_timestamp": self.ingestion_timestamp,
        }
