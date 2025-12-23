"""
Módulo de chunking para documentos legais brasileiros.

Transforma LegalDocument (JSON estruturado) em chunks prontos para Milvus,
com enriquecimento via LLM (context_header, thesis_text, synthetic_questions).

Suporta parent-child retrieval:
- Chunk pai (ARTICLE): texto completo do artigo
- Chunks filhos (PARAGRAPH/INCISO): dispositivos com parent_chunk_id
"""

from .chunk_models import LegalChunk, ChunkLevel, ThesisType, ChunkingResult
from .law_chunker import LawChunker
from .chunk_materializer import (
    ChunkMaterializer,
    MaterializedChunk,
    MaterializationResult,
    ChunkMetadata,
    DeviceType,
)

__all__ = [
    # Modelos
    "LegalChunk",
    "ChunkLevel",
    "ThesisType",
    "ChunkingResult",
    # Chunker original
    "LawChunker",
    # Parent-child materializer
    "ChunkMaterializer",
    "MaterializedChunk",
    "MaterializationResult",
    "ChunkMetadata",
    "DeviceType",
]
