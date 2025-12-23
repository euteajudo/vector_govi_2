"""
Módulo de chunking para documentos legais brasileiros.

Transforma LegalDocument (JSON estruturado) em chunks prontos para Milvus,
com enriquecimento via LLM (context_header, thesis_text, synthetic_questions).
"""

from .chunk_models import LegalChunk, ChunkLevel, ThesisType
from .law_chunker import LawChunker

__all__ = [
    "LegalChunk",
    "ChunkLevel",
    "ThesisType",
    "LawChunker",
]
