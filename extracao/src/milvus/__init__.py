"""
Modulo de integracao com Milvus.

Define schemas e funcoes para operacoes no Milvus.

Schemas disponíveis:
- v2 (leis_v2): Schema original com embeddings BGE-M3
- v3 (leis_v3): Novo schema com parent-child + citations + proveniência
"""

from .schema import create_legal_chunks_schema, COLLECTION_NAME, create_indexes
from .schema_v3 import (
    create_legal_chunks_schema_v3,
    create_indexes_v3,
    COLLECTION_NAME_V3,
)

__all__ = [
    # Schema v2 (legado)
    "create_legal_chunks_schema",
    "COLLECTION_NAME",
    "create_indexes",
    # Schema v3 (atual)
    "create_legal_chunks_schema_v3",
    "create_indexes_v3",
    "COLLECTION_NAME_V3",
]
