"""
Modulo de integracao com Milvus.

Define schemas e funcoes para operacoes no Milvus.
"""

from .schema import create_legal_chunks_schema, COLLECTION_NAME

__all__ = ["create_legal_chunks_schema", "COLLECTION_NAME"]
