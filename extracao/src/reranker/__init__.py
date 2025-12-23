"""
Modulo de reranking para documentos legais.

Componentes:
- ColBERTReranker: Late interaction (MaxSim) via BGE-M3
- BGEReranker: Cross-encoder via bge-reranker-v2-m3

ColBERT vs Cross-Encoder:
- ColBERT: Mais rapido em batch, captura match exato de termos
- Cross-Encoder: Mais preciso em geral, mas mais lento
"""

from .colbert_reranker import ColBERTReranker

__all__ = ["ColBERTReranker"]
