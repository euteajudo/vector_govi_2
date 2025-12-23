"""
Modulo de enriquecimento de chunks com LLM.

Implementa Contextual Retrieval da Anthropic:
- context_header: Frase contextualizando o chunk
- thesis_text: Resumo/tese do dispositivo
- thesis_type: Classificacao do tipo de norma
- synthetic_questions: Perguntas que o chunk responde
"""

from .chunk_enricher import ChunkEnricher

__all__ = ["ChunkEnricher"]
