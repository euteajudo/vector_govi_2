"""
Gerador de respostas RAG para documentos legais.

Combina retrieval (HybridSearcher) com generation (vLLM) para
produzir respostas completas com citacoes.

Uso:
    from rag import AnswerGenerator

    generator = AnswerGenerator()
    response = generator.generate(
        query="Quando o ETP pode ser dispensado?",
        top_k=5,
    )

    print(response.answer)
    for citation in response.citations:
        print(f"  - {citation.text}")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .citation_formatter import CitationFormatter, Citation, format_citation_from_hit
from .answer_models import AnswerResponse, AnswerMetadata, QueryRequest

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuracao para geracao de resposta."""

    # Modelo LLM
    model: str = "Qwen/Qwen3-8B-AWQ"
    temperature: float = 0.3
    max_tokens: int = 2048

    # vLLM
    base_url: str = "http://localhost:8000/v1"
    timeout: float = 120.0

    # Retrieval
    top_k: int = 5
    use_hyde: bool = True
    use_reranker: bool = True

    # Collection
    collection_name: str = "leis_v3"

    @classmethod
    def default(cls) -> "GenerationConfig":
        """Configuracao padrao."""
        return cls()

    @classmethod
    def fast(cls) -> "GenerationConfig":
        """Configuracao para baixa latencia."""
        return cls(
            temperature=0.1,
            max_tokens=1024,
            top_k=3,
            use_hyde=False,
            use_reranker=False,
        )


@dataclass
class RAGContext:
    """Contexto montado para o LLM."""

    chunks_text: str
    citations: list[Citation]
    chunk_ids: list[str]
    total_chunks: int


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """Voce e um assistente especializado em legislacao brasileira de licitacoes e contratacoes publicas.

Sua funcao e responder perguntas com base EXCLUSIVAMENTE nos trechos de documentos legais fornecidos no contexto.

REGRAS IMPORTANTES:
1. Responda APENAS com base no contexto fornecido
2. Se a informacao nao estiver no contexto, diga "Nao encontrei essa informacao nos documentos consultados"
3. SEMPRE cite os artigos/paragrafos que embasam sua resposta usando o formato [Art. X] ou [Art. X, Par. Y]
4. Seja objetivo e preciso - respostas juridicas exigem exatidao
5. Use linguagem formal apropriada para documentos legais
6. Se houver multiplas disposicoes relevantes, mencione todas

FORMATO DA RESPOSTA:
- Responda de forma clara e estruturada
- Inclua as citacoes entre colchetes no corpo do texto
- Se necessario, organize em topicos"""

USER_PROMPT_TEMPLATE = """CONTEXTO (Trechos de documentos legais):
{context}

---

PERGUNTA: {query}

Com base EXCLUSIVAMENTE no contexto acima, responda a pergunta citando os artigos relevantes."""


# =============================================================================
# ANSWER GENERATOR
# =============================================================================

class AnswerGenerator:
    """
    Gerador de respostas RAG para perguntas juridicas.

    Combina HybridSearcher para retrieval com vLLM para generation,
    produzindo respostas com citacoes formatadas.
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        searcher=None,
        llm_client=None,
    ):
        """
        Inicializa o gerador.

        Args:
            config: Configuracao de geracao
            searcher: HybridSearcher (cria novo se nao fornecido)
            llm_client: VLLMClient (cria novo se nao fornecido)
        """
        self.config = config or GenerationConfig.default()
        self._searcher = searcher
        self._llm_client = llm_client
        self._citation_formatter = CitationFormatter()

    # =========================================================================
    # LAZY LOADING
    # =========================================================================

    @property
    def searcher(self):
        """Carrega searcher sob demanda."""
        if self._searcher is None:
            from search import HybridSearcher, SearchConfig

            logger.info("Carregando HybridSearcher...")
            search_config = SearchConfig.default()
            search_config.collection_name = self.config.collection_name
            search_config.use_hyde = self.config.use_hyde

            self._searcher = HybridSearcher(config=search_config)
        return self._searcher

    @property
    def llm_client(self):
        """Carrega LLM client sob demanda."""
        if self._llm_client is None:
            from llm.vllm_client import VLLMClient, LLMConfig

            logger.info("Carregando vLLM client...")
            llm_config = LLMConfig(
                base_url=self.config.base_url,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )
            self._llm_client = VLLMClient(config=llm_config)
        return self._llm_client

    # =========================================================================
    # GERACAO PRINCIPAL
    # =========================================================================

    def generate(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters=None,
    ) -> AnswerResponse:
        """
        Gera resposta completa para a pergunta.

        Args:
            query: Pergunta do usuario
            top_k: Numero de chunks para contexto
            filters: Filtros de busca (SearchFilter)

        Returns:
            AnswerResponse com resposta e citacoes
        """
        start_time = time.perf_counter()
        top_k = top_k or self.config.top_k

        # 1. Retrieval
        logger.info(f"Buscando contexto para: {query[:50]}...")
        retrieval_start = time.perf_counter()

        search_result = self.searcher.search(
            query=query,
            top_k=top_k,
            filters=filters,
            use_reranker=self.config.use_reranker,
        )
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        if not search_result.hits:
            return self._empty_response(query, retrieval_time)

        # 2. Montar contexto
        context = self._build_context(search_result.hits)

        # 3. Gerar resposta
        logger.info(f"Gerando resposta com {len(context.citations)} chunks...")
        generation_start = time.perf_counter()

        answer_text = self._generate_answer(query, context)
        generation_time = (time.perf_counter() - generation_start) * 1000

        # 4. Calcular confianca
        confidence = self._calculate_confidence(search_result.hits)

        # 5. Extrair documentos fonte
        sources = self._extract_sources(search_result.hits)

        # 6. Montar resposta
        total_time = (time.perf_counter() - start_time) * 1000

        metadata = AnswerMetadata(
            model=self.config.model,
            latency_ms=int(total_time),
            retrieval_ms=int(retrieval_time),
            generation_ms=int(generation_time),
            chunks_retrieved=len(search_result.hits),
            chunks_used=len(context.citations),
        )

        return AnswerResponse(
            success=True,
            query=query,
            answer=answer_text,
            confidence=confidence,
            citations=context.citations,
            sources=sources,
            metadata=metadata,
        )

    # =========================================================================
    # METODOS AUXILIARES
    # =========================================================================

    def _build_context(self, hits) -> RAGContext:
        """
        Monta contexto formatado para o LLM.

        Args:
            hits: Lista de SearchHit

        Returns:
            RAGContext com texto e citacoes
        """
        chunks_parts = []
        citations = []
        chunk_ids = []

        for i, hit in enumerate(hits, 1):
            # Formata citacao
            citation = format_citation_from_hit(hit)
            citations.append(citation)
            chunk_ids.append(hit.chunk_id)

            # Monta bloco de contexto
            header = f"[{i}] {citation.text}"
            if hit.device_type and hit.device_type != "article":
                header += f" ({hit.device_type})"

            chunk_block = f"{header}\n{hit.text}"
            chunks_parts.append(chunk_block)

        chunks_text = "\n\n---\n\n".join(chunks_parts)

        return RAGContext(
            chunks_text=chunks_text,
            citations=citations,
            chunk_ids=chunk_ids,
            total_chunks=len(hits),
        )

    def _generate_answer(self, query: str, context: RAGContext) -> str:
        """
        Gera resposta usando o LLM.

        Args:
            query: Pergunta
            context: Contexto montado

        Returns:
            Texto da resposta
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context.chunks_text,
            query=query,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm_client.chat(messages)

        # Remove /think tags se presentes (Qwen3)
        answer = response.strip()
        if "<think>" in answer:
            # Remove bloco de pensamento
            import re
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        return answer

    def _calculate_confidence(self, hits) -> float:
        """
        Calcula score de confianca baseado nos hits.

        Formula:
        - Base: media ponderada das relevancies (peso = relevancia^2)
        - Penalidade: se menos de 2 citacoes, reduz 20%
        - Bonus: se top citacao > 0.9, adiciona 5%
        """
        if not hits:
            return 0.0

        # Usa rerank_score se disponivel, senao final_score
        scores = []
        for hit in hits:
            if hit.rerank_score is not None and hit.rerank_score > 0:
                score = hit.rerank_score
            else:
                score = hit.final_score
            scores.append(score)

        # Media ponderada (pesos = score^2)
        weights = [s ** 2 for s in scores]
        total_weight = sum(weights)

        if total_weight == 0:
            return 0.0

        weighted_avg = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Ajustes
        confidence = weighted_avg

        # Penalidade por poucos hits
        if len(hits) < 2:
            confidence *= 0.8

        # Bonus por hit muito relevante
        if scores and scores[0] > 0.9:
            confidence = min(1.0, confidence + 0.05)

        return round(confidence, 3)

    def _extract_sources(self, hits) -> list[dict]:
        """
        Extrai documentos fonte unicos.

        Args:
            hits: Lista de SearchHit

        Returns:
            Lista de dicts com document_id, tipo, numero, ano
        """
        seen = set()
        sources = []

        for hit in hits:
            doc_id = hit.document_id
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                sources.append({
                    "document_id": doc_id,
                    "tipo_documento": hit.document_type,
                    "numero": hit.document_number,
                    "ano": hit.year,
                })

        return sources

    def _empty_response(self, query: str, retrieval_time: float) -> AnswerResponse:
        """Retorna resposta vazia quando nao ha resultados."""
        return AnswerResponse(
            success=True,
            query=query,
            answer="Nao encontrei informacoes relevantes nos documentos consultados para responder essa pergunta.",
            confidence=0.0,
            citations=[],
            sources=[],
            metadata=AnswerMetadata(
                model=self.config.model,
                latency_ms=int(retrieval_time),
                retrieval_ms=int(retrieval_time),
                generation_ms=0,
                chunks_retrieved=0,
                chunks_used=0,
            ),
        )

    def close(self):
        """Libera recursos."""
        if self._searcher:
            self._searcher.disconnect()
        if self._llm_client:
            self._llm_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# FUNCAO HELPER
# =============================================================================

def generate_answer(
    query: str,
    top_k: int = 5,
    config: Optional[GenerationConfig] = None,
) -> AnswerResponse:
    """
    Funcao helper para geracao rapida de resposta.

    Cria generator temporario e gera resposta.
    Para multiplas queries, use AnswerGenerator diretamente.

    Args:
        query: Pergunta do usuario
        top_k: Numero de chunks para contexto
        config: Configuracao opcional

    Returns:
        AnswerResponse com resposta e citacoes
    """
    with AnswerGenerator(config=config) as generator:
        return generator.generate(query, top_k=top_k)
