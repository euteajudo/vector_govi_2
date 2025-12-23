"""
HyDEExpander - Hypothetical Document Embeddings para query expansion.

Tecnica que gera documentos hipoteticos que responderiam a query,
e usa os embeddings desses documentos para melhorar a busca.

Referencia: https://arxiv.org/abs/2212.10496
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from llm.vllm_client import VLLMClient, LLMConfig


# =============================================================================
# PROMPTS PARA HYDE
# =============================================================================

HYDE_SYSTEM_PROMPT = """Voce e um especialista em direito administrativo brasileiro.
Sua tarefa e gerar trechos de documentos legais que responderiam a pergunta do usuario.

REGRAS:
1. Gere trechos curtos no estilo de artigos de lei ou instrucao normativa
2. Use linguagem juridica formal brasileira
3. Os trechos devem ser especificos e tecnicos
4. Responda APENAS com JSON valido"""


HYDE_USER_PROMPT = """Dada a pergunta abaixo, gere {n_docs} trechos de documentos legais brasileiros
que seriam a resposta ideal para essa pergunta.

PERGUNTA: {query}

Gere {n_docs} trechos curtos (50-100 palavras cada) no estilo de artigos de lei,
decreto ou instrucao normativa que responderiam essa pergunta diretamente.

Responda em JSON:
{{
    "hypothetical_docs": [
        "Art. X ... [trecho 1]",
        "Art. Y ... [trecho 2]",
        "Art. Z ... [trecho 3]"
    ]
}}"""


# =============================================================================
# HYDE EXPANDER
# =============================================================================

@dataclass
class HyDEResult:
    """Resultado da expansao HyDE."""
    original_query: str
    hypothetical_docs: list[str]
    combined_dense: list[float]
    combined_sparse: dict[int, float]
    query_weight: float
    doc_weight: float


class HyDEExpander:
    """
    Hypothetical Document Embeddings para query expansion.

    Gera documentos hipoteticos usando LLM e combina seus embeddings
    com o embedding da query original para melhorar a busca.

    Uso:
        expander = HyDEExpander(llm_client, embedder)

        # Expansao basica
        result = expander.expand(query)

        # Usar embeddings combinados na busca
        search_dense = result.combined_dense
        search_sparse = result.combined_sparse
    """

    def __init__(
        self,
        llm_client: Optional[VLLMClient] = None,
        embedder = None,  # BGEM3Embedder
        n_hypothetical: int = 3,
        query_weight: float = 0.4,
        doc_weight: float = 0.6,
    ):
        """
        Inicializa o expander.

        Args:
            llm_client: Cliente LLM para gerar documentos
            embedder: Embedder BGE-M3 para gerar vetores
            n_hypothetical: Numero de documentos hipoteticos a gerar
            query_weight: Peso do embedding da query (0-1)
            doc_weight: Peso dos embeddings dos docs hipoteticos (0-1)
        """
        if llm_client:
            self.llm = llm_client
            self._owns_llm = False
        else:
            self.llm = VLLMClient(LLMConfig.for_enrichment())
            self._owns_llm = True

        self.embedder = embedder
        self.n_hypothetical = n_hypothetical
        self.query_weight = query_weight
        self.doc_weight = doc_weight

        # Normaliza pesos
        total = query_weight + doc_weight
        self.query_weight = query_weight / total
        self.doc_weight = doc_weight / total

    def close(self):
        """Fecha recursos."""
        if self._owns_llm and self.llm:
            self.llm.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def generate_hypothetical_docs(self, query: str, n_docs: int = None) -> list[str]:
        """
        Gera documentos hipoteticos para a query.

        Args:
            query: Pergunta do usuario
            n_docs: Numero de docs (default: self.n_hypothetical)

        Returns:
            Lista de trechos de documentos hipoteticos
        """
        n_docs = n_docs or self.n_hypothetical

        user_prompt = HYDE_USER_PROMPT.format(
            query=query,
            n_docs=n_docs,
        )

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.7,  # Um pouco de variacao para diversidade
        )

        # Parse resposta
        docs = self._parse_hyde_response(response)

        return docs

    def _parse_hyde_response(self, response: str) -> list[str]:
        """Parseia resposta do LLM."""
        import json
        import re

        response = response.strip()

        # Remove markdown code blocks
        if response.startswith("```"):
            response = re.sub(r"^```\w*\n?", "", response)
            response = re.sub(r"\n?```$", "", response)

        try:
            data = json.loads(response)
            return data.get("hypothetical_docs", [])
        except json.JSONDecodeError:
            # Tenta encontrar JSON
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get("hypothetical_docs", [])
                except:
                    pass

            # Fallback: tenta extrair textos entre aspas
            docs = re.findall(r'"([^"]{50,})"', response)
            if docs:
                return docs[:self.n_hypothetical]

            return []

    def expand(self, query: str, n_docs: int = None) -> HyDEResult:
        """
        Expande a query usando HyDE.

        Gera documentos hipoteticos, calcula embeddings e combina
        com o embedding da query original.

        Args:
            query: Pergunta do usuario
            n_docs: Numero de docs hipoteticos

        Returns:
            HyDEResult com embeddings combinados
        """
        if not self.embedder:
            raise ValueError("Embedder nao configurado")

        n_docs = n_docs or self.n_hypothetical

        # 1. Gera documentos hipoteticos
        hypothetical_docs = self.generate_hypothetical_docs(query, n_docs)

        if not hypothetical_docs:
            # Fallback: usa apenas a query
            result = self.embedder.encode_hybrid([query])
            return HyDEResult(
                original_query=query,
                hypothetical_docs=[],
                combined_dense=result['dense'][0],
                combined_sparse=result['sparse'][0],
                query_weight=1.0,
                doc_weight=0.0,
            )

        # 2. Gera embeddings da query e dos docs
        all_texts = [query] + hypothetical_docs
        embeddings = self.embedder.encode_hybrid(all_texts)

        query_dense = np.array(embeddings['dense'][0])
        query_sparse = embeddings['sparse'][0]

        doc_denses = [np.array(e) for e in embeddings['dense'][1:]]
        doc_sparses = embeddings['sparse'][1:]

        # 3. Combina embeddings dense (media ponderada)
        doc_dense_avg = np.mean(doc_denses, axis=0)
        combined_dense = (
            self.query_weight * query_dense +
            self.doc_weight * doc_dense_avg
        )

        # 4. Combina embeddings sparse (uniao com pesos)
        combined_sparse = self._combine_sparse(
            query_sparse, doc_sparses,
            self.query_weight, self.doc_weight
        )

        return HyDEResult(
            original_query=query,
            hypothetical_docs=hypothetical_docs,
            combined_dense=combined_dense.tolist(),
            combined_sparse=combined_sparse,
            query_weight=self.query_weight,
            doc_weight=self.doc_weight,
        )

    def _combine_sparse(
        self,
        query_sparse: dict,
        doc_sparses: list[dict],
        query_weight: float,
        doc_weight: float,
    ) -> dict:
        """Combina embeddings sparse."""
        # Coleta todos os indices
        all_indices = set(query_sparse.keys())
        for ds in doc_sparses:
            all_indices.update(ds.keys())

        # Combina valores
        combined = {}
        doc_weight_each = doc_weight / len(doc_sparses) if doc_sparses else 0

        for idx in all_indices:
            value = query_weight * query_sparse.get(idx, 0)
            for ds in doc_sparses:
                value += doc_weight_each * ds.get(idx, 0)
            if value > 0:
                combined[idx] = value

        return combined

    def get_expanded_embedding(self, query: str) -> dict:
        """
        Retorna embeddings expandidos no formato esperado pelo HybridSearcher.

        Args:
            query: Pergunta do usuario

        Returns:
            Dict com 'dense' e 'sparse' prontos para busca
        """
        result = self.expand(query)
        return {
            'dense': result.combined_dense,
            'sparse': result.combined_sparse,
        }
