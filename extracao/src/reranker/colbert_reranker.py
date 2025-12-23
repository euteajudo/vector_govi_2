"""
ColBERT Reranker usando BGE-M3.

Usa late interaction (MaxSim) para precisao maxima em documentos juridicos.
O BGE-M3 ja suporta ColBERT nativamente via return_colbert_vecs=True.

ColBERT vs Cross-Encoder:
  Cross-Encoder (BGE-Reranker atual):
    [Query + Doc] -> Encoder -> 1 score

  ColBERT (MaxSim):
    Query -> Encoder -> [v1, v2, v3, ...]  (1 vetor por token)
    Doc   -> Encoder -> [v1, v2, v3, ...]  (1 vetor por token)

    Para cada token da query:
      -> Encontra max similarity com todos tokens do doc

    Score final = media dos max similarities

Vantagem: Captura match exato de termos juridicos como "interdependentes", "correlatas".

Uso:
    from reranker import ColBERTReranker

    reranker = ColBERTReranker()

    results = reranker.rerank(
        query="contratacoes interdependentes e correlatas",
        documents=["Doc 1...", "Doc 2...", "Doc 3..."],
        top_k=5
    )
    # results = [(2, 0.92), (0, 0.85), (1, 0.71)]  # (index, score)
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ColBERTConfig:
    """Configuracao do ColBERT Reranker."""

    model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True
    device: Optional[str] = None  # Auto-detect
    max_length: int = 8192
    batch_size: int = 32

    @classmethod
    def default(cls) -> "ColBERTConfig":
        return cls()

    @classmethod
    def fast(cls) -> "ColBERTConfig":
        """Configuracao otimizada para velocidade."""
        return cls(use_fp16=True, max_length=2048, batch_size=64)


class ColBERTReranker:
    """
    Reranker usando ColBERT MaxSim do BGE-M3.

    Usa late interaction para capturar match exato de termos,
    especialmente importante para vocabulario juridico.

    Attributes:
        config: Configuracao do modelo
        model: Modelo BGE-M3 carregado
        _initialized: Se o modelo foi carregado
    """

    def __init__(self, config: Optional[ColBERTConfig] = None):
        """
        Inicializa o ColBERT Reranker.

        O modelo e carregado de forma lazy (apenas quando chamado).

        Args:
            config: Configuracao do modelo. Usa default se nao fornecido.
        """
        self.config = config or ColBERTConfig.default()
        self.model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Carrega o modelo se ainda nao foi carregado (lazy loading)."""
        if self._initialized:
            return

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding nao instalado. "
                "Instale com: pip install FlagEmbedding"
            )

        logger.info(f"Carregando ColBERT Reranker ({self.config.model_name})...")

        self.model = BGEM3FlagModel(
            self.config.model_name,
            use_fp16=self.config.use_fp16,
            device=self.config.device,
        )

        self._initialized = True
        logger.info("ColBERT Reranker carregado com sucesso")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documentos usando ColBERT MaxSim.

        Args:
            query: Pergunta do usuario
            documents: Lista de textos dos documentos candidatos
            top_k: Quantos retornar (None = todos, ordenados)

        Returns:
            Lista de (doc_index, score) ordenada por score descendente
        """
        self._ensure_initialized()

        if not documents:
            return []

        if top_k is None:
            top_k = len(documents)

        # Gerar ColBERT embeddings (multi-vector por documento)
        query_output = self.model.encode(
            [query],
            batch_size=1,
            max_length=self.config.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )

        doc_output = self.model.encode(
            documents,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )

        # query_vecs shape: (n_query_tokens, 1024)
        query_vecs = query_output["colbert_vecs"][0]

        # Calcular MaxSim para cada documento
        scores = []
        for i, doc_vecs in enumerate(doc_output["colbert_vecs"]):
            score = self._maxsim(query_vecs, doc_vecs)
            scores.append((i, score))

        # Ordenar por score descendente
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def _maxsim(self, query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
        """
        Calcula MaxSim score entre query e documento.

        MaxSim: Para cada token da query, encontra a maxima similaridade
        com qualquer token do documento. Score final e a media dos maximos.

        Args:
            query_vecs: (n_query_tokens, embedding_dim)
            doc_vecs: (n_doc_tokens, embedding_dim)

        Returns:
            Score de similaridade (0-1)
        """
        # Normalizar para similaridade de cosseno
        q_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
        d_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)

        # Matriz de similaridade: (n_query_tokens, n_doc_tokens)
        sim_matrix = np.dot(q_norm, d_norm.T)

        # MaxSim: max por linha (cada token da query pega seu melhor match)
        max_sims = sim_matrix.max(axis=1)

        # Score final: media dos maximos
        return float(max_sims.mean())

    def rerank_with_details(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank com detalhes adicionais (para debug/analise).

        Returns:
            Lista de dicts com index, score e estatisticas
        """
        self._ensure_initialized()

        if not documents:
            return []

        if top_k is None:
            top_k = len(documents)

        query_output = self.model.encode(
            [query],
            batch_size=1,
            max_length=self.config.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )

        doc_output = self.model.encode(
            documents,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )

        query_vecs = query_output["colbert_vecs"][0]
        n_query_tokens = len(query_vecs)

        results = []
        for i, doc_vecs in enumerate(doc_output["colbert_vecs"]):
            # Normalizar
            q_norm = query_vecs / (
                np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8
            )
            d_norm = doc_vecs / (
                np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8
            )

            # Matriz de similaridade
            sim_matrix = np.dot(q_norm, d_norm.T)
            max_sims = sim_matrix.max(axis=1)

            results.append({
                "index": i,
                "score": float(max_sims.mean()),
                "n_query_tokens": n_query_tokens,
                "n_doc_tokens": len(doc_vecs),
                "min_token_score": float(max_sims.min()),
                "max_token_score": float(max_sims.max()),
                "std_token_score": float(max_sims.std()),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def compute_scores(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Calcula scores para todos os documentos (sem ordenar).

        Args:
            query: Pergunta do usuario
            documents: Lista de textos

        Returns:
            Lista de scores na mesma ordem dos documentos
        """
        self._ensure_initialized()

        if not documents:
            return []

        query_output = self.model.encode(
            [query],
            batch_size=1,
            max_length=self.config.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )

        doc_output = self.model.encode(
            documents,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )

        query_vecs = query_output["colbert_vecs"][0]

        scores = []
        for doc_vecs in doc_output["colbert_vecs"]:
            score = self._maxsim(query_vecs, doc_vecs)
            scores.append(score)

        return scores

    def __repr__(self) -> str:
        return (
            f"ColBERTReranker(model={self.config.model_name!r}, "
            f"initialized={self._initialized})"
        )


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Teste do ColBERT Reranker")
    print("=" * 60)

    # Documentos de teste
    query = "contratacoes interdependentes e correlatas"

    documents = [
        "Art. 6 O ETP devera evidenciar o problema a ser resolvido...",
        "Art. 3 Para fins do disposto nesta IN: III - contratacoes correlatas: aquelas cujos objetos sejam similares; IV - contratacoes interdependentes: aquelas que guardam relacao direta...",
        "Art. 14 A elaboracao do ETP e facultada nas hipoteses...",
    ]

    reranker = ColBERTReranker()

    print(f"\nQuery: {query}")
    print(f"Documentos: {len(documents)}")

    # Rerank
    print("\n--- Reranking ---")
    results = reranker.rerank(query, documents)

    for idx, score in results:
        print(f"\n{idx+1}. Score: {score:.4f}")
        print(f"   {documents[idx][:80]}...")

    # Com detalhes
    print("\n--- Com Detalhes ---")
    detailed = reranker.rerank_with_details(query, documents)

    for r in detailed:
        print(f"\nDoc {r['index']+1}: Score={r['score']:.4f}")
        print(f"  Query tokens: {r['n_query_tokens']}, Doc tokens: {r['n_doc_tokens']}")
        print(f"  Min/Max token score: {r['min_token_score']:.4f} / {r['max_token_score']:.4f}")
