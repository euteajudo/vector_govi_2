"""
Wrapper para BGE-Reranker-v2-m3 - Modelo de Reranking Multilíngue.

BGE-Reranker-v2-m3:
- Cross-encoder multilíngue (100+ idiomas)
- Treinado junto com BGE-M3 (se complementam)
- Score: 0-1 (relevância query-documento)
- Licença: Apache 2.0

Uso típico (2-stage retrieval):
1. BGE-M3 busca top 100 candidatos (rápido, bi-encoder)
2. BGE-Reranker reordena para top 10 (preciso, cross-encoder)

Uso:
    from embeddings.bge_reranker import BGEReranker

    reranker = BGEReranker()

    # Rerank uma lista de documentos
    scores = reranker.compute_scores(
        query="O que é ETP?",
        documents=["Doc 1...", "Doc 2...", "Doc 3..."]
    )
    # scores = [0.95, 0.23, 0.87]

    # Rerank e retorna ordenado
    results = reranker.rerank(
        query="O que é ETP?",
        documents=[{"text": "...", "id": 1}, ...],
        top_k=5
    )
"""

import logging
from dataclasses import dataclass
from typing import Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACAO
# =============================================================================

@dataclass
class RerankerConfig:
    """Configuracao do modelo de reranking."""

    # Modelo
    model_name: str = "BAAI/bge-reranker-v2-m3"

    # Performance
    use_fp16: bool = True
    batch_size: int = 32
    max_length: int = 1024  # Reranker usa contexto menor que embedder

    # Device
    device: Optional[str] = None  # None = auto-detect

    # Normalize scores (0-1)
    normalize: bool = True

    @classmethod
    def default(cls) -> "RerankerConfig":
        return cls()

    @classmethod
    def fast(cls) -> "RerankerConfig":
        """Configuracao otimizada para velocidade."""
        return cls(
            use_fp16=True,
            batch_size=64,
            max_length=512,
        )


# =============================================================================
# WRAPPER BGE-RERANKER
# =============================================================================

class BGEReranker:
    """
    Wrapper para reranking com BGE-Reranker-v2-m3.

    Cross-encoder que recebe pares (query, document) e retorna
    score de relevância. Mais preciso que bi-encoder mas mais lento.

    Attributes:
        config: Configuracao do modelo
        model: Modelo reranker carregado
        _initialized: Se o modelo foi carregado
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Inicializa o reranker.

        O modelo e carregado de forma lazy (apenas quando chamado).

        Args:
            config: Configuracao do modelo. Usa default se nao fornecido.
        """
        self.config = config or RerankerConfig.default()
        self.model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Carrega o modelo se ainda nao foi carregado (lazy loading)."""
        if self._initialized:
            return

        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "FlagEmbedding nao instalado. "
                "Instale com: pip install FlagEmbedding"
            )

        logger.info(f"Carregando reranker {self.config.model_name}...")

        self.model = FlagReranker(
            self.config.model_name,
            use_fp16=self.config.use_fp16,
            device=self.config.device,
        )

        self._initialized = True
        logger.info(f"Reranker carregado com sucesso")

    # =========================================================================
    # COMPUTE SCORES
    # =========================================================================

    def compute_scores(
        self,
        query: str,
        documents: list[str],
        batch_size: Optional[int] = None,
    ) -> list[float]:
        """
        Calcula scores de relevância para pares query-documento.

        Args:
            query: Query de busca
            documents: Lista de documentos para rankear
            batch_size: Tamanho do batch

        Returns:
            Lista de scores (0-1 se normalize=True)
        """
        self._ensure_initialized()

        if not documents:
            return []

        batch_size = batch_size or self.config.batch_size

        # Cria pares (query, doc) para o reranker
        pairs = [[query, doc] for doc in documents]

        logger.debug(f"Calculando scores para {len(pairs)} pares...")

        # Compute scores
        scores = self.model.compute_score(
            pairs,
            batch_size=batch_size,
            max_length=self.config.max_length,
            normalize=self.config.normalize,
        )

        # FlagReranker pode retornar float ou lista
        if isinstance(scores, (int, float)):
            scores = [scores]

        return [float(s) for s in scores]

    def compute_score_single(self, query: str, document: str) -> float:
        """
        Calcula score para um único par query-documento.

        Args:
            query: Query de busca
            document: Documento para avaliar

        Returns:
            Score de relevância
        """
        scores = self.compute_scores(query, [document])
        return scores[0] if scores else 0.0

    # =========================================================================
    # RERANK
    # =========================================================================

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_key: str = "text",
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Reordena documentos por relevância.

        Args:
            query: Query de busca
            documents: Lista de dicts com documentos
            text_key: Chave do texto no dict
            top_k: Retorna apenas top K (None = todos)
            return_scores: Adiciona score ao resultado

        Returns:
            Lista de documentos reordenados
        """
        if not documents:
            return []

        # Extrai textos
        texts = [doc.get(text_key, "") for doc in documents]

        # Calcula scores
        scores = self.compute_scores(query, texts)

        # Combina documentos com scores
        results = []
        for doc, score in zip(documents, scores):
            result = dict(doc)
            if return_scores:
                result["rerank_score"] = score
            results.append(result)

        # Ordena por score (maior primeiro)
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Aplica top_k
        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_with_ids(
        self,
        query: str,
        documents: list[tuple[str, str]],  # [(id, text), ...]
        top_k: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Reordena documentos e retorna IDs com scores.

        Args:
            query: Query de busca
            documents: Lista de tuplas (id, text)
            top_k: Retorna apenas top K

        Returns:
            Lista de tuplas (id, score) ordenadas
        """
        if not documents:
            return []

        ids = [doc[0] for doc in documents]
        texts = [doc[1] for doc in documents]

        scores = self.compute_scores(query, texts)

        # Combina e ordena
        results = list(zip(ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    # =========================================================================
    # UTILITARIOS
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"BGEReranker(model={self.config.model_name!r}, "
            f"normalize={self.config.normalize}, "
            f"initialized={self._initialized})"
        )


# =============================================================================
# MOCK PARA TESTES
# =============================================================================

class MockReranker:
    """
    Reranker falso para testes sem GPU/modelo.

    Gera scores baseados em overlap de palavras.
    """

    def __init__(self):
        logger.warning("Usando MockReranker - scores sao simulados!")

    def compute_scores(
        self,
        query: str,
        documents: list[str],
        **kwargs
    ) -> list[float]:
        """Gera scores baseados em overlap de palavras."""
        query_words = set(query.lower().split())

        scores = []
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            max_possible = max(len(query_words), 1)
            score = min(overlap / max_possible, 1.0)
            scores.append(score)

        return scores

    def compute_score_single(self, query: str, document: str) -> float:
        return self.compute_scores(query, [document])[0]

    def rerank(
        self,
        query: str,
        documents: list[dict],
        text_key: str = "text",
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> list[dict]:
        texts = [doc.get(text_key, "") for doc in documents]
        scores = self.compute_scores(query, texts)

        results = []
        for doc, score in zip(documents, scores):
            result = dict(doc)
            if return_scores:
                result["rerank_score"] = score
            results.append(result)

        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_with_ids(
        self,
        query: str,
        documents: list[tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        ids = [doc[0] for doc in documents]
        texts = [doc[1] for doc in documents]
        scores = self.compute_scores(query, texts)

        results = list(zip(ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results


# =============================================================================
# FACTORY
# =============================================================================

def get_reranker(
    use_mock: bool = False,
    config: Optional[RerankerConfig] = None,
) -> BGEReranker | MockReranker:
    """
    Factory para criar reranker.

    Args:
        use_mock: Se True, retorna MockReranker (para testes)
        config: Configuracao do reranker

    Returns:
        Reranker configurado
    """
    if use_mock:
        return MockReranker()

    return BGEReranker(config=config)


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Teste do BGEReranker")
    print("=" * 60)

    # Query e documentos de teste
    query = "O que e ETP e para que serve?"

    documents = [
        {
            "id": "art-3",
            "text": "Art. 3 Para fins do disposto nesta Instrucao Normativa, considera-se: I - Estudo Tecnico Preliminar - ETP: documento constitutivo da primeira etapa do planejamento de uma contratacao...",
        },
        {
            "id": "art-6",
            "text": "Art. 6 O ETP devera evidenciar o problema a ser resolvido e a melhor solucao, de modo a permitir a avaliacao da viabilidade tecnica...",
        },
        {
            "id": "art-14",
            "text": "Art. 14 A elaboracao do ETP: I - e facultada nas hipoteses dos incisos I, II, VII e VIII do art. 75...",
        },
    ]

    # Usa mock para teste sem modelo
    reranker = get_reranker(use_mock=True)

    print(f"\nReranker: {reranker}")
    print(f"Query: {query}")
    print(f"Documentos: {len(documents)}")

    # Rerank
    print("\n--- Reranking ---")
    results = reranker.rerank(query, documents, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['id']}] Score: {result['rerank_score']:.4f}")
        print(f"   {result['text'][:80]}...")

    # Teste com modelo real
    print("\n" + "=" * 60)
    print("Teste com modelo real (se disponivel)")
    print("=" * 60)

    try:
        real_reranker = get_reranker(use_mock=False)
        texts = [doc["text"] for doc in documents]
        scores = real_reranker.compute_scores(query, texts)

        print(f"\nScores reais:")
        for doc, score in zip(documents, scores):
            print(f"  {doc['id']}: {score:.4f}")

    except ImportError as e:
        print(f"Modelo nao disponivel: {e}")
    except Exception as e:
        print(f"Erro: {e}")
