"""
Wrapper para BGE-M3 - Modelo de Embedding Multilíngue.

BGE-M3 (BAAI General Embedding - Multilingual, Multi-Functionality, Multi-Granularity):
- Dense: 1024 dimensões
- Sparse: Learned sparse (superior ao BM25 estatístico)
- Suporta até 8192 tokens de contexto
- Multilíngue (100+ idiomas, incluindo português)
- Licença: Apache 2.0

Por que usar Sparse do BGE-M3 em vez de BM25:
1. Learned sparse é treinado junto com dense - se complementam
2. Aprende pesos semânticos para tokens, não apenas frequência
3. Captura sinônimos: "requisitante" ~ "demandante" ~ "solicitante"
4. Especialmente importante para português jurídico
5. Consistência: mesmo modelo para dense e sparse

Uso:
    from embeddings import BGEM3Embedder

    embedder = BGEM3Embedder()

    # Retorna dense + sparse
    result = embedder.encode_hybrid(["Texto 1", "Texto 2"])
    dense = result["dense"]   # List[List[float]] - 1024d
    sparse = result["sparse"] # List[Dict[int, float]] - token_id -> weight

    # Apenas dense (retrocompatível)
    embeddings = embedder.encode(["Texto 1", "Texto 2"])
"""

import logging
from dataclasses import dataclass
from typing import Optional, TypedDict
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# TIPOS
# =============================================================================

class HybridEmbeddings(TypedDict):
    """Resultado do encode híbrido (dense + sparse)."""
    dense: list[list[float]]           # Dense embeddings (1024d)
    sparse: list[dict[int, float]]     # Sparse embeddings {token_id: weight}


class SingleHybridEmbedding(TypedDict):
    """Resultado do encode híbrido para um único texto."""
    dense: list[float]
    sparse: dict[int, float]


# =============================================================================
# CONFIGURACAO
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuracao do modelo de embedding."""

    # Modelo
    model_name: str = "BAAI/bge-m3"

    # Dimensoes do embedding denso
    embedding_dim: int = 1024

    # Performance
    use_fp16: bool = True
    batch_size: int = 32
    max_length: int = 8192  # BGE-M3 suporta ate 8192 tokens

    # Normalizacao
    normalize_embeddings: bool = True

    # Device
    device: Optional[str] = None  # None = auto-detect (cuda se disponivel)

    # Cache
    cache_dir: Optional[str] = None

    # Sparse config
    return_sparse: bool = True  # Retornar sparse embeddings por padrao

    @classmethod
    def default(cls) -> "EmbeddingConfig":
        return cls()

    @classmethod
    def fast(cls) -> "EmbeddingConfig":
        """Configuracao otimizada para velocidade."""
        return cls(
            use_fp16=True,
            batch_size=64,
            max_length=2048,
        )

    @classmethod
    def dense_only(cls) -> "EmbeddingConfig":
        """Apenas dense embeddings (mais rapido)."""
        return cls(return_sparse=False)


# =============================================================================
# WRAPPER BGE-M3
# =============================================================================

class BGEM3Embedder:
    """
    Wrapper para geracao de embeddings com BGE-M3.

    Gera embeddings densos (1024d) e sparse (learned) para busca hibrida.

    O sparse embedding do BGE-M3 e um modelo aprendido (learned sparse)
    que foi treinado junto com o dense embedding, permitindo que ambos
    se complementem de forma otimizada durante a busca hibrida.

    Attributes:
        config: Configuracao do modelo
        model: Modelo BGE-M3 carregado
        _initialized: Se o modelo foi carregado
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Inicializa o embedder.

        O modelo e carregado de forma lazy (apenas quando encode e chamado).

        Args:
            config: Configuracao do modelo. Usa default se nao fornecido.
        """
        self.config = config or EmbeddingConfig.default()
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

        logger.info(f"Carregando modelo {self.config.model_name}...")

        self.model = BGEM3FlagModel(
            self.config.model_name,
            use_fp16=self.config.use_fp16,
            device=self.config.device,
        )

        self._initialized = True
        logger.info(f"Modelo carregado com sucesso (dim={self.config.embedding_dim})")

    # =========================================================================
    # ENCODE HIBRIDO (DENSE + SPARSE)
    # =========================================================================

    def encode_hybrid(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> HybridEmbeddings:
        """
        Gera embeddings dense e sparse para lista de textos.

        Args:
            texts: Lista de textos para embedding
            batch_size: Tamanho do batch (default: config.batch_size)

        Returns:
            HybridEmbeddings com 'dense' e 'sparse'
        """
        self._ensure_initialized()

        if not texts:
            return {"dense": [], "sparse": []}

        batch_size = batch_size or self.config.batch_size

        logger.debug(f"Gerando embeddings hibridos para {len(texts)} textos...")

        # BGE-M3 encode com dense e sparse
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.config.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        # Extrai embeddings densos
        dense_vecs = output["dense_vecs"]

        # Normaliza se configurado
        if self.config.normalize_embeddings:
            dense_vecs = self._normalize(dense_vecs)

        # Extrai sparse embeddings
        # BGE-M3 retorna: {"lexical_weights": [{"token_id": weight, ...}, ...]}
        sparse_vecs = self._process_sparse(output.get("lexical_weights", []))

        return {
            "dense": dense_vecs.tolist(),
            "sparse": sparse_vecs,
        }

    def encode_hybrid_single(self, text: str) -> SingleHybridEmbedding:
        """
        Gera embedding hibrido para um unico texto.

        Args:
            text: Texto para embedding

        Returns:
            SingleHybridEmbedding com 'dense' e 'sparse'
        """
        result = self.encode_hybrid([text])
        return {
            "dense": result["dense"][0] if result["dense"] else [],
            "sparse": result["sparse"][0] if result["sparse"] else {},
        }

    # =========================================================================
    # ENCODE DENSE ONLY (RETROCOMPATIVEL)
    # =========================================================================

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Gera apenas embeddings densos para lista de textos.

        Metodo retrocompativel - retorna apenas dense embeddings.

        Args:
            texts: Lista de textos para embedding
            batch_size: Tamanho do batch (default: config.batch_size)
            show_progress: Mostrar barra de progresso

        Returns:
            Lista de embeddings densos (cada um com 1024 dimensoes)
        """
        self._ensure_initialized()

        if not texts:
            return []

        batch_size = batch_size or self.config.batch_size

        logger.debug(f"Gerando embeddings densos para {len(texts)} textos...")

        # BGE-M3 encode apenas dense
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.config.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        # Extrai embeddings densos
        dense_vecs = output["dense_vecs"]

        # Normaliza se configurado
        if self.config.normalize_embeddings:
            dense_vecs = self._normalize(dense_vecs)

        return dense_vecs.tolist()

    def encode_single(self, text: str) -> list[float]:
        """
        Gera embedding denso para um unico texto.

        Args:
            text: Texto para embedding

        Returns:
            Embedding com 1024 dimensoes
        """
        embeddings = self.encode([text])
        return embeddings[0] if embeddings else []

    # =========================================================================
    # ENCODE SPARSE ONLY
    # =========================================================================

    def encode_sparse(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> list[dict[int, float]]:
        """
        Gera apenas embeddings sparse para lista de textos.

        Args:
            texts: Lista de textos para embedding
            batch_size: Tamanho do batch

        Returns:
            Lista de sparse embeddings {token_id: weight}
        """
        self._ensure_initialized()

        if not texts:
            return []

        batch_size = batch_size or self.config.batch_size

        logger.debug(f"Gerando embeddings sparse para {len(texts)} textos...")

        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.config.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        return self._process_sparse(output.get("lexical_weights", []))

    def encode_sparse_single(self, text: str) -> dict[int, float]:
        """
        Gera embedding sparse para um unico texto.

        Args:
            text: Texto para embedding

        Returns:
            Sparse embedding {token_id: weight}
        """
        sparse = self.encode_sparse([text])
        return sparse[0] if sparse else {}

    # =========================================================================
    # CONVERSAO PARA MILVUS
    # =========================================================================

    def sparse_to_milvus(self, sparse: dict[int, float]) -> dict:
        """
        Converte sparse embedding para formato Milvus SparseFloatVector.

        Milvus espera: {"indices": [int, ...], "values": [float, ...]}
        ou simplesmente um dict {int: float} que ele converte internamente.

        Args:
            sparse: Sparse embedding {token_id: weight}

        Returns:
            Dict no formato Milvus
        """
        if not sparse:
            return {}

        # Milvus aceita dict diretamente para SparseFloatVector
        # Mas podemos converter para formato explícito se necessário
        return dict(sparse)

    def sparse_batch_to_milvus(self, sparse_list: list[dict[int, float]]) -> list[dict]:
        """
        Converte batch de sparse embeddings para formato Milvus.

        Args:
            sparse_list: Lista de sparse embeddings

        Returns:
            Lista de dicts no formato Milvus
        """
        return [self.sparse_to_milvus(s) for s in sparse_list]

    # =========================================================================
    # UTILITARIOS
    # =========================================================================

    def _process_sparse(self, lexical_weights: list) -> list[dict[int, float]]:
        """
        Processa sparse embeddings do BGE-M3.

        BGE-M3 retorna lexical_weights como lista de dicts {token_id_str: weight}.
        Convertemos para {token_id_int: weight}.
        """
        result = []

        for weights in lexical_weights:
            if isinstance(weights, dict):
                # Converte string keys para int e filtra pesos muito baixos
                processed = {}
                for token_id, weight in weights.items():
                    try:
                        tid = int(token_id) if isinstance(token_id, str) else token_id
                        # Filtra pesos muito baixos para economia de espaco
                        if abs(weight) > 1e-6:
                            processed[tid] = float(weight)
                    except (ValueError, TypeError):
                        continue
                result.append(processed)
            else:
                result.append({})

        return result

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normaliza embeddings para norma L2 = 1."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Evita divisao por zero
        return embeddings / norms

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calcula similaridade cosseno entre dois embeddings densos.

        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding

        Returns:
            Score de similaridade (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def sparse_similarity(
        self,
        sparse1: dict[int, float],
        sparse2: dict[int, float],
    ) -> float:
        """
        Calcula similaridade entre dois sparse embeddings.

        Usa produto interno normalizado.

        Args:
            sparse1: Primeiro sparse embedding
            sparse2: Segundo sparse embedding

        Returns:
            Score de similaridade
        """
        if not sparse1 or not sparse2:
            return 0.0

        # Produto interno
        common_keys = set(sparse1.keys()) & set(sparse2.keys())
        dot_product = sum(sparse1[k] * sparse2[k] for k in common_keys)

        # Normas
        norm1 = np.sqrt(sum(v**2 for v in sparse1.values()))
        norm2 = np.sqrt(sum(v**2 for v in sparse2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @property
    def embedding_dimension(self) -> int:
        """Retorna dimensao do embedding denso."""
        return self.config.embedding_dim

    def __repr__(self) -> str:
        return (
            f"BGEM3Embedder(model={self.config.model_name!r}, "
            f"dim={self.config.embedding_dim}, "
            f"sparse={self.config.return_sparse}, "
            f"initialized={self._initialized})"
        )


# =============================================================================
# MOCK PARA TESTES
# =============================================================================

class MockEmbedder:
    """
    Embedder falso para testes sem GPU/modelo.

    Gera vetores aleatorios de 1024 dimensoes e sparse simulado.
    """

    def __init__(self, dim: int = 1024, seed: int = 42):
        self.dim = dim
        self._seed = seed
        np.random.seed(seed)
        logger.warning("Usando MockEmbedder - embeddings sao aleatorios!")

    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Gera embeddings densos aleatorios."""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            vec = np.random.randn(self.dim)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec.tolist())
        return embeddings

    def encode_single(self, text: str) -> list[float]:
        return self.encode([text])[0]

    def encode_sparse(self, texts: list[str], **kwargs) -> list[dict[int, float]]:
        """Gera sparse embeddings simulados."""
        result = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            # Simula 20-50 tokens com pesos
            num_tokens = np.random.randint(20, 50)
            token_ids = np.random.randint(0, 30000, num_tokens)
            weights = np.random.rand(num_tokens) * 2 - 0.5
            sparse = {int(tid): float(w) for tid, w in zip(token_ids, weights) if abs(w) > 0.1}
            result.append(sparse)
        return result

    def encode_sparse_single(self, text: str) -> dict[int, float]:
        return self.encode_sparse([text])[0]

    def encode_hybrid(self, texts: list[str], **kwargs) -> HybridEmbeddings:
        """Gera embeddings hibridos simulados."""
        return {
            "dense": self.encode(texts),
            "sparse": self.encode_sparse(texts),
        }

    def encode_hybrid_single(self, text: str) -> SingleHybridEmbedding:
        result = self.encode_hybrid([text])
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0],
        }

    def sparse_to_milvus(self, sparse: dict[int, float]) -> dict:
        return dict(sparse)

    def sparse_batch_to_milvus(self, sparse_list: list[dict[int, float]]) -> list[dict]:
        return sparse_list

    @property
    def embedding_dimension(self) -> int:
        return self.dim


# =============================================================================
# FACTORY
# =============================================================================

def get_embedder(
    use_mock: bool = False,
    config: Optional[EmbeddingConfig] = None,
) -> BGEM3Embedder | MockEmbedder:
    """
    Factory para criar embedder.

    Args:
        use_mock: Se True, retorna MockEmbedder (para testes)
        config: Configuracao do embedder

    Returns:
        Embedder configurado
    """
    if use_mock:
        return MockEmbedder(dim=config.embedding_dim if config else 1024)

    return BGEM3Embedder(config=config)


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Teste do BGEM3Embedder (Dense + Sparse)")
    print("=" * 60)

    # Textos de teste - sinonimos juridicos
    texts = [
        "O requisitante deve elaborar o documento de formalizacao da demanda.",
        "O demandante deve preparar o documento de formalizacao da demanda.",
        "O solicitante deve criar o documento de formalizacao da demanda.",
    ]

    # Usa mock para teste sem modelo
    embedder = get_embedder(use_mock=True)

    print(f"\nEmbedder: {embedder}")
    print(f"Dimensao dense: {embedder.embedding_dimension}")

    # Gera embeddings hibridos
    print("\n--- Embeddings Hibridos ---")
    result = embedder.encode_hybrid(texts)

    print(f"Dense embeddings: {len(result['dense'])} x {len(result['dense'][0])}d")
    print(f"Sparse embeddings: {len(result['sparse'])} vetores")

    for i, sparse in enumerate(result["sparse"]):
        print(f"  Texto {i+1}: {len(sparse)} tokens com peso")

    # Testa similaridade dense
    print("\n--- Similaridade Dense ---")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = embedder.similarity(result["dense"][i], result["dense"][j]) if hasattr(embedder, 'similarity') else 0
            print(f"  Texto {i+1} vs {j+1}: {sim:.4f}")

    # Formato Milvus
    print("\n--- Formato Milvus (Sparse) ---")
    milvus_sparse = embedder.sparse_batch_to_milvus(result["sparse"])
    print(f"Convertido para Milvus: {len(milvus_sparse)} vetores")

    print("\n" + "=" * 60)
    print("Teste com modelo real (se disponivel)")
    print("=" * 60)

    try:
        real_embedder = get_embedder(use_mock=False)
        real_result = real_embedder.encode_hybrid(texts[:1])
        print(f"Dense: {len(real_result['dense'][0])} dimensoes")
        print(f"Sparse: {len(real_result['sparse'][0])} tokens")
    except ImportError as e:
        print(f"Modelo nao disponivel: {e}")
    except Exception as e:
        print(f"Erro: {e}")
