"""
Cliente remoto para embeddings BGE-M3.

Chama API FastAPI rodando no POD para gerar embeddings,
evitando carregar o modelo localmente.

Uso:
    from pod import RemoteEmbedder

    embedder = RemoteEmbedder()
    result = embedder.encode_hybrid(["texto 1", "texto 2"])
    # result = {'dense': [[...]], 'sparse': [{...}]}
"""

import logging
from typing import Optional
import requests

from .config import PodConfig, get_pod_config

logger = logging.getLogger(__name__)


class RemoteEmbedder:
    """
    Cliente para servidor de embeddings remoto.

    O servidor roda no POD e expoe API para gerar embeddings BGE-M3.
    """

    def __init__(self, config: Optional[PodConfig] = None):
        self.config = config or get_pod_config()
        self.base_url = self.config.embedding_base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> bool:
        """Verifica se servidor esta online."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Embedding server offline: {e}")
            return False

    def encode_dense(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """
        Gera embeddings densos.

        Args:
            texts: Lista de textos
            batch_size: Tamanho do batch

        Returns:
            Lista de vetores densos (1024d)
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/encode/dense",
                json={"texts": texts, "batch_size": batch_size},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings densos: {e}")
            raise

    def encode_sparse(self, texts: list[str]) -> list[dict]:
        """
        Gera embeddings esparsos (lexical weights).

        Args:
            texts: Lista de textos

        Returns:
            Lista de dicts {token_id: weight}
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/encode/sparse",
                json={"texts": texts},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings esparsos: {e}")
            raise

    def encode_hybrid(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> dict:
        """
        Gera embeddings hibridos (dense + sparse).

        Args:
            texts: Lista de textos
            batch_size: Tamanho do batch

        Returns:
            Dict com 'dense' e 'sparse'
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/encode/hybrid",
                json={"texts": texts, "batch_size": batch_size},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "dense": data["dense"],
                "sparse": data["sparse"],
            }
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings hibridos: {e}")
            raise

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> dict:
        """
        Gera embeddings em batches grandes.

        Args:
            texts: Lista de textos
            batch_size: Tamanho do batch
            show_progress: Se deve mostrar progresso

        Returns:
            Dict com 'dense' e 'sparse' para todos os textos
        """
        all_dense = []
        all_sparse = []

        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            result = self.encode_hybrid(batch, batch_size=len(batch))
            all_dense.extend(result["dense"])
            all_sparse.extend(result["sparse"])

            if show_progress:
                processed = min(i + batch_size, total)
                logger.info(f"Embeddings: {processed}/{total}")

        return {"dense": all_dense, "sparse": all_sparse}

    def close(self):
        """Fecha sessao HTTP."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Funcao helper
def get_remote_embedder(config: Optional[PodConfig] = None) -> RemoteEmbedder:
    """Cria cliente de embeddings remoto."""
    return RemoteEmbedder(config)
