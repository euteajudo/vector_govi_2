"""
Cliente para GPU Server no POD.

Acessa o servidor FastAPI unificado via HTTP direto,
sem necessidade de tunel SSH.

Uso:
    from pod import GPUClient

    client = GPUClient("http://195.26.233.70:8080")

    # Embeddings
    result = client.embed_hybrid(["texto 1", "texto 2"])
    # result = {'dense': [[...]], 'sparse': [{...}]}

    # LLM
    response = client.chat([
        {"role": "user", "content": "Ola!"}
    ])
    # response = {'content': '...', 'usage': {...}}
"""

import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


class GPUClient:
    """
    Cliente unificado para GPU Server no POD.

    Acessa embeddings (BGE-M3) e LLM (vLLM) via HTTP.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: int = 180,
    ):
        """
        Args:
            base_url: URL do GPU Server (ex: http://195.26.233.70:8080)
            timeout: Timeout em segundos para requisicoes
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> dict:
        """Verifica saude do servidor."""
        try:
            resp = self.session.get(
                f"{self.base_url}/health",
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Health check falhou: {e}")
            return {"status": "error", "error": str(e)}

    def is_healthy(self) -> bool:
        """Retorna True se servidor esta saudavel."""
        health = self.health_check()
        return health.get("status") == "ok"

    # ========================================================================
    # Embeddings
    # ========================================================================

    def embed_hybrid(
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
                f"{self.base_url}/embed/hybrid",
                json={"texts": texts, "batch_size": batch_size},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "dense": data["dense"],
                "sparse": data["sparse"],
            }
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings: {e}")
            raise

    def embed_dense(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[list[float]]:
        """Gera apenas embeddings densos."""
        try:
            resp = self.session.post(
                f"{self.base_url}/embed/dense",
                json={"texts": texts, "batch_size": batch_size},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings densos: {e}")
            raise

    def embed_batch(
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
            result = self.embed_hybrid(batch, batch_size=len(batch))
            all_dense.extend(result["dense"])
            all_sparse.extend(result["sparse"])

            if show_progress:
                processed = min(i + batch_size, total)
                logger.info(f"Embeddings: {processed}/{total}")

        return {"dense": all_dense, "sparse": all_sparse}

    # ========================================================================
    # LLM
    # ========================================================================

    def chat(
        self,
        messages: list[dict],
        model: str = "Qwen/Qwen3-8B-AWQ",
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[dict] = None,
    ) -> dict:
        """
        Envia mensagens para o LLM.

        Args:
            messages: Lista de mensagens [{"role": "user", "content": "..."}]
            model: Nome do modelo
            temperature: Temperatura (0.0 = deterministico)
            max_tokens: Maximo de tokens na resposta
            response_format: Formato de resposta (ex: {"type": "json_object"})

        Returns:
            Dict com 'content', 'model', 'usage', 'time_ms'
        """
        try:
            payload = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if response_format:
                payload["response_format"] = response_format

            resp = self.session.post(
                f"{self.base_url}/llm/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Erro ao chamar LLM: {e}")
            raise

    def chat_simple(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Chat simplificado - retorna apenas o conteudo.

        Args:
            prompt: Prompt do usuario
            system: Prompt de sistema (opcional)
            **kwargs: Argumentos extras para chat()

        Returns:
            Conteudo da resposta (string)
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = self.chat(messages, **kwargs)
        return result["content"]

    def list_models(self) -> list[str]:
        """Lista modelos disponiveis."""
        try:
            resp = self.session.get(
                f"{self.base_url}/llm/models",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Erro ao listar modelos: {e}")
            return []

    # ========================================================================
    # Context Manager
    # ========================================================================

    def close(self):
        """Fecha sessao HTTP."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# Helper
# ============================================================================

def get_gpu_client(base_url: str = "http://localhost:8080") -> GPUClient:
    """Cria cliente GPU."""
    return GPUClient(base_url)
