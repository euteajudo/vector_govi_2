"""
Configuração centralizada para o Pipeline RAG.

Usa variáveis de ambiente com fallback para valores padrão (localhost).
No RunPod, as variáveis são definidas no .env ou docker-compose.yml.

Uso:
    from src.config import settings

    # Acessar configurações
    print(settings.vllm_url)      # http://localhost:8000/v1
    print(settings.milvus_host)   # localhost
    print(settings.redis_url)     # redis://localhost:6379/0
"""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """Configurações do sistema."""

    # vLLM
    vllm_host: str = os.getenv("VLLM_HOST", "localhost")
    vllm_port: int = int(os.getenv("VLLM_PORT", "8000"))

    # Milvus
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))

    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))

    # HuggingFace
    hf_home: str = os.getenv("HF_HOME", "")
    hf_token: str = os.getenv("HF_TOKEN", "")

    # Modelos
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B-AWQ")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # GPU
    gpu_memory_utilization: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
    max_model_len: int = int(os.getenv("MAX_MODEL_LEN", "16000"))

    # Collection Milvus
    collection_name: str = os.getenv("MILVUS_COLLECTION", "leis_v3")

    @property
    def vllm_url(self) -> str:
        """URL completa do vLLM."""
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"

    @property
    def redis_url(self) -> str:
        """URL completa do Redis."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def milvus_uri(self) -> str:
        """URI do Milvus."""
        return f"http://{self.milvus_host}:{self.milvus_port}"


@lru_cache()
def get_settings() -> Settings:
    """Retorna singleton das configurações."""
    return Settings()


# Singleton para acesso direto
settings = get_settings()


# Função helper para atualizar em runtime (testes)
def override_settings(**kwargs) -> Settings:
    """Sobrescreve configurações (útil para testes)."""
    get_settings.cache_clear()
    for key, value in kwargs.items():
        os.environ[key.upper()] = str(value)
    return get_settings()
