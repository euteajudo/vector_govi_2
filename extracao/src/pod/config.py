"""
Configuracoes para conexao com RunPod.

Suporta dois modos:
1. LOCAL_TO_POD: Codigo roda local, usa servicos do POD via SSH tunnel
2. ON_POD: Codigo roda diretamente no POD

Tuneis SSH necessarios (LOCAL_TO_POD):
- Local 8000 -> POD 8000 (vLLM)
- Local 8100 -> POD 8100 (Embedding Server)
- Reverso POD 19530 -> Local 19530 (Milvus)
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PodConfig:
    """Configuracao de conexao com POD."""

    # Modo de execucao
    mode: str = "HTTP_DIRECT"  # HTTP_DIRECT (novo) ou SSH_TUNNEL (legado)

    # POD SSH (legado - para fallback)
    pod_ip: str = "195.26.233.70"
    pod_ssh_port: int = 57457
    ssh_key_path: str = "~/.ssh/id_ed25519"

    # GPU Server (HTTP direto via RunPod TCP)
    gpu_server_host: str = "195.26.233.70"
    gpu_server_port: int = 55278  # Porta TCP exposta no RunPod (interna 8080 -> externa 55278)

    # vLLM (interno no POD - acessado via GPU Server)
    vllm_model: str = "Qwen/Qwen3-8B-AWQ"

    # Milvus (VPS - acessivel de qualquer lugar)
    milvus_host: str = "77.37.43.160"
    milvus_port: int = 19530

    # Redis (no POD - para Celery)
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Celery
    celery_concurrency: int = 8

    @property
    def gpu_server_url(self) -> str:
        """URL do GPU Server (HTTP direto)."""
        return f"http://{self.gpu_server_host}:{self.gpu_server_port}"

    # Legado - compatibilidade
    @property
    def vllm_base_url(self) -> str:
        if self.mode == "HTTP_DIRECT":
            return f"{self.gpu_server_url}/llm"
        return f"http://localhost:8000/v1"

    @property
    def embedding_base_url(self) -> str:
        if self.mode == "HTTP_DIRECT":
            return f"{self.gpu_server_url}/embed"
        return "http://localhost:8100"

    @classmethod
    def from_env(cls) -> "PodConfig":
        """Cria config a partir de variaveis de ambiente."""
        return cls(
            mode=os.getenv("POD_MODE", "LOCAL_TO_POD"),
            pod_ip=os.getenv("POD_IP", "195.26.233.70"),
            pod_ssh_port=int(os.getenv("POD_SSH_PORT", "57457")),
            vllm_host=os.getenv("VLLM_HOST", "localhost"),
            vllm_port=int(os.getenv("VLLM_PORT", "8000")),
            vllm_model=os.getenv("VLLM_MODEL", "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"),
            embedding_host=os.getenv("EMBEDDING_HOST", "localhost"),
            embedding_port=int(os.getenv("EMBEDDING_PORT", "8100")),
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=int(os.getenv("MILVUS_PORT", "19530")),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            celery_concurrency=int(os.getenv("CELERY_CONCURRENCY", "8")),
        )

    @classmethod
    def for_local_to_pod(cls) -> "PodConfig":
        """Config para rodar local usando POD via tunnels."""
        return cls(mode="LOCAL_TO_POD")

    @classmethod
    def for_on_pod(cls) -> "PodConfig":
        """Config para rodar diretamente no POD."""
        return cls(
            mode="ON_POD",
            milvus_host="77.37.43.160",  # VPS
            milvus_port=19530,
        )


# Singleton
_config: Optional[PodConfig] = None


def get_pod_config() -> PodConfig:
    """Retorna configuracao do POD (singleton)."""
    global _config
    if _config is None:
        _config = PodConfig.from_env()
    return _config


def set_pod_config(config: PodConfig):
    """Define configuracao do POD."""
    global _config
    _config = config
