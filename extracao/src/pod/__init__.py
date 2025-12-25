"""
Modulo POD - Estrutura paralela para execucao em RunPod.

Este modulo contem componentes para rodar o pipeline usando
GPU remota do POD para embeddings e LLM.

Arquitetura:
- Milvus: LOCAL (Windows) - acesso via tunel SSH reverso
- vLLM: POD (GPU) - API em localhost:8000
- Embeddings: POD (GPU) - FastAPI em localhost:8100
- Redis: POD - para Celery workers

Uso:
    # No POD: iniciar servicos
    bash scripts/pod/start_pod_services.sh

    # No POD: iniciar workers Celery
    bash scripts/pod/start_celery_workers.sh 8

    # Local: rodar pipeline usando POD
    python scripts/pod/run_pipeline_pod.py --input data/lei_14133.md
"""

from .config import PodConfig, get_pod_config, set_pod_config
from .remote_embedder import RemoteEmbedder, get_remote_embedder
from .gpu_client import GPUClient, get_gpu_client

__all__ = [
    "PodConfig",
    "get_pod_config",
    "set_pod_config",
    "RemoteEmbedder",
    "get_remote_embedder",
    "GPUClient",
    "get_gpu_client",
]
