"""
Configuracao Celery para workers no POD.

O POD tem GPU maior, entao podemos ter mais workers concorrentes.
Redis roda localmente no POD (localhost:6379).
"""

from celery import Celery

from .config import get_pod_config

config = get_pod_config()

app = Celery(
    "pod_enrichment",
    broker=f"redis://{config.redis_host}:{config.redis_port}/0",
    backend=f"redis://{config.redis_host}:{config.redis_port}/0",
    include=["src.pod.tasks"],
)

app.conf.update(
    # Tempo maximo por task (10 min)
    task_time_limit=600,

    # Rate limit: mais tasks por minuto no POD
    task_default_rate_limit="20/m",

    # 1 task por vez por worker (evita OOM na GPU)
    worker_prefetch_multiplier=1,

    # Retry automatico se worker morrer
    task_acks_late=True,

    # Serialização
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="America/Sao_Paulo",

    # Resultados expiram em 1 hora
    result_expires=3600,
)
