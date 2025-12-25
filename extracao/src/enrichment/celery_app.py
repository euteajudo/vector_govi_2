"""
Configuracao do Celery para enriquecimento paralelo.

Requer Redis rodando:
    docker run -d --name redis -p 6379:6379 redis:alpine

Iniciar worker na VPS:
    cd /root/vector_govi_2/extracao
    source venv/bin/activate
    celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=4

Monitorar:
    celery -A src.enrichment.celery_app flower --port=5555

Acessar Flower:
    http://77.37.43.160:5555
"""

import os
from celery import Celery

# Configuracao do ambiente
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

# Configuracao do Celery com Redis
app = Celery(
    "enrichment",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    include=["src.enrichment.tasks", "src.enrichment.tasks_http"],
)

# Configuracoes
app.conf.update(
    # Serialização
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="America/Sao_Paulo",
    enable_utc=True,

    # Task settings
    task_track_started=True,
    task_time_limit=600,  # 10 minutos max por task
    task_soft_time_limit=300,  # 5 minutos soft limit

    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Rate limiting (evita sobrecarregar vLLM)
    task_default_rate_limit="10/m",  # 10 tasks por minuto

    # Prefetch (quantas tasks pegar de uma vez)
    worker_prefetch_multiplier=1,  # Uma task por vez por worker
)

if __name__ == "__main__":
    app.start()
