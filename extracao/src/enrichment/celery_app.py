"""
Configuracao do Celery para enriquecimento paralelo.

Requer Redis rodando:
    docker run -d --name redis -p 6379:6379 redis:alpine

Iniciar worker:
    cd extracao
    celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=2

Monitorar:
    celery -A src.enrichment.celery_app flower  # (requer: pip install flower)
"""

from celery import Celery

# Configuracao do Celery com Redis
app = Celery(
    "enrichment",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["src.enrichment.tasks"],
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
