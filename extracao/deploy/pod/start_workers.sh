#!/bin/bash
# Script para iniciar Celery workers no POD
# Executa no POD via web terminal ou SSH

cd /workspace/extracao

# 1. Verifica se Redis está rodando
if ! pgrep -x "redis-server" > /dev/null; then
    echo "[INFO] Iniciando Redis..."
    redis-server --daemonize yes
    sleep 2
fi

# Verifica Redis
redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[OK] Redis rodando"
else
    echo "[ERRO] Redis não está respondendo"
    exit 1
fi

# 2. Ativa ambiente virtual
source venv/bin/activate

# 3. Exporta variáveis
export PYTHONPATH=/workspace/extracao/src:/workspace/extracao:$PYTHONPATH
export REDIS_HOST=localhost
export REDIS_PORT=6379
export GPU_SERVER_URL=http://localhost:8080
export MILVUS_HOST=77.37.43.160
export MILVUS_PORT=19530

# 4. Inicia workers Celery (10 workers)
echo "[INFO] Iniciando 10 workers Celery..."
celery -A src.enrichment.celery_app worker \
    --loglevel=info \
    --concurrency=10 \
    --queues=enrichment \
    --hostname=pod-worker@%h \
    --pidfile=/tmp/celery-worker.pid \
    --logfile=/workspace/celery-worker.log \
    --detach

# 5. Verifica se iniciou
sleep 3
if [ -f /tmp/celery-worker.pid ]; then
    PID=$(cat /tmp/celery-worker.pid)
    echo "[OK] Workers iniciados (PID: $PID)"
    echo "[OK] Log: /workspace/celery-worker.log"
else
    echo "[ERRO] Falha ao iniciar workers"
    exit 1
fi

# 6. Inicia Flower (opcional, para monitoramento)
echo "[INFO] Iniciando Flower (dashboard)..."
celery -A src.enrichment.celery_app flower \
    --port=5555 \
    --persistent=True \
    --db=/workspace/flower.db \
    --detach

echo "[OK] Flower rodando em http://localhost:5555"
echo ""
echo "=== SETUP CONCLUIDO ==="
echo "Workers: 10"
echo "Redis: localhost:6379"
echo "Flower: http://localhost:5555"
echo ""
echo "Para verificar status:"
echo "  celery -A src.enrichment.celery_app inspect active"
