#!/bin/bash
# ============================================================================
# Script para iniciar Celery workers no POD
#
# O POD tem GPU maior (48GB), então podemos ter mais workers.
# Cada worker processa 1 task por vez (evita OOM).
#
# Uso:
#   cd /workspace/pipeline/extracao
#   bash scripts/pod/start_celery_workers.sh [NUM_WORKERS]
#
# Exemplo:
#   bash scripts/pod/start_celery_workers.sh 8
# ============================================================================

set -e

# Numero de workers (default: 8)
NUM_WORKERS=${1:-8}

echo "=============================================="
echo "  Iniciando $NUM_WORKERS Celery Workers"
echo "=============================================="

cd /workspace/pipeline/extracao

# Verifica se Redis está rodando
if ! pgrep -x "redis-server" > /dev/null; then
    echo "[ERRO] Redis não está rodando. Execute start_pod_services.sh primeiro."
    exit 1
fi

# Verifica se vLLM está rodando
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "[AVISO] vLLM não está respondendo. Workers podem falhar."
fi

# Verifica se Embedding Server está rodando
if ! curl -s http://localhost:8100/health > /dev/null 2>&1; then
    echo "[AVISO] Embedding Server não está respondendo. Workers podem falhar."
fi

# Para workers antigos se existirem
echo ""
echo "[1/2] Parando workers antigos..."
pkill -f "celery.*pod_enrichment" 2>/dev/null || true
sleep 2

# Inicia workers
echo ""
echo "[2/2] Iniciando $NUM_WORKERS workers..."

mkdir -p /workspace/logs

for i in $(seq 1 $NUM_WORKERS); do
    WORKER_NAME="pod_worker_$i"
    LOG_FILE="/workspace/logs/celery_worker_$i.log"

    echo "  Iniciando $WORKER_NAME..."

    nohup celery -A src.pod.celery_app worker \
        --loglevel=info \
        --concurrency=1 \
        --hostname="$WORKER_NAME@%h" \
        > "$LOG_FILE" 2>&1 &

    sleep 1
done

echo ""
echo "=============================================="
echo "  $NUM_WORKERS workers iniciados!"
echo ""
echo "  Monitoramento:"
echo "    # Ver workers ativos"
echo "    celery -A src.pod.celery_app inspect active"
echo ""
echo "    # Ver filas"
echo "    celery -A src.pod.celery_app inspect reserved"
echo ""
echo "    # Dashboard web (Flower)"
echo "    celery -A src.pod.celery_app flower --port=5555"
echo ""
echo "  Logs:"
echo "    tail -f /workspace/logs/celery_worker_*.log"
echo "=============================================="
