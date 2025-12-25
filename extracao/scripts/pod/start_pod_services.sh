#!/bin/bash
# ============================================================================
# Script de startup para servicos no POD RunPod
#
# Inicia:
# 1. vLLM com Qwen3-30B-A3B-AWQ na porta 8000
# 2. Embedding Server (BGE-M3) na porta 8100
# 3. Redis para Celery na porta 6379
#
# Uso no POD:
#   cd /workspace/pipeline/extracao
#   bash scripts/pod/start_pod_services.sh
# ============================================================================

set -e

echo "=============================================="
echo "  Iniciando servicos no POD"
echo "=============================================="

# Diretorio base
WORKSPACE="/workspace/pipeline/extracao"
cd $WORKSPACE

# ============================================================================
# 1. Verifica GPU
# ============================================================================
echo ""
echo "[1/4] Verificando GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# 2. Inicia Redis (background)
# ============================================================================
echo "[2/4] Iniciando Redis..."
if pgrep -x "redis-server" > /dev/null; then
    echo "  Redis ja esta rodando"
else
    redis-server --daemonize yes --port 6379
    echo "  Redis iniciado na porta 6379"
fi

# ============================================================================
# 3. Inicia vLLM (background)
# ============================================================================
echo ""
echo "[3/4] Iniciando vLLM..."

VLLM_MODEL="stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"
VLLM_PORT=8000
VLLM_LOG="/workspace/logs/vllm.log"

mkdir -p /workspace/logs

# Verifica se vLLM ja esta rodando
if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "  vLLM ja esta rodando na porta $VLLM_PORT"
else
    echo "  Iniciando vLLM com modelo $VLLM_MODEL..."
    nohup python -m vllm.entrypoints.openai.api_server \
        --model $VLLM_MODEL \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --max-model-len 16000 \
        --gpu-memory-utilization 0.85 \
        --dtype auto \
        --trust-remote-code \
        > $VLLM_LOG 2>&1 &

    echo "  Aguardando vLLM iniciar (pode levar 2-3 min)..."
    for i in {1..60}; do
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            echo "  vLLM pronto na porta $VLLM_PORT"
            break
        fi
        sleep 5
        echo "    Aguardando... ($i/60)"
    done
fi

# ============================================================================
# 4. Inicia Embedding Server (background)
# ============================================================================
echo ""
echo "[4/4] Iniciando Embedding Server (BGE-M3)..."

EMBED_PORT=8100
EMBED_LOG="/workspace/logs/embedding_server.log"

# Verifica se embedding server ja esta rodando
if curl -s http://localhost:$EMBED_PORT/health > /dev/null 2>&1; then
    echo "  Embedding Server ja esta rodando na porta $EMBED_PORT"
else
    echo "  Iniciando Embedding Server..."
    nohup python -m src.pod.embedding_server > $EMBED_LOG 2>&1 &

    echo "  Aguardando Embedding Server iniciar..."
    for i in {1..30}; do
        if curl -s http://localhost:$EMBED_PORT/health > /dev/null 2>&1; then
            echo "  Embedding Server pronto na porta $EMBED_PORT"
            break
        fi
        sleep 2
        echo "    Aguardando... ($i/30)"
    done
fi

# ============================================================================
# Status Final
# ============================================================================
echo ""
echo "=============================================="
echo "  Status dos Servicos"
echo "=============================================="
echo ""
echo "Redis:            $(pgrep -x redis-server > /dev/null && echo 'OK (6379)' || echo 'FALHOU')"
echo "vLLM:             $(curl -s http://localhost:8000/health > /dev/null && echo 'OK (8000)' || echo 'Aguardando...')"
echo "Embedding Server: $(curl -s http://localhost:8100/health > /dev/null && echo 'OK (8100)' || echo 'Aguardando...')"
echo ""
echo "Logs:"
echo "  vLLM:     tail -f /workspace/logs/vllm.log"
echo "  Embeddings: tail -f /workspace/logs/embedding_server.log"
echo ""
echo "=============================================="
echo "  Servicos iniciados! Agora configure os tuneis SSH:"
echo ""
echo "  No Windows (PowerShell):"
echo "    # Tunel Local (vLLM + Embeddings)"
echo "    ssh -L 8000:localhost:8000 -L 8100:localhost:8100 root@195.26.233.70 -p 57457"
echo ""
echo "    # Tunel Reverso (Milvus)"
echo "    ssh -R 19530:localhost:19530 root@195.26.233.70 -p 57457"
echo "=============================================="
