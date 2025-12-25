#!/bin/bash
# ============================================================================
# Script de startup do GPU Server no POD RunPod
#
# Inicia:
# 1. vLLM com Qwen3-8B-AWQ na porta 8000 (interno)
# 2. GPU Server (FastAPI) na porta 8080 (exposto via TCP)
#
# O GPU Server unifica:
# - Embeddings (BGE-M3) em /embed/*
# - LLM (proxy vLLM) em /llm/*
#
# Uso no POD:
#   cd /workspace/pipeline/extracao
#   bash scripts/pod/start_gpu_server.sh
#
# Depois, no Windows, acesse diretamente:
#   http://195.26.233.70:8080/health
# ============================================================================

set -e

echo "=============================================="
echo "  GPU Server - POD Startup"
echo "=============================================="

WORKSPACE="/workspace/pipeline/extracao"
cd $WORKSPACE

# ============================================================================
# 1. Verifica GPU
# ============================================================================
echo ""
echo "[1/3] Verificando GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# 2. Inicia vLLM (background, porta 8000 interna)
# ============================================================================
echo "[2/3] Iniciando vLLM..."

VLLM_MODEL="Qwen/Qwen3-8B-AWQ"
VLLM_PORT=8000
VLLM_LOG="/workspace/logs/vllm.log"

mkdir -p /workspace/logs

if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "  vLLM ja esta rodando na porta $VLLM_PORT"
else
    echo "  Iniciando vLLM com modelo $VLLM_MODEL..."
    nohup python -m vllm.entrypoints.openai.api_server \
        --model $VLLM_MODEL \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --max-model-len 16000 \
        --gpu-memory-utilization 0.7 \
        --dtype auto \
        --trust-remote-code \
        > $VLLM_LOG 2>&1 &

    echo "  Aguardando vLLM iniciar (pode levar 2-3 min)..."
    for i in {1..60}; do
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            echo "  vLLM pronto!"
            break
        fi
        sleep 5
        echo "    Aguardando... ($i/60)"
    done
fi

# ============================================================================
# 3. Inicia GPU Server (porta 8080 exposta)
# ============================================================================
echo ""
echo "[3/3] Iniciando GPU Server..."

GPU_PORT=8080
GPU_LOG="/workspace/logs/gpu_server.log"

if curl -s http://localhost:$GPU_PORT/health > /dev/null 2>&1; then
    echo "  GPU Server ja esta rodando na porta $GPU_PORT"
else
    echo "  Iniciando GPU Server..."
    nohup python -m src.pod.gpu_server --port $GPU_PORT > $GPU_LOG 2>&1 &

    echo "  Aguardando GPU Server iniciar..."
    for i in {1..30}; do
        if curl -s http://localhost:$GPU_PORT/health > /dev/null 2>&1; then
            echo "  GPU Server pronto!"
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
echo "vLLM (interno):   $(curl -s http://localhost:8000/health > /dev/null && echo 'OK (8000)' || echo 'Aguardando...')"
echo "GPU Server:       $(curl -s http://localhost:8080/health > /dev/null && echo 'OK (8080)' || echo 'Aguardando...')"
echo ""
echo "=============================================="
echo "  IMPORTANTE: Expor porta 8080 no RunPod!"
echo ""
echo "  1. Va em 'Edit Pod' no RunPod"
echo "  2. Adicione porta TCP: 8080"
echo "  3. Salve e aguarde restart"
echo ""
echo "  Depois, acesse do Windows:"
echo "    curl http://195.26.233.70:8080/health"
echo "=============================================="
echo ""
echo "Logs:"
echo "  vLLM:       tail -f /workspace/logs/vllm.log"
echo "  GPU Server: tail -f /workspace/logs/gpu_server.log"
echo "=============================================="
