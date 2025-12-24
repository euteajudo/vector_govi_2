#!/bin/bash
# =============================================================================
# Start Services Script - Pipeline RAG (RunPod)
# =============================================================================
# Imagem: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Uso: chmod +x start_services.sh && ./start_services.sh
# =============================================================================

set -e

PROJECT_DIR="/workspace/rag-pipeline"
cd $PROJECT_DIR

# Carrega variáveis de ambiente
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"

echo "=============================================="
echo "  Iniciando Serviços - Pipeline RAG"
echo "=============================================="
echo ""

# Criar diretório de logs
mkdir -p $PROJECT_DIR/logs

# -----------------------------------------------------------------------------
# 1. Verificar dependências Python
# -----------------------------------------------------------------------------
echo "[1/6] Verificando dependências Python..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Instalando dependências..."
    pip install -q -r deploy/requirements.txt
fi
echo "Dependências: OK"

# -----------------------------------------------------------------------------
# 2. Iniciar Redis (background)
# -----------------------------------------------------------------------------
echo "[2/6] Iniciando Redis..."
if ! pgrep -x "redis-server" > /dev/null; then
    redis-server --daemonize yes --dir $PROJECT_DIR/data --appendonly yes
    sleep 2
fi
redis-cli ping && echo "Redis: OK"

# -----------------------------------------------------------------------------
# 3. Iniciar Milvus Standalone (Docker)
# -----------------------------------------------------------------------------
echo "[3/6] Iniciando Milvus..."
if ! docker ps 2>/dev/null | grep -q milvus-standalone; then
    # Remove container antigo se existir
    docker rm -f milvus-standalone 2>/dev/null || true

    docker run -d --name milvus-standalone \
        -p 19530:19530 -p 9091:9091 \
        -v ${PROJECT_DIR}/data/milvus:/var/lib/milvus \
        milvusdb/milvus:v2.4-latest \
        milvus run standalone

    echo "Aguardando Milvus iniciar (30s)..."
    sleep 30
fi

# Verificar Milvus
for i in {1..10}; do
    if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "Milvus: OK"
        break
    fi
    echo "Aguardando Milvus... ($i/10)"
    sleep 5
done

# -----------------------------------------------------------------------------
# 4. Iniciar vLLM (background)
# -----------------------------------------------------------------------------
echo "[4/6] Iniciando vLLM..."
if ! pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    # Instalar vLLM se necessário
    if ! python -c "import vllm" 2>/dev/null; then
        echo "Instalando vLLM..."
        pip install -q vllm
    fi

    nohup python -m vllm.entrypoints.openai.api_server \
        --model ${LLM_MODEL:-Qwen/Qwen3-8B-AWQ} \
        --max-model-len ${MAX_MODEL_LEN:-16000} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.85} \
        --dtype auto \
        --trust-remote-code \
        --port 8000 \
        > $PROJECT_DIR/logs/vllm.log 2>&1 &

    echo "Aguardando vLLM carregar modelo (~60s)..."
    echo "Acompanhe: tail -f $PROJECT_DIR/logs/vllm.log"

    # Aguarda vLLM ficar pronto
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM: OK"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "vLLM: Timeout (verifique logs/vllm.log)"
        fi
        sleep 5
    done
else
    echo "vLLM: já está rodando"
fi

# -----------------------------------------------------------------------------
# 5. Iniciar Celery Worker (background)
# -----------------------------------------------------------------------------
echo "[5/6] Iniciando Celery Worker..."
if ! pgrep -f "celery.*worker" > /dev/null; then
    nohup celery -A src.enrichment.celery_app worker \
        --loglevel=info \
        --concurrency=1 \
        > $PROJECT_DIR/logs/celery.log 2>&1 &
    sleep 3
fi
echo "Celery Worker: OK"

# -----------------------------------------------------------------------------
# 6. Iniciar Streamlit Dashboard
# -----------------------------------------------------------------------------
echo "[6/6] Iniciando Streamlit Dashboard..."
if ! pgrep -f "streamlit.*app.py" > /dev/null; then
    nohup streamlit run src/dashboard/app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        > $PROJECT_DIR/logs/streamlit.log 2>&1 &
    sleep 3
fi
echo "Streamlit: OK"

echo ""
echo "=============================================="
echo "  TODOS OS SERVIÇOS INICIADOS!"
echo "=============================================="
echo ""
echo "Serviços disponíveis:"
echo "  - Dashboard: http://localhost:8501 (ou porta HTTP do RunPod)"
echo "  - vLLM API:  http://localhost:8000"
echo "  - Milvus:    localhost:19530"
echo "  - Redis:     localhost:6379"
echo ""
echo "Logs:"
echo "  - vLLM:      tail -f $PROJECT_DIR/logs/vllm.log"
echo "  - Celery:    tail -f $PROJECT_DIR/logs/celery.log"
echo "  - Streamlit: tail -f $PROJECT_DIR/logs/streamlit.log"
echo ""
echo "Para parar todos os serviços:"
echo "  ./deploy/stop_services.sh"
echo ""
