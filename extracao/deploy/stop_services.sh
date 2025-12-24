#!/bin/bash
# =============================================================================
# Stop Services Script - Pipeline RAG
# =============================================================================

echo "Parando serviços..."

# Para Streamlit
pkill -f "streamlit" 2>/dev/null && echo "Streamlit: parado" || echo "Streamlit: não estava rodando"

# Para Celery
pkill -f "celery.*worker" 2>/dev/null && echo "Celery: parado" || echo "Celery: não estava rodando"

# Para vLLM
pkill -f "vllm.entrypoints" 2>/dev/null && echo "vLLM: parado" || echo "vLLM: não estava rodando"

# Para Redis
redis-cli shutdown 2>/dev/null && echo "Redis: parado" || echo "Redis: não estava rodando"

# Para Milvus (Docker)
docker stop milvus-standalone 2>/dev/null && echo "Milvus: parado" || echo "Milvus: não estava rodando"

echo ""
echo "Todos os serviços foram parados."
