#!/bin/bash
# =============================================================================
# Variáveis de Ambiente para RunPod
# =============================================================================
# Source este arquivo antes de rodar qualquer script:
#   source deploy/env.sh
# =============================================================================

# Serviços (ajuste se estiver usando Docker Compose)
export VLLM_HOST="${VLLM_HOST:-localhost}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export MILVUS_HOST="${MILVUS_HOST:-localhost}"
export MILVUS_PORT="${MILVUS_PORT:-19530}"
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"

# HuggingFace
export HF_HOME="${HF_HOME:-/workspace/rag-pipeline/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"

# GPU
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16000}"

# Modelos
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-8B-AWQ}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
export RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"

# Milvus Collection
export MILVUS_COLLECTION="${MILVUS_COLLECTION:-leis_v3}"

# Python path
export PYTHONPATH="${PYTHONPATH}:/workspace/rag-pipeline"

echo "Variáveis de ambiente configuradas:"
echo "  VLLM_HOST=$VLLM_HOST:$VLLM_PORT"
echo "  MILVUS_HOST=$MILVUS_HOST:$MILVUS_PORT"
echo "  REDIS_HOST=$REDIS_HOST:$REDIS_PORT"
echo "  HF_HOME=$HF_HOME"
echo "  LLM_MODEL=$LLM_MODEL"
