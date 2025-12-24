#!/bin/bash
# =============================================================================
# RunPod Setup Script - Pipeline RAG para Documentos Legais
# =============================================================================
# Imagem: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Uso: chmod +x runpod_setup.sh && ./runpod_setup.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  RunPod Setup - Pipeline RAG Legal"
echo "=============================================="

# -----------------------------------------------------------------------------
# 1. Verificar GPU
# -----------------------------------------------------------------------------
echo ""
echo "[1/7] Verificando GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# -----------------------------------------------------------------------------
# 2. Verificar Python e PyTorch
# -----------------------------------------------------------------------------
echo "[2/7] Verificando ambiente Python..."
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# -----------------------------------------------------------------------------
# 3. Criar diretório do projeto
# -----------------------------------------------------------------------------
echo "[3/7] Criando diretório do projeto..."
PROJECT_DIR="/workspace/rag-pipeline"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Criar estrutura
mkdir -p src scripts data deploy logs .cache/huggingface

# -----------------------------------------------------------------------------
# 4. Instalar dependências do sistema
# -----------------------------------------------------------------------------
echo "[4/7] Instalando dependências do sistema..."
apt-get update -qq
apt-get install -y -qq redis-server tmux htop

# -----------------------------------------------------------------------------
# 5. Instalar dependências Python
# -----------------------------------------------------------------------------
echo "[5/7] Instalando dependências Python..."

# Criar requirements mínimo se não existir
if [ ! -f "$PROJECT_DIR/deploy/requirements.txt" ]; then
    cat << 'EOF' > /tmp/requirements.txt
# Core
docling>=2.15.0
openai>=1.50.0
FlagEmbedding>=1.3.0
sentence-transformers>=3.0.0
pymilvus>=2.5.0

# Task Queue
celery>=5.4.0
redis>=5.0.0
flower>=2.0.0

# API & Web
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
streamlit>=1.30.0

# Framework
langgraph>=0.2.0
langchain-core>=0.3.0

# Data
pydantic>=2.10.0
pydantic-settings>=2.0.0
pandas>=2.0.0

# Utils
python-dotenv>=1.0.0
rich>=13.0.0
tqdm>=4.66.0
httpx>=0.27.0
structlog>=24.0.0
EOF
    pip install -q -r /tmp/requirements.txt
else
    pip install -q -r $PROJECT_DIR/deploy/requirements.txt
fi

# -----------------------------------------------------------------------------
# 6. Baixar modelos HuggingFace
# -----------------------------------------------------------------------------
echo "[6/7] Baixando modelos HuggingFace (pode demorar ~10min)..."

export HF_HOME="$PROJECT_DIR/.cache/huggingface"

python << 'EOF'
import os
os.environ["HF_HOME"] = "/workspace/rag-pipeline/.cache/huggingface"

from huggingface_hub import snapshot_download

models = [
    ("Qwen/Qwen3-8B-AWQ", "LLM principal"),
    ("BAAI/bge-m3", "Embeddings"),
    ("BAAI/bge-reranker-v2-m3", "Reranker"),
]

for model, desc in models:
    print(f"Baixando {model} ({desc})...")
    try:
        snapshot_download(model, cache_dir=os.environ["HF_HOME"])
        print(f"  ✓ OK")
    except Exception as e:
        print(f"  ✗ Erro: {e}")

print("\n✓ Modelos baixados!")
EOF

# -----------------------------------------------------------------------------
# 7. Configurar variáveis de ambiente
# -----------------------------------------------------------------------------
echo "[7/7] Configurando ambiente..."

cat << 'EOF' > $PROJECT_DIR/.env
# RunPod Environment
HF_HOME=/workspace/rag-pipeline/.cache/huggingface
TRANSFORMERS_CACHE=/workspace/rag-pipeline/.cache/huggingface

# Serviços
VLLM_HOST=localhost
VLLM_PORT=8000
MILVUS_HOST=localhost
MILVUS_PORT=19530
REDIS_HOST=localhost
REDIS_PORT=6379

# GPU
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=16000

# Modelos
LLM_MODEL=Qwen/Qwen3-8B-AWQ
EOF

# Adicionar ao bashrc
echo "source $PROJECT_DIR/.env" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$PROJECT_DIR" >> ~/.bashrc
echo "cd $PROJECT_DIR" >> ~/.bashrc

echo ""
echo "=============================================="
echo "  SETUP CONCLUÍDO!"
echo "=============================================="
echo ""
echo "Próximos passos:"
echo ""
echo "1. Faça upload do código para $PROJECT_DIR"
echo "   Use o File Browser do RunPod ou:"
echo "   scp -i ~/.ssh/runpod_ed25519 -r ./extracao/* root@<pod-ip>:$PROJECT_DIR/"
echo ""
echo "2. Inicie os serviços:"
echo "   cd $PROJECT_DIR/deploy"
echo "   ./start_services.sh"
echo ""
echo "3. Acesse o Dashboard:"
echo "   http://<pod-ip>:8501"
echo ""
