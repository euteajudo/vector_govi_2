# =============================================================================
# Dockerfile - Pipeline RAG para Documentos Legais
# =============================================================================
# Build: docker build -t rag-pipeline .
# Run: docker run --gpus all -p 8501:8501 -p 8080:8080 rag-pipeline
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Define Python 3.11 como padrão
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Cria ambiente virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Instala PyTorch com CUDA 12.1
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copia requirements e instala dependências
COPY deploy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia código fonte
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY data/ /app/data/

# Copia arquivos de configuração
COPY pyproject.toml /app/

# Variáveis de ambiente
ENV VLLM_HOST=vllm
ENV VLLM_PORT=8000
ENV MILVUS_HOST=milvus-standalone
ENV MILVUS_PORT=19530
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Cria diretório para cache
RUN mkdir -p /app/.cache/huggingface

# Expõe portas
# 8501: Streamlit Dashboard
# 8080: FastAPI (futuro)
# 5555: Flower (Celery monitor)
EXPOSE 8501 8080 5555

# Comando padrão (pode ser sobrescrito)
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
