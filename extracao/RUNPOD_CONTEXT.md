# Contexto para Claude Code no RunPod

## O que é este projeto

Sistema RAG (Retrieval-Augmented Generation) para documentos legais brasileiros (leis, decretos, instruções normativas). Permite fazer perguntas sobre legislação e receber respostas com citações precisas.

## Arquitetura

```
PDF → Docling → SpanParser → ArticleOrchestrator (LLM) → ChunkMaterializer
                                                              ↓
                                                    ChunkEnricher (LLM)
                                                              ↓
Query → HyDE → BGE-M3 (embeddings) → Milvus (busca híbrida) → Reranker → LLM → Resposta
```

## Stack

| Componente | Tecnologia |
|------------|------------|
| LLM | Qwen 3 8B-AWQ (via vLLM) |
| Embeddings | BGE-M3 (FlagEmbedding) |
| Reranker | BGE-Reranker-v2-m3 |
| Vector DB | Milvus |
| Task Queue | Celery + Redis |
| Dashboard | Streamlit |

## Estado Atual

- ✅ Pipeline de extração funcionando
- ✅ Chunking com parent-child
- ✅ Busca híbrida (dense + sparse)
- ✅ Answer Generator com citações
- ✅ Dashboard Streamlit
- ⏳ Deploy no RunPod (em andamento)

## O que precisa ser feito no RunPod

### 1. Configurar ambiente
```bash
cd /workspace
unzip rag-pipeline.zip -d rag-pipeline
cd rag-pipeline
```

### 2. Instalar dependências
```bash
# Configurar cache no volume persistente
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
mkdir -p /workspace/models

# Criar symlink
ln -sf /workspace/models ~/.cache/huggingface

# Instalar dependências (o PyTorch já vem na imagem)
pip install docling openai FlagEmbedding sentence-transformers pymilvus
pip install celery redis streamlit fastapi uvicorn
pip install langgraph langchain-core pydantic rich tqdm httpx
```

### 3. Instalar e configurar vLLM
```bash
# Instalar vLLM
pip install vllm

# Iniciar vLLM (em background)
HF_HOME=/workspace/models python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B-AWQ \
    --max-model-len 16000 \
    --gpu-memory-utilization 0.85 \
    --port 8000 &
```

### 4. Instalar Milvus (opção Lite para teste)
```bash
pip install "pymilvus[model]"

# No código, usar:
# from pymilvus import MilvusClient
# client = MilvusClient("./milvus.db")  # Arquivo local
```

### 5. Iniciar Redis
```bash
apt-get update && apt-get install -y redis-server
redis-server --daemonize yes
```

### 6. Testar o sistema
```bash
cd /workspace/rag-pipeline
export PYTHONPATH=/workspace/rag-pipeline
python scripts/test_answer_generator.py --query "Quais os critérios de julgamento?"
```

### 7. Iniciar Dashboard
```bash
streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

## Estrutura do Projeto

```
/workspace/rag-pipeline/
├── src/
│   ├── rag/              # Answer Generator, citações
│   ├── search/           # Busca híbrida, HyDE, reranker
│   ├── parsing/          # SpanParser, ArticleOrchestrator
│   ├── chunking/         # ChunkMaterializer
│   ├── enrichment/       # ChunkEnricher, Celery tasks
│   ├── embeddings/       # BGE-M3
│   ├── llm/              # VLLMClient
│   ├── milvus/           # Schema Milvus
│   └── dashboard/        # Streamlit app
├── scripts/              # Scripts utilitários
├── data/                 # PDFs (L14133.pdf, INs)
└── deploy/               # Arquivos de deploy
```

## Variáveis de Ambiente Importantes

```bash
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export PYTHONPATH=/workspace/rag-pipeline
export VLLM_HOST=localhost
export VLLM_PORT=8000
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

## Documentos de Teste

- `data/L14133.pdf` - Lei 14.133/2021 (Nova Lei de Licitações)
- `data/INSTRUÇÃO NORMATIVA SEGES Nº 58_2022.pdf` - IN sobre ETP
- `data/INSTRUÇÃO NORMATIVA SEGES _ME Nº 65...pdf` - IN sobre pesquisa de preços

## Queries de Teste

```
"Quais os critérios de julgamento?"
"Quando o ETP pode ser dispensado?"
"Como fazer pesquisa de preços?"
"O que é contratação direta?"
```

## GPU Disponível

A100-SXM4-80GB - Excelente para:
- vLLM com Qwen 8B-AWQ (~6GB VRAM)
- BGE-M3 embeddings (~2GB VRAM)
- BGE-Reranker (~2GB VRAM)
- Sobra ~70GB para batching

## Problemas Conhecidos

1. **vLLM 0.13 + PyTorch 2.9**: Pode haver conflito de templates. Se `import vllm` falhar, reinstalar:
   ```bash
   pip install --force-reinstall vllm torch
   ```

2. **Milvus no RunPod**: Usar Milvus Lite (arquivo local) ou Zilliz Cloud

3. **Cache de modelos**: Sempre usar `/workspace/models` (volume persistente)
