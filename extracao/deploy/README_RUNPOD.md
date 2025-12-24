# Deploy no RunPod.io - Pipeline RAG para Documentos Legais

## Requisitos Mínimos

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| GPU | RTX 4090 (24GB) | A100 40GB |
| VRAM | 24GB | 40GB+ |
| RAM | 32GB | 64GB |
| Disco | 100GB | 200GB |
| CUDA | 12.1+ | 12.1+ |

### GPUs Recomendadas no RunPod

| GPU | VRAM | Custo/hora | Performance |
|-----|------|------------|-------------|
| RTX 4090 | 24GB | ~$0.44 | Boa |
| A100 40GB | 40GB | ~$1.14 | Excelente |
| A100 80GB | 80GB | ~$1.89 | Máxima |
| H100 80GB | 80GB | ~$3.89 | Premium |

---

## Opção 1: Deploy Rápido (Recomendado)

### Passo 1: Criar Pod no RunPod

1. Acesse [runpod.io](https://runpod.io)
2. Clique em **Deploy** → **GPU Pods**
3. Selecione:
   - **Template**: `RunPod Pytorch 2.1` ou `NVIDIA CUDA 12.1`
   - **GPU**: A100 40GB (recomendado)
   - **Container Disk**: 100GB
   - **Volume Disk**: 100GB (para modelos)

4. Configure portas expostas:
   ```
   8501/http  # Streamlit Dashboard
   8000/http  # vLLM API
   5555/http  # Flower (Celery)
   ```

5. Clique em **Deploy**

### Passo 2: Conectar via SSH

```bash
ssh root@<pod-ip> -i ~/.ssh/id_ed25519
```

### Passo 3: Executar Setup

```bash
# Baixar script de setup
curl -O https://raw.githubusercontent.com/SEU_REPO/main/deploy/runpod_setup.sh
chmod +x runpod_setup.sh
./runpod_setup.sh
```

### Passo 4: Upload do Código

**Via SCP (do seu PC):**
```bash
scp -r ./extracao/* root@<pod-ip>:/workspace/rag-pipeline/
```

**Via RunPod File Browser:**
1. No painel do RunPod, clique em "Connect" → "HTTP 8501"
2. Use o uploader web para enviar os arquivos

### Passo 5: Iniciar Serviços

```bash
cd /workspace/rag-pipeline/deploy
./start_services.sh
```

### Passo 6: Acessar Dashboard

No painel do RunPod, clique em **Connect** → **HTTP 8501**

---

## Opção 2: Deploy com Docker Compose

### Passo 1: Criar Pod (mesmas configurações acima)

### Passo 2: Upload do Código

```bash
scp -r ./extracao root@<pod-ip>:/workspace/rag-pipeline
```

### Passo 3: Iniciar com Docker Compose

```bash
cd /workspace/rag-pipeline/deploy
docker-compose up -d
```

### Passo 4: Verificar Status

```bash
docker-compose ps
docker-compose logs -f
```

---

## Estrutura de Arquivos no Pod

```
/workspace/rag-pipeline/
├── src/                    # Código fonte
│   ├── chunking/
│   ├── dashboard/
│   ├── embeddings/
│   ├── enrichment/
│   ├── extract/
│   ├── llm/
│   ├── milvus/
│   ├── parsing/
│   ├── rag/
│   └── search/
├── scripts/                # Scripts utilitários
├── data/                   # PDFs e dados
│   ├── L14133.pdf
│   └── ...
├── deploy/                 # Arquivos de deploy
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   └── *.sh
├── logs/                   # Logs dos serviços
├── .cache/huggingface/     # Cache de modelos
└── .env                    # Variáveis de ambiente
```

---

## Comandos Úteis

### Verificar GPU
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Monitor contínuo
```

### Logs dos Serviços
```bash
# vLLM
tail -f /workspace/rag-pipeline/logs/vllm.log

# Celery
tail -f /workspace/rag-pipeline/logs/celery.log

# Docker Compose
docker-compose logs -f vllm
docker-compose logs -f rag-app
```

### Testar Serviços
```bash
# vLLM
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

# Milvus
curl http://localhost:9091/healthz

# Redis
redis-cli ping
```

### Testar RAG
```bash
cd /workspace/rag-pipeline
source .venv/bin/activate
python scripts/test_answer_generator.py --query "Quais os critérios de julgamento?"
```

---

## Estimativa de Performance

### Com A100 40GB

| Operação | Tempo Estimado |
|----------|----------------|
| Startup vLLM | ~30s |
| Query simples (sem HyDE) | ~15s |
| Query completa (com HyDE) | ~30s |
| Enriquecimento por chunk | ~5s |

### Com RTX 4090 (24GB)

| Operação | Tempo Estimado |
|----------|----------------|
| Startup vLLM | ~45s |
| Query simples (sem HyDE) | ~25s |
| Query completa (com HyDE) | ~50s |
| Enriquecimento por chunk | ~10s |

---

## Troubleshooting

### vLLM não inicia
```bash
# Verificar VRAM disponível
nvidia-smi

# Reduzir uso de memória
export GPU_MEMORY_UTILIZATION=0.75
export MAX_MODEL_LEN=8000
```

### Milvus não conecta
```bash
# Reiniciar Milvus
docker restart milvus-standalone
sleep 30
curl http://localhost:9091/healthz
```

### Erro de importação Python
```bash
cd /workspace/rag-pipeline
export PYTHONPATH=$PYTHONPATH:/workspace/rag-pipeline
```

### Celery não processa tasks
```bash
# Verificar conexão Redis
redis-cli ping

# Reiniciar worker
pkill -f "celery.*worker"
celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=1
```

---

## Custos Estimados

### Sessão de Teste (2 horas)
| GPU | Custo |
|-----|-------|
| RTX 4090 | ~$0.88 |
| A100 40GB | ~$2.28 |

### Uso Diário (8 horas)
| GPU | Custo |
|-----|-------|
| RTX 4090 | ~$3.52 |
| A100 40GB | ~$9.12 |

---

## Próximos Passos

1. **Testar query básica**: Verificar se o sistema responde corretamente
2. **Indexar documentos**: Rodar pipeline com seus PDFs
3. **Benchmark**: Comparar latência com sua máquina local
4. **Ajustar parâmetros**: Otimizar `gpu_memory_utilization` e `max_model_len`
