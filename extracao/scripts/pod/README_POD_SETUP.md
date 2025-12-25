# Instruções para Configurar GPU Server no POD

## Contexto

O sistema RAG precisa de um **GPU Server** rodando no POD que expõe:
1. **Embeddings BGE-M3** (dense + sparse) na porta 8080
2. **Proxy para vLLM** local

O computador Windows vai acessar esse servidor via HTTP direto (sem SSH tunnel).

## Passo 1: Baixar o Script de Setup

Execute no terminal:

```bash
curl -sL https://dpaste.com/BSTJYSFN5.txt > /workspace/setup_pod.sh
```

## Passo 2: Executar o Setup

```bash
bash /workspace/setup_pod.sh
```

Isso vai:
- Criar o arquivo `/workspace/rag-pipeline/gpu_server.py`
- Instalar dependências: `fastapi uvicorn httpx pydantic FlagEmbedding`

## Passo 3: Verificar se vLLM está Rodando

O GPU Server precisa do vLLM rodando na porta 8000:

```bash
curl -s http://localhost:8000/health
```

Se não estiver rodando, inicie com:

```bash
# Verifica se container existe
docker ps -a | grep vllm

# Se não existir, cria:
docker run -d --name vllm --gpus all \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B-AWQ \
  --max-model-len 16000 \
  --gpu-memory-utilization 0.9

# Se existir mas estiver parado:
docker start vllm

# Verifica logs:
docker logs -f vllm
```

## Passo 4: Iniciar GPU Server

```bash
cd /workspace/rag-pipeline
python gpu_server.py --port 8080
```

O servidor deve mostrar:
```
INFO:     Iniciando GPU Server em 0.0.0.0:8080
INFO:     vLLM: http://localhost:8000
INFO:     Uvicorn running on http://0.0.0.0:8080
```

## Passo 5: Testar

No mesmo POD (outro terminal):

```bash
# Health check
curl http://localhost:8080/health

# Teste de embedding
curl -X POST http://localhost:8080/embed/hybrid \
  -H "Content-Type: application/json" \
  -d '{"texts": ["teste de embedding"], "batch_size": 1}'
```

## Passo 6: Confirmar Acesso Externo

A porta 8080 deve estar exposta no RunPod como TCP.
O acesso externo será via: `http://195.26.233.70:55278`

## Arquitetura

```
Windows (local)                          POD (RunPod)
================                         ==============

Pipeline HTTP     ───HTTP:55278──────►   GPU Server (:8080)
                                              │
Milvus (:19530)   ◄──────────────────────    │
                                              │
                                              ▼
                                         vLLM (:8000)
                                         Qwen3-8B-AWQ
                                              │
                                              ▼
                                         BGE-M3
                                         (FlagEmbedding)
```

## Troubleshooting

### GPU Server não inicia
```bash
# Verifica se a porta está livre
netstat -tlnp | grep 8080

# Mata processo se necessário
kill $(lsof -t -i:8080)
```

### vLLM não responde
```bash
# Verifica container
docker logs vllm --tail 50

# Reinicia se necessário
docker restart vllm
```

### Erro de GPU/CUDA
```bash
# Verifica GPU
nvidia-smi

# Verifica CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Código Completo do GPU Server

Se preferir criar manualmente o arquivo `gpu_server.py`:

```python
# Cole o conteúdo de https://dpaste.com/BSTJYSFN5.txt
# Ou baixe diretamente com curl
```
