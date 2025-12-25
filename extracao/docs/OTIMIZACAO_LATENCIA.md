# Otimização de Latência do Sistema RAG

**Data:** 25 de Dezembro de 2025
**Ambiente:** RunPod com A100-SXM4-80GB (81GB VRAM)

## Resumo Executivo

Este documento descreve o processo de investigação e solução de um problema de alta latência no sistema RAG para documentos legais brasileiros. A latência foi reduzida de **~40 segundos** para **~10 segundos** por query após a otimização.

---

## 1. Problema Identificado

### Sintoma
O sistema apresentava latência de **30-40 segundos** por query, apesar de rodar em uma GPU A100 com 80GB de VRAM e recursos computacionais abundantes.

### Métricas Iniciais
| Métrica | Valor |
|---------|-------|
| Latência total | ~40s |
| Retrieval (HyDE + busca + rerank) | ~31s |
| Geração LLM | ~6-9s |
| GPU disponível | A100-SXM4-80GB |
| VRAM livre | ~23GB (após vLLM) |

---

## 2. Processo de Diagnóstico

### 2.1 Teste de Componentes Individuais

Executamos testes isolados para identificar o gargalo:

```python
# Teste de latência por componente
=== 1. MILVUS (via túnel SSH) ===
Conexão + Load: 2.29s  ✓ OK

=== 2. vLLM ===
Resposta curta: 0.30s  ✓ OK

=== 3. BGE-M3 (Embeddings) ===
Carregar modelo: 12.41s  ⚠️ PROBLEMA
Gerar embedding: 1.39s   ✓ OK

=== 4. BGE-Reranker ===
Carregar reranker: 1.84s  ⚠️ PROBLEMA
Reranking: 7.25s          ⚠️ LENTO
```

### 2.2 Causa Raiz Identificada

**Os modelos BGE-M3 e BGE-Reranker estavam sendo recarregados a cada requisição.**

Análise do código em `src/search/hybrid_searcher.py`:

```python
@property
def embedder(self):
    if self._embedder is None:
        # PROBLEMA: Cria nova instância a cada requisição!
        from embeddings import BGEM3Embedder, EmbeddingConfig
        self._embedder = BGEM3Embedder(...)
    return self._embedder
```

O mesmo problema existia em:
- `src/search/contextual_retriever.py` (linha 141)
- `src/embeddings/bge_reranker.py`

---

## 3. Solução Implementada

### 3.1 Padrão Singleton para Modelos

Criamos um módulo `src/model_pool.py` que mantém os modelos como singletons:

```python
"""
Singleton pool para modelos compartilhados.
Mantém BGE-M3 e Reranker em memória para evitar recarregamento.
"""

_embedder = None
_reranker = None
_raw_embedder = None

def get_embedder():
    """Retorna wrapper BGEM3Embedder singleton."""
    global _embedder
    if _embedder is None:
        from embeddings import BGEM3Embedder, EmbeddingConfig
        _embedder = BGEM3Embedder(EmbeddingConfig(use_fp16=True))
        _embedder._ensure_initialized()
    return _embedder

def get_reranker():
    """Retorna wrapper BGEReranker singleton."""
    global _reranker
    if _reranker is None:
        from embeddings import BGEReranker, RerankerConfig
        _reranker = BGEReranker(RerankerConfig(use_fp16=True))
        _reranker._ensure_initialized()
    return _reranker
```

### 3.2 Atualização dos Componentes

Modificamos `hybrid_searcher.py` e `contextual_retriever.py` para usar o singleton:

```python
# ANTES
def _ensure_embedder(self):
    if self._embedder is None:
        from FlagEmbedding import BGEM3FlagModel
        self._embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# DEPOIS
def _ensure_embedder(self):
    if self._embedder is None:
        from model_pool import get_raw_embedder
        self._embedder = get_raw_embedder()
```

### 3.3 Warmup Server

Criamos um servidor FastAPI (`scripts/warmup_server.py`) que mantém os modelos carregados na GPU:

```bash
# Iniciar warmup server
nohup python scripts/warmup_server.py > /workspace/warmup.log 2>&1 &

# Endpoints disponíveis
GET  /health  - Health check
GET  /status  - Status dos modelos e GPU
POST /embed   - Gerar embeddings
POST /rerank  - Reranking de documentos
```

---

## 4. Resultados

### 4.1 Melhoria de Latência

| Cenário | Retrieval | Total | Melhoria |
|---------|-----------|-------|----------|
| Antes (cold a cada req) | 30.8s | 40s | - |
| Depois (1ª query, cold) | 18.9s | 32s | -20% |
| Depois (2ª+ query, warm) | **4.2s** | **10s** | **-75%** |

### 4.2 Teste de Múltiplas Queries

```
Query 1 (cold): 32.35s | Retrieval: 18901ms
Query 2 (warm):  9.76s | Retrieval:  4176ms  ← -78%
Query 3 (warm): 13.29s | Retrieval:  4417ms  ← -77%
```

### 4.3 Uso de VRAM

| Componente | VRAM |
|------------|------|
| vLLM (Qwen3-8B-AWQ) | ~58 GB (70%) |
| BGE-M3 + Reranker | ~2.2 GB |
| Livre | ~21 GB |

---

## 5. Aprendizados Importantes

### 5.1 Singleton vs Processo

**O padrão singleton funciona apenas dentro do mesmo processo Python.**

- ✅ Múltiplas queries no mesmo processo → modelos reutilizados
- ❌ Processos separados → cada um carrega seus próprios modelos

**Implicações:**
- O Dashboard Streamlit mantém o processo ativo → queries rápidas após warmup
- Scripts CLI criam novo processo a cada execução → sempre cold start
- O warmup server mantém modelos na GPU, mas não compartilha entre processos

### 5.2 Lazy Loading vs Eager Loading

O código original usava lazy loading (carrega quando necessário), o que é bom para economia de memória, mas ruim para latência em produção.

**Recomendação:** Para produção, fazer eager loading no startup da aplicação:

```python
# No início do servidor/dashboard
from model_pool import preload_models
preload_models()  # Carrega tudo de uma vez
```

### 5.3 Memória GPU vs RAM

Os modelos BGE-M3 e Reranker são carregados na **GPU (VRAM)**, não na RAM do sistema.

```python
# Verificar uso de VRAM
import torch
print(f'GPU alocada: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
```

### 5.4 Breakdown da Latência (warm)

| Componente | Tempo | % do Total |
|------------|-------|------------|
| HyDE (LLM gera doc hipotético) | ~3-4s | 30% |
| Embedding da query | ~0.5s | 5% |
| Busca Milvus (via túnel SSH) | ~1-2s | 15% |
| Reranking | ~2-3s | 25% |
| Geração LLM (resposta final) | ~5-7s | 50% |

---

## 6. Arquivos Modificados

```
src/
├── model_pool.py              # NOVO - Singleton pool
├── search/
│   ├── hybrid_searcher.py     # Usa singleton
│   └── contextual_retriever.py # Usa singleton

scripts/
└── warmup_server.py           # NOVO - Servidor de warmup
```

---

## 7. Como Usar em Produção

### 7.1 Para o Dashboard Streamlit

O Streamlit mantém o processo ativo, então basta garantir que os modelos são carregados no início:

```python
# No início de app.py
import streamlit as st
from model_pool import preload_models

@st.cache_resource
def init_models():
    preload_models()
    return True

init_models()
```

### 7.2 Para API FastAPI

```python
from fastapi import FastAPI
from model_pool import preload_models

app = FastAPI()

@app.on_event("startup")
async def startup():
    preload_models()
```

### 7.3 Para Scripts CLI

Para scripts que rodam uma vez e terminam, não há como evitar o cold start. Opções:

1. Aceitar o cold start (~15-20s extra na primeira query)
2. Manter um servidor intermediário (warmup_server) e chamá-lo via HTTP
3. Processar múltiplas queries no mesmo script

---

## 8. Comandos Úteis

```bash
# Verificar uso de VRAM
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Iniciar warmup server
export PYTHONPATH=/workspace/rag-pipeline/src
nohup python scripts/warmup_server.py > /workspace/warmup.log 2>&1 &

# Verificar status do warmup
curl http://localhost:8100/status

# Testar latência
python scripts/test_answer_generator.py --query "Sua pergunta"
```

---

## 9. Cache Semantico (Implementado!)

Apos a otimizacao de modelos, implementamos um **Cache Semantico com Busca Hibrida** que reduz ainda mais a latencia para queries similares.

### Resultados do Cache

| Cenario | Latencia | Melhoria |
|---------|----------|----------|
| Sem cache (warm) | ~10s | baseline |
| Cache HIT | ~50ms | **99.5%** |

### Como Funciona

1. **Busca Hibrida**: Combina embeddings densos (semantica) + esparsos (keywords)
2. **RRF Ranking**: Reciprocal Rank Fusion para combinar scores
3. **Redis + Milvus**: Milvus armazena embeddings, Redis armazena respostas

### Vantagem da Busca Hibrida

```
Query cacheada:  "O que diz o Art. 72?"
Nova query:      "O que diz o Art. 75?"

Busca apenas densa: Similaridade 0.96 → Retornaria ERRADO
Busca hibrida:      RRF score 0.018 → NAO retorna (correto!)
```

**Documentacao completa:** `docs/CACHE_SEMANTICO.md`

---

## 10. Proximos Passos (Sugestoes)

1. ~~**Integrar warmup no Streamlit:**~~ ✅ Feito
2. ~~**Cache de respostas:**~~ ✅ Implementado (Cache Semantico)
3. **Otimizar HyDE:** Considerar desabilitar HyDE para queries simples (economia de ~3-4s)
4. **Milvus local:** Rodar Milvus no mesmo pod para eliminar latencia de rede
5. **Fine-tuning:** Usar queries curadas do cache para fine-tuning do modelo

---

## 11. Conclusao

A otimização de latência foi bem-sucedida, reduzindo o tempo de resposta de **~40s para ~10s** (melhoria de 75%). A causa raiz era o recarregamento desnecessário de modelos a cada requisição, resolvido com o padrão singleton.

O aprendizado principal é que **modelos de ML devem ser tratados como recursos compartilhados** em aplicações de produção, não como objetos descartáveis criados a cada request.
