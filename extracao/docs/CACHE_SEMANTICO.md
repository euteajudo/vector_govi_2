# Cache Semantico com Busca Hibrida

## Visao Geral

O Cache Semantico e um sistema de cache inteligente que armazena respostas do RAG e as reutiliza para perguntas similares. Diferente de um cache tradicional (que usa match exato), o cache semantico usa **embeddings** para encontrar perguntas parecidas, mesmo que escritas de forma diferente.

### Exemplo Pratico

```
Usuario A pergunta: "Quando o ETP pode ser dispensado?"
→ Sistema processa (busca + LLM) em ~10 segundos
→ Resposta salva no cache

Usuario B pergunta: "Em quais casos posso dispensar o ETP?"
→ Cache encontra pergunta similar (92% similaridade)
→ Retorna resposta em ~50ms (200x mais rapido!)
```

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         FLUXO DE QUERY                          │
└─────────────────────────────────────────────────────────────────┘

  Usuario faz pergunta
         │
         ▼
  ┌──────────────┐
  │ AnswerGenerator │
  └──────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │            SEMANTIC CACHE                     │
  │  ┌────────────────┐    ┌─────────────────┐   │
  │  │    MILVUS      │    │     REDIS       │   │
  │  │                │    │                 │   │
  │  │ query_cache_v2 │    │ DB 1 (cache)    │   │
  │  │ - dense_emb    │    │ - response:hash │   │
  │  │ - sparse_emb   │    │ - feedback:hash │   │
  │  │ - query_hash   │    │ - stats:*       │   │
  │  └────────────────┘    └─────────────────┘   │
  └──────────────────────────────────────────────┘
         │
         ├── HIT? ──────────────────────────────────► Retorna resposta cached
         │                                            (latencia ~50ms)
         │
         └── MISS? ──► Pipeline RAG completo ──────► Salva no cache
                       (busca + rerank + LLM)         (latencia ~10s)
                       (latencia ~10s)
```

## Por que Busca Hibrida?

### O Problema da Busca Apenas Densa

A busca densa (semantic search) e excelente para entender o **significado** das palavras, mas pode confundir termos especificos:

```
Query cacheada:  "O que diz o Art. 72 sobre dispensa?"
Nova query:      "O que diz o Art. 75 sobre dispensa?"

Similaridade semantica: 0.96 (muito alta!)
→ Cache retornaria resposta ERRADA (Art. 72 em vez de Art. 75)
```

### A Solucao: Busca Hibrida (Dense + Sparse)

Combinamos dois tipos de busca:

| Tipo | O que captura | Exemplo |
|------|---------------|---------|
| **Dense** | Significado semantico | "dispensar ETP" ≈ "ETP pode ser dispensado" |
| **Sparse** | Keywords exatas (BM25-like) | "Art. 75" ≠ "Art. 72" |

```
Query cacheada:  "O que diz o Art. 72 sobre dispensa?"
Nova query:      "O que diz o Art. 75 sobre dispensa?"

Dense score:  0.96 (alta - semanticamente similares)
Sparse score: 0.30 (baixa - "72" vs "75" sao diferentes)
RRF final:    0.018 (abaixo do threshold 0.025)
→ Cache NAO retorna resposta (correto!)
```

## Como Funciona o RRF (Reciprocal Rank Fusion)

O RRF combina os rankings de multiplas buscas usando a formula:

```
RRF(doc) = Σ 1/(k + rank_i(doc))
```

Onde:
- `k = 60` (constante padrao)
- `rank_i(doc)` = posicao do documento na busca i

### Exemplos de RRF Score

| Cenario | Rank Dense | Rank Sparse | RRF Score |
|---------|------------|-------------|-----------|
| Match perfeito em ambas | #1 | #1 | 1/61 + 1/61 = **0.0328** |
| Bom match | #1 | #3 | 1/61 + 1/63 = **0.0322** |
| Match apenas semantico | #1 | #10 | 1/61 + 1/70 = **0.0307** |
| Match fraco | #5 | #8 | 1/65 + 1/68 = **0.0301** |

**Threshold atual: 0.025** (ajustavel em `CacheConfig.similarity_threshold`)

## Estrutura do Codigo

### Arquivos Principais

```
src/
├── cache/
│   ├── __init__.py              # Exporta SemanticCache, CacheConfig
│   └── semantic_cache.py        # Implementacao principal
│
├── rag/
│   └── answer_generator.py      # Integra cache no pipeline
│
└── dashboard/
    └── app.py                   # Pagina de monitoramento + feedback
```

### Collection Milvus: `query_cache_v2`

```python
fields = [
    # Identificacao
    FieldSchema(name="id", dtype=INT64, is_primary=True, auto_id=True),
    FieldSchema(name="query_hash", dtype=VARCHAR, max_length=64),
    FieldSchema(name="query_text", dtype=VARCHAR, max_length=2000),

    # Embeddings para busca hibrida
    FieldSchema(name="dense_embedding", dtype=FLOAT_VECTOR, dim=1024),   # BGE-M3
    FieldSchema(name="sparse_embedding", dtype=SPARSE_FLOAT_VECTOR),     # Lexical

    # Metadados
    FieldSchema(name="created_at", dtype=INT64),

    # Feedback RLHF
    FieldSchema(name="likes", dtype=INT64),
    FieldSchema(name="dislikes", dtype=INT64),
    FieldSchema(name="times_served", dtype=INT64),
]
```

### Indices

| Campo | Tipo de Indice | Metrica |
|-------|----------------|---------|
| `dense_embedding` | HNSW (M=16, ef=256) | COSINE |
| `sparse_embedding` | SPARSE_INVERTED_INDEX | IP (Inner Product) |

## Fluxo Detalhado

### 1. Cache GET (Busca)

```python
def get(self, query: str) -> Optional[dict]:
    # 1. Gera embeddings da query
    dense, sparse = self._get_embeddings(query)

    # 2. Prepara busca hibrida
    dense_search = AnnSearchRequest(
        data=[dense],
        anns_field="dense_embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=5,
    )

    sparse_search = AnnSearchRequest(
        data=[sparse],
        anns_field="sparse_embedding",
        param={"metric_type": "IP"},
        limit=5,
    )

    # 3. Executa busca hibrida com RRF
    results = collection.hybrid_search(
        reqs=[dense_search, sparse_search],
        ranker=RRFRanker(k=60),
        limit=1,
    )

    # 4. Verifica threshold
    if results[0][0].score < 0.025:  # RRF threshold
        return None  # MISS

    # 5. Busca resposta no Redis
    query_hash = results[0][0].entity.get("query_hash")
    response = redis.get(f"response:{query_hash}")

    return json.loads(response)  # HIT!
```

### 2. Cache SET (Armazenamento)

```python
def set(self, query: str, response: dict, ttl: int = 86400):
    # 1. Gera hash e embeddings
    query_hash = sha256(query.lower())[:32]
    dense, sparse = self._get_embeddings(query)

    # 2. Insere no Milvus
    collection.insert([{
        "query_hash": query_hash,
        "query_text": query,
        "dense_embedding": dense,
        "sparse_embedding": sparse,
        "created_at": timestamp,
        "likes": 0,
        "dislikes": 0,
        "times_served": 0,
    }])

    # 3. Salva resposta no Redis (com TTL)
    redis.setex(
        f"response:{query_hash}",
        ttl,  # 24 horas padrao
        json.dumps(response)
    )
```

## Sistema de Feedback (RLHF Simplificado)

O cache implementa um sistema de feedback que permite:

1. **Usuarios avaliam respostas** com like/dislike
2. **Dashboard mostra** queries com problemas (muitos dislikes)
3. **Administrador pode** invalidar respostas ruins

### Metricas Coletadas

```python
# Redis keys
"feedback:{query_hash}:likes"      # Contador de likes
"feedback:{query_hash}:dislikes"   # Contador de dislikes
"served:{query_hash}"              # Vezes que foi servida do cache
"stats:cache_hits"                 # Total de hits
"stats:cache_misses"               # Total de misses
```

### Dashboard de Monitoramento

Pagina `/Cache` no dashboard Streamlit mostra:

- **Estatisticas gerais**: queries cacheadas, hit rate, memoria
- **Feedback**: likes, dislikes, taxa de aprovacao
- **Crescimento**: grafico de queries adicionadas por dia
- **Top queries**: respostas mais bem avaliadas
- **Queries com problemas**: candidatas a revisao/invalidacao

## Configuracao

```python
@dataclass
class CacheConfig:
    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "query_cache_v2"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1  # DB separado para cache

    # Comportamento
    similarity_threshold: float = 0.025  # RRF score minimo
    ttl_seconds: int = 86400             # 24 horas
    max_cache_size: int = 10000          # Maximo de queries

    # Embedding
    embedding_dim: int = 1024  # BGE-M3
```

## Integracao com AnswerGenerator

O cache e integrado automaticamente:

```python
class AnswerGenerator:
    def generate(self, query: str, skip_cache: bool = False):
        # 1. Verifica cache (se habilitado)
        if self.config.use_cache and not skip_cache and self.cache:
            cached = self.cache.get(query)
            if cached:
                return AnswerResponse.from_dict(cached)

        # 2. Pipeline RAG completo (busca + rerank + LLM)
        search_result = self.searcher.search(query)
        context = self._build_context(search_result.hits)
        answer = self._generate_answer(query, context)

        # 3. Salva no cache
        if self.config.use_cache and self.cache:
            self.cache.set(query, response.to_dict())

        return response
```

### Configuracoes de Geracao

```python
# Com cache (padrao)
config = GenerationConfig.default()
config.use_cache = True

# Sem cache (forca nova geracao)
response = generator.generate(query, skip_cache=True)

# Desabilita cache completamente
config.use_cache = False
```

## Consideracoes de Performance

### Latencia

| Cenario | Latencia | Componentes |
|---------|----------|-------------|
| Cache HIT | ~50ms | embedding(30ms) + milvus(10ms) + redis(5ms) |
| Cache MISS | ~10s | embedding(30ms) + busca(4s) + rerank(2s) + LLM(4s) |

### Armazenamento

| Componente | Por Query | 10K Queries |
|------------|-----------|-------------|
| Milvus (dense) | ~4KB | ~40MB |
| Milvus (sparse) | ~2KB | ~20MB |
| Redis (response) | ~5KB | ~50MB |
| **Total** | ~11KB | ~110MB |

### Escalabilidade

- **Milvus**: Suporta milhoes de vetores com HNSW
- **Redis**: Memoria configuravel, LRU automatico
- **BGE-M3**: Batching para multiplas queries

## Troubleshooting

### Cache nao esta funcionando

```bash
# Verifica conexao Milvus (tunel SSH necessario)
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port=19530); print('OK')"

# Verifica Redis
redis-cli -n 1 ping
```

### Hit rate muito baixo

1. **Threshold muito alto**: Reduza `similarity_threshold` (ex: 0.020)
2. **Queries muito diversas**: Normal no inicio, melhora com uso
3. **Sparse dominando**: Verifique se queries tem muitos termos tecnicos

### Respostas incorretas do cache

```python
# Invalida query especifica
cache.invalidate("query problematica aqui")

# Limpa cache inteiro (cuidado!)
cache.clear()
```

## Valor de Negocio

O cache semantico oferece:

1. **Reducao de custos**: Menos chamadas ao LLM (economia de tokens)
2. **Menor latencia**: 200x mais rapido para queries similares
3. **Base de conhecimento**: Queries + respostas curadas pelos usuarios
4. **Insights de uso**: Quais perguntas sao mais frequentes
5. **Qualidade**: Feedback permite melhorar respostas ao longo do tempo

### Estrategia de Crescimento

```
Fase 1 (Lancamento):
- Cache vazio, todos os requests sao MISS
- Usuarios fazem perguntas, cache cresce organicamente

Fase 2 (Crescimento):
- Hit rate aumenta conforme queries se repetem
- Feedback identifica respostas problematicas
- Administrador curadoria queries com muitos dislikes

Fase 3 (Maturidade):
- Hit rate estabiliza em 40-60%
- Base de queries "curadas" com alta aprovacao
- Possibilidade de usar dados para fine-tuning
```

## Referencias

- [BGE-M3: Multi-Functionality, Multi-Linguality, Multi-Granularity](https://arxiv.org/abs/2402.03216)
- [Milvus Hybrid Search](https://milvus.io/docs/hybrid_search.md)
- [RRF: Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
