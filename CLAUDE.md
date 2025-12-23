# 📋 Pipeline de Extração RAG para Leis - Documentação de Decisões

> **Projeto**: Sistema RAG para orgaos publicos
> **Data de Inicio**: 21/12/2024
> **Status**: Fase 4 - RAG Completo (Em Progresso)

---

## 🎯 Objetivo do Projeto

Desenvolver um sistema RAG (Retrieval-Augmented Generation) completo e comercializável para órgãos públicos brasileiros, começando pela extração e indexação de documentos legais (leis, decretos, instruções normativas).

---

## ⚖️ Requisitos de Licenciamento

| Requisito           | Decisão                                        |
| ------------------- | ---------------------------------------------- |
| Licenças permitidas | Apache 2.0, MIT, BSD, PostgreSQL License       |
| Licenças proibidas  | GPL, AGPL, SSPL, proprietárias                 |
| Motivo              | Produto comercial para venda a órgãos públicos |

---

## 🛠️ Stack Tecnológica Definida

### Backend

| Componente    | Tecnologia | Versão | Licença | Justificativa               |
| ------------- | ---------- | ------ | ------- | --------------------------- |
| Linguagem     | Python     | 3.11+  | PSF     | Expertise da equipe         |
| Framework API | FastAPI    | latest | MIT     | Performance, tipagem, async |
| Validação     | Pydantic   | v2     | MIT     | Integração nativa FastAPI   |

### Extração de Documentos

| Componente     | Tecnologia | Versão | Licença    | Justificativa                             |
| -------------- | ---------- | ------ | ---------- | ----------------------------------------- |
| Parser PDF     | Docling    | 2.15+  | MIT (IBM)  | Preserva hierarquia, markdown estruturado |
| OCR (fallback) | PaddleOCR  | latest | Apache 2.0 | Multilíngue, português                    |

### Framework Agêntico

| Componente   | Tecnologia | Versão | Licença | Justificativa                             |
| ------------ | ---------- | ------ | ------- | ----------------------------------------- |
| Orquestração | LangGraph  | 0.2+   | MIT     | Grafos de estado, checkpointing, flexível |

### LLM

| Componente      | Tecnologia        | Versão  | Licença    | Justificativa                                   |
| --------------- | ----------------- | ------- | ---------- | ----------------------------------------------- |
| Extracao        | **Qwen 3 8B-AWQ** | latest  | Apache 2.0 | Extracao JSON estruturado (94% qualidade)       |
| Enriquecimento  | **Qwen 3 4B-AWQ** | latest  | Apache 2.0 | context_header, thesis, questions (2x mais rapido) |
| Runtime Prod    | **vLLM**          | 0.13+   | Apache 2.0 | Docker, API OpenAI-compatible, quantizacao AWQ  |
| Hardware        | GPU 12GB          | -       | -          | 8B-AWQ: 5.7GB, 4B-AWQ: 2.5GB (ambos cabem)      |

**Decisao**: Usar **dois modelos especializados** por tarefa:
- **Qwen 3 8B-AWQ**: Extracao (tarefa complexa, precisa de capacidade)
- **Qwen 3 4B-AWQ**: Enriquecimento (tarefa simples, precisa de velocidade)

**Estrategia de Roteamento de Modelos**:

O vLLM nao suporta multiplos modelos simultaneamente na mesma GPU. Estrategia adotada:

| Estrategia | Descricao |
|------------|-----------|
| **Pipeline Sequencial** | Trocar modelo entre fases do pipeline |
| Fase 1 (Extracao) | vLLM com 8B-AWQ → PDF para JSON |
| Fase 2 (Enriquecimento) | vLLM com 4B-AWQ → Chunks enriquecidos |
| Fase 3 (Embedding) | BGE-M3 (CPU/GPU separada) |

**Troca de Modelo** (via script ou docker-compose):
```bash
# Trocar para 8B (extracao)
docker stop vllm && docker rm vllm
docker run -d --name vllm --gpus all -v huggingface-cache:/root/.cache/huggingface \
  -p 8000:8000 vllm/vllm-openai:latest --model Qwen/Qwen3-8B-AWQ

# Trocar para 4B (enriquecimento)
docker stop vllm && docker rm vllm
docker run -d --name vllm --gpus all -v huggingface-cache:/root/.cache/huggingface \
  -p 8000:8000 vllm/vllm-openai:latest --model Qwen/Qwen3-4B-AWQ
```

**Modelos no Cache Docker** (volume `huggingface-cache`):
```
models--Qwen--Qwen3-4B-AWQ  →  2.5GB
models--Qwen--Qwen3-8B-AWQ  →  5.7GB
```

**Vantagens vLLM em Producao**:

- Continuous batching (maior throughput)
- PagedAttention (uso eficiente de VRAM)
- API compativel com OpenAI (facil migracao)
- Tensor parallelism para multiplas GPUs
- Quantizacao nativa (AWQ, GPTQ)

**Justificativa dos modelos** (21/12/2024 - apos benchmarks extensivos):

- **8B para Extracao**: Unico modelo local que extraiu corretamente alineas (sub_items)
- **4B para Enriquecimento**: Mesma qualidade que 8B, porem 2x mais rapido
- Ambos com 256K de contexto (8x mais que Qwen 2.5)
- Licenca Apache 2.0 (100% comercial)
- Forte em portugues juridico

### Embeddings & Reranking

| Componente | Tecnologia             | Versão | Licença    | Justificativa                               |
| ---------- | ---------------------- | ------ | ---------- | ------------------------------------------- |
| Embedding  | **BGE-M3**             | latest | Apache 2.0 | Multilíngue, 8k contexto, híbrido           |
| Reranker   | **bge-reranker-v2-m3** | latest | Apache 2.0 | Cross-encoder multilíngue, melhora precisão |
| Runtime    | FlagEmbedding          | latest | Apache 2.0 | Biblioteca oficial BAAI                     |

**IMPORTANTE: Onde cada componente roda**

```
┌─────────────────────────────────────────────────────────────────┐
│                        COMPUTADOR                               │
│                                                                 │
│  ┌─────────────────────┐     ┌─────────────────────────────┐   │
│  │   Docker Container  │     │      Python Local           │   │
│  │       (vLLM)        │     │   (FlagEmbedding)           │   │
│  │                     │     │                             │   │
│  │  Qwen 4B/8B (LLM)   │     │  BGE-M3 (embeddings)        │   │
│  │  API: localhost:8000│     │  BGE-Reranker (rerank)      │   │
│  └─────────────────────┘     └─────────────────────────────┘   │
│           │                              │                      │
│           └──────────┬───────────────────┘                      │
│                      ▼                                          │
│              ┌───────────────┐                                  │
│              │  GPU (VRAM)   │  ← Compartilham GPU              │
│              └───────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

- **vLLM (Docker)**: Serve LLMs via API HTTP, roda no container
- **FlagEmbedding (Local)**: Carrega BGE-M3 e Reranker diretamente no Python

**Estrategia de Retrieval 2-Stage** (Testado 22/12/2024):

```
Query → BGE-M3 (Stage 1) → Top 10 → BGE-Reranker (Stage 2) → Top 3
           ↓                              ↓
     Busca rapida              Reordenacao precisa
     (bi-encoder)              (cross-encoder)
```

| Stage | Componente | Funcao | Velocidade |
|-------|------------|--------|------------|
| 1 | BGE-M3 (dense + sparse) | Busca inicial no Milvus | Rapido |
| 2 | BGE-Reranker-v2-m3 | Reordena resultados | Lento mas preciso |

**Benchmark 2-Stage Retrieval** (22/12/2024):

| Query | Stage 1 Top 1 | Stage 2 Top 1 | Melhoria |
|-------|---------------|---------------|----------|
| "O que e ETP?" | Art-6 (0.032) | Art-3 (0.80) | Promoveu 5→1 |
| "ETP dispensado?" | Art-14 (0.033) | Art-14 (0.98) | Confirmou 98% |
| "contratacoes correlatas" | Art-9 (0.032) | Art-3 (0.87) | Promoveu 3→1 |

O reranker **corrige** o ranking inicial, promovendo documentos relevantes.

**Benchmark ColBERT vs Cross-Encoder** (22/12/2024):

Testamos dois metodos de reranking para documentos juridicos:

| Metodo | Tecnica | Score Medio | Concordancia |
|--------|---------|-------------|--------------|
| **Cross-Encoder** (BGE-Reranker) | Query+Doc juntos | **0.91** | 80% |
| ColBERT (MaxSim) | Late interaction | 0.62 | 80% |

Resultados por query:

| Query | Cross-Encoder | ColBERT | Acordo |
|-------|---------------|---------|--------|
| "O que e ETP?" | Art-3 (0.80) | Art-3 (0.62) | ✓ |
| "Quando ETP dispensado?" | Art-14 (0.98) | Art-14 (0.63) | ✓ |
| "contratacoes interdependentes" | Art-3 (0.87) | Art-3 (0.47) | ✓ |
| "responsaveis elaboracao ETP" | Art-8 (0.97) | Art-8 (0.67) | ✓ |
| "sistema ETP digital funciona?" | Art-17 (0.96) | Art-4 (0.72) | ✗ |

**Decisao**: Manter **Cross-Encoder (BGE-Reranker)** como reranker principal:
- Scores mais altos e discriminativos (0.91 vs 0.62)
- 80% de concordancia com ColBERT
- Velocidade similar em producao

**ColBERT** disponivel como alternativa para queries com termos tecnicos exatos.

**Campos de Enriquecimento** (Contextual Retrieval):

| Campo | Descricao | Usado em |
|-------|-----------|----------|
| `text` | Texto original do artigo | Armazenamento |
| `enriched_text` | Contexto + texto + perguntas | **Embedding** (dense_vector) |
| `context_header` | Frase contextualizando o artigo | enriched_text |
| `thesis_text` | Resumo do que o artigo determina | **Embedding** (thesis_vector) |
| `thesis_type` | Tipo: definicao, procedimento, etc | Filtro |
| `synthetic_questions` | Perguntas que o artigo responde | enriched_text |

O `enriched_text` combina todos os campos para melhor recuperacao semantica:
```
[CONTEXTO: Este artigo da IN 58/2022 define os conceitos basicos...]

Art. 3 Para fins do disposto nesta Instrucao Normativa, considera-se:
I - Estudo Tecnico Preliminar - ETP: documento constitutivo...

[PERGUNTAS RELACIONADAS:
- Qual e a funcao do Sistema ETP Digital?
- Quem assume a funcao de requisitante?]
```

### Armazenamento

| Componente   | Tecnologia            | Versão | Licença    | Justificativa             |
| ------------ | --------------------- | ------ | ---------- | ------------------------- |
| Vector Store | Milvus Standalone     | 2.6    | Apache 2.0 | Já em produção, escalável |
| Collection   | Especializada em leis | -      | -          | Otimização de schema      |

### Infraestrutura Docker (Produção)

```yaml
# docker-compose.prod.yml
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --max-model-len 32768
      --gpu-memory-utilization 0.9
      --dtype auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  milvus:
    image: milvusdb/milvus:v2.6-latest
    # ... configuração existente
```

### Frontend (Futuro)

| Componente | Tecnologia   | Versão | Licença | Justificativa    |
| ---------- | ------------ | ------ | ------- | ---------------- |
| Framework  | Next.js      | 14+    | MIT     | SSR, performance |
| UI Library | React        | 18+    | MIT     | Padrão mercado   |
| Styling    | Tailwind CSS | 3+     | MIT     | Utility-first    |
| Components | shadcn/ui    | latest | MIT     | Customizável     |
| Icons      | Lucide React | latest | ISC     | Leve, moderno    |

---

## 📊 Decisões de Arquitetura

### 1. Arquitetura de Extração (3 Abordagens Testadas)

#### 🏆 Resultado do Benchmark (21/12/2024)

| Métrica                 | Extractor Simples | LangGraph Pipeline |   Híbrido   |
| ----------------------- | :---------------: | :----------------: | :---------: |
| **Capítulos Corretos**  |       4 ✅        |        2 ❌        |    4 ✅     |
| **Total Artigos**       |       19 ✅       |       19 ✅        |    19 ✅    |
| **Artigos Soltos**      |       0 ✅        |        3 ❌        |    0 ✅     |
| **Schema Correto**      |       OK ✅       |      ERRO ❌       |    OK ✅    |
| **Sub-items (alíneas)** |      SIM ✅       |       NÃO ❌       |   SIM ✅    |
| **Metadados Completos** |      SIM ✅       |       NÃO ❌       |   SIM ✅    |
| **SCORE TOTAL**         |    **100%** 🏆    |      **30%**       | **100%** 🏆 |

#### Insight Importante

O **Extractor Simples** e o **Híbrido** tiveram resultados **IDÊNTICOS** (100%). Isso prova que:

1. **O Extractor Simples é o motor principal** - ele faz o trabalho pesado
2. **O Pydantic Schema é a chave** - guia o LLM perfeitamente (similar ao LlamaExtract)
3. **LangGraph é orquestrador, não extrator** - não melhora qualidade, apenas gerencia fluxo

#### Quando Usar Cada Abordagem

| Cenário                     | Recomendação      |
| --------------------------- | ----------------- |
| **Scripts rápidos**         | Extractor Simples |
| **APIs/Microservices**      | Extractor Simples |
| **Prototipagem**            | Extractor Simples |
| **Produção robusta**        | Pipeline Híbrido  |
| **Multi-documento**         | Pipeline Híbrido  |
| **Com retry/checkpointing** | Pipeline Híbrido  |

### 2. API de Extração (Estilo LlamaExtract)

Criamos uma API elegante inspirada no [LlamaExtract](https://developers.llamaindex.ai/python/cloud/llamaextract/), mas 100% open-source.

```python
from extract import Extractor, ExtractConfig
from models.legal_document import LegalDocument

# Extração simples
extractor = Extractor()
result = extractor.extract("documento.pdf", schema=LegalDocument)
print(result.data)

# Com configuração customizada
config = ExtractConfig.for_legal_documents()
result = extractor.extract("lei.pdf", schema=LegalDocument, config=config)
```

#### Módulos Criados

```
extracao/src/
├── extract/                    # API de extração (estilo LlamaExtract)
│   ├── __init__.py
│   ├── config.py               # ExtractConfig, ExtractMode, ChunkMode
│   └── extractor.py            # Extractor, ExtractionAgent, ExtractionResult
├── models/                     # Schemas Pydantic
│   ├── __init__.py
│   └── legal_document.py       # LegalDocument, Chapter, Article, etc.
├── pipeline/                   # Pipeline LangGraph
│   ├── __init__.py
│   └── hybrid_pipeline.py      # Pipeline híbrido (LangGraph + Extractor)
└── agents/                     # Agentes LangGraph (legado)
    ├── __init__.py
    └── pipeline_agent.py       # Pipeline LangGraph original
```

### 3. Estratégia de Extração Final

```
PDF → Docling (Markdown) → Extractor (Pydantic + Qwen 3 8B) → BGE-M3 (Embedding) → Milvus
```

**Decisão**: Pipeline com **Extractor Simples** como motor principal.

**Fluxo detalhado**:

1. **Docling** extrai PDF → Markdown estruturado
2. **Extractor** (Pydantic + Qwen 3 8B) → JSON estruturado validado
3. **BGE-M3** → Gera embeddings dos chunks
4. **Milvus** → Armazena vetores e metadados

**Por que o Extractor Simples venceu**:

- Schema Pydantic no prompt = LLM sabe exatamente o que gerar
- Uma chamada focada > Múltiplas chamadas genéricas
- Menos complexidade = Menos erros
- Validação Pydantic integrada

### 4. Schema Pydantic para Documentos Legais

```python
class LegalDocument(BaseModel):
    """Modelo principal para documentos legais brasileiros."""

    document_type: str = Field(..., description="LEI, DECRETO, etc")
    issuing_body: str = Field(..., description="Nome do órgão emissor")
    issuing_body_acronym: Optional[str] = Field(None, description="Sigla")
    number: str = Field(..., description="Número do documento")
    date: str = Field(..., description="Data YYYY-MM-DD")
    ementa: str = Field(..., description="Resumo oficial")
    publication_details: Optional[PublicationDetails] = None
    chapters: list[Chapter] = Field(..., min_length=1)
    signatory: Optional[str] = None

class Chapter(BaseModel):
    chapter_number: Optional[str] = Field(None, examples=["I", "II"])
    title: str
    articles: list[Article] = Field(..., min_length=1)

class Article(BaseModel):
    article_number: str = Field(..., examples=["1", "2", "10"])
    title: Optional[str] = None
    content: str
    items: list[Item] = Field(default_factory=list)
    paragraphs: list[Paragraph] = Field(default_factory=list)

class Item(BaseModel):
    item_identifier: str = Field(..., examples=["I", "II", "III"])
    description: str
    sub_items: list[SubItem] = Field(default_factory=list)

class SubItem(BaseModel):
    item_identifier: str = Field(..., examples=["a", "b", "c"])
    description: str

class Paragraph(BaseModel):
    paragraph_identifier: str = Field(..., examples=["1", "2", "unico"])
    content: str
```

### 5. Configuração de Extração

```python
class ExtractConfig(BaseModel):
    """Configuração de extração (similar ao LlamaExtract)."""

    extraction_mode: ExtractMode = ExtractMode.BALANCED
    extraction_target: ExtractTarget = ExtractTarget.PER_DOC
    chunk_mode: ChunkMode = ChunkMode.SECTION
    system_prompt: Optional[str] = None
    llm: LLMConfig = Field(default_factory=LLMConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @classmethod
    def for_legal_documents(cls) -> "ExtractConfig":
        """Preset otimizado para documentos legais brasileiros."""
        return cls(
            extraction_mode=ExtractMode.BALANCED,
            chunk_mode=ChunkMode.ARTICLE,
            system_prompt="Especialista em documentos legais brasileiros...",
            llm=LLMConfig(model="qwen3:8b", temperature=0.0),
            validation=ValidationConfig(min_quality_score=0.98),
        )
```

### 6. Estratégia de Chunking

**Abordagem**: Chunking Agêntico Hierárquico

**Regras**:

1. Nunca separar inciso do seu artigo pai
2. Manter contexto hierárquico em cada chunk
3. Tamanho alvo: 500-1000 tokens
4. Parent-child linking para expansão de contexto

**Estrutura de Chunk**:

```python
{
    "id": "uuid",
    "law_id": "lei-12345-2024",
    "content": "Texto do chunk",
    "content_type": "artigo | paragrafo | inciso",
    "hierarchy": {
        "lei": "Lei nº 12.345/2024",
        "capitulo": "Capítulo I",
        "secao": "Seção II",
        "artigo": "Art. 5º"
    },
    "parent_context": "Contexto do pai",
    "metadata": {
        "position": 45,
        "tokens": 234,
        "keywords": ["direito", "cidadão"]
    }
}
```

### 7. Collection Milvus para Leis

**Nome**: `leis_v2`

**Schema Principal** (30 campos):

| Campo | Tipo | Indice | Descricao |
|-------|------|--------|-----------|
| id | INT64 | Primary | Auto-gerado |
| chunk_id | VARCHAR(200) | - | ID hierarquico |
| text | VARCHAR(65535) | - | Texto original |
| enriched_text | VARCHAR(65535) | - | Contexto + texto + perguntas |
| dense_vector | FLOAT_VECTOR(1024) | HNSW | Embedding do enriched_text |
| thesis_vector | FLOAT_VECTOR(1024) | HNSW | Embedding do thesis_text |
| sparse_vector | SPARSE_FLOAT_VECTOR | SPARSE_INVERTED | Learned sparse BGE-M3 |
| context_header | VARCHAR(2000) | - | Frase de contexto |
| thesis_text | VARCHAR(5000) | - | Resumo do artigo |
| thesis_type | VARCHAR(100) | - | definicao, procedimento, etc |
| synthetic_questions | VARCHAR(10000) | - | Perguntas relacionadas |
| article_number | VARCHAR(32) | INVERTED | Numero do artigo |
| tipo_documento | VARCHAR(64) | INVERTED | LEI, DECRETO, IN |
| ano | INT64 | INVERTED | Ano do documento |

**Indices para Busca Hibrida**:
- `dense_vector`: HNSW (COSINE, M=16, efConstruction=256)
- `thesis_vector`: HNSW (COSINE, M=16, efConstruction=256)
- `sparse_vector`: SPARSE_INVERTED_INDEX (IP, drop_ratio=0.2)

### 8. Modulo de Busca Hibrida (22/12/2024)

Modulo reutilizavel para busca 2-stage com BGE-M3 + BGE-Reranker.

**Estrutura**:
```
src/search/
├── __init__.py          # Exports publicos
├── config.py            # SearchConfig, SearchMode, RerankMode
├── models.py            # SearchHit, SearchResult, SearchFilter
└── hybrid_searcher.py   # HybridSearcher (classe principal)
```

**Uso Basico**:
```python
from search import HybridSearcher, SearchConfig

# Busca com configuracao padrao (3-way hybrid + reranker)
with HybridSearcher() as searcher:
    result = searcher.search("O que e pesquisa de precos?", top_k=5)

    for hit in result.hits:
        print(f"Art. {hit.article_number}: {hit.final_score:.2f}")
        print(f"  {hit.context_header}")
```

**Configuracoes Pre-definidas**:

| Config | Modo | Reranker | Uso |
|--------|------|----------|-----|
| `SearchConfig.default()` | 3-way hybrid | Cross-encoder | Producao |
| `SearchConfig.fast()` | 2-way hybrid | Nenhum | Baixa latencia |
| `SearchConfig.precise()` | 3-way hybrid | Cross-encoder | Maxima precisao |
| `SearchConfig.dense_only()` | Dense | Nenhum | Debug |

**Busca com Filtros**:
```python
from search import HybridSearcher, SearchFilter

filters = SearchFilter(
    document_type="IN",
    year=2021,
    thesis_types=["definicao", "procedimento"],
)
result = searcher.search("definicoes basicas", filters=filters)
```

**Campos Usados na Busca Hibrida**:

| Vetor | Campo Fonte | Peso | Descricao |
|-------|-------------|------|-----------|
| `dense_vector` | `enriched_text` | 50% | Semantica geral (contexto + texto + perguntas) |
| `sparse_vector` | `enriched_text` | 30% | Termos especificos (learned sparse) |
| `thesis_vector` | `thesis_text` | 20% | Essencia/resumo do artigo |

**Pipeline de Busca 2-Stage**:

```
Query
  │
  ▼
[BGE-M3] → Embedding hibrido (dense + sparse)
  │
  ▼
[Stage 1: Milvus Hybrid Search]
  ├─ Dense (50%) → ANN no dense_vector
  ├─ Sparse (30%) → Inverted Index no sparse_vector
  └─ Thesis (20%) → ANN no thesis_vector
  │
  ▼
[WeightedRanker] → Top 20 candidatos
  │
  ▼
[Stage 2: BGE-Reranker Cross-Encoder]
  └─ Rerank usando enriched_text
  │
  ▼
Top 5 final (ordenado por relevancia)
```

**Benchmark do Modulo** (22/12/2024 - IN 65/2021 Pesquisa de Precos):

| Query | Top 1 | Rerank Score | Tempo Total |
|-------|-------|--------------|-------------|
| Como fazer pesquisa de precos? | Art. 3 (definicoes) | **0.91** | 43s |
| Prazo de validade dos precos? | Art. 6 (120 dias) | **0.94** | 25s |
| O que e preco estimado? | Art. 2 (definicoes) | **0.78** | 25s |

**Performance**:
- Stage 1 (Milvus): ~70-100ms (apos warmup)
- Stage 2 (Reranker): ~25s (modelo cross-encoder)
- Primeira execucao: ~40s (carrega modelos)

---

## 🚀 Roadmap

### Fase 1 - MVP Extração ✅ (Completo)

- [x] Definição de stack
- [x] Documentação de decisões
- [x] Setup projeto Python
- [x] Teste Docling com PDF de lei
- [x] Análise da estrutura extraída
- [x] Benchmark de modelos LLM (8 modelos testados)
- [x] Seleção do modelo: **qwen3:8b**
- [x] Implementação do Extractor com Pydantic
- [x] Comparação: Extractor vs LangGraph vs Híbrido
- [x] API estilo LlamaExtract (open-source)
- [x] Validação Pydantic integrada

### Fase 2 - Chunking Agentico ✅ (Completo)

- [x] Implementacao do LawChunker (chunk_models.py, law_chunker.py)
- [x] Prompts de enriquecimento (enrichment_prompts.py)
- [x] Cliente vLLM (vllm_client.py)
- [x] Benchmark 4B vs 8B para enriquecimento
- [x] Decisao: 4B para enriquecimento, 8B para extracao
- [x] Integracao completa LawChunker + vLLM + BGE-M3
- [x] Pipeline run_pipeline.py funcional

### Fase 3 - Embeddings + Storage ✅ (Completo)

- [x] Setup vLLM em Docker (producao)
- [x] Download modelos AWQ no volume Docker
- [x] Configurar BGE-M3 (embeddings dense + sparse)
- [x] Configurar bge-reranker-v2-m3 (reranking)
- [x] Schema Milvus (leis_v2 com 30 campos)
- [x] Pipeline de indexacao (run_pipeline.py)
- [x] Busca hibrida (dense + sparse + thesis = 3 vetores)
- [x] 2-Stage retrieval (BGE-M3 + Reranker)
- [x] ColBERT Reranker implementado (alternativa)
- [x] Benchmark ColBERT vs Cross-Encoder (80% concordancia)
- [x] Correcao: usar enriched_text nos embeddings
- [x] Correcao: usar 3 vetores na busca hibrida
- [x] Modulo de busca reutilizavel (`src/search/`)

### Fase 4 - RAG Completo (Em Progresso)

- [x] Modulo de busca hibrida (HybridSearcher)
- [ ] API de busca com FastAPI
- [ ] Integração retrieval + generation (vLLM)
- [ ] Prompts especializados para resposta juridica
- [ ] Avaliação de qualidade (RAGAS ou similar)

### Fase 5 - Interface

- [ ] Setup Next.js
- [ ] Interface de upload
- [ ] Interface de busca
- [ ] Dashboard de monitoramento

---

## 📝 Notas de Desenvolvimento

### 21/12/2024

**Manha - Extracao**:
- Projeto iniciado
- Stack definida com foco em licencas Apache 2.0/MIT
- Decisao de usar JSON estruturado ao inves de Markdown puro
- Estrutura do projeto criada em `extracao/`
- Ambiente virtual configurado com Docling instalado
- **Teste Docling**: Extracao da IN SEGES 58/2022 bem-sucedida (3.5s)
- **Benchmark completo**: 8 modelos Qwen testados
- **Selecao**: qwen3:8b (94% qualidade, unico com alineas)
- **Comparacao de abordagens**: Extractor (100%) vs LangGraph (30%) vs Hibrido (100%)
- **Decisao**: Extractor Simples como motor principal
- **API criada**: Estilo LlamaExtract, 100% open-source

**Tarde - Chunking e Enriquecimento**:
- Implementado LawChunker (chunking hierarquico por artigo)
- Criados prompts de enriquecimento (Contextual Retrieval da Anthropic)
- Implementado cliente vLLM (API OpenAI-compatible)
- Setup vLLM em Docker com modelos AWQ
- Download Qwen 3 4B-AWQ e 8B-AWQ no volume Docker
- **Benchmark enriquecimento**: 4B vs 8B testados
- **Resultado**: 4B tem mesma qualidade, porem 2x mais rapido (7.4s vs 14.5s/chunk)
- **Decisao**: 4B para enriquecimento, 8B para extracao
- **Estrategia de roteamento**: Pipeline sequencial (trocar modelo entre fases)

**Arquivos criados**:
```
src/chunking/
  chunk_models.py      # LegalChunk dataclass
  law_chunker.py       # LawChunker pipeline
  enrichment_prompts.py # Prompts Contextual Retrieval
src/llm/
  vllm_client.py       # VLLMClient, LLMConfig
src/embeddings/
  bge_m3.py           # BGEM3Embedder (a integrar)
src/milvus/
  schema.py           # leis_v2 collection schema
tests/
  test_chunking.py    # Teste de chunking
  test_enrichment.py  # Benchmark de enriquecimento
```

### 22/12/2024

**Manha - Embeddings e Retrieval**:
- Integrado BGE-M3 com pipeline (dense 1024d + sparse)
- Criada collection `leis_v2` no Milvus (30 campos, 6 indices)
- Pipeline completo: JSON → Chunks → LLM → Embeddings → Milvus
- Testada busca hibrida (dense + sparse + RRF/Weighted)
- **2-Stage Retrieval**: BGE-M3 + BGE-Reranker funcionando

**Tarde - ColBERT e Correcoes**:
- Implementado ColBERT Reranker (`src/reranker/colbert_reranker.py`)
- **Benchmark ColBERT vs Cross-Encoder**: 80% concordancia, Cross-Encoder tem scores maiores
- **Decisao**: Cross-Encoder como principal, ColBERT como alternativa
- **Bug encontrado**: `enriched_text` nao estava sendo usado nos embeddings!
- **Correcao**: `law_chunker.py` agora usa `enriched_text` para `dense_vector`
- **Correcao**: Busca hibrida agora usa 3 vetores (dense + sparse + thesis)
- **Correcao**: Reranking agora usa `enriched_text` em vez de `text`

**Arquivos criados/modificados**:
```
src/reranker/
  __init__.py          # Modulo de reranking
  colbert_reranker.py  # ColBERT MaxSim reranker
src/embeddings/
  bge_reranker.py      # BGE-Reranker Cross-Encoder
src/chunking/
  law_chunker.py       # CORRIGIDO: usar enriched_text
tests/
  test_2stage_retrieval.py      # ATUALIZADO: 3 vetores + enriched_text
  test_colbert_vs_crossencoder.py # Benchmark comparativo
scripts/
  init_milvus.py       # Inicializa collection leis_v2
  run_pipeline.py      # Pipeline completo
```

**Tarde - Modulo de Busca Reutilizavel**:
- Criado modulo `src/search/` com API limpa para busca
- `HybridSearcher`: classe principal com busca 2-stage
- `SearchConfig`: configuracoes pre-definidas (default, fast, precise)
- `SearchHit`, `SearchResult`: dataclasses para resultados
- `SearchFilter`: filtros por tipo_documento, ano, thesis_type
- Testado com IN 65/2021 (pesquisa de precos)
- **Resultado**: Reranker scores 0.78-0.94 para queries relevantes
- **Descoberta**: Collection `leis_v2` contem IN 65, nao IN 58

**Arquivos criados**:
```
src/search/
  __init__.py          # Exports publicos
  config.py            # SearchConfig, SearchMode, RerankMode
  models.py            # SearchHit, SearchResult, SearchFilter
  hybrid_searcher.py   # HybridSearcher (classe principal)
tests/
  test_search_module.py # Teste completo do modulo
```

---

## 🧪 Benchmark de Modelos LLM (21/12/2024)

### Metodologia

- **Documento teste**: Instrução Normativa SEGES Nº 58/2022 (19 artigos, 4 capítulos)
- **Tarefa**: Converter Markdown (extraído pelo Docling) para JSON estruturado
- **Critérios**: Extração de items (incisos), paragraphs, sub_items (alíneas), títulos
- **Hardware**: NVIDIA RTX 4070 12GB VRAM

### Resultados Completos

| Modelo            | Disco     | GPU Load     | items        | paragraphs | sub_items       | Títulos | Qualidade          |
| ----------------- | --------- | ------------ | ------------ | ---------- | --------------- | ------- | ------------------ |
| qwen2.5:7b        | 4.7GB     | 100% GPU     | ❌ Mistura § | ❌         | ❌              | ❌      | ⭐⭐⭐ 75%         |
| qwen2.5-coder:7b  | 4.7GB     | 100% GPU     | ❌           | ❌         | ❌              | ❌      | ⭐⭐ 60%           |
| qwen2.5-coder:14b | 9GB       | 94% GPU      | ✅           | ✅         | ❌              | ❌      | ⭐⭐⭐⭐ 93%       |
| qwen3:4b          | 2.5GB     | 100% GPU     | ✅           | ✅         | ❌              | ❌      | ⭐⭐⭐⭐ 90%       |
| **qwen3:8b** ⭐   | **4.9GB** | **100% GPU** | ✅           | ✅         | **✅ Alíneas!** | ✅      | **⭐⭐⭐⭐⭐ 94%** |
| qwen3:14b         | 9GB       | 74%/26% CPU  | ✅           | ✅         | ⚠️ Parcial      | ❌      | ⭐⭐⭐⭐ 92%       |
| qwen3-coder:30b   | 18GB      | Local MoE    | ✅           | ✅         | ✅ (vazio)      | ✅      | ⭐⭐⭐⭐⭐ 97%     |
| qwen3-coder:480b  | Cloud     | Cloud        | ✅           | ✅         | ✅ (vazio)      | ✅      | ⭐⭐⭐⭐⭐ 98%     |

### Análise

**Observações Importantes**:

1. **qwen3:8b** foi o único modelo local que extraiu corretamente **alíneas (sub_items: a, b, c, d)**
2. Modelos com **offloading CPU** (qwen3:14b) tiveram qualidade inferior aos que cabem 100% na GPU
3. Família **Qwen 3** é significativamente melhor que **Qwen 2.5** para extração estruturada
4. Modelos **coder** não mostraram vantagem significativa para esta tarefa específica
5. O modelo cloud (480b) é apenas 4% melhor que o qwen3:8b local

**Decisão Final**: **qwen3:8b**

| Critério         | qwen3:8b         |
| ---------------- | ---------------- |
| Tamanho em disco | 4.9GB            |
| Uso de VRAM      | ~10GB (100% GPU) |
| Contexto máximo  | 256K tokens      |
| Qualidade JSON   | 94%              |
| Extração alíneas | ✅ Único local   |
| Custo            | Gratuito (local) |
| Licença          | Apache 2.0       |

### Modelos Mantidos no Sistema

```
qwen3:8b                 - 5.2GB  - Produção
qwen3-coder:480b-cloud   - Cloud  - Backup/Comparação
```

### Modelos Removidos

- qwen2.5:7b (75% qualidade)
- qwen2.5-coder:7b (60% qualidade - muito incompleto)
- qwen2.5-coder:14b (93% mas precisa de contexto reduzido)
- qwen3:4b (90% - substituído pelo 8b)
- qwen3:14b (92% - offloading prejudica qualidade)
- qwen3-coder:30b (97% - não cabe na GPU 12GB)

---

## 🧪 Benchmark de Abordagens de Extração (21/12/2024)

### Metodologia

- **Documento teste**: Instrução Normativa SEGES Nº 58/2022 (19 artigos, 4 capítulos)
- **Tarefa**: Extrair JSON estruturado completo com validação
- **Modelo**: qwen3:8b

### Resultados

| Abordagem             |  Score   | Capítulos | Schema  | Alíneas | Metadados |
| --------------------- | :------: | :-------: | :-----: | :-----: | :-------: |
| **Extractor Simples** | **100%** |   4 ✅    |  OK ✅  | SIM ✅  |  SIM ✅   |
| LangGraph Pipeline    |   30%    |   2 ❌    | ERRO ❌ | NÃO ❌  |  NÃO ❌   |
| **Híbrido**           | **100%** |   4 ✅    |  OK ✅  | SIM ✅  |  SIM ✅   |

### Conclusão

O **Pydantic Schema** é o diferencial. Quando o LLM recebe o JSON Schema completo do Pydantic, ele sabe exatamente o que gerar.

---

## 🧪 Benchmark de Enriquecimento: 4B vs 8B (21/12/2024)

### Metodologia

- **Documento teste**: IN SEGES 58/2022 (3 artigos selecionados: Art. 3, 6, 14)
- **Tarefa**: Gerar context_header, thesis_text, thesis_type, synthetic_questions
- **Runtime**: vLLM 0.13 com quantizacao AWQ
- **Hardware**: NVIDIA RTX 4070 12GB VRAM

### Resultados Comparativos

| Metrica | **4B-AWQ** | **8B-AWQ** | Diferenca |
|---------|-----------|-----------|-----------|
| Taxa de sucesso | 100% (3/3) | 100% (3/3) | = |
| Acuracia thesis_type | 66.7% (2/3) | 66.7% (2/3) | = |
| Tempo total | **22.20s** | 43.46s | **-49%** |
| Tempo medio/chunk | **7.40s** | 14.49s | **-49%** |

### Tempo por Artigo

| Artigo | Tipo Esperado | 4B (tempo) | 8B (tempo) | Speedup | Acerto |
|--------|---------------|-----------|-----------|---------|--------|
| Art. 3 | definicao | 9.25s | 16.81s | **1.8x** | OK |
| Art. 6 | procedimento | 6.43s | 12.76s | **2.0x** | ERRO* |
| Art. 14 | excecao | 6.52s | 13.89s | **2.1x** | OK |

*O erro no Art. 6 ocorreu em ambos os modelos - e problema do prompt, nao do modelo.

### Qualidade das Saidas (4B)

**Art. 3 (Definicoes)**:
```
context_header: Este artigo da IN 58/2022 define os conceitos basicos para
                elaboracao de ETP no ambito federal

thesis_text: Estabelece definicoes de termos tecnicos relacionados ao
             planejamento de contratacoes publicas, incluindo ETP, sistema
             digital de gestao e tipos de contratacoes correlatas ou interdependentes

synthetic_questions:
- Qual e a definicao de ETP segundo a IN 58/2022?
- Quais sao as caracteristicas do Sistema ETP Digital?
- O que caracteriza contratacoes correlatas?
```

### Decisao Final

| Criterio | 4B-AWQ | 8B-AWQ | Vencedor |
|----------|--------|--------|----------|
| Qualidade | 100% | 100% | Empate |
| Velocidade | 7.4s/chunk | 14.5s/chunk | **4B** |
| VRAM | 2.5GB | 5.7GB | **4B** |
| Batch potencial | Maior | Menor | **4B** |

**Conclusao**: O **Qwen 3 4B-AWQ** e a escolha certa para enriquecimento:
- Mesma qualidade que o 8B
- 2x mais rapido
- Metade da VRAM (permite batch maior no futuro)

---

## 🔍 Arquitetura de Validacao (Pos-MVP)

> **Status**: Planejado - Implementar apos MVP funcional

### Problema

A extracao tem duas etapas que precisam de validacao:

1. **PDF → Markdown (Docling)**: Como saber se o Docling extraiu corretamente?
2. **Markdown → JSON (Qwen3)**: Como saber se o LLM estruturou corretamente?

### Decisao: Validacao Assincrona (Nao-Bloqueante)

Em vez de validar sincronamente (bloqueando o pipeline), a validacao roda em **paralelo**:

```
                    ┌─────────────────────────────┐
                    │      Processo Principal      │
                    │    (nao bloqueia, rapido)    │
                    └─────────────────────────────┘
                                  │
PDF → Docling → Markdown ─────────┼─────────→ Qwen3 → JSON → Milvus
                                  │
                                  │ (fork assincrono)
                                  ▼
                    ┌─────────────────────────────┐
                    │    Processo de Validacao     │
                    │   (paralelo, pode demorar)   │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Alerta para Humano         │
                    │   (so se score < threshold)  │
                    └─────────────────────────────┘
```

### Beneficios

| Beneficio | Descricao |
|-----------|-----------|
| **Desacoplamento** | Processo principal nao espera validacao |
| **Escala independente** | Workers de extracao e validacao escalam separadamente |
| **Human-in-the-loop inteligente** | Humano so e acionado quando necessario |
| **Retry sem reprocessar** | Se validacao falhar, pode re-validar sem re-extrair |

### Estados do Documento

```
PROCESSANDO → EXTRAIDO → INDEXADO
                 │
                 └──→ VALIDANDO → VALIDADO ✓
                           │
                           └──→ SUSPEITO → REVISAO_HUMANA → CORRIGIDO
```

O documento pode estar **indexado e buscavel** mesmo enquanto validacao roda.

### Validacao 1: PDF → Markdown (Heuristicas)

Usar **PyMuPDF** para extrair texto bruto do PDF e comparar contagens com Markdown do Docling:

| Elemento | Contagem PDF | Contagem Markdown | Status |
|----------|--------------|-------------------|--------|
| Artigos (Art.) | 19 | 19 | ✅ |
| Paragrafos (§) | 10 | 10 | ✅ |
| Incisos (I, II) | 25 | 25 | ✅ |
| Capitulos | 4 | 4 | ✅ |

**Alertar humano se**: Discrepancia > 5% ou elementos estruturais faltando.

### Validacao 2: Markdown → JSON (Ja Implementado)

Modulo `extraction_utils.py` com:

- **DoclingValidator**: Conta elementos no Markdown
- **ExtractionValidator**: Compara JSON vs Markdown
- **AutoFixer**: Corrige erros conhecidos automaticamente

### Threshold de Alerta

| Score | Acao |
|-------|------|
| >= 98% | Log apenas, sem alerta |
| 95-98% | Warning, revisao opcional |
| < 95% | Alerta para humano |
| Erros estruturais | Alerta imediato |

### Implementacao Futura

**Fase 1 (Simples)**:
- Fila em memoria ou SQLite
- Worker separado (thread/processo)
- Alerta por log/arquivo

**Fase 2 (Producao)**:
- Redis/RabbitMQ como fila
- Celery para workers
- Dashboard para humano revisar

**Fase 3 (Escala)**:
- Evento no MinIO quando PDF e processado
- Workers distribuidos
- Metricas e observabilidade

### Amostragem para Calibracao

Mesmo com score 100%, **10% dos documentos** vao para revisao aleatoria para:
- Calibrar confianca no sistema
- Detectar erros sistematicos
- Melhorar heuristicas ao longo do tempo

---

## 🔗 Referências

- [Docling Documentation](https://ds4sd.github.io/docling/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
- [Milvus 2.6 Documentation](https://milvus.io/docs)
- [Qwen 3 Models](https://huggingface.co/Qwen)
- [Ollama](https://ollama.com/) - Runtime local para LLMs
- [LlamaExtract](https://developers.llamaindex.ai/python/cloud/llamaextract/) - Inspiração para API
- [vLLM](https://docs.vllm.ai/) - Runtime de produção
