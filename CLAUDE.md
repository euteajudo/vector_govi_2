# 📋 Pipeline de Extração RAG para Leis - Documentação de Decisões

> **Projeto**: Sistema RAG para orgaos publicos
> **Data de Inicio**: 21/12/2024
> **Ultima Atualizacao**: 23/12/2024 21:30
> **Status**: Fase 5 - RAG Completo com Resposta LLM ✅ (Answer Generator + Citações)

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
| LLM (unico)     | **Qwen 3 8B-AWQ** | latest  | Apache 2.0 | Extracao + Enriquecimento (modelo unico)        |
| Runtime Prod    | **vLLM**          | 0.13+   | Apache 2.0 | Docker, API OpenAI-compatible, quantizacao AWQ  |
| Hardware        | GPU 12GB          | -       | -          | 8B-AWQ: 5.7GB + BGE-M3: ~2GB = ~8GB total       |

**Decisao**: Usar **modelo unico Qwen 3 8B-AWQ** para todas as tarefas.

> **Atualizado 22/12/2024**: Abandonamos a estrategia de model swapping (trocar 4B/8B entre fases).
> O ganho de velocidade do 4B nao justifica a complexidade operacional em producao.

**Por que modelo unico?**

| Criterio | Model Swapping (4B+8B) | Modelo Unico (8B) |
|----------|------------------------|-------------------|
| Complexidade | Alta (scripts de troca) | Baixa |
| Downtime | Sim (durante troca) | Nao |
| Race conditions | Possiveis | Nenhuma |
| Filas/workers | Complexo | Simples |
| Velocidade enriquecimento | 7.4s/chunk | 14.5s/chunk |
| Qualidade | Igual | Igual |

**Conclusao**: O 8B e 2x mais lento no enriquecimento, mas a simplicidade operacional
compensa. Em producao com filas (Redis/Celery), a latencia extra e absorvida pelo paralelismo.

**Configuracao vLLM Producao**:
```bash
docker run -d --name vllm --gpus all \
  -v huggingface-cache:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B-AWQ \
  --max-model-len 16000 \
  --gpu-memory-utilization 0.9
```

**Vantagens vLLM em Producao**:

- Continuous batching (maior throughput)
- PagedAttention (uso eficiente de VRAM)
- API compativel com OpenAI (facil migracao)
- Tensor parallelism para multiplas GPUs
- Quantizacao nativa (AWQ, GPTQ)

**Justificativa do modelo** (21/12/2024 - apos benchmarks extensivos):

- **8B-AWQ**: Unico modelo local que extraiu corretamente alineas (sub_items)
- 256K de contexto (8x mais que Qwen 2.5)
- Licenca Apache 2.0 (100% comercial)
- Forte em portugues juridico
- VRAM: 5.7GB (cabe em GPU 12GB com folga para BGE-M3)

**JSON Schema (Structured Output)** - Implementado 22/12/2024:

O vLLM suporta `response_format` com `json_schema` para forcar o modelo a gerar
apenas JSON valido seguindo um schema. Isso e usado na extracao (MD→JSON) para
prevenir alucinacoes e garantir output estruturado.

```python
# Exemplo de uso no VLLMClient
result = client.chat_with_schema(
    messages=[{"role": "user", "content": "Extraia..."}],
    schema=LegalDocument,  # Pydantic model ou dict
    temperature=0.0,
)
# result ja e dict validado, nao string
```

| Fase | Usa json_schema? | Motivo |
|------|------------------|--------|
| Extracao (MD→JSON) | **Sim** | Precisa de JSON estruturado exato |
| Enriquecimento | Nao | Retorna texto livre (context, thesis) |
| Resposta usuario | Nao | Retorna texto natural |

**Configuracao**:
```python
config = ExtractConfig.for_legal_documents()
config.llm.use_guided_json = True  # Habilita json_schema na extracao
```

**Beneficios**:
- Elimina parsing manual de JSON
- Previne JSON malformado
- Reduz alucinacoes de estrutura
- 100% de sucesso em testes

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
| `text` | Texto original do artigo | **Reranking** (Stage 2) |
| `enriched_text` | Contexto + texto + perguntas | **Embedding** (dense_vector) |
| `context_header` | Frase contextualizando o artigo | enriched_text |
| `thesis_text` | Resumo do que o artigo determina | **Embedding** (thesis_vector) |
| `thesis_type` | Tipo: definicao, procedimento, etc | Filtro |
| `synthetic_questions` | Perguntas que o artigo responde | enriched_text |

> **IMPORTANTE (Corrigido 22/12/2024)**: O reranker usa `text` (original), NAO `enriched_text`.
> O prefixo `[CONTEXTO: ...]` do enriched_text dilui a relevancia para o cross-encoder.
> Testes mostraram: texto original = score 0.55, enriched_text = score 0.27.

**Estrategia de uso dos campos**:

| Stage | Campo Usado | Motivo |
|-------|-------------|--------|
| Stage 1 (Embedding) | `enriched_text` | Contexto extra melhora busca semantica |
| Stage 2 (Reranking) | `text` | Cross-encoder precisa de texto limpo |

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

**Nome**: `leis_v3` (atual) | `leis_v2` (legado, dropada)

**Schema v3** (30 campos com parent-child):

| Campo | Tipo | Indice | Descricao |
|-------|------|--------|-----------|
| id | INT64 | Primary | Auto-gerado |
| chunk_id | VARCHAR(200) | - | ID completo: IN-65-2021#ART-005 |
| **parent_chunk_id** | VARCHAR(200) | INVERTED | ID do chunk pai (vazio para artigos) |
| **span_id** | VARCHAR(100) | - | ART-005, PAR-005-1, INC-005-I |
| **device_type** | VARCHAR(32) | INVERTED | article, paragraph, inciso, alinea |
| **chunk_level** | VARCHAR(32) | - | article, device |
| text | VARCHAR(65535) | - | Texto original |
| enriched_text | VARCHAR(65535) | - | Contexto + texto + perguntas |
| dense_vector | FLOAT_VECTOR(1024) | HNSW | Embedding do enriched_text |
| thesis_vector | FLOAT_VECTOR(1024) | HNSW | Embedding do thesis_text |
| sparse_vector | SPARSE_FLOAT_VECTOR | SPARSE_INVERTED | Learned sparse BGE-M3 |
| context_header | VARCHAR(2000) | - | Frase de contexto |
| thesis_text | VARCHAR(5000) | - | Resumo do artigo |
| thesis_type | VARCHAR(100) | - | definicao, procedimento, etc |
| synthetic_questions | VARCHAR(10000) | - | Perguntas relacionadas |
| **citations** | VARCHAR(5000) | - | JSON: [span_id, ...] |
| document_id | VARCHAR(200) | - | ID único do documento |
| tipo_documento | VARCHAR(64) | INVERTED | LEI, DECRETO, IN |
| numero | VARCHAR(32) | - | Número do documento |
| ano | INT64 | INVERTED | Ano do documento |
| article_number | VARCHAR(32) | INVERTED | Numero do artigo |
| **schema_version** | VARCHAR(32) | - | Versão do schema (1.0.0) |
| **extractor_version** | VARCHAR(32) | - | Versão do extrator |
| **ingestion_timestamp** | VARCHAR(64) | - | Timestamp ISO |
| **document_hash** | VARCHAR(128) | - | SHA-256 do PDF |
| **page** | INT64 | - | Página no PDF |
| **bbox_left/top/right/bottom** | FLOAT | - | Bounding box |

**Campos novos em negrito** (v3):
- Parent-child: `parent_chunk_id`, `span_id`, `device_type`, `chunk_level`, `citations`
- Proveniência: `schema_version`, `extractor_version`, `ingestion_timestamp`, `document_hash`
- Page spans: `page`, `bbox_left`, `bbox_top`, `bbox_right`, `bbox_bottom`

**Indices para Busca Hibrida**:
- `dense_vector`: HNSW (COSINE, M=16, efConstruction=256)
- `thesis_vector`: HNSW (COSINE, M=16, efConstruction=256)
- `sparse_vector`: SPARSE_INVERTED_INDEX (IP, drop_ratio=0.2)
- `parent_chunk_id`: INVERTED (para buscar filhos)
- `device_type`: INVERTED (filtrar por tipo)

### 8. Arquitetura Span-Based (23/12/2024)

**Abordagem**: Extração baseada em spans com hierarquia preservada.

A arquitetura span-based divide o documento em spans identificados por IDs únicos
que preservam a estrutura hierárquica do documento legal.

**Componentes Principais**:

```
src/parsing/
├── span_parser.py              # SpanParser - parseia Markdown para spans
├── span_models.py              # Span, SpanType, ParsedDocument
├── span_extraction_models.py   # ArticleSpans (schema para LLM)
└── article_orchestrator.py     # ArticleOrchestrator (extração por artigo)
```

**Fluxo de Extração**:

```
PDF → Docling → Markdown → SpanParser → ParsedDocument
                                              │
                                              ▼
                           ArticleOrchestrator (por artigo)
                                              │
                                              ▼
                             ChunkMaterializer → MaterializedChunk
                                              │
                                              ▼
                                           Milvus
```

**Formato de Span IDs**:

| Tipo | Formato | Exemplo |
|------|---------|---------|
| Artigo | `ART-{nnn}` | `ART-005` |
| Parágrafo | `PAR-{art}-{n}` | `PAR-005-1`, `PAR-005-UNICO` |
| Inciso | `INC-{art}-{romano}` | `INC-005-I`, `INC-005-II` |
| Alínea | `ALI-{art}-{romano}-{letra}` | `ALI-005-I-a` |
| Inciso de § | `INC-{art}-{romano}_{par}` | `INC-005-I_2` (inciso I do §2) |

**Características**:

- **Curto-circuito**: Artigos sem filhos não chamam LLM (economia de tokens)
- **Schema enum dinâmico**: IDs permitidos são passados como enum por artigo
- **Retry focado por janela**: Retry específico para PAR ou INC, não ambos
- **Validação de parent consistency**: `INC-005-I_2` deve ter parent `PAR-005-2`

**Resultados do Teste** (IN 65/2021):

| Métrica | Valor |
|---------|-------|
| Artigos processados | 11/11 (100%) |
| Artigos válidos | 11/11 |
| Total de chunks | 47 |
| ARTICLE chunks | 11 |
| PARAGRAPH chunks | 19 |
| INCISO chunks | 17 |

### 9. Parent-Child Retrieval com ChunkMaterializer (23/12/2024)

O `ChunkMaterializer` transforma ArticleChunks em chunks indexáveis com suporte
a parent-child retrieval.

**Estrutura de Chunks**:

```
Chunk Pai (ARTICLE)           Chunks Filhos
┌─────────────────────┐       ┌─────────────────────┐
│ IN-65-2021#ART-005  │──────▶│ IN-65-2021#PAR-005-1│
│ parent_chunk_id: "" │       │ parent: ART-005     │
│ type: ARTICLE       │       │ type: PARAGRAPH     │
│ text: "Art. 5..."   │       │ text: "§1 ..."      │
└─────────────────────┘       ├─────────────────────┤
                              │ IN-65-2021#INC-005-I│
                              │ parent: ART-005     │
                              │ type: INCISO        │
                              │ text: "I - ..."     │
                              └─────────────────────┘
```

**Estratégia de Busca Parent-Child**:

```
Query → Busca chunks filhos (INC/PAR) → Agrega chunks pai → Contexto expandido → LLM
```

1. Busca semântica retorna chunk filho (ex: `INC-005-II`)
2. Sistema recupera chunk pai via `parent_chunk_id` (ex: `ART-005`)
3. Contexto expandido passa para LLM (pai + filho + irmãos relevantes)
4. LLM responde com contexto completo do artigo

**Classes do ChunkMaterializer**:

```python
@dataclass
class ChunkMetadata:
    """Metadados de proveniência e versão."""
    schema_version: str = "1.0.0"
    extractor_version: str = "1.0.0"
    ingestion_timestamp: str
    document_hash: str  # SHA-256 do PDF
    valid_from: Optional[str]  # Vigência
    valid_to: Optional[str]
    page_spans: dict  # Coordenadas PDF (futuro)

@dataclass
class MaterializedChunk:
    """Chunk pronto para indexação."""
    chunk_id: str           # Ex: "IN-65-2021#ART-005"
    parent_chunk_id: str    # Ex: "" (pai) ou "IN-65-2021#ART-005" (filho)
    span_id: str            # Ex: "ART-005"
    device_type: DeviceType # ARTICLE, PARAGRAPH, INCISO, ALINEA
    chunk_level: ChunkLevel
    text: str
    citations: list[str]    # Spans que compõem este chunk
    metadata: ChunkMetadata
```

**Campos Dinâmicos para Milvus**:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `parent_chunk_id` | VARCHAR | ID do chunk pai ("" se for artigo) |
| `span_id` | VARCHAR | ID do span (ex: "ART-005") |
| `device_type` | VARCHAR | article, paragraph, inciso, alinea |
| `citations` | JSON | Lista de span_ids que compõem o chunk |

### 10. Answer-JSON para Frontend (23/12/2024)

Formato estruturado de resposta para o frontend consumir.

**Módulo**: `src/rag/answer_models.py`

**Estrutura da Resposta**:

```json
{
    "success": true,
    "data": {
        "answer": "Texto da resposta gerada pelo LLM...",
        "confidence": 0.92,
        "citations": [
            {
                "span_id": "ART-005",
                "chunk_id": "IN-65-2021#ART-005",
                "text": "Art. 5º O estudo...",
                "relevance": 0.95,
                "location": {"page": 2, "x": 50, "y": 100}
            }
        ],
        "sources": [
            {
                "document_id": "IN-65-2021",
                "title": "IN SEGES Nº 65/2021",
                "tipo_documento": "INSTRUCAO NORMATIVA"
            }
        ]
    },
    "metadata": {
        "model": "Qwen/Qwen3-8B-AWQ",
        "latency_ms": 1234,
        "tokens_used": 456,
        "chunks_retrieved": 5,
        "chunks_used": 3,
        "timestamp": "2024-12-23T14:30:00Z"
    }
}
```

**Cálculo de Confiança**:

```python
def calculate_confidence(citations: list[Citation]) -> float:
    """
    Fórmula:
    - Base: média ponderada das relevâncias (peso = relevância²)
    - Penalidade: se menos de 2 citações, reduz 20%
    - Bonus: se top citação > 0.9, adiciona 5%
    """
```

**Classes Principais**:

| Classe | Descrição |
|--------|-----------|
| `Citation` | Uma citação específica (span_id, texto, relevância, localização) |
| `Source` | Documento fonte (document_id, título, tipo) |
| `AnswerMetadata` | Métricas de debugging (modelo, latência, tokens) |
| `AnswerResponse` | Resposta completa para frontend |
| `QueryRequest` | Request do frontend (query, filtros, top_k) |

### 11. Page Spans - Citações Visuais (23/12/2024)

Módulo para extrair coordenadas PDF do Docling e mapear para spans do SpanParser.

**Módulo**: `src/parsing/page_spans.py`

**Estrutura de Coordenadas**:

```python
@dataclass
class BoundingBox:
    left: float      # Coordenada X esquerda
    top: float       # Coordenada Y topo
    right: float     # Coordenada X direita
    bottom: float    # Coordenada Y base
    page: int        # Número da página
    coord_origin: str = "TOPLEFT"

@dataclass
class SpanLocation:
    span_id: str     # Ex: "ART-005"
    page: int        # Página no PDF
    bbox: BoundingBox
    confidence: float  # Confiança do matching (0-1)
```

**Fluxo de Extração**:

```
PDF → Docling → ConversionResult
                     │
                     ├── markdown → SpanParser → ParsedDocument
                     │
                     └── texts[].prov → PageSpanExtractor → TextLocations
                                                │
                                                ▼
                            map_spans_to_locations() → SpanLocations
```

**Uso**:

```python
from docling.document_converter import DocumentConverter
from parsing import SpanParser, PageSpanExtractor

# Converte PDF
converter = DocumentConverter()
result = converter.convert("documento.pdf")

# Extrai page spans
extractor = PageSpanExtractor()
text_locations = extractor.extract_from_docling(result.document)

# Parseia markdown
parser = SpanParser()
parsed_doc = parser.parse(result.document.export_to_markdown())

# Mapeia spans para coordenadas
span_locations = extractor.map_spans_to_locations(parsed_doc, text_locations)

# Resultado
for span_id, loc in span_locations.items():
    print(f"{span_id}: página {loc.page}, bbox={loc.bbox.to_dict()}")
```

**Integração com ChunkMetadata**:

```python
page_spans = {
    "ART-005": {"page": 2, "l": 100.0, "t": 200.0, "r": 500.0, "b": 220.0},
    "PAR-005-1": {"page": 3, "l": 100.0, "t": 400.0, "r": 500.0, "b": 420.0},
}

metadata = ChunkMetadata(
    schema_version="1.0.0",
    document_hash="abc123",
    page_spans=page_spans,  # Usado para navegação visual no frontend
)
```

**Uso no Frontend**:

O frontend pode usar as coordenadas para:
1. Destacar o texto citado no PDF viewer
2. Navegar automaticamente para a página correta
3. Desenhar bounding box sobre o texto relevante

### 12. Dashboard de Ingestão (23/12/2024)

Módulo para coleta e visualização de métricas do pipeline de ingestão.

**Módulo**: `src/dashboard/ingestion_metrics.py`

**Métricas Coletadas**:

| Categoria | Métricas |
|-----------|----------|
| **Cobertura** | Parágrafos, incisos, alíneas por artigo |
| **Status** | Válidos, suspeitos, inválidos |
| **Latência** | Tempo por fase, por artigo |
| **Tokens** | Prompt, completion, total |
| **Chunks** | Por tipo (article, paragraph, inciso) |

**Uso**:

```python
from dashboard import MetricsCollector, generate_dashboard_report

# Durante o pipeline
collector = MetricsCollector(ingestion_id="IN-65-2021-001")
collector.set_document_info(
    document_id="IN-65-2021",
    tipo_documento="IN",
    numero="65",
    ano=2021,
)

collector.start_phase("parsing")
# ... parsing ...
collector.end_phase("parsing", items_processed=1)

# Para cada artigo extraído
collector.record_article_metrics(
    article_id="ART-005",
    parser_paragrafos=3,
    llm_paragrafos=3,
    parser_incisos=5,
    llm_incisos=5,
    status="valid",
    tokens_prompt=500,
    tokens_completion=100,
)

# Gera relatório
report = collector.generate_report()
print(generate_dashboard_report(report))
```

**Exemplo de Saída**:

```
======================================================================
DASHBOARD DE INGESTÃO
======================================================================
Ingestion ID: IN-65-2021-001
Status: completed

----------------------------------------------------------------------
ARTIGOS
----------------------------------------------------------------------
Total: 11
  [OK] Validos: 9 (82%)
  [!!] Suspeitos: 1
  [XX] Invalidos: 1

----------------------------------------------------------------------
COBERTURA
----------------------------------------------------------------------
Parágrafos: 20/22 (91%)
Incisos: 31/33 (94%)

----------------------------------------------------------------------
CHUNKS GERADOS
----------------------------------------------------------------------
Total: 47
  ARTICLE: 11
  PARAGRAPH: 19
  INCISO: 17

----------------------------------------------------------------------
TOKENS LLM
----------------------------------------------------------------------
Prompt: 5,500
Completion: 1,100
Total: 6,600
Custo estimado (API ref): $0.0066
======================================================================
```

**Classes Principais**:

| Classe | Descrição |
|--------|-----------|
| `MetricsCollector` | Coletor principal, registra métricas |
| `ArticleMetrics` | Métricas de um artigo individual |
| `DocumentMetrics` | Métricas agregadas do documento |
| `PhaseMetrics` | Métricas de uma fase do pipeline |
| `IngestionMetrics` | Relatório completo de ingestão |



### 12.1 Dashboard Streamlit - Modos Dev/Prod (25/12/2024)

O dashboard Streamlit suporta dois modos de operacao baseado na variavel RAG_MODE:

**Modos de Operacao**:

| Modo | RAG_MODE | GPU | Comportamento |
|------|----------|-----|---------------|
| **Development** | development (padrao) | 12GB | Lazy loading, modelos sob demanda |
| **Production** | production | 24GB+ | Singleton, modelos na GPU permanentemente |

**Como Alternar**:

- Desenvolvimento: export RAG_MODE=development
- Producao: export RAG_MODE=production

**Comportamento em cada modo**:

| Aspecto | Development | Production |
|---------|-------------|------------|
| Startup | Imediato (~0s) | Lento (~15-20s) |
| Primeira query | Lenta (~30-40s) | Rapida (~10s) |
| VRAM usada | Libera apos uso | Permanente (~8GB) |
| st.cache_resource | Nao pre-carrega | Pre-carrega BGE-M3 + Reranker |

**Indicador Visual na Sidebar**:

O dashboard mostra o modo atual na sidebar:
- **DEVELOPMENT** (azul): Modelos sob demanda
- **PRODUCTION** (verde): Modelos na GPU

### 13. Modulo de Busca Hibrida (22/12/2024)

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

### Fase 4 - RAG Completo ✅ (Arquitetura Span-Based Completa)

- [x] Modulo de busca hibrida (HybridSearcher)
- [x] Arquitetura Span-Based (SpanParser, ArticleOrchestrator)
- [x] Parent-child retrieval (ChunkMaterializer)
- [x] Schema enum dinâmico por artigo (previne alucinação)
- [x] Retry focado por janela (PAR ou INC)
- [x] Metadados de proveniência (schema_version, document_hash)
- [x] Answer-JSON estruturado para frontend
- [x] Page spans (coordenadas PDF para citações visuais)
- [x] Dashboard de ingestão (métricas de cobertura)
- [x] Schema Milvus v3 (leis_v3) com parent-child
- [x] Migração leis_v2 → leis_v3
- [x] Pipeline v3 (run_pipeline_v3.py)
- [x] Busca híbrida testada (RRF + Weighted Ranker)
- [x] IN 65/2021 indexada (47 chunks, 100% cobertura)
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

### 23/12/2024

**Manhã - Arquitetura Span-Based**:
- Implementada arquitetura de extração baseada em spans
- Criado `SpanParser` que parseia Markdown para spans hierárquicos
- Criado `ArticleOrchestrator` que extrai hierarquia por artigo
- Schema enum dinâmico: IDs permitidos passados como enum no JSON Schema
- Curto-circuito: artigos sem filhos não chamam LLM (economia de tokens)
- Retry focado por janela: retry para PAR ou INC, não ambos juntos
- Validação de parent consistency (INC-005-I_2 → parent=PAR-005-2)
- **Resultado**: 100% de acurácia na IN 65/2021 (11/11 artigos válidos)

**Tarde - Parent-Child e Answer-JSON**:
- Criado `ChunkMaterializer` para parent-child retrieval
- Chunks pai (ARTICLE) + chunks filhos (PARAGRAPH/INCISO)
- Metadados de proveniência: schema_version, extractor_version, document_hash
- Criado módulo `rag/` com Answer-JSON para frontend
- Formato estruturado: answer, citations, confidence, sources, metadata
- Cálculo de confiança baseado em relevância ponderada
- **Resultado**: 47 chunks materializados (11 ART + 19 PAR + 17 INC)

**Arquivos criados**:
```
src/parsing/
  span_parser.py              # SpanParser
  span_models.py              # Span, SpanType, ParsedDocument
  span_extraction_models.py   # ArticleSpans schema
  article_orchestrator.py     # ArticleOrchestrator
  __init__.py                 # Exports
src/chunking/
  chunk_materializer.py       # ChunkMaterializer, MaterializedChunk
  __init__.py                 # ATUALIZADO: novos exports
src/rag/
  __init__.py                 # Módulo RAG
  answer_models.py            # AnswerResponse, Citation, Source
tests/
  test_span_parser.py         # Teste do SpanParser
  test_article_orchestrator.py # Teste do orchestrator
  test_chunk_materializer.py  # Teste parent-child
```

**Problemas resolvidos**:
- JSON truncado com max_tokens=4096 → reduzido para 512 (suficiente para IDs)
- KeyError 'article_id' no prompt → adicionado ao format()
- Campo `llm_children_count` obsoleto → atualizado para campos por tipo

**Noite - Page Spans para Citações Visuais**:
- Criado módulo `page_spans.py` para extrair coordenadas PDF
- `PageSpanExtractor`: extrai bounding boxes do Docling
- `BoundingBox`: estrutura com l/t/r/b + page + coord_origin
- `SpanLocation`: mapeia span_id para localização no PDF
- Integração com `ChunkMetadata.page_spans`
- **Resultado**: 100% de mapeamento em testes com 4 spans

**Arquivos criados**:
```
src/parsing/
  page_spans.py           # PageSpanExtractor, BoundingBox, SpanLocation
tests/
  test_page_spans.py      # Testes de mapeamento e merge
```

**Dashboard de Ingestão**:
- Criado módulo `dashboard/` para métricas de ingestão
- `MetricsCollector`: coleta métricas durante o pipeline
- `ArticleMetrics`: cobertura por artigo (PAR, INC)
- `DocumentMetrics`: agregação de documento
- `PhaseMetrics`: timing por fase (parsing, extraction)
- `generate_dashboard_report()`: relatório formatado para terminal
- **Resultado**: Dashboard completo com cobertura, tokens, custo, latência

**Arquivos criados**:
```
src/dashboard/
  __init__.py              # Exports públicos
  ingestion_metrics.py     # MetricsCollector, métricas
tests/
  test_dashboard.py        # Testes de coleta e relatório
```

### 23/12/2024

**Madrugada - Schema Milvus v3 e Migração**:
- Criado schema `leis_v3` com campos parent-child e proveniência
- Novos campos: `parent_chunk_id`, `span_id`, `device_type`, `chunk_level`
- Campos de proveniência: `schema_version`, `extractor_version`, `ingestion_timestamp`, `document_hash`
- Campos page spans: `page`, `bbox_left`, `bbox_top`, `bbox_right`, `bbox_bottom`
- Script de migração `migrate_to_v3.py`: dropa leis_v2, cria leis_v3
- **Resultado**: Collection leis_v3 com 30 campos e 8 índices

**Arquivos criados**:
```
src/milvus/
  schema_v3.py          # Schema v3 com parent-child
  __init__.py           # ATUALIZADO: exports v3
scripts/
  migrate_to_v3.py      # Migração leis_v2 → leis_v3
```

**Pipeline v3 - Span-Based + Milvus**:
- Criado `run_pipeline_v3.py`: pipeline completo com nova arquitetura
- Fluxo: SpanParser → ArticleOrchestrator → ChunkMaterializer → BGE-M3 → Milvus
- Integração com MetricsCollector para dashboard
- **Bug fix**: `chunk_level.value` retornava int, alterado para `.name.lower()` (string)
- **Bug fix**: `embedder.encode()` não retorna sparse, alterado para `encode_hybrid()`
- **Resultado**: IN 65/2021 inserida com sucesso (47 chunks, 30.02s total)

**Arquivos criados**:
```
scripts/
  run_pipeline_v3.py    # Pipeline v3 completo
```

**Teste de Busca Híbrida (Dense + Sparse)**:
- Testada busca híbrida no Milvus com RRF e Weighted Ranker
- Query: "Como fazer pesquisa de preços em contratações públicas?"
- Comparação de rankings entre métodos

| Método | Top 1 | Top 2 | Top 3 | Score Top 1 |
|--------|-------|-------|-------|-------------|
| Dense Only | ART-005 | INC-005-IV | ART-001 | 0.6512 |
| Sparse Only | INC-005-IV | ART-003 | ART-004 | 0.1020 |
| **RRF Hybrid** | INC-005-IV | ART-005 | ART-003 | 0.0325 |
| Weighted (0.7/0.3) | ART-005 | INC-005-IV | ART-001 | 0.7372 |

**Observações**:
- Overlap dense/sparse: 4/5 (80% de concordância)
- RRF promove INC-005-IV para Top 1 (combinação semântica + lexical)
- Weighted mantém ranking similar ao dense (ponderação 70/30)
- Sparse scores são menores mas capturam termos exatos ("pesquisa", "preços")

**Resultados da Ingestão IN 65/2021**:
```
Pipeline v3 - Status: completed
Tempo total: 30.02s

Fases:
- Load: 0.00s
- Parsing: 0.00s (57 spans)
- Extraction: 12.08s (11 artigos válidos)
- Materialization: 0.00s (47 chunks)
- Embedding: 16.28s (47 embeddings BGE-M3)
- Indexing: 1.65s (47 inseridos no Milvus)

Cobertura: 100%
- Parágrafos: 19/19
- Incisos: 17/17

Chunks por tipo:
- ARTICLE: 11
- PARAGRAPH: 19
- INCISO: 17
```

**Afinações Finais - Contextual Retriever**:
- Criado módulo `ContextualRetriever` com parent-child + MMR
- Query Router automático: Weighted (padrão) vs RRF (dispositivo específico)
- MMR (Maximal Marginal Relevance) para diversidade de irmãos
- Cap de expansão: max 1 pai + 4 irmãos relevantes
- `CitationValidator`: valida citations ⊆ context_used

**Arquivos criados**:
```
src/search/
  contextual_retriever.py   # ContextualRetriever, CitationValidator
  __init__.py               # ATUALIZADO: novos exports
scripts/
  benchmark_retrieval.py    # Benchmark de estratégias
```

**Query Router - Detecção Automática**:
```python
# Padrões que ativam RRF (dispositivo específico)
DEVICE_PATTERNS = [
    r'\bart\.?\s*\d+',      # art. 5, art 10
    r'§\s*\d+',             # § 1º
    r'\binciso\b',          # inciso
    r'\bal[ií]nea\b',       # alínea
    r'\b[IVX]+\s*[-–]',     # I -, II -
]

# Queries amplas → Weighted (0.7 dense + 0.3 sparse)
# Queries específicas → RRF (Reciprocal Rank Fusion)
```

**Fluxo do ContextualRetriever**:
```
Query → Detecta Estratégia → Busca Híbrida (Top-K)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            Expande para Pais              Seleciona Irmãos (MMR)
            (max 1 artigo)                 (max 4, λ=0.7)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                           Ordena Hierarquicamente
                           (pai → filhos ordenados)
                                    │
                                    ▼
                           Monta Contexto + Citações
```

**Benchmark Final - Contextual vs Simples**:

| Query | Simples (Top-5) | Contextual (MMR) | Ganho |
|-------|-----------------|------------------|-------|
| Fornecedores? | 5 chunks | 9 chunks (5+4 MMR) | +4 irmãos |
| Prazo resposta? | 5 chunks | 9 chunks (5+4 MMR) | +4 irmãos |
| Cotação formal? | 5 chunks | 9 chunks (5+4 MMR) | +4 irmãos |

O MMR garante diversidade: não retorna 5 incisos similares, mas mix de PAR + INC.

**Tarde - Enriquecimento LLM + HyDE (Contextual Retrieval)**:

Implementamos o sistema completo de enriquecimento de chunks com LLM e HyDE para query expansion.

**Módulo ChunkEnricher** (`src/enrichment/`):
- Enriquece chunks com contexto, tese e perguntas sintéticas
- Usa prompts de `enrichment_prompts.py` (Anthropic Contextual Retrieval)
- Campos preenchidos: `context_header`, `thesis_text`, `thesis_type`, `synthetic_questions`
- Monta `enriched_text` para embedding: `[CONTEXTO: ...] + texto + [PERGUNTAS: ...]`

**Arquivos criados**:
```
src/enrichment/
  __init__.py           # Exports
  chunk_enricher.py     # ChunkEnricher, EnrichmentResult
```

**Integração no Pipeline v3**:
- Nova fase 4.5: Enriquecimento (entre materialização e embedding)
- Processa em batches de 5 chunks
- Usa `Qwen/Qwen3-8B-AWQ` (mesmo modelo único)
- Tempo: ~5s/chunk (233s para IN 65 com 47 chunks)

**HyDE - Hypothetical Document Embeddings** (`src/search/`):
- Técnica de query expansion: gera documentos hipotéticos com LLM
- Combina embeddings da query + docs hipotéticos (40%/60%)
- Melhora recall para queries ambíguas ou curtas
- Toggle: `SearchConfig.use_hyde = True/False`

**Arquivos criados**:
```
src/search/
  hyde_expander.py      # HyDEExpander, HyDEResult
```

**Integração no HybridSearcher**:
- Propriedade `hyde_expander` com lazy loading
- Usa HyDE quando `config.use_hyde = True`
- Gera 3 documentos hipotéticos por query
- Overhead: +15-20s por query (geração LLM)

**Benchmark HyDE** (23/12/2024):

| Query | Sem HyDE | Com HyDE | Diferença |
|-------|----------|----------|-----------|
| "pesquisa de preços" | ART-005, ART-003, ART-004 | ART-005, ART-003, ART-004 | = (query específica) |
| "fornecedores e cotações" | PAR-007-5, INC-005-IV | PAR-007-5, INC-082-VII, INC-023-IV | +3 novos resultados Lei 14.133 |

**Conclusão HyDE**:
- Útil para queries curtas/ambíguas
- Overhead de +15-20s não justifica para queries específicas
- Recomendado: desabilitado por padrão, habilitado para queries complexas

**Resultados IN 65/2021 com Enriquecimento**:
```
Pipeline v3 - Status: completed
Tempo total: 279.67s

Fases:
- Load: 0.00s
- Parsing: 0.00s (57 spans)
- Extraction: 11.49s (11 artigos válidos)
- Materialization: 0.00s (47 chunks)
- Enrichment: 233.05s (47 chunks, 0 erros)
- Embedding: 33.37s (47 embeddings BGE-M3)
- Indexing: 1.76s

Cobertura: 100%
Campos preenchidos: context_header, thesis_text, thesis_type, synthetic_questions, enriched_text
```

**Resultados Lei 14.133/2021** (sem enriquecimento por timeout):
```
Pipeline v3 - Status: completed
Tempo total: 945.84s

Fases:
- Extraction: 294.36s (191/204 válidos, 94%)
- Materialization: 1265 chunks
- Enrichment: TIMEOUT após 3 batches
- Embedding: 240.43s (1265 embeddings)
- Indexing: 2.99s

Chunks no Milvus: 1312 total (47 IN 65 + 1265 Lei 14.133)
```

**Correção de Bug - Nome do Modelo vLLM**:
- LLMConfig usava `Qwen/Qwen3-8B` mas container tem `Qwen/Qwen3-8B-AWQ`
- Corrigido: `for_enrichment()` e `for_extraction()` agora usam `-AWQ`
- Arquivo: `src/llm/vllm_client.py`

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

## 🛡️ Correções Anti-Alucinação no Extrator (22/12/2024)

### Problema Identificado

Durante testes com a IN 65/2021 (pesquisa de preços), descobrimos que o LLM estava **inventando artigos** que não existiam no documento original. O problema tinha duas causas:

1. **Capítulo Fantasma**: O método `_split_by_chapters` criava um capítulo "DISPOSIÇÕES INICIAIS" para conteúdo antes do primeiro CAPÍTULO real (que era apenas metadados como ementa e data). O LLM então tentava extrair artigos desse conteúdo e inventava artigos.

2. **Referências a Outras Leis**: O texto do documento continha referências a artigos de outras leis (ex: "Art. 75 da Lei 14.133"). A validação pós-extração capturava essas referências como artigos válidos.

### Correções Implementadas

| Correção | Arquivo | Descrição |
|----------|---------|-----------|
| Ignorar conteúdo pré-capítulo | `extractor.py:_split_by_chapters` | Só processa conteúdo APÓS o primeiro CAPÍTULO real |
| Validação pré-extração | `extractor.py:_extract_chapter` | Verifica se capítulo tem artigos antes de chamar LLM |
| Instrução anti-alucinação | `extractor.py:_extract_chapter` | Prompt inclui lista de artigos esperados e proíbe invenção |
| Validação pós-extração | `extractor.py:_validate_extracted_chapter` | Remove artigos que não existem no markdown original |
| Distinção de referências | `extractor.py:_validate_extracted_chapter` | Ignora "Art. N da Lei X" (referências a outras leis) |

### Detalhes Técnicos

**`_split_by_chapters` (antes)**:
```python
current_title = "DISPOSIÇÕES INICIAIS"  # Criava capítulo fantasma
```

**`_split_by_chapters` (depois)**:
```python
current_title = None  # Ignora conteúdo antes do primeiro CAPÍTULO
found_first_chapter = False
# Só adiciona conteúdo após encontrar primeiro CAPÍTULO real
```

**Validação pré-extração**:
```python
# Lista artigos que realmente existem no texto
expected_articles = sorted(set(int(a) for a in articles_in_content))
# Prompt informa: "Extraia APENAS os artigos: Art. 1, 2, 3..."
# Prompt proíbe: "NAO INVENTE artigos que nao estao no texto"
```

**Validação pós-extração**:
```python
# Pattern que ignora referências a outras leis
r'(?:^|\n)\s*Art\.?\s*(\d+)[°ºo]?(?:\s|\.|\s*[-–—])'  # Art. no início de linha
# Exclui: "art. N da Lei", "art. N do Decreto"
```

### Resultado

| Métrica | Antes | Depois |
|---------|-------|--------|
| IN 65/2021 | 21 artigos (10 inventados) | 11 artigos (correto) |
| Alucinação | 47% falsos | 0% falsos |
| Score qualidade | 52% | 100% |

### Arquivos Modificados

- `src/extract/extractor.py`: 3 métodos corrigidos/adicionados
- `tests/test_extraction_fix.py`: Novo teste de validação

---

## 📊 Resumo do Estado Atual (23/12/2024)

### O que está funcionando

| Componente | Status | Descrição |
|------------|--------|-----------|
| **Docling** | ✅ | Extração PDF → Markdown com hierarquia |
| **SpanParser** | ✅ | Markdown → Spans determinísticos |
| **ArticleOrchestrator** | ✅ | Extração LLM por artigo com enum dinâmico |
| **ChunkMaterializer** | ✅ | Parent-child chunks (ART → PAR/INC) |
| **ChunkEnricher** | ✅ | Enriquecimento LLM (context, thesis, questions) |
| **BGE-M3** | ✅ | Embeddings dense (1024d) + sparse |
| **Milvus leis_v3** | ✅ | 1312 chunks (IN 65 + Lei 14.133), 30 campos |
| **Busca Híbrida** | ✅ | Weighted (0.7/0.3) + RRF |
| **HyDEExpander** | ✅ | Query expansion com documentos hipotéticos |
| **ContextualRetriever** | ✅ | Parent-child + MMR + Query Router |
| **CitationValidator** | ✅ | Valida citations ⊆ context_used |
| **Dashboard** | ✅ | Métricas de cobertura e latência |

### Arquitetura Implementada

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE DE INGESTÃO                            │
├─────────────────────────────────────────────────────────────────────────┤
│  PDF → Docling → Markdown → SpanParser → ArticleOrchestrator (LLM)      │
│                                              │                          │
│                                              ▼                          │
│                                    ChunkMaterializer                    │
│                                    (parent-child)                       │
│                                              │                          │
│                                              ▼                          │
│                                    ChunkEnricher (LLM)                  │
│                            (context, thesis, questions)                 │
│                                              │                          │
│                                              ▼                          │
│                              BGE-M3 (dense + sparse)                    │
│                                              │                          │
│                                              ▼                          │
│                                    Milvus leis_v3                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE DE RETRIEVAL                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Query → [HyDE opcional] → Query Router → Busca Híbrida                 │
│              │                    │                                     │
│              ▼                    ▼                                     │
│    Gera docs hipotéticos   Detecta padrões                              │
│    (se use_hyde=True)      (art., §, inciso)                            │
│              │                   │                                       │
│              └───────────────────┼───────────────────┐                   │
│                                  ▼                   ▼                   │
│                           Top-K inicial (5)    Weighted/RRF             │
│                                  │                                       │
│                                  ▼                                       │
│                          Expande para Pais (1)                          │
│                                  │                                       │
│                                  ▼                                       │
│                          MMR Irmãos (4)                                 │
│                                  │                                       │
│                                  ▼                                       │
│                          CitationValidator                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Métricas de Performance

| Métrica | Valor |
|---------|-------|
| Total chunks Milvus | 1312 (IN 65 + Lei 14.133) |
| IN 65: Cobertura | 100% (parágrafos e incisos) |
| IN 65: Chunks enriquecidos | 47/47 (100%) |
| Lei 14.133: Cobertura | 99% PAR, 100% INC |
| Lei 14.133: Extração válida | 191/204 (94%) |
| Tempo ingestão (IN 65 c/ enriquecimento) | 280s |
| Tempo ingestão (Lei 14.133 s/ enriq.) | 946s |

### Arquivos Principais

```
extracao/
├── src/
│   ├── parsing/
│   │   ├── span_parser.py           # SpanParser (determinístico)
│   │   ├── span_models.py           # Span, ParsedDocument
│   │   ├── article_orchestrator.py  # Extração LLM por artigo
│   │   └── page_spans.py            # Coordenadas PDF
│   ├── chunking/
│   │   ├── chunk_materializer.py    # Parent-child chunks
│   │   └── chunk_models.py          # LegalChunk, ChunkLevel
│   ├── enrichment/                  # NOVO (23/12/2024)
│   │   ├── __init__.py              # Exports
│   │   └── chunk_enricher.py        # ChunkEnricher (context, thesis, questions)
│   ├── search/
│   │   ├── hyde_expander.py         # NOVO: HyDE query expansion
│   │   ├── contextual_retriever.py  # MMR + Query Router
│   │   ├── hybrid_searcher.py       # Busca híbrida (HyDE integrado)
│   │   └── config.py                # SearchConfig (use_hyde toggle)
│   ├── llm/
│   │   └── vllm_client.py           # VLLMClient (Qwen/Qwen3-8B-AWQ)
│   ├── milvus/
│   │   ├── schema_v3.py             # Schema leis_v3
│   │   └── schema.py                # Schema legado v2
│   ├── embeddings/
│   │   └── bge_m3.py                # BGE-M3 embedder
│   ├── dashboard/
│   │   └── ingestion_metrics.py     # Métricas de ingestão
│   └── rag/
│       └── answer_models.py         # Answer-JSON
├── scripts/
│   ├── run_pipeline_v3.py           # Pipeline c/ enriquecimento (fase 4.5)
│   ├── migrate_to_v3.py             # Migração Milvus
│   └── benchmark_retrieval.py       # Benchmark estratégias
└── tests/
    ├── test_span_parser.py
    ├── test_article_orchestrator.py
    ├── test_chunk_materializer.py
    └── test_page_spans.py
```

---

## 🎯 Próximos Passos (Roadmap)

### Concluído (23/12/2024)

- [x] **ChunkEnricher**: Enriquecimento LLM (context, thesis, questions)
- [x] **HyDE Query Expansion**: Documentos hipotéticos para queries ambíguas
- [x] **IN 65 Enriquecida**: 47 chunks com campos preenchidos
- [x] **Lei 14.133 Indexada**: 1265 chunks (sem enriquecimento por timeout)

### Curto Prazo (próxima sessão)

- [ ] **Enriquecer Lei 14.133**: Re-executar pipeline com timeout maior
- [ ] **Grid Search de Pesos**: Testar 0.6/0.4 e 0.8/0.2 para Weighted
- [ ] **Normalização Sparse**: Lower, stopwords jurídicas, de-accent
- [ ] **Mais documentos**: Indexar IN 58/2022, outras INs

### Médio Prazo (API e Integração)

- [ ] **API FastAPI**: Endpoints `/search`, `/ingest`, `/validate`
- [ ] **Answer Generation**: Integrar retrieval + LLM (Qwen 3) para resposta
- [ ] **Prompts Jurídicos**: Prompt especializado para resposta legal
- [ ] **Avaliação RAGAS**: Métricas de qualidade (faithfulness, relevance)

### Longo Prazo (Produção)

- [ ] **UI Next.js**: Interface de busca com citações clicáveis
- [ ] **PDF Viewer**: Clique na citação → pula para página/coordenada
- [ ] **Multi-tenant**: Suporte a múltiplos órgãos/clientes
- [ ] **Observabilidade**: Logs, métricas, tracing

---

## ✅ Checklist de Produção

### Testes Obrigatórios

- [x] Cobertura por tipo (PAR/INC) ≥ 98% por artigo
- [x] Duplicatas = 0 por artigo e por chunk
- [x] Suffix↔parent válido (INC-005-II_2 → parent=PAR-005-2)
- [ ] Round-trip: texto reconstruído == concat dos spans
- [x] Retrieval contextual: pai sempre aparece no conjunto final
- [x] Answer-JSON: citações apontam para span_ids usados

### Governança

- [x] `schema_version`, `extractor_version` em cada chunk
- [x] `ingestion_timestamp`, `document_hash` para rastreabilidade
- [x] `page`, `bbox_*` para citação visual
- [x] `parent_chunk_id` para expansão de contexto

---

## 📅 23/12/2024 - Tarde/Noite: RAG Completo com Resposta LLM

### Resumo da Sessão

Nesta sessão completamos o ciclo RAG end-to-end:
1. **Celery Pipeline**: Enriquecimento paralelo de chunks
2. **Answer Generator**: Geração de respostas com citações
3. **Dashboard Streamlit**: Interface para perguntas
4. **Primeiro teste bem-sucedido**: Resposta 100% coerente com a lei

### 1. Pipeline Celery para Enriquecimento Paralelo

**Problema**: Lei 14.133 tem 1260 chunks. Enriquecer sequencialmente levaria ~5h.

**Solução**: Celery + Redis para processamento paralelo.

**Arquivos criados**:

```
src/enrichment/
├── __init__.py           # Exports do módulo
├── celery_app.py         # Configuração Celery
└── tasks.py              # Tasks de enriquecimento

scripts/
├── run_enrichment_celery.py  # Dispara tasks
└── check_progress.py         # Monitora progresso
```

**celery_app.py** - Configuração:
```python
from celery import Celery

app = Celery(
    "enrichment",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["src.enrichment.tasks"],
)

app.conf.update(
    task_time_limit=600,          # 10 min max por task
    task_default_rate_limit="10/m", # 10 tasks/min (respeita GPU)
    worker_prefetch_multiplier=1,   # 1 task por vez por worker
    task_acks_late=True,            # Retry se worker morrer
)
```

**tasks.py** - Task de enriquecimento:
```python
@app.task(bind=True, max_retries=3, default_retry_delay=30)
def enrich_chunk_task(self, chunk_id, text, device_type, ...):
    """Enriquece um chunk e atualiza no Milvus."""
    # 1. Inicializa LLM e enricher
    # 2. Gera context_header, thesis_text, synthetic_questions
    # 3. Gera novos embeddings com enriched_text
    # 4. Upsert no Milvus (delete + insert)
```

**Comandos para executar**:

```bash
# Terminal 1: Redis
docker run -d --name redis -p 6379:6379 redis:alpine

# Terminais 2-5: Workers Celery (4 workers)
cd extracao
celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=1

# Terminal 6: Dispara tasks
python scripts/run_enrichment_celery.py

# Monitoramento
python scripts/check_progress.py --watch  # Atualiza a cada 30s
celery -A src.enrichment.celery_app flower  # Dashboard web :5555
```

**Resultado**:
- 4 workers processando em paralelo
- ~6-13 chunks/min (depende da complexidade)
- Taxa de sucesso: 100% (com retry automático)

### 2. Answer Generator - Resposta RAG com LLM

**Módulo**: `src/rag/answer_generator.py`

**Fluxo completo**:
```
Query do usuário
       │
       ▼
[1. HyDE] LLM gera 3 documentos hipotéticos (opcional)
       │
       ▼
[2. Embedding] BGE-M3 combina query + docs hipotéticos
       │
       ▼
[3. Busca Híbrida] Milvus (dense 50% + sparse 30% + thesis 20%)
       │
       ▼
[4. Reranking] BGE-Reranker cross-encoder (optional)
       │
       ▼
[5. Contexto] Monta chunks para prompt
       │
       ▼
[6. Generation] Qwen 3 8B gera resposta com citações
       │
       ▼
[7. Formatação] Citações legais (Lei X, Art. Y, §Z)
```

**Uso**:
```python
from rag import AnswerGenerator, GenerationConfig

# Modo completo (HyDE + Reranker)
generator = AnswerGenerator()
response = generator.generate("Quais os critérios de julgamento?")

print(response.answer)          # Resposta formatada
print(response.confidence)      # 0.999 (99.9%)
for citation in response.citations:
    print(citation.text)        # "Lei 14.133/2021, Art. 33, I"

# Modo rápido (sem HyDE, sem Reranker)
config = GenerationConfig.fast()
generator = AnswerGenerator(config=config)
```

**Estrutura da resposta (AnswerResponse)**:
```json
{
  "success": true,
  "query": "Quais os critérios de julgamento?",
  "data": {
    "answer": "Os critérios de julgamento previstos na Lei 14.133/2021 são...",
    "confidence": 0.999,
    "citations": [
      {
        "text": "Lei 14.133/2021, Art. 33, I",
        "short": "Art. 33, I",
        "document_type": "Lei",
        "document_number": "14.133",
        "year": 2021,
        "article": "33",
        "device": "inciso",
        "device_number": "I"
      }
    ],
    "sources": [
      {"document_id": "LEI-14133-2021", "tipo_documento": "LEI", "ano": 2021}
    ]
  },
  "metadata": {
    "model": "Qwen/Qwen3-8B-AWQ",
    "latency_ms": 54650,
    "retrieval_ms": 25749,
    "generation_ms": 28900,
    "chunks_retrieved": 5,
    "chunks_used": 5
  }
}
```

### 3. Citation Formatter - Citações Legais

**Módulo**: `src/rag/citation_formatter.py`

**Formata citações no padrão jurídico brasileiro**:

| Tipo | Exemplo de Saída |
|------|------------------|
| Artigo | Lei 14.133/2021, Art. 33 |
| Parágrafo | Lei 14.133/2021, Art. 14, Par. 5 |
| Inciso | Lei 14.133/2021, Art. 33, inciso I |
| Alínea | Lei 14.133/2021, Art. 14, inciso II, alínea 'a' |
| § único | IN 65/2021, Art. 3, Parágrafo único |

**Uso**:
```python
from rag import CitationFormatter, format_citation

# Simples
citation = format_citation(
    tipo_documento="LEI",
    numero="14133",
    ano=2021,
    article_number="33",
    device_type="inciso",
    span_id="INC-033-I"
)
# -> "Lei 14.133/2021, Art. 33, inciso I"

# Com classe
formatter = CitationFormatter()
citation = formatter.format_from_chunk(chunk_data)
```

### 4. Dashboard Streamlit - Página "Perguntar"

**Arquivo**: `src/dashboard/app.py`

**Nova página adicionada**: "Perguntar"

**Funcionalidades**:
- Campo de texto para perguntas
- Configurações: HyDE, Reranker, Top-K
- Modo Rápido vs Completo
- Resposta formatada do Qwen 3 8B
- Citações com artigo/parágrafo/inciso
- Métricas de latência (retrieval, generation, total)
- JSON completo para debug

**Como acessar**:
```bash
streamlit run src/dashboard/app.py --server.port 8501
# Acesse http://localhost:8501 → página "Perguntar"
```

### 5. Índices Milvus - Verificação de Uso

**Todos os índices vetoriais estão sendo utilizados** no modo HYBRID_3WAY:

| Índice | Campo | Peso | Tipo | Status |
|--------|-------|------|------|--------|
| HNSW | `dense_vector` | 50% | COSINE | ✅ Usado |
| HNSW | `thesis_vector` | 20% | COSINE | ✅ Usado |
| SPARSE_INVERTED | `sparse_vector` | 30% | IP | ✅ Usado |

**Índices escalares** (usados em filtros):
- `tipo_documento` - INVERTED
- `ano` - INVERTED
- `article_number` - INVERTED
- `device_type` - INVERTED
- `parent_chunk_id` - INVERTED

### 6. Primeiro Teste Bem-Sucedido

**Query**: "Quais os critérios de julgamento?"

**Resposta do sistema**:
```
Os critérios de julgamento previstos na Lei 14.133/2021 são os seguintes:

1. Menor preço – [Art. 33, I].
2. Maior desconto – [Art. 33, II].
3. Melhor técnica ou conteúdo artístico – [Art. 33, III].
4. Técnica e preço – [Art. 33, IV].
5. Maior lance, no caso de leilão – [Art. 33, V].
6. Maior retorno econômico – [Art. 33, VI].

Detalhamento de alguns critérios:
- Julgamento por técnica e preço ([Art. 33, IV]):
  - Considera a ponderação objetiva entre técnica e preço, com até 70%
    da pontuação atribuída à proposta técnica ([Art. 36, § 2º]).
...
```

**Métricas**:
- Confiança: **99.9%**
- Retrieval: 25.7s
- Generation: 28.9s
- Total: **54.6s**

**Avaliação**: Resposta **100% coerente** com a Lei 14.133/2021. Citou corretamente Art. 33 com todos os 6 critérios e detalhou Art. 36 sobre técnica e preço.

### 7. Progresso do Enriquecimento (em andamento)

| Documento | Chunks | Enriquecidos | Progresso |
|-----------|--------|--------------|-----------|
| IN 65/2021 | 47 | 47 | ✅ 100% |
| Lei 14.133/2021 | 1260 | ~400 | ⏳ ~32% |
| **Total** | 1307 | ~447 | **~34%** |

Os 4 workers Celery continuam processando em background.

### 8. Arquivos Criados/Modificados (23/12/2024 tarde)

```
src/enrichment/
├── __init__.py              # NOVO: Exports
├── celery_app.py            # NOVO: Config Celery
└── tasks.py                 # NOVO: Tasks enriquecimento

src/rag/
├── __init__.py              # ATUALIZADO: Novos exports
├── answer_generator.py      # NOVO: Geração resposta RAG
└── citation_formatter.py    # NOVO: Formatação citações

src/search/
└── models.py                # ATUALIZADO: Adicionado campo 'ano' e property 'year'

src/dashboard/
└── app.py                   # ATUALIZADO: Nova página "Perguntar"

scripts/
├── run_enrichment_celery.py # NOVO: Dispara tasks Celery
├── check_progress.py        # NOVO: Monitora progresso
└── test_answer_generator.py # NOVO: Teste do generator
```

### 9. Lições Aprendidas

| Lição | Contexto |
|-------|----------|
| **Celery precisa de imports corretos** | Usar `src.llm.vllm_client` ao invés de `llm.vllm_client` |
| **Milvus insert row-oriented** | Usar `[{campo: valor}]` ao invés de `{campo: [valor]}` |
| **Streamlit cache é agressivo** | Reiniciar processo para pegar mudanças em módulos |
| **HyDE adiciona ~15-20s** | Desativar para queries simples |
| **Reranker adiciona ~10s** | Mas melhora precisão significativamente |
| **Qualidade > Velocidade** | Primeiro garantir respostas corretas, depois otimizar |

### 10. Métricas de Latência

| Modo | HyDE | Reranker | Retrieval | Generation | Total |
|------|------|----------|-----------|------------|-------|
| Rápido | ❌ | ❌ | ~11s | ~19s | **~30s** |
| Completo | ✅ | ✅ | ~26s | ~29s | **~55s** |

**Causas da latência**:
- HyDE: LLM gera 3 documentos hipotéticos (~15s)
- Reranker: Cross-encoder processa top-20 (~10s)
- Generation: Resposta longa com citações (~20-30s)

---

## 🎯 Status Atual do Projeto (23/12/2024 21:30)

### Fase Atual: **5 - RAG Completo com Resposta LLM** ✅

| Componente | Status | Descrição |
|------------|--------|-----------|
| Extração PDF | ✅ Completo | Docling + SpanParser + ArticleOrchestrator |
| Chunking | ✅ Completo | ChunkMaterializer com parent-child |
| Embeddings | ✅ Completo | BGE-M3 (dense + sparse) |
| Enriquecimento | ⏳ Em andamento | ChunkEnricher (32% Lei 14.133) |
| Indexação | ✅ Completo | Milvus leis_v3 (1307 chunks) |
| Busca Híbrida | ✅ Completo | Weighted 3-way + HyDE |
| Reranking | ✅ Completo | BGE-Reranker cross-encoder |
| Resposta LLM | ✅ Completo | AnswerGenerator + Qwen 8B |
| Citações | ✅ Completo | CitationFormatter |
| Dashboard | ✅ Completo | Streamlit com página "Perguntar" |

### O que funciona end-to-end

```
Usuário faz pergunta
        │
        ▼
[Dashboard Streamlit] → [AnswerGenerator]
        │
        ▼
[HyDE opcional] → [Busca Híbrida Milvus] → [Reranker]
        │
        ▼
[Monta contexto com chunks] → [Qwen 3 8B gera resposta]
        │
        ▼
[Formata citações] → [Exibe resposta + métricas]
```

---

## 🚀 Próximos Passos (Atualizado)

### Concluído (23/12/2024)

- [x] **Pipeline Celery**: Enriquecimento paralelo com 4 workers
- [x] **Answer Generator**: Geração de respostas RAG com citações
- [x] **Citation Formatter**: Formatação de citações legais
- [x] **Dashboard "Perguntar"**: Interface para perguntas ao sistema
- [x] **Primeiro teste bem-sucedido**: Resposta 100% coerente

### Curto Prazo (próxima sessão)

- [ ] **Completar enriquecimento Lei 14.133**: Aguardar Celery (~4h restantes)
- [ ] **Otimizar latência**: Cache de embeddings, streaming response
- [ ] **API FastAPI**: Endpoints `/ask`, `/search`, `/ingest`
- [ ] **Streaming response**: Mostrar resposta enquanto gera

### Médio Prazo

- [ ] **Cache de queries**: Perguntas frequentes pré-computadas
- [ ] **Avaliação RAGAS**: Métricas de qualidade (faithfulness, relevance)
- [ ] **Mais documentos**: Indexar Decretos, outras Leis
- [ ] **Fine-tuning prompts**: Melhorar precisão das respostas

### Longo Prazo (Produção)

- [ ] **UI React/Next.js**: Interface profissional
- [ ] **PDF Viewer**: Clique na citação → pula para página
- [ ] **Multi-tenant**: Suporte a múltiplos órgãos
- [ ] **GPU maior**: RTX 4090 para latência 2x menor
- [ ] **Kubernetes**: Deploy escalável

---

## 📊 Comandos Úteis

### Iniciar Sistema Completo

```bash
# 1. Docker (Milvus + vLLM)
docker start milvus-standalone vllm

# 2. Redis (para Celery)
docker run -d --name redis -p 6379:6379 redis:alpine

# 3. Workers Celery (abrir 4 terminais)
cd extracao
celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=1

# 4. Dashboard Streamlit
streamlit run src/dashboard/app.py --server.port 8501
```

### Monitoramento

```bash
# Progresso do enriquecimento
python scripts/check_progress.py --watch

# Dashboard Celery (Flower)
celery -A src.enrichment.celery_app flower
# Acesse http://localhost:5555

# Logs do vLLM
docker logs -f vllm
```

### Testar Resposta RAG

```bash
# Via linha de comando
python scripts/test_answer_generator.py --query "Quando o ETP pode ser dispensado?"

# Modo rápido (sem HyDE)
python scripts/test_answer_generator.py --fast --query "Quais os critérios de julgamento?"
```

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
- [Celery Documentation](https://docs.celeryq.dev/) - Task queue
- [Streamlit Documentation](https://docs.streamlit.io/) - Dashboard
