# 📚 Pipeline de Extração para Sistema RAG

Pipeline de extração agêntico para documentos legais brasileiros (leis, decretos, instruções normativas).

## 🚀 Quick Start

### 1. Instalar dependências

```bash
# Usando pip
pip install -e ".[dev]"

# Ou usando uv (recomendado - mais rápido)
uv pip install -e ".[dev]"
```

### 2. Testar extração com Docling

```bash
# Coloque um PDF de lei na pasta tests/fixtures/
python scripts/test_docling_extraction.py tests/fixtures/sua_lei.pdf
```

## 📁 Estrutura do Projeto

```
extracao/
├── src/
│   ├── extractors/      # Extratores de documentos (Docling)
│   ├── agents/          # Agentes LangGraph
│   ├── chunking/        # Estratégias de chunking
│   ├── embeddings/      # Integração BGE-M3
│   ├── llm/             # Cliente Ollama/Qwen
│   ├── storage/         # Integração Milvus
│   └── api/             # Endpoints FastAPI
├── scripts/             # Scripts utilitários
├── tests/
│   └── fixtures/        # PDFs de teste
└── pyproject.toml
```

## 🛠️ Stack

| Componente | Tecnologia |
|------------|------------|
| Extração | Docling |
| Agentes | LangGraph |
| LLM | Qwen 2.5 via Ollama |
| Embeddings | BGE-M3 |
| Vector Store | Milvus 2.6 |
| API | FastAPI |

## 📋 Documentação

Veja [claude.md](../claude.md) para decisões de arquitetura e roadmap.

