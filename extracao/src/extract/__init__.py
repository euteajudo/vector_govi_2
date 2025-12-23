"""
Módulo de Extração - API similar ao LlamaExtract.

Este módulo fornece uma API elegante e simples para extração de dados
estruturados de documentos, usando tecnologias 100% open-source.

Stack:
- Docling: Parsing de PDF/DOCX/HTML para Markdown
- Qwen 3 8B: Transformação MD -> JSON estruturado
- Pydantic: Definição e validação de schemas
- LangGraph: Orquestração agêntica (opcional)

Uso básico:
    from extract import Extractor, ExtractConfig
    from models.legal_document import LegalDocument
    
    # Extração simples
    extractor = Extractor()
    result = extractor.extract("documento.pdf", schema=LegalDocument)
    print(result.data)
    
    # Com configuração customizada
    config = ExtractConfig.for_legal_documents()
    result = extractor.extract("lei.pdf", schema=LegalDocument, config=config)

Usando Extraction Agent (workflows reutilizáveis):
    from extract import ExtractionAgent
    
    agent = ExtractionAgent(
        name="lei-parser",
        schema=LegalDocument,
        config=ExtractConfig.for_legal_documents(),
    )
    
    # Extrair múltiplos documentos
    result1 = agent.extract("lei1.pdf")
    result2 = agent.extract("lei2.pdf")
    
    # Ver estatísticas
    print(agent.get_stats())
"""

from .config import (
    ExtractConfig,
    ExtractMode,
    ExtractTarget,
    ChunkMode,
    LLMConfig,
    DoclingConfig,
    ValidationConfig,
)

from .extractor import (
    Extractor,
    ExtractionAgent,
    ExtractionResult,
)

__all__ = [
    # Config
    "ExtractConfig",
    "ExtractMode",
    "ExtractTarget",
    "ChunkMode",
    "LLMConfig",
    "DoclingConfig",
    "ValidationConfig",
    # Extractor
    "Extractor",
    "ExtractionAgent",
    "ExtractionResult",
]

