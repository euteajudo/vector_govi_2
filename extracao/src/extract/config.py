"""
Configuração de Extração - Inspirado no LlamaExtract.

Este módulo define as configurações que controlam o processo de extração,
replicando a arquitetura elegante do LlamaExtract mas usando tecnologias
open-source (Docling + Qwen 3 8B + Pydantic).

Exemplo de uso:
    from extract.config import ExtractConfig, ExtractMode, ChunkMode
    
    config = ExtractConfig(
        extraction_mode=ExtractMode.BALANCED,
        chunk_mode=ChunkMode.SECTION,
        system_prompt="Foco em artigos de lei",
    )
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ExtractMode(str, Enum):
    """
    Modo de extração - controla qualidade vs velocidade.
    
    Equivalente ao LlamaExtract:
    - FAST: Processamento rápido, documentos simples
    - BALANCED: Bom equilíbrio velocidade/precisão
    - MULTIMODAL: Para documentos com imagens/tabelas
    - PREMIUM: Máxima precisão com OCR avançado
    """
    FAST = "fast"
    BALANCED = "balanced"
    MULTIMODAL = "multimodal"
    PREMIUM = "premium"


class ChunkMode(str, Enum):
    """
    Estratégia de divisão do documento.
    
    - PAGE: Divide por páginas
    - SECTION: Divide por seções semânticas (capítulos, artigos)
    - ARTICLE: Divide por artigos (específico para leis)
    """
    PAGE = "page"
    SECTION = "section"
    ARTICLE = "article"


class ExtractTarget(str, Enum):
    """
    Escopo da extração.
    
    - PER_DOC: Aplica schema ao documento inteiro
    - PER_PAGE: Aplica schema a cada página
    - PER_CHUNK: Aplica schema a cada chunk
    """
    PER_DOC = "per_doc"
    PER_PAGE = "per_page"
    PER_CHUNK = "per_chunk"


class LLMProvider(str, Enum):
    """Provider do LLM."""
    OLLAMA = "ollama"
    VLLM = "vllm"


class LLMConfig(BaseModel):
    """Configuração do LLM para extração."""

    provider: LLMProvider = Field(
        default=LLMProvider.VLLM,
        description="Provider do LLM (ollama ou vllm)"
    )
    model: str = Field(
        default="Qwen/Qwen3-8B-AWQ",
        description="Nome do modelo"
    )
    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="URL base da API (vLLM: 8000, Ollama: 11434)"
    )
    api_key: str = Field(
        default="not-needed",
        description="API key (vLLM não requer)"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0, le=2.0,
        description="Temperatura do modelo (0 = determinístico)"
    )
    max_tokens: int = Field(
        default=16384,
        description="Máximo de tokens a gerar"
    )
    timeout: int = Field(
        default=300,
        description="Timeout em segundos"
    )

    @classmethod
    def for_vllm(cls, model: str = "Qwen/Qwen3-8B-AWQ") -> "LLMConfig":
        """Configuração para vLLM."""
        return cls(
            provider=LLMProvider.VLLM,
            model=model,
            base_url="http://localhost:8000/v1",
        )

    @classmethod
    def for_ollama(cls, model: str = "qwen3:8b") -> "LLMConfig":
        """Configuração para Ollama."""
        return cls(
            provider=LLMProvider.OLLAMA,
            model=model,
            base_url="http://localhost:11434/v1",
        )


class DoclingConfig(BaseModel):
    """Configuração do Docling para parsing."""
    
    ocr_enabled: bool = Field(
        default=True,
        description="Habilitar OCR para PDFs escaneados"
    )
    table_extraction: bool = Field(
        default=True,
        description="Extrair tabelas estruturadas"
    )
    image_extraction: bool = Field(
        default=False,
        description="Extrair imagens do documento"
    )


class ValidationConfig(BaseModel):
    """Configuração de validação."""
    
    validate_schema: bool = Field(
        default=True,
        description="Validar output contra schema Pydantic"
    )
    max_fix_attempts: int = Field(
        default=3,
        description="Máximo de tentativas de correção"
    )
    min_quality_score: float = Field(
        default=0.95,
        description="Score mínimo de qualidade (0-1)"
    )
    validate_completeness: bool = Field(
        default=True,
        description="Verificar se todos os elementos foram extraídos"
    )


class ExtractConfig(BaseModel):
    """
    Configuração principal de extração.
    
    Equivalente ao ExtractConfig do LlamaExtract, mas para stack open-source.
    
    Exemplo:
        config = ExtractConfig(
            extraction_mode=ExtractMode.BALANCED,
            chunk_mode=ChunkMode.SECTION,
            system_prompt="Extraia todos os artigos de lei",
        )
    """
    
    # === MODO DE EXTRAÇÃO ===
    extraction_mode: ExtractMode = Field(
        default=ExtractMode.BALANCED,
        description="Modo de extração (FAST, BALANCED, MULTIMODAL, PREMIUM)"
    )
    
    extraction_target: ExtractTarget = Field(
        default=ExtractTarget.PER_DOC,
        description="Escopo da extração"
    )
    
    chunk_mode: ChunkMode = Field(
        default=ChunkMode.SECTION,
        description="Estratégia de divisão do documento"
    )
    
    # === PROMPTS ===
    system_prompt: Optional[str] = Field(
        default=None,
        description="Instruções adicionais para o LLM"
    )
    
    # === FILTROS ===
    page_range: Optional[str] = Field(
        default=None,
        description="Páginas específicas para extrair (ex: '1-5,10-15')"
    )
    
    # === CONFIGURAÇÕES AVANÇADAS ===
    high_resolution_mode: bool = Field(
        default=False,
        description="OCR de alta resolução (mais lento)"
    )
    
    invalidate_cache: bool = Field(
        default=False,
        description="Ignorar cache e reprocessar"
    )
    
    # === EXTENSÕES (metadata adicional) ===
    cite_sources: bool = Field(
        default=False,
        description="Incluir citações das fontes"
    )
    
    use_reasoning: bool = Field(
        default=False,
        description="Incluir raciocínio do modelo"
    )
    
    confidence_scores: bool = Field(
        default=False,
        description="Incluir scores de confiança"
    )
    
    # === SUB-CONFIGURAÇÕES ===
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuração do LLM"
    )
    
    docling: DoclingConfig = Field(
        default_factory=DoclingConfig,
        description="Configuração do Docling"
    )
    
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Configuração de validação"
    )
    
    class Config:
        use_enum_values = True
    
    @classmethod
    def fast(cls) -> "ExtractConfig":
        """Preset para extração rápida."""
        return cls(
            extraction_mode=ExtractMode.FAST,
            llm=LLMConfig(
                temperature=0.0,
                max_tokens=8192,
            ),
            validation=ValidationConfig(
                max_fix_attempts=1,
                min_quality_score=0.8,
            ),
        )

    @classmethod
    def balanced(cls) -> "ExtractConfig":
        """Preset balanceado (padrão)."""
        return cls(
            extraction_mode=ExtractMode.BALANCED,
            llm=LLMConfig(
                temperature=0.0,
                max_tokens=16384,
            ),
            validation=ValidationConfig(
                max_fix_attempts=3,
                min_quality_score=0.95,
            ),
        )

    @classmethod
    def premium(cls) -> "ExtractConfig":
        """Preset para máxima qualidade."""
        return cls(
            extraction_mode=ExtractMode.PREMIUM,
            high_resolution_mode=True,
            cite_sources=True,
            use_reasoning=True,
            confidence_scores=True,
            llm=LLMConfig(
                temperature=0.0,
                max_tokens=16384,
            ),
            validation=ValidationConfig(
                max_fix_attempts=5,
                min_quality_score=0.99,
            ),
        )

    @classmethod
    def for_legal_documents(cls) -> "ExtractConfig":
        """Preset otimizado para documentos legais brasileiros."""
        return cls(
            extraction_mode=ExtractMode.BALANCED,
            chunk_mode=ChunkMode.ARTICLE,
            system_prompt=(
                "Voce e um especialista em extracao de documentos legais brasileiros. "
                "IMPORTANTE: Extraia ABSOLUTAMENTE TODOS os artigos do documento, do Art. 1 ao ultimo. "
                "NAO pule nenhum artigo, mesmo que pareca repetitivo ou similar. "
                "Cada artigo deve aparecer exatamente uma vez na estrutura. "
                "Mantenha a estrutura hierarquica exata: Capitulo > Artigo > Incisos > Alineas."
            ),
            llm=LLMConfig(
                provider=LLMProvider.VLLM,
                model="Qwen/Qwen3-8B-AWQ",
                base_url="http://localhost:8000/v1",
                max_tokens=8000,  # Ajustado para caber no contexto de 16k
            ),
            validation=ValidationConfig(
                validate_completeness=True,
                max_fix_attempts=5,
                min_quality_score=0.98,
            ),
        )


# =============================================================================
# EXEMPLOS DE USO
# =============================================================================

if __name__ == "__main__":
    # Configuração básica
    config = ExtractConfig(
        extraction_mode=ExtractMode.BALANCED,
        chunk_mode=ChunkMode.SECTION,
    )
    print("=== Configuração Básica ===")
    print(config.model_dump_json(indent=2))
    
    # Preset para documentos legais
    print("\n=== Preset Legal Documents ===")
    legal_config = ExtractConfig.for_legal_documents()
    print(f"Mode: {legal_config.extraction_mode}")
    print(f"Chunk: {legal_config.chunk_mode}")
    print(f"LLM: {legal_config.llm.model}")
    print(f"System Prompt: {legal_config.system_prompt[:50]}...")

