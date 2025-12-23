"""
Modelos de dados para chunks de documentos legais.

Define a estrutura LegalChunk que representa um chunk pronto para
insercao no Milvus com todos os campos necessarios para busca hibrida:
- dense_vector: embedding denso BGE-M3 (1024d)
- thesis_vector: embedding da tese/resumo (1024d)
- sparse_vector: learned sparse BGE-M3 (superior ao BM25)

O sparse do BGE-M3 e um modelo aprendido que captura sinonimos
e variacoes linguisticas do portugues juridico.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class ChunkLevel(Enum):
    """Nível hierárquico do chunk."""
    DOCUMENT = 0    # Documento inteiro (raro, para contexto)
    CHAPTER = 1     # Capítulo inteiro (para contexto amplo)
    ARTICLE = 2     # Artigo completo (principal, retrieval padrão)
    DEVICE = 3      # Dispositivo isolado (§, inciso com alíneas)


class ThesisType(Enum):
    """Tipo de tese/conteúdo do dispositivo legal."""
    DEFINICAO = "definicao"         # Define conceitos
    PROCEDIMENTO = "procedimento"   # Estabelece procedimentos
    PRAZO = "prazo"                 # Define prazos
    REQUISITO = "requisito"         # Estabelece requisitos
    COMPETENCIA = "competencia"     # Define competências
    VEDACAO = "vedacao"             # Proíbe algo
    EXCECAO = "excecao"             # Estabelece exceções
    SANCAO = "sancao"               # Define sanções/penalidades
    DISPOSICAO = "disposicao"       # Disposição geral


@dataclass
class LegalChunk:
    """
    Chunk de documento legal pronto para Milvus.

    Contem todos os campos necessarios para busca hibrida com BGE-M3:
    - dense_vector: embedding denso do texto (1024d)
    - thesis_vector: embedding denso da tese/resumo (1024d)
    - sparse_vector: learned sparse BGE-M3 {token_id: weight}

    O sparse do BGE-M3 e superior ao BM25 porque:
    1. Treinado junto com dense - se complementam
    2. Aprende pesos semanticos, nao apenas frequencia
    3. Captura sinonimos: "requisitante" ~ "demandante" ~ "solicitante"

    Attributes:
        chunk_id: ID hierarquico unico (ex: "IN-SEGES-58-2022#CAP-I#ART-3")
        parent_id: ID do chunk pai (ex: "IN-SEGES-58-2022#CAP-I")
        chunk_index: Indice sequencial global no documento
        chunk_level: Nivel hierarquico (CHAPTER, ARTICLE, DEVICE)

        text: Texto original do dispositivo
        enriched_text: Contexto + texto + perguntas (para LLM)

        context_header: Frase contextualizando o dispositivo
        thesis_text: Resumo objetivo do que determina/define
        thesis_type: Classificacao do tipo de conteudo
        synthetic_questions: Perguntas que o chunk responde (separadas por \\n)

        document_id: ID unico do documento
        tipo_documento: LEI, DECRETO, INSTRUCAO NORMATIVA, etc
        numero: Numero do documento
        ano: Ano do documento
        chapter_number: Numero do capitulo (I, II, III)
        chapter_title: Titulo do capitulo
        article_number: Numero do artigo
        article_title: Titulo do artigo (se houver)

        has_items: Se o artigo tem incisos
        has_paragraphs: Se o artigo tem paragrafos
        item_count: Quantidade de incisos
        paragraph_count: Quantidade de paragrafos

        token_count: Contagem de tokens do texto
        char_start: Posicao inicial no documento original
        char_end: Posicao final no documento original

        dense_vector: Embedding denso BGE-M3 (1024d)
        thesis_vector: Embedding denso da tese (1024d)
        sparse_vector: Learned sparse BGE-M3 {token_id: weight}
    """

    # === IDs Hierárquicos ===
    chunk_id: str
    parent_id: str
    chunk_index: int
    chunk_level: ChunkLevel

    # === Conteúdo ===
    text: str
    enriched_text: str = ""

    # === Enriquecimento (gerado por LLM) ===
    context_header: str = ""
    thesis_text: str = ""
    thesis_type: str = "disposicao"
    synthetic_questions: str = ""

    # === Hierarquia Legal ===
    document_id: str = ""
    tipo_documento: str = ""
    numero: str = ""
    ano: int = 0
    chapter_number: str = ""
    chapter_title: str = ""
    article_number: str = ""
    article_title: str = ""

    # === Flags Estruturais ===
    has_items: bool = False
    has_paragraphs: bool = False
    item_count: int = 0
    paragraph_count: int = 0

    # === Metadados ===
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0

    # === Embeddings BGE-M3 (preenchidos pelo pipeline) ===
    dense_vector: Optional[list[float]] = None       # 1024d
    thesis_vector: Optional[list[float]] = None      # 1024d
    sparse_vector: Optional[dict[int, float]] = None # {token_id: weight}

    def to_dict(self) -> dict:
        """
        Converte para dicionário compatível com Milvus.

        Nota: O campo 'id' é auto-gerado pelo Milvus (autoID=true).
        """
        return {
            # Campos de texto
            "text": self.text,
            "enriched_text": self.enriched_text,
            "context_header": self.context_header,
            "thesis_text": self.thesis_text,
            "thesis_type": self.thesis_type,
            "synthetic_questions": self.synthetic_questions,

            # Hierarquia
            "document_id": self.document_id,
            "tipo_documento": self.tipo_documento,
            "numero": self.numero,
            "ano": self.ano,
            "section": f"Capítulo {self.chapter_number}" if self.chapter_number else "",
            "section_type": "capitulo",
            "section_title": self.chapter_title,
            "chunk_index": self.chunk_index,

            # Embeddings BGE-M3
            "dense_vector": self.dense_vector or [],
            "thesis_vector": self.thesis_vector or [],
            "sparse_vector": self.sparse_vector or {},

            # Campos adicionais (dynamic fields no Milvus)
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "chunk_level": self.chunk_level.value,
            "article_number": self.article_number,
            "article_title": self.article_title or "",
            "has_items": self.has_items,
            "has_paragraphs": self.has_paragraphs,
            "item_count": self.item_count,
            "paragraph_count": self.paragraph_count,
            "token_count": self.token_count,
        }

    def to_json(self) -> str:
        """Serializa para JSON."""
        d = self.to_dict()
        # Remove embeddings do JSON (muito grande)
        d.pop("dense_vector", None)
        d.pop("thesis_vector", None)
        d.pop("sparse_vector", None)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "LegalChunk":
        """Cria LegalChunk a partir de dicionário."""
        return cls(
            chunk_id=data.get("chunk_id", ""),
            parent_id=data.get("parent_id", ""),
            chunk_index=data.get("chunk_index", 0),
            chunk_level=ChunkLevel(data.get("chunk_level", 2)),
            text=data.get("text", ""),
            enriched_text=data.get("enriched_text", ""),
            context_header=data.get("context_header", ""),
            thesis_text=data.get("thesis_text", ""),
            thesis_type=data.get("thesis_type", "disposicao"),
            synthetic_questions=data.get("synthetic_questions", ""),
            document_id=data.get("document_id", ""),
            tipo_documento=data.get("tipo_documento", ""),
            numero=data.get("numero", ""),
            ano=data.get("ano", 0),
            chapter_number=data.get("chapter_number", ""),
            chapter_title=data.get("chapter_title", ""),
            article_number=data.get("article_number", ""),
            article_title=data.get("article_title", ""),
            has_items=data.get("has_items", False),
            has_paragraphs=data.get("has_paragraphs", False),
            item_count=data.get("item_count", 0),
            paragraph_count=data.get("paragraph_count", 0),
            token_count=data.get("token_count", 0),
            char_start=data.get("char_start", 0),
            char_end=data.get("char_end", 0),
            dense_vector=data.get("dense_vector"),
            thesis_vector=data.get("thesis_vector"),
            sparse_vector=data.get("sparse_vector"),
        )

    def __repr__(self) -> str:
        return (
            f"LegalChunk(id={self.chunk_id!r}, "
            f"type={self.thesis_type}, "
            f"items={self.item_count}, "
            f"paragraphs={self.paragraph_count}, "
            f"tokens={self.token_count})"
        )


@dataclass
class ChunkingResult:
    """Resultado do processo de chunking de um documento."""

    chunks: list[LegalChunk] = field(default_factory=list)
    document_id: str = ""
    total_chunks: int = 0
    total_tokens: int = 0
    processing_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """Retorna resumo do resultado."""
        level_counts = {}
        type_counts = {}

        for chunk in self.chunks:
            level = chunk.chunk_level.name
            level_counts[level] = level_counts.get(level, 0) + 1

            t = chunk.thesis_type
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "document_id": self.document_id,
            "total_chunks": len(self.chunks),
            "total_tokens": sum(c.token_count for c in self.chunks),
            "chunks_by_level": level_counts,
            "chunks_by_type": type_counts,
            "processing_time": f"{self.processing_time_seconds:.2f}s",
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }
