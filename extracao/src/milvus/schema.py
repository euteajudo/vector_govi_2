"""
Schema Milvus para chunks de documentos legais.

Collection: leis_v2

Busca hibrida com BGE-M3:
- dense_vector: embedding denso 1024d (busca semantica)
- thesis_vector: embedding denso da tese 1024d
- sparse_vector: learned sparse BGE-M3 (superior ao BM25)

O sparse do BGE-M3 e um modelo aprendido que:
1. Foi treinado junto com dense - se complementam
2. Aprende pesos semanticos, nao apenas frequencia
3. Captura sinonimos: "requisitante" ~ "demandante" ~ "solicitante"
4. Especialmente importante para portugues juridico

Uso:
    from pymilvus import connections, Collection
    from milvus.schema import create_legal_chunks_schema, COLLECTION_NAME

    connections.connect(host="localhost", port="19530")
    schema = create_legal_chunks_schema()
    collection = Collection(COLLECTION_NAME, schema)
"""

from pymilvus import FieldSchema, CollectionSchema, DataType

# Nome da collection
COLLECTION_NAME = "leis_v2"


def create_legal_chunks_schema() -> CollectionSchema:
    """
    Cria schema para collection de chunks legais.

    Campos:
    - IDs hierarquicos (chunk_id, parent_id)
    - Conteudo (text, enriched_text)
    - Vetores BGE-M3 (dense_vector, thesis_vector, sparse_vector)
    - Enriquecimento (context_header, thesis_text, synthetic_questions)
    - Metadados do documento
    - Hierarquia legal

    Returns:
        CollectionSchema configurado
    """
    fields = [
        # === ID (auto-gerado) ===
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="ID auto-gerado"
        ),

        # === IDs Hierarquicos ===
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=200,
            description="ID hierarquico: IN-SEGES-58-2022#CAP-I#ART-3"
        ),
        FieldSchema(
            name="parent_id",
            dtype=DataType.VARCHAR,
            max_length=200,
            description="ID do chunk pai"
        ),
        FieldSchema(
            name="chunk_level",
            dtype=DataType.INT64,
            description="Nivel: 0=doc, 1=cap, 2=art, 3=dispositivo"
        ),
        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT64,
            description="Indice sequencial no documento"
        ),

        # === Conteudo ===
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            enable_match=True,
            description="Texto original - usado para ColBERT reranking"
        ),
        FieldSchema(
            name="enriched_text",
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="Contexto + texto + perguntas"
        ),

        # === Vetores BGE-M3 (1024d) ===
        FieldSchema(
            name="dense_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024,
            description="Embedding denso BGE-M3 do texto"
        ),
        FieldSchema(
            name="thesis_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024,
            description="Embedding denso BGE-M3 da tese"
        ),
        FieldSchema(
            name="sparse_vector",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
            description="Learned sparse BGE-M3 {token_id: weight}"
        ),

        # === Enriquecimento ===
        FieldSchema(
            name="context_header",
            dtype=DataType.VARCHAR,
            max_length=2000,
            description="Frase contextualizando o dispositivo"
        ),
        FieldSchema(
            name="thesis_text",
            dtype=DataType.VARCHAR,
            max_length=5000,
            description="Resumo objetivo do dispositivo"
        ),
        FieldSchema(
            name="thesis_type",
            dtype=DataType.VARCHAR,
            max_length=100,
            description="Tipo: definicao, procedimento, prazo, etc"
        ),
        FieldSchema(
            name="synthetic_questions",
            dtype=DataType.VARCHAR,
            max_length=10000,
            description="Perguntas que o chunk responde"
        ),

        # === Metadados do Documento ===
        FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=500,
            description="ID unico do documento"
        ),
        FieldSchema(
            name="tipo_documento",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="LEI, DECRETO, INSTRUCAO NORMATIVA"
        ),
        FieldSchema(
            name="numero",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Numero do documento"
        ),
        FieldSchema(
            name="ano",
            dtype=DataType.INT64,
            description="Ano do documento"
        ),

        # === Hierarquia Legal ===
        FieldSchema(
            name="chapter_number",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Numero do capitulo (I, II, III)"
        ),
        FieldSchema(
            name="chapter_title",
            dtype=DataType.VARCHAR,
            max_length=500,
            description="Titulo do capitulo"
        ),
        FieldSchema(
            name="article_number",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Numero do artigo"
        ),
        FieldSchema(
            name="article_title",
            dtype=DataType.VARCHAR,
            max_length=500,
            description="Titulo do artigo"
        ),

        # === Flags Estruturais ===
        FieldSchema(
            name="has_items",
            dtype=DataType.BOOL,
            description="Se tem incisos"
        ),
        FieldSchema(
            name="has_paragraphs",
            dtype=DataType.BOOL,
            description="Se tem paragrafos"
        ),
        FieldSchema(
            name="item_count",
            dtype=DataType.INT64,
            description="Quantidade de incisos"
        ),
        FieldSchema(
            name="paragraph_count",
            dtype=DataType.INT64,
            description="Quantidade de paragrafos"
        ),

        # === Legado (compatibilidade) ===
        FieldSchema(
            name="section",
            dtype=DataType.VARCHAR,
            max_length=500,
            description="Secao (legado)"
        ),
        FieldSchema(
            name="section_type",
            dtype=DataType.VARCHAR,
            max_length=100,
            description="Tipo de secao (legado)"
        ),
        FieldSchema(
            name="section_title",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Titulo da secao (legado)"
        ),

        # === Rastreabilidade ===
        FieldSchema(
            name="token_count",
            dtype=DataType.INT64,
            description="Quantidade de tokens"
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Chunks legais: BGE-M3 (1024d dense + sparse) + ColBERT rerank",
        enable_dynamic_field=True
    )

    return schema


def create_indexes(collection) -> None:
    """
    Cria indices para busca hibrida.

    Indices:
    - dense_vector: HNSW para busca semantica rapida
    - thesis_vector: HNSW para busca na tese
    - sparse_vector: SPARSE_INVERTED_INDEX para learned sparse

    Args:
        collection: Collection do Milvus
    """
    # Indice HNSW para dense_vector
    dense_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 256
        }
    }
    collection.create_index("dense_vector", dense_index)

    # Indice HNSW para thesis_vector
    thesis_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 256
        }
    }
    collection.create_index("thesis_vector", thesis_index)

    # Indice para sparse_vector (learned sparse BGE-M3)
    sparse_index = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",  # Inner Product para sparse
        "params": {
            "drop_ratio_build": 0.2  # Remove 20% dos tokens menos importantes
        }
    }
    collection.create_index("sparse_vector", sparse_index)

    # Indices escalares para filtros
    collection.create_index("tipo_documento", {"index_type": "INVERTED"})
    collection.create_index("ano", {"index_type": "INVERTED"})
    collection.create_index("article_number", {"index_type": "INVERTED"})


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("Schema para collection:", COLLECTION_NAME)
    print()

    schema = create_legal_chunks_schema()

    print("Campos:")
    for field in schema.fields:
        print(f"  - {field.name}: {field.dtype.name}", end="")
        if hasattr(field, 'dim') and field.dim:
            print(f" ({field.dim}d)", end="")
        if hasattr(field, 'max_length') and field.max_length:
            print(f" (max={field.max_length})", end="")
        print()

    print()
    print("Vetores:")
    print("  - dense_vector: BGE-M3 dense 1024d")
    print("  - thesis_vector: BGE-M3 dense 1024d")
    print("  - sparse_vector: BGE-M3 learned sparse")
