"""
Schema Milvus v3 para chunks de documentos legais.

Collection: leis_v3

Novos campos para parent-child retrieval:
- parent_chunk_id: ID do chunk pai (vazio para artigos)
- span_id: ID do span (ART-005, PAR-005-1, INC-005-I)
- device_type: Tipo do dispositivo (article, paragraph, inciso, alinea)
- citations: Lista de span_ids que compõem o chunk

Metadados de proveniência:
- schema_version: Versão do schema
- extractor_version: Versão do extrator
- ingestion_timestamp: Timestamp de ingestão
- document_hash: Hash SHA-256 do PDF

Uso:
    from pymilvus import connections, Collection
    from milvus.schema_v3 import create_legal_chunks_schema_v3, COLLECTION_NAME_V3

    connections.connect(host="localhost", port="19530")
    schema = create_legal_chunks_schema_v3()
    collection = Collection(COLLECTION_NAME_V3, schema)
"""

from pymilvus import FieldSchema, CollectionSchema, DataType

# Nome da collection
COLLECTION_NAME_V3 = "leis_v3"


def create_legal_chunks_schema_v3() -> CollectionSchema:
    """
    Cria schema v3 para collection de chunks legais.

    Principais mudanças da v2:
    - parent_chunk_id: Para parent-child retrieval
    - span_id: ID único do span (ART-005, PAR-005-1)
    - device_type: article/paragraph/inciso/alinea
    - citations: JSON com lista de span_ids
    - Campos de proveniência

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

        # === IDs Parent-Child ===
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=200,
            description="ID completo: IN-65-2021#ART-005"
        ),
        FieldSchema(
            name="parent_chunk_id",
            dtype=DataType.VARCHAR,
            max_length=200,
            description="ID do chunk pai (vazio para artigos)"
        ),
        FieldSchema(
            name="span_id",
            dtype=DataType.VARCHAR,
            max_length=100,
            description="ID do span: ART-005, PAR-005-1, INC-005-I"
        ),
        FieldSchema(
            name="device_type",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Tipo: article, paragraph, inciso, alinea"
        ),
        FieldSchema(
            name="chunk_level",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Nivel: article, device, sub_device"
        ),

        # === Conteudo ===
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            enable_match=True,
            description="Texto original - usado para reranking"
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
            description="Embedding denso BGE-M3 do enriched_text"
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

        # === Citations (lista de span_ids) ===
        FieldSchema(
            name="citations",
            dtype=DataType.VARCHAR,
            max_length=5000,
            description="JSON: lista de span_ids que compõem o chunk"
        ),

        # === Metadados do Documento ===
        FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=200,
            description="ID unico do documento"
        ),
        FieldSchema(
            name="tipo_documento",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="LEI, DECRETO, IN"
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
            name="article_number",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Numero do artigo (1, 2, 5)"
        ),

        # === Proveniência ===
        FieldSchema(
            name="schema_version",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Versao do schema (1.0.0)"
        ),
        FieldSchema(
            name="extractor_version",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Versao do extrator (1.0.0)"
        ),
        FieldSchema(
            name="ingestion_timestamp",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Timestamp ISO de ingestao"
        ),
        FieldSchema(
            name="document_hash",
            dtype=DataType.VARCHAR,
            max_length=128,
            description="SHA-256 do PDF original"
        ),

    ]

    schema = CollectionSchema(
        fields=fields,
        description="Chunks legais v3: parent-child + citations + proveniencia",
        enable_dynamic_field=True
    )

    return schema


def create_indexes_v3(collection) -> None:
    """
    Cria indices para busca hibrida.

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

    # Indice para sparse_vector
    sparse_index = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
        "params": {
            "drop_ratio_build": 0.2
        }
    }
    collection.create_index("sparse_vector", sparse_index)

    # Indices escalares para filtros
    collection.create_index("tipo_documento", {"index_type": "INVERTED"})
    collection.create_index("ano", {"index_type": "INVERTED"})
    collection.create_index("article_number", {"index_type": "INVERTED"})
    collection.create_index("device_type", {"index_type": "INVERTED"})
    collection.create_index("parent_chunk_id", {"index_type": "INVERTED"})


if __name__ == "__main__":
    print("Schema v3 para collection:", COLLECTION_NAME_V3)
    print()

    schema = create_legal_chunks_schema_v3()

    print("Campos:")
    for field in schema.fields:
        print(f"  - {field.name}: {field.dtype.name}", end="")
        if hasattr(field, 'dim') and field.dim:
            print(f" ({field.dim}d)", end="")
        if hasattr(field, 'max_length') and field.max_length:
            print(f" (max={field.max_length})", end="")
        print()

    print()
    print("Novos campos (parent-child):")
    print("  - parent_chunk_id: ID do pai ('' para artigos)")
    print("  - span_id: ART-005, PAR-005-1, INC-005-I")
    print("  - device_type: article, paragraph, inciso, alinea")
    print("  - citations: JSON [span_id, ...]")
    print()
    print("Novos campos (proveniencia):")
    print("  - schema_version, extractor_version")
    print("  - ingestion_timestamp, document_hash")
