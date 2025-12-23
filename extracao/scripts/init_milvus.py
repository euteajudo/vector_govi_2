"""
Script para inicializar a collection leis_v2 no Milvus.

Uso:
    python scripts/init_milvus.py
    python scripts/init_milvus.py --drop  # Recria a collection
"""

import sys
import argparse
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pymilvus import connections, Collection, utility
from milvus.schema import create_legal_chunks_schema, create_indexes, COLLECTION_NAME


def init_collection(drop_existing: bool = False) -> Collection:
    """
    Inicializa a collection leis_v2.

    Args:
        drop_existing: Se True, dropa a collection existente

    Returns:
        Collection configurada
    """
    print("=" * 60)
    print("INICIALIZACAO DA COLLECTION MILVUS")
    print("=" * 60)

    # Conecta ao Milvus
    print("\n[1/4] Conectando ao Milvus...")
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("      Conectado!")

    # Verifica se collection existe
    exists = utility.has_collection(COLLECTION_NAME)
    print(f"\n[2/4] Collection '{COLLECTION_NAME}' existe: {exists}")

    if exists and drop_existing:
        print(f"      Dropando collection existente...")
        utility.drop_collection(COLLECTION_NAME)
        exists = False
        print("      Collection dropada!")

    if exists:
        print(f"      Usando collection existente.")
        collection = Collection(COLLECTION_NAME)
    else:
        # Cria schema
        print(f"\n[3/4] Criando collection '{COLLECTION_NAME}'...")
        schema = create_legal_chunks_schema()
        collection = Collection(
            name=COLLECTION_NAME,
            schema=schema,
            using="default"
        )
        print("      Collection criada!")

        # Cria indices
        print("\n[4/4] Criando indices...")
        create_indexes(collection)
        print("      Indices criados!")

    # Carrega collection
    print("\n[5/5] Carregando collection...")
    collection.load()
    print("      Collection carregada!")

    # Estatisticas
    print("\n" + "=" * 60)
    print("ESTATISTICAS")
    print("=" * 60)
    print(f"  Nome: {COLLECTION_NAME}")
    print(f"  Entidades: {collection.num_entities}")
    print(f"  Campos: {len(collection.schema.fields)}")

    print("\n  Campos principais:")
    for field in collection.schema.fields[:10]:
        dtype = field.dtype.name
        extra = ""
        if hasattr(field, 'dim') and field.dim:
            extra = f" ({field.dim}d)"
        elif hasattr(field, 'max_length') and field.max_length:
            extra = f" (max={field.max_length})"
        print(f"    - {field.name}: {dtype}{extra}")

    if len(collection.schema.fields) > 10:
        print(f"    ... e mais {len(collection.schema.fields) - 10} campos")

    print("\n  Indices:")
    for index in collection.indexes:
        print(f"    - {index.field_name}: {index.params.get('index_type', 'N/A')}")

    return collection


def main():
    parser = argparse.ArgumentParser(description="Inicializa collection Milvus")
    parser.add_argument("--drop", action="store_true", help="Dropa collection existente")
    args = parser.parse_args()

    try:
        collection = init_collection(drop_existing=args.drop)
        print("\n[OK] Collection pronta para uso!")
        return 0
    except Exception as e:
        print(f"\n[ERRO] {e}")
        return 1
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    sys.exit(main())
