"""
Script de migração: leis_v2 -> leis_v3

Este script:
1. Conecta ao Milvus
2. Dropa a collection leis_v2 (se existir)
3. Cria a nova collection leis_v3 com campos parent-child
4. Cria os índices necessários

Uso:
    python scripts/migrate_to_v3.py

ATENÇÃO: Este script APAGA todos os dados da collection leis_v2!
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymilvus import connections, utility, Collection
from milvus.schema_v3 import (
    create_legal_chunks_schema_v3,
    create_indexes_v3,
    COLLECTION_NAME_V3,
)
from milvus.schema import COLLECTION_NAME as OLD_COLLECTION


def migrate_to_v3(
    drop_old: bool = True,
    host: str = "localhost",
    port: str = "19530",
):
    """
    Migra para collection v3.

    Args:
        drop_old: Se deve dropar a collection antiga
        host: Host do Milvus
        port: Porta do Milvus
    """
    print("=" * 60)
    print("MIGRACAO PARA SCHEMA V3")
    print("=" * 60)

    # 1. Conecta ao Milvus
    print("\n[1/5] Conectando ao Milvus...")
    try:
        connections.connect(host=host, port=port)
        print(f"      Conectado a {host}:{port}")
    except Exception as e:
        print(f"      ERRO: Nao foi possivel conectar: {e}")
        return False

    # 2. Lista collections existentes
    print("\n[2/5] Verificando collections existentes...")
    existing = utility.list_collections()
    print(f"      Collections: {existing}")

    # 3. Dropa collection antiga (se existir e solicitado)
    if drop_old:
        print(f"\n[3/5] Dropando collection antiga ({OLD_COLLECTION})...")
        if OLD_COLLECTION in existing:
            old_collection = Collection(OLD_COLLECTION)
            count = old_collection.num_entities
            print(f"      Entidades na collection antiga: {count}")

            utility.drop_collection(OLD_COLLECTION)
            print(f"      Collection {OLD_COLLECTION} dropada!")
        else:
            print(f"      Collection {OLD_COLLECTION} nao existe, pulando...")

        # Dropa v3 se existir (para recriar limpa)
        if COLLECTION_NAME_V3 in existing:
            print(f"      Dropando collection existente {COLLECTION_NAME_V3}...")
            utility.drop_collection(COLLECTION_NAME_V3)
    else:
        print("\n[3/5] Pulando drop de collection antiga (drop_old=False)")

    # 4. Cria collection v3
    print(f"\n[4/5] Criando collection {COLLECTION_NAME_V3}...")
    schema = create_legal_chunks_schema_v3()

    collection = Collection(
        name=COLLECTION_NAME_V3,
        schema=schema,
        using="default",
    )
    print(f"      Collection {COLLECTION_NAME_V3} criada!")

    # Lista campos
    print("\n      Campos do schema:")
    for field in schema.fields:
        tipo = field.dtype.name
        if hasattr(field, 'dim') and field.dim:
            tipo += f" ({field.dim}d)"
        print(f"        - {field.name}: {tipo}")

    # 5. Cria indices
    print(f"\n[5/5] Criando indices...")
    create_indexes_v3(collection)
    print("      Indices criados!")

    # Lista indices
    print("\n      Indices criados:")
    for index in collection.indexes:
        print(f"        - {index.field_name}: {index.params}")

    # Carrega collection
    print("\n      Carregando collection na memoria...")
    collection.load()
    print("      Collection carregada!")

    # Resumo
    print("\n" + "=" * 60)
    print("MIGRACAO CONCLUIDA")
    print("=" * 60)
    print(f"  Collection antiga: {OLD_COLLECTION} (dropada)")
    print(f"  Collection nova: {COLLECTION_NAME_V3}")
    print(f"  Campos: {len(schema.fields)}")
    print()
    print("Novos campos parent-child:")
    print("  - parent_chunk_id: ID do chunk pai")
    print("  - span_id: ART-005, PAR-005-1, INC-005-I")
    print("  - device_type: article, paragraph, inciso, alinea")
    print("  - citations: JSON lista de span_ids")
    print()
    print("Novos campos proveniencia:")
    print("  - schema_version, extractor_version")
    print("  - ingestion_timestamp, document_hash")

    # Desconecta
    connections.disconnect("default")

    return True


def check_milvus_status(host: str = "localhost", port: str = "19530"):
    """Verifica status do Milvus."""
    print("\n" + "=" * 60)
    print("STATUS DO MILVUS")
    print("=" * 60)

    try:
        connections.connect(host=host, port=port)
        print(f"  Conectado: {host}:{port}")

        collections = utility.list_collections()
        print(f"  Collections: {collections}")

        for name in collections:
            col = Collection(name)
            print(f"\n  {name}:")
            print(f"    Entidades: {col.num_entities}")
            print(f"    Campos: {len(col.schema.fields)}")

        connections.disconnect("default")
        return True

    except Exception as e:
        print(f"  ERRO: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migra collection Milvus para v3")
    parser.add_argument("--host", default="localhost", help="Host do Milvus")
    parser.add_argument("--port", default="19530", help="Porta do Milvus")
    parser.add_argument("--status", action="store_true", help="Apenas mostra status")
    parser.add_argument("--keep-old", action="store_true", help="Nao dropa collection antiga")

    args = parser.parse_args()

    if args.status:
        check_milvus_status(args.host, args.port)
    else:
        print("\n" + "!" * 60)
        print("ATENCAO: Este script vai APAGAR a collection leis_v2!")
        print("!" * 60)
        print()

        confirm = input("Deseja continuar? (digite 'sim' para confirmar): ")
        if confirm.lower() == "sim":
            migrate_to_v3(
                drop_old=not args.keep_old,
                host=args.host,
                port=args.port,
            )
        else:
            print("\nMigracao cancelada.")
