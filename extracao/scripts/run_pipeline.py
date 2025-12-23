"""
Pipeline completo: JSON -> Chunks Enriquecidos -> Milvus

Uso:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --no-llm    # Sem enriquecimento
    python scripts/run_pipeline.py --no-milvus # Sem inserir no Milvus
"""

import json
import sys
import time
import argparse
import logging
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from models.legal_document import LegalDocument
from chunking.law_chunker import LawChunker, ChunkerConfig
from llm.vllm_client import VLLMClient, LLMConfig
from embeddings.bge_m3 import BGEM3Embedder, EmbeddingConfig
from milvus.schema import COLLECTION_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_document(json_path: Path) -> LegalDocument:
    """Carrega documento legal do JSON."""
    logger.info(f"Carregando documento: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Pode ser resultado direto ou wrapper com "data"
    if "data" in data:
        doc_data = data["data"]
    else:
        doc_data = data

    return LegalDocument.model_validate(doc_data)


def create_vllm_client() -> VLLMClient:
    """Cria cliente vLLM para enriquecimento."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-8B-AWQ",  # Modelo unico para extracao + enriquecimento
        temperature=0.0,
        max_tokens=1024,
    )
    return VLLMClient(config=config)


def create_embedder() -> BGEM3Embedder:
    """Cria embedder BGE-M3 para vetorizacao."""
    config = EmbeddingConfig(
        use_fp16=True,
        batch_size=16,
        max_length=8192,
    )
    return BGEM3Embedder(config=config)


def insert_into_milvus(chunks: list, collection_name: str = COLLECTION_NAME):
    """Insere chunks no Milvus."""
    from pymilvus import connections, Collection

    logger.info(f"Conectando ao Milvus...")
    connections.connect(host="localhost", port="19530")

    collection = Collection(collection_name)

    # Prepara dados para inserir
    # Milvus espera lista de valores por campo
    data = {
        "chunk_id": [],
        "parent_id": [],
        "chunk_level": [],
        "chunk_index": [],
        "text": [],
        "enriched_text": [],
        "dense_vector": [],
        "thesis_vector": [],
        "sparse_vector": [],
        "context_header": [],
        "thesis_text": [],
        "thesis_type": [],
        "synthetic_questions": [],
        "document_id": [],
        "tipo_documento": [],
        "numero": [],
        "ano": [],
        "chapter_number": [],
        "chapter_title": [],
        "article_number": [],
        "article_title": [],
        "has_items": [],
        "has_paragraphs": [],
        "item_count": [],
        "paragraph_count": [],
        "section": [],
        "section_type": [],
        "section_title": [],
        "token_count": [],
    }

    for chunk in chunks:
        data["chunk_id"].append(chunk.chunk_id)
        data["parent_id"].append(chunk.parent_id)
        data["chunk_level"].append(chunk.chunk_level.value)
        data["chunk_index"].append(chunk.chunk_index)
        data["text"].append(chunk.text[:65535])  # Limita tamanho
        data["enriched_text"].append((chunk.enriched_text or "")[:65535])

        # Vetores - usa placeholder se nao tiver
        if chunk.dense_vector:
            data["dense_vector"].append(chunk.dense_vector)
        else:
            data["dense_vector"].append([0.0] * 1024)

        if chunk.thesis_vector:
            data["thesis_vector"].append(chunk.thesis_vector)
        else:
            data["thesis_vector"].append([0.0] * 1024)

        if chunk.sparse_vector:
            data["sparse_vector"].append(chunk.sparse_vector)
        else:
            data["sparse_vector"].append({0: 0.0})  # Sparse vazio

        # Enriquecimento
        data["context_header"].append((chunk.context_header or "")[:2000])
        data["thesis_text"].append((chunk.thesis_text or "")[:5000])
        data["thesis_type"].append(chunk.thesis_type or "disposicao")
        data["synthetic_questions"].append((chunk.synthetic_questions or "")[:10000])

        # Metadados
        data["document_id"].append(chunk.document_id or "")
        data["tipo_documento"].append(chunk.tipo_documento or "")
        data["numero"].append(chunk.numero or "")
        data["ano"].append(chunk.ano or 0)
        data["chapter_number"].append(chunk.chapter_number or "")
        data["chapter_title"].append(chunk.chapter_title or "")
        data["article_number"].append(chunk.article_number or "")
        data["article_title"].append(chunk.article_title or "")

        # Flags
        data["has_items"].append(chunk.has_items or False)
        data["has_paragraphs"].append(chunk.has_paragraphs or False)
        data["item_count"].append(chunk.item_count or 0)
        data["paragraph_count"].append(chunk.paragraph_count or 0)

        # Legado
        data["section"].append("")
        data["section_type"].append("")
        data["section_title"].append("")

        data["token_count"].append(chunk.token_count or 0)

    logger.info(f"Inserindo {len(chunks)} chunks no Milvus...")

    # Converte para formato de lista de listas
    insert_data = [data[field] for field in data.keys()]

    result = collection.insert(insert_data)
    collection.flush()

    logger.info(f"Inseridos {result.insert_count} chunks")
    logger.info(f"Total na collection: {collection.num_entities}")

    connections.disconnect("default")

    return result


def run_pipeline(
    json_path: Path,
    use_llm: bool = True,
    use_embeddings: bool = True,
    use_milvus: bool = True,
    batch_size: int = 1,
):
    """
    Executa pipeline completo.

    Args:
        json_path: Caminho para JSON do documento
        use_llm: Se deve enriquecer com LLM
        use_embeddings: Se deve gerar embeddings BGE-M3
        use_milvus: Se deve inserir no Milvus
        batch_size: Tamanho do batch para LLM
    """
    print("=" * 60)
    print("PIPELINE DE PROCESSAMENTO")
    print("=" * 60)

    total_start = time.time()

    # 1. Carrega documento
    print("\n[1/5] Carregando documento...")
    doc = load_document(json_path)
    print(f"      Documento: {doc.document_type} {doc.number}")
    print(f"      Capitulos: {len(doc.chapters)}")
    print(f"      Artigos: {sum(len(c.articles) for c in doc.chapters)}")

    # 2. Configura chunker
    print("\n[2/5] Configurando chunker...")

    llm_client = None
    if use_llm:
        try:
            llm_client = create_vllm_client()
            models = llm_client.list_models()
            if models:
                print(f"      vLLM conectado: {models[0]}")
            else:
                print("      AVISO: vLLM nao respondeu, continuando sem LLM")
                llm_client = None
        except Exception as e:
            print(f"      AVISO: Erro ao conectar vLLM: {e}")
            llm_client = None

    embedder = None
    if use_embeddings:
        try:
            print("      Inicializando BGE-M3...")
            embedder = create_embedder()
            print("      BGE-M3 pronto!")
        except Exception as e:
            print(f"      AVISO: Erro ao criar embedder: {e}")
            embedder = None

    config = ChunkerConfig(
        enrich_with_llm=llm_client is not None,
        generate_embeddings=embedder is not None,
        batch_size=batch_size,
    )

    chunker = LawChunker(
        llm_client=llm_client,
        embedding_model=embedder,
        config=config,
    )

    print(f"      Enriquecimento LLM: {'Sim' if config.enrich_with_llm else 'Nao'}")
    print(f"      Embeddings BGE-M3: {'Sim' if config.generate_embeddings else 'Nao'}")

    # 3. Executa chunking + enriquecimento
    print("\n[3/5] Executando chunking + enriquecimento...")
    chunk_start = time.time()

    result = chunker.chunk_document(doc)

    chunk_time = time.time() - chunk_start
    print(f"      Chunks gerados: {result.total_chunks}")
    print(f"      Tokens totais: {result.total_tokens}")
    print(f"      Tempo: {chunk_time:.2f}s")

    if result.errors:
        print(f"      ERROS: {result.errors}")

    # Mostra preview
    print("\n      Preview dos chunks:")
    for chunk in result.chunks[:3]:
        print(f"        - {chunk.chunk_id}")
        if chunk.context_header:
            print(f"          Header: {chunk.context_header[:80]}...")
        if chunk.thesis_type:
            print(f"          Tipo: {chunk.thesis_type}")
        if chunk.dense_vector:
            print(f"          Dense: {len(chunk.dense_vector)}d")
        if chunk.sparse_vector:
            print(f"          Sparse: {len(chunk.sparse_vector)} tokens")

    if len(result.chunks) > 3:
        print(f"        ... e mais {len(result.chunks) - 3} chunks")

    # 4. Verifica embeddings
    has_embeddings = result.chunks and result.chunks[0].dense_vector is not None
    print(f"\n[4/5] Embeddings: {'Gerados' if has_embeddings else 'Nao gerados'}")

    # 5. Insere no Milvus
    if use_milvus and result.chunks:
        print("\n[5/5] Inserindo no Milvus...")
        try:
            insert_result = insert_into_milvus(result.chunks)
            print(f"      Sucesso!")
        except Exception as e:
            print(f"      ERRO: {e}")
    else:
        print("\n[5/5] Pulando insercao no Milvus")

    # Resumo
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"  Documento: {doc.document_type} {doc.number}/{doc.date[:4]}")
    print(f"  Chunks: {result.total_chunks}")
    print(f"  Tokens: {result.total_tokens}")
    print(f"  Tempo total: {total_time:.2f}s")
    if result.total_chunks > 0:
        print(f"  Tempo/chunk: {total_time / result.total_chunks:.2f}s")

    # Salva resultado
    output_path = json_path.parent / f"{json_path.stem}_chunks.json"

    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "parent_id": c.parent_id,
            "chunk_level": c.chunk_level.name,
            "text": c.text[:500] + "..." if len(c.text) > 500 else c.text,
            "context_header": c.context_header,
            "thesis_text": c.thesis_text,
            "thesis_type": c.thesis_type,
            "synthetic_questions": c.synthetic_questions,
            "token_count": c.token_count,
        }
        for c in result.chunks
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "document_id": result.document_id,
            "total_chunks": result.total_chunks,
            "total_tokens": result.total_tokens,
            "processing_time": result.processing_time_seconds,
            "chunks": chunks_data,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  Resultado salvo em: {output_path}")

    # Fecha cliente
    if llm_client:
        llm_client.close()

    return result


def main():
    parser = argparse.ArgumentParser(description="Pipeline de processamento de documentos legais")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "output" / "resultado_extracao_vllm_v3.json"),
        help="Caminho para JSON do documento"
    )
    parser.add_argument("--no-llm", action="store_true", help="Nao usar LLM para enriquecimento")
    parser.add_argument("--no-embeddings", action="store_true", help="Nao gerar embeddings BGE-M3")
    parser.add_argument("--no-milvus", action="store_true", help="Nao inserir no Milvus")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size para LLM")

    args = parser.parse_args()

    json_path = Path(args.input)
    if not json_path.exists():
        print(f"ERRO: Arquivo nao encontrado: {json_path}")
        return 1

    try:
        run_pipeline(
            json_path=json_path,
            use_llm=not args.no_llm,
            use_embeddings=not args.no_embeddings,
            use_milvus=not args.no_milvus,
            batch_size=args.batch_size,
        )
        return 0
    except Exception as e:
        logger.exception(f"Erro no pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
