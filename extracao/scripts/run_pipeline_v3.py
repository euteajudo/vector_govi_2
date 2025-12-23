"""
Pipeline v3: Span-Based Extraction -> Parent-Child Chunks -> Milvus leis_v3

Processa documentos legais usando:
1. SpanParser: Parseia markdown para spans hierárquicos
2. ArticleOrchestrator: Extrai hierarquia por artigo via LLM
3. ChunkMaterializer: Gera chunks parent-child
4. BGE-M3: Gera embeddings (dense + sparse)
5. Milvus: Insere na collection leis_v3

Uso:
    python scripts/run_pipeline_v3.py
    python scripts/run_pipeline_v3.py --input data/output/in_65_markdown.md
    python scripts/run_pipeline_v3.py --no-llm  # Sem extração LLM (apenas parser)
"""

import json
import sys
import time
import argparse
import logging
import hashlib
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsing import SpanParser, ArticleOrchestrator, OrchestratorConfig
from chunking import ChunkMaterializer, ChunkMetadata, DeviceType
from llm.vllm_client import VLLMClient, LLMConfig
from embeddings.bge_m3 import BGEM3Embedder, EmbeddingConfig
from milvus.schema_v3 import COLLECTION_NAME_V3
from dashboard import MetricsCollector, generate_dashboard_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_vllm_client() -> VLLMClient:
    """Cria cliente vLLM."""
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-8B-AWQ",
        temperature=0.0,
        max_tokens=512,
    )
    return VLLMClient(config=config)


def create_embedder() -> BGEM3Embedder:
    """Cria embedder BGE-M3."""
    config = EmbeddingConfig(
        use_fp16=True,
        batch_size=8,
        max_length=8192,
    )
    return BGEM3Embedder(config=config)


def compute_file_hash(file_path: Path) -> str:
    """Computa SHA-256 do arquivo."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def insert_into_milvus_v3(chunks: list, collection_name: str = COLLECTION_NAME_V3):
    """Insere chunks na collection v3."""
    from pymilvus import connections, Collection

    logger.info(f"Conectando ao Milvus...")
    connections.connect(host="localhost", port="19530")

    collection = Collection(collection_name)

    # Prepara dados
    data = {
        "chunk_id": [],
        "parent_chunk_id": [],
        "span_id": [],
        "device_type": [],
        "chunk_level": [],
        "text": [],
        "enriched_text": [],
        "dense_vector": [],
        "thesis_vector": [],
        "sparse_vector": [],
        "context_header": [],
        "thesis_text": [],
        "thesis_type": [],
        "synthetic_questions": [],
        "citations": [],
        "document_id": [],
        "tipo_documento": [],
        "numero": [],
        "ano": [],
        "article_number": [],
        "schema_version": [],
        "extractor_version": [],
        "ingestion_timestamp": [],
        "document_hash": [],
        "page": [],
        "bbox_left": [],
        "bbox_top": [],
        "bbox_right": [],
        "bbox_bottom": [],
    }

    for chunk in chunks:
        data["chunk_id"].append(chunk.chunk_id)
        data["parent_chunk_id"].append(chunk.parent_chunk_id or "")
        data["span_id"].append(chunk.span_id)
        data["device_type"].append(chunk.device_type.value)
        data["chunk_level"].append(chunk.chunk_level.name.lower())
        data["text"].append(chunk.text[:65535])
        data["enriched_text"].append((chunk.enriched_text or chunk.text)[:65535])

        # Vetores
        if chunk.dense_vector:
            data["dense_vector"].append(chunk.dense_vector)
        else:
            data["dense_vector"].append([0.0] * 1024)

        # thesis_vector - usa dense_vector se não tiver
        if hasattr(chunk, 'thesis_vector') and chunk.thesis_vector:
            data["thesis_vector"].append(chunk.thesis_vector)
        else:
            data["thesis_vector"].append(chunk.dense_vector or [0.0] * 1024)

        # sparse_vector
        if chunk.sparse_vector:
            data["sparse_vector"].append(chunk.sparse_vector)
        else:
            data["sparse_vector"].append({0: 0.0})

        # Enriquecimento
        data["context_header"].append((chunk.context_header or "")[:2000])
        data["thesis_text"].append((chunk.thesis_text or "")[:5000])
        data["thesis_type"].append(chunk.thesis_type or "disposicao")
        data["synthetic_questions"].append((chunk.synthetic_questions or "")[:10000])

        # Citations como JSON
        data["citations"].append(json.dumps(chunk.citations or []))

        # Metadados
        data["document_id"].append(chunk.document_id or "")
        data["tipo_documento"].append(chunk.tipo_documento or "")
        data["numero"].append(chunk.numero or "")
        data["ano"].append(chunk.ano or 0)
        data["article_number"].append(chunk.article_number or "")

        # Proveniência
        meta = chunk.metadata
        data["schema_version"].append(meta.schema_version or "1.0.0")
        data["extractor_version"].append(meta.extractor_version or "1.0.0")
        data["ingestion_timestamp"].append(meta.ingestion_timestamp or datetime.utcnow().isoformat())
        data["document_hash"].append(meta.document_hash or "")

        # Page spans (placeholder por enquanto)
        data["page"].append(0)
        data["bbox_left"].append(0.0)
        data["bbox_top"].append(0.0)
        data["bbox_right"].append(0.0)
        data["bbox_bottom"].append(0.0)

    logger.info(f"Inserindo {len(chunks)} chunks no Milvus...")

    # Converte para formato de lista
    insert_data = [data[field] for field in data.keys()]

    result = collection.insert(insert_data)
    collection.flush()

    logger.info(f"Inseridos {result.insert_count} chunks")
    logger.info(f"Total na collection: {collection.num_entities}")

    connections.disconnect("default")

    return result


def run_pipeline_v3(
    markdown_path: Path,
    document_id: str = "IN-65-2021",
    tipo_documento: str = "INSTRUCAO NORMATIVA",
    numero: str = "65",
    ano: int = 2021,
    use_llm: bool = True,
    use_embeddings: bool = True,
    use_milvus: bool = True,
):
    """
    Executa pipeline v3 completo.

    Args:
        markdown_path: Caminho para arquivo markdown
        document_id: ID do documento
        tipo_documento: Tipo (IN, LEI, DECRETO)
        numero: Número do documento
        ano: Ano
        use_llm: Se deve usar LLM para extração
        use_embeddings: Se deve gerar embeddings
        use_milvus: Se deve inserir no Milvus
    """
    print("=" * 70)
    print("PIPELINE V3 - SPAN-BASED + PARENT-CHILD")
    print("=" * 70)

    total_start = time.time()

    # Inicializa métricas
    collector = MetricsCollector(ingestion_id=f"{document_id}-{int(time.time())}")
    collector.set_document_info(
        document_id=document_id,
        tipo_documento=tipo_documento,
        numero=numero,
        ano=ano,
    )

    # 1. Carrega markdown
    print("\n[1/6] Carregando markdown...")
    collector.start_phase("load")

    if not markdown_path.exists():
        print(f"      ERRO: Arquivo nao encontrado: {markdown_path}")
        return None

    markdown = markdown_path.read_text(encoding="utf-8")
    file_hash = compute_file_hash(markdown_path)

    print(f"      Arquivo: {markdown_path.name}")
    print(f"      Tamanho: {len(markdown):,} caracteres")
    print(f"      Hash: {file_hash[:16]}...")

    collector.end_phase("load", items_processed=1)

    # 2. Parseia com SpanParser
    print("\n[2/6] Parseando com SpanParser...")
    collector.start_phase("parsing")
    parse_start = time.time()

    parser = SpanParser()
    parsed_doc = parser.parse(markdown)

    parse_time = time.time() - parse_start
    print(f"      Spans totais: {len(parsed_doc.spans)}")
    print(f"      Artigos: {len(parsed_doc.articles)}")
    print(f"      Tempo: {parse_time:.2f}s")

    collector.end_phase("parsing", items_processed=len(parsed_doc.spans))

    # 3. Extrai hierarquia com LLM
    if use_llm:
        print("\n[3/6] Extraindo hierarquia com ArticleOrchestrator...")
        collector.start_phase("extraction")
        extract_start = time.time()

        try:
            llm_client = create_vllm_client()
            models = llm_client.list_models()
            if models:
                print(f"      vLLM conectado: {models[0]}")
            else:
                print("      AVISO: vLLM nao respondeu")
                use_llm = False
        except Exception as e:
            print(f"      ERRO conectando vLLM: {e}")
            use_llm = False

        if use_llm:
            config = OrchestratorConfig(
                temperature=0.0,
                max_tokens=512,
                strict_validation=False,
                enable_retry=True,
                max_retries=2,
            )

            orchestrator = ArticleOrchestrator(llm_client, config)
            extraction_result = orchestrator.extract_all_articles(parsed_doc)

            extract_time = time.time() - extract_start
            print(f"      Artigos extraidos: {extraction_result.total_articles}")
            print(f"      Validos: {extraction_result.valid_articles}")
            print(f"      Suspeitos: {extraction_result.suspect_articles}")
            print(f"      Tempo: {extract_time:.2f}s")

            # Registra métricas por artigo
            for chunk in extraction_result.chunks:
                collector.record_article_metrics(
                    article_id=chunk.article_id,
                    article_number=chunk.article_number,
                    parser_paragrafos=chunk.parser_paragrafos_count,
                    llm_paragrafos=chunk.llm_paragrafos_count,
                    parser_incisos=chunk.parser_incisos_count,
                    llm_incisos=chunk.llm_incisos_count,
                    status=chunk.status.value,
                    validation_notes=chunk.validation_notes,
                    retry_count=chunk.retry_count,
                )

            collector.end_phase("extraction", items_processed=extraction_result.total_articles)

            # Fecha cliente
            llm_client.close()
        else:
            extraction_result = None
    else:
        print("\n[3/6] Pulando extração LLM...")
        extraction_result = None
        collector.end_phase("extraction", items_processed=0)

    # 4. Materializa chunks
    print("\n[4/6] Materializando chunks parent-child...")
    collector.start_phase("materialization")
    mat_start = time.time()

    metadata = ChunkMetadata(
        schema_version="1.0.0",
        extractor_version="1.0.0",
        document_hash=file_hash,
        pdf_path=str(markdown_path),
    )

    materializer = ChunkMaterializer(
        document_id=document_id,
        tipo_documento=tipo_documento,
        numero=numero,
        ano=ano,
        metadata=metadata,
    )

    if extraction_result:
        all_chunks = materializer.materialize_all(
            extraction_result.chunks,
            parsed_doc,
            include_children=True,
        )
    else:
        # Sem LLM, cria chunks apenas dos artigos
        all_chunks = []
        for article in parsed_doc.articles:
            from chunking.chunk_models import ChunkLevel
            from chunking.chunk_materializer import MaterializedChunk

            chunk = MaterializedChunk(
                chunk_id=f"{document_id}#{article.span_id}",
                parent_chunk_id="",
                span_id=article.span_id,
                device_type=DeviceType.ARTICLE,
                chunk_level=ChunkLevel.ARTICLE,
                text=article.text,
                document_id=document_id,
                tipo_documento=tipo_documento,
                numero=numero,
                ano=ano,
                article_number=article.identifier or "",
                citations=[article.span_id],
                metadata=metadata,
            )
            all_chunks.append(chunk)

    mat_time = time.time() - mat_start
    print(f"      Total chunks: {len(all_chunks)}")

    # Contagens por tipo
    article_count = sum(1 for c in all_chunks if c.device_type == DeviceType.ARTICLE)
    par_count = sum(1 for c in all_chunks if c.device_type == DeviceType.PARAGRAPH)
    inc_count = sum(1 for c in all_chunks if c.device_type == DeviceType.INCISO)

    print(f"      ARTICLE: {article_count}")
    print(f"      PARAGRAPH: {par_count}")
    print(f"      INCISO: {inc_count}")
    print(f"      Tempo: {mat_time:.2f}s")

    collector.set_chunk_counts(
        total=len(all_chunks),
        articles=article_count,
        paragraphs=par_count,
        incisos=inc_count,
    )
    collector.end_phase("materialization", items_processed=len(all_chunks))

    # 5. Gera embeddings
    if use_embeddings:
        print("\n[5/6] Gerando embeddings BGE-M3...")
        collector.start_phase("embedding")
        embed_start = time.time()

        try:
            embedder = create_embedder()
            print("      BGE-M3 carregado!")

            # Gera embeddings para cada chunk
            texts = [c.text for c in all_chunks]
            print(f"      Processando {len(texts)} textos...")

            # Processa em batch usando encode_hybrid (dense + sparse)
            for i, chunk in enumerate(all_chunks):
                # Usa texto para embedding (idealmente seria enriched_text)
                text = chunk.text

                # Gera embedding hibrido (dense + sparse)
                result = embedder.encode_hybrid([text])

                chunk.dense_vector = result['dense'][0]
                chunk.sparse_vector = result['sparse'][0]

                if (i + 1) % 10 == 0:
                    print(f"      Processados: {i + 1}/{len(all_chunks)}")

            embed_time = time.time() - embed_start
            print(f"      Embeddings gerados: {len(all_chunks)}")
            print(f"      Tempo: {embed_time:.2f}s")

            collector.end_phase("embedding", items_processed=len(all_chunks))

        except Exception as e:
            print(f"      ERRO gerando embeddings: {e}")
            collector.end_phase("embedding", errors=1)
    else:
        print("\n[5/6] Pulando embeddings...")

    # 6. Insere no Milvus
    if use_milvus and all_chunks:
        print("\n[6/6] Inserindo no Milvus leis_v3...")
        collector.start_phase("indexing")
        index_start = time.time()

        try:
            insert_result = insert_into_milvus_v3(all_chunks)
            index_time = time.time() - index_start
            print(f"      Tempo: {index_time:.2f}s")
            collector.end_phase("indexing", items_processed=len(all_chunks))
        except Exception as e:
            print(f"      ERRO inserindo no Milvus: {e}")
            collector.end_phase("indexing", errors=1)
            collector.set_error(str(e))
    else:
        print("\n[6/6] Pulando inserção no Milvus")

    # Gera relatório
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    report = collector.generate_report()
    print(generate_dashboard_report(report))

    print(f"\nTempo total: {total_time:.2f}s")

    # Salva relatório
    report_path = markdown_path.parent / f"{markdown_path.stem}_pipeline_v3_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    print(f"Relatório salvo: {report_path}")

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Pipeline v3 - Span-Based + Parent-Child")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "output" / "in_65_markdown.md"),
        help="Caminho para arquivo markdown"
    )
    parser.add_argument("--document-id", default="IN-65-2021", help="ID do documento")
    parser.add_argument("--tipo", default="INSTRUCAO NORMATIVA", help="Tipo do documento")
    parser.add_argument("--numero", default="65", help="Numero do documento")
    parser.add_argument("--ano", type=int, default=2021, help="Ano do documento")
    parser.add_argument("--no-llm", action="store_true", help="Nao usar LLM")
    parser.add_argument("--no-embeddings", action="store_true", help="Nao gerar embeddings")
    parser.add_argument("--no-milvus", action="store_true", help="Nao inserir no Milvus")

    args = parser.parse_args()

    markdown_path = Path(args.input)

    try:
        run_pipeline_v3(
            markdown_path=markdown_path,
            document_id=args.document_id,
            tipo_documento=args.tipo,
            numero=args.numero,
            ano=args.ano,
            use_llm=not args.no_llm,
            use_embeddings=not args.no_embeddings,
            use_milvus=not args.no_milvus,
        )
        return 0
    except Exception as e:
        logger.exception(f"Erro no pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
