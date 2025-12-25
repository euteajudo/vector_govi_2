"""
Pipeline v3 usando POD para GPU (embeddings e LLM).

Estrutura:
- Milvus: LOCAL (acesso via tunel SSH reverso no POD)
- vLLM: POD (localhost:8000 via tunel)
- Embeddings: POD (localhost:8100 via tunel)

Pre-requisitos:
1. Tunel SSH ativo (tunnel_monitor.bat ou keep_tunnel_alive.ps1)
2. vLLM rodando no POD
3. Embedding server rodando no POD

Uso:
    # Ingestao completa (com enriquecimento)
    python scripts/pod/run_pipeline_pod.py --input data/output/L14133.md

    # Ingestao rapida (sem enriquecimento - para fazer depois com Celery)
    python scripts/pod/run_pipeline_pod.py --input data/output/L14133.md --no-enrichment

    # Especificar documento
    python scripts/pod/run_pipeline_pod.py --input data/output/L14133.md \\
        --document-id LEI-14133-2021 --tipo LEI --numero 14133 --ano 2021
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from parsing import SpanParser, ArticleOrchestrator, OrchestratorConfig
from chunking import ChunkMaterializer, ChunkMetadata, DeviceType
from llm.vllm_client import VLLMClient, LLMConfig
from milvus.schema_v3 import COLLECTION_NAME_V3
from dashboard import MetricsCollector, generate_dashboard_report
from pod.config import PodConfig, get_pod_config
from pod.remote_embedder import RemoteEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_vllm_client(config: PodConfig) -> VLLMClient:
    """Cria cliente vLLM apontando para POD."""
    llm_config = LLMConfig(
        base_url=config.vllm_base_url,
        model=config.vllm_model,
        temperature=0.0,
        max_tokens=512,
    )
    return VLLMClient(config=llm_config)


def create_remote_embedder(config: PodConfig) -> RemoteEmbedder:
    """Cria cliente de embeddings remoto."""
    return RemoteEmbedder(config)


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

    logger.info("Conectando ao Milvus local...")
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

        if hasattr(chunk, 'thesis_vector') and chunk.thesis_vector:
            data["thesis_vector"].append(chunk.thesis_vector)
        else:
            data["thesis_vector"].append(chunk.dense_vector or [0.0] * 1024)

        if chunk.sparse_vector:
            data["sparse_vector"].append(chunk.sparse_vector)
        else:
            data["sparse_vector"].append({0: 0.0})

        # Enriquecimento
        data["context_header"].append((chunk.context_header or "")[:2000])
        data["thesis_text"].append((chunk.thesis_text or "")[:5000])
        data["thesis_type"].append(chunk.thesis_type or "disposicao")
        data["synthetic_questions"].append((chunk.synthetic_questions or "")[:10000])

        data["citations"].append(json.dumps(chunk.citations or []))

        # Metadados
        data["document_id"].append(chunk.document_id or "")
        data["tipo_documento"].append(chunk.tipo_documento or "")
        data["numero"].append(chunk.numero or "")
        data["ano"].append(chunk.ano or 0)
        data["article_number"].append(chunk.article_number or "")

        # Proveniencia
        meta = chunk.metadata
        data["schema_version"].append(meta.schema_version or "1.0.0")
        data["extractor_version"].append(meta.extractor_version or "1.0.0")
        data["ingestion_timestamp"].append(meta.ingestion_timestamp or datetime.utcnow().isoformat())
        data["document_hash"].append(meta.document_hash or "")

        # Page spans
        data["page"].append(0)
        data["bbox_left"].append(0.0)
        data["bbox_top"].append(0.0)
        data["bbox_right"].append(0.0)
        data["bbox_bottom"].append(0.0)

    logger.info(f"Inserindo {len(chunks)} chunks no Milvus...")

    insert_data = [data[field] for field in data.keys()]
    result = collection.insert(insert_data)
    collection.flush()

    logger.info(f"Inseridos {result.insert_count} chunks")
    logger.info(f"Total na collection: {collection.num_entities}")

    connections.disconnect("default")
    return result


def run_pipeline_pod(
    markdown_path: Path,
    document_id: str = "LEI-14133-2021",
    tipo_documento: str = "LEI",
    numero: str = "14133",
    ano: int = 2021,
    use_llm: bool = True,
    use_enrichment: bool = False,  # Default False - fazer depois com Celery
    use_embeddings: bool = True,
    use_milvus: bool = True,
    pod_config: PodConfig = None,
):
    """
    Executa pipeline v3 usando POD para GPU.

    Args:
        markdown_path: Caminho para arquivo markdown
        document_id: ID do documento
        tipo_documento: Tipo (IN, LEI, DECRETO)
        numero: Numero do documento
        ano: Ano
        use_llm: Se deve usar LLM para extracao (via POD)
        use_enrichment: Se deve usar LLM para enriquecimento (desabilitado por padrao)
        use_embeddings: Se deve gerar embeddings (via POD)
        use_milvus: Se deve inserir no Milvus (local)
        pod_config: Configuracao do POD
    """
    config = pod_config or get_pod_config()

    print("=" * 70)
    print("PIPELINE V3 - POD MODE (GPU REMOTA)")
    print("=" * 70)
    print(f"vLLM: {config.vllm_base_url}")
    print(f"Embeddings: {config.embedding_base_url}")
    print(f"Milvus: {config.milvus_host}:{config.milvus_port}")
    print("=" * 70)

    total_start = time.time()

    # Inicializa metricas
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

    # 2. Parseia com SpanParser (local - nao precisa GPU)
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

    # 3. Extrai hierarquia com LLM (POD)
    if use_llm:
        print("\n[3/6] Extraindo hierarquia com LLM (POD)...")
        collector.start_phase("extraction")
        extract_start = time.time()

        try:
            llm_client = create_vllm_client(config)
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
            orch_config = OrchestratorConfig(
                temperature=0.0,
                max_tokens=512,
                strict_validation=False,
                enable_retry=True,
                max_retries=2,
            )

            orchestrator = ArticleOrchestrator(llm_client, orch_config)
            extraction_result = orchestrator.extract_all_articles(parsed_doc)

            extract_time = time.time() - extract_start
            print(f"      Artigos extraidos: {extraction_result.total_articles}")
            print(f"      Validos: {extraction_result.valid_articles}")
            print(f"      Suspeitos: {extraction_result.suspect_articles}")
            print(f"      Tempo: {extract_time:.2f}s")

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
            llm_client.close()
        else:
            extraction_result = None
    else:
        print("\n[3/6] Pulando extracao LLM...")
        extraction_result = None
        collector.end_phase("extraction", items_processed=0)

    # 4. Materializa chunks (local)
    print("\n[4/6] Materializando chunks parent-child...")
    collector.start_phase("materialization")
    mat_start = time.time()

    metadata = ChunkMetadata(
        schema_version="1.0.0",
        extractor_version="1.0.0-pod",
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

    # 4.5. Enriquecimento (PULAR - fazer depois com Celery)
    if use_enrichment:
        print("\n[4.5/6] Enriquecimento habilitado - executando...")
        # TODO: Implementar enriquecimento via POD
        print("      (Nao implementado ainda - use Celery workers)")
    else:
        print("\n[4.5/6] Pulando enriquecimento (usar Celery depois)...")

    # 5. Gera embeddings (POD)
    if use_embeddings:
        print("\n[5/6] Gerando embeddings via POD...")
        collector.start_phase("embedding")
        embed_start = time.time()

        try:
            embedder = create_remote_embedder(config)

            # Verifica conexao
            if not embedder.health_check():
                print("      ERRO: Servidor de embeddings offline!")
                print(f"      URL: {config.embedding_base_url}")
                use_embeddings = False
            else:
                print("      Servidor de embeddings conectado!")

                # Processa em batches
                texts = [c.text for c in all_chunks]
                print(f"      Processando {len(texts)} textos...")

                batch_size = 32
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i + batch_size]
                    batch_texts = [c.text for c in batch_chunks]

                    result = embedder.encode_hybrid(batch_texts)

                    for j, chunk in enumerate(batch_chunks):
                        chunk.dense_vector = result["dense"][j]
                        chunk.sparse_vector = result["sparse"][j]
                        # thesis_vector = dense por enquanto
                        chunk.thesis_vector = result["dense"][j]

                    processed = min(i + batch_size, len(all_chunks))
                    print(f"      Processados: {processed}/{len(all_chunks)}")

                embedder.close()

                embed_time = time.time() - embed_start
                print(f"      Embeddings gerados: {len(all_chunks)}")
                print(f"      Tempo: {embed_time:.2f}s")

                collector.end_phase("embedding", items_processed=len(all_chunks))

        except Exception as e:
            print(f"      ERRO gerando embeddings: {e}")
            import traceback
            traceback.print_exc()
            collector.end_phase("embedding", errors=1)
    else:
        print("\n[5/6] Pulando embeddings...")

    # 6. Insere no Milvus (local)
    if use_milvus and all_chunks and use_embeddings:
        print("\n[6/6] Inserindo no Milvus local...")
        collector.start_phase("indexing")
        index_start = time.time()

        try:
            insert_result = insert_into_milvus_v3(all_chunks)
            index_time = time.time() - index_start
            print(f"      Tempo: {index_time:.2f}s")
            collector.end_phase("indexing", items_processed=len(all_chunks))
        except Exception as e:
            print(f"      ERRO inserindo no Milvus: {e}")
            import traceback
            traceback.print_exc()
            collector.end_phase("indexing", errors=1)
            collector.set_error(str(e))
    else:
        print("\n[6/6] Pulando insercao no Milvus")

    # Relatorio
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    report = collector.generate_report()
    print(generate_dashboard_report(report))

    print(f"\nTempo total: {total_time:.2f}s")

    # Salva relatorio
    report_path = markdown_path.parent / f"{markdown_path.stem}_pipeline_pod_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    print(f"Relatorio salvo: {report_path}")

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Pipeline v3 - POD Mode")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Caminho para arquivo markdown"
    )
    parser.add_argument("--document-id", default="LEI-14133-2021", help="ID do documento")
    parser.add_argument("--tipo", default="LEI", help="Tipo do documento")
    parser.add_argument("--numero", default="14133", help="Numero do documento")
    parser.add_argument("--ano", type=int, default=2021, help="Ano do documento")
    parser.add_argument("--no-llm", action="store_true", help="Nao usar LLM para extracao")
    parser.add_argument("--with-enrichment", action="store_true", help="Incluir enriquecimento (lento)")
    parser.add_argument("--no-embeddings", action="store_true", help="Nao gerar embeddings")
    parser.add_argument("--no-milvus", action="store_true", help="Nao inserir no Milvus")

    # POD config
    parser.add_argument("--vllm-url", default=None, help="URL do vLLM (ex: http://localhost:8000/v1)")
    parser.add_argument("--embedding-url", default=None, help="URL do embedding server")

    args = parser.parse_args()

    # Configura POD
    config = get_pod_config()
    if args.vllm_url:
        config.vllm_host = args.vllm_url.split("://")[1].split(":")[0]
        config.vllm_port = int(args.vllm_url.split(":")[-1].replace("/v1", ""))
    if args.embedding_url:
        config.embedding_host = args.embedding_url.split("://")[1].split(":")[0]
        config.embedding_port = int(args.embedding_url.split(":")[-1])

    markdown_path = Path(args.input)

    try:
        run_pipeline_pod(
            markdown_path=markdown_path,
            document_id=args.document_id,
            tipo_documento=args.tipo,
            numero=args.numero,
            ano=args.ano,
            use_llm=not args.no_llm,
            use_enrichment=args.with_enrichment,
            use_embeddings=not args.no_embeddings,
            use_milvus=not args.no_milvus,
            pod_config=config,
        )
        return 0
    except Exception as e:
        logger.exception(f"Erro no pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
