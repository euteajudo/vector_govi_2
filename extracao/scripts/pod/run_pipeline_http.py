"""
Pipeline v3 usando GPU Server via HTTP direto (sem SSH tunnel).

Pre-requisitos:
1. GPU Server rodando no POD (python -m src.pod.gpu_server --port 8080)
2. Porta 8080 exposta no RunPod (Edit Pod -> TCP ports)
3. Milvus rodando localmente

Uso:
    # Com GPU Server na porta padrao
    python scripts/pod/run_pipeline_http.py --input data/output/L14133.md

    # Especificando URL do GPU Server
    python scripts/pod/run_pipeline_http.py --input data/output/L14133.md \\
        --gpu-url http://195.26.233.70:8080

    # Sem extracao LLM (apenas parse + embeddings)
    python scripts/pod/run_pipeline_http.py --input data/output/L14133.md --no-llm
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
from milvus.schema_v3 import COLLECTION_NAME_V3
from dashboard import MetricsCollector, generate_dashboard_report
from pod.config import PodConfig, get_pod_config
from pod.gpu_client import GPUClient
from validation import PostIngestionValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GPUClientLLMAdapter:
    """Adaptador para usar GPUClient como LLM client no ArticleOrchestrator."""

    def __init__(self, gpu_client: GPUClient, model: str = "qwen3-8b"):
        self.client = gpu_client
        self.model = model

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict = None,
    ) -> dict:
        """Compativel com interface do VLLMClient."""
        result = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return {
            "content": result["content"],
            "usage": result.get("usage", {}),
        }

    def chat_with_schema(
        self,
        messages: list[dict],
        schema: dict,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict:
        """Chat com guided JSON usando json_schema.

        Retorna o JSON parseado diretamente (nao o wrapper do chat).
        Compativel com VLLMClient.chat_with_schema().
        """
        import json

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "article_spans",
                "strict": True,
                "schema": schema
            }
        }
        result = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        # Extrai content e parseia como JSON
        content = result.get("content", "{}")
        return json.loads(content)

    def list_models(self) -> list[str]:
        return self.client.list_models()

    def close(self):
        self.client.close()


def compute_file_hash(file_path: Path) -> str:
    """Computa SHA-256 do arquivo."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def insert_into_milvus_v3(chunks: list, collection_name: str = COLLECTION_NAME_V3):
    """Insere chunks na collection v3."""
    import os
    from pymilvus import connections, Collection

    milvus_host = os.getenv("MILVUS_HOST", "77.37.43.160")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    logger.info(f"Conectando ao Milvus ({milvus_host}:{milvus_port})...")
    connections.connect(host=milvus_host, port=milvus_port)

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

    logger.info(f"Inserindo {len(chunks)} chunks no Milvus...")

    insert_data = [data[field] for field in data.keys()]
    result = collection.insert(insert_data)
    collection.flush()

    logger.info(f"Inseridos {result.insert_count} chunks")
    logger.info(f"Total na collection: {collection.num_entities}")

    connections.disconnect("default")
    return result


def run_pipeline_http(
    markdown_path: Path,
    gpu_url: str = "http://195.26.233.70:55278",
    document_id: str = "LEI-14133-2021",
    tipo_documento: str = "LEI",
    numero: str = "14133",
    ano: int = 2021,
    use_llm: bool = True,
    use_embeddings: bool = True,
    use_milvus: bool = True,
    validate: bool = True,
    auto_fix: bool = True,
):
    """
    Executa pipeline v3 usando GPU Server via HTTP direto.

    Args:
        markdown_path: Caminho para arquivo markdown
        gpu_url: URL do GPU Server (ex: http://195.26.233.70:8080)
        document_id: ID do documento
        tipo_documento: Tipo (IN, LEI, DECRETO)
        numero: Numero do documento
        ano: Ano
        use_llm: Se deve usar LLM para extracao
        use_embeddings: Se deve gerar embeddings
        use_milvus: Se deve inserir no Milvus
        validate: Se deve validar chunks apos insercao
        auto_fix: Se deve corrigir erros automaticamente
    """
    print("=" * 70)
    print("PIPELINE V3 - HTTP DIRETO (SEM SSH TUNNEL)")
    print("=" * 70)
    print(f"GPU Server: {gpu_url}")
    print(f"Milvus: localhost:19530")
    print("=" * 70)

    total_start = time.time()

    # Inicializa GPU Client
    gpu_client = GPUClient(gpu_url)

    # Verifica conexao
    print("\n[0/7] Verificando conexao com GPU Server...")
    health = gpu_client.health_check()
    if health.get("status") != "ok":
        print(f"      ERRO: GPU Server offline ou inacessivel")
        print(f"      Status: {health}")
        print("\n      Verifique:")
        print("      1. GPU Server esta rodando no POD?")
        print("      2. Porta 8080 esta exposta no RunPod?")
        return None

    print(f"      Embeddings: {'OK' if health.get('embedding_model_loaded') else 'Carregando...'}")
    print(f"      vLLM: {'OK' if health.get('vllm_available') else 'Offline'}")
    print(f"      GPU: {'Disponivel' if health.get('gpu_available') else 'Nao disponivel'}")

    # Inicializa metricas
    collector = MetricsCollector(ingestion_id=f"{document_id}-{int(time.time())}")
    collector.set_document_info(
        document_id=document_id,
        tipo_documento=tipo_documento,
        numero=numero,
        ano=ano,
    )

    # 1. Carrega markdown
    print("\n[1/7] Carregando markdown...")
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
    print("\n[2/7] Parseando com SpanParser...")
    collector.start_phase("parsing")
    parse_start = time.time()

    parser = SpanParser()
    parsed_doc = parser.parse(markdown)

    parse_time = time.time() - parse_start
    print(f"      Spans totais: {len(parsed_doc.spans)}")
    print(f"      Artigos: {len(parsed_doc.articles)}")
    print(f"      Tempo: {parse_time:.2f}s")

    collector.end_phase("parsing", items_processed=len(parsed_doc.spans))

    # 3. Extrai hierarquia com LLM (GPU Server)
    if use_llm and health.get('vllm_available'):
        print("\n[3/7] Extraindo hierarquia com LLM (GPU Server)...")
        collector.start_phase("extraction")
        extract_start = time.time()

        # Cria adaptador LLM
        llm_adapter = GPUClientLLMAdapter(gpu_client)

        models = llm_adapter.list_models()
        if models:
            print(f"      Modelo: {models[0]}")

        orch_config = OrchestratorConfig(
            temperature=0.0,
            max_tokens=512,
            strict_validation=False,
            enable_retry=True,
            max_retries=2,
        )

        orchestrator = ArticleOrchestrator(llm_adapter, orch_config)
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
    else:
        if not health.get('vllm_available'):
            print("\n[3/7] vLLM offline - pulando extracao...")
        else:
            print("\n[3/7] Pulando extracao LLM...")
        extraction_result = None
        collector.end_phase("extraction", items_processed=0)

    # 4. Materializa chunks (local)
    print("\n[4/7] Materializando chunks parent-child...")
    collector.start_phase("materialization")
    mat_start = time.time()

    metadata = ChunkMetadata(
        schema_version="1.0.0",
        extractor_version="1.0.0-http",
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

    # 5. Gera embeddings (GPU Server)
    if use_embeddings and health.get('embedding_model_loaded'):
        print("\n[5/7] Gerando embeddings via GPU Server...")
        collector.start_phase("embedding")
        embed_start = time.time()

        try:
            # Processa em batches
            texts = [c.text for c in all_chunks]
            print(f"      Processando {len(texts)} textos...")

            batch_size = 32
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                batch_texts = [c.text for c in batch_chunks]

                result = gpu_client.embed_hybrid(batch_texts)

                for j, chunk in enumerate(batch_chunks):
                    chunk.dense_vector = result["dense"][j]
                    chunk.sparse_vector = result["sparse"][j]
                    chunk.thesis_vector = result["dense"][j]

                processed = min(i + batch_size, len(all_chunks))
                print(f"      Processados: {processed}/{len(all_chunks)}")

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
        print("\n[5/7] Pulando embeddings...")

    # 6. Insere no Milvus (local)
    if use_milvus and all_chunks and use_embeddings:
        print("\n[6/7] Inserindo no Milvus local...")
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
        print("\n[6/7] Pulando insercao no Milvus")

    # 7. Validacao pos-ingestao
    validation_result = None
    if validate and use_milvus and all_chunks:
        print("\n[7/7] Validando chunks inseridos...")
        collector.start_phase("validation")
        val_start = time.time()

        try:
            validator = PostIngestionValidator(
                gpu_server=gpu_url,
                milvus_host="77.37.43.160",
            )
            validation_result = validator.validate_document(document_id, auto_fix=auto_fix)

            val_time = time.time() - val_start
            print(f"      Total chunks: {validation_result.total_chunks}")
            print(f"      Validos: {validation_result.valid_chunks} ({validation_result.success_rate:.1%})")
            print(f"      Erros encontrados: {len(validation_result.errors)}")
            print(f"      Corrigidos: {len(validation_result.fixed_chunks)}")
            print(f"      Nao corrigiveis: {len(validation_result.unfixable_chunks)}")
            print(f"      Tempo: {val_time:.2f}s")

            if validation_result.unfixable_chunks:
                print("\n      [ATENCAO] Chunks que precisam de revisao manual:")
                for chunk_id in validation_result.unfixable_chunks:
                    print(f"        - {chunk_id}")

            collector.end_phase("validation", items_processed=validation_result.total_chunks)

        except Exception as e:
            print(f"      ERRO na validacao: {e}")
            import traceback
            traceback.print_exc()
            collector.end_phase("validation", errors=1)
    else:
        print("\n[7/7] Pulando validacao")

    # Cleanup
    gpu_client.close()

    # Relatorio
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    report = collector.generate_report()
    print(generate_dashboard_report(report))

    print(f"\nTempo total: {total_time:.2f}s")

    # Salva relatorio
    report_path = markdown_path.parent / f"{markdown_path.stem}_pipeline_http_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    print(f"Relatorio salvo: {report_path}")

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Pipeline v3 - HTTP Direto")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Caminho para arquivo markdown"
    )
    parser.add_argument(
        "--gpu-url",
        type=str,
        default="http://195.26.233.70:55278",
        help="URL do GPU Server (ex: http://IP:55278)"
    )
    parser.add_argument("--document-id", default="LEI-14133-2021", help="ID do documento")
    parser.add_argument("--tipo", default="LEI", help="Tipo do documento")
    parser.add_argument("--numero", default="14133", help="Numero do documento")
    parser.add_argument("--ano", type=int, default=2021, help="Ano do documento")
    parser.add_argument("--no-llm", action="store_true", help="Nao usar LLM para extracao")
    parser.add_argument("--no-embeddings", action="store_true", help="Nao gerar embeddings")
    parser.add_argument("--no-milvus", action="store_true", help="Nao inserir no Milvus")
    parser.add_argument("--no-validate", action="store_true", help="Nao validar apos insercao")
    parser.add_argument("--no-fix", action="store_true", help="Nao corrigir erros automaticamente")

    args = parser.parse_args()

    markdown_path = Path(args.input)

    try:
        run_pipeline_http(
            markdown_path=markdown_path,
            gpu_url=args.gpu_url,
            document_id=args.document_id,
            tipo_documento=args.tipo,
            numero=args.numero,
            ano=args.ano,
            use_llm=not args.no_llm,
            use_embeddings=not args.no_embeddings,
            use_milvus=not args.no_milvus,
            validate=not args.no_validate,
            auto_fix=not args.no_fix,
        )
        return 0
    except Exception as e:
        logger.exception(f"Erro no pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
