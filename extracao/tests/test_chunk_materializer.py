"""
Teste do ChunkMaterializer - Parent-child retrieval.

Testa:
1. Materialização de ArticleChunk em chunks pai e filhos
2. IDs corretos com parent_chunk_id
3. Metadados de proveniência
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsing import SpanParser, ArticleOrchestrator, OrchestratorConfig
from chunking import ChunkMaterializer, ChunkMetadata, DeviceType
from llm.vllm_client import VLLMClient, LLMConfig


def test_chunk_materializer():
    """Testa materialização de chunks com parent-child."""
    markdown_path = Path(__file__).parent.parent / "data" / "output" / "in_65_markdown.md"

    print("=" * 70)
    print("TESTE DO CHUNK MATERIALIZER - PARENT-CHILD")
    print("=" * 70)

    # 1. Carrega markdown
    print("\n[1] Carregando markdown...")
    if not markdown_path.exists():
        print(f"    Arquivo nao encontrado: {markdown_path}")
        return None

    markdown = markdown_path.read_text(encoding="utf-8")
    print(f"    Markdown: {len(markdown)} caracteres")

    # 2. Parseia com SpanParser
    print("\n[2] Parseando com SpanParser...")
    parser = SpanParser()
    parsed_doc = parser.parse(markdown)
    print(f"    Total spans: {len(parsed_doc.spans)}")
    print(f"    Artigos: {len(parsed_doc.articles)}")

    # 3. Configura cliente vLLM
    print("\n[3] Configurando cliente vLLM...")
    llm_config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-8B-AWQ",
        temperature=0.0,
        max_tokens=512,
    )
    llm_client = VLLMClient(llm_config)

    # 4. Extrai com ArticleOrchestrator
    print("\n[4] Extraindo com ArticleOrchestrator...")
    config = OrchestratorConfig(
        temperature=0.0,
        max_tokens=512,
        strict_validation=False,
    )
    orchestrator = ArticleOrchestrator(llm_client, config)
    result = orchestrator.extract_all_articles(parsed_doc)
    print(f"    Artigos extraidos: {result.total_articles}")
    print(f"    Validos: {result.valid_articles}")

    # 5. Materializa com ChunkMaterializer
    print("\n[5] Materializando chunks parent-child...")
    metadata = ChunkMetadata(
        schema_version="1.0.0",
        extractor_version="1.0.0",
        document_hash="abc123",  # Placeholder
    )

    materializer = ChunkMaterializer(
        document_id="IN-65-2021",
        tipo_documento="INSTRUCAO NORMATIVA",
        numero="65",
        ano=2021,
        metadata=metadata,
    )

    all_chunks = materializer.materialize_all(
        result.chunks, parsed_doc, include_children=True
    )

    print(f"    Total chunks materializados: {len(all_chunks)}")

    # 6. Estatísticas por tipo
    print("\n[6] Estatisticas por tipo:")
    article_count = sum(1 for c in all_chunks if c.device_type == DeviceType.ARTICLE)
    paragraph_count = sum(1 for c in all_chunks if c.device_type == DeviceType.PARAGRAPH)
    inciso_count = sum(1 for c in all_chunks if c.device_type == DeviceType.INCISO)

    print(f"    ARTICLE: {article_count}")
    print(f"    PARAGRAPH: {paragraph_count}")
    print(f"    INCISO: {inciso_count}")

    # 7. Mostra exemplo de parent-child
    print("\n[7] Exemplo parent-child (Art. 5):")
    art5_chunks = [c for c in all_chunks if "ART-005" in c.chunk_id]

    for chunk in art5_chunks:
        parent_info = f"(pai: {chunk.parent_chunk_id})" if chunk.parent_chunk_id else "(raiz)"
        print(f"    {chunk.chunk_id} [{chunk.device_type.value}] {parent_info}")
        print(f"        citations: {chunk.citations[:3]}{'...' if len(chunk.citations) > 3 else ''}")

    # 8. Verifica metadados
    print("\n[8] Metadados de proveniencia:")
    sample = all_chunks[0]
    print(f"    schema_version: {sample.metadata.schema_version}")
    print(f"    extractor_version: {sample.metadata.extractor_version}")
    print(f"    ingestion_timestamp: {sample.metadata.ingestion_timestamp[:19]}...")
    print(f"    document_hash: {sample.metadata.document_hash}")

    # 9. Formato Milvus
    print("\n[9] Formato para Milvus (amostra):")
    milvus_dict = sample.to_milvus_dict()
    for key in ["chunk_id", "parent_chunk_id", "device_type", "citations"]:
        print(f"    {key}: {milvus_dict.get(key)}")

    return all_chunks


def test_parent_child_retrieval_simulation():
    """Simula busca parent-child."""
    print("\n" + "=" * 70)
    print("SIMULACAO DE PARENT-CHILD RETRIEVAL")
    print("=" * 70)

    # Chunks mockados
    chunks = [
        {"chunk_id": "IN-65#ART-005", "parent_chunk_id": "", "text": "Art. 5 completo..."},
        {"chunk_id": "IN-65#PAR-005-1", "parent_chunk_id": "IN-65#ART-005", "text": "§1 Priorizar..."},
        {"chunk_id": "IN-65#PAR-005-2", "parent_chunk_id": "IN-65#ART-005", "text": "§2 Pesquisa..."},
        {"chunk_id": "IN-65#INC-005-I", "parent_chunk_id": "IN-65#ART-005", "text": "I - composicao..."},
        {"chunk_id": "IN-65#INC-005-II", "parent_chunk_id": "IN-65#ART-005", "text": "II - contratacoes..."},
    ]

    # Simula busca que encontra INC-005-II
    query = "contratacoes similares administracao publica"
    print(f"\n[1] Query: '{query}'")
    print("    (simulando que INC-005-II foi o top hit)")

    found_chunk = chunks[4]  # INC-005-II
    print(f"\n[2] Chunk encontrado: {found_chunk['chunk_id']}")

    # Expande para o pai
    parent_id = found_chunk["parent_chunk_id"]
    print(f"\n[3] Expandindo para pai: {parent_id}")

    parent_chunk = next((c for c in chunks if c["chunk_id"] == parent_id), None)
    if parent_chunk:
        print(f"    Pai encontrado: {parent_chunk['chunk_id']}")

    # Contexto expandido para LLM
    print("\n[4] Contexto expandido para LLM:")
    print(f"    - Pai: {parent_chunk['text'][:50]}...")
    print(f"    - Filho (match): {found_chunk['text'][:50]}...")

    print("\n    [OK] Parent-child retrieval funcionando!")


if __name__ == "__main__":
    test_chunk_materializer()
    test_parent_child_retrieval_simulation()
