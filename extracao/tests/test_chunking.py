"""
Teste do LawChunker com IN SEGES 58/2022.

Executa chunking sem LLM/embeddings para validar a estrutura.
"""

import json
import sys
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from models.legal_document import LegalDocument
from chunking.law_chunker import LawChunker, ChunkerConfig
from chunking.chunk_models import ChunkLevel


def main():
    print("=" * 70)
    print("TESTE DO LAWCHUNKER - IN SEGES 58/2022")
    print("=" * 70)

    # Carrega documento extraído
    json_path = Path(__file__).parent.parent / "data" / "output" / "resultado_extracao_vllm_v3.json"

    if not json_path.exists():
        print(f"ERRO: Arquivo não encontrado: {json_path}")
        return 1

    print(f"\n1. Carregando documento: {json_path.name}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    doc = LegalDocument.model_validate(data["data"])

    print(f"   Documento: {doc.document_type} nº {doc.number}")
    print(f"   Órgão: {doc.issuing_body_acronym}")
    print(f"   Capítulos: {len(doc.chapters)}")

    total_articles = sum(len(ch.articles) for ch in doc.chapters)
    print(f"   Artigos: {total_articles}")

    # Chunker sem LLM/embeddings (apenas estrutura)
    print("\n2. Executando chunking (modo fast - sem LLM/embeddings)...")

    chunker = LawChunker(config=ChunkerConfig.fast())
    result = chunker.chunk_document(doc)

    print(f"\n3. RESULTADO:")
    print("-" * 70)
    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))

    # Detalhes dos chunks
    print("\n4. CHUNKS GERADOS:")
    print("-" * 70)

    for i, chunk in enumerate(result.chunks, 1):
        level_icon = {
            ChunkLevel.DOCUMENT: "[DOC]",
            ChunkLevel.CHAPTER: "[CAP]",
            ChunkLevel.ARTICLE: "[ART]",
            ChunkLevel.DEVICE: "[DEV]",
        }.get(chunk.chunk_level, "[???]")

        print(f"\n{level_icon} Chunk {i}: {chunk.chunk_id}")
        print(f"   Parent: {chunk.parent_id}")
        print(f"   Level: {chunk.chunk_level.name}")
        print(f"   Article: {chunk.article_number}")
        print(f"   Items: {chunk.item_count}, Paragraphs: {chunk.paragraph_count}")
        print(f"   Tokens: {chunk.token_count}")
        print(f"   Text preview: {chunk.text[:80]}...")

    # Estatísticas
    print("\n5. ESTATÍSTICAS:")
    print("-" * 70)

    levels = {}
    for chunk in result.chunks:
        level = chunk.chunk_level.name
        levels[level] = levels.get(level, 0) + 1

    print(f"   Por nível:")
    for level, count in sorted(levels.items()):
        print(f"     - {level}: {count}")

    tokens = [c.token_count for c in result.chunks]
    print(f"\n   Tokens:")
    print(f"     - Total: {sum(tokens)}")
    print(f"     - Média: {sum(tokens) / len(tokens):.1f}")
    print(f"     - Mínimo: {min(tokens)}")
    print(f"     - Máximo: {max(tokens)}")

    # Valida chunks
    print("\n6. VALIDAÇÃO:")
    print("-" * 70)

    errors = []

    # Todos os artigos têm chunks?
    article_chunks = [c for c in result.chunks if "ART-" in c.chunk_id]
    if len(article_chunks) < total_articles:
        errors.append(f"Faltam chunks de artigos: {len(article_chunks)} < {total_articles}")

    # IDs únicos?
    ids = [c.chunk_id for c in result.chunks]
    if len(ids) != len(set(ids)):
        errors.append("IDs duplicados encontrados!")

    # Todos têm parent_id válido?
    for chunk in result.chunks:
        if not chunk.parent_id:
            errors.append(f"Chunk sem parent_id: {chunk.chunk_id}")

    if errors:
        print("   ERROS:")
        for err in errors:
            print(f"     [X] {err}")
    else:
        print("   [OK] Todos os chunks validos!")

    # Salva resultado
    output_path = Path(__file__).parent.parent / "data" / "output" / "chunks_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "parent_id": c.parent_id,
            "chunk_level": c.chunk_level.name,
            "article_number": c.article_number,
            "chapter_number": c.chapter_number,
            "item_count": c.item_count,
            "paragraph_count": c.paragraph_count,
            "token_count": c.token_count,
            "text": c.text,
        }
        for c in result.chunks
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "document_id": result.document_id,
                "summary": result.summary(),
                "chunks": chunks_data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n7. Resultado salvo em: {output_path}")

    print("\n" + "=" * 70)
    print("TESTE CONCLUÍDO")
    print("=" * 70)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
