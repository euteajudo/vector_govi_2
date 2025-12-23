"""
Teste do pipeline 2-stage retrieval com BGE-M3 + BGE-Reranker.

Demonstra:
1. Busca inicial com BGE-M3 (dense + sparse + thesis) no Milvus
2. Reranking com BGE-Reranker-v2-m3 usando enriched_text

Uso:
    python tests/test_2stage_retrieval.py
"""

import sys
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from embeddings import BGEM3Embedder, EmbeddingConfig, BGEReranker, RerankerConfig


def test_2stage_retrieval():
    """Testa pipeline completo de 2-stage retrieval."""

    print("=" * 70)
    print("TESTE: 2-Stage Retrieval (BGE-M3 + BGE-Reranker)")
    print("=" * 70)

    # =========================================================================
    # 1. SETUP
    # =========================================================================

    print("\n[1/5] Conectando ao Milvus...")
    connections.connect(host="localhost", port="19530")
    collection = Collection("leis_v2")
    collection.load()
    print(f"      Collection: leis_v2 ({collection.num_entities} entidades)")

    print("\n[2/5] Carregando BGE-M3...")
    embedder = BGEM3Embedder(EmbeddingConfig(use_fp16=True))

    print("\n[3/5] Carregando BGE-Reranker...")
    reranker = BGEReranker(RerankerConfig(use_fp16=True))

    # =========================================================================
    # 2. QUERIES DE TESTE
    # =========================================================================

    queries = [
        "O que e ETP e qual sua finalidade?",
        "Quando o ETP e dispensado ou facultado?",
        "contratacoes interdependentes e correlatas",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        # =====================================================================
        # STAGE 1: Busca inicial com BGE-M3 (top 10)
        # =====================================================================

        print("\n--- STAGE 1: Busca BGE-M3 (dense + sparse) ---")

        # Gera embedding da query
        query_embedding = embedder.encode_hybrid_single(query)

        # Busca dense (COSINE - mesmo do indice)
        dense_req = AnnSearchRequest(
            data=[query_embedding["dense"]],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=10,
        )

        # Busca sparse (IP - padrao para sparse)
        sparse_req = AnnSearchRequest(
            data=[query_embedding["sparse"]],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=10,
        )

        # Busca thesis (COSINE - mesmo embedding para thesis)
        thesis_req = AnnSearchRequest(
            data=[query_embedding["dense"]],
            anns_field="thesis_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=10,
        )

        # Hybrid search com 3 vetores (dense=0.5, sparse=0.3, thesis=0.2)
        stage1_results = collection.hybrid_search(
            reqs=[dense_req, sparse_req, thesis_req],
            rerank=WeightedRanker(0.5, 0.3, 0.2),
            limit=10,
            output_fields=["chunk_id", "text", "enriched_text", "article_number", "thesis_type", "context_header"],
        )

        # Extrai documentos do stage 1
        stage1_docs = []
        print(f"\n  Top 10 candidatos (Stage 1):")
        for i, hit in enumerate(stage1_results[0], 1):
            # Usa enriched_text se disponivel, senao usa text
            enriched = hit.entity.get("enriched_text", "")
            text = hit.entity.get("text", "")
            doc = {
                "chunk_id": hit.entity.get("chunk_id"),
                "text": text,
                "enriched_text": enriched or text,  # Fallback para text
                "article": hit.entity.get("article_number"),
                "thesis_type": hit.entity.get("thesis_type"),
                "context_header": hit.entity.get("context_header", ""),
                "stage1_score": hit.distance,
            }
            stage1_docs.append(doc)
            print(f"  {i:2}. Score: {hit.distance:.4f} | Art-{doc['article']} | {doc['thesis_type']}")

        # =====================================================================
        # STAGE 2: Reranking com BGE-Reranker (top 3)
        # =====================================================================

        print("\n--- STAGE 2: Reranking BGE-Reranker ---")

        # Rerank usando enriched_text (contexto + texto + perguntas)
        reranked = reranker.rerank(
            query=query,
            documents=stage1_docs,
            text_key="enriched_text",  # Usa texto enriquecido
            top_k=5,
            return_scores=True,
        )

        print(f"\n  Top 5 apos reranking (Stage 2):")
        for i, doc in enumerate(reranked, 1):
            print(f"  {i}. Rerank: {doc['rerank_score']:.4f} | Stage1: {doc['stage1_score']:.4f} | Art-{doc['article']} | {doc['thesis_type']}")
            if doc.get("context_header"):
                print(f"     Header: {doc['context_header'][:60]}...")

        # =====================================================================
        # COMPARACAO
        # =====================================================================

        print("\n--- COMPARACAO Stage1 vs Stage2 ---")

        # Pega top 3 de cada
        top3_stage1 = [d["article"] for d in stage1_docs[:3]]
        top3_stage2 = [d["article"] for d in reranked[:3]]

        print(f"  Stage 1 Top 3: {top3_stage1}")
        print(f"  Stage 2 Top 3: {top3_stage2}")

        if top3_stage1 == top3_stage2:
            print("  -> Mesma ordem (reranker concordou)")
        else:
            print("  -> Ordem diferente (reranker reordenou!)")

    # =========================================================================
    # CLEANUP
    # =========================================================================

    connections.disconnect("default")
    print("\n" + "=" * 70)
    print("TESTE CONCLUIDO")
    print("=" * 70)


if __name__ == "__main__":
    test_2stage_retrieval()
