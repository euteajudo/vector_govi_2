"""
Teste comparativo: ColBERT vs Cross-Encoder (BGE-Reranker)

Compara os dois metodos de reranking para documentos juridicos.

Uso:
    python tests/test_colbert_vs_crossencoder.py
"""

import sys
import time
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from reranker.colbert_reranker import ColBERTReranker
from embeddings.bge_reranker import BGEReranker
from embeddings.bge_m3 import BGEM3Embedder, EmbeddingConfig


def main():
    print("=" * 70)
    print("TESTE COMPARATIVO: ColBERT vs Cross-Encoder")
    print("=" * 70)

    # Conectar ao Milvus
    print("\n[1/4] Conectando ao Milvus...")
    connections.connect(host="localhost", port="19530")
    collection = Collection("leis_v2")
    collection.load()
    print(f"      Collection: leis_v2 ({collection.num_entities} entidades)")

    # Carregar modelos
    print("\n[2/4] Carregando BGE-M3 (embeddings)...")
    embedder = BGEM3Embedder(EmbeddingConfig(use_fp16=True))

    print("\n[3/4] Carregando Cross-Encoder (BGE-Reranker)...")
    cross_encoder = BGEReranker()

    print("\n[4/4] Carregando ColBERT Reranker...")
    colbert = ColBERTReranker()

    print("\nModelos carregados!")

    # Queries de teste
    queries = [
        "O que e ETP e para que serve?",
        "Quando o ETP e dispensado ou facultado?",
        "contratacoes interdependentes e correlatas",
        "quem sao os responsaveis pela elaboracao do ETP?",
        "sistema ETP digital como funciona?",
    ]

    # Resultados agregados
    results_summary = []

    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        # Stage 1: Recall com BGE-M3 hybrid search
        print("\n--- Stage 1: Hybrid Search (BGE-M3) ---")

        # Gerar embeddings da query
        query_embedding = embedder.encode_hybrid_single(query)

        # Busca hibrida (dense + sparse)
        dense_req = AnnSearchRequest(
            data=[query_embedding["dense"]],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=10,
        )

        sparse_req = AnnSearchRequest(
            data=[query_embedding["sparse"]],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=10,
        )

        ranker = WeightedRanker(0.7, 0.3)

        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=10,
            output_fields=["article_number", "text", "thesis_type", "chunk_id"],
        )

        candidates = []
        for hit in results[0]:
            candidates.append({
                "article_number": hit.entity.get("article_number"),
                "text": hit.entity.get("text"),
                "thesis_type": hit.entity.get("thesis_type"),
                "chunk_id": hit.entity.get("chunk_id"),
                "hybrid_score": hit.distance,
            })

        print(f"Top 5 candidatos (Stage 1):")
        for i, c in enumerate(candidates[:5]):
            print(f"  {i+1}. Art-{c['article_number']} ({c['thesis_type']}): {c['hybrid_score']:.4f}")

        # Extrair textos para reranking
        texts = [c["text"] for c in candidates]

        # Stage 2A: Cross-Encoder (BGE-Reranker)
        print("\n--- Stage 2A: Cross-Encoder (BGE-Reranker) ---")

        start_time = time.time()
        ce_scores = cross_encoder.compute_scores(query, texts)
        ce_time = (time.time() - start_time) * 1000

        # Ordenar por score
        ce_results = list(zip(range(len(candidates)), ce_scores))
        ce_results.sort(key=lambda x: x[1], reverse=True)

        print(f"Tempo: {ce_time:.1f}ms")
        print(f"Top 3 (Cross-Encoder):")
        for idx, score in ce_results[:3]:
            c = candidates[idx]
            print(f"  Art-{c['article_number']} ({c['thesis_type']}): {score:.4f}")

        # Stage 2B: ColBERT (MaxSim)
        print("\n--- Stage 2B: ColBERT (MaxSim) ---")

        start_time = time.time()
        colbert_results = colbert.rerank(query, texts, top_k=10)
        colbert_time = (time.time() - start_time) * 1000

        print(f"Tempo: {colbert_time:.1f}ms")
        print(f"Top 3 (ColBERT):")
        for idx, score in colbert_results[:3]:
            c = candidates[idx]
            print(f"  Art-{c['article_number']} ({c['thesis_type']}): {score:.4f}")

        # Comparacao
        print("\n--- COMPARACAO ---")

        ce_top1_idx = ce_results[0][0]
        colbert_top1_idx = colbert_results[0][0]

        ce_top1 = candidates[ce_top1_idx]
        colbert_top1 = candidates[colbert_top1_idx]

        print(f"| Metodo        | Top 1  | Score  | Tempo    |")
        print(f"|---------------|--------|--------|----------|")
        print(f"| Cross-Encoder | Art-{ce_top1['article_number']:2} | {ce_results[0][1]:.4f} | {ce_time:6.1f}ms |")
        print(f"| ColBERT       | Art-{colbert_top1['article_number']:2} | {colbert_results[0][1]:.4f} | {colbert_time:6.1f}ms |")

        # Verificar concordancia
        if ce_top1["article_number"] == colbert_top1["article_number"]:
            agreement = "CONCORDAM"
            print(f"\n  Ambos concordam: Art-{ce_top1['article_number']}")
        else:
            agreement = "DIVERGEM"
            print(f"\n  Divergencia: Cross-Encoder=Art-{ce_top1['article_number']} vs ColBERT=Art-{colbert_top1['article_number']}")

        # Salvar resultado
        results_summary.append({
            "query": query[:40] + "..." if len(query) > 40 else query,
            "ce_top1": ce_top1["article_number"],
            "ce_score": ce_results[0][1],
            "ce_time": ce_time,
            "colbert_top1": colbert_top1["article_number"],
            "colbert_score": colbert_results[0][1],
            "colbert_time": colbert_time,
            "agreement": agreement,
        })

    # Resumo final
    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)

    print("\n| Query                                    | CE     | ColBERT | Acordo   |")
    print("|------------------------------------------|--------|---------|----------|")
    for r in results_summary:
        print(f"| {r['query']:40} | Art-{r['ce_top1']:2} | Art-{r['colbert_top1']:2}  | {r['agreement']:8} |")

    # Estatisticas
    agreements = sum(1 for r in results_summary if r["agreement"] == "CONCORDAM")
    total = len(results_summary)

    avg_ce_time = sum(r["ce_time"] for r in results_summary) / total
    avg_colbert_time = sum(r["colbert_time"] for r in results_summary) / total

    avg_ce_score = sum(r["ce_score"] for r in results_summary) / total
    avg_colbert_score = sum(r["colbert_score"] for r in results_summary) / total

    print(f"\nEstatisticas:")
    print(f"  Concordancia: {agreements}/{total} ({agreements/total*100:.0f}%)")
    print(f"  Tempo medio Cross-Encoder: {avg_ce_time:.1f}ms")
    print(f"  Tempo medio ColBERT: {avg_colbert_time:.1f}ms")
    print(f"  Score medio Cross-Encoder: {avg_ce_score:.4f}")
    print(f"  Score medio ColBERT: {avg_colbert_score:.4f}")

    # Recomendacao
    print("\n" + "=" * 70)
    print("RECOMENDACAO")
    print("=" * 70)

    if avg_colbert_score > avg_ce_score * 1.1:
        print("  ColBERT tem scores significativamente maiores (>10%).")
        print("  Considere usar ColBERT para queries com termos exatos.")
    elif avg_ce_score > avg_colbert_score * 1.1:
        print("  Cross-Encoder tem scores significativamente maiores (>10%).")
        print("  Mantenha o Cross-Encoder como principal reranker.")
    else:
        print("  Scores similares entre os metodos.")
        print("  Cross-Encoder e mais rapido, use como padrao.")

    if avg_ce_time < avg_colbert_time:
        print(f"  Cross-Encoder e {avg_colbert_time/avg_ce_time:.1f}x mais rapido.")
    else:
        print(f"  ColBERT e {avg_ce_time/avg_colbert_time:.1f}x mais rapido.")

    # Cleanup
    connections.disconnect("default")

    print("\n" + "=" * 70)
    print("TESTE FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
