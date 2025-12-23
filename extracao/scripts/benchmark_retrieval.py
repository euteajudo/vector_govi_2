"""
Benchmark de estratégias de retrieval.

Compara:
- Dense only (BGE-M3 1024d)
- Sparse only (BGE-M3 learned sparse)
- Hybrid RRF (Reciprocal Rank Fusion)
- Hybrid Weighted (0.7 dense + 0.3 sparse)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker, WeightedRanker
from FlagEmbedding import BGEM3FlagModel


def run_benchmark():
    connections.connect(host='localhost', port='19530')
    col = Collection('leis_v3')
    col.load()

    print('=' * 80)
    print('BENCHMARK: COMPARACAO DE ESTRATEGIAS DE RETRIEVAL')
    print('=' * 80)

    # Carrega BGE-M3
    print('\nCarregando BGE-M3...')
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # Queries de teste com respostas esperadas
    test_cases = [
        {
            'query': 'Como fazer pesquisa de precos em contratacoes publicas?',
            'expected': ['ART-005', 'ART-001', 'ART-003'],
            'best_match': 'ART-005'
        },
        {
            'query': 'Quantos fornecedores preciso consultar?',
            'expected': ['INC-005-IV'],
            'best_match': 'INC-005-IV'
        },
        {
            'query': 'O que deve conter o documento de pesquisa de precos?',
            'expected': ['ART-003', 'INC-003-I', 'INC-003-II'],
            'best_match': 'ART-003'
        },
        {
            'query': 'Quando posso usar contratacao direta por inexigibilidade?',
            'expected': ['ART-007'],
            'best_match': 'ART-007'
        },
        {
            'query': 'Qual o prazo para resposta do fornecedor?',
            'expected': ['INC-005-I_2', 'PAR-005-2'],
            'best_match': 'INC-005-I_2'
        }
    ]

    results = {
        'dense': {'scores': [], 'hits': [], 'top1_correct': 0},
        'sparse': {'scores': [], 'hits': [], 'top1_correct': 0},
        'rrf': {'scores': [], 'hits': [], 'top1_correct': 0},
        'weighted': {'scores': [], 'hits': [], 'top1_correct': 0}
    }

    print(f'\nExecutando {len(test_cases)} queries de teste...\n')

    for i, tc in enumerate(test_cases):
        query = tc['query']
        expected = tc['expected']
        best = tc['best_match']

        print(f'Query {i+1}: {query[:50]}...')
        print(f'  Esperado: {best}')

        # Gera embeddings
        enc = model.encode([query], return_dense=True, return_sparse=True)
        dense_vec = enc['dense_vecs'][0].tolist()
        sparse_dict = enc['lexical_weights'][0]
        sparse_vec = {int(k): float(v) for k, v in sparse_dict.items() if abs(v) > 1e-6}

        # 1. Dense only
        dense_res = col.search(
            data=[dense_vec],
            anns_field='dense_vector',
            param={'metric_type': 'COSINE', 'params': {'ef': 64}},
            limit=5,
            output_fields=['span_id']
        )
        top1_dense = dense_res[0][0].entity.get('span_id')
        score_dense = dense_res[0][0].score
        results['dense']['scores'].append(score_dense)
        results['dense']['hits'].append(top1_dense)
        if top1_dense == best:
            results['dense']['top1_correct'] += 1

        # 2. Sparse only
        sparse_res = col.search(
            data=[sparse_vec],
            anns_field='sparse_vector',
            param={'metric_type': 'IP', 'params': {}},
            limit=5,
            output_fields=['span_id']
        )
        top1_sparse = sparse_res[0][0].entity.get('span_id')
        score_sparse = sparse_res[0][0].score
        results['sparse']['scores'].append(score_sparse)
        results['sparse']['hits'].append(top1_sparse)
        if top1_sparse == best:
            results['sparse']['top1_correct'] += 1

        # 3. Hybrid RRF
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field='dense_vector',
            param={'metric_type': 'COSINE', 'params': {'ef': 64}},
            limit=10
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field='sparse_vector',
            param={'metric_type': 'IP', 'params': {}},
            limit=10
        )
        rrf_res = col.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=60),
            limit=5,
            output_fields=['span_id']
        )
        top1_rrf = rrf_res[0][0].entity.get('span_id')
        score_rrf = rrf_res[0][0].score
        results['rrf']['scores'].append(score_rrf)
        results['rrf']['hits'].append(top1_rrf)
        if top1_rrf == best:
            results['rrf']['top1_correct'] += 1

        # 4. Hybrid Weighted (0.7 dense + 0.3 sparse)
        weighted_res = col.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=WeightedRanker(0.7, 0.3),
            limit=5,
            output_fields=['span_id']
        )
        top1_weighted = weighted_res[0][0].entity.get('span_id')
        score_weighted = weighted_res[0][0].score
        results['weighted']['scores'].append(score_weighted)
        results['weighted']['hits'].append(top1_weighted)
        if top1_weighted == best:
            results['weighted']['top1_correct'] += 1

        # Mostra resultados desta query
        ok_dense = "[OK]" if top1_dense == best else ""
        ok_sparse = "[OK]" if top1_sparse == best else ""
        ok_rrf = "[OK]" if top1_rrf == best else ""
        ok_weighted = "[OK]" if top1_weighted == best else ""

        print(f'  Dense:    {top1_dense:15} ({score_dense:.4f}) {ok_dense}')
        print(f'  Sparse:   {top1_sparse:15} ({score_sparse:.4f}) {ok_sparse}')
        print(f'  RRF:      {top1_rrf:15} ({score_rrf:.4f}) {ok_rrf}')
        print(f'  Weighted: {top1_weighted:15} ({score_weighted:.4f}) {ok_weighted}')
        print()

    # Resumo
    print('=' * 80)
    print('RESUMO DO BENCHMARK')
    print('=' * 80)

    print('\n--- ACURACIA TOP-1 (Hit Rate) ---')
    for method in ['dense', 'sparse', 'rrf', 'weighted']:
        correct = results[method]['top1_correct']
        total = len(test_cases)
        pct = (correct / total) * 100
        bar = '#' * int(pct / 5)
        print(f'  {method:10}: {correct}/{total} ({pct:5.1f}%) {bar}')

    print('\n--- SCORE MEDIO ---')
    for method in ['dense', 'sparse', 'rrf', 'weighted']:
        avg = sum(results[method]['scores']) / len(results[method]['scores'])
        print(f'  {method:10}: {avg:.4f}')

    print('\n--- DETALHES POR QUERY ---')
    for i, tc in enumerate(test_cases):
        query_short = tc['query'][:40]
        best = tc['best_match']

        # Verifica quem acertou
        winners = []
        for method in ['dense', 'sparse', 'rrf', 'weighted']:
            if results[method]['hits'][i] == best:
                winners.append(method)

        status = ", ".join(winners) if winners else f"MISS (todos erraram, esperado: {best})"
        print(f'  Q{i+1}: {status}')

    # Conclusao
    print('\n' + '=' * 80)
    print('CONCLUSAO')
    print('=' * 80)

    best_method = max(['dense', 'sparse', 'rrf', 'weighted'],
                      key=lambda m: results[m]['top1_correct'])
    best_acc = results[best_method]['top1_correct'] / len(test_cases) * 100

    print(f'\nMELHOR ESTRATEGIA: {best_method.upper()} ({best_acc:.0f}% acuracia)')

    # Analise comparativa
    dense_acc = results['dense']['top1_correct'] / len(test_cases) * 100
    sparse_acc = results['sparse']['top1_correct'] / len(test_cases) * 100
    rrf_acc = results['rrf']['top1_correct'] / len(test_cases) * 100
    weighted_acc = results['weighted']['top1_correct'] / len(test_cases) * 100

    print(f'\nComparativo de acuracia:')
    print(f'  Dense:    {dense_acc:.0f}%')
    print(f'  Sparse:   {sparse_acc:.0f}%')
    print(f'  RRF:      {rrf_acc:.0f}%')
    print(f'  Weighted: {weighted_acc:.0f}%')

    hybrid_best = max(rrf_acc, weighted_acc)
    if hybrid_best > dense_acc:
        diff = hybrid_best - dense_acc
        print(f'\n>>> HIBRIDO SUPEROU DENSE em {diff:.0f} pontos percentuais!')
    elif hybrid_best == dense_acc:
        print(f'\n>>> HIBRIDO EMPATOU com DENSE')
    else:
        diff = dense_acc - hybrid_best
        print(f'\n>>> DENSE SUPEROU HIBRIDO em {diff:.0f} pontos percentuais')

    connections.disconnect('default')

    return results


if __name__ == "__main__":
    run_benchmark()
