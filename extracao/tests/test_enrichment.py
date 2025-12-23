"""
Teste de enriquecimento de chunks com vLLM.

Testa a geracao de:
- context_header
- thesis_text
- thesis_type
- synthetic_questions

Uso:
    python tests/test_enrichment.py
    python tests/test_enrichment.py --model Qwen/Qwen3-4B-AWQ  # Para benchmark
"""

import json
import sys
import time
import argparse
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from llm.vllm_client import VLLMClient, LLMConfig
from chunking.enrichment_prompts import (
    build_enrichment_prompt,
    parse_enrichment_response,
    build_enriched_text,
)


# Chunks de teste (extraidos da IN SEGES 58/2022)
TEST_CHUNKS = [
    {
        "article_number": "3",
        "article_title": "Definicoes",
        "chapter_number": "I",
        "chapter_title": "DISPOSICOES PRELIMINARES",
        "text": """Art. 3 Para fins do disposto nesta Instrucao Normativa, considera-se:
I - Estudo Tecnico Preliminar - ETP: documento constitutivo da primeira etapa do planejamento de uma contratacao que caracteriza o interesse publico envolvido e a sua melhor solucao e da base ao anteprojeto, ao termo de referencia ou ao projeto basico a serem elaborados caso se conclua pela viabilidade da contratacao;
II - Sistema ETP Digital: ferramenta informatizada integrante da plataforma do Sistema Integrado de Administracao de Servicos Gerais - Siasg;
III - contratacoes correlatas: aquelas cujos objetos sejam similares ou correspondentes entre si;
IV - contratacoes interdependentes: aquelas que, por guardarem relacao direta na execucao do objeto, devem ser contratadas juntamente para a plena satisfacao da necessidade da Administracao.""",
        "expected_type": "definicao",
    },
    {
        "article_number": "6",
        "article_title": None,
        "chapter_number": "II",
        "chapter_title": "ELABORACAO",
        "text": """Art. 6 O ETP devera evidenciar o problema a ser resolvido e a melhor solucao, de modo a permitir a avaliacao da viabilidade tecnica, socioeconomica e ambiental da contratacao.""",
        "expected_type": "procedimento",
    },
    {
        "article_number": "14",
        "article_title": None,
        "chapter_number": "II",
        "chapter_title": "ELABORACAO",
        "text": """Art. 14 A elaboracao do ETP:
I - e facultada nas hipoteses dos incisos I, II, VII e VIII do art. 75 e do S 7 do art. 90 da Lei n 14.133, de 2021; e
II - e dispensada na hipotese do inciso III do art. 75 da Lei n 14.133, de 2021, e nos casos de prorrogacoes dos contratos de servicos e fornecimentos continuos.""",
        "expected_type": "excecao",
    },
]


def test_enrichment(client: VLLMClient, chunk: dict, verbose: bool = True) -> dict:
    """Testa enriquecimento de um chunk."""

    system_prompt, user_prompt = build_enrichment_prompt(
        text=chunk["text"],
        document_type="INSTRUCAO NORMATIVA",
        number="58",
        year=2022,
        issuing_body="SEGES/ME",
        chapter_number=chunk["chapter_number"],
        chapter_title=chunk["chapter_title"],
        article_number=chunk["article_number"],
        article_title=chunk.get("article_title"),
    )

    start_time = time.time()

    response = client.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    elapsed = time.time() - start_time

    # Parse response
    try:
        enrichment = parse_enrichment_response(response)
        success = True
    except Exception as e:
        enrichment = {"error": str(e), "raw_response": response[:500]}
        success = False

    result = {
        "article": chunk["article_number"],
        "expected_type": chunk.get("expected_type"),
        "actual_type": enrichment.get("thesis_type"),
        "type_match": enrichment.get("thesis_type") == chunk.get("expected_type"),
        "elapsed_seconds": elapsed,
        "success": success,
        "enrichment": enrichment,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Art. {chunk['article_number']} - {chunk.get('article_title', 'Sem titulo')}")
        print(f"{'='*60}")
        print(f"Tempo: {elapsed:.2f}s")
        print(f"Sucesso: {success}")

        if success:
            print(f"\ncontext_header:")
            print(f"  {enrichment.get('context_header', 'N/A')[:200]}")

            print(f"\nthesis_text:")
            print(f"  {enrichment.get('thesis_text', 'N/A')[:300]}")

            print(f"\nthesis_type: {enrichment.get('thesis_type', 'N/A')}")
            print(f"  Esperado: {chunk.get('expected_type')}")
            print(f"  Match: {'OK' if result['type_match'] else 'ERRO'}")

            print(f"\nsynthetic_questions:")
            questions = enrichment.get('synthetic_questions', '')
            for q in questions.split('\n')[:3]:
                if q.strip():
                    print(f"  - {q.strip()}")
        else:
            print(f"\nErro: {enrichment.get('error')}")
            print(f"Resposta raw: {enrichment.get('raw_response', 'N/A')}")

    return result


def run_benchmark(model: str = None, verbose: bool = True):
    """Executa benchmark de enriquecimento."""

    print("=" * 60)
    print("BENCHMARK DE ENRIQUECIMENTO")
    print("=" * 60)

    # Conecta ao vLLM
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model=model or "Qwen/Qwen3-8B-AWQ",
        temperature=0.0,
        max_tokens=1024,
    )

    client = VLLMClient(config=config)

    # Verifica conexao
    models = client.list_models()
    if not models:
        print("ERRO: Nao foi possivel conectar ao vLLM")
        return None

    print(f"\nModelo: {config.model}")
    print(f"Modelos disponiveis: {models}")
    print(f"Chunks de teste: {len(TEST_CHUNKS)}")

    # Executa testes
    results = []
    total_time = 0

    for chunk in TEST_CHUNKS:
        result = test_enrichment(client, chunk, verbose=verbose)
        results.append(result)
        total_time += result["elapsed_seconds"]

    # Estatisticas
    print("\n" + "=" * 60)
    print("ESTATISTICAS")
    print("=" * 60)

    success_count = sum(1 for r in results if r["success"])
    type_match_count = sum(1 for r in results if r.get("type_match"))

    print(f"\nResultados:")
    print(f"  - Total: {len(results)}")
    print(f"  - Sucesso: {success_count}/{len(results)}")
    print(f"  - Tipo correto: {type_match_count}/{len(results)}")
    print(f"  - Tempo total: {total_time:.2f}s")
    print(f"  - Tempo medio: {total_time/len(results):.2f}s por chunk")

    # Detalhes por chunk
    print("\nDetalhes:")
    for r in results:
        status = "OK" if r["success"] else "ERRO"
        type_status = "OK" if r.get("type_match") else "X"
        print(f"  Art. {r['article']:>2}: {status} | tipo={type_status} | {r['elapsed_seconds']:.2f}s")

    client.close()

    return {
        "model": config.model,
        "total_chunks": len(results),
        "success_rate": success_count / len(results),
        "type_accuracy": type_match_count / len(results),
        "total_time": total_time,
        "avg_time": total_time / len(results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Teste de enriquecimento com vLLM")
    parser.add_argument("--model", type=str, help="Modelo a usar (ex: Qwen/Qwen3-4B-AWQ)")
    parser.add_argument("--quiet", action="store_true", help="Modo silencioso")
    args = parser.parse_args()

    result = run_benchmark(model=args.model, verbose=not args.quiet)

    if result:
        # Salva resultado
        output_path = Path(__file__).parent.parent / "data" / "output" / "enrichment_benchmark.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove enrichment details para salvar
        save_result = {k: v for k, v in result.items() if k != "results"}
        save_result["results_summary"] = [
            {
                "article": r["article"],
                "success": r["success"],
                "type_match": r.get("type_match"),
                "elapsed": r["elapsed_seconds"],
            }
            for r in result["results"]
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_result, f, ensure_ascii=False, indent=2)

        print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    main()
