"""
Script para converter PDF para Markdown com Docling e processar com Pipeline v3.

Uso:
    python scripts/convert_and_process.py data/L14133.pdf
    python scripts/convert_and_process.py data/L14133.pdf --no-llm
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path = None) -> Path:
    """
    Converte PDF para Markdown usando Docling.

    Args:
        pdf_path: Caminho para o PDF
        output_dir: Diretório de saída (default: data/output)

    Returns:
        Caminho para o arquivo markdown gerado
    """
    from docling.document_converter import DocumentConverter

    print(f"\n[DOCLING] Convertendo PDF para Markdown...")
    print(f"      Arquivo: {pdf_path.name}")
    print(f"      Tamanho: {pdf_path.stat().st_size / 1024:.1f} KB")

    start = time.time()

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    # Exporta para markdown
    markdown = result.document.export_to_markdown()

    # Define output path
    if output_dir is None:
        output_dir = pdf_path.parent / "output"
    output_dir.mkdir(exist_ok=True)

    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(markdown, encoding="utf-8")

    elapsed = time.time() - start
    print(f"      Tempo: {elapsed:.2f}s")
    print(f"      Caracteres: {len(markdown):,}")
    print(f"      Salvo em: {md_path}")

    return md_path


def extract_document_info(pdf_name: str) -> dict:
    """Extrai informações do documento a partir do nome do arquivo."""

    # Lei 14.133
    if "14133" in pdf_name or "L14133" in pdf_name:
        return {
            "document_id": "LEI-14133-2021",
            "tipo_documento": "LEI",
            "numero": "14133",
            "ano": 2021,
        }

    # IN 65
    if "65" in pdf_name and ("IN" in pdf_name.upper() or "INSTRUÇÃO" in pdf_name.upper()):
        return {
            "document_id": "IN-65-2021",
            "tipo_documento": "INSTRUCAO NORMATIVA",
            "numero": "65",
            "ano": 2021,
        }

    # IN 58
    if "58" in pdf_name:
        return {
            "document_id": "IN-58-2022",
            "tipo_documento": "INSTRUCAO NORMATIVA",
            "numero": "58",
            "ano": 2022,
        }

    # Default
    return {
        "document_id": pdf_name.replace(".pdf", ""),
        "tipo_documento": "DOCUMENTO",
        "numero": "",
        "ano": 2024,
    }


def main():
    parser = argparse.ArgumentParser(description="Converte PDF e processa com Pipeline v3")
    parser.add_argument("pdf_path", type=str, help="Caminho para o PDF")
    parser.add_argument("--no-llm", action="store_true", help="Não usar LLM")
    parser.add_argument("--no-embeddings", action="store_true", help="Não gerar embeddings")
    parser.add_argument("--no-milvus", action="store_true", help="Não inserir no Milvus")
    parser.add_argument("--skip-conversion", action="store_true", help="Pular conversão (usa MD existente)")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"ERRO: Arquivo não encontrado: {pdf_path}")
        return 1

    print("=" * 70)
    print("PIPELINE COMPLETO: PDF -> Markdown -> Chunks -> Milvus")
    print("=" * 70)

    total_start = time.time()

    # 1. Converte PDF para Markdown
    output_dir = pdf_path.parent / "output"
    md_path = output_dir / f"{pdf_path.stem}.md"

    if args.skip_conversion and md_path.exists():
        print(f"\n[DOCLING] Usando markdown existente: {md_path}")
    else:
        md_path = convert_pdf_to_markdown(pdf_path, output_dir)

    # 2. Extrai informações do documento
    doc_info = extract_document_info(pdf_path.name)
    print(f"\n[INFO] Documento identificado:")
    print(f"      ID: {doc_info['document_id']}")
    print(f"      Tipo: {doc_info['tipo_documento']}")
    print(f"      Número: {doc_info['numero']}")
    print(f"      Ano: {doc_info['ano']}")

    # 3. Executa Pipeline v3
    print("\n" + "=" * 70)

    from run_pipeline_v3 import run_pipeline_v3

    chunks = run_pipeline_v3(
        markdown_path=md_path,
        document_id=doc_info["document_id"],
        tipo_documento=doc_info["tipo_documento"],
        numero=doc_info["numero"],
        ano=doc_info["ano"],
        use_llm=not args.no_llm,
        use_embeddings=not args.no_embeddings,
        use_milvus=not args.no_milvus,
    )

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"TEMPO TOTAL: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
