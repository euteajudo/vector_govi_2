"""
Script para dividir documentos grandes em partes e processar separadamente.

Divide o documento por capítulos ou número de artigos e processa cada parte
com o pipeline completo (incluindo enriquecimento).

Uso:
    # Dividir por capítulos (cada capítulo = 1 parte)
    python scripts/split_and_process.py --input data/output/L14133.md --split-by chapters

    # Dividir em N partes iguais
    python scripts/split_and_process.py --input data/output/L14133.md --split-by articles --parts 3

    # Apenas dividir (sem processar)
    python scripts/split_and_process.py --input data/output/L14133.md --parts 3 --split-only

    # Processar uma parte específica
    python scripts/split_and_process.py --input data/output/L14133.md --process-part 2
"""

import sys
import re
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsing import SpanParser


@dataclass
class DocumentPart:
    """Uma parte do documento."""
    part_number: int
    total_parts: int
    start_article: int
    end_article: int
    article_count: int
    markdown: str
    chapters: list[str]


def analyze_document(markdown_path: Path) -> dict:
    """Analisa o documento e retorna estatísticas."""
    markdown = markdown_path.read_text(encoding="utf-8")

    parser = SpanParser()
    parsed = parser.parse(markdown)

    # Conta capítulos
    chapters = []
    current_chapter = None
    articles_per_chapter = {}

    for span in parsed.spans:
        if span.span_type.name == "CHAPTER":
            current_chapter = span.text[:50]
            chapters.append(current_chapter)
            articles_per_chapter[current_chapter] = 0
        elif span.span_type.name == "ARTICLE" and current_chapter:
            articles_per_chapter[current_chapter] += 1

    return {
        "total_spans": len(parsed.spans),
        "total_articles": len(parsed.articles),
        "total_chapters": len(chapters),
        "chapters": chapters,
        "articles_per_chapter": articles_per_chapter,
        "markdown_size": len(markdown),
    }


def split_by_chapters(markdown_path: Path) -> list[DocumentPart]:
    """Divide o documento por capítulos."""
    markdown = markdown_path.read_text(encoding="utf-8")

    # Encontra posições dos capítulos
    chapter_pattern = r'^(#{1,2}\s*(?:CAPÍTULO|CAPITULO|CAP[ÍI]TULO)\s+[IVXLC]+.*?)$'
    matches = list(re.finditer(chapter_pattern, markdown, re.MULTILINE | re.IGNORECASE))

    if not matches:
        print("Nenhum capítulo encontrado. Usando documento inteiro.")
        return [DocumentPart(
            part_number=1,
            total_parts=1,
            start_article=1,
            end_article=999,
            article_count=0,
            markdown=markdown,
            chapters=["Documento completo"],
        )]

    parts = []
    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)

        chapter_markdown = markdown[start_pos:end_pos]
        chapter_title = match.group(1).strip()

        # Conta artigos neste capítulo
        article_matches = re.findall(r'Art\.\s*(\d+)', chapter_markdown)
        article_count = len(article_matches)

        if article_count > 0:
            start_art = int(article_matches[0])
            end_art = int(article_matches[-1])
        else:
            start_art = 0
            end_art = 0

        parts.append(DocumentPart(
            part_number=i + 1,
            total_parts=len(matches),
            start_article=start_art,
            end_article=end_art,
            article_count=article_count,
            markdown=chapter_markdown,
            chapters=[chapter_title],
        ))

    return parts


def split_by_articles(markdown_path: Path, num_parts: int) -> list[DocumentPart]:
    """Divide o documento em N partes com número similar de artigos."""
    markdown = markdown_path.read_text(encoding="utf-8")

    # Encontra todos os artigos
    article_pattern = r'^(Art\.\s*\d+[°ºo]?\s*)'
    matches = list(re.finditer(article_pattern, markdown, re.MULTILINE))

    if not matches:
        print("Nenhum artigo encontrado!")
        return []

    total_articles = len(matches)
    articles_per_part = total_articles // num_parts

    print(f"Total de artigos: {total_articles}")
    print(f"Artigos por parte: ~{articles_per_part}")

    parts = []

    for i in range(num_parts):
        start_idx = i * articles_per_part
        if i == num_parts - 1:
            # Última parte pega todos os restantes
            end_idx = total_articles
        else:
            end_idx = (i + 1) * articles_per_part

        start_pos = matches[start_idx].start()
        end_pos = matches[end_idx].start() if end_idx < total_articles else len(markdown)

        # Pega os artigos desta parte
        part_articles = [m for m in matches[start_idx:end_idx]]
        article_numbers = [re.search(r'(\d+)', m.group()).group(1) for m in part_articles]

        # Extrai capítulos desta parte
        part_markdown = markdown[start_pos:end_pos]
        chapter_matches = re.findall(
            r'^#{1,2}\s*((?:CAPÍTULO|CAPITULO)\s+[IVXLC]+.*?)$',
            part_markdown, re.MULTILINE | re.IGNORECASE
        )

        parts.append(DocumentPart(
            part_number=i + 1,
            total_parts=num_parts,
            start_article=int(article_numbers[0]) if article_numbers else 0,
            end_article=int(article_numbers[-1]) if article_numbers else 0,
            article_count=len(article_numbers),
            markdown=part_markdown,
            chapters=chapter_matches or [f"Parte {i+1}"],
        ))

    return parts


def save_parts(parts: list[DocumentPart], base_path: Path) -> list[Path]:
    """Salva as partes como arquivos separados."""
    output_dir = base_path.parent
    base_name = base_path.stem

    saved_paths = []
    for part in parts:
        part_name = f"{base_name}_part{part.part_number}of{part.total_parts}.md"
        part_path = output_dir / part_name

        # Adiciona header com informações da parte
        header = f"""<!--
PARTE {part.part_number} de {part.total_parts}
Artigos: {part.start_article} a {part.end_article} ({part.article_count} artigos)
Capítulos: {', '.join(part.chapters[:3])}{'...' if len(part.chapters) > 3 else ''}
-->

"""
        part_path.write_text(header + part.markdown, encoding="utf-8")
        saved_paths.append(part_path)

        print(f"  Parte {part.part_number}: {part_path.name}")
        print(f"    Artigos {part.start_article}-{part.end_article} ({part.article_count} artigos)")

    return saved_paths


def process_part(
    part_path: Path,
    document_id: str,
    part_number: int,
    total_parts: int,
    tipo: str = "LEI",
    numero: str = "14133",
    ano: int = 2021,
) -> bool:
    """Processa uma parte com o pipeline v3."""

    # Ajusta document_id para incluir parte
    part_doc_id = f"{document_id}-P{part_number}"

    cmd = [
        sys.executable,
        "scripts/run_pipeline_v3.py",
        "--input", str(part_path),
        "--document-id", part_doc_id,
        "--tipo", tipo,
        "--numero", numero,
        "--ano", str(ano),
    ]

    print(f"\n{'='*70}")
    print(f"PROCESSANDO PARTE {part_number}/{total_parts}")
    print(f"{'='*70}")
    print(f"Arquivo: {part_path.name}")
    print(f"Document ID: {part_doc_id}")
    print(f"Comando: {' '.join(cmd)}")
    print("="*70 + "\n")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Divide e processa documentos grandes"
    )
    parser.add_argument("--input", "-i", required=True, help="Arquivo markdown")
    parser.add_argument(
        "--split-by",
        choices=["chapters", "articles"],
        default="articles",
        help="Como dividir: por capítulos ou número de artigos"
    )
    parser.add_argument("--parts", "-n", type=int, default=3, help="Número de partes (para split-by=articles)")
    parser.add_argument("--split-only", action="store_true", help="Apenas dividir, não processar")
    parser.add_argument("--process-part", type=int, help="Processar apenas uma parte específica")
    parser.add_argument("--analyze-only", action="store_true", help="Apenas analisar o documento")

    # Metadados do documento
    parser.add_argument("--document-id", default="LEI-14133-2021")
    parser.add_argument("--tipo", default="LEI")
    parser.add_argument("--numero", default="14133")
    parser.add_argument("--ano", type=int, default=2021)

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Arquivo não encontrado: {input_path}")
        return

    # Análise
    if args.analyze_only:
        print("="*70)
        print("ANÁLISE DO DOCUMENTO")
        print("="*70)

        stats = analyze_document(input_path)
        print(f"Arquivo: {input_path.name}")
        print(f"Tamanho: {stats['markdown_size']:,} caracteres")
        print(f"Spans: {stats['total_spans']}")
        print(f"Artigos: {stats['total_articles']}")
        print(f"Capítulos: {stats['total_chapters']}")

        print("\nArtigos por capítulo:")
        for chapter, count in stats['articles_per_chapter'].items():
            print(f"  {chapter}: {count} artigos")

        return

    # Divisão
    print("="*70)
    print("DIVIDINDO DOCUMENTO")
    print("="*70)

    if args.split_by == "chapters":
        parts = split_by_chapters(input_path)
    else:
        parts = split_by_articles(input_path, args.parts)

    print(f"\nDocumento dividido em {len(parts)} partes:")

    # Salva as partes
    saved_paths = save_parts(parts, input_path)

    if args.split_only:
        print("\n[SPLIT-ONLY] Partes salvas. Use --process-part N para processar.")
        return

    # Processamento
    if args.process_part:
        # Processa apenas uma parte
        part_idx = args.process_part - 1
        if part_idx < 0 or part_idx >= len(saved_paths):
            print(f"Parte {args.process_part} não existe. Partes: 1-{len(saved_paths)}")
            return

        success = process_part(
            saved_paths[part_idx],
            args.document_id,
            args.process_part,
            len(saved_paths),
            args.tipo,
            args.numero,
            args.ano,
        )

        print(f"\nParte {args.process_part}: {'OK' if success else 'ERRO'}")
    else:
        # Processa todas as partes sequencialmente
        print("\n" + "="*70)
        print("PROCESSANDO TODAS AS PARTES")
        print("="*70)

        results = []
        for i, part_path in enumerate(saved_paths):
            success = process_part(
                part_path,
                args.document_id,
                i + 1,
                len(saved_paths),
                args.tipo,
                args.numero,
                args.ano,
            )
            results.append(success)

        # Resumo
        print("\n" + "="*70)
        print("RESUMO")
        print("="*70)
        for i, success in enumerate(results):
            print(f"  Parte {i+1}: {'OK' if success else 'ERRO'}")

        total_ok = sum(results)
        print(f"\nTotal: {total_ok}/{len(results)} partes processadas com sucesso")


if __name__ == "__main__":
    main()
