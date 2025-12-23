"""
Validadores para garantir 100% de precisão na extração de documentos legais.

Este módulo contém:
1. DoclingValidator - Valida e conta elementos no Markdown do Docling
2. ExtractionValidator - Compara extração JSON vs Markdown original
3. AutoFixer - Corrige erros conhecidos automaticamente

Uso:
    from extraction_validator import (
        DoclingValidator,
        ExtractionValidator,
        AutoFixer
    )
    
    # 1. Validar Markdown do Docling
    docling_val = DoclingValidator(markdown)
    docling_report = docling_val.validate()
    
    # 2. Extrair com LLM...
    
    # 3. Validar extração
    extraction_val = ExtractionValidator(markdown, json_data)
    extraction_report = extraction_val.validate()
    
    # 4. Corrigir erros automaticamente
    fixer = AutoFixer(json_data, markdown)
    fixed_data = fixer.fix_all()
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# PADRÕES REGEX PARA DOCUMENTOS LEGAIS BRASILEIROS
# =============================================================================

PATTERNS = {
    # Artigos: "Art. 1º", "Art. 10.", "Art. 100"
    "article": r'Art\.?\s*(\d+)[ºº°.]?',
    
    # Parágrafos: "§ 1º", "§ 2º", "§1º", "Parágrafo único"
    "paragraph": r'§\s*(\d+)[ºº°]?|[Pp]ar[áa]grafo\s+[úu]nico',
    
    # Parágrafos em lista numerada do Docling: "12. § 1º"
    "paragraph_numbered": r'^\d+\.\s*§\s*(\d+)[ºº°]?',
    
    # Incisos: "I -", "II -", "III -", "IV-", "V –"
    # Pode vir como bullet "- I -" ou lista "2. IV -"
    "item": r'(?:^|\n)\s*(?:-\s*)?(?:\d+\.\s*)?([IVX]+)\s*[-–—]',
    
    # Alíneas: "a)", "b)", "c)"
    "sub_item": r'(?:^|\n)\s*(?:-\s*)?([a-z])\)',
    
    # Capítulos: "CAPÍTULO I", "CAPITULO II"
    "chapter": r'CAP[ÍI]TULO\s+([IVX]+)',
}


# =============================================================================
# DATACLASSES PARA RELATÓRIOS
# =============================================================================

@dataclass
class ElementCount:
    """Contagem de elementos encontrados."""
    articles: list[str] = field(default_factory=list)
    paragraphs: list[tuple[str, str]] = field(default_factory=list)  # (art, §)
    items: list[tuple[str, str]] = field(default_factory=list)  # (art, inciso)
    sub_items: list[tuple[str, str, str]] = field(default_factory=list)  # (art, inciso, alínea)
    chapters: list[str] = field(default_factory=list)
    
    @property
    def total_articles(self) -> int:
        return len(self.articles)
    
    @property
    def total_paragraphs(self) -> int:
        return len(self.paragraphs)
    
    @property
    def total_items(self) -> int:
        return len(self.items)
    
    @property
    def total_sub_items(self) -> int:
        return len(self.sub_items)
    
    @property
    def total_chapters(self) -> int:
        return len(self.chapters)


@dataclass
class ValidationReport:
    """Relatório de validação."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    expected: Optional[ElementCount] = None
    found: Optional[ElementCount] = None
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def summary(self) -> str:
        lines = [
            f"Valid: {self.is_valid}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        if self.expected and self.found:
            lines.extend([
                f"Articles: {self.found.total_articles}/{self.expected.total_articles}",
                f"Paragraphs: {self.found.total_paragraphs}/{self.expected.total_paragraphs}",
                f"Items: {self.found.total_items}/{self.expected.total_items}",
                f"Chapters: {self.found.total_chapters}/{self.expected.total_chapters}",
            ])
        if self.errors:
            lines.append("--- ERRORS ---")
            lines.extend(self.errors[:10])  # Primeiros 10
        if self.warnings:
            lines.append("--- WARNINGS ---")
            lines.extend(self.warnings[:10])
        return "\n".join(lines)


# =============================================================================
# VALIDADOR DO MARKDOWN DO DOCLING
# =============================================================================

class DoclingValidator:
    """
    Valida e conta elementos no Markdown gerado pelo Docling.
    
    Uso:
        validator = DoclingValidator(markdown_text)
        report = validator.validate()
        counts = validator.count_elements()
    """
    
    def __init__(self, markdown: str):
        self.markdown = markdown
        self._counts: Optional[ElementCount] = None
    
    def count_elements(self) -> ElementCount:
        """Conta todos os elementos legais no Markdown."""
        if self._counts:
            return self._counts
        
        counts = ElementCount()
        
        # Contar capítulos
        for match in re.finditer(PATTERNS["chapter"], self.markdown, re.IGNORECASE):
            counts.chapters.append(match.group(1))
        
        # Contar artigos
        for match in re.finditer(PATTERNS["article"], self.markdown):
            counts.articles.append(match.group(1))
        
        # Contar parágrafos (incluindo formato lista numerada do Docling)
        # IMPORTANTE: Distinguir parágrafos próprios de referências a outras leis
        current_article = None
        for line in self.markdown.split('\n'):
            # Detectar artigo atual
            art_match = re.search(PATTERNS["article"], line)
            if art_match:
                current_article = art_match.group(1)
            
            # Detectar parágrafos próprios do documento
            # Padrão: linha COMEÇA com § ou com "12. §" (lista numerada Docling)
            # Isso distingue de referências que aparecem NO MEIO do texto
            
            line_stripped = line.strip()
            
            # § 1º no início da linha
            para_match = re.match(r'^§\s*(\d+)[ºº°]?', line_stripped)
            
            # Lista numerada do Docling: "12. § 1º"
            if not para_match:
                para_match = re.match(r'^\d+\.\s*§\s*(\d+)[ºº°]?', line_stripped)
            
            # Parágrafo único
            if not para_match:
                if re.match(r'^[Pp]ar[áa]grafo\s+[úu]nico', line_stripped):
                    if current_article:
                        counts.paragraphs.append((current_article, "unico"))
                    continue
            
            if para_match and current_article:
                para_id = para_match.group(1)
                counts.paragraphs.append((current_article, para_id))
        
        # Contar incisos
        current_article = None
        for line in self.markdown.split('\n'):
            art_match = re.search(PATTERNS["article"], line)
            if art_match:
                current_article = art_match.group(1)
            
            item_match = re.search(PATTERNS["item"], line)
            if item_match and current_article:
                counts.items.append((current_article, item_match.group(1)))
        
        # Contar alíneas
        current_article = None
        current_item = None
        for line in self.markdown.split('\n'):
            art_match = re.search(PATTERNS["article"], line)
            if art_match:
                current_article = art_match.group(1)
                current_item = None
            
            item_match = re.search(PATTERNS["item"], line)
            if item_match:
                current_item = item_match.group(1)
            
            sub_match = re.search(PATTERNS["sub_item"], line)
            if sub_match and current_article and current_item:
                counts.sub_items.append((current_article, current_item, sub_match.group(1)))
        
        self._counts = counts
        return counts
    
    def validate(self) -> ValidationReport:
        """Valida o Markdown do Docling."""
        report = ValidationReport(is_valid=True)
        counts = self.count_elements()
        report.expected = counts
        
        # Verificar se há artigos
        if counts.total_articles == 0:
            report.add_error("Nenhum artigo encontrado no Markdown")
        
        # Verificar sequência de artigos
        try:
            art_nums = sorted([int(a) for a in counts.articles])
            expected_seq = list(range(1, max(art_nums) + 1))
            missing = set(expected_seq) - set(art_nums)
            if missing:
                report.add_warning(f"Artigos possivelmente faltando no MD: {sorted(missing)}")
        except (ValueError, TypeError):
            pass
        
        # Verificar problemas conhecidos do Docling
        if "<!-- image -->" in self.markdown:
            report.add_warning("Markdown contém marcadores de imagem que podem afetar extração")
        
        # Verificar lista numerada com § (problema comum)
        if re.search(r'^\d+\.\s*§', self.markdown, re.MULTILINE):
            report.add_warning("Parágrafos em formato lista numerada detectados (ex: '12. § 1º')")
        
        # Verificar Art. como item de lista (erro do Docling)
        if re.search(r'^-\s*Art\.', self.markdown, re.MULTILINE):
            report.add_error("Artigo detectado como item de lista (- Art.) - erro do Docling")
        
        return report
    
    def get_article_paragraphs_map(self) -> dict[str, list[str]]:
        """Retorna mapa de artigo -> lista de parágrafos."""
        counts = self.count_elements()
        result = {}
        for art, para in counts.paragraphs:
            if art not in result:
                result[art] = []
            result[art].append(para)
        return result
    
    def get_article_items_map(self) -> dict[str, list[str]]:
        """Retorna mapa de artigo -> lista de incisos."""
        counts = self.count_elements()
        result = {}
        for art, item in counts.items:
            if art not in result:
                result[art] = []
            result[art].append(item)
        return result


# =============================================================================
# VALIDADOR DA EXTRAÇÃO JSON
# =============================================================================

class ExtractionValidator:
    """
    Compara extração JSON com Markdown original.
    
    Uso:
        validator = ExtractionValidator(markdown, json_data)
        report = validator.validate()
    """
    
    def __init__(self, markdown: str, json_data: dict):
        self.markdown = markdown
        self.json_data = json_data
        self.docling_validator = DoclingValidator(markdown)
    
    def count_json_elements(self) -> ElementCount:
        """Conta elementos no JSON extraído."""
        counts = ElementCount()
        
        for chapter in self.json_data.get("chapters", []):
            if chapter.get("chapter_number"):
                counts.chapters.append(chapter["chapter_number"])
            
            for article in chapter.get("articles", []):
                art_num = article.get("article_number", "")
                counts.articles.append(art_num)
                
                for para in article.get("paragraphs", []):
                    counts.paragraphs.append((art_num, para.get("paragraph_identifier", "")))
                
                for item in article.get("items", []):
                    item_id = item.get("item_identifier", "")
                    counts.items.append((art_num, item_id))
                    
                    for sub in item.get("sub_items", []):
                        counts.sub_items.append((art_num, item_id, sub.get("item_identifier", "")))
        
        return counts
    
    def validate(self) -> ValidationReport:
        """Valida extração comparando com Markdown original."""
        report = ValidationReport(is_valid=True)
        
        expected = self.docling_validator.count_elements()
        found = self.count_json_elements()
        
        report.expected = expected
        report.found = found
        
        # Comparar artigos
        if found.total_articles != expected.total_articles:
            report.add_error(
                f"Artigos: esperado {expected.total_articles}, extraído {found.total_articles}"
            )
            
            # Detalhar quais faltam
            expected_arts = set(expected.articles)
            found_arts = set(found.articles)
            missing = expected_arts - found_arts
            extra = found_arts - expected_arts
            
            if missing:
                report.add_error(f"Artigos faltando: {sorted(missing, key=lambda x: int(x) if x.isdigit() else 0)}")
            if extra:
                report.add_warning(f"Artigos extras: {sorted(extra)}")
        
        # Comparar parágrafos
        if found.total_paragraphs != expected.total_paragraphs:
            report.add_error(
                f"Parágrafos: esperado {expected.total_paragraphs}, extraído {found.total_paragraphs}"
            )
            
            # Detalhar por artigo
            expected_map = self.docling_validator.get_article_paragraphs_map()
            found_map = {}
            for art, para in found.paragraphs:
                if art not in found_map:
                    found_map[art] = []
                found_map[art].append(para)
            
            for art, expected_paras in expected_map.items():
                found_paras = found_map.get(art, [])
                if len(found_paras) != len(expected_paras):
                    report.add_error(
                        f"Art. {art}: esperado {len(expected_paras)} §, extraído {len(found_paras)}"
                    )
        
        # Comparar incisos (warning, não erro crítico)
        if found.total_items != expected.total_items:
            diff = abs(found.total_items - expected.total_items)
            if diff > 2:  # Tolerância de 2
                report.add_warning(
                    f"Incisos: esperado ~{expected.total_items}, extraído {found.total_items}"
                )
        
        # Comparar capítulos
        if found.total_chapters != expected.total_chapters:
            report.add_warning(
                f"Capítulos: esperado {expected.total_chapters}, extraído {found.total_chapters}"
            )
        
        return report


# =============================================================================
# AUTO-FIXER PARA ERROS CONHECIDOS
# =============================================================================

class AutoFixer:
    """
    Corrige automaticamente erros conhecidos na extração.
    
    Erros corrigidos:
    1. Parágrafos classificados como incisos
    2. Capítulos vazios
    3. Incisos duplicados
    4. Sub-items fantasmas
    """
    
    def __init__(self, json_data: dict, markdown: str):
        self.data = json_data
        self.markdown = markdown
        self.docling_validator = DoclingValidator(markdown)
        self.fixes_applied: list[str] = []
    
    def fix_all(self) -> dict:
        """Aplica todas as correções."""
        self._fix_paragraphs_as_items()
        self._fix_empty_chapters()
        self._fix_phantom_sub_items()
        self._fix_duplicate_content()
        return self.data
    
    def _fix_paragraphs_as_items(self):
        """
        Corrige incisos que são na verdade parágrafos.
        
        Detecta quando:
        1. O Markdown original tem § mas o JSON tem inciso
        2. O texto do inciso começa com padrão típico de parágrafo
        """
        expected_map = self.docling_validator.get_article_paragraphs_map()
        
        for chapter in self.data.get("chapters", []):
            for article in chapter.get("articles", []):
                art_num = article.get("article_number", "")
                
                # Verificar se este artigo deveria ter parágrafos
                expected_paras = expected_map.get(art_num, [])
                current_paras = article.get("paragraphs", [])
                
                if len(expected_paras) > len(current_paras):
                    # Faltam parágrafos - verificar se estão nos items
                    items = article.get("items", [])
                    items_to_remove = []
                    
                    for i, item in enumerate(items):
                        desc = item.get("description", "")
                        
                        # Padrões que indicam que é parágrafo, não inciso
                        paragraph_indicators = [
                            # Texto começa com referência a órgãos/entidades
                            desc.startswith("Os órgãos"),
                            desc.startswith("As entidades"),
                            desc.startswith("As informações"),
                            desc.startswith("A definição"),
                            desc.startswith("O disposto"),
                            desc.startswith("Em caso de"),
                            desc.startswith("Caso,"),
                            desc.startswith("Em todos os casos"),
                            # Identificador numérico (1, 2) em vez de romano (I, II)
                            item.get("item_identifier", "").isdigit(),
                        ]
                        
                        if any(paragraph_indicators):
                            items_to_remove.append(i)
                            current_paras.append({
                                "paragraph_identifier": str(len(current_paras) + 1),
                                "content": desc
                            })
                            self.fixes_applied.append(
                                f"Art. {art_num}: Movido inciso '{item.get('item_identifier')}' para parágrafo"
                            )
                    
                    # Remover items movidos (de trás pra frente)
                    for i in reversed(items_to_remove):
                        items.pop(i)
                    
                    article["paragraphs"] = current_paras
    
    def _fix_empty_chapters(self):
        """Remove capítulos sem artigos."""
        chapters = self.data.get("chapters", [])
        original_count = len(chapters)
        
        self.data["chapters"] = [
            ch for ch in chapters
            if ch.get("articles") and len(ch["articles"]) > 0
        ]
        
        removed = original_count - len(self.data["chapters"])
        if removed > 0:
            self.fixes_applied.append(f"Removidos {removed} capítulos vazios")
    
    def _fix_phantom_sub_items(self):
        """Remove sub_items que duplicam o texto do item pai."""
        for chapter in self.data.get("chapters", []):
            for article in chapter.get("articles", []):
                for item in article.get("items", []):
                    sub_items = item.get("sub_items", [])
                    
                    if len(sub_items) == 1:
                        # Se só tem 1 sub_item e o texto é similar ao pai, remover
                        sub_desc = sub_items[0].get("description", "")[:50]
                        item_desc = item.get("description", "")
                        
                        if sub_desc in item_desc:
                            item["sub_items"] = []
                            self.fixes_applied.append(
                                f"Art. {article.get('article_number')}, "
                                f"Inciso {item.get('item_identifier')}: "
                                f"Removido sub_item duplicado"
                            )
    
    def _fix_duplicate_content(self):
        """Remove duplicações de conteúdo entre artigos."""
        seen_contents = {}
        
        for chapter in self.data.get("chapters", []):
            for article in chapter.get("articles", []):
                content = article.get("content", "")[:100]  # Primeiros 100 chars
                art_num = article.get("article_number", "")
                
                if content in seen_contents:
                    # Conteúdo duplicado - verificar se é o mesmo artigo
                    other_art = seen_contents[content]
                    if other_art != art_num:
                        self.fixes_applied.append(
                            f"AVISO: Art. {art_num} tem conteúdo similar ao Art. {other_art}"
                        )
                else:
                    seen_contents[content] = art_num
    
    def get_report(self) -> str:
        """Retorna relatório das correções aplicadas."""
        if not self.fixes_applied:
            return "Nenhuma correção necessária"
        
        lines = [f"Correções aplicadas: {len(self.fixes_applied)}"]
        lines.extend(self.fixes_applied)
        return "\n".join(lines)


# =============================================================================
# FUNÇÃO DE CONVENIÊNCIA
# =============================================================================

def validate_and_fix(markdown: str, json_data: dict) -> tuple[dict, ValidationReport, str]:
    """
    Função de conveniência que valida e corrige em uma chamada.
    
    Returns:
        (fixed_data, validation_report, fix_report)
    """
    # 1. Validar Markdown original
    docling_val = DoclingValidator(markdown)
    docling_report = docling_val.validate()
    
    # 2. Corrigir JSON
    fixer = AutoFixer(json_data, markdown)
    fixed_data = fixer.fix_all()
    fix_report = fixer.get_report()
    
    # 3. Validar extração corrigida
    extraction_val = ExtractionValidator(markdown, fixed_data)
    final_report = extraction_val.validate()
    
    # Mesclar warnings do Docling
    for w in docling_report.warnings:
        final_report.add_warning(f"[Docling] {w}")
    for e in docling_report.errors:
        final_report.add_warning(f"[Docling] {e}")
    
    return fixed_data, final_report, fix_report


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Exemplo com Markdown simulado
    test_markdown = """
CAPÍTULO I

DISPOSIÇÕES GERAIS

Art. 1º Esta norma estabelece regras.

Art. 2º Considera-se:

- I - primeiro item;
- II - segundo item.

§ 1º Primeiro parágrafo.

§ 2º Segundo parágrafo.

Art. 3º Outro artigo.

CAPÍTULO II

DISPOSIÇÕES FINAIS

Art. 4º Entra em vigor na data.

Parágrafo único. Revogam-se disposições.
"""
    
    # Simular extração com erro (§ como inciso)
    test_json = {
        "chapters": [
            {
                "chapter_number": "I",
                "title": "DISPOSIÇÕES GERAIS",
                "articles": [
                    {
                        "article_number": "1",
                        "content": "Esta norma estabelece regras.",
                        "items": [],
                        "paragraphs": []
                    },
                    {
                        "article_number": "2",
                        "content": "Considera-se:",
                        "items": [
                            {"item_identifier": "I", "description": "primeiro item;", "sub_items": []},
                            {"item_identifier": "II", "description": "segundo item.", "sub_items": []},
                            {"item_identifier": "1", "description": "Os órgãos devem seguir.", "sub_items": []},
                            {"item_identifier": "2", "description": "As entidades também.", "sub_items": []}
                        ],
                        "paragraphs": []
                    },
                    {
                        "article_number": "3",
                        "content": "Outro artigo.",
                        "items": [],
                        "paragraphs": []
                    }
                ]
            },
            {
                "chapter_number": "II",
                "title": "DISPOSIÇÕES FINAIS",
                "articles": [
                    {
                        "article_number": "4",
                        "content": "Entra em vigor na data.",
                        "items": [],
                        "paragraphs": [
                            {"paragraph_identifier": "unico", "content": "Revogam-se disposições."}
                        ]
                    }
                ]
            }
        ]
    }
    
    print("=" * 60)
    print("TESTE DO VALIDADOR")
    print("=" * 60)
    
    # Validar Markdown
    print("\n--- Validação Docling ---")
    docling_val = DoclingValidator(test_markdown)
    docling_report = docling_val.validate()
    print(docling_report.summary())
    
    # Validar extração (antes da correção)
    print("\n--- Validação Extração (ANTES) ---")
    extraction_val = ExtractionValidator(test_markdown, test_json)
    before_report = extraction_val.validate()
    print(before_report.summary())
    
    # Aplicar correções
    print("\n--- Aplicando Correções ---")
    fixer = AutoFixer(test_json, test_markdown)
    fixed_data = fixer.fix_all()
    print(fixer.get_report())
    
    # Validar após correção
    print("\n--- Validação Extração (DEPOIS) ---")
    extraction_val2 = ExtractionValidator(test_markdown, fixed_data)
    after_report = extraction_val2.validate()
    print(after_report.summary())
