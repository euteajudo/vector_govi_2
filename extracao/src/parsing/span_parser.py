"""
SpanParser - Regex-first parser for Brazilian legal documents.

This parser uses deterministic regex patterns to identify the hierarchical
structure of legal documents. The LLM never discovers structure - it only
classifies or enriches the spans identified here.

Hierarchy (Brazilian Legal Documents):
    CAPÍTULO > Seção > Subseção > Artigo > §/Inciso > Alínea > Item

Regex Patterns:
    - Capítulo:  CAPÍTULO I, CAPÍTULO II...
    - Seção:     Seção I, Seção II...
    - Artigo:    Art. 1º, Art. 2º, Art. 10...
    - Parágrafo: § 1º, § 2º, Parágrafo único
    - Inciso:    I -, II -, III -... (numerais romanos)
    - Alínea:    a), b), c)...
    - Item:      1), 2), 3)... (dentro de alíneas)
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from .span_models import Span, SpanType, ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class ParserConfig:
    """Configuração do parser."""

    # Padrões podem ser customizados por tipo de documento
    include_headers: bool = True
    include_texto_livre: bool = False  # Texto entre estruturas
    normalize_whitespace: bool = True
    extract_titles: bool = True  # Títulos de artigos/seções


class SpanParser:
    """
    Parser determinístico para documentos legais brasileiros.

    Usa regex para identificar a estrutura hierárquica do documento,
    gerando spans com IDs únicos que podem ser referenciados pelo LLM.

    Usage:
        parser = SpanParser()
        doc = parser.parse(markdown_text)

        for span in doc.articles:
            print(f"{span.span_id}: {span.text[:50]}...")

        # Markdown anotado para o LLM
        annotated = doc.to_annotated_markdown()
    """

    # =========================================================================
    # REGEX PATTERNS - Estrutura Legal Brasileira
    # =========================================================================

    # Capítulo: "CAPÍTULO I", "CAPITULO II", "CAP. III"
    PATTERN_CAPITULO = re.compile(
        r'^(?:CAP[ÍI]TULO|CAP\.?)\s+([IVXLC]+)\b[^\n]*',
        re.IGNORECASE | re.MULTILINE
    )

    # Seção: "Seção I", "SEÇÃO II"
    PATTERN_SECAO = re.compile(
        r'^SE[ÇC][ÃA]O\s+([IVXLC]+)\b[^\n]*',
        re.IGNORECASE | re.MULTILINE
    )

    # Subseção: "Subseção I", "SUBSEÇÃO II"
    PATTERN_SUBSECAO = re.compile(
        r'^SUBSE[ÇC][ÃA]O\s+([IVXLC]+)\b[^\n]*',
        re.IGNORECASE | re.MULTILINE
    )

    # Lookahead comum para estruturas de nível superior
    # Usado para parar captura em: Capítulo, Seção, Subseção
    ESTRUTURA_SUPERIOR = r'(?:CAP[ÍI]TULO|SE[ÇC][ÃA]O|SUBSE[ÇC][ÃA]O)'

    # Artigo: "Art. 1º", "Art. 2o", "- Art. 10"
    # Captura: grupo 1 = número, resto = conteúdo (até próximo artigo ou estrutura superior)
    # NÃO para nos incisos/parágrafos - eles serão extraídos depois
    PATTERN_ARTIGO = re.compile(
        rf'^[-*]?\s*Art\.?\s*(\d+)[°ºo]?\s*[-.]?\s*(.+?)(?=\n[-*]?\s*Art\.?\s*\d+[°ºo]?(?:\s|[-.])|^{ESTRUTURA_SUPERIOR}|\Z)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    # Parágrafo: "§ 1º", "§ 2o", "Parágrafo único", "- § 3º"
    # NÃO para nos incisos/alíneas - eles serão extraídos depois
    PATTERN_PARAGRAFO = re.compile(
        rf'^[-*]?\s*(?:§\s*(\d+)[°ºo]?|[Pp]ar[áa]grafo\s+[úu]nico)\s*[-.]?\s*(.+?)(?=\n[-*]?\s*§\s*\d+|\n[-*]?\s*Art\.?\s*\d+[°ºo]?|^{ESTRUTURA_SUPERIOR}|\Z)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    # Numerais romanos: I-XX
    ROMAN_NUMERALS = r'(?:I{1,3}|IV|VI{0,3}|IX|X{1,3}|XI{1,3}|XIV|XV|XVI{0,3}|XIX|XX{0,3})'

    # Inciso: "- I  -", "- II –", "III -", etc.
    # Formato Docling varia: "- I  -  texto" ou "III - texto" (com ou sem bullet)
    # NÃO para nas alíneas - elas serão extraídas depois
    PATTERN_INCISO = re.compile(
        rf'^[-*]?\s*({ROMAN_NUMERALS})\s*[-–—]\s*(.+?)(?=\n[-*]?\s*{ROMAN_NUMERALS}\s*[-–—]|\n[-*]?\s*§\s*\d+|\n[-*]?\s*Art\.?\s*\d+[°ºo]?|\n{ESTRUTURA_SUPERIOR}|\Z)',
        re.MULTILINE | re.DOTALL
    )

    # Alínea: "a)", "b)", "c)"
    PATTERN_ALINEA = re.compile(
        rf'^[-*]?\s*([a-z])\)\s*(.+?)(?=\n[-*]?\s*[a-z]\)|^[-*]?\s*{ROMAN_NUMERALS}\s*[-–]|^[-*]?\s*§|^[-*]?\s*Art\.?\s*\d+|^{ESTRUTURA_SUPERIOR}|\Z)',
        re.MULTILINE | re.DOTALL
    )

    # Item numérico: "1)", "2)", "3)" (dentro de alíneas, raro)
    PATTERN_ITEM = re.compile(
        r'^[-*]?\s*(\d+)\)\s*(.+?)(?=\n[-*]?\s*\d+\)|^[-*]?\s*[a-z]\)|^[-*]?\s*(?:I{1,3}|IV|VI{0,3})\s*[-–]|\Z)',
        re.MULTILINE | re.DOTALL
    )

    def __init__(self, config: Optional[ParserConfig] = None):
        """Inicializa o parser."""
        self.config = config or ParserConfig()

    def parse(self, markdown: str) -> ParsedDocument:
        """
        Parseia markdown e retorna documento com spans identificados.

        Args:
            markdown: Texto em markdown (output do Docling)

        Returns:
            ParsedDocument com todos os spans indexados
        """
        doc = ParsedDocument(source_text=markdown)

        # Normaliza whitespace se configurado
        if self.config.normalize_whitespace:
            markdown = self._normalize_whitespace(markdown)

        # 1. Extrai metadados do cabeçalho
        self._extract_header(markdown, doc)

        # 2. Extrai capítulos
        self._extract_capitulos(markdown, doc)

        # 3. Extrai artigos (principal)
        self._extract_artigos(markdown, doc)

        # 4. Para cada artigo, extrai subdivisões
        for span in list(doc.articles):
            self._extract_article_children(span, doc)

        logger.info(
            f"Parsed document: {len(doc.spans)} spans, "
            f"{len(doc.articles)} articles, "
            f"{len(doc.capitulos)} chapters"
        )

        return doc

    def _normalize_whitespace(self, text: str) -> str:
        """Normaliza espaços em branco."""
        # Remove espaços múltiplos (mantém quebras de linha)
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Remove linhas em branco múltiplas
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _extract_header(self, markdown: str, doc: ParsedDocument):
        """Extrai cabeçalho do documento."""
        if not self.config.include_headers:
            return

        # Encontra primeiro capítulo ou artigo
        first_cap = self.PATTERN_CAPITULO.search(markdown)
        first_art = self.PATTERN_ARTIGO.search(markdown)

        end_pos = len(markdown)
        if first_cap:
            end_pos = min(end_pos, first_cap.start())
        if first_art:
            end_pos = min(end_pos, first_art.start())

        if end_pos > 100:  # Só se tiver conteúdo significativo
            header_text = markdown[:end_pos].strip()
            if header_text:
                span = Span(
                    span_id="HDR-001",
                    span_type=SpanType.HEADER,
                    text=header_text,
                    start_pos=0,
                    end_pos=end_pos,
                )
                doc.add_span(span)

                # Extrai metadados do header
                self._parse_header_metadata(header_text, doc)

    def _parse_header_metadata(self, header: str, doc: ParsedDocument):
        """Extrai metadados do cabeçalho."""
        # Tipo de documento
        tipo_match = re.search(
            r'(LEI|DECRETO|INSTRU[ÇC][ÃA]O NORMATIVA|PORTARIA|RESOLU[ÇC][ÃA]O)',
            header,
            re.IGNORECASE
        )
        if tipo_match:
            doc.metadata["document_type"] = tipo_match.group(1).upper()

        # Número
        num_match = re.search(r'N[°ºo]?\s*(\d+)', header, re.IGNORECASE)
        if num_match:
            doc.metadata["number"] = num_match.group(1)

        # Data
        data_match = re.search(
            r'(\d{1,2})\s+(?:DE\s+)?(\w+)\s+(?:DE\s+)?(\d{4})',
            header,
            re.IGNORECASE
        )
        if data_match:
            doc.metadata["date_raw"] = data_match.group(0)

    def _extract_capitulos(self, markdown: str, doc: ParsedDocument):
        """Extrai capítulos do documento."""
        for match in self.PATTERN_CAPITULO.finditer(markdown):
            numero = match.group(1)
            text = match.group(0).strip()

            # Busca título na próxima linha
            end_pos = match.end()
            next_newline = markdown.find('\n', end_pos)
            if next_newline != -1:
                # Próxima linha pode ser o título
                next_line_end = markdown.find('\n', next_newline + 1)
                if next_line_end == -1:
                    next_line_end = len(markdown)
                next_line = markdown[next_newline:next_line_end].strip()

                # Se não começa com Art. ou outro padrão, é título
                if next_line and not re.match(r'^[-*]?\s*(Art\.|§|[IVXLC]+\s*[-–]|[a-z]\))', next_line, re.IGNORECASE):
                    text = f"{text}\n{next_line}"
                    end_pos = next_line_end

            span = Span(
                span_id=f"CAP-{numero}",
                span_type=SpanType.CAPITULO,
                text=text,
                identifier=numero,
                start_pos=match.start(),
                end_pos=end_pos,
            )
            doc.add_span(span)

    def _extract_artigos(self, markdown: str, doc: ParsedDocument):
        """Extrai artigos do documento."""
        for match in self.PATTERN_ARTIGO.finditer(markdown):
            numero = match.group(1)
            content = match.group(2).strip() if match.group(2) else ""

            # Limpa conteúdo (remove subdivisões que serão extraídas depois)
            content_lines = content.split('\n')
            main_content = []
            for line in content_lines:
                # Para no primeiro inciso, parágrafo, ou alínea
                if re.match(r'^[-*]?\s*(§|[IVXLC]+\s*[-–]|[a-z]\))', line.strip(), re.IGNORECASE):
                    break
                main_content.append(line)

            text = '\n'.join(main_content).strip()

            # Encontra capítulo pai
            parent_id = self._find_parent_capitulo(match.start(), doc)

            span = Span(
                span_id=f"ART-{numero.zfill(3)}",
                span_type=SpanType.ARTIGO,
                text=f"Art. {numero}º {text}" if text else f"Art. {numero}º",
                identifier=numero,
                parent_id=parent_id,
                start_pos=match.start(),
                end_pos=match.end(),
                metadata={"full_match": match.group(0)},
            )
            doc.add_span(span)

    def _find_parent_capitulo(self, position: int, doc: ParsedDocument) -> Optional[str]:
        """Encontra capítulo que contém a posição."""
        parent = None
        for cap in doc.capitulos:
            if cap.start_pos < position:
                parent = cap.span_id
        return parent

    def _extract_article_children(self, article: Span, doc: ParsedDocument):
        """Extrai parágrafos, incisos e alíneas de um artigo."""
        full_text = article.metadata.get("full_match", "")
        if not full_text:
            return

        art_num = article.identifier.zfill(3)

        # Encontra onde começam os parágrafos (se existirem)
        first_par = self.PATTERN_PARAGRAFO.search(full_text)
        if first_par:
            caput_text = full_text[:first_par.start()]
            paragrafos_text = full_text[first_par.start():]
        else:
            caput_text = full_text
            paragrafos_text = ""

        # Extrai incisos do caput (antes dos parágrafos)
        self._extract_incisos(caput_text, art_num, article.span_id, doc)

        # Extrai parágrafos (que por sua vez extraem seus próprios incisos)
        if paragrafos_text:
            self._extract_paragrafos(paragrafos_text, art_num, article.span_id, doc)

    def _extract_paragrafos(
        self,
        text: str,
        art_num: str,
        parent_id: str,
        doc: ParsedDocument
    ):
        """Extrai parágrafos de um artigo."""
        for match in self.PATTERN_PARAGRAFO.finditer(text):
            numero = match.group(1)
            content = match.group(2).strip() if match.group(2) else ""

            # Determina identificador
            if numero:
                identifier = numero
                span_id = f"PAR-{art_num}-{numero}"
            else:
                identifier = "único"
                span_id = f"PAR-{art_num}-UNICO"

            # Limpa conteúdo (para no primeiro inciso ou alínea)
            content_lines = content.split('\n')
            main_content = []
            for line in content_lines:
                if re.match(r'^[-*]?\s*([IVXLC]+\s*[-–]|[a-z]\))', line.strip()):
                    break
                main_content.append(line)

            clean_content = '\n'.join(main_content).strip()

            span = Span(
                span_id=span_id,
                span_type=SpanType.PARAGRAFO,
                text=f"§ {identifier}º {clean_content}" if identifier != "único" else f"Parágrafo único. {clean_content}",
                identifier=identifier,
                parent_id=parent_id,
                metadata={"full_match": match.group(0)},
            )
            doc.add_span(span)

            # Extrai incisos dentro do parágrafo (parent_id vincula ao parágrafo)
            self._extract_incisos(match.group(0), art_num, span_id, doc)

    def _extract_incisos(
        self,
        text: str,
        art_num: str,
        parent_id: str,
        doc: ParsedDocument
    ):
        """Extrai incisos de um artigo ou parágrafo."""
        for match in self.PATTERN_INCISO.finditer(text):
            romano = match.group(1)
            content = match.group(2).strip() if match.group(2) else ""

            # Limpa conteúdo (para na primeira alínea)
            content_lines = content.split('\n')
            main_content = []
            for line in content_lines:
                if re.match(r'^[-*]?\s*[a-z]\)', line.strip()):
                    break
                main_content.append(line)

            clean_content = '\n'.join(main_content).strip()

            # ID base: INC-{art}-{romano}
            base_id = f"INC-{art_num}-{romano}"
            span_id = base_id

            # Se ID já existe, adiciona sufixo para desambiguar
            suffix = 2
            while doc.get_span(span_id) is not None:
                span_id = f"{base_id}_{suffix}"
                suffix += 1

            span = Span(
                span_id=span_id,
                span_type=SpanType.INCISO,
                text=f"{romano} - {clean_content}",
                identifier=romano,
                parent_id=parent_id,
                metadata={"full_match": match.group(0)},
            )
            doc.add_span(span)

            # Extrai alíneas dentro do inciso
            self._extract_alineas(match.group(0), art_num, romano, span_id, doc)

    def _extract_alineas(
        self,
        text: str,
        art_num: str,
        inciso: str,
        parent_id: str,
        doc: ParsedDocument
    ):
        """Extrai alíneas de um inciso."""
        for match in self.PATTERN_ALINEA.finditer(text):
            letra = match.group(1)
            content = match.group(2).strip() if match.group(2) else ""

            span_id = f"ALI-{art_num}-{inciso}-{letra}"

            span = Span(
                span_id=span_id,
                span_type=SpanType.ALINEA,
                text=f"{letra}) {content}",
                identifier=letra,
                parent_id=parent_id,
            )
            doc.add_span(span)

    def parse_to_annotated(self, markdown: str) -> str:
        """
        Parseia e retorna markdown anotado com span_ids.

        Este é o formato que será enviado ao LLM para classificação.
        O LLM só precisa selecionar IDs, não gerar texto.

        Returns:
            Markdown com cada linha prefixada por [SPAN_ID]
        """
        doc = self.parse(markdown)
        return doc.to_annotated_markdown()


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def roman_to_int(roman: str) -> int:
    """Converte numeral romano para inteiro."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
    result = 0
    prev = 0
    for char in reversed(roman.upper()):
        curr = values.get(char, 0)
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result


def int_to_roman(num: int) -> str:
    """Converte inteiro para numeral romano."""
    val = [100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    result = ''
    for i, v in enumerate(val):
        while num >= v:
            result += syms[i]
            num -= v
    return result
