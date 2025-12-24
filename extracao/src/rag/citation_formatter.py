"""
Formatador de citacoes legais para frontend.

Gera citacoes formatadas a partir dos metadados do Milvus.

Exemplos de saida:
- "Lei 14.133/2021, Art. 14"
- "Lei 14.133/2021, Art. 14, Par. 5"
- "Lei 14.133/2021, Art. 14, inciso II"
- "Lei 14.133/2021, Art. 14, inciso II, alinea 'a'"
- "IN 65/2021, Art. 3, Par. unico"

Uso:
    from rag.citation_formatter import CitationFormatter, format_citation

    # Formato simples
    citation = format_citation(
        tipo_documento="LEI",
        numero="14133",
        ano=2021,
        article_number="14",
        device_type="paragraph",
        span_id="PAR-014-5"
    )
    # -> "Lei 14.133/2021, Art. 14, Par. 5"

    # Com classe
    formatter = CitationFormatter()
    citation = formatter.format_from_chunk(chunk_data)
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class Citation:
    """Citacao formatada com metadados."""

    # Texto formatado da citacao
    text: str

    # Texto curto (para tags/badges)
    short: str

    # Componentes individuais
    document_type: str
    document_number: str
    year: int
    article: str
    device: Optional[str] = None
    device_number: Optional[str] = None

    # IDs para linking
    chunk_id: str = ""
    document_id: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "short": self.short,
            "document_type": self.document_type,
            "document_number": self.document_number,
            "year": self.year,
            "article": self.article,
            "device": self.device,
            "device_number": self.device_number,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
        }


class CitationFormatter:
    """Formatador de citacoes legais."""

    # Mapeamento de tipos de documento para nome legivel
    DOC_TYPE_NAMES = {
        "LEI": "Lei",
        "DECRETO": "Decreto",
        "IN": "IN",
        "INSTRUCAO NORMATIVA": "IN",
        "PORTARIA": "Portaria",
        "RESOLUCAO": "Resolucao",
        "MP": "MP",
        "MEDIDA PROVISORIA": "MP",
        "LC": "LC",
        "LEI COMPLEMENTAR": "LC",
        "EC": "EC",
        "EMENDA CONSTITUCIONAL": "EC",
    }

    # Mapeamento de device_type para nome legivel
    DEVICE_NAMES = {
        "article": "Art.",
        "paragraph": "Par.",
        "inciso": "inciso",
        "alinea": "alinea",
        "item": "item",
    }

    def __init__(self, use_symbols: bool = True):
        """
        Args:
            use_symbols: Se True, usa simbolos (Par. -> SS, ordinal -> o)
        """
        self.use_symbols = use_symbols

    def format_from_chunk(self, chunk: dict) -> Citation:
        """
        Formata citacao a partir de dados de um chunk do Milvus.

        Args:
            chunk: Dict com campos do chunk (tipo_documento, numero, ano, etc)

        Returns:
            Citation formatada
        """
        return self.format(
            tipo_documento=chunk.get("tipo_documento", ""),
            numero=chunk.get("numero", ""),
            ano=chunk.get("ano", 0),
            article_number=chunk.get("article_number", ""),
            device_type=chunk.get("device_type", "article"),
            span_id=chunk.get("span_id", ""),
            chunk_id=chunk.get("chunk_id", ""),
            document_id=chunk.get("document_id", ""),
        )

    def format(
        self,
        tipo_documento: str,
        numero: str,
        ano: int,
        article_number: str,
        device_type: str = "article",
        span_id: str = "",
        chunk_id: str = "",
        document_id: str = "",
    ) -> Citation:
        """
        Formata citacao completa.

        Args:
            tipo_documento: Tipo do documento (LEI, IN, DECRETO)
            numero: Numero do documento
            ano: Ano do documento
            article_number: Numero do artigo
            device_type: Tipo do dispositivo (article, paragraph, inciso, alinea)
            span_id: ID do span (ex: PAR-014-5, INC-002-I)
            chunk_id: ID completo do chunk
            document_id: ID do documento

        Returns:
            Citation com texto formatado
        """
        # Nome do tipo de documento
        doc_name = self.DOC_TYPE_NAMES.get(tipo_documento.upper(), tipo_documento)

        # Formata numero do documento (14133 -> 14.133)
        doc_number = self._format_doc_number(numero)

        # Base: "Lei 14.133/2021"
        doc_ref = f"{doc_name} {doc_number}/{ano}"

        # Artigo: "Art. 14"
        art_ref = f"Art. {article_number}"

        # Dispositivo especifico
        device_ref = ""
        device_number = None

        if device_type != "article" and span_id:
            device_ref, device_number = self._format_device(device_type, span_id)

        # Monta citacao completa
        if device_ref:
            full_text = f"{doc_ref}, {art_ref}, {device_ref}"
        else:
            full_text = f"{doc_ref}, {art_ref}"

        # Versao curta (para badges)
        short_text = f"Art. {article_number}"
        if device_number:
            if device_type == "paragraph":
                short_text = f"Art. {article_number}, Par. {device_number}"
            elif device_type == "inciso":
                short_text = f"Art. {article_number}, {device_number}"

        return Citation(
            text=full_text,
            short=short_text,
            document_type=doc_name,
            document_number=doc_number,
            year=ano,
            article=article_number,
            device=device_type if device_type != "article" else None,
            device_number=device_number,
            chunk_id=chunk_id,
            document_id=document_id,
        )

    def _format_doc_number(self, numero: str) -> str:
        """Formata numero do documento (14133 -> 14.133)."""
        if not numero:
            return ""

        # Remove caracteres nao numericos
        num_clean = re.sub(r'[^\d]', '', str(numero))

        if not num_clean:
            return numero

        # Adiciona separador de milhar
        try:
            num_int = int(num_clean)
            if num_int >= 1000:
                return f"{num_int:,}".replace(",", ".")
            return str(num_int)
        except ValueError:
            return numero

    def _format_device(self, device_type: str, span_id: str) -> tuple[str, str]:
        """
        Formata referencia ao dispositivo.

        Args:
            device_type: Tipo (paragraph, inciso, alinea)
            span_id: ID do span (ex: PAR-014-5, INC-002-I, ALI-002-I-a)

        Returns:
            Tuple (texto formatado, numero do dispositivo)
        """
        parts = span_id.split("-")

        if device_type == "paragraph":
            # PAR-014-5 -> "Par. 5" ou "Paragrafo unico"
            if len(parts) >= 3:
                par_num = parts[-1]
                if par_num.upper() == "UNICO":
                    return "Paragrafo unico", "unico"
                else:
                    return f"Par. {par_num}", par_num

        elif device_type == "inciso":
            # INC-002-I -> "inciso I"
            # INC-002-I_2 -> "inciso I do Par. 2"
            if len(parts) >= 3:
                inc_part = parts[-1]

                # Verifica se e inciso de paragrafo (INC-XXX-I_2)
                if "_" in inc_part:
                    inc_num, par_num = inc_part.split("_", 1)
                    return f"inciso {inc_num} do Par. {par_num}", inc_num
                else:
                    return f"inciso {inc_part}", inc_part

        elif device_type == "alinea":
            # ALI-002-I-a -> "alinea 'a'"
            # ALI-002-I_2-a -> "alinea 'a' do inciso I do Par. 2"
            if len(parts) >= 4:
                alinea = parts[-1]
                return f"alinea '{alinea}'", alinea

        return "", ""

    def format_multiple(self, chunks: list[dict]) -> list[Citation]:
        """Formata multiplas citacoes."""
        return [self.format_from_chunk(chunk) for chunk in chunks]

    def group_by_article(self, citations: list[Citation]) -> dict[str, list[Citation]]:
        """
        Agrupa citacoes por artigo.

        Retorna dict: "Art. 14" -> [Citation, Citation, ...]
        """
        grouped = {}
        for cit in citations:
            key = f"Art. {cit.article}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(cit)
        return grouped


# Funcao de conveniencia
def format_citation(
    tipo_documento: str,
    numero: str,
    ano: int,
    article_number: str,
    device_type: str = "article",
    span_id: str = "",
    **kwargs
) -> str:
    """
    Formata citacao simples (retorna apenas texto).

    Uso:
        citation = format_citation(
            tipo_documento="LEI",
            numero="14133",
            ano=2021,
            article_number="14",
            device_type="paragraph",
            span_id="PAR-014-5"
        )
        # -> "Lei 14.133/2021, Art. 14, Par. 5"
    """
    formatter = CitationFormatter()
    result = formatter.format(
        tipo_documento=tipo_documento,
        numero=numero,
        ano=ano,
        article_number=article_number,
        device_type=device_type,
        span_id=span_id,
        **kwargs
    )
    return result.text


def format_citation_from_hit(hit) -> Citation:
    """
    Formata citacao a partir de um SearchHit.

    Uso:
        result = searcher.search("query")
        for hit in result.hits:
            citation = format_citation_from_hit(hit)
            print(citation.text)
    """
    formatter = CitationFormatter()
    return formatter.format(
        tipo_documento=hit.document_type or "",
        numero=hit.document_number or "",
        ano=hit.year or 0,
        article_number=hit.article_number or "",
        device_type=hit.device_type or "article",
        span_id=hit.span_id or "",
        chunk_id=hit.chunk_id or "",
        document_id=hit.document_id or "",
    )
