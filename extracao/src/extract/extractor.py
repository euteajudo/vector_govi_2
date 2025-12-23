"""
Extrator Principal - API similar ao LlamaExtract.

Este módulo fornece uma API elegante e simples para extração de dados
estruturados de documentos, usando tecnologias open-source.

Stack:
- Docling: Parsing de PDF para Markdown
- Qwen 3 8B: Transformação MD -> JSON
- Pydantic: Validação de schema
- LangGraph: Orquestração agêntica

Exemplo de uso:
    from extract import Extractor
    from models.legal_document import LegalDocument
    
    extractor = Extractor()
    result = extractor.extract(
        document="path/to/law.pdf",
        schema=LegalDocument,
    )
    print(result.data)
"""

import json
import logging
import re
from pathlib import Path
from typing import TypeVar, Generic, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel
from openai import OpenAI

from .config import ExtractConfig, ExtractMode, ChunkMode, LLMProvider

logger = logging.getLogger(__name__)


# Type variable para o schema
T = TypeVar("T", bound=BaseModel)


@dataclass
class ExtractionResult(Generic[T]):
    """Resultado de uma extração."""
    
    # Dados extraídos (validados pelo schema)
    data: Optional[T] = None
    
    # Dados brutos (antes da validação)
    raw_data: dict = field(default_factory=dict)
    
    # Metadados
    success: bool = False
    quality_score: float = 0.0
    
    # Timing
    extraction_time_seconds: float = 0.0
    
    # Logs e erros
    logs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    # Extensões opcionais
    sources: Optional[list[dict]] = None  # cite_sources
    reasoning: Optional[str] = None       # use_reasoning
    confidence: Optional[dict] = None     # confidence_scores
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "data": self.data.model_dump() if self.data else self.raw_data,
            "success": self.success,
            "quality_score": self.quality_score,
            "extraction_time_seconds": self.extraction_time_seconds,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class Extractor:
    """
    Extrator de dados estruturados de documentos.
    
    API similar ao LlamaExtract, mas usando stack open-source.
    
    Exemplo:
        extractor = Extractor()
        
        # Extração simples
        result = extractor.extract("doc.pdf", schema=MySchema)
        
        # Com configuração customizada
        result = extractor.extract(
            "doc.pdf",
            schema=MySchema,
            config=ExtractConfig.for_legal_documents(),
        )
    """
    
    def __init__(self, config: Optional[ExtractConfig] = None):
        """
        Inicializa o extrator.
        
        Args:
            config: Configuração padrão (pode ser sobrescrita em extract())
        """
        self.default_config = config or ExtractConfig.balanced()
        self._docling_converter = None
    
    @property
    def docling_converter(self):
        """Lazy loading do Docling converter."""
        if self._docling_converter is None:
            from docling.document_converter import DocumentConverter
            self._docling_converter = DocumentConverter()
        return self._docling_converter
    
    def extract(
        self,
        document: Union[str, Path, bytes],
        schema: type[T],
        config: Optional[ExtractConfig] = None,
    ) -> ExtractionResult[T]:
        """
        Extrai dados estruturados de um documento.
        
        Args:
            document: Caminho do arquivo, Path, ou bytes do documento
            schema: Classe Pydantic que define a estrutura de saída
            config: Configuração de extração (usa default se não fornecido)
        
        Returns:
            ExtractionResult com dados extraídos e metadados
        """
        config = config or self.default_config
        result = ExtractionResult[T]()
        start_time = datetime.now()
        
        try:
            # 1. Parse do documento
            result.logs.append("[DOCLING] Iniciando parsing do documento...")
            markdown_raw = self._parse_document(document, config)
            result.logs.append(f"[DOCLING] OK - {len(markdown_raw)} caracteres extraidos")

            # 2. Pré-processamento do Markdown (desabilitado temporariamente para teste)
            # As tags podem estar confundindo o modelo
            # result.logs.append("[PREPROCESS] Normalizando estrutura...")
            # markdown = self._preprocess_markdown(markdown_raw)
            # result.logs.append("[PREPROCESS] OK - Tags de estrutura adicionadas")
            markdown = markdown_raw  # Usar markdown original

            # 3. Pré-análise (para validação posterior)
            expected_articles = self._pre_analyze(markdown, config)
            result.logs.append(f"[ANALYZE] Esperados {len(expected_articles)} artigos")

            # 4. Extração com LLM
            result.logs.append(f"[LLM] Extraindo com {config.llm.model}...")
            raw_json = self._extract_with_llm(markdown, schema, config)
            result.logs.append("[LLM] OK - JSON extraido")

            # 5. Pós-processamento para corrigir erros comuns
            result.logs.append("[POSTPROCESS] Corrigindo erros comuns...")
            raw_json = self._postprocess_extraction(raw_json, markdown_raw)
            result.raw_data = raw_json
            result.logs.append("[POSTPROCESS] OK - Correções aplicadas")

            # 6. Validação estrutural
            result.logs.append("[VALIDATE-STRUCT] Verificando estrutura...")
            issues = self._validate_extraction(raw_json, markdown_raw)
            for issue_type, issue_list in issues.items():
                for issue in issue_list:
                    result.warnings.append(f"{issue_type}: {issue}")
            if any(issues.values()):
                result.logs.append(f"[VALIDATE-STRUCT] {len([i for v in issues.values() for i in v])} avisos")
            else:
                result.logs.append("[VALIDATE-STRUCT] OK - Sem problemas")

            # 7. Validação com Pydantic
            result.logs.append("[VALIDATE-SCHEMA] Validando contra schema...")
            validated_data, validation_errors = self._validate_schema(raw_json, schema)

            if validated_data:
                result.data = validated_data
                result.success = True
                result.logs.append("[VALIDATE-SCHEMA] OK - Schema validado")
            else:
                result.errors.extend(validation_errors)
                result.warnings.append("Schema validation failed, using raw data")

            # 8. Calcular score de qualidade
            extracted_articles = self._get_extracted_articles(raw_json)
            result.quality_score = self._calculate_quality(
                expected_articles, extracted_articles
            )
            result.logs.append(f"[QUALITY] Score: {result.quality_score:.1%}")

            # 9. Correção se necessário
            if result.quality_score < config.validation.min_quality_score:
                result.logs.append("[FIX] Tentando corrigir artigos faltantes...")
                missing = set(expected_articles) - set(extracted_articles)
                
                if missing and config.validation.max_fix_attempts > 0:
                    fixed_data = self._fix_missing(
                        markdown, raw_json, list(missing), schema, config
                    )
                    result.raw_data = fixed_data
                    
                    # Revalidar
                    validated_data, _ = self._validate_schema(fixed_data, schema)
                    if validated_data:
                        result.data = validated_data
                    
                    # Recalcular score
                    new_articles = self._get_extracted_articles(fixed_data)
                    result.quality_score = self._calculate_quality(
                        expected_articles, new_articles
                    )
                    result.logs.append(f"[FIX] Novo score: {result.quality_score:.1%}")
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Extraction failed: {str(e)}")
        
        result.extraction_time_seconds = (datetime.now() - start_time).total_seconds()
        return result
    
    def _parse_document(
        self,
        document: Union[str, Path, bytes],
        config: ExtractConfig
    ) -> str:
        """Parse documento para Markdown usando Docling."""
        
        # Se já é string (markdown), retorna direto
        if isinstance(document, str):
            path = Path(document)
            if path.exists():
                if path.suffix.lower() == ".md":
                    return path.read_text(encoding="utf-8")
                # É um arquivo para processar
                result = self.docling_converter.convert(str(path))
                return result.document.export_to_markdown()
            else:
                # É markdown direto
                return document
        
        elif isinstance(document, Path):
            if document.suffix.lower() == ".md":
                return document.read_text(encoding="utf-8")
            result = self.docling_converter.convert(str(document))
            return result.document.export_to_markdown()
        
        elif isinstance(document, bytes):
            # TODO: Implementar extração de bytes
            raise NotImplementedError("Extração de bytes ainda não implementada")
        
        raise ValueError(f"Tipo de documento não suportado: {type(document)}")
    
    def _pre_analyze(self, markdown: str, config: ExtractConfig) -> list[int]:
        """Pré-análise do documento para identificar elementos esperados."""
        articles = []
        pattern = r'Art\.?\s*(\d+)[°ºo]?'

        for match in re.finditer(pattern, markdown):
            art_num = int(match.group(1))
            if art_num not in articles:
                articles.append(art_num)

        return sorted(articles)

    def _preprocess_markdown(self, markdown: str) -> str:
        """
        Pré-processa o Markdown para facilitar a extração.

        Normaliza formatação e destaca elementos estruturais.
        """
        text = markdown

        # 1. Normalizar parágrafos - garantir § visível
        # "§ 1o" -> "[PARAGRAFO] § 1o"
        text = re.sub(
            r'(?<!\[PARAGRAFO\] )(§\s*\d+[ºo°]?)',
            r'[PARAGRAFO] \1',
            text
        )
        text = re.sub(
            r'(?i)(?<!\[PARAGRAFO\] )(par[aá]grafo\s+[uú]nico)',
            r'[PARAGRAFO] \1',
            text
        )

        # 2. Normalizar incisos - destacar numerais romanos
        # "I - texto" -> "[INCISO] I - texto"
        text = re.sub(
            r'(?m)^(\s*)(I{1,3}|IV|VI{0,3}|IX|XI{0,3})\s*[-–—]\s*',
            r'\1[INCISO] \2 - ',
            text
        )

        # 3. Normalizar alíneas - destacar letras
        # "a) texto" -> "[ALINEA] a) texto"
        text = re.sub(
            r'(?m)^(\s*)([a-z])\)\s*',
            r'\1[ALINEA] \2) ',
            text
        )

        # 4. Normalizar artigos
        # "Art. 1o" -> "[ARTIGO] Art. 1o"
        text = re.sub(
            r'(?m)^(\s*)(Art\.?\s*\d+[°ºo]?)',
            r'\1[ARTIGO] \2',
            text
        )

        return text

    def _fix_paragraphs_from_markdown(self, article: dict, art_num: str, markdown: str) -> None:
        """
        Verifica no markdown original se o artigo tem § e corrige items incorretos.
        """
        # Encontrar conteúdo do artigo no markdown
        pattern = rf'(?:^|\n)\s*Art\.?\s*{art_num}[°ºo\.]?\s+(.+?)(?=(?:^|\n)\s*Art\.?\s*\d+|CAP[ÍI]TULO|$)'
        match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
        if not match:
            return

        art_content = match.group(1)

        # Contar § no artigo original
        para_matches = re.findall(r'§\s*(\d+)[ºo°]?', art_content)
        para_unico = re.search(r'(?i)par[aá]grafo\s+[uú]nico', art_content)

        expected_paras = len(para_matches) + (1 if para_unico else 0)
        current_paras = len(article.get("paragraphs", []))

        # Se há § no markdown mas não na extração, verificar items
        if expected_paras > current_paras and article.get("items"):
            items = article.get("items", [])
            items_to_move = []

            # Mapear número romano para número decimal
            roman_to_num = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}

            for i, item in enumerate(items):
                identifier = item.get("item_identifier", "")
                # Verificar se este "inciso" é na verdade um parágrafo
                if identifier in roman_to_num:
                    num = roman_to_num[identifier]
                    # Procurar se existe § com esse número no markdown
                    if str(num) in para_matches:
                        items_to_move.append((i, num, item.get("description", "")))

            # Mover items para paragraphs
            if items_to_move:
                if "paragraphs" not in article:
                    article["paragraphs"] = []

                for idx, para_num, content in reversed(items_to_move):
                    article["paragraphs"].insert(0, {
                        "paragraph_identifier": str(para_num),
                        "content": content
                    })
                    article["items"].pop(idx)

                # Ordenar paragraphs por número
                article["paragraphs"].sort(
                    key=lambda p: (
                        0 if p["paragraph_identifier"] == "unico" else int(p["paragraph_identifier"])
                    )
                )

    def _postprocess_extraction(self, data: dict, markdown: str = "") -> dict:
        """
        Pós-processa a extração para corrigir erros comuns.

        Corrige problemas de classificação § vs incisos e remove duplicações.
        """
        if "chapters" not in data:
            return data

        # Remover capítulos vazios (sem artigos)
        data["chapters"] = [
            ch for ch in data["chapters"]
            if ch.get("articles") and len(ch["articles"]) > 0
        ]

        for chapter in data["chapters"]:
            for article in chapter.get("articles", []):
                # 0. Verificar no markdown se artigo tem § e corrigir items que são parágrafos
                art_num = article.get("article_number", "")
                if markdown and art_num:
                    self._fix_paragraphs_from_markdown(article, art_num, markdown)
                # 1. Mover § dos items para paragraphs
                items_to_remove = []
                for i, item in enumerate(article.get("items", [])):
                    identifier = item.get("item_identifier", "")
                    description = item.get("description", "")

                    # Detectar se é parágrafo disfarçado de inciso
                    is_paragraph = (
                        "§" in identifier or
                        "§" in description[:20] or
                        identifier.lower() in ["1", "2", "3", "unico", "único"] and
                        not re.match(r'^[IVXLC]+$', identifier)
                    )

                    if is_paragraph:
                        # Mover para paragraphs
                        para_id = identifier.replace("§", "").strip()
                        if not para_id or para_id in ["1o", "1º", "2o", "2º"]:
                            para_id = re.search(r'\d+', identifier) or "1"
                            if hasattr(para_id, 'group'):
                                para_id = para_id.group()

                        if "paragraphs" not in article:
                            article["paragraphs"] = []

                        article["paragraphs"].append({
                            "paragraph_identifier": para_id,
                            "content": description
                        })
                        items_to_remove.append(i)

                # Remover items movidos (do fim para o início)
                for i in reversed(items_to_remove):
                    article["items"].pop(i)

                # 2. Limpar sub_items fantasmas (vazios ou inválidos)
                for item in article.get("items", []):
                    if "sub_items" in item:
                        # Filtrar apenas sub_items válidos
                        valid_sub_items = [
                            si for si in item["sub_items"]
                            if si.get("description", "").strip() and
                               re.match(r'^[a-z]$', si.get("item_identifier", ""))
                        ]
                        item["sub_items"] = valid_sub_items

                # 3. Remover duplicações de conteúdo
                seen_content = set()
                unique_items = []
                for item in article.get("items", []):
                    content_key = item.get("description", "")[:100]
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        unique_items.append(item)
                article["items"] = unique_items

        return data

    def _validate_extraction(self, data: dict, markdown: str) -> dict:
        """
        Valida a extração e retorna relatório de problemas.

        Compara elementos extraídos com o documento original.
        """
        issues = {
            "missing_paragraphs": [],
            "missing_items": [],
            "duplicated_content": [],
            "phantom_sub_items": [],
        }

        # Identificar artigos do documento (para não contar referências a outras leis)
        extracted_articles = set()
        for chapter in data.get("chapters", []):
            for article in chapter.get("articles", []):
                extracted_articles.add(str(article.get("article_number", "")))

        # Contar § apenas em artigos do próprio documento
        paragraphs_in_md = 0
        for art_num in extracted_articles:
            # Encontrar conteúdo do artigo
            pattern = rf'Art\.?\s*{art_num}[°ºo\.]?(.+?)(?=Art\.?\s*\d+|CAP[ÍI]TULO|$)'
            match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1)
                paragraphs_in_md += len(re.findall(r'§\s*\d+[ºo°]?', content))
                paragraphs_in_md += len(re.findall(r'(?i)par[aá]grafo\s+[uú]nico', content))

        # Contar § na extração
        paragraphs_extracted = 0
        for chapter in data.get("chapters", []):
            for article in chapter.get("articles", []):
                paragraphs_extracted += len(article.get("paragraphs", []))

        if paragraphs_extracted < paragraphs_in_md:
            issues["missing_paragraphs"].append(
                f"Esperados {paragraphs_in_md} §, extraídos {paragraphs_extracted}"
            )

        # Contar alíneas no markdown original
        alineas_in_md = len(re.findall(r'(?m)^\s*[a-z]\)\s*\S', markdown))

        # Contar sub_items na extração
        sub_items_extracted = 0
        for chapter in data.get("chapters", []):
            for article in chapter.get("articles", []):
                for item in article.get("items", []):
                    sub_items_extracted += len(item.get("sub_items", []))

        if sub_items_extracted > alineas_in_md:
            issues["phantom_sub_items"].append(
                f"Alíneas no MD: {alineas_in_md}, sub_items extraídos: {sub_items_extracted}"
            )

        return issues

    def _get_llm_client(self, config: ExtractConfig) -> OpenAI:
        """Retorna cliente OpenAI configurado para o provider."""
        return OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            timeout=config.llm.timeout,
        )

    def _get_vllm_client(self, config: ExtractConfig):
        """Retorna cliente VLLMClient para uso com guided_json."""
        from llm.vllm_client import VLLMClient, LLMConfig as VLLMConfig

        vllm_config = VLLMConfig(
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=float(config.llm.timeout),
        )
        return VLLMClient(config=vllm_config)

    def _extract_with_llm(
        self,
        markdown: str,
        schema: type[BaseModel],
        config: ExtractConfig
    ) -> dict:
        """Extrai JSON estruturado usando LLM."""

        # Verificar se documento é grande e precisa de chunking
        # ~4 chars = 1 token, contexto 16k - 4k para resposta = 12k tokens input = 48k chars
        if len(markdown) > 10000 and config.chunk_mode == ChunkMode.ARTICLE:
            return self._extract_by_chapters(markdown, schema, config)

        return self._extract_single_pass(markdown, schema, config)

    def _extract_single_pass(
        self,
        markdown: str,
        schema: type[BaseModel],
        config: ExtractConfig
    ) -> dict:
        """Extrai em uma única passada."""
        system_prompt = self._build_system_prompt(schema, config)
        max_doc_chars = 50000

        # Usar guided_json se habilitado (só vLLM)
        if config.llm.use_guided_json and config.llm.provider == LLMProvider.VLLM:
            logger.info("Usando guided_json para extração estruturada")
            user_prompt = f"""DOCUMENTO:
{markdown[:max_doc_chars]}

Extraia o documento para JSON seguindo a estrutura do schema. /no_think"""

            vllm_client = self._get_vllm_client(config)
            try:
                result = vllm_client.chat_with_schema(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=schema,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens,
                )
                return result
            finally:
                vllm_client.close()

        # Fallback: extração tradicional com parsing manual
        json_schema = schema.model_json_schema()
        schema_str = json.dumps(json_schema, indent=2, ensure_ascii=False)

        user_prompt = f"""DOCUMENTO:
{markdown[:max_doc_chars]}

JSON SCHEMA:
{schema_str}

Retorne APENAS o JSON extraido, sem explicacoes. /no_think"""

        client = self._get_llm_client(config)
        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )

        content = response.choices[0].message.content
        return self._extract_json_from_response(content)

    def _extract_by_chapters(
        self,
        markdown: str,
        schema: type[BaseModel],
        config: ExtractConfig
    ) -> dict:
        """Extrai documento dividindo por capítulos."""
        # Dividir por capítulos
        chapters = self._split_by_chapters(markdown)

        # Extrair metadados do documento (do início)
        header = markdown[:1500]
        doc_metadata = self._extract_document_metadata(header, config)

        # Extrair cada capítulo
        extracted_chapters = []
        for chapter_title, chapter_content in chapters:
            chapter_data = self._extract_chapter(
                chapter_title, chapter_content, config
            )
            if chapter_data:
                # Validação pós-extração: remover artigos inventados
                chapter_data = self._validate_extracted_chapter(
                    chapter_data, chapter_content
                )
                # Só adiciona se ainda tiver artigos após validação
                if chapter_data.get("articles"):
                    extracted_chapters.append(chapter_data)

        # Combinar resultados
        result = doc_metadata.copy()
        result["chapters"] = extracted_chapters
        return result

    def _validate_extracted_chapter(
        self,
        chapter_data: dict,
        chapter_content: str
    ) -> dict:
        """
        Valida artigos extraídos contra o conteúdo original.

        Remove artigos inventados (que não existem no markdown original).
        Isso é uma proteção adicional contra alucinação do LLM.

        IMPORTANTE: Distingue entre artigos do próprio documento e
        referências a artigos de outras leis (ex: "Art. 75 da Lei 14.133").
        """
        # Identificar artigos que são DO PRÓPRIO DOCUMENTO (não referências)
        # Pattern: "Art. N" no início de linha/parágrafo ou após quebra
        # Exclui: "art. N da Lei X", "art. N do Decreto Y", etc.
        real_articles = set()

        # Pattern 1: Art. no início de linha ou após marcadores de estrutura
        # Captura artigos que são definições do documento
        # Inclui formato Docling: "- Art. N°" (bullet antes do Art.)
        for match in re.finditer(
            r'(?:^|\n)\s*[-*]?\s*Art\.?\s*(\d+)[°ºo]?(?:\s|\.|\s*[-–—])',
            chapter_content,
            re.IGNORECASE | re.MULTILINE
        ):
            real_articles.add(match.group(1))

        # Pattern 2: Art. seguido de conteúdo do artigo (não referência)
        # Exclui padrões como "art. N da Lei", "art. N do Decreto"
        for match in re.finditer(
            r'Art\.?\s*(\d+)[°ºo]?\s+(?![dD][aoeu])',  # Não seguido de "da", "do", "de", "du"
            chapter_content,
            re.IGNORECASE
        ):
            # Verifica se não é referência a outra lei
            start = match.start()
            context_after = chapter_content[match.end():match.end()+30].lower()
            if not any(ref in context_after for ref in ['lei ', 'decreto ', 'instrução ', 'portaria ']):
                real_articles.add(match.group(1))

        # Filtrar apenas artigos reais
        valid_articles = []
        for article in chapter_data.get("articles", []):
            art_num_raw = str(article.get("article_number", ""))
            # Normaliza: extrai apenas o número do article_number
            # Ex: "Art. 7°" -> "7", "7" -> "7", "Art. Art. 10" -> "10"
            art_num_match = re.search(r'(\d+)', art_num_raw)
            art_num = art_num_match.group(1) if art_num_match else art_num_raw

            # Atualiza article_number para formato limpo
            if art_num_match:
                article["article_number"] = art_num

            if art_num in real_articles:
                valid_articles.append(article)
            else:
                # Log para debug: artigo inventado detectado e removido
                logger.warning(
                    f"Artigo inventado removido: Art. {art_num} "
                    f"(nao existe no capitulo '{chapter_data.get('title', '')}')"
                )

        chapter_data["articles"] = valid_articles
        return chapter_data

    def _split_by_chapters(self, markdown: str) -> list[tuple[str, str]]:
        """
        Divide markdown por capítulos.

        IMPORTANTE: Ignora conteúdo antes do primeiro CAPÍTULO real,
        pois esse conteúdo é apenas metadados do documento (ementa, data, etc.)
        e não contém artigos. Incluir esse conteúdo como "capítulo" causaria
        alucinação do LLM (inventar artigos que não existem).
        """
        # Pattern para encontrar capítulos
        pattern = r'(CAP[ÍI]TULO\s+[IVXLC]+[^\n]*)'
        parts = re.split(pattern, markdown, flags=re.IGNORECASE)

        chapters = []
        current_title = None  # Não criar capítulo fantasma
        current_content = []
        found_first_chapter = False

        for i, part in enumerate(parts):
            if re.match(r'CAP[ÍI]TULO', part, re.IGNORECASE):
                # Salvar capítulo anterior (apenas se já encontramos um capítulo real)
                if current_content and found_first_chapter:
                    chapters.append((current_title, '\n'.join(current_content)))
                current_title = part.strip()
                current_content = []
                found_first_chapter = True
            else:
                # Só adiciona conteúdo se já encontramos o primeiro capítulo
                if found_first_chapter:
                    current_content.append(part)
                # Conteúdo antes do primeiro capítulo é ignorado (metadados)

        # Salvar último capítulo
        if current_content and found_first_chapter:
            chapters.append((current_title, '\n'.join(current_content)))

        # Se não encontrou nenhum capítulo, documento não tem estrutura de capítulos
        # Nesse caso, retornar documento inteiro como único bloco (sem título fantasma)
        if not chapters:
            # Verificar se tem artigos no documento
            has_articles = bool(re.search(r'Art\.?\s*\d+', markdown, re.IGNORECASE))
            if has_articles:
                chapters.append(("DISPOSIÇÕES GERAIS", markdown))

        return chapters

    def _extract_document_metadata(self, header: str, config: ExtractConfig) -> dict:
        """Extrai metadados do documento (tipo, número, data, ementa)."""
        prompt = f"""Extraia os metadados deste documento legal:

{header}

Retorne JSON com:
{{"document_type": "tipo (LEI, DECRETO, INSTRUCAO NORMATIVA, etc)",
"issuing_body": "orgao emissor",
"issuing_body_acronym": "sigla ou null",
"number": "numero do documento",
"date": "data no formato YYYY-MM-DD",
"ementa": "resumo oficial do documento",
"signatory": "nome de quem assinou ou null"}}

Retorne APENAS o JSON. /no_think"""

        client = self._get_llm_client(config)
        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )

        return self._extract_json_from_response(response.choices[0].message.content)

    def _get_chapter_schema(self) -> dict:
        """Retorna JSON Schema para extração de capítulo."""
        return {
            "type": "object",
            "properties": {
                "chapter_number": {
                    "type": ["string", "null"],
                    "description": "Número romano do capítulo (I, II, III...)"
                },
                "title": {
                    "type": "string",
                    "description": "Título do capítulo"
                },
                "articles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "article_number": {
                                "type": "string",
                                "description": "Número do artigo (1, 2, 3...)"
                            },
                            "title": {
                                "type": ["string", "null"],
                                "description": "Título do artigo ou null"
                            },
                            "content": {
                                "type": "string",
                                "description": "Texto principal do artigo"
                            },
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item_identifier": {"type": "string"},
                                        "description": {"type": "string"},
                                        "sub_items": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "item_identifier": {"type": "string"},
                                                    "description": {"type": "string"}
                                                },
                                                "required": ["item_identifier", "description"]
                                            }
                                        }
                                    },
                                    "required": ["item_identifier", "description", "sub_items"]
                                }
                            },
                            "paragraphs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "paragraph_identifier": {"type": "string"},
                                        "content": {"type": "string"}
                                    },
                                    "required": ["paragraph_identifier", "content"]
                                }
                            }
                        },
                        "required": ["article_number", "content", "items", "paragraphs"]
                    }
                }
            },
            "required": ["title", "articles"]
        }

    def _extract_chapter(
        self,
        chapter_title: str,
        chapter_content: str,
        config: ExtractConfig
    ) -> dict:
        """Extrai artigos de um capítulo."""
        # Validação pré-extração: verificar se o texto realmente tem artigos
        articles_in_content = re.findall(r'Art\.?\s*(\d+)', chapter_content, re.IGNORECASE)
        if not articles_in_content:
            # Não há artigos neste capítulo, retornar vazio
            return {
                "chapter_number": None,
                "title": chapter_title,
                "articles": []
            }

        # Lista de artigos esperados para validação
        expected_articles = sorted(set(int(a) for a in articles_in_content))
        expected_str = ", ".join(str(a) for a in expected_articles)

        system_prompt = """Voce e um extrator de documentos legais brasileiros.
REGRAS CRITICAS:
1. Extraia APENAS os artigos que EXISTEM no texto
2. NAO INVENTE artigos que nao estao no texto
3. Incisos sao I, II, III... -> items
4. Paragrafos sao § 1o, § 2o, Paragrafo unico -> paragraphs
5. Alineas sao a), b), c) -> sub_items
PROIBIDO: Criar artigos inventados ou com conteudo fabricado."""

        user_prompt = f"""Extraia os artigos deste capitulo:

CAPITULO: {chapter_title}
ARTIGOS ESPERADOS: Art. {expected_str}

CONTEUDO:
{chapter_content[:8000]}

/no_think"""

        # Usar guided_json se habilitado (só vLLM)
        if config.llm.use_guided_json and config.llm.provider == LLMProvider.VLLM:
            vllm_client = self._get_vllm_client(config)
            try:
                result = vllm_client.chat_with_schema(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=self._get_chapter_schema(),
                    temperature=0.0,
                    max_tokens=config.llm.max_tokens,
                )
                return result
            except Exception as e:
                logger.warning(f"guided_json falhou para capítulo, usando fallback: {e}")
            finally:
                vllm_client.close()

        # Fallback: extração tradicional
        prompt = f"""Extraia TODOS os artigos deste capitulo de documento legal:

CAPITULO: {chapter_title}

CONTEUDO:
{chapter_content[:8000]}

Retorne JSON com:
{{"chapter_number": "numero romano ou null",
"title": "titulo do capitulo",
"articles": [
  {{"article_number": "numero (ex: 1, 2, 10)",
   "title": "titulo do artigo ou null",
   "content": "texto principal do artigo",
   "items": [{{"item_identifier": "I, II, III...", "description": "texto", "sub_items": []}}],
   "paragraphs": [{{"paragraph_identifier": "1, 2, unico", "content": "texto"}}]
  }}
]}}

REGRAS CRITICAS:
1. Extraia APENAS os artigos que EXISTEM no texto: Art. {expected_str}
2. NAO INVENTE artigos que nao estao no texto
3. Se nao encontrar um artigo no texto, NAO o inclua no JSON
4. Incisos sao I, II, III... -> items
5. Paragrafos sao § 1o, § 2o, Paragrafo unico -> paragraphs
6. Alineas sao a), b), c) -> sub_items

PROIBIDO: Criar artigos inventados ou com conteudo fabricado.

Retorne APENAS o JSON. /no_think"""

        client = self._get_llm_client(config)
        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=config.llm.max_tokens,
        )

        try:
            return self._extract_json_from_response(response.choices[0].message.content)
        except Exception:
            return {
                "chapter_number": None,
                "title": chapter_title,
                "articles": []
            }

    def _extract_json_from_response(self, content: str) -> dict:
        """Extrai JSON da resposta do LLM, ignorando texto extra."""
        # Tentar parsear diretamente
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Procurar JSON entre ```json e ```
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Procurar primeiro { e último }
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Não foi possível extrair JSON da resposta: {content[:500]}...")
    
    def _build_system_prompt(
        self,
        schema: type[BaseModel],
        config: ExtractConfig
    ) -> str:
        """Constrói o system prompt baseado na configuração."""

        base_prompt = """Voce e um extrator de dados estruturados de alta precisao.
Sua tarefa e extrair informacoes do documento e retornar em formato JSON
seguindo EXATAMENTE o schema fornecido.

REGRAS GERAIS:
1. Extraia TODOS os elementos do documento
2. Mantenha a estrutura hierarquica
3. Use os nomes de campos EXATOS do schema
4. Retorne APENAS JSON valido, sem explicacoes
5. Se um campo e opcional e nao existe, use null"""

        # Adicionar prompt customizado
        if config.system_prompt:
            base_prompt += f"\n\nINSTRUCOES ADICIONAIS:\n{config.system_prompt}"

        # Adicionar instruções específicas para documentos legais
        if config.chunk_mode == ChunkMode.ARTICLE:
            base_prompt += """

REGRAS CRITICAS PARA DOCUMENTOS LEGAIS BRASILEIROS:

1. PARAGRAFOS vs INCISOS - DISTINCAO OBRIGATORIA:
   - PARAGRAFO: Comeca com "§" (ex: "§ 1o", "§ 2o") ou "Paragrafo unico" -> vai em "paragraphs"
   - INCISO: Comeca com numeral romano (I, II, III, IV, V...) -> vai em "items"
   - ATENCAO: "§ 1o" NAO e inciso I! E paragrafo primeiro!
   - Se vir "§" ou "Paragrafo", SEMPRE coloque em paragraphs com paragraph_identifier

2. ESTRUTURA HIERARQUICA CORRETA:
   - Artigo (Art. 1o, Art. 2o...)
     - items[]: Incisos com numeral romano (I, II, III...)
       - sub_items[]: Alineas com letras (a, b, c...)
     - paragraphs[]: Paragrafos com § (§ 1o, § 2o, Paragrafo unico)

3. REGRAS DE CAMPOS:
   - article_number: apenas o numero (1, 2, 3), sem "Art." ou "o"
   - item_identifier: numeral romano EXATO (I, II, III, IV, V, VI...)
   - sub_items[].item_identifier: letra minuscula (a, b, c, d...)
   - paragraph_identifier: numero (1, 2, 3) ou "unico" para paragrafo unico

4. EVITAR ERROS COMUNS:
   - NAO duplique conteudo entre artigos
   - NAO invente alineas/sub_items que nao existem no texto
   - NAO confunda "§" (paragrafo) com numeral romano (inciso)
   - Se texto nao tem alineas (a, b, c), sub_items deve ser []
   - Cada elemento deve aparecer UMA UNICA VEZ na estrutura"""

        return base_prompt
    
    def _validate_schema(
        self,
        data: dict,
        schema: type[T]
    ) -> tuple[Optional[T], list[str]]:
        """Valida dados contra schema Pydantic."""
        try:
            validated = schema.model_validate(data)
            return validated, []
        except Exception as e:
            errors = []
            if hasattr(e, 'errors'):
                for err in e.errors():
                    loc = ' -> '.join(str(x) for x in err['loc'])
                    errors.append(f"{loc}: {err['msg']}")
            else:
                errors.append(str(e))
            return None, errors
    
    def _get_extracted_articles(self, data: dict) -> list[int]:
        """Extrai lista de números de artigos do JSON."""
        articles = []
        
        for chapter in data.get("chapters", []):
            for article in chapter.get("articles", []):
                art_str = str(article.get("article_number", "0"))
                match = re.search(r'\d+', art_str)
                if match:
                    articles.append(int(match.group()))
        
        return sorted(set(articles))
    
    def _calculate_quality(
        self,
        expected: list[int],
        extracted: list[int]
    ) -> float:
        """Calcula score de qualidade da extração."""
        if not expected:
            return 1.0 if not extracted else 0.5
        
        expected_set = set(expected)
        extracted_set = set(extracted)
        
        correct = len(expected_set & extracted_set)
        return correct / len(expected_set)
    
    def _fix_missing(
        self,
        markdown: str,
        current_data: dict,
        missing: list[int],
        schema: type[BaseModel],
        config: ExtractConfig
    ) -> dict:
        """Tenta extrair artigos faltantes individualmente."""

        fixed_articles = []
        client = self._get_llm_client(config)

        for art_num in missing[:config.validation.max_fix_attempts]:
            # Encontrar texto do artigo
            pattern = rf'Art\.?\s*{art_num}[°ºo]?\s*(.+?)(?=Art\.?\s*\d+[°ºo]?|CAPITULO|CAPÍTULO|$)'
            match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)

            if not match:
                continue

            art_text = f"Art. {art_num} " + match.group(1).strip()[:2500]

            prompt = f"""Extraia este artigo para JSON:

ARTIGO:
{art_text}

JSON:
{{"article_number": "{art_num}", "title": null, "content": "texto", "items": [], "paragraphs": []}}

Retorne APENAS o JSON do artigo. /no_think"""

            try:
                response = client.chat.completions.create(
                    model=config.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2048,
                )

                article = self._extract_json_from_response(response.choices[0].message.content)
                article["article_number"] = str(art_num)
                fixed_articles.append(article)

            except Exception:
                continue

        # Merge com dados existentes
        if fixed_articles and current_data.get("chapters"):
            current_data["chapters"][0]["articles"].extend(fixed_articles)

        return current_data


# =============================================================================
# EXTRACTION AGENT - Para workflows reutilizáveis
# =============================================================================

class ExtractionAgent(Generic[T]):
    """
    Agente de extração reutilizável.
    
    Equivalente ao Extraction Agent do LlamaExtract.
    Encapsula schema + configuração para uso repetido.
    
    Exemplo:
        agent = ExtractionAgent(
            name="lei-parser",
            schema=LegalDocument,
            config=ExtractConfig.for_legal_documents(),
        )
        
        result = agent.extract("lei1.pdf")
        result = agent.extract("lei2.pdf")
    """
    
    def __init__(
        self,
        name: str,
        schema: type[T],
        config: Optional[ExtractConfig] = None,
    ):
        self.name = name
        self.schema = schema
        self.config = config or ExtractConfig.balanced()
        self._extractor = Extractor(self.config)
        self._history: list[ExtractionResult] = []
    
    def extract(
        self,
        document: Union[str, Path, bytes],
        config_override: Optional[ExtractConfig] = None,
    ) -> ExtractionResult[T]:
        """Extrai usando o schema e config do agente."""
        config = config_override or self.config
        result = self._extractor.extract(document, self.schema, config)
        self._history.append(result)
        return result
    
    def get_history(self) -> list[ExtractionResult]:
        """Retorna histórico de extrações."""
        return self._history
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do agente."""
        if not self._history:
            return {"total": 0}
        
        successful = sum(1 for r in self._history if r.success)
        avg_quality = sum(r.quality_score for r in self._history) / len(self._history)
        avg_time = sum(r.extraction_time_seconds for r in self._history) / len(self._history)
        
        return {
            "total": len(self._history),
            "successful": successful,
            "success_rate": successful / len(self._history),
            "avg_quality_score": avg_quality,
            "avg_extraction_time": avg_time,
        }


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Adicionar src ao path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.legal_document import LegalDocument
    
    # Criar extrator
    extractor = Extractor()
    
    # Configuração para documentos legais
    config = ExtractConfig.for_legal_documents()
    
    print("=== Extractor Open-Source ===")
    print(f"Mode: {config.extraction_mode}")
    print(f"LLM: {config.llm.model}")
    print(f"Schema: {LegalDocument.__name__}")
    
    # Testar com arquivo de exemplo
    test_file = Path("data/output")
    md_files = list(test_file.glob("*_extracted.md"))
    
    if md_files:
        print(f"\nExtraindo: {md_files[0].name}")
        result = extractor.extract(
            md_files[0],
            schema=LegalDocument,
            config=config,
        )
        
        print(f"\nResultado:")
        print(f"  Success: {result.success}")
        print(f"  Quality: {result.quality_score:.1%}")
        print(f"  Time: {result.extraction_time_seconds:.2f}s")
        print(f"  Errors: {len(result.errors)}")
        
        for log in result.logs:
            print(f"  {log}")

