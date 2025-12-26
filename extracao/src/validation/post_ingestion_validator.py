"""Validador pós-ingestão para detectar e corrigir erros em chunks."""
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Tipos de erros detectados."""
    THESIS_TYPE_MARKDOWN = "thesis_type_markdown"  # Contém ** ou markdown
    THESIS_TYPE_INVALID = "thesis_type_invalid"    # Valor não reconhecido
    THESIS_TEXT_EMPTY = "thesis_text_empty"        # Vazio quando deveria ter
    THESIS_TEXT_MARKDOWN = "thesis_text_markdown"  # Contém markdown residual
    ENRICHED_TEXT_EMPTY = "enriched_text_empty"    # Sem texto enriquecido
    CONTEXT_HEADER_EMPTY = "context_header_empty"  # Sem contexto
    EMBEDDING_ZERO = "embedding_zero"              # Vetor zerado


@dataclass
class ValidationError:
    """Representa um erro de validação."""
    chunk_id: str
    error_type: ErrorType
    field: str
    current_value: str
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ValidationResult:
    """Resultado da validação de um documento."""
    document_id: str
    total_chunks: int = 0
    valid_chunks: int = 0
    errors: list = field(default_factory=list)
    fixed_chunks: list = field(default_factory=list)
    unfixable_chunks: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.valid_chunks / self.total_chunks

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def summary(self) -> str:
        """Gera resumo da validação."""
        lines = [
            f"=== Validação: {self.document_id} ===",
            f"Total chunks: {self.total_chunks}",
            f"Válidos: {self.valid_chunks} ({self.success_rate:.1%})",
            f"Erros encontrados: {len(self.errors)}",
            f"Corrigidos automaticamente: {len(self.fixed_chunks)}",
            f"Não corrigíveis: {len(self.unfixable_chunks)}",
        ]

        if self.unfixable_chunks:
            lines.append("\n[ATENÇÃO] Chunks que precisam de revisão manual:")
            for chunk_id in self.unfixable_chunks:
                lines.append(f"  - {chunk_id}")

        return "\n".join(lines)


class PostIngestionValidator:
    """Validador pós-ingestão com correção automática."""

    VALID_THESIS_TYPES = [
        'definicao', 'procedimento', 'obrigacao',
        'proibicao', 'excecao', 'prazo', 'disposicao'
    ]

    def __init__(self, gpu_server: str = None, milvus_host: str = None, milvus_port: str = "19530"):
        """
        Inicializa o validador.

        Args:
            gpu_server: URL do servidor GPU para correções LLM
            milvus_host: Host do Milvus
            milvus_port: Porta do Milvus
        """
        import os
        self.gpu_server = gpu_server or os.getenv('GPU_SERVER', 'http://195.26.233.70:55278')
        self.milvus_host = milvus_host or os.getenv('MILVUS_HOST', '77.37.43.160')
        self.milvus_port = milvus_port
        self._collection = None

    def _connect_milvus(self):
        """Conecta ao Milvus."""
        from pymilvus import connections, Collection
        connections.connect(host=self.milvus_host, port=self.milvus_port)
        self._collection = Collection('leis_v3')
        self._collection.load()

    def _disconnect_milvus(self):
        """Desconecta do Milvus."""
        from pymilvus import connections
        connections.disconnect('default')
        self._collection = None

    def validate_chunk(self, chunk: dict) -> list:
        """
        Valida um chunk individual.

        Returns:
            Lista de ValidationError encontrados
        """
        errors = []
        chunk_id = chunk.get('chunk_id', 'unknown')

        # 1. Validar thesis_type
        thesis_type = chunk.get('thesis_type', '')
        if thesis_type:
            # Verifica se contém markdown
            if '**' in thesis_type or '*' in thesis_type:
                # Tenta extrair o tipo correto
                clean_type = re.sub(r'\*+', '', thesis_type).strip().lower()
                suggested = None
                if clean_type in self.VALID_THESIS_TYPES:
                    suggested = clean_type

                errors.append(ValidationError(
                    chunk_id=chunk_id,
                    error_type=ErrorType.THESIS_TYPE_MARKDOWN,
                    field='thesis_type',
                    current_value=thesis_type,
                    suggested_fix=suggested,
                    auto_fixable=suggested is not None
                ))
            elif thesis_type.lower() not in self.VALID_THESIS_TYPES:
                errors.append(ValidationError(
                    chunk_id=chunk_id,
                    error_type=ErrorType.THESIS_TYPE_INVALID,
                    field='thesis_type',
                    current_value=thesis_type,
                    suggested_fix='disposicao',  # fallback
                    auto_fixable=True
                ))

        # 2. Validar thesis_text
        thesis_text = chunk.get('thesis_text', '')
        if not thesis_text and thesis_type:
            errors.append(ValidationError(
                chunk_id=chunk_id,
                error_type=ErrorType.THESIS_TEXT_EMPTY,
                field='thesis_text',
                current_value='',
                suggested_fix=None,
                auto_fixable=False  # Precisa de LLM
            ))
        elif thesis_text and '**' in thesis_text:
            clean_text = re.sub(r'\*+', '', thesis_text).strip()
            errors.append(ValidationError(
                chunk_id=chunk_id,
                error_type=ErrorType.THESIS_TEXT_MARKDOWN,
                field='thesis_text',
                current_value=thesis_text[:100],
                suggested_fix=clean_text,
                auto_fixable=True
            ))

        # 3. Validar enriched_text
        enriched_text = chunk.get('enriched_text', '')
        if not enriched_text:
            errors.append(ValidationError(
                chunk_id=chunk_id,
                error_type=ErrorType.ENRICHED_TEXT_EMPTY,
                field='enriched_text',
                current_value='',
                auto_fixable=False
            ))

        # 4. Validar context_header
        context_header = chunk.get('context_header', '')
        if not context_header:
            errors.append(ValidationError(
                chunk_id=chunk_id,
                error_type=ErrorType.CONTEXT_HEADER_EMPTY,
                field='context_header',
                current_value='',
                auto_fixable=False
            ))

        return errors

    def validate_document(self, document_id: str, auto_fix: bool = True) -> ValidationResult:
        """
        Valida todos os chunks de um documento.

        Args:
            document_id: ID do documento (ex: "IN-65-2021")
            auto_fix: Se True, tenta corrigir erros automaticamente

        Returns:
            ValidationResult com detalhes da validação
        """
        result = ValidationResult(document_id=document_id)

        try:
            self._connect_milvus()

            # Busca todos os chunks do documento
            chunks = self._collection.query(
                expr=f'document_id == "{document_id}"',
                output_fields=['*'],
                limit=10000
            )

            result.total_chunks = len(chunks)
            logger.info(f"Validando {len(chunks)} chunks de {document_id}")

            for chunk in chunks:
                chunk_id = chunk.get('chunk_id', 'unknown')
                errors = self.validate_chunk(chunk)

                if not errors:
                    result.valid_chunks += 1
                else:
                    result.errors.extend(errors)

                    if auto_fix:
                        fixed = self._auto_fix_chunk(chunk, errors)
                        if fixed:
                            result.fixed_chunks.append(chunk_id)
                            result.valid_chunks += 1
                        else:
                            result.unfixable_chunks.append(chunk_id)
                    else:
                        result.unfixable_chunks.append(chunk_id)

            self._collection.flush()

        finally:
            self._disconnect_milvus()

        return result

    def _auto_fix_chunk(self, chunk: dict, errors: list) -> bool:
        """
        Tenta corrigir um chunk automaticamente.

        Returns:
            True se todas as correções foram aplicadas
        """
        chunk_id = chunk.get('chunk_id')
        needs_llm = False
        updates = {}

        for error in errors:
            if error.auto_fixable and error.suggested_fix:
                updates[error.field] = error.suggested_fix
                logger.info(f"[AUTO-FIX] {chunk_id}: {error.field} = '{error.suggested_fix}'")
            elif error.error_type in [ErrorType.THESIS_TEXT_EMPTY, ErrorType.ENRICHED_TEXT_EMPTY]:
                needs_llm = True

        # Se precisa de LLM, chama para gerar campos faltantes
        if needs_llm:
            try:
                llm_updates = self._fix_with_llm(chunk)
                updates.update(llm_updates)
            except Exception as e:
                logger.error(f"Erro ao corrigir {chunk_id} via LLM: {e}")
                return False

        # Aplica as correções no Milvus
        if updates:
            try:
                self._update_chunk(chunk, updates)
                return True
            except Exception as e:
                logger.error(f"Erro ao atualizar {chunk_id}: {e}")
                return False

        return len([e for e in errors if not e.auto_fixable]) == 0

    def _fix_with_llm(self, chunk: dict) -> dict:
        """
        Usa LLM para corrigir campos faltantes.

        Returns:
            Dict com campos corrigidos
        """
        import requests

        chunk_id = chunk.get('chunk_id')
        text = chunk.get('text', '')[:2000]

        logger.info(f"[LLM-FIX] Gerando thesis para {chunk_id}")

        system_prompt = 'Voce e um assistente especializado em direito administrativo brasileiro. Responda sempre em portugues do Brasil, de forma direta e concisa. /no_think'

        resp = requests.post(f'{self.gpu_server}/llm/chat', json={
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'Analise este dispositivo legal:\n\n{text}\n\nResponda no formato:\nTIPO: [uma palavra: definicao/procedimento/obrigacao/proibicao/excecao/prazo]\nRESUMO: [uma frase resumindo o que o artigo determina]\n\nNao use markdown.'}
            ],
            'temperature': 0.0, 'max_tokens': 150
        }, timeout=120)

        content = resp.json().get('content', '')
        # Remove tags <think>
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        content = content.strip()

        # Parse resposta
        thesis_type, thesis_text = self._parse_thesis_response(content)

        # Gera embedding do thesis_text
        embed_resp = requests.post(f'{self.gpu_server}/embed/hybrid', json={
            'texts': [thesis_text]
        }, timeout=60)
        thesis_vector = embed_resp.json()['dense'][0]

        return {
            'thesis_type': thesis_type,
            'thesis_text': thesis_text[:5000],
            'thesis_vector': thesis_vector
        }

    def _parse_thesis_response(self, text: str) -> tuple:
        """Parse da resposta do LLM."""
        text = re.sub(r'\*\*', '', text)

        thesis_type = 'disposicao'
        thesis_text = text

        tipo_match = re.search(r'TIPO:?\s*(\w+)', text, re.IGNORECASE)
        if tipo_match:
            tipo_encontrado = tipo_match.group(1).lower()
            if tipo_encontrado in self.VALID_THESIS_TYPES:
                thesis_type = tipo_encontrado

        resumo_match = re.search(r'RESUMO:?\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if resumo_match:
            thesis_text = resumo_match.group(1).strip()
        else:
            thesis_text = re.sub(r'^.*?TIPO:?\s*\w+\s*', '', text, flags=re.IGNORECASE).strip()

        return thesis_type, thesis_text

    def _update_chunk(self, chunk: dict, updates: dict):
        """Atualiza um chunk no Milvus."""
        chunk_id = chunk.get('chunk_id')

        # Prepara row sem 'id' e com updates
        row = {}
        for k, v in chunk.items():
            if k != 'id':
                row[k] = v

        row.update(updates)

        # Delete e insert
        self._collection.delete(expr=f'chunk_id == "{chunk_id}"')
        self._collection.insert([row])

        logger.info(f"[UPDATED] {chunk_id}: {list(updates.keys())}")


def run_validation(document_id: str, auto_fix: bool = True, gpu_server: str = None):
    """
    Função auxiliar para executar validação.

    Args:
        document_id: ID do documento
        auto_fix: Se True, corrige erros automaticamente
        gpu_server: URL do servidor GPU

    Returns:
        ValidationResult
    """
    validator = PostIngestionValidator(gpu_server=gpu_server)
    result = validator.validate_document(document_id, auto_fix=auto_fix)

    print(result.summary())

    if result.has_errors:
        print("\nDetalhes dos erros:")
        for error in result.errors:
            print(f"  [{error.error_type.value}] {error.chunk_id}")
            print(f"    Campo: {error.field}")
            print(f"    Valor: {error.current_value[:50]}...")
            if error.suggested_fix:
                print(f"    Sugestão: {error.suggested_fix}")
            print()

    return result


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    doc_id = sys.argv[1] if len(sys.argv) > 1 else "IN-65-2021"
    auto_fix = '--no-fix' not in sys.argv

    run_validation(doc_id, auto_fix=auto_fix)
