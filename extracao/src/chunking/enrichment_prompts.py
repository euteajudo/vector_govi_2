"""
Prompts para enriquecimento de chunks com LLM.

Implementa a técnica de Contextual Retrieval da Anthropic:
- Gera perguntas sintéticas que o chunk responde
- Cria resumos/teses do conteúdo
- Adiciona contexto hierárquico

Referência: https://www.anthropic.com/news/contextual-retrieval
"""

# =============================================================================
# PROMPT DE ENRIQUECIMENTO PRINCIPAL
# =============================================================================

ENRICHMENT_SYSTEM_PROMPT = """Você é um especialista em direito administrativo brasileiro.
Sua tarefa é analisar dispositivos legais e gerar metadados estruturados para um sistema de busca semântica.

REGRAS:
1. Seja objetivo e técnico
2. Use linguagem clara, sem juridiquês desnecessário
3. Foque no conteúdo normativo, não em repetir o texto
4. Perguntas devem ser práticas, como um servidor público faria
5. Responda APENAS com JSON válido, sem explicações"""


ENRICHMENT_USER_PROMPT = """Analise este dispositivo legal e gere os metadados solicitados.

═══════════════════════════════════════════════════════════════════════════════
DOCUMENTO: {document_type} nº {number}/{year}
ÓRGÃO EMISSOR: {issuing_body}
CAPÍTULO: {chapter_number} - {chapter_title}
ARTIGO: {article_number}{article_title_suffix}
═══════════════════════════════════════════════════════════════════════════════

TEXTO DO DISPOSITIVO:
{text}

═══════════════════════════════════════════════════════════════════════════════

Gere um JSON com os seguintes campos:

{{
    "context_header": "Frase de 1-2 linhas contextualizando este dispositivo no documento (máx 200 caracteres)",
    "thesis_text": "Resumo objetivo do que este dispositivo determina ou define, sem repetir o texto literal (máx 500 caracteres)",
    "thesis_type": "escolha UM: definicao | procedimento | prazo | requisito | competencia | vedacao | excecao | sancao | disposicao",
    "synthetic_questions": [
        "Pergunta prática 1 que um servidor público faria",
        "Pergunta prática 2",
        "Pergunta prática 3",
        "Pergunta prática 4",
        "Pergunta prática 5"
    ]
}}

INSTRUÇÕES PARA CADA CAMPO:

• context_header: Situe o leitor. Ex: "Este artigo da IN 58/2022 define os conceitos básicos para elaboração de ETP no âmbito federal"

• thesis_text: Capture a essência normativa. NÃO repita o texto. Ex: "Estabelece que ETP, requisitante e área técnica são conceitos fundamentais, podendo ser exercidos pelo mesmo servidor"

• thesis_type:
  - definicao: Define conceitos, termos técnicos
  - procedimento: Estabelece como fazer algo, etapas
  - prazo: Define prazos, cronogramas
  - requisito: Estabelece condições, exigências
  - competencia: Define quem pode/deve fazer
  - vedacao: Proíbe algo, estabelece limitações
  - excecao: Estabelece exceções a regras
  - sancao: Define penalidades, consequências
  - disposicao: Disposições gerais que não se encaixam

• synthetic_questions: Perguntas que um servidor faria ao consultar a norma. Ex:
  - "O que é ETP segundo a IN 58?"
  - "Quem pode ser requisitante?"
  - "O requisitante pode ser também área técnica?"

Responda APENAS com o JSON, sem texto adicional."""


# =============================================================================
# PROMPT PARA BATCH (múltiplos chunks)
# =============================================================================

BATCH_ENRICHMENT_SYSTEM_PROMPT = """Você é um especialista em direito administrativo brasileiro.
Analise múltiplos dispositivos legais e gere metadados para cada um.
Responda com uma lista JSON contendo os metadados de cada dispositivo na ordem recebida."""


BATCH_ENRICHMENT_USER_PROMPT = """Analise os seguintes dispositivos legais e gere metadados para cada um.

═══════════════════════════════════════════════════════════════════════════════
DOCUMENTO: {document_type} nº {number}/{year}
ÓRGÃO EMISSOR: {issuing_body}
═══════════════════════════════════════════════════════════════════════════════

{chunks_text}

═══════════════════════════════════════════════════════════════════════════════

Para CADA dispositivo acima, gere um objeto JSON com:
- context_header (máx 200 chars)
- thesis_text (máx 500 chars)
- thesis_type (definicao|procedimento|prazo|requisito|competencia|vedacao|excecao|sancao|disposicao)
- synthetic_questions (lista de 5 perguntas práticas)

Responda com uma lista JSON na mesma ordem dos dispositivos:
[
    {{"context_header": "...", "thesis_text": "...", "thesis_type": "...", "synthetic_questions": [...]}},
    ...
]"""


# =============================================================================
# TEMPLATE PARA MONTAR CHUNK NO BATCH
# =============================================================================

CHUNK_TEMPLATE = """
---[ DISPOSITIVO {index} ]---
CAPÍTULO: {chapter_number} - {chapter_title}
ARTIGO: {article_number}{article_title_suffix}

{text}
"""


# =============================================================================
# PROMPT PARA CLASSIFICAÇÃO DE TIPO (mais leve)
# =============================================================================

CLASSIFICATION_PROMPT = """Classifique o tipo deste dispositivo legal:

TEXTO: {text}

Responda apenas com uma palavra:
definicao | procedimento | prazo | requisito | competencia | vedacao | excecao | sancao | disposicao"""


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def build_enrichment_prompt(
    text: str,
    document_type: str,
    number: str,
    year: int,
    issuing_body: str,
    chapter_number: str,
    chapter_title: str,
    article_number: str,
    article_title: str | None = None,
) -> tuple[str, str]:
    """
    Constrói prompt de enriquecimento para um chunk.

    Returns:
        Tuple (system_prompt, user_prompt)
    """
    article_title_suffix = f" - {article_title}" if article_title else ""

    user_prompt = ENRICHMENT_USER_PROMPT.format(
        document_type=document_type,
        number=number,
        year=year,
        issuing_body=issuing_body,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        article_number=article_number,
        article_title_suffix=article_title_suffix,
        text=text,
    )

    return ENRICHMENT_SYSTEM_PROMPT, user_prompt


def build_batch_enrichment_prompt(
    chunks: list[dict],
    document_type: str,
    number: str,
    year: int,
    issuing_body: str,
) -> tuple[str, str]:
    """
    Constrói prompt de enriquecimento em batch para múltiplos chunks.

    Args:
        chunks: Lista de dicts com keys: text, chapter_number, chapter_title,
                article_number, article_title

    Returns:
        Tuple (system_prompt, user_prompt)
    """
    chunks_text_parts = []

    for i, chunk in enumerate(chunks, 1):
        article_title_suffix = (
            f" - {chunk['article_title']}" if chunk.get("article_title") else ""
        )

        chunk_text = CHUNK_TEMPLATE.format(
            index=i,
            chapter_number=chunk.get("chapter_number", ""),
            chapter_title=chunk.get("chapter_title", ""),
            article_number=chunk.get("article_number", ""),
            article_title_suffix=article_title_suffix,
            text=chunk["text"],
        )
        chunks_text_parts.append(chunk_text)

    chunks_text = "\n".join(chunks_text_parts)

    user_prompt = BATCH_ENRICHMENT_USER_PROMPT.format(
        document_type=document_type,
        number=number,
        year=year,
        issuing_body=issuing_body,
        chunks_text=chunks_text,
    )

    return BATCH_ENRICHMENT_SYSTEM_PROMPT, user_prompt


def build_enriched_text(
    text: str,
    context_header: str,
    synthetic_questions: list[str],
) -> str:
    """
    Monta texto enriquecido para embedding.

    O texto enriquecido inclui:
    - Contexto no início
    - Texto original
    - Perguntas que o trecho responde

    Isso melhora o recall na busca semântica (Contextual Retrieval).
    """
    questions_text = "\n".join(f"- {q}" for q in synthetic_questions)

    return f"""[CONTEXTO: {context_header}]

{text}

Perguntas que este trecho responde:
{questions_text}"""


def parse_enrichment_response(response: str) -> dict:
    """
    Parseia resposta do LLM de enriquecimento.

    Args:
        response: Resposta JSON do LLM

    Returns:
        Dict com context_header, thesis_text, thesis_type, synthetic_questions
    """
    import json
    import re

    # Tenta extrair JSON da resposta
    response = response.strip()

    # Remove markdown code blocks se presentes
    if response.startswith("```"):
        # Remove ```json e ``` finais
        response = re.sub(r"^```\w*\n?", "", response)
        response = re.sub(r"\n?```$", "", response)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Tenta encontrar JSON na resposta
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Não foi possível parsear resposta: {response[:200]}")

    # Valida campos obrigatórios
    required = ["context_header", "thesis_text", "thesis_type", "synthetic_questions"]
    for field in required:
        if field not in data:
            raise ValueError(f"Campo obrigatório ausente: {field}")

    # Normaliza synthetic_questions para string
    questions = data["synthetic_questions"]
    if isinstance(questions, list):
        data["synthetic_questions"] = "\n".join(questions)

    # Valida thesis_type
    valid_types = {
        "definicao", "procedimento", "prazo", "requisito",
        "competencia", "vedacao", "excecao", "sancao", "disposicao"
    }
    if data["thesis_type"] not in valid_types:
        data["thesis_type"] = "disposicao"

    return data


def parse_batch_enrichment_response(response: str, expected_count: int) -> list[dict]:
    """
    Parseia resposta do LLM de enriquecimento em batch.

    Args:
        response: Resposta JSON do LLM (lista)
        expected_count: Número esperado de itens

    Returns:
        Lista de dicts com metadados de cada chunk
    """
    import json
    import re

    response = response.strip()

    # Remove markdown code blocks
    if response.startswith("```"):
        response = re.sub(r"^```\w*\n?", "", response)
        response = re.sub(r"\n?```$", "", response)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Tenta encontrar array JSON
        match = re.search(r"\[[\s\S]*\]", response)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Não foi possível parsear resposta batch: {response[:200]}")

    if not isinstance(data, list):
        raise ValueError("Resposta deve ser uma lista JSON")

    if len(data) != expected_count:
        raise ValueError(
            f"Esperado {expected_count} itens, recebido {len(data)}"
        )

    # Valida e normaliza cada item
    results = []
    for item in data:
        normalized = parse_enrichment_response(json.dumps(item))
        results.append(normalized)

    return results
