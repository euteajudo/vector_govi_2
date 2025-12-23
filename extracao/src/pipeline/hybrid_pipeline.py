"""
Pipeline Híbrido: LangGraph + Extractor Simples.

Combina o melhor dos dois mundos:
- LangGraph: Orquestração, validação, retry, logging
- Extractor Simples: Extração principal com Pydantic

Arquitetura:
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH (Orquestrador)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │  DOCLING │───▶│  EXTRACTOR   │───▶│  VALIDATOR   │         │
│   │  (Parse) │    │  (Pydantic)  │    │  (Quality)   │         │
│   └──────────┘    └──────────────┘    └──────────────┘         │
│                          │                    │                 │
│                          │              ┌─────▼─────┐           │
│                          │              │  DECIDE   │           │
│                          │              └─────┬─────┘           │
│                          │                    │                 │
│                    ┌─────▼────────────────────▼─────┐           │
│                    │           FINALIZER            │           │
│                    └────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

import json
import re
from typing import TypedDict, Optional, Any
from pathlib import Path
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# Imports locais
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract import Extractor, ExtractConfig, ExtractionResult
from models.legal_document import LegalDocument, validate_extraction, count_articles


# =============================================================================
# ESTADO DO PIPELINE
# =============================================================================

class PipelineStatus(str, Enum):
    """Status do pipeline."""
    PENDING = "pending"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    FIXING = "fixing"
    COMPLETED = "completed"
    FAILED = "failed"


class HybridState(TypedDict):
    """Estado compartilhado do pipeline híbrido."""
    
    # Input
    input_path: str
    input_type: str  # "pdf" ou "markdown"
    
    # Docling output
    markdown: str
    parse_success: bool
    parse_error: Optional[str]
    
    # Extractor output
    extraction_result: Optional[dict]  # ExtractionResult serializado
    raw_json: dict
    validated_data: Optional[dict]  # LegalDocument serializado
    
    # Validação
    expected_articles: list[int]
    extracted_articles: list[int]
    missing_articles: list[int]
    quality_score: float
    is_valid: bool
    validation_errors: list[str]
    
    # Controle
    status: str
    attempt: int
    max_attempts: int
    
    # Logs
    logs: list[str]
    errors: list[str]
    
    # Timing
    start_time: str
    end_time: Optional[str]
    duration_seconds: float


# =============================================================================
# NODES
# =============================================================================

def node_parse(state: HybridState) -> HybridState:
    """
    Node 1: Parse do documento com Docling.
    
    - Se já tem markdown, pula
    - Se é PDF, usa Docling
    - Se é .md, lê direto
    """
    state["status"] = PipelineStatus.PARSING.value
    state["logs"].append("[PARSE] Iniciando...")
    
    input_path = Path(state["input_path"])
    
    # Já tem markdown?
    if state.get("markdown"):
        state["logs"].append("[PARSE] Markdown ja fornecido, pulando")
        state["parse_success"] = True
        return state
    
    # Verificar arquivo
    if not input_path.exists():
        state["parse_error"] = f"Arquivo nao encontrado: {input_path}"
        state["parse_success"] = False
        state["errors"].append(state["parse_error"])
        return state
    
    try:
        if input_path.suffix.lower() == ".md":
            # Ler markdown direto
            state["markdown"] = input_path.read_text(encoding="utf-8")
            state["input_type"] = "markdown"
            state["logs"].append(f"[PARSE] Markdown lido: {len(state['markdown']):,} chars")
        else:
            # Usar Docling para PDF
            from docling.document_converter import DocumentConverter
            
            converter = DocumentConverter()
            result = converter.convert(str(input_path))
            state["markdown"] = result.document.export_to_markdown()
            state["input_type"] = "pdf"
            state["logs"].append(f"[PARSE] PDF convertido: {len(state['markdown']):,} chars")
        
        state["parse_success"] = True
        
        # Pre-analise: contar artigos esperados
        pattern = r'Art\.?\s*(\d+)[°ºo]?'
        articles = []
        for match in re.finditer(pattern, state["markdown"]):
            art_num = int(match.group(1))
            if art_num not in articles:
                articles.append(art_num)
        state["expected_articles"] = sorted(articles)
        state["logs"].append(f"[PARSE] Artigos esperados: {len(articles)} (Art. {min(articles)} a {max(articles)})")
        
    except Exception as e:
        state["parse_error"] = str(e)
        state["parse_success"] = False
        state["errors"].append(f"[PARSE] Erro: {e}")
    
    return state


def node_extract(state: HybridState) -> HybridState:
    """
    Node 2: Extração com Extractor Simples (Pydantic).
    
    Este é o coração do pipeline - usa o Extractor que ganhou o benchmark!
    """
    state["status"] = PipelineStatus.EXTRACTING.value
    state["logs"].append(f"[EXTRACT] Tentativa {state['attempt'] + 1}/{state['max_attempts']}...")
    
    if not state["parse_success"]:
        state["errors"].append("[EXTRACT] Parse falhou, pulando extracao")
        return state
    
    try:
        # Criar extractor com config otimizada
        config = ExtractConfig.for_legal_documents()
        extractor = Extractor(config)
        
        # Extrair usando o markdown
        result = extractor.extract(
            document=state["markdown"],
            schema=LegalDocument,
            config=config,
        )
        
        # Serializar resultado
        state["extraction_result"] = {
            "success": result.success,
            "quality_score": result.quality_score,
            "extraction_time_seconds": result.extraction_time_seconds,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        
        state["raw_json"] = result.raw_data
        
        if result.data:
            state["validated_data"] = result.data.model_dump()
        
        # Coletar artigos extraidos
        extracted = []
        for chapter in result.raw_data.get("chapters", []):
            for article in chapter.get("articles", []):
                try:
                    num = int(article.get("article_number", 0))
                    if num not in extracted:
                        extracted.append(num)
                except:
                    pass
        
        state["extracted_articles"] = sorted(extracted)
        
        # Adicionar logs do extractor
        for log in result.logs:
            state["logs"].append(f"  {log}")
        
        state["logs"].append(f"[EXTRACT] Extraidos: {len(extracted)} artigos")
        
    except Exception as e:
        state["errors"].append(f"[EXTRACT] Erro: {e}")
        import traceback
        state["errors"].append(traceback.format_exc())
    
    return state


def node_validate(state: HybridState) -> HybridState:
    """
    Node 3: Validação de qualidade.
    
    - Compara artigos esperados vs extraídos
    - Valida schema Pydantic
    - Calcula score de qualidade
    """
    state["status"] = PipelineStatus.VALIDATING.value
    state["logs"].append("[VALIDATE] Validando resultado...")
    
    expected = set(state["expected_articles"])
    extracted = set(state["extracted_articles"])
    
    # Calcular diferenças
    missing = sorted(list(expected - extracted))
    extra = sorted(list(extracted - expected))
    
    state["missing_articles"] = missing
    
    # Calcular score
    if len(expected) > 0:
        score = len(expected & extracted) / len(expected)
    else:
        score = 0.0
    
    state["quality_score"] = score
    
    # Validar com Pydantic
    if state["raw_json"]:
        is_valid, errors = validate_extraction(state["raw_json"])
        state["is_valid"] = is_valid and len(missing) == 0
        state["validation_errors"] = errors
    else:
        state["is_valid"] = False
        state["validation_errors"] = ["Nenhum JSON extraido"]
    
    state["logs"].append(f"[VALIDATE] Score: {score:.1%}")
    state["logs"].append(f"[VALIDATE] Faltando: {len(missing)} artigos")
    state["logs"].append(f"[VALIDATE] Pydantic valido: {state['is_valid']}")
    
    if missing:
        state["logs"].append(f"[VALIDATE] Artigos faltando: {missing}")
    
    return state


def node_decide(state: HybridState) -> str:
    """
    Router: Decide próximo passo baseado na validação.
    
    - Se score >= 95% e válido -> finalizar
    - Se tentativas esgotadas -> finalizar
    - Senão -> tentar corrigir
    """
    score = state["quality_score"]
    is_valid = state["is_valid"]
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]
    
    if score >= 0.95 and is_valid:
        state["logs"].append("[DECIDE] -> FINALIZAR (sucesso)")
        return "finalize"
    
    if attempt >= max_attempts - 1:
        state["logs"].append(f"[DECIDE] -> FINALIZAR (max tentativas: {max_attempts})")
        return "finalize"
    
    if state["missing_articles"]:
        state["logs"].append("[DECIDE] -> CORRIGIR (artigos faltando)")
        return "fix"
    
    state["logs"].append("[DECIDE] -> FINALIZAR")
    return "finalize"


def node_fix(state: HybridState) -> HybridState:
    """
    Node 4: Correção de artigos faltantes.
    
    Extrai individualmente cada artigo faltante.
    """
    state["status"] = PipelineStatus.FIXING.value
    state["attempt"] += 1
    
    missing = state["missing_articles"]
    state["logs"].append(f"[FIX] Corrigindo {len(missing)} artigos faltantes...")
    
    if not missing:
        return state
    
    import ollama
    
    markdown = state["markdown"]
    fixed_articles = []
    
    for art_num in missing[:5]:  # Limitar a 5 por vez
        state["logs"].append(f"[FIX] Extraindo Art. {art_num}...")
        
        # Encontrar texto do artigo
        pattern = rf'Art\.?\s*{art_num}[°ºo]?\s*(.+?)(?=Art\.?\s*\d+[°ºo]?|CAPITULO|CAPÍTULO|$)'
        match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
        
        if not match:
            state["logs"].append(f"[FIX] Art. {art_num} nao encontrado no markdown")
            continue
        
        art_text = f"Art. {art_num} " + match.group(1).strip()[:3000]
        
        prompt = f"""Extraia este artigo de lei para JSON estruturado.

ARTIGO:
{art_text}

SCHEMA:
{{
  "article_number": "{art_num}",
  "title": null,
  "content": "texto do caput",
  "items": [
    {{"item_identifier": "I", "description": "texto", "sub_items": []}}
  ],
  "paragraphs": [
    {{"paragraph_identifier": "1", "content": "texto"}}
  ]
}}

REGRAS:
- article_number: apenas numero ({art_num})
- item_identifier: numeral romano (I, II, III)
- paragraph_identifier: numero (1, 2) ou "unico"
- sub_items: alineas com letras (a, b, c)

Retorne APENAS o JSON do artigo:"""

        try:
            response = ollama.chat(
                model="qwen3:8b",
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_ctx": 8192, "num_predict": 4096},
            )
            
            article = json.loads(response["message"]["content"])
            article["article_number"] = str(art_num)
            fixed_articles.append(article)
            state["logs"].append(f"[FIX] Art. {art_num} OK")
            
        except Exception as e:
            state["logs"].append(f"[FIX] Art. {art_num} ERRO: {e}")
    
    # Merge artigos corrigidos
    if fixed_articles and state["raw_json"].get("chapters"):
        # Adicionar ao primeiro capítulo (simplificação)
        state["raw_json"]["chapters"][0]["articles"].extend(fixed_articles)
        
        # Atualizar lista de extraídos
        for art in fixed_articles:
            try:
                num = int(art["article_number"])
                if num not in state["extracted_articles"]:
                    state["extracted_articles"].append(num)
            except:
                pass
        
        state["extracted_articles"].sort()
        state["logs"].append(f"[FIX] {len(fixed_articles)} artigos adicionados")
    
    return state


def node_finalize(state: HybridState) -> HybridState:
    """
    Node 5: Finalização.
    
    - Ordenar artigos por número
    - Calcular métricas finais
    - Registrar tempo total
    """
    state["status"] = PipelineStatus.COMPLETED.value
    state["end_time"] = datetime.now().isoformat()
    
    # Calcular duração
    start = datetime.fromisoformat(state["start_time"])
    end = datetime.fromisoformat(state["end_time"])
    state["duration_seconds"] = (end - start).total_seconds()
    
    # Ordenar artigos nos capítulos
    if state["raw_json"].get("chapters"):
        for chapter in state["raw_json"]["chapters"]:
            articles = chapter.get("articles", [])
            articles.sort(
                key=lambda a: int(re.search(r'\d+', str(a.get("article_number", "0"))).group() or 0)
            )
    
    state["logs"].append(f"[FINALIZE] Concluido em {state['duration_seconds']:.2f}s")
    state["logs"].append(f"[FINALIZE] Score final: {state['quality_score']:.1%}")
    state["logs"].append(f"[FINALIZE] Artigos: {len(state['extracted_articles'])}/{len(state['expected_articles'])}")
    
    return state


# =============================================================================
# CONSTRUIR PIPELINE
# =============================================================================

def build_hybrid_pipeline():
    """Constrói o grafo LangGraph do pipeline híbrido."""
    
    graph = StateGraph(HybridState)
    
    # Adicionar nodes
    graph.add_node("parse", node_parse)
    graph.add_node("extract", node_extract)
    graph.add_node("validate", node_validate)
    graph.add_node("fix", node_fix)
    graph.add_node("finalize", node_finalize)
    
    # Entry point
    graph.set_entry_point("parse")
    
    # Edges lineares
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "validate")
    
    # Edge condicional após validação
    graph.add_conditional_edges(
        "validate",
        node_decide,
        {
            "fix": "fix",
            "finalize": "finalize",
        }
    )
    
    # Fix volta para validate
    graph.add_edge("fix", "validate")
    
    # Finalize termina
    graph.add_edge("finalize", END)
    
    return graph.compile()


# =============================================================================
# API PÚBLICA
# =============================================================================

def run_hybrid_pipeline(
    input_path: str = None,
    markdown: str = None,
    max_attempts: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Executa o pipeline híbrido.
    
    Args:
        input_path: Caminho para PDF ou .md
        markdown: Markdown direto (opcional)
        max_attempts: Máximo de tentativas de correção
        verbose: Mostrar logs em tempo real
    
    Returns:
        dict com resultado da extração
    """
    # Estado inicial
    initial_state: HybridState = {
        "input_path": input_path or "",
        "input_type": "",
        "markdown": markdown or "",
        "parse_success": False,
        "parse_error": None,
        "extraction_result": None,
        "raw_json": {},
        "validated_data": None,
        "expected_articles": [],
        "extracted_articles": [],
        "missing_articles": [],
        "quality_score": 0.0,
        "is_valid": False,
        "validation_errors": [],
        "status": PipelineStatus.PENDING.value,
        "attempt": 0,
        "max_attempts": max_attempts,
        "logs": [],
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": 0.0,
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE HIBRIDO: LangGraph + Extractor Simples")
        print("=" * 70)
    
    # Construir e executar
    pipeline = build_hybrid_pipeline()
    
    final_state = None
    printed_logs = set()
    
    for state_update in pipeline.stream(initial_state):
        for node_name, node_state in state_update.items():
            final_state = node_state
            
            if verbose:
                for log in node_state.get("logs", []):
                    if log not in printed_logs:
                        print(log)
                        printed_logs.add(log)
    
    if verbose:
        print("\n" + "=" * 70)
        print("RESULTADO FINAL")
        print("=" * 70)
        print(f"  Status: {final_state['status']}")
        print(f"  Score: {final_state['quality_score']:.1%}")
        print(f"  Artigos: {len(final_state['extracted_articles'])}/{len(final_state['expected_articles'])}")
        print(f"  Tempo: {final_state['duration_seconds']:.2f}s")
        print(f"  Tentativas: {final_state['attempt'] + 1}")
        
        if final_state["errors"]:
            print(f"\n  Erros:")
            for err in final_state["errors"][:3]:
                print(f"    - {err[:100]}")
    
    return {
        "json": final_state["raw_json"],
        "validated_data": final_state["validated_data"],
        "quality_score": final_state["quality_score"],
        "is_valid": final_state["is_valid"],
        "expected_count": len(final_state["expected_articles"]),
        "extracted_count": len(final_state["extracted_articles"]),
        "missing": final_state["missing_articles"],
        "duration_seconds": final_state["duration_seconds"],
        "logs": final_state["logs"],
        "errors": final_state["errors"],
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Encontrar arquivo de teste
    output_dir = Path(__file__).parent.parent.parent / "data" / "output"
    md_files = list(output_dir.glob("*_extracted.md"))
    
    if md_files:
        test_file = md_files[0]
        print(f"Testando com: {test_file.name}")
        
        result = run_hybrid_pipeline(
            input_path=str(test_file),
            max_attempts=2,
            verbose=True,
        )
        
        # Salvar resultado
        output_file = output_dir / f"{test_file.stem}_hybrid.json"
        output_file.write_text(
            json.dumps(result["json"], indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n[SALVO] {output_file}")
    else:
        print("Nenhum arquivo .md encontrado em data/output/")

