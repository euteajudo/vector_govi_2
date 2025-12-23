"""
Pipeline de Extração - Combina LangGraph + Extractor Simples.

Este módulo fornece o pipeline híbrido que usa:
- LangGraph para orquestração, validação e retry
- Extractor Simples (Pydantic) para extração principal

Uso:
    from pipeline import run_hybrid_pipeline
    
    result = run_hybrid_pipeline(
        input_path="documento.pdf",
        max_attempts=3,
        verbose=True,
    )
    
    print(result["json"])
    print(f"Score: {result['quality_score']:.1%}")
"""

from .hybrid_pipeline import (
    run_hybrid_pipeline,
    build_hybrid_pipeline,
    HybridState,
    PipelineStatus,
)

__all__ = [
    "run_hybrid_pipeline",
    "build_hybrid_pipeline",
    "HybridState",
    "PipelineStatus",
]

