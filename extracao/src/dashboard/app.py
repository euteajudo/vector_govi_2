"""
Dashboard Streamlit para visualização de métricas do Pipeline RAG.

Páginas:
- Visão Geral: Métricas da collection e status dos serviços
- Busca: Interface para testar busca híbrida
- Ingestão: Métricas de ingestão de documentos
- Chunks: Explorador de chunks indexados

Uso:
    streamlit run extracao/src/dashboard/app.py
"""

import sys
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Optional

# Configuração da página
st.set_page_config(
    page_title="Pipeline RAG - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONEXÕES E CACHE
# =============================================================================

@st.cache_resource
def get_milvus_connection():
    """Conecta ao Milvus (cached)."""
    try:
        from pymilvus import connections, Collection, utility

        connections.connect(
            alias="default",
            host="localhost",
            port="19530",
        )
        return True
    except Exception as e:
        st.error(f"Erro ao conectar ao Milvus: {e}")
        return False


@st.cache_resource
def get_collection_info():
    """Obtém informações da collection."""
    try:
        from pymilvus import Collection, utility

        if not utility.has_collection("leis_v3"):
            return None

        collection = Collection("leis_v3")
        collection.load()

        return {
            "name": "leis_v3",
            "num_entities": collection.num_entities,
            "schema": collection.schema,
            "indexes": collection.indexes,
        }
    except Exception as e:
        st.error(f"Erro ao obter collection: {e}")
        return None


def get_collection_stats():
    """Obtém estatísticas detalhadas da collection."""
    try:
        from pymilvus import Collection, connections

        # Garante que a conexão existe
        if not connections.has_connection("default"):
            get_milvus_connection()

        collection = Collection("leis_v3")
        collection.load()

        # Busca todos os chunks para estatísticas
        results = collection.query(
            expr="id > 0",
            output_fields=[
                "device_type", "tipo_documento", "ano",
                "article_number", "document_id"
            ],
            limit=10000,
        )

        if not results:
            return None

        df = pd.DataFrame(results)

        return {
            "total": len(df),
            "by_device_type": df["device_type"].value_counts().to_dict() if "device_type" in df else {},
            "by_tipo_documento": df["tipo_documento"].value_counts().to_dict() if "tipo_documento" in df else {},
            "by_ano": df["ano"].value_counts().to_dict() if "ano" in df else {},
            "documents": df["document_id"].nunique() if "document_id" in df else 0,
            "articles": df[df["device_type"] == "article"]["article_number"].nunique() if "device_type" in df else 0,
        }
    except Exception as e:
        st.error(f"Erro ao obter estatísticas: {e}")
        return None


# =============================================================================
# PÁGINA: VISÃO GERAL
# =============================================================================

def page_overview():
    """Página de visão geral do sistema."""
    st.header("Visão Geral do Sistema")

    # Status dos serviços
    st.subheader("Status dos Serviços")

    col1, col2, col3 = st.columns(3)

    with col1:
        milvus_ok = get_milvus_connection()
        if milvus_ok:
            st.success("Milvus: Conectado")
        else:
            st.error("Milvus: Offline")

    with col2:
        # Verifica vLLM
        try:
            import requests
            resp = requests.get("http://localhost:8000/health", timeout=2)
            if resp.status_code == 200:
                st.success("vLLM: Online")
            else:
                st.warning("vLLM: Resposta inesperada")
        except:
            st.error("vLLM: Offline")

    with col3:
        # Verifica BGE-M3
        try:
            from embeddings import BGEM3Embedder
            st.success("BGE-M3: Disponível")
        except:
            st.warning("BGE-M3: Não carregado")

    st.divider()

    # Informações da Collection
    st.subheader("Collection Milvus")

    info = get_collection_info()
    if info:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Collection", info["name"])

        with col2:
            st.metric("Total de Chunks", f"{info['num_entities']:,}")

        with col3:
            st.metric("Campos", len(info["schema"].fields))

        # Estatísticas detalhadas
        stats = get_collection_stats()
        if stats:
            st.divider()
            st.subheader("Estatísticas")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Documentos", stats["documents"])

            with col2:
                st.metric("Artigos", stats["articles"])

            with col3:
                st.metric("Total Chunks", stats["total"])

            with col4:
                # Calcula cobertura
                article_count = stats["by_device_type"].get("article", 0)
                device_count = stats["total"] - article_count
                st.metric("Dispositivos", device_count)

            # Gráficos
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Chunks por Tipo**")
                if stats["by_device_type"]:
                    df = pd.DataFrame(
                        list(stats["by_device_type"].items()),
                        columns=["Tipo", "Quantidade"]
                    )
                    st.bar_chart(df.set_index("Tipo"))

            with col2:
                st.write("**Chunks por Documento**")
                if stats["by_tipo_documento"]:
                    df = pd.DataFrame(
                        list(stats["by_tipo_documento"].items()),
                        columns=["Tipo", "Quantidade"]
                    )
                    st.bar_chart(df.set_index("Tipo"))
    else:
        st.warning("Collection 'leis_v3' não encontrada. Execute o pipeline de ingestão primeiro.")


# =============================================================================
# PÁGINA: BUSCA
# =============================================================================

def page_search():
    """Página de busca híbrida."""
    st.header("Busca Híbrida")

    # Verifica conexão
    if not get_milvus_connection():
        st.error("Milvus não está conectado.")
        return

    # Configurações na sidebar
    with st.sidebar:
        st.subheader("Configurações de Busca")

        top_k = st.slider("Top K", 1, 20, 5)
        use_reranker = st.checkbox("Usar Reranker", value=False)

        st.divider()

        # Filtros
        st.subheader("Filtros")
        filter_tipo = st.selectbox(
            "Tipo de Documento",
            ["Todos", "LEI", "DECRETO", "IN"],
        )
        filter_ano = st.number_input(
            "Ano",
            min_value=0,
            max_value=2030,
            value=0,
            help="0 = Todos os anos"
        )

    # Campo de busca
    query = st.text_input(
        "Digite sua consulta:",
        placeholder="Ex: Como fazer pesquisa de preços em contratações públicas?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("Buscar", type="primary")

    if search_button and query:
        with st.spinner("Buscando..."):
            try:
                from search import HybridSearcher, SearchConfig, SearchFilter
                from search.config import RerankMode

                # Configura busca
                config = SearchConfig.default()
                if not use_reranker:
                    config.rerank_mode = RerankMode.NONE

                # Prepara filtros
                filters = None
                if filter_tipo != "Todos" or filter_ano > 0:
                    filters = SearchFilter(
                        document_type=filter_tipo if filter_tipo != "Todos" else None,
                        year=filter_ano if filter_ano > 0 else None,
                    )

                # Executa busca
                with HybridSearcher(config) as searcher:
                    result = searcher.search(
                        query=query,
                        top_k=top_k,
                        filters=filters,
                        use_reranker=use_reranker,
                    )

                # Mostra métricas
                st.divider()
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Resultados", result.total_found)
                with col2:
                    st.metric("Stage 1", f"{result.stage1_time_ms:.0f}ms")
                with col3:
                    st.metric("Stage 2", f"{result.stage2_time_ms:.0f}ms")
                with col4:
                    total_time = result.stage1_time_ms + result.stage2_time_ms
                    st.metric("Total", f"{total_time:.0f}ms")

                st.divider()

                # Mostra resultados
                if result.hits:
                    for i, hit in enumerate(result.hits, 1):
                        device = hit.device_type or "article"
                        with st.expander(
                            f"#{i} - Art. {hit.article_number} | "
                            f"Score: {hit.final_score:.4f} | "
                            f"Tipo: {device}",
                            expanded=(i <= 3)
                        ):
                            # Metadados
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Documento:** {hit.document_number}")
                            with col2:
                                st.write(f"**Tipo:** {hit.document_type}")
                            with col3:
                                st.write(f"**Chunk ID:** {hit.chunk_id}")

                            # Contexto
                            if hit.context_header:
                                st.info(f"**Contexto:** {hit.context_header}")

                            # Texto
                            st.write("**Texto:**")
                            st.markdown(f"```\n{hit.text}\n```")

                            # Thesis
                            if hit.thesis_text:
                                st.write("**Tese:**")
                                st.write(hit.thesis_text)

                            # Scores detalhados
                            st.write("**Scores:**")
                            scores_df = pd.DataFrame([{
                                "Milvus Score": f"{hit.score:.4f}",
                                "Rerank Score": f"{hit.rerank_score:.4f}" if hit.rerank_score else "N/A",
                                "Final Score": f"{hit.final_score:.4f}",
                            }])
                            st.dataframe(scores_df, hide_index=True)
                else:
                    st.warning("Nenhum resultado encontrado.")

            except Exception as e:
                st.error(f"Erro na busca: {e}")
                import traceback
                st.code(traceback.format_exc())


# =============================================================================
# PÁGINA: INGESTÃO
# =============================================================================

def page_ingestion():
    """Página de métricas de ingestão."""
    st.header("Métricas de Ingestão")

    # Carregar arquivo de métricas
    st.subheader("Carregar Métricas")

    uploaded_file = st.file_uploader(
        "Selecione um arquivo JSON de métricas",
        type=["json"],
    )

    # Ou usar métricas de exemplo
    use_example = st.checkbox("Usar métricas de exemplo")

    metrics_data = None

    if uploaded_file:
        try:
            metrics_data = json.load(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")

    elif use_example:
        # Métricas de exemplo (IN 65/2021)
        metrics_data = {
            "ingestion_id": "IN-65-2021-001",
            "timestamp": "2024-12-23T10:30:00Z",
            "status": "completed",
            "total_duration_seconds": 30.02,
            "phases": [
                {"name": "load", "duration_seconds": 0.01, "items_processed": 1, "errors": 0},
                {"name": "parsing", "duration_seconds": 0.05, "items_processed": 57, "errors": 0},
                {"name": "extraction", "duration_seconds": 12.08, "items_processed": 11, "errors": 0},
                {"name": "materialization", "duration_seconds": 0.02, "items_processed": 47, "errors": 0},
                {"name": "embedding", "duration_seconds": 16.28, "items_processed": 47, "errors": 0},
                {"name": "indexing", "duration_seconds": 1.65, "items_processed": 47, "errors": 0},
            ],
            "document": {
                "document_id": "IN-65-2021",
                "tipo_documento": "IN",
                "numero": "65",
                "ano": 2021,
                "articles": {
                    "total": 11,
                    "valid": 11,
                    "suspect": 0,
                    "invalid": 0,
                    "success_rate": "100%",
                },
                "coverage": {
                    "paragrafos": "19/19",
                    "paragrafos_pct": "100%",
                    "incisos": "17/17",
                    "incisos_pct": "100%",
                },
                "tokens": {
                    "prompt": 5500,
                    "completion": 1100,
                    "total": 6600,
                },
                "chunks": {
                    "total": 47,
                    "articles": 11,
                    "paragraphs": 19,
                    "incisos": 17,
                },
            },
            "articles": [
                {"article_id": f"ART-{str(i).zfill(3)}", "article_number": str(i), "status": "valid",
                 "coverage": {"paragrafos": "2/2", "incisos": "3/3"}, "tokens": {"total": 600}}
                for i in range(1, 12)
            ],
        }

    if metrics_data:
        st.divider()

        # Header
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ID", metrics_data["ingestion_id"])
        with col2:
            st.metric("Status", metrics_data["status"].upper())
        with col3:
            st.metric("Tempo Total", f"{metrics_data['total_duration_seconds']:.2f}s")

        st.divider()

        # Documento
        doc = metrics_data["document"]
        st.subheader(f"Documento: {doc['tipo_documento']} {doc['numero']}/{doc['ano']}")

        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Artigos",
                doc["articles"]["total"],
                delta=f"{doc['articles']['success_rate']} válidos"
            )

        with col2:
            st.metric(
                "Parágrafos",
                doc["coverage"]["paragrafos"],
                delta=doc["coverage"]["paragrafos_pct"]
            )

        with col3:
            st.metric(
                "Incisos",
                doc["coverage"]["incisos"],
                delta=doc["coverage"]["incisos_pct"]
            )

        with col4:
            st.metric(
                "Chunks",
                doc["chunks"]["total"],
            )

        st.divider()

        # Fases do Pipeline
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fases do Pipeline")
            phases_df = pd.DataFrame(metrics_data["phases"])
            phases_df["duration"] = phases_df["duration_seconds"].apply(lambda x: f"{x:.2f}s")
            st.dataframe(
                phases_df[["name", "duration", "items_processed", "errors"]],
                hide_index=True,
                use_container_width=True,
            )

            # Gráfico de tempo por fase
            st.bar_chart(
                phases_df.set_index("name")["duration_seconds"]
            )

        with col2:
            st.subheader("Chunks por Tipo")
            chunks = doc["chunks"]
            chunks_df = pd.DataFrame([
                {"Tipo": "Articles", "Quantidade": chunks["articles"]},
                {"Tipo": "Paragraphs", "Quantidade": chunks["paragraphs"]},
                {"Tipo": "Incisos", "Quantidade": chunks["incisos"]},
            ])
            st.dataframe(chunks_df, hide_index=True, use_container_width=True)
            st.bar_chart(chunks_df.set_index("Tipo"))

        st.divider()

        # Tokens LLM
        st.subheader("Consumo de Tokens LLM")
        tokens = doc["tokens"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Prompt", f"{tokens['prompt']:,}")
        with col2:
            st.metric("Completion", f"{tokens['completion']:,}")
        with col3:
            st.metric("Total", f"{tokens['total']:,}")
        with col4:
            # Custo estimado (referência API)
            cost = (tokens['total'] / 1000) * 0.001
            st.metric("Custo Est.", f"${cost:.4f}")

        st.divider()

        # Artigos individuais
        st.subheader("Artigos")

        if metrics_data.get("articles"):
            articles_df = pd.DataFrame(metrics_data["articles"])

            # Adiciona indicador visual de status
            def status_icon(status):
                if status == "valid":
                    return "✅"
                elif status == "suspect":
                    return "⚠️"
                return "❌"

            articles_df["status_icon"] = articles_df["status"].apply(status_icon)

            st.dataframe(
                articles_df[["article_id", "status_icon", "status"]],
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("Carregue um arquivo de métricas ou marque 'Usar métricas de exemplo'.")


# =============================================================================
# PÁGINA: EXPLORADOR DE CHUNKS
# =============================================================================

def page_chunks():
    """Página para explorar chunks indexados."""
    st.header("Explorador de Chunks")

    if not get_milvus_connection():
        st.error("Milvus não está conectado.")
        return

    # Filtros
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_device = st.selectbox(
            "Tipo de Dispositivo",
            ["Todos", "article", "paragraph", "inciso", "alinea"],
        )

    with col2:
        filter_article = st.text_input(
            "Número do Artigo",
            placeholder="Ex: 5",
        )

    with col3:
        limit = st.slider("Limite", 10, 100, 20)

    if st.button("Carregar Chunks", type="primary"):
        with st.spinner("Carregando..."):
            try:
                from pymilvus import Collection, connections

                # Garante que a conexão existe
                if not connections.has_connection("default"):
                    get_milvus_connection()

                collection = Collection("leis_v3")
                collection.load()

                # Monta expressão de filtro
                expr_parts = []
                if filter_device != "Todos":
                    expr_parts.append(f'device_type == "{filter_device}"')
                if filter_article:
                    expr_parts.append(f'article_number == "{filter_article}"')

                expr = " && ".join(expr_parts) if expr_parts else "id > 0"

                # Busca chunks
                results = collection.query(
                    expr=expr,
                    output_fields=[
                        "chunk_id", "span_id", "device_type", "article_number",
                        "text", "context_header", "thesis_text", "thesis_type",
                        "document_id", "tipo_documento", "parent_chunk_id",
                    ],
                    limit=limit,
                )

                if results:
                    st.success(f"Encontrados {len(results)} chunks")

                    for i, chunk in enumerate(results, 1):
                        with st.expander(
                            f"{chunk.get('span_id', 'N/A')} | "
                            f"Art. {chunk.get('article_number', 'N/A')} | "
                            f"{chunk.get('device_type', 'N/A')}",
                            expanded=(i == 1)
                        ):
                            # Metadados
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Chunk ID:** `{chunk.get('chunk_id', 'N/A')}`")
                                st.write(f"**Parent:** `{chunk.get('parent_chunk_id', 'N/A') or 'Nenhum (raiz)'}`")
                            with col2:
                                st.write(f"**Documento:** {chunk.get('document_id', 'N/A')}")
                                st.write(f"**Tipo Doc:** {chunk.get('tipo_documento', 'N/A')}")

                            # Contexto
                            if chunk.get("context_header"):
                                st.info(f"**Contexto:** {chunk['context_header']}")

                            # Texto
                            st.write("**Texto:**")
                            st.markdown(f"```\n{chunk.get('text', 'N/A')[:1000]}...\n```")

                            # Thesis
                            col1, col2 = st.columns(2)
                            with col1:
                                if chunk.get("thesis_type"):
                                    st.write(f"**Tipo de Tese:** {chunk['thesis_type']}")
                            with col2:
                                if chunk.get("thesis_text"):
                                    st.write(f"**Tese:** {chunk['thesis_text'][:200]}...")
                else:
                    st.warning("Nenhum chunk encontrado com os filtros selecionados.")

            except Exception as e:
                st.error(f"Erro ao carregar chunks: {e}")
                import traceback
                st.code(traceback.format_exc())


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Função principal do dashboard."""

    # Sidebar - Navegação
    st.sidebar.title("Pipeline RAG")
    st.sidebar.caption("Dashboard de Métricas")

    page = st.sidebar.radio(
        "Navegação",
        ["Visão Geral", "Busca", "Ingestão", "Chunks"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Atualizado: {datetime.now().strftime('%H:%M:%S')}")

    # Renderiza página selecionada
    if page == "Visão Geral":
        page_overview()
    elif page == "Busca":
        page_search()
    elif page == "Ingestão":
        page_ingestion()
    elif page == "Chunks":
        page_chunks()


if __name__ == "__main__":
    main()
