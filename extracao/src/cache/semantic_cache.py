"""
Cache Semântico para respostas do RAG.

Usa busca HÍBRIDA (dense + sparse) para encontrar queries similares,
garantindo precisão tanto semântica quanto por keywords.

Arquitetura:
- Milvus: Armazena embeddings dense + sparse para busca híbrida
- Redis: Armazena respostas completas (JSON)

Uso:
    from cache.semantic_cache import SemanticCache

    cache = SemanticCache()

    # Verifica cache
    cached = cache.get("Quando o ETP pode ser dispensado?")
    if cached:
        return cached  # Hit! ~10ms

    # Miss - executa pipeline
    response = generator.generate(query)

    # Salva no cache
    cache.set(query, response)
"""

import json
import logging
import hashlib
import time
from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime

import redis
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
    AnnSearchRequest,
    RRFRanker,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuração do cache semântico."""

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "query_cache_v2"  # Nova versão com híbrido

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1  # DB separado para cache

    # Cache behavior
    # RRF score: 1/(k+rank). Top match em ambas buscas ≈ 0.033
    # Threshold 0.025 = match bom em ambas (top 3)
    similarity_threshold: float = 0.025
    ttl_seconds: int = 86400  # 24 horas
    max_cache_size: int = 10000  # Máximo de queries cacheadas

    # Embedding
    embedding_dim: int = 1024  # BGE-M3 dense dimension

    # Híbrido
    dense_weight: float = 0.6  # Peso da busca densa
    sparse_weight: float = 0.4  # Peso da busca esparsa


class SemanticCache:
    """
    Cache semântico para respostas RAG com busca híbrida.

    Encontra queries similares usando embeddings dense + sparse
    e retorna respostas cacheadas quando a similaridade é alta.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._embedder = None
        self._redis: Optional[redis.Redis] = None
        self._collection: Optional[Collection] = None
        self._connected = False

    # =========================================================================
    # CONEXÕES
    # =========================================================================

    def _ensure_connected(self):
        """Garante conexão com Milvus e Redis."""
        if self._connected:
            return

        # Conecta ao Redis
        self._redis = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            decode_responses=True,
        )
        self._redis.ping()
        logger.info("Redis conectado para cache")

        # Conecta ao Milvus
        connections.connect(
            alias="cache",
            host=self.config.milvus_host,
            port=self.config.milvus_port,
        )

        # Cria ou carrega collection
        self._ensure_collection()
        self._connected = True

    def _ensure_collection(self):
        """Cria collection de cache com suporte a busca híbrida."""
        if utility.has_collection(self.config.collection_name, using="cache"):
            self._collection = Collection(
                self.config.collection_name,
                using="cache"
            )
            self._collection.load()
            logger.info(
                f"Collection '{self.config.collection_name}' carregada "
                f"({self._collection.num_entities} queries cacheadas)"
            )
            return

        # Cria nova collection com suporte a híbrido
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="query_hash",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="query_text",
                dtype=DataType.VARCHAR,
                max_length=2000,
            ),
            # Embedding denso (semântico)
            FieldSchema(
                name="dense_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.embedding_dim,
            ),
            # Embedding esparso (keywords/BM25-like)
            FieldSchema(
                name="sparse_embedding",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
            ),
            # Feedback do usuário (RLHF simplificado)
            FieldSchema(
                name="likes",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="dislikes",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="times_served",
                dtype=DataType.INT64,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Cache semântico de queries RAG com busca híbrida"
        )

        self._collection = Collection(
            name=self.config.collection_name,
            schema=schema,
            using="cache",
        )

        # Cria índice para dense (HNSW)
        dense_index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        }
        self._collection.create_index(
            field_name="dense_embedding",
            index_params=dense_index_params,
        )
        logger.info("Índice HNSW criado para dense_embedding")

        # Cria índice para sparse (SPARSE_INVERTED_INDEX)
        sparse_index_params = {
            "metric_type": "IP",  # Inner Product para sparse
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2},
        }
        self._collection.create_index(
            field_name="sparse_embedding",
            index_params=sparse_index_params,
        )
        logger.info("Índice SPARSE_INVERTED_INDEX criado para sparse_embedding")

        self._collection.load()
        logger.info(f"Collection '{self.config.collection_name}' criada com busca híbrida")

    def _ensure_embedder(self):
        """Carrega embedder (singleton)."""
        if self._embedder is None:
            from model_pool import get_raw_embedder
            self._embedder = get_raw_embedder()

    # =========================================================================
    # OPERAÇÕES DE CACHE
    # =========================================================================

    def _normalize_query(self, query: str) -> str:
        """Normaliza query para comparação."""
        return query.strip().lower()

    def _hash_query(self, query: str) -> str:
        """Gera hash da query normalizada."""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _get_embeddings(self, query: str) -> tuple[list[float], dict]:
        """
        Gera embeddings dense e sparse da query.

        Returns:
            Tuple (dense_embedding, sparse_embedding)
        """
        self._ensure_embedder()
        result = self._embedder.encode(
            [query],
            return_dense=True,
            return_sparse=True,
        )

        dense = result['dense_vecs'][0].tolist()

        # Converte sparse para formato Milvus {índice: valor}
        sparse_raw = result['lexical_weights'][0]
        # sparse_raw é um dict {token_id: weight}
        sparse = {int(k): float(v) for k, v in sparse_raw.items()}

        return dense, sparse

    def get(self, query: str) -> Optional[dict]:
        """
        Busca resposta no cache usando busca híbrida.

        Args:
            query: Pergunta do usuário

        Returns:
            Resposta cacheada ou None se não encontrada
        """
        self._ensure_connected()
        start_time = time.perf_counter()

        try:
            # Gera embeddings
            dense_embedding, sparse_embedding = self._get_embeddings(query)

            # Prepara busca híbrida
            # Request para busca densa
            dense_search = AnnSearchRequest(
                data=[dense_embedding],
                anns_field="dense_embedding",
                param={
                    "metric_type": "COSINE",
                    "params": {"ef": 64},
                },
                limit=5,
            )

            # Request para busca esparsa
            sparse_search = AnnSearchRequest(
                data=[sparse_embedding],
                anns_field="sparse_embedding",
                param={
                    "metric_type": "IP",
                    "params": {},
                },
                limit=5,
            )

            # Executa busca híbrida com RRF
            results = self._collection.hybrid_search(
                reqs=[dense_search, sparse_search],
                ranker=RRFRanker(k=60),  # k=60 é valor padrão do RRF
                limit=1,
                output_fields=["query_hash", "query_text"],
            )

            if not results or not results[0]:
                logger.debug(f"Cache MISS (sem resultados): {query[:50]}...")
                self.record_miss()
                return None

            hit = results[0][0]
            rrf_score = hit.score  # RRF score (não é similaridade direta)

            # RRF score típico varia de 0 a ~0.03 para matches bons
            # Convertemos para uma escala mais intuitiva
            # Threshold baseado em experimentação
            if rrf_score < self.config.similarity_threshold:
                logger.debug(
                    f"Cache MISS (RRF score {rrf_score:.4f} < {self.config.similarity_threshold}): "
                    f"{query[:50]}..."
                )
                self.record_miss()
                return None

            # Cache HIT - busca resposta no Redis
            query_hash = hit.entity.get("query_hash")

            # Incrementa contador de vezes servida
            self.increment_served(query_hash)
            cached_query = hit.entity.get("query_text")

            response_json = self._redis.get(f"response:{query_hash}")

            if not response_json:
                logger.warning(f"Cache inconsistente: Milvus tem, Redis não")
                return None

            response = json.loads(response_json)

            # Adiciona metadados de cache
            elapsed = (time.perf_counter() - start_time) * 1000
            response["_cache"] = {
                "hit": True,
                "rrf_score": round(rrf_score, 4),
                "matched_query": cached_query,
                "latency_ms": round(elapsed, 2),
                "search_type": "hybrid",
            }

            logger.info(
                f"Cache HIT (RRF={rrf_score:.4f}): "
                f"'{query[:30]}...' → '{cached_query[:30]}...'"
            )

            return response

        except Exception as e:
            logger.error(f"Erro ao buscar cache: {e}")
            return None

    def set(
        self,
        query: str,
        response: dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Salva resposta no cache com embeddings híbridos.

        Args:
            query: Pergunta original
            response: Resposta do RAG (dict)
            ttl: Tempo de vida em segundos (None = config default)

        Returns:
            True se salvou com sucesso
        """
        self._ensure_connected()
        ttl = ttl or self.config.ttl_seconds

        try:
            query_hash = self._hash_query(query)
            dense_embedding, sparse_embedding = self._get_embeddings(query)

            # Verifica se já existe (evita duplicatas)
            existing = self._collection.query(
                expr=f'query_hash == "{query_hash}"',
                output_fields=["id"],
                limit=1,
            )

            if existing:
                logger.debug(f"Query já cacheada: {query[:50]}...")
                # Atualiza TTL no Redis
                self._redis.expire(f"response:{query_hash}", ttl)
                return True

            # Salva embeddings no Milvus
            self._collection.insert([
                {
                    "query_hash": query_hash,
                    "query_text": query[:2000],  # Limita tamanho
                    "dense_embedding": dense_embedding,
                    "sparse_embedding": sparse_embedding,
                    "created_at": int(datetime.now().timestamp()),
                    "likes": 0,
                    "dislikes": 0,
                    "times_served": 0,
                }
            ])

            # Salva resposta no Redis
            response_copy = dict(response)
            # Remove campos que não devem ser cacheados
            response_copy.pop("_cache", None)

            self._redis.setex(
                f"response:{query_hash}",
                ttl,
                json.dumps(response_copy, ensure_ascii=False, default=str),
            )

            logger.info(f"Cache SET (híbrido): {query[:50]}... (TTL={ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
            return False

    def invalidate(self, query: str) -> bool:
        """Remove query específica do cache."""
        self._ensure_connected()

        try:
            query_hash = self._hash_query(query)

            # Remove do Redis
            self._redis.delete(f"response:{query_hash}")

            # Remove do Milvus
            self._collection.delete(expr=f'query_hash == "{query_hash}"')

            logger.info(f"Cache invalidado: {query[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Erro ao invalidar cache: {e}")
            return False

    def clear(self) -> bool:
        """Limpa todo o cache."""
        self._ensure_connected()

        try:
            # Limpa Redis (keys do DB de cache)
            self._redis.flushdb()

            # Recria collection Milvus
            utility.drop_collection(self.config.collection_name, using="cache")
            self._ensure_collection()

            logger.info("Cache limpo completamente")
            return True

        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")
            return False

    def stats(self) -> dict:
        """Retorna estatísticas do cache."""
        self._ensure_connected()

        try:
            # Milvus
            num_queries = self._collection.num_entities

            # Redis
            redis_keys = self._redis.dbsize()
            redis_memory = self._redis.info("memory").get("used_memory_human", "N/A")

            # Estatísticas de hits/misses
            hits = int(self._redis.get("stats:cache_hits") or 0)
            misses = int(self._redis.get("stats:cache_misses") or 0)
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0

            return {
                "queries_cached": num_queries,
                "redis_keys": redis_keys,
                "redis_memory": redis_memory,
                "similarity_threshold": self.config.similarity_threshold,
                "ttl_seconds": self.config.ttl_seconds,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate_pct": round(hit_rate, 2),
                "search_type": "hybrid (dense + sparse)",
            }

        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # FEEDBACK (RLHF SIMPLIFICADO)
    # =========================================================================

    def record_feedback(self, query_hash: str, is_like: bool) -> bool:
        """
        Registra feedback do usuário (like ou dislike).

        Args:
            query_hash: Hash da query
            is_like: True para like, False para dislike

        Returns:
            True se registrou com sucesso
        """
        self._ensure_connected()

        try:
            field = "likes" if is_like else "dislikes"

            # Busca valor atual
            results = self._collection.query(
                expr=f'query_hash == "{query_hash}"',
                output_fields=["id", field],
                limit=1,
            )

            if not results:
                logger.warning(f"Query não encontrada para feedback: {query_hash}")
                return False

            # Incrementa (Milvus não suporta update direto, então usamos Redis)
            redis_key = f"feedback:{query_hash}:{field}"
            self._redis.incr(redis_key)

            logger.info(f"Feedback registrado: {query_hash} +1 {field}")
            return True

        except Exception as e:
            logger.error(f"Erro ao registrar feedback: {e}")
            return False

    def increment_served(self, query_hash: str) -> bool:
        """Incrementa contador de vezes que a resposta foi servida do cache."""
        self._ensure_connected()

        try:
            self._redis.incr(f"served:{query_hash}")
            self._redis.incr("stats:cache_hits")
            return True
        except Exception as e:
            logger.error(f"Erro ao incrementar served: {e}")
            return False

    def record_miss(self) -> bool:
        """Registra um cache miss."""
        self._ensure_connected()
        try:
            self._redis.incr("stats:cache_misses")
            return True
        except:
            return False

    def get_detailed_stats(self) -> dict:
        """
        Retorna estatísticas detalhadas para o dashboard.

        Inclui:
        - Crescimento do cache
        - Distribuição de likes/dislikes
        - Queries mais servidas
        - Taxa de acerto
        """
        self._ensure_connected()

        try:
            basic_stats = self.stats()

            # Busca todas as queries com feedback
            all_queries = self._collection.query(
                expr="id > 0",
                output_fields=[
                    "query_hash", "query_text", "created_at",
                    "likes", "dislikes", "times_served"
                ],
                limit=10000,
            )

            # Calcula totais de feedback do Redis
            total_likes = 0
            total_dislikes = 0
            total_served = 0

            queries_with_feedback = []
            for q in all_queries:
                qhash = q.get("query_hash", "")
                likes = int(self._redis.get(f"feedback:{qhash}:likes") or 0)
                dislikes = int(self._redis.get(f"feedback:{qhash}:dislikes") or 0)
                served = int(self._redis.get(f"served:{qhash}") or 0)

                total_likes += likes
                total_dislikes += dislikes
                total_served += served

                if likes > 0 or dislikes > 0 or served > 0:
                    queries_with_feedback.append({
                        "query_hash": qhash,
                        "query_text": q.get("query_text", "")[:100],
                        "likes": likes,
                        "dislikes": dislikes,
                        "times_served": served,
                        "score": likes - dislikes,
                        "created_at": q.get("created_at", 0),
                    })

            # Ordena por score (likes - dislikes)
            queries_with_feedback.sort(key=lambda x: x["score"], reverse=True)

            return {
                **basic_stats,
                "total_likes": total_likes,
                "total_dislikes": total_dislikes,
                "total_times_served": total_served,
                "queries_with_feedback": len(queries_with_feedback),
                "top_queries": queries_with_feedback[:10],
                "worst_queries": queries_with_feedback[-10:][::-1] if len(queries_with_feedback) > 10 else [],
                "approval_rate": round(
                    total_likes / (total_likes + total_dislikes) * 100, 2
                ) if (total_likes + total_dislikes) > 0 else 0,
            }

        except Exception as e:
            logger.error(f"Erro ao obter estatísticas detalhadas: {e}")
            return {"error": str(e)}

    def get_growth_data(self, days: int = 30) -> list:
        """
        Retorna dados de crescimento do cache para gráfico.

        Args:
            days: Número de dias para buscar

        Returns:
            Lista de {date, count} para cada dia
        """
        self._ensure_connected()

        try:
            from datetime import datetime, timedelta

            # Busca todas as queries com timestamp (limite Milvus = 16384)
            all_queries = self._collection.query(
                expr="id > 0",
                output_fields=["created_at"],
                limit=16000,
            )

            # Agrupa por dia
            counts_by_day = {}
            for q in all_queries:
                ts = q.get("created_at", 0)
                if ts > 0:
                    date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    counts_by_day[date] = counts_by_day.get(date, 0) + 1

            # Preenche dias faltantes
            result = []
            today = datetime.now()
            for i in range(days, -1, -1):
                date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                result.append({
                    "date": date,
                    "queries_added": counts_by_day.get(date, 0),
                })

            # Adiciona total acumulado
            cumulative = 0
            for r in result:
                cumulative += r["queries_added"]
                r["total_cached"] = cumulative

            return result

        except Exception as e:
            logger.error(f"Erro ao obter dados de crescimento: {e}")
            return []

    def close(self):
        """Fecha conexões."""
        if self._redis:
            self._redis.close()
        if self._connected:
            connections.disconnect("cache")
        self._connected = False

    def __enter__(self):
        self._ensure_connected()
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# DECORATOR PARA CACHE AUTOMÁTICO
# =============================================================================

def cached_response(cache: SemanticCache):
    """
    Decorator para adicionar cache semântico a funções de geração.

    Uso:
        cache = SemanticCache()

        @cached_response(cache)
        def generate(query: str) -> dict:
            # ... pipeline RAG ...
            return response
    """
    def decorator(func):
        def wrapper(query: str, *args, **kwargs):
            # Tenta cache
            cached = cache.get(query)
            if cached:
                return cached

            # Executa função
            response = func(query, *args, **kwargs)

            # Salva no cache
            if isinstance(response, dict):
                cache.set(query, response)

            return response

        return wrapper
    return decorator


# =============================================================================
# TESTE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Teste do Cache Semântico com Busca Híbrida")
    print("=" * 60)

    cache = SemanticCache()

    # Teste 1: Set
    print("\n--- Teste SET ---")
    response = {
        "answer": "O ETP pode ser dispensado nas hipóteses do art. 75...",
        "confidence": 0.95,
        "citations": ["Art. 75, I", "Art. 75, II"],
    }
    cache.set("Quando o ETP pode ser dispensado?", response)

    # Teste 2: Get exato
    print("\n--- Teste GET (query exata) ---")
    result = cache.get("Quando o ETP pode ser dispensado?")
    print(f"Hit: {result is not None}")
    if result:
        print(f"RRF Score: {result.get('_cache', {}).get('rrf_score')}")

    # Teste 3: Get similar (semântico)
    print("\n--- Teste GET (query semanticamente similar) ---")
    result = cache.get("Quando posso dispensar o ETP?")
    print(f"Hit: {result is not None}")
    if result:
        print(f"RRF Score: {result.get('_cache', {}).get('rrf_score')}")
        print(f"Matched: {result.get('_cache', {}).get('matched_query')}")

    # Teste 4: Get diferente (não deve dar match)
    print("\n--- Teste GET (query diferente) ---")
    result = cache.get("Qual o prazo para recurso em licitação?")
    print(f"Hit: {result is not None}")

    # Teste 5: Query com artigo diferente (sparse deve ajudar)
    print("\n--- Teste GET (artigo diferente - sparse deve distinguir) ---")
    response2 = {
        "answer": "O art. 72 trata de...",
        "confidence": 0.90,
        "citations": ["Art. 72"],
    }
    cache.set("O que diz o art. 72?", response2)

    result = cache.get("O que diz o art. 75?")
    print(f"Hit para art. 75: {result is not None}")
    if result:
        print(f"Matched: {result.get('_cache', {}).get('matched_query')}")
    else:
        print("Correto! Não deu match (artigos diferentes)")

    # Stats
    print("\n--- Estatísticas ---")
    print(cache.stats())

    cache.close()
