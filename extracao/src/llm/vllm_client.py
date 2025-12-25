"""
Cliente vLLM com API OpenAI-compatible.

O vLLM expoe uma API compativel com OpenAI, permitindo usar
o mesmo codigo para vLLM, Ollama, ou OpenAI.

Uso:
    from llm import VLLMClient

    client = VLLMClient(base_url="http://localhost:8000/v1")

    response = client.chat([
        {"role": "system", "content": "Voce e um assistente."},
        {"role": "user", "content": "O que e ETP?"}
    ])

    print(response)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Any
import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACAO
# =============================================================================

@dataclass
class LLMConfig:
    """Configuracao do cliente LLM."""

    # Conexao
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"  # vLLM nao precisa de API key
    timeout: float = 120.0  # segundos

    # Modelo
    model: str = "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"  # Modelo padrao

    # Geracao
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0

    # Retry
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def for_enrichment(cls, model: str = "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ") -> "LLMConfig":
        """Config otimizada para enriquecimento de chunks."""
        return cls(
            model=model,
            temperature=0.0,
            max_tokens=1024,  # Enriquecimento nao precisa de muito
            timeout=300.0,  # 5 minutos para batches grandes
        )

    @classmethod
    def for_extraction(cls, model: str = "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ") -> "LLMConfig":
        """Config para extracao estruturada (JSON complexo)."""
        return cls(
            model=model,
            temperature=0.0,
            max_tokens=4096,  # Extração precisa de mais tokens
        )


# =============================================================================
# CLIENTE VLLM
# =============================================================================

class VLLMClient:
    """
    Cliente para vLLM com API OpenAI-compatible.

    Suporta:
    - Chat completions
    - Retry automatico
    - Logging de metricas
    - Compativel com Ollama tambem

    Attributes:
        config: Configuracao do cliente
        _client: Cliente HTTP
    """

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Inicializa o cliente.

        Args:
            config: Configuracao. Se nao fornecido, usa default.
            **kwargs: Sobrescreve campos da config (ex: base_url="...")
        """
        self.config = config or LLMConfig()

        # Aplica kwargs sobre a config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        )

        logger.info(f"VLLMClient inicializado: {self.config.base_url}")

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Envia mensagens e retorna resposta.

        Args:
            messages: Lista de mensagens [{"role": "...", "content": "..."}]
            temperature: Temperatura (default: config)
            max_tokens: Max tokens (default: config)
            model: Modelo (default: config)
            **kwargs: Parametros extras para a API

        Returns:
            Texto da resposta do modelo
        """
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
            **kwargs,
        }

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()

                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()

                elapsed = time.time() - start_time
                data = response.json()

                # Extrai texto da resposta
                content = data["choices"][0]["message"]["content"]

                # Log metricas
                usage = data.get("usage", {})
                logger.debug(
                    f"LLM response: {elapsed:.2f}s, "
                    f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
                    f"completion_tokens={usage.get('completion_tokens', '?')}"
                )

                return content

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

            except httpx.TimeoutException as e:
                logger.warning(f"Timeout (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

            except Exception as e:
                logger.error(f"Erro inesperado: {e}")
                raise

        raise RuntimeError("Falha apos todas as tentativas")

    def chat_json(
        self,
        messages: list[dict],
        **kwargs,
    ) -> dict:
        """
        Envia mensagens e retorna resposta parseada como JSON.

        NOTA: Este metodo faz parsing POS-RESPOSTA (pode falhar).
        Para extracao estruturada, use chat_with_schema() que forca
        o modelo a gerar JSON valido via guided_json.

        Args:
            messages: Lista de mensagens
            **kwargs: Parametros para chat()

        Returns:
            Dict parseado da resposta JSON
        """
        response = self.chat(messages, **kwargs)

        # Tenta parsear JSON
        response = response.strip()

        # Remove markdown code blocks se presentes
        if response.startswith("```"):
            import re
            response = re.sub(r"^```\w*\n?", "", response)
            response = re.sub(r"\n?```$", "", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Tenta encontrar JSON na resposta
            import re
            # Tenta objeto
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                return json.loads(match.group())
            # Tenta array
            match = re.search(r"\[[\s\S]*\]", response)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Nao foi possivel parsear JSON: {response[:200]}") from e

    def chat_with_schema(
        self,
        messages: list[dict],
        schema: type,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> dict:
        """
        Envia mensagens com response_format json_schema para forcar output estruturado.

        IMPORTANTE: Este metodo usa o recurso json_schema do vLLM que
        forca o modelo a gerar APENAS tokens validos para o schema.
        Muito mais confiavel que chat_json() para extracao estruturada.

        Uso:
            from pydantic import BaseModel

            class Article(BaseModel):
                article_number: str
                content: str

            result = client.chat_with_schema(
                messages=[{"role": "user", "content": "Extraia..."}],
                schema=Article,
            )

        Args:
            messages: Lista de mensagens
            schema: Classe Pydantic ou dict com JSON Schema
            temperature: Temperatura (default: 0 para extracao)
            max_tokens: Max tokens (default: config)
            model: Modelo (default: config)

        Returns:
            Dict validado contra o schema
        """
        # Extrai JSON Schema do Pydantic ou usa dict direto
        if hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()
            schema_name = schema.__name__
        elif isinstance(schema, dict):
            json_schema = schema
            schema_name = "extraction_schema"
        else:
            raise ValueError(f"Schema deve ser Pydantic BaseModel ou dict, recebido: {type(schema)}")

        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.0,  # 0 para extracao
            "max_tokens": max_tokens or self.config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                },
            },
        }

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()

                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()

                elapsed = time.time() - start_time
                data = response.json()

                # Extrai texto da resposta
                content = data["choices"][0]["message"]["content"]

                # Log metricas
                usage = data.get("usage", {})
                logger.debug(
                    f"LLM json_schema response: {elapsed:.2f}s, "
                    f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
                    f"completion_tokens={usage.get('completion_tokens', '?')}"
                )

                # Com json_schema, o output JA e JSON valido
                return json.loads(content)

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

            except json.JSONDecodeError as e:
                # Isso nao deveria acontecer com json_schema
                logger.error(f"JSON invalido mesmo com json_schema: {e}")
                raise

            except Exception as e:
                logger.error(f"Erro inesperado: {e}")
                raise

        raise RuntimeError("Falha apos todas as tentativas")

    def list_models(self) -> list[str]:
        """Lista modelos disponiveis no servidor."""
        try:
            response = self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Erro ao listar modelos: {e}")
            return []

    def health_check(self) -> bool:
        """Verifica se o servidor esta respondendo."""
        try:
            models = self.list_models()
            return len(models) > 0
        except Exception:
            return False

    def close(self):
        """Fecha o cliente HTTP."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"VLLMClient(url={self.config.base_url!r}, model={self.config.model!r})"


# =============================================================================
# CLIENTE OLLAMA (mesmo protocolo)
# =============================================================================

class OllamaClient(VLLMClient):
    """
    Cliente para Ollama (mesmo protocolo OpenAI-compatible).

    Uso:
        client = OllamaClient(model="qwen3:8b")
        response = client.chat([...])
    """

    def __init__(self, model: str = "qwen3:8b", **kwargs):
        config = LLMConfig(
            base_url="http://localhost:11434/v1",
            model=model,
            **{k: v for k, v in kwargs.items() if hasattr(LLMConfig, k)}
        )
        super().__init__(config=config)


# =============================================================================
# FACTORY
# =============================================================================

def get_llm_client(
    provider: str = "vllm",
    model: Optional[str] = None,
    **kwargs,
) -> VLLMClient:
    """
    Factory para criar cliente LLM.

    Args:
        provider: "vllm" ou "ollama"
        model: Nome do modelo
        **kwargs: Config adicional

    Returns:
        Cliente configurado
    """
    if provider == "ollama":
        return OllamaClient(model=model or "qwen3:8b", **kwargs)

    # vLLM default
    config = LLMConfig(model=model or "Qwen/Qwen3-8B", **kwargs)
    return VLLMClient(config=config)


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Teste do VLLMClient")
    print("=" * 60)

    # Testa conexao
    client = VLLMClient()

    print(f"\nCliente: {client}")
    print(f"Health check: {client.health_check()}")

    models = client.list_models()
    print(f"Modelos disponiveis: {models}")

    if models:
        # Teste simples
        print("\n--- Teste de Chat ---")
        response = client.chat([
            {"role": "user", "content": "O que significa ETP em licitacoes?"}
        ], max_tokens=200)
        print(f"Resposta: {response[:300]}...")

        # Teste JSON
        print("\n--- Teste de JSON ---")
        try:
            json_response = client.chat_json([
                {"role": "system", "content": "Responda apenas com JSON valido."},
                {"role": "user", "content": 'Classifique: {"tipo": "definicao ou procedimento", "confianca": 0.0-1.0} para: "ETP e o Estudo Tecnico Preliminar"'}
            ], max_tokens=100)
            print(f"JSON: {json_response}")
        except Exception as e:
            print(f"Erro: {e}")

    client.close()
