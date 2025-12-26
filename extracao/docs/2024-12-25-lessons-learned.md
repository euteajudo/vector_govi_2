# Lições Aprendidas - 25/12/2024

## Resumo da Sessão

Nesta sessão completamos a ingestão da Lei 14.133/2021 (1277 chunks) e iniciamos o processo de enriquecimento. Enfrentamos diversos desafios técnicos que resultaram em aprendizados importantes.

---

## 1. Arquitetura Distribuída

### Descoberta: Separação de Responsabilidades

A arquitetura final que funciona:

```
┌─────────────────────────────────────────────────────────┐
│                    WINDOWS (Local)                       │
│  - Orquestração do pipeline                              │
│  - Scripts de disparo                                    │
│  - Monitoramento                                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      POD (RunPod)                        │
│  - GPU Server (vLLM + BGE-M3)                           │
│  - Celery Workers (10x)                                  │
│  - Redis (broker)                                        │
│  - Processamento local = baixa latência                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      VPS (Contabo)                       │
│  - Milvus Standalone                                     │
│  - Armazenamento persistente                             │
│  - IP fixo: 77.37.43.160                                │
└─────────────────────────────────────────────────────────┘
```

### Lição
- Processar no POD (onde está a GPU) e enviar apenas resultados para VPS
- Evita latência de rede para LLM e embeddings
- Celery no POD permite paralelismo (10 workers = 10x mais rápido)

---

## 2. Problemas com GPUClientLLMAdapter

### Problema 1: Retorno incorreto de `chat_with_schema()`

**Sintoma**: LLM retornava 500 para artigos COM filhos (150 falhas), mas funcionava para artigos SEM filhos (54 OK).

**Causa raiz**: `GPUClientLLMAdapter.chat_with_schema()` retornava o wrapper `{"content": "...", "usage": {...}}` em vez do JSON parseado.

**Correção** (`run_pipeline_http.py`):
```python
def chat_with_schema(self, messages, schema, ...):
    result = self.chat(messages=messages, ...)
    content = result.get("content", "{}")
    return json.loads(content)  # Retorna JSON parseado
```

### Problema 2: Nome do modelo incorreto

**Sintoma**: Erro 500 do vLLM "model does not exist".

**Causa**: O modelo estava registrado como `qwen3-8b` mas o código usava `Qwen/Qwen3-8B-AWQ`.

**Correção**: Alterar para `model="qwen3-8b"` no adapter.

### Lição
- Sempre verificar se o wrapper está extraindo o conteúdo corretamente
- Testar manualmente com `curl` para validar nomes de modelos

---

## 3. SSH do RunPod

### Problema
O SSH do RunPod tem formato especial e não executa comandos normalmente:
```
ssh user@ssh.runpod.io -i key "comando"
# Erro: "Your SSH client doesn't support PTY"
```

### Descoberta
- O RunPod SSH não é SSH tradicional
- Mesmo com `-T` (sem PTY), comandos não executam
- Alternativas: Web Terminal (precisa habilitar) ou Jupyter

### Lição
- Para executar comandos no POD, usar Web Terminal ou Jupyter
- O GPU Server HTTP funciona sem problemas (porta 55278)

---

## 4. Enriquecimento de Chunks

### Performance Medida

| Método | Tempo/chunk | 1277 chunks |
|--------|-------------|-------------|
| Sequencial HTTP | 4.5s | 1.6 horas |
| Celery 10 workers | 0.5s (paralelo) | ~10 minutos |

### Campos Gerados
- `context_header`: Frase contextualizando o dispositivo
- `thesis_text`: Resumo do que o dispositivo determina
- `thesis_type`: Classificação (definição, requisito, vedação, etc.)
- `synthetic_questions`: Perguntas que o dispositivo responde
- `enriched_text`: Texto completo para embedding

### Problema: LLM gerando `<think>` tags
O modelo Qwen 3 às vezes gera tags `<think>` antes do JSON. O fallback no parser lida com isso.

---

## 5. Validação Pós-Ingestão

### Módulo Criado
`src/validation/post_ingestion_validator.py`

### Erros Detectáveis
- `context_header` vazio (não enriquecido)
- `thesis_text` vazio
- `thesis_type` inválido
- Resíduos de markdown no texto
- Chunks duplicados

### Resultado Lei 14.133
- 1277 chunks ingeridos
- 154 enriquecidos (12%)
- 1123 pendentes de enriquecimento

---

## 6. Arquivos Importantes Criados

### Scripts
- `scripts/pod/monitor_enrichment.py` - Monitor de progresso em tempo real
- `scripts/pod/dispatch_celery.py` - Dispara tasks para Celery
- `scripts/fix_art001.py` - Correção de artigos específicos

### Deploy
- `deploy/pod/start_workers.sh` - Inicia Celery workers no POD

### Código
- `src/pod/tasks_local.py` - Tasks Celery para rodar localmente no POD
- `src/validation/post_ingestion_validator.py` - Validador pós-ingestão

---

## 7. Comandos Úteis

### Verificar progresso do enriquecimento
```bash
python scripts/pod/monitor_enrichment.py --watch
```

### Testar GPU Server
```bash
curl http://195.26.233.70:55278/health
```

### Verificar chunks no Milvus
```python
from pymilvus import connections, Collection
connections.connect(host="77.37.43.160", port=19530)
collection = Collection("leis_v3")
total = collection.query(expr='document_id == "LEI-14133-2021"', limit=10000)
print(f"Total: {len(total)}")
```

---

## 8. Próximos Passos

1. **Iniciar Celery no POD** (via Web Terminal)
   - Redis + 10 workers
   - Flower para monitoramento

2. **Completar enriquecimento** (~10 min com Celery)

3. **Validar qualidade** com `post_ingestion_validator.py`

4. **Testar RAG completo** com perguntas sobre a Lei 14.133

---

## Métricas Finais

| Métrica | Valor |
|---------|-------|
| Documento | Lei 14.133/2021 |
| Chunks totais | 1277 |
| Artigos válidos | 192/204 (94%) |
| Parágrafos | 434 |
| Incisos | 639 |
| Tempo de ingestão | ~2.5 min |
| Tempo de enriquecimento (estimado) | ~10 min (Celery) |
