# PolyRAG — Architecture Reference

## Folder layout

```
polyrag/
├── config.yaml                 # ALL parameters (models, thresholds, router utterances, ports)
├── pyproject.toml · uv.lock · .python-version (3.12)
├── data_drop/                  # hot folder: files dropped here get ingested
├── data/
│   ├── sql/relacional.db       # RAG 1 (SQLite)
│   ├── qdrant/                 # Qdrant storage
│   ├── graph/graph.json        # RAG 3 persisted graph
│   └── processed/              # ingested files moved here
├── models/                     # .gguf files (gitignored)
├── runtime/
│   ├── llama/llama-server.exe  # Vulkan build (b10453)
│   └── qdrant/qdrant.exe       # standalone (v1.19.0)
├── scripts/
│   ├── start_llm_servers.ps1   # 3 llama-server instances
│   ├── start_qdrant.ps1 · start_backend.ps1 · stop_all.ps1
│   └── evaluate.py             # offline benchmark (recall@k etc.)
├── backend/app/
│   ├── main.py                 # FastAPI factory, lifespan, OTel instrumentation, CORS
│   ├── core/                   # config.py · telemetry.py · llama_client.py
│   ├── api/v1/                 # health · chat · ingest · telemetry · router aggregator
│   ├── pipeline/               # orchestrator · tabularity · router · cache
│   ├── rags/                   # base.py · relational.py · vectorial.py · graph.py
│   ├── ingest/                 # watcher · loaders · ocr · chunking
│   └── schemas/                # pydantic DTOs (chat, ingest, telemetry)
├── frontend/                   # SvelteKit 2 + Tailwind v4.3 + marked (Dia 5)
└── tests/                      # test_router_math · test_cache · test_api_integration
```

## Ingest pipeline (order matters)

1. **watcher** polls `data_drop/` every `watch_interval_s` (2s default), detects new files by hash/mtime.
2. **split** images (`.png .jpg .jpeg .webp .bmp`) from text/table files.
3. **ocr** (`ingest/ocr.py`) sends images as base64 to llama-server :8081 (GLM-OCR) with
   `ingest.ocr_prompt` → returns text (tables reproduced as Markdown).
4. **loaders** (`ingest/loaders.py`): `.txt/.md` → str; `.csv/.xlsx` → pandas DataFrame.
5. **chunking** (`ingest/chunking.py`): `chunk_size` 512, `chunk_overlap` 64.
6. **router** (`pipeline/router.py`) decides the destination base for each chunk:
   - Stage 1 (heuristic): tabularity score ≥ 0.8 → `relational`.
   - Stage 2 (embeddings): cosine against each route's utterances → top1/top2/margin.
   - Stage 3 (LLM-as-judge): only in the gray zone (low margin).
7. **store** chunk into the chosen RAG (SQL, Qdrant, or graph). Move file to `data/processed/`.

## Query pipeline (orchestrator)

1. **CAG cache** (`pipeline/cache.py`, FAISS `IndexFlatIP` in RAM, L2-normalized): if cosine ≥
   `cache.similarity_threshold` (0.92) → return cached answer immediately (span `cache.hit=true`).
2. **router**: same engine as ingestion, input = user question. Decides the route(s):
   - single route if confident (top1 ≥ tau_high AND margin ≥ delta_margin);
   - **multi-route fan-out** if `top2 ≥ tau_multi` (e.g. 0.5) → query the top-N routes in parallel;
   - gray zone → LLM-as-judge returns which routes (JSON list), not just one.
3. **RAG query**: each selected base retrieves top-k. When fan-out happened, merge rankings with
   **Reciprocal Rank Fusion (RRF)**: `score = Σ_rotas 1/(k + rank)`, k=60 — deterministic, scale-agnostic.
4. **LLM** (`core/llama_client.py`, Qwen3-14B :8080, temperature 0.2): final answer over the
   fused multi-source context tagged with origins (`[SQL]`, `[GRAFO]`).
5. Store `(question_embedding, answer, route)` into CAG for future hits.

## The 3 RAGs

- **RAG 1 — relational** (`rags/relational.py`): pandas DataFrame → SQLite tables; Text-to-SQL via
  Qwen3-14B with schema context → **read-only validation** (only `SELECT`) → execute → answer as
  Markdown table + SQL shown. Deterministic correctness (right SQL = exact number).
- **RAG 2 — vectorial** (`rags/vectorial.py`): chunks → BGE-M3 embed (:8082) → Qdrant collection
  (`distance: Cosine`, `vector_size: 1024`). Query: embed → HNSW search top-k → context → LLM
  synthesis with citations.
- **RAG 3 — graph** (`rags/graph.py`): Qwen3-14B NER/OpenIE extracts triples
  `(subject, relation, object)` per chunk → `networkx.DiGraph` (nodes = entities + chunks; edges =
  relations + entity↔chunk) persisted to `graph.json` → query: match question entities via
  embeddings → **`nx.personalized_pagerank`** → rank chunks → multi-hop answer.

## API contract (FastAPI, `/api/v1/`)

| Method | Route | Description |
|---|---|---|
| GET | `/api/v1/health` | status of llm/ocr/embeddings/qdrant |
| POST | `/api/v1/chat` | `{message}` → `{answer, route, trace_id, sources[]}` |
| WS | `/api/v1/chat/stream` | token streaming + pipeline events |
| POST | `/api/v1/ingest` | direct file upload |
| GET | `/api/v1/ingest/jobs` | ingestion job status |
| WS | `/api/v1/telemetry/stream` | live OTel spans (feeds frontend panel) |
| GET | `/api/v1/telemetry/traces` | ring buffer of recent traces |
| DELETE | `/api/v1/cache` | clear CAG |

## config.yaml essentials

```yaml
server: {host: 127.0.0.1, port: 8000}
llama:
  bin_path: runtime/llama/llama-server.exe
  llm:      {model_path: models/Qwen3-14B-Q4_K_M.gguf, host: 127.0.0.1, port: 8080, ctx_size: 16384, n_gpu_layers: -1, temperature: 0.2}
  ocr:      {model_path: models/GLM-OCR-Q8_0.gguf, mmproj_path: models/mmproj-GLM-OCR-Q8_0.gguf, host: 127.0.0.1, port: 8081, ctx_size: 8192, n_gpu_layers: -1}
  embeddings: {model_path: models/bge-m3-Q8_0.gguf, host: 127.0.0.1, port: 8082, ctx_size: 8192, n_gpu_layers: -1, dimensions: 1024}
qdrant: {bin_path: runtime/qdrant/qdrant.exe, host: 127.0.0.1, port: 6333, collection: polyrag_docs, distance: Cosine, vector_size: 1024}
router:
  tau_high: 0.62
  tau_low: 0.35
  delta_margin: 0.08
  llm_judge_enabled: true
  routes:  # each route: description + utterances (the anchors that "train" the router)
    relational: {description: "...", utterances: ["total de vendas por região", ...]}
    vectorial:  {description: "...", utterances: ["resuma o artigo sobre...", ...]}
    graph:      {description: "...", utterances: ["se o fornecedor A falhar, quais contratos são afetados", ...]}
cache: {enabled: true, similarity_threshold: 0.92, max_entries: 10000}
ingest: {watch_interval_s: 2.0, chunk_size: 512, chunk_overlap: 64, ocr_prompt: "...", vision_extensions: [.png,.jpg,.jpeg,.webp,.bmp], text_extensions: [.txt,.md], table_extensions: [.csv,.xlsx]}
telemetry: {enabled: true, service_name: polyrag, ring_buffer_size: 500}
```
