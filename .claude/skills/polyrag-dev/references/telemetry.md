# PolyRAG — Telemetry & Evaluation Reference

## OpenTelemetry setup

- **TracerProvider** with a **custom `SpanExporter`** (`core/telemetry.py`). No Collector, no
  Grafana/Prometheus.
- The exporter serializes each span to JSON (OTLP-inspired) into:
  - an **in-memory ring buffer** (`collections.deque(maxlen=telemetry.ring_buffer_size)`), served by
    `GET /api/v1/telemetry/traces`;
  - a **WebSocket broadcast** (`WS /api/v1/telemetry/stream`) → live frontend panel.
- `FastAPIInstrumentor.instrument_app(app)` for automatic HTTP spans.

## Standard spans

| Span name | When | Key attributes |
|---|---|---|
| `pipeline.ingest` | a file is processed in data_drop | `ingest.file`, `ingest.file_type`, `ingest.ocr_used` |
| `pipeline.ocr` | image → text via GLM-OCR | `ocr.model`, `ocr.chars`, `ocr.latency_ms` |
| `pipeline.router` | routing a chunk or a question | `router.route`, `router.score_top1`, `router.score_top2`, `router.margin`, `router.decision_stage` (`heuristic` \| `embedding` \| `llm_judge`), `router.llm_judge_used`, `router.routes` (list, when fan-out), `router.fan_out` (bool) |
| `pipeline.cache` | CAG lookup | `cache.hit` (bool), `cache.cosine`, `cache.entries` |
| `pipeline.rag.<route>` | retrieval from the chosen base | `rag.engine` (`sqlite` \| `qdrant` \| `graph`), `rag.top_k`, `rag.scores`, `rag.entities` (graph) |
| `pipeline.llm` | final generation | `llm.model`, `llm.prompt_tokens`, `llm.completion_tokens`, `llm.temperature` |
| `eval.*` | offline evaluation only (`service.name=polyrag-eval`) | see Evaluation below |

Every span carries `duration_ms` (end − start). Spans are nested under one trace per user request:
`Cache → Router → RAG → LLM`.

## Frontend panel (SvelteKit)

Right column of the UI consumes `WS /api/v1/telemetry/stream` and renders:
- timeline of spans per trace with latencies,
- badge with chosen route + margin,
- cache hit/miss indicator.

## Evaluation (offline) — `scripts/evaluate.py`

**Key concept:** real recall requires ground truth, so it is measured **offline** with a **golden
set** (50–100 questions), not in production in real time.

Golden set item: `{question, correct_route, relevant_chunks[], expected_answer}`.

### Metrics

```
retrieval recall@k    = |relevant ∩ top-k retrieved| / |relevant|
retrieval precision@k = |relevant ∩ top-k retrieved| / k
router_accuracy       = (questions routed to correct_route) / total
answer_correctness    = generated answer vs expected answer  (rubric, below)
```

### The distinction (diagnostic)

```
wrong answer?
  ├─ recall LOW  → info never entered context  → problem in RETRIEVAL
  └─ recall HIGH → info was there, LLM erred   → problem in GENERATION
```

- `recall` = "was the right information in the context?" (math, high achievable)
- `answer_correctness` = "did the right answer reach the user, even reworded?" (LLM rubric)

### LLM-as-judge rubric (temperature 0, JSON output)

```
"You are an answer evaluator. Compare the GENERATED ANSWER with the EXPECTED ANSWER.
Reply only JSON: {"correct": 0|1, "justification": "..."}
- correct = 1 if the generated answer conveys the SAME information as the expected answer,
  even if worded differently.
- correct = 0 if essential information is missing or contradicted."
```

### Realistic targets

| Metric | Target |
|---|---|
| recall@10 (synthetic/controlled) | 90–98% |
| recall@1 | 60–85% |
| router_accuracy | 90–97% |
| answer_correctness (RAG 1 / SQL) | 90%+ (deterministic: right SQL = exact number) |
| answer_correctness (RAG 2/3, free text) | 70–90% (99% unrealistic with a local 14B LLM) |

### Report format

Always report recall and correctness **separately**, with the top-k (recall@1 ≠ recall@10) and the
golden-set context. Export as spans with `service.name=polyrag-eval`:
`eval.recall@1`, `eval.recall@5`, `eval.recall@10`, `eval.precision@10`, `eval.router_accuracy`,
`eval.answer_correctness`, `eval.route`.

### Production proxies (no golden set live)

Show in the real-time panel: `retrieval.top1_score`, `retrieval.scores[]`,
`retrieval.used_in_answer` (attribution), `router.margin`, `cache.hit`, and the determinism signal
`same_question_same_route` (ask the same question N times → identical route/answer).
