---
name: polyrag-dev
description: >-
  Use this skill whenever working on the PolyRAG project — a unified/federated multi-modal RAG
  framework with semantic routing and CAG (cache-augmented generation), running natively on
  Windows + AMD GPU via the Vulkan backend (no PyTorch, no vLLM, no Ollama, no Docker/WSL).
  Trigger it for any task on PolyRAG: implementing or modifying pipeline modules (data_drop
  ingestion, GLM-OCR image text extraction, semantic router, the 3 RAGs — relational/SQLite,
  vectorial/Qdrant, graph/networkx — CAG FAISS cache, orchestrator, FastAPI /api/v1 routes,
  OpenTelemetry telemetry, SvelteKit frontend), writing evaluation scripts (recall@k, precision@k,
  router_accuracy, answer_correctness with golden sets), choosing models or library versions
  (Qwen3-14B, GLM-OCR, BGE-M3 via llama.cpp), or explaining the math (cosine similarity, L2
  normalization, margin scoring, Personalized PageRank). Always consult this skill before writing
  or reviewing PolyRAG code so the architectural decisions, deterministic-design principle, pinned
  versions, and hard constraints are respected — even if the user does not explicitly say
  "PolyRAG" but references the project's architecture, pipeline, models, ports, or math.
---

# PolyRAG — Development Guide

PolyRAG is a reference-implementation RAG framework that federates **3 heterogeneous data stores**
(relational SQL, vector store, knowledge graph) behind a single unified pipeline with **semantic
routing** and a **semantic cache (CAG)**. Everything runs 100% locally on consumer hardware.

**Core thesis:** *deterministic where it matters, AI where it makes a difference.* Routing, cache
hit/miss, and graph ranking are pure math (100% deterministic and testable); the LLM is reserved
for disambiguation and final answer generation only.

## 1. Hard constraints (never violate)

1. **No PyTorch, no vLLM, no Ollama, no Docker, no WSL.** Native Windows + AMD GPU via **Vulkan**.
2. **No `hipporag` library** (requires `torch==2.5.1` + `vllm` — Linux/CUDA-only). GraphRAG is
   reimplemented with `networkx`.
3. **No `semantic-router` library.** The router is a custom NumPy/FAISS engine (math visible).
4. **Python 3.12** (semantic-router's `<3.14` pin and websockets `>=3.11` conflict on other versions).
5. All model inference goes through **`llama-server`** (OpenAI-compatible API). Never import a model
   directly into Python.
6. **Determinism:** routing/cache/threshold decisions are arithmetic — same input, same output, bit
   for bit. Temperature is fixed low (0.2). LLM-as-judge runs at temperature 0.
7. **Config-driven:** all parameters live in `config.yaml` (models, thresholds, router utterances,
   ports). No hardcoded paths or magic numbers in code.
8. Every pipeline stage emits an **OpenTelemetry span** (see `references/telemetry.md`).

## 2. The system at a glance

```
data_drop/  →  watcher  →  split images  →  GLM-OCR (OCR)  →  chunking
        →  SEMANTIC ROUTER (heuristic → cosine+margin → LLM-as-judge)
        →  [RAG1 SQLite] [RAG2 Qdrant] [RAG3 graph networkx]
USER QUERY → CAG cache (FAISS RAM, cosine ≥ 0.92 = hit)
        →  router (same engine) → RAG → LLM answer  →  WS stream + OTel spans
```

**Models (3 `llama-server` instances, all OpenAI-compatible):**

| Role | Model | File | Port |
|---|---|---|---|
| LLM (reasoning, SQL, NER, judge, answers) | Qwen3-14B | `Qwen3-14B-Q4_K_M.gguf` | 8080 |
| OCR (images in data_drop) | GLM-OCR (~830M params) | `GLM-OCR-Q8_0.gguf` + `mmproj-GLM-OCR-Q8_0.gguf` | 8081 |
| Embeddings (BGE-M3, 1024 dims, PT/EN) | BGE-M3 | `bge-m3-Q8_0.gguf` | 8082 |

VRAM budget: ~9.5GB (Qwen3-14B) + ~2.3GB (GLM-OCR) + ~0.6GB (BGE-M3) ≈ 12.4GB of 16GB — all three
stay resident. Qdrant standalone runs as a separate process (v1.19.0, Windows native binary).

## 3. Development workflow

When implementing or modifying PolyRAG code, follow this order:

1. **Read the relevant reference** (see table below) before writing code.
2. **Check constraints** (section 1) — do not introduce any prohibited dependency.
3. **Config first:** add any new tunable parameter to `config.yaml` and load it via `core/config.py`.
4. **Instrument:** every stage must emit a span with the standard attributes from
   `references/telemetry.md`. Never return from a stage without a span.
5. **Keep it deterministic:** the routing/cache math must be pure functions (numpy/FAISS only),
   unit-testable without any LLM.
6. **Async:** use `async def` + the OpenAI async client for all model I/O.

## 4. Reference files (read the relevant one before coding)

| When working on... | Read |
|---|---|
| Pipeline, folder layout, modules, config.yaml, API contract, model/ports | `references/architecture.md` |
| Cosine similarity, L2 norm, margin, thresholds, FAISS vs HNSW, PageRank/PPR, quantization | `references/math.md` |
| Why decisions were made (hipporag reimplementation, custom router, custom OTel exporter, GLM-OCR, determinism) | `references/decisions.md` |
| Pinned library versions + official docs links | `references/versions.md` |
| Span names, attributes, OTel exporter design, evaluation metrics (recall@k, answer_correctness rubric) | `references/telemetry.md` |
| How to interact with the user: adaptive teaching, consent-before-action protocol, quizzes | `references/interaction.md` |

## 5. Code standards

- **Directory layout:** `backend/app/` with `core/`, `api/v1/`, `pipeline/`, `rags/`, `ingest/`, `schemas/`.
- **RAG interface:** all RAGs implement `RAGBase` with `ingest()` and `query()` (see architecture).
- **Spans:** name spans `pipeline.cache`, `pipeline.router`, `pipeline.rag.<route>`, `pipeline.llm`.
- **Responses:** the API returns `{answer, route, trace_id, sources[]}`.
- **Tests:** math (router, cache) must have pure unit tests; no LLM required to run them.

## 6. Before writing ANY PolyRAG code

- [ ] No prohibited dependency (torch, vllm, ollama, docker, semantic-router lib, hipporag lib)?
- [ ] Parameters in `config.yaml`, not hardcoded?
- [ ] Stage emits a standard OTel span?
- [ ] Deterministic paths are pure functions (testable without LLM)?
- [ ] Matches the folder/module layout in architecture.md?

## 7. Interaction protocol (MANDATORY)

These two rules override everything else. Read `references/interaction.md` for the full protocol.

1. **Consent before any action — always.** NEVER modify files, create/delete anything, or run
   terminal commands without first asking the user and getting their explicit approval. Present what
   you plan to do, wait for a "go". No exceptions, even for small edits. The user's machine and
   repository belong to them; every change is their decision.

2. **Adaptive teaching — always know where the user is.** Before explaining a topic or writing code,
   assess the user's current knowledge level. Ask a quick question or a small quiz about the concept
   at hand (e.g., "do you remember what cosine similarity measures?"). Use their answer to decide how
   deep to go: skip topics they clearly master, explain thoroughly what they don't. When a milestone
   is reached (module implemented, decision made, concept covered), verify understanding with a short
   quiz before moving to the next step. Never assume prior knowledge; never assume ignorance — check
   first.
