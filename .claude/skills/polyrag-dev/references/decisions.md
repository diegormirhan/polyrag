# PolyRAG — Engineering Decisions & Rationale

Every decision below is deliberate. Do not reverse them without strong reason and user agreement.

## 1. No PyTorch / vLLM / Ollama / Docker — Vulkan on AMD

- **Why:** maximal hardware compatibility (AMD, Intel iGPU, even some ARM) and native Windows without
  containers. CUDA is NVIDIA-only; Docker/WSL adds setup burden.
- **Consequence:** all inference via **`llama.cpp` Vulkan build** (`llama-server`). Verified on
  AMD Radeon RX 9060 XT 16GB (RDNA 4).

## 2. `hipporag` library is BANNED — reimplement HippoRAG 2

- **Why:** the PyPI package `hipporag==2.0.0a4` requires `torch==2.5.1`, `transformers==4.45.2`,
  `vllm==0.6.6.post1` (Linux/CUDA-only) plus hard pins (openai==1.91.1, pydantic==2.10.4) that break
  the whole environment. It cannot install on this stack.
- **Decision:** implement the **HippoRAG 2 algorithm** (OSU-NLP, paper arXiv 2502.14802) ourselves:
  NER/OpenIE triples via Qwen3.5-4B → `networkx.DiGraph` → `nx.personalized_pagerank`. This is a
  portfolio **plus**, not a workaround.

## 3. `semantic-router` library is BANNED — custom NumPy/FAISS engine

- **Why:** the library's default encoders are OpenAI/Cohere; wiring a custom local encoder is awkward,
  and it pins Python `<3.14`. More importantly, the project's value is *showing the math* — cosine,
  margin, thresholds implemented and unit-tested explicitly.
- **Decision:** custom `pipeline/router.py` engine; `semantic-router` used only as conceptual design
  reference (Routes, utterances, threshold, margin).

## 4. OpenTelemetry without a Collector — custom exporter + WebSocket

- **Why:** no external servers (no Grafana/Prometheus/Collector). OTLP itself needs a Collector to
  receive; the browser can't consume gRPC-OTLP natively.
- **Decision:** OTel SDK (official) with a **custom `SpanExporter`** that serializes spans to JSON
  (OTLP-inspired: trace_id, span_id, parent_span_id, name, start/end, duration_ms, attributes) →
  in-memory ring buffer + **WebSocket broadcast** to the frontend. A real OTLP exporter may be wired
  later behind a config flag.

## 5. GLM-OCR instead of a general VLM

- **Why:** the data_drop pipeline needs **OCR** (text + tables out of images), and GLM-OCR is
  specialized for it. It is tiny (~830M params → Q8_0 file is 0.89GB, mmproj Q8_0 0.45GB, ~2.3GB VRAM
  total), so all 3 models stay resident in 16GB.
- **Note:** OCR models have specific prompt structures — consult llama.cpp `docs/multimodal.md`
  (GLM-OCR support) before changing the OCR prompt.

## 6. Determinism where it matters, AI where it makes a difference

- **Why:** non-deterministic agents/function-calling make systems unpredictable, unauditable, and
  hard to test — a top enterprise pain point with off-the-shelf RAG.
- **Decision:** heuristic detection, cosine routing, cache hit/miss, and PageRank are **pure math**
  (same input → same output). The LLM is called only for (a) gray-zone route disambiguation
  (temperature 0) and (b) final answer generation (temperature 0.2). This is the project's
  differentiator and its main pitch.

## 7. SQLite (stdlib) not PostgreSQL

- **Why:** zero-install, single file, perfect for a portable starter kit. PostgreSQL via Docker
  violates the no-Docker constraint. The built-in `sqlite3` module is modern enough; `pysqlite3` is
  unnecessary.

## 8. Qdrant standalone (no Docker)

- **Why:** Qdrant publishes a **native Windows binary** (`qdrant-x86_64-pc-windows-msvc.zip`,
  v1.19.0). Runs as a separate process → real vector DB architecture, zero containers.

## 9. Python 3.12 (pinned)

- **Why:** the intersection of all constraints: `semantic-router` (<3.14) is moot now but websockets
  needs ≥3.11, faiss-cpu/hipporag-era libs need ≥3.10, and 3.14 lacks wheels for some native deps.
  Use **uv** as the package manager (lockfile for reproducibility).

## 10. "Federated" terminology note

The academic **FedRAG** (2024) means privacy-preserving federation *between organizations*. PolyRAG
uses "federated" to mean **multiple heterogeneous stores** (relational + vector + graph) behind one
unified gateway — that is "multi-store / hybrid RAG" in industry terms. Use that wording in
interviews/presentations to be precise.

## 11. Multi-route fan-out + RRF for composite questions

- **Why:** some questions need data from more than one base ("total de vendas em março, e isso fere
  compliance?" → SQL + graph). A single-route router would answer incompletely.
- **Decision:** if `top2 ≥ tau_multi` (0.5), query the top-N routes in parallel and merge with
  **Reciprocal Rank Fusion** (`score = Σ 1/(k + rank)`, k=60). The *decision* of how many routes is a
  threshold (deterministic, matches the project thesis); the LLM only arbitrates which routes in the
  gray zone. RRF is scale-agnostic (each RAG scores on a different scale) and pure arithmetic.

## Positioning summary (for presentations)

> "Not a replacement for LlamaIndex — a reference implementation with a design principle:
> deterministic where it matters. It runs on democratic hardware (AMD/Vulkan, no CUDA, no Docker)
> and exposes every decision (scores, margin, spans) for audit. LlamaIndex is the productivity
> choice; PolyRAG is the transparency + portability reference."
