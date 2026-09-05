# PolyRAG

[![CI](https://github.com/diegormirhan/polyrag/actions/workflows/ci.yml/badge.svg)](https://github.com/diegormirhan/polyrag/actions/workflows/ci.yml)

**A federated RAG system that routes questions with arithmetic instead of an agent, and shows you the arithmetic while it does it.**

Drop a spreadsheet, a scanned receipt and a policy document into one folder. PolyRAG reads each one,
splits it, and decides — per chunk — whether it belongs in a SQL table, a vector index, or a
knowledge graph. Ask a question and the same decision runs again to pick where to look. Every step
reports itself, so you can watch the routing happen and see the numbers behind it.

Runs entirely on one machine: AMD GPU through Vulkan, no CUDA, no Docker, no cloud.

---

## The problem it takes seriously

Agent-based RAG delegates routing to a model. That makes behaviour unpredictable — the same question
can take a different path on Tuesday, and nobody can explain why. That is what stops these systems
from being testable, debuggable or auditable.

PolyRAG's premise is the opposite: **use deterministic maths wherever it can decide, and reserve the
model for what only a model can do.**

| Step | How it decides | Deterministic? |
|---|---|---|
| Is this file tabular? | Regex + comma/digit density | ✅ |
| Which store does this belong in? | Cosine similarity + margin | ✅ |
| Is this question already answered? | Cosine ≥ threshold | ✅ |
| Which graph chunks matter? | Personalized PageRank | ✅ |
| Genuine tie between two routes | A model breaks it | ⚠️ rare, and marked as such |
| Writing the answer | A model | ⚠️ temperature 0 |

The claim is not a new algorithm. It is that **routing does not need one** — and that a system which
can explain its own decisions is worth more than one that guesses well.

---

## Architecture

```mermaid
flowchart TB
    subgraph ingest["Ingestion"]
        drop["data_drop/"] --> kind{"file type"}
        kind -->|image| ocr["GLM-OCR"]
        kind -->|pdf, docx, pptx| doc["loaders<br/>keeps heading structure"]
        kind -->|csv, xlsx| table["pandas"]
        ocr --> chunk["chunking<br/>paragraphs, then meaning"]
        doc --> chunk
    end

    subgraph router["Semantic router — one engine, both directions"]
        s1["1 · heuristic<br/>tabularity score"] --> s2["2 · cosine + margin<br/>against config anchors"]
        s2 -->|gray zone only| s3["3 · model tiebreak"]
    end

    chunk --> router
    table --> sql
    question(["question"]) --> cache["CAG<br/>FAISS in RAM"]
    cache -->|miss| router

    router --> sql[("RAG 1 · SQLite<br/>Text-to-SQL")]
    router --> vec[("RAG 2 · Qdrant<br/>HNSW")]
    router --> graph[("RAG 3 · networkx<br/>Personalized PageRank")]

    sql --> llm["Qwen3.5-4B<br/>writes the answer"]
    vec --> llm
    graph --> llm

    llm --> ui["SvelteKit<br/>chat + live pipeline panel"]
    router -.spans.-> otel["OpenTelemetry"]
    otel -.WebSocket.-> ui
```

Four models served by `llama.cpp` over Vulkan, each on its own port, all speaking the OpenAI API:
**Qwen3.5-4B** (reasoning, SQL, entity extraction, answers, route tiebreaks), **GLM-OCR** (images),
**BGE-M3** (embeddings). Qdrant runs standalone alongside them.

---

## The router

The same component decides where a chunk is **stored** and where a question is **searched** — the
input is the only difference. It runs three stages, cheapest first:

**1 · Heuristic.** Comma and digit density. A spreadsheet is tabular by definition and never reaches
the model. Cost: microseconds.

**2 · Cosine + margin.** Each route is defined in `config.yaml` by a handful of example phrases,
embedded once at startup. The input is embedded and compared against all of them; the best match per
route becomes that route's score.

```
margin = score_top1 − score_top2

score_top1 ≥ tau_high  AND  margin ≥ delta   →  take it, no model involved
score_top1 < tau_low                          →  fall back to free text
otherwise                                     →  genuine tie, ask a model
```

This is a 1-nearest-neighbour classifier with a rejection rule. Cost: **8–13 ms**, measured.

**3 · Model tiebreak.** Only in the gray zone. It is asked twice with the two routes swapped, and
its answer is only trusted if both orderings agree — small models favour whichever option is listed
first. The UI marks these answers, because that is the one case where a model, not arithmetic, chose.

---

## Three stores, because there are three kinds of data

| | Holds | Retrieved by | Why not the others |
|---|---|---|---|
| **RAG 1** · SQLite | Rows and columns | Generated SQL | Aggregation must be exact. Correct SQL gives the correct number; no model does the arithmetic. |
| **RAG 2** · Qdrant | Free text | Cosine over HNSW | No schema to impose, no relations to model — only meaning. |
| **RAG 3** · networkx | Entities and relations | Personalized PageRank | Multi-hop. "If A fails, what breaks?" needs topology, not similarity. |

RAG 3 is HippoRAG 2 reimplemented from the [paper](https://arxiv.org/pdf/2502.14802): the model
extracts subject–relation–object triples, networkx holds the graph, and the walk teleports back to
the question's entities so the ranking means "relevant to *this* question".

The official `hipporag` package needs `torch` and `vLLM`, which are Linux/CUDA only. Reimplementing
it was a constraint, and it turned out to be the more useful outcome — the maths is visible and
testable rather than hidden behind an import.

---

## What is measured

Numbers from this machine (RX 9060 XT, 16 GB), not estimates:

| | |
|---|---|
| Routing decision (deterministic stage) | **8–13 ms** |
| Routing decision on the wire, streaming | **23 ms** |
| Time to first token | **639 ms** |
| Cache hit, including a paraphrase | **8 ms** |
| Share of a request spent on model calls | **~79 %** |

Two changes worth their own line, because both were found by instrumenting rather than guessing:

- **OpenIE 42.7 s → 7.4 s.** The GPU was 3.6 GB over capacity and streaming weights from system RAM.
  Sleeping the two intermittently-used models and halving the KV cache fixed it.
- **Answer accuracy 4/6 → 6/6** on a repeated multi-hop question, by dropping generation temperature
  from 0.2 to 0. Sampling was silently corrupting a third of the answers. *(n = 6, one question —
  enough to justify a free change, not enough to be a benchmark.)*

**Retrieval quality is not measured yet.** `recall@k`, `precision@k` and answer correctness against a
golden set are the next milestone, and this README will carry the curve when it exists. Reporting
numbers before then would be decoration.

---

## Running it

Needs Python 3.12, Node 22, and roughly 6 GB of VRAM.

```bash
# 1 · dependencies
uv sync
npm --prefix frontend ci

# 2 · runtimes (llama.cpp Vulkan build + Qdrant), unpacked into bin/
#     and GGUF models into models/ — see config.yaml for the expected paths

# 3 · everything the backend needs: four llama-servers and Qdrant
uv run python scripts/start_servers.py

# 4 · the API
uv run uvicorn --app-dir backend app.main:app --port 8000

# 5 · the interface
npm --prefix frontend run dev
```

Then open <http://localhost:5173>, or <http://localhost:8000/docs> for the API.

Drop a file into `data_drop/` and it is ingested within seconds — the hot folder is watched. Or use
the Corpus tab, which is the same code path.

Everything is driven by `config.yaml`: ports, model paths, thresholds, prompts, and the example
phrases that define each route. **Adding a route is a config change, not a code change.**

---

## Verifying it

```bash
uv run pytest tests/ -q          # 24 tests, no servers required
uv run ruff check backend/ scripts/ tests/
npm --prefix frontend run check  # 179 files, 0 errors
```

The unit tests cover cosine and margin, the read-only SQL guard, triple parsing and the cache. They
need no GPU and no network — which is the point of keeping that code pure. The scripts under
`tests/` named `*_integration.py` are run by hand against a live stack and print their results;
pytest collects nothing from them.

---

## Known limitations

Stated because they are real, not because they are theoretical:

- **Retrieval ranking is weak on dense documents.** On one questionnaire, three of five retrieved
  passages were irrelevant and all five scored between 0.418 and 0.465 — no score threshold
  separates that. Reranking is the likely answer; measurement comes first.
- **Graph search re-embeds every entity on every query.** 318 ms of a 514 ms graph lookup, growing
  with the corpus. Entity vectors belong in the ingestion step.
- **Source files (`.py`, `.ts`, …) are not supported.** Doing it properly needs AST-aware chunking
  and probably a fourth route; adding the extension to the config would route code by accident
  rather than by decision.
- **Images and tables embedded inside PDFs and DOCX are skipped.** Only their text is extracted.
- **A filename can only be ingested once.** Re-ingesting a changed file requires renaming it.
- **`llama.cpp` is not bit-for-bit deterministic even at temperature 0.** The maths in this project
  is; the model steps have residual variance, and saying otherwise would be the imprecision this
  project criticises.

---

## Stack

`FastAPI 0.141` · `Qdrant 1.19` · `networkx 3.6` · `faiss-cpu 1.15` · `OpenTelemetry 1.44` ·
`SvelteKit 2.63` · `Svelte 5` · `llama.cpp` (Vulkan) · `uv`

No PyTorch, no vLLM, no Ollama, no Docker, no WSL.
