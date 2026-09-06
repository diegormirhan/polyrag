# PolyRAG — Math Reference

Every concept below is used somewhere in the project. Keep the implementations pure functions
(numpy/FAISS only) so they are unit-testable without any LLM.

## Embeddings — "meaning becomes geometry"

An **embedding** is a vector (list of numbers) that represents a text's meaning. Similar texts point
in similar directions in a **1024-dimensional space** (BGE-M3). BGE-M3 is multilingual (PT-BR + EN).

- Use: router, CAG, RAG 2, RAG 3 entity matching.
- Generated via llama-server :8082 `/v1/embeddings` (never import a model into Python).

## L2 norm and normalization

```
‖v‖ = √(v₁² + v₂² + ... + vₙ²)
normalize(v) = v / ‖v‖   → unit vector (length 1)
```

After normalizing, **cosine similarity == dot product** (much cheaper). Always L2-normalize before
putting vectors into `faiss.IndexFlatIP` (inner product = cosine on unit vectors).

## Cosine similarity

```
                a · b          a₁b₁ + a₂b₂ + ... + aₙbₙ
cos(θ) = ───────────────── = ─────────────────────────────
              ‖a‖ × ‖b‖        √(Σaᵢ²) × √(Σbᵢ²)
```

- `cos = 1` → same direction (identical meaning); `cos ≈ 0` → unrelated.
- **Why cosine, not Euclidean distance:** meaning is in *direction*, not magnitude. Cosine ignores
  vector length (a long document is a "longer" vector but the topic is the direction).

## Margin scoring (the router's confidence)

The router embeds the input, computes cosine against every anchor utterance of every route, keeps
the best score per route, then:

```
top1 = best route score      top2 = second-best route score
margin = top1 − top2
```

Decision rules (from `config.yaml`):

```
top1 ≥ tau_high (0.45) AND margin ≥ delta_margin (0.08)  → route top1 (confident)
top1 < tau_low (0.35)                                    → fallback → vectorial
otherwise                                                → gray zone → content evidence
```

This is a **1-NN classifier with a rejection rule** — a classic ML concept, good to cite.

The gray zone does NOT call a model. It re-scores the question against the section headings each
store declares (`RAGBase.content_anchors`) and takes the max per route against the configured
anchors. Measured: the LLM judge that used to live here scored 8/10 where geometry alone scored
9/10; the headings took routing on unseen questions from 82% to 95%.

## FAISS IndexFlatIP vs HNSW (Qdrant)

- **`IndexFlatIP`** (used in CAG + router): exact brute-force search over all N vectors using SIMD.
  With N ≤ ~10k it is microseconds. It is exact (not approximate).
- **HNSW** (inside Qdrant for RAG 2): approximate search over millions of vectors. A multi-layer
  navigable graph (top layer = long-range links, lower layers = finer links). Search is ~log(N) and
  trade exactness for speed (recall ~99%). Analogy: highway → avenue → street.

## PageRank and Personalized PageRank (RAG 3)

A **graph** = nodes (entities: "Fornecedor A", "Lei 13.709", ...) + directed edges (relations).
**PageRank** = importance of a node from the importance of those pointing to it (Google, 1998).
Modeled as a **random walk**: a walker picks a random outgoing edge each step; the long-run fraction
of time on each node is the PageRank.

```
PR(nó) = (1−d)/N  +  d × Σ PR(vizinho) / out_degree(vizinho)      d = 0.85
```

Mathematically this is the **principal eigenvector of the transition matrix** (stationary
distribution of a Markov chain). `networkx` solves it by iteration (~100 steps).

**Personalized PageRank** = the teleport term returns **only to the seed nodes** (the entities found
in the user's question) instead of any node. So it measures *relevance to THIS question*, enabling
**multi-hop reasoning**: importance flows across edges for several hops (A → contract Y → clause Z)
without any programmed rules.

Use: `nx.personalized_pagerank(G, personalization={seed: 1.0, ...})`.

> ⚠️ **API warning (validated during skill eval):** recent networkx releases (3.6+) **removed**
> `nx.personalized_pagerank` and made `nx.pagerank` require `scipy`. Do not rely on them. Implement a
> small **pure-numpy power iteration** instead (validated to match networkx to <1e-4 and
> bit-for-bit deterministic):
>
> ```
> r_new = (1−d)·personalization + d · (Σ r[u]/outdeg(u) for edges u→v) + d·dangling·p
> iterate until Σ|Δr| < tol
> ```
>
> Vectorize with `np.add.at` over flattened edge arrays; remember to handle dangling nodes (outdeg 0)
> by redistributing their mass into the personalization vector.

## Quantization (why models fit in VRAM)

Models natively store weights as 16-bit floats (FP16). **Quantization** compresses to ~4 bits
(Q4_K_M) — 4× less memory, minimal quality loss.

```
A 14B model in FP16: 14B × 2 bytes ≈ 28 GB  → too big
The same in Q4:   14B × 0.5 byte ≈ 9 GB   → fits in 16GB (this project runs a 4B: ~2.9GB)
```

## SQL as relational algebra (interview bonus)

SQL is set theory: `WHERE` = selection (σ), `SELECT cols` = projection (π), `JOIN` = filtered
Cartesian product (⨝), `GROUP BY` = partition + aggregation.

## Reciprocal Rank Fusion (RRF) — merging multi-route results

Used when a question is **composite** and the router fans out to more than one RAG. Each RAG ranks
chunks on a different scale (SQL = aggregated value, Qdrant = cosine, graph = PageRank), so raw
scores are not comparable. RRF uses only the ranks:

```
score_final(chunk) = Σ_rotas  1 / (k + rank_na_rota)        k = 60 (classic constant)

example: chunk ranked #2 in SQL and #5 in graph
         → 1/(60+2) + 1/(60+5) = 0.0161 + 0.0154 = 0.0315
```

- **Deterministic** (pure arithmetic), **scale-agnostic** (works across heterogeneous rankers).
- Chunks retrieved by multiple routes get boosted; the final list is sorted by `score_final`.

## Key constants

- BGE-M3 embedding dimension: **1024**
- Default thresholds: `tau_high = 0.45`, `tau_low = 0.35`, `delta_margin = 0.08`,
  `tau_multi = 0.5` (if top2 ≥ tau_multi → fan-out to multiple routes)
- CAG hit threshold: `cosine ≥ 0.80`
- Chunk size 512 tokens, overlap 64
- RRF constant: `k = 60`
