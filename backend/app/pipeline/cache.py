from __future__ import annotations

import faiss
import numpy as np

from app.core.config import Settings


class SemanticCache:
    """CAG — a repeated (or nearly repeated) question is answered straight from RAM.

    Takes vectors, not text: the embedding is computed once by the orchestrator and
    reused here. That keeps this class pure math + storage, so it can be unit-tested
    with handmade vectors without any llama-server running.
    """

    def __init__(self, settings: Settings) -> None:
        self._threshold = settings.cache.similarity_threshold
        self._max_entries = settings.cache.max_entries
        # IndexFlatIP = exact brute-force search by inner product. Vectors arrive
        # already normalized, so the inner product IS the cosine similarity —
        # no division by norms, same trick used in the router.
        self._index = faiss.IndexFlatIP(settings.llama.embeddings.dimensions)
        self._answers: list[str] = []

    def lookup(self, vector: list[float]) -> str | None:
        """Answer stored for the nearest known question, if it is close enough."""
        if self._index.ntotal == 0:
            return None
        scores, ids = self._index.search(_as_batch(vector), 1)
        if scores[0][0] < self._threshold:
            return None
        return self._answers[ids[0][0]]

    def store(self, vector: list[float], answer: str) -> None:
        # Full cache simply stops accepting new entries. No eviction policy until
        # a real workload proves one is needed.
        if self._index.ntotal >= self._max_entries:
            return
        self._index.add(_as_batch(vector))
        self._answers.append(answer)

    def clear(self) -> None:
        self._index.reset()
        self._answers.clear()


def _as_batch(vector: list[float]) -> np.ndarray:
    # FAISS always works on batches: a single vector goes in as a 1-row matrix.
    return np.array([vector], dtype="float32")
