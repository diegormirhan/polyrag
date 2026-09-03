from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any


def content_id(content: str) -> str:
    """Deterministic ID derived from the content itself.

    The same text always yields the same ID, so re-ingesting a chunk overwrites
    the existing record instead of creating a duplicate — re-running ingestion
    used to fill the top-k with copies of the same chunk. uuid5 rather than a
    plain hash because Qdrant only accepts a UUID or an integer as a point id.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, content))


class RAGBase(ABC):
    """Common shape for the three RAG backends (relational, vectorial, graph).

    Deliberately loose on types: each RAG takes a different kind of content
    (DataFrame vs. text chunk) and returns a different kind of result, so the
    concrete classes narrow `Any` in their own signatures instead of forcing a
    shared type that would not mean anything.
    """

    @abstractmethod
    async def ingest(self, content: Any) -> None:
        """Store content into this RAG's backend. Shape of `content` is RAG-specific."""

    @abstractmethod
    async def query(self, question: str, top_k: int = 5) -> list[Any]:
        """Retrieve top_k relevant items for `question`. Shape of results is RAG-specific."""

    @abstractmethod
    async def stats(self) -> dict[str, Any]:
        """What this backend currently holds. Each store counts different things."""
