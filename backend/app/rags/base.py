from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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
