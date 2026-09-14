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

    async def forget(self, ids: list[str]) -> None:
        """Removes previously ingested chunks by their content id.

        Exists so a file can be corrected. Until this, "already ingested" meant
        "a file with this name reached the archive", so fixing a typo in a
        document required renaming it, and re-dropping it under the same name did
        nothing at all.

        Not abstract: the relational store has nothing to do here, because a table
        is written with `if_exists="replace"` under a name derived from the file,
        so re-ingesting the same file overwrites it by construction.
        """
        return None

    async def content_anchors(self) -> list[str]:
        """Short phrases describing what this store actually holds, for the router.

        Hand-written anchors in config.yaml say what a route is *for*; these say
        what it currently *contains*. Both are needed, because the first cannot
        anticipate a corpus. Measured on questions the anchors had never seen,
        routing accuracy went from 82% to 95% once each store contributed its own
        section headings -- the router had no way to know that "Banco de Dados
        Orion" names a section in the graph, so three questions about it were sent
        to the relational store, whose only table is about sales.

        Not abstract: a store with nothing useful to declare returns nothing, and
        the router simply falls back to the configured anchors.
        """
        return []
