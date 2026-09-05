from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from opentelemetry import trace

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients
from app.ingest import loaders, ocr
from app.ingest.chunking import chunks
from app.ingest.watcher import FileKind, IngestFile, Watcher
from app.pipeline.router import Router
from app.rags.base import RAGBase
from app.rags.factory import build_rags

_tracer = trace.get_tracer("polyrag.pipeline")


@dataclass(frozen=True)
class IngestReport:
    path: Path
    routes: list[str]  # the route each stored chunk went to, in order


def _table_name(path: Path, index: int | None = None) -> str:
    """SQLite-safe table name derived from the file name."""
    name = re.sub(r"\W+", "_", path.stem).strip("_").lower() or "table"
    if name[0].isdigit():
        name = f"t_{name}"
    return name if index is None else f"{name}_{index}"


def _markdown_table_to_frame(markdown: str) -> pd.DataFrame:
    """Markdown table (what OCR emits for tables in images) back into a DataFrame.

    Returns an empty frame when the text contains no table at all.
    """
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in markdown.strip().splitlines()
        if line.strip().startswith("|")
    ]
    if not rows:
        return pd.DataFrame()
    header, *body = rows
    # Drop the |---|:--:| separator row — it is layout, not data.
    body = [row for row in body if not all(set(cell) <= set("-: ") for cell in row)]
    return pd.DataFrame(body, columns=header)


class Ingestor:
    """The ingestion path: file -> text -> chunks -> router -> the right RAG.

    Mirror image of the Orchestrator. Same router, same three RAGs, opposite
    question: this one asks "where does this content belong?", that one asks
    "where should this question be answered from?".
    """

    def __init__(
        self,
        clients: LlamaClients,
        router: Router,
        rags: dict[str, RAGBase],
        settings: Settings,
    ) -> None:
        self._clients = clients
        self._router = router
        self._rags = rags
        self._settings = settings
        self._watcher = Watcher(settings)

    @classmethod
    async def create(cls, clients: LlamaClients, settings: Settings | None = None) -> Ingestor:
        settings = settings or load_config()
        router = await Router.create(clients, settings)
        return cls(clients, router, await build_rags(clients, settings), settings)

    async def ingest_pending(self) -> list[IngestReport]:
        """One pass over data_drop/. The infinite watch loop belongs to the caller."""
        return [await self.ingest_file(f) for f in await self._watcher.poll_once()]

    async def ingest_file(self, ingest_file: IngestFile) -> IngestReport:
        with _tracer.start_as_current_span("pipeline.ingest") as span:
            span.set_attributes(
                {
                    "ingest.file": ingest_file.path.name,
                    "ingest.kind": ingest_file.kind.value,
                }
            )
            if ingest_file.kind == FileKind.TABLE:
                # A spreadsheet is tabular by definition, so routing it would be asking
                # a question whose answer is already known. Only chunks get routed.
                frame = loaders.load_table(ingest_file.path)
                await self._rags["relational"].ingest((frame, _table_name(ingest_file.path)))
                routes = ["relational"]
            else:
                routes = await self._route_chunks(await self._to_text(ingest_file), ingest_file.path)

            span.set_attributes({"ingest.chunks": len(routes), "ingest.routes": routes})
            self._archive(ingest_file.path)
            self._record(ingest_file.path.name, routes)
            return IngestReport(ingest_file.path, routes)

    async def _to_text(self, ingest_file: IngestFile) -> str:
        if ingest_file.kind == FileKind.VISION:
            return await ocr.extract_text(ingest_file.path, self._clients)
        if ingest_file.kind == FileKind.DOCUMENT:
            return loaders.load_document(ingest_file.path)
        return loaders.load_text(ingest_file.path)

    async def _route_chunks(self, text: str, path: Path) -> list[str]:
        routes = []
        for index, chunk in enumerate(await chunks(text, self._clients, self._settings)):
            with _tracer.start_as_current_span("pipeline.ingest.chunk") as span:
                decision = await self._router.route(chunk)
                stored_in = await self._store(decision.route, chunk, path, index)
                # router.route and ingest.stored_in are separate attributes on purpose:
                # when they disagree, a chunk was sent to `relational` and turned out to
                # hold no table. That divergence is the interesting signal, not an error.
                span.set_attributes(
                    {
                        "chunk.index": index,
                        "chunk.chars": len(chunk),
                        "router.route": decision.route,
                        "router.decision_stage": decision.decision_stage,
                        "router.margin": decision.margin,
                        "ingest.stored_in": stored_in,
                        **{f"router.score.{r}": s for r, s in decision.scores.items()},
                    }
                )
            routes.append(stored_in)
        return routes

    async def _store(self, route: str, chunk: str, path: Path, index: int) -> str:
        """Stores the chunk and returns the route it actually landed in."""
        if route == "relational":
            frame = _markdown_table_to_frame(chunk)
            if not frame.empty:
                await self._rags[route].ingest((frame, _table_name(path, index)))
                return route
            # The router reaches `relational` for content that merely reads as
            # tabular. The path is the embedding stage, which weighs meaning and
            # never inspects structure at all — measured, the receipt that caused
            # this scores 0.53 on the tabularity heuristic, well under its 0.8
            # threshold, so the heuristic was not the one that let it through.
            # SQLite needs real rows and columns, so a chunk holding no table is
            # kept as free text.
            route = "vectorial"

        await self._rags[route].ingest(chunk)
        return route

    def _record(self, name: str, routes: list[str]) -> None:
        """Persists where a file's chunks landed.

        Without this the mapping lives only in the running process, so a restart
        loses the one thing the corpus view exists to show. Rewritten whole rather
        than appended: the file is small, and a truncated append would be worse
        than a lost entry.
        """
        manifest = Path(self._settings.paths.ingest_manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        entries[name] = {"chunks": len(routes), "routes": routes}
        manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    def _archive(self, path: Path) -> None:
        # Moving into data/processed/ is what marks a file as done: the watcher
        # defines "new" as "not present in that folder".
        processed = Path(self._settings.paths.processed)
        processed.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(processed / path.name))
