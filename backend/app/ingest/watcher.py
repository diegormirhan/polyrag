from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncIterator
import asyncio

from app.core.config import Settings, load_config

# A closed set of allowed "kinds" of file. Using an Enum instead of raw strings
# means typos get caught by the type checker instead of silently misrouting a file.
class FileKind(str, Enum):
    VISION = "vision"      # image -> needs OCR before it becomes text
    TEXT = "text"           # already plain text, read as-is
    TABLE = "table"          # csv/xlsx -> read as a pandas DataFrame
    DOCUMENT = "document"     # pdf/docx/pptx -> needs a format-specific extractor


# Small immutable pair: which file, and what kind it is. frozen=True stops anything
# downstream from accidentally mutating it once the watcher hands it off.
@dataclass(frozen=True)
class IngestFile:
    path: Path
    kind: FileKind


# Pure function: extension -> FileKind, driven entirely by config.yaml (no hardcoded
# extension lists here). Returns None for anything not listed, so it gets skipped.
def _classify(path: Path, settings: Settings) -> FileKind | None:
    suffix = path.suffix.lower()
    if suffix in settings.ingest.vision_extensions:
        return FileKind.VISION
    if suffix in settings.ingest.text_extensions:
        return FileKind.TEXT
    if suffix in settings.ingest.table_extensions:
        return FileKind.TABLE
    if suffix in settings.ingest.document_extensions:
        return FileKind.DOCUMENT
    return None


class Watcher:
    """Polls data_drop/ and yields new, recognized files."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_config()

    def _candidates(self) -> list[Path]:
        # "New" = sitting in data_drop/ and NOT already moved into data/processed/.
        # If it's still in data_drop, by definition it hasn't been ingested yet.
        data_drop = Path(self._settings.paths.data_drop)
        processed = Path(self._settings.paths.processed)
        processed_names = {p.name for p in processed.glob("*")} if processed.exists() else set()

        return [
            p
            for p in data_drop.glob("*")
            if p.is_file() and p.name not in processed_names
        ]

    async def poll_once(self) -> list[IngestFile]:
        # One check-cycle, no waiting/looping inside — kept separate from watch()
        # so it can be unit-tested directly (call it, assert on the result, done).
        ready: list[IngestFile] = []
        for path in self._candidates():
            kind = _classify(path, self._settings)
            if kind is not None:
                ready.append(IngestFile(path=path, kind=kind))
        return ready

    async def watch(self) -> AsyncIterator[IngestFile]:
        # Infinite async generator: every watch_interval_s seconds, poll once and
        # hand out (yield) each ready file, one at a time, to whoever does
        # `async for f in watcher.watch(): ...`
        interval = self._settings.ingest.watch_interval_s
        while True:
            for ingest_file in await self.poll_once():
                yield ingest_file
            await asyncio.sleep(interval)