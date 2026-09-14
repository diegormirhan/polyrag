from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.core.config import Settings, load_config


def file_digest(path: Path) -> str:
    """SHA-256 of the file's bytes — the identity the pipeline tracks it by.

    Bytes rather than the name, because the name is what the user types and the
    content is what was ingested. Read in blocks so a large PDF is not held in
    memory to be hashed.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# A closed set of allowed "kinds" of file. Using an Enum instead of raw strings
# means typos get caught by the type checker instead of silently misrouting a file.
class FileKind(StrEnum):
    VISION = "vision"  # image -> needs OCR before it becomes text
    TEXT = "text"  # already plain text, read as-is
    TABLE = "table"  # csv/xlsx -> read as a pandas DataFrame
    DOCUMENT = "document"  # pdf/docx/pptx -> needs a format-specific extractor


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

    def _ingested_digests(self) -> set[str]:
        """The content digest of every file already ingested, from the manifest.

        The manifest, and no longer the archive folder, is what "already ingested"
        means. Keying on the name made a file ingestable exactly once: correcting a
        typo in a document and dropping it back under the same name did nothing,
        and the only way to re-ingest was to rename the file. Keying on the content
        makes the right thing happen for both cases -- the same bytes are skipped
        however they are named, and changed bytes are ingested even under a name
        already seen.
        """
        manifest = Path(self._settings.paths.ingest_manifest)
        if not manifest.exists():
            return set()
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        return {entry["digest"] for entry in entries.values() if "digest" in entry}

    def _candidates(self) -> list[Path]:
        data_drop = Path(self._settings.paths.data_drop)
        ingested = self._ingested_digests()
        return [path for path in data_drop.glob("*") if path.is_file() and file_digest(path) not in ingested]

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
