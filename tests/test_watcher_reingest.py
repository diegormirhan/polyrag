"""What the watcher considers new, now that identity is content and not a name.

Until this, "already ingested" meant "a file with this name reached the archive",
so correcting a document and dropping it back under the same name did nothing at
all; the only way to re-ingest was to rename the file.
"""

import asyncio
import json
from pathlib import Path

from app.core.config import load_config
from app.ingest.watcher import Watcher, file_digest


def _settings(tmp_path):
    settings = load_config()
    drop = tmp_path / "drop"
    drop.mkdir()
    return settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={
                    "data_drop": str(drop),
                    "processed": str(tmp_path / "processed"),
                    "ingest_manifest": str(tmp_path / "manifest.json"),
                }
            )
        }
    )


def _manifest(settings, entries: dict) -> None:
    Path(settings.paths.ingest_manifest).write_text(json.dumps(entries), encoding="utf-8")


def test_an_unseen_file_is_offered(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "drop" / "notas.md").write_text("conteudo original", encoding="utf-8")
    assert [f.path.name for f in asyncio.run(Watcher(settings).poll_once())] == ["notas.md"]


def test_the_same_bytes_are_skipped(tmp_path):
    settings = _settings(tmp_path)
    path = tmp_path / "drop" / "notas.md"
    path.write_text("conteudo original", encoding="utf-8")
    _manifest(settings, {"notas.md": {"digest": file_digest(path), "stored": []}})
    assert asyncio.run(Watcher(settings).poll_once()) == []


def test_a_corrected_file_under_the_same_name_is_offered_again(tmp_path):
    """The limitation this replaced: same name, different content, was invisible."""
    settings = _settings(tmp_path)
    path = tmp_path / "drop" / "notas.md"
    path.write_text("conteudo original", encoding="utf-8")
    _manifest(settings, {"notas.md": {"digest": file_digest(path), "stored": []}})

    path.write_text("conteudo corrigido", encoding="utf-8")
    assert [f.path.name for f in asyncio.run(Watcher(settings).poll_once())] == ["notas.md"]


def test_the_same_content_under_a_new_name_is_skipped(tmp_path):
    """The other half of content identity: a copy is not a new document."""
    settings = _settings(tmp_path)
    original = tmp_path / "drop" / "notas.md"
    original.write_text("conteudo original", encoding="utf-8")
    _manifest(settings, {"notas.md": {"digest": file_digest(original), "stored": []}})
    original.rename(tmp_path / "drop" / "notas-copia.md")
    assert asyncio.run(Watcher(settings).poll_once()) == []
