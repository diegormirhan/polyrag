from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from app.api.v1.deps import get_settings
from app.core.config import Settings
from app.rags.base import RAGBase
from app.schemas import CorpusFile, CorpusResponse

router = APIRouter()


def get_rags(request: Request) -> dict[str, RAGBase]:
    return request.app.state.rags


@router.get("/corpus", response_model=CorpusResponse)
async def corpus(
    rags: dict[str, RAGBase] = Depends(get_rags),
    settings: Settings = Depends(get_settings),
) -> CorpusResponse:
    """What has been ingested, and what each store ended up holding.

    The file list comes from data/processed rather than an in-memory job log,
    because that folder IS the record: the watcher defines "already ingested" as
    "present there", so it survives restarts and matches what the pipeline believes.
    """
    manifest_path = Path(settings.paths.ingest_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    processed = Path(settings.paths.processed)
    files = sorted(
        (
            CorpusFile(
                name=path.name,
                size_bytes=path.stat().st_size,
                ingested_at=path.stat().st_mtime,
                **manifest.get(path.name, {}),
            )
            for path in processed.glob("*")
            if path.is_file()
        ),
        key=lambda file: file.ingested_at,
        reverse=True,
    )

    names = list(rags)
    stats = await asyncio.gather(*(rags[name].stats() for name in names))
    return CorpusResponse(files=files, stores=dict(zip(names, stats)))
