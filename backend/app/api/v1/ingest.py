from __future__ import annotations

from collections import deque
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.v1.deps import get_ingestor, get_jobs, get_settings
from app.core.config import Settings
from app.pipeline.ingestor import Ingestor
from app.schemas import IngestReportOut

router = APIRouter()


@router.post("/ingest", response_model=list[IngestReportOut])
async def ingest(
    file: UploadFile,
    ingestor: Ingestor = Depends(get_ingestor),
    jobs: deque = Depends(get_jobs),
    settings: Settings = Depends(get_settings),
) -> list[IngestReportOut]:
    """Uploads a file into data_drop/ and ingests everything waiting there.

    The upload lands in the hot folder rather than being handled specially, so an
    uploaded file and a file dropped in by hand take exactly the same path — one
    code path to reason about, one to keep working.
    """
    name = Path(file.filename or "upload").name
    # The watcher defines "new" as "not already in data/processed", so uploading a
    # known name used to write the file into the hot folder, ingest nothing, return
    # an empty list, and leave the upload sitting there forever. Refusing up front
    # says what happened and leaves no orphan behind.
    if (Path(settings.paths.processed) / name).exists():
        raise HTTPException(
            status_code=409,
            detail=f"{name} has already been ingested. Rename it to ingest a new version.",
        )

    destination = Path(settings.paths.data_drop) / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(await file.read())

    reports = await ingestor.ingest_pending()
    jobs.extend(reports)
    return [IngestReportOut(file=r.path.name, chunks=len(r.routes), routes=r.routes) for r in reports]


@router.get("/ingest/jobs", response_model=list[IngestReportOut])
async def jobs(jobs: deque = Depends(get_jobs)) -> list[IngestReportOut]:
    return [IngestReportOut(file=r.path.name, chunks=len(r.routes), routes=r.routes) for r in jobs]
