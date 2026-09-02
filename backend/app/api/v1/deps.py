from __future__ import annotations

from collections import deque

from fastapi import Request

from app.core.config import Settings, load_config
from app.pipeline.ingestor import Ingestor
from app.pipeline.orchestrator import Orchestrator


def get_settings() -> Settings:
    """Never `Depends(load_config)` directly.

    FastAPI turns a dependency's parameters into request parameters, and
    load_config takes `path` — so depending on it straight would publish
    `?path=` on every endpoint using it and let a caller point the loader at any
    file on disk. A zero-argument wrapper is the whole fix.
    """
    return load_config()


def get_orchestrator(request: Request) -> Orchestrator:
    # Built once in the lifespan: constructing it per request would rebuild the
    # router anchors and reload the graph on every message.
    return request.app.state.orchestrator


def get_ingestor(request: Request) -> Ingestor:
    return request.app.state.ingestor


def get_jobs(request: Request) -> deque:
    return request.app.state.jobs
