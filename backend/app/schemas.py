from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    cache_hit: bool
    trace_id: str  # ties the answer to its spans in the telemetry panel
    sources: list[Any]
    # All null on a cache hit: the answer came from RAM, so no routing happened.
    route: str | None = None
    decision_stage: str | None = None
    margin: float | None = None
    scores: dict[str, float] = Field(default_factory=dict)


class IngestReportOut(BaseModel):
    file: str
    chunks: int
    routes: list[str]  # where each chunk actually landed, which is not always
    # the route the router asked for — see Ingestor._store


class ServiceHealth(BaseModel):
    name: str
    url: str
    up: bool
    sleeping: bool | None = None  # llama-servers only; asleep still counts as up


class HealthResponse(BaseModel):
    ok: bool
    services: list[ServiceHealth]
