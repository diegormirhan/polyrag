from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat, corpus, health, ingest, telemetry

# The version lives in the prefix, in one place: /api/v2 later means a second
# aggregator, not a rename spread across every endpoint file.
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(ingest.router, tags=["ingest"])
api_router.include_router(corpus.router, tags=["ingest"])
api_router.include_router(telemetry.router, tags=["telemetry"])
