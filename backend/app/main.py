from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.api.v1.router import api_router
from app.core.config import load_config
from app.core.llama_client import LlamaClients
from app.core.telemetry import setup_telemetry
from app.pipeline.cache import SemanticCache
from app.pipeline.ingestor import Ingestor
from app.pipeline.orchestrator import Orchestrator
from app.pipeline.router import Router
from app.rags.factory import build_rags


async def _watch_data_drop(app: FastAPI, interval: float) -> None:
    """The hot folder: anything dropped in data_drop/ gets ingested on its own."""
    while True:
        try:
            for report in await app.state.ingestor.ingest_pending():
                app.state.jobs.append(report)
        except Exception as error:
            # Not precaution: a chunk that looked tabular but held no table already
            # crashed ingestion once during development. In a background task that
            # failure would have killed ingestion for the whole session, silently.
            print(f"[ingest] falhou: {error!r}")
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_config()

    # Router and RAGs are built once and shared by both pipelines. GraphRAG keeps
    # the graph in memory, so separate copies would hide everything the ingestor
    # writes from everything the search reads.
    clients = LlamaClients(settings)
    router = await Router.create(clients, settings)
    rags = await build_rags(clients, settings)
    cache = SemanticCache(settings) if settings.cache.enabled else None

    app.state.orchestrator = Orchestrator(clients, router, cache, rags, settings)
    app.state.ingestor = Ingestor(clients, router, rags, settings)
    app.state.jobs = deque(maxlen=100)

    watcher = asyncio.create_task(_watch_data_drop(app, settings.ingest.watch_interval_s))
    yield
    watcher.cancel()


def create_app() -> FastAPI:
    settings = load_config()
    app = FastAPI(
        title="PolyRAG",
        version="0.1.0",
        description="Federated multi-modal RAG with deterministic semantic routing.",
        lifespan=lifespan,
    )
    # Before instrument_app, so the request spans land in the same provider as the
    # pipeline spans and share one trace per request.
    app.state.exporter = setup_telemetry(settings)
    # The telemetry routes must not be instrumented, or the stream feeds itself:
    # sending a span over the WebSocket makes the ASGI layer emit a "websocket send"
    # span, which is exported, which is sent, forever. Measured before the fix: 499
    # of the 500 spans in the ring buffer were websocket sends, and every real
    # pipeline span had been pushed out of the buffer.
    FastAPIInstrumentor.instrument_app(app, excluded_urls="api/v1/telemetry")
    app.include_router(api_router)
    return app


app = create_app()
