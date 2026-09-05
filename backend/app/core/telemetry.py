from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from app.core.config import Settings


def _to_payload(span: ReadableSpan) -> dict[str, Any]:
    """The shape the frontend timeline consumes."""
    context = span.get_span_context()
    return {
        "trace_id": f"{context.trace_id:032x}",
        "span_id": f"{context.span_id:016x}",
        # The panel draws a tree, so each span has to say who its parent is.
        "parent_span_id": f"{span.parent.span_id:016x}" if span.parent else None,
        "name": span.name,
        "start_ms": (span.start_time or 0) / 1e6,
        "duration_ms": ((span.end_time or 0) - (span.start_time or 0)) / 1e6,
        "attributes": dict(span.attributes or {}),
    }


class LiveSpanExporter(SpanExporter):
    """Keeps recent spans in RAM and pushes each one to live subscribers.

    No Collector, no Jaeger, no extra process — the panel is fed straight from
    the SDK. Using OpenTelemetry rather than ad-hoc event dicts is what makes
    "point it at Jaeger in production" a one-line change instead of a rewrite.
    """

    def __init__(self, ring_buffer_size: int) -> None:
        # maxlen makes this self-trimming: the oldest span falls off on append.
        self.recent: deque[dict] = deque(maxlen=ring_buffer_size)
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            payload = _to_payload(span)
            self.recent.append(payload)
            for queue in self._subscribers:
                queue.put_nowait(payload)
        return SpanExportResult.SUCCESS


def setup_telemetry(settings: Settings) -> LiveSpanExporter:
    """Installs the global tracer provider. Returns the exporter the API reads from."""
    exporter = LiveSpanExporter(settings.telemetry.ring_buffer_size)
    provider = TracerProvider(resource=Resource.create({"service.name": settings.telemetry.service_name}))
    # Simple, not Batch: the panel is live, so a span has to leave the SDK the
    # moment it ends instead of waiting for a batch window to close.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter
