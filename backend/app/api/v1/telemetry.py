from __future__ import annotations

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.get("/telemetry/traces")
async def traces(request: Request) -> list[dict]:
    """The ring buffer of recent spans — what the panel draws on first load."""
    return list(request.app.state.exporter.recent)


@router.websocket("/telemetry/stream")
async def stream(websocket: WebSocket) -> None:
    exporter = websocket.app.state.exporter
    await websocket.accept()
    queue = exporter.subscribe()
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        # Without this the exporter keeps pushing into a queue nobody reads, and
        # it grows for as long as the process lives.
        exporter.unsubscribe(queue)
