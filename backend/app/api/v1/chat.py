from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from opentelemetry import trace

from app.api.v1.deps import get_orchestrator
from app.pipeline.orchestrator import Orchestrator
from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    result = await orchestrator.answer(body.message)
    decision = result.decision
    return ChatResponse(
        answer=result.answer,
        cache_hit=result.cache_hit,
        # The request span and the pipeline spans share a trace, so this id is what
        # the panel filters on to show exactly the steps behind this answer.
        trace_id=f"{trace.get_current_span().get_span_context().trace_id:032x}",
        sources=result.sources,
        route=decision.route if decision else None,
        decision_stage=decision.decision_stage if decision else None,
        margin=decision.margin if decision else None,
        scores=decision.scores if decision else {},
    )


@router.delete("/cache", status_code=204)
async def clear_cache(orchestrator: Orchestrator = Depends(get_orchestrator)) -> None:
    orchestrator.clear_cache()


@router.websocket("/chat/stream")
async def chat_stream(websocket: WebSocket) -> None:
    """Streams one answer: the routing decision first, then sources, then tokens.

    The decision goes out before retrieval even starts, so the panel can draw the
    route and its margin while the RAG is still working — the wait shows the
    pipeline thinking instead of a spinner.
    """
    orchestrator: Orchestrator = websocket.app.state.orchestrator
    await websocket.accept()
    try:
        while True:
            question = (await websocket.receive_json())["message"]
            async for event in orchestrator.answer_stream(question):
                await websocket.send_json(_serialize(event))
    except WebSocketDisconnect:
        pass


def _serialize(event: dict) -> dict:
    """Domain objects out, JSON in. The orchestrator should not know about the wire."""
    if event["type"] == "decision":
        decision = event["decision"]
        return {
            "type": "decision",
            "route": decision.route,
            "decision_stage": decision.decision_stage,
            "margin": decision.margin,
            "scores": decision.scores,
        }
    return event
