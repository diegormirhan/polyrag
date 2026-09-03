from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator

import pandas as pd
from opentelemetry import trace
_tracer = trace.get_tracer("polyrag.pipeline")

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, chat_stream, embed
from app.core.vectors import normalize
from app.pipeline.cache import SemanticCache
from app.pipeline.router import RouteDecision, Router
from app.rags.base import RAGBase
from app.rags.factory import build_rags

_tracer = trace.get_tracer("polyrag.pipeline")

@dataclass(frozen=True)
class ChatResult:
    answer: str
    cache_hit: bool
    decision: RouteDecision | None  # None when the cache answered — no routing happened
    sources: list[Any]


def _format_context(route: str, results: list[Any]) -> str:
    # SQL rows are dicts of columns; the other two RAGs return chunks of text.
    if route == "relational":
        return pd.DataFrame(results).to_markdown(index=False)
    return "\n\n".join(item["text"] for item in results)


class Orchestrator:
    """The search path: Cache -> Router -> RAG -> LLM."""

    def __init__(
        self,
        clients: LlamaClients,
        router: Router,
        cache: SemanticCache | None,
        rags: dict[str, RAGBase],
        settings: Settings,
    ) -> None:
        self._clients = clients
        self._router = router
        self._cache = cache
        self._rags = rags
        self._settings = settings

    @classmethod
    async def create(cls, clients: LlamaClients, settings: Settings | None = None) -> "Orchestrator":
        settings = settings or load_config()
        rags = await build_rags(clients, settings)
        # A disabled cache is simply no cache: one source of truth, instead of an
        # `enabled` flag checked again at every call site.
        cache = SemanticCache(settings) if settings.cache.enabled else None
        router = await Router.create(clients, settings)
        return cls(clients, router, cache, rags, settings)

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    async def answer(self, question: str) -> ChatResult:
        """Collects the stream into one result, for callers that want the whole answer."""
        answer, cache_hit, decision, sources = "", False, None, []
        async for event in self.answer_stream(question):
            match event["type"]:
                case "token":
                    answer += event["text"]
                case "cache_hit":
                    cache_hit = True
                case "decision":
                    decision = event["decision"]
                case "sources":
                    sources = event["sources"]
        return ChatResult(answer, cache_hit, decision, sources)

    async def answer_stream(self, question: str) -> AsyncIterator[dict]:
        """The search path, emitting each step as it happens.

        This is the primary implementation and answer() consumes it, so the
        streaming and non-streaming responses cannot drift apart. Events carry
        domain objects (RouteDecision); turning them into JSON is the API layer's
        job, not this one's.

        Spans live here rather than inside cache.py and router.py: those stay pure
        (the cache is testable with handmade vectors, no server involved), and
        RouteDecision already carries every number the panel needs.
        """
        with _tracer.start_as_current_span("pipeline.chat"):
            vector = normalize((await embed(self._clients.embeddings, [question]))[0])

            with _tracer.start_as_current_span("pipeline.cache") as span:
                cached = self._cache.lookup(vector) if self._cache is not None else None
                span.set_attribute("cache.hit", cached is not None)
            if cached is not None:
                yield {"type": "cache_hit"}
                yield {"type": "token", "text": cached}
                yield {"type": "done"}
                return

            # On a miss the router embeds the question a second time. Left as is: this
            # path already costs seconds of SQL/LLM work, so one extra embedding call is
            # noise, and keeping route() self-contained is worth more than the microopt.
            with _tracer.start_as_current_span("pipeline.router") as span:
                decision = await self._router.route(question)
                span.set_attributes({
                    "router.route": decision.route,
                    "router.decision_stage": decision.decision_stage,
                    "router.score_top1": decision.score_top1,
                    "router.score_top2": decision.score_top2,
                    "router.margin": decision.margin,
                    # Flattened one key per route: OTel attributes take primitives,
                    # never a dict.
                    **{f"router.score.{r}": s for r, s in decision.scores.items()},
                })
            # Emitted before retrieval so the panel can draw the decision while the
            # RAG is still working — the wait becomes the product.
            yield {"type": "decision", "decision": decision}

            with _tracer.start_as_current_span(f"pipeline.rag.{decision.route}") as span:
                results = await self._rags[decision.route].query(
                    question, top_k=self._settings.orchestrator.top_k
                )
                span.set_attribute("rag.results", len(results))
            yield {"type": "sources", "sources": results}

            if not results:
                yield {"type": "token", "text": "Não encontrei nada nas bases sobre isso."}
                yield {"type": "done"}
                return

            prompt = self._settings.orchestrator.answer_prompt.format(
                context=_format_context(decision.route, results),
                question=question,
            )
            answer = ""
            async for piece in chat_stream(
                self._clients.llm,
                [{"role": "user", "content": prompt}],
                temperature=self._settings.llama.llm.temperature,
            ):
                answer += piece
                yield {"type": "token", "text": piece}

            if self._cache is not None:
                self._cache.store(vector, answer)
            yield {"type": "done"}
