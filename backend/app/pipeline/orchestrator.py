from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pandas as pd
from opentelemetry import trace

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


def _edges_first(results: list[Any]) -> list[Any]:
    """Reorders a ranked list so the best passages sit at both ends of the prompt.

    Liu et al. 2023, "Lost in the Middle": a model attends to the start and the end
    of a long context and least to its middle, and accuracy on a fact placed in the
    middle drops measurably. Handing the model a list sorted best-to-worst puts the
    second-best passage exactly where it is least likely to be read.

    Ranks 1,3,5 go out in order and 2,4 come back reversed, so rank 1 opens the
    context and rank 2 closes it: [1,2,3,4,5] -> [1,3,5,4,2]. Pure list slicing,
    no scores involved -- the retrieval step already did the ranking, and this only
    decides where each rank is placed.

    Not applied to SQL rows: their order is the ORDER BY that selected them, and
    shuffling it would misrepresent the query's own answer.
    """
    return results[0::2] + results[1::2][::-1]


def _format_context(route: str, results: list[Any]) -> str:
    # SQL rows are dicts of columns; the other two RAGs return chunks of text.
    if route == "relational":
        sql = next((row["sql"] for row in results if "sql" in row), None)
        rows = [row for row in results if "sql" not in row]
        # The query is part of the context, not decoration: one row is meaningless
        # without the ORDER BY that selected it. Measured, `ORDER BY receita ASC
        # LIMIT 1` returned the right row and the model answered "only a list of
        # regions, impossible to determine".
        table = pd.DataFrame(rows).to_markdown(index=False)
        return f"Consulta SQL executada:\n{sql}\n\nResultado:\n{table}"
    return "\n\n".join(item["text"] for item in _edges_first(results))


# What each store is called in a multi-source context, so the model can say where
# a fact came from and the reader can check it.
ORIGIN = {"relational": "SQL", "vectorial": "TEXT", "graph": "GRAPH"}


def _join_context(found: list[tuple[str, list[Any]]]) -> str:
    """One block per store consulted, labelled — or the bare block when there was one.

    The single-store case is left byte-identical on purpose: it is the path every
    published measurement was taken on, and tagging it would change the prompt for
    80 questions in order to describe a feature none of them uses.

    Blocks are concatenated rather than rank-fused. RRF is the right tool for
    merging two rankings *of the same items*; here the stores hold disjoint
    content, and one of them returns a SQL result table rather than a ranked list
    at all, so fusing would be inventing an ordering between things that were
    never in competition.
    """
    if len(found) == 1:
        route, results = found[0]
        return _format_context(route, results)
    return "\n\n".join(f"[{ORIGIN[route]}]\n{_format_context(route, results)}" for route, results in found)


# Where a question goes when the store the router chose has nothing for it.
# The vector store is the only one that can hold arbitrary text, which is why
# it is also the ingestor's fallback.
FALLBACK_ROUTE = "vectorial"


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
    async def create(cls, clients: LlamaClients, settings: Settings | None = None) -> Orchestrator:
        settings = settings or load_config()
        rags = await build_rags(clients, settings)
        # A disabled cache is simply no cache: one source of truth, instead of an
        # `enabled` flag checked again at every call site.
        cache = SemanticCache(settings) if settings.cache.enabled else None
        router = await Router.create(clients, settings, rags)
        return cls(clients, router, cache, rags, settings)

    async def _retrieve(
        self, route: str, question: str, top_k: int, fallback_from: str | None = None
    ) -> list[Any]:
        with _tracer.start_as_current_span(f"pipeline.rag.{route}") as span:
            results = await self._rags[route].query(question, top_k=top_k)
            span.set_attribute("rag.results", len(results))
            if fallback_from:
                span.set_attribute("rag.fallback_from", fallback_from)
            return results

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
                cached = self._cache.lookup(vector, question) if self._cache is not None else None
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
                span.set_attributes(
                    {
                        "router.route": decision.route,
                        "router.routes": list(decision.routes),
                        "router.fan_out": decision.fan_out,
                        "router.decision_stage": decision.decision_stage,
                        "router.score_top1": decision.score_top1,
                        "router.score_top2": decision.score_top2,
                        "router.margin": decision.margin,
                        # Flattened one key per route: OTel attributes take primitives,
                        # never a dict.
                        **{f"router.score.{r}": s for r, s in decision.scores.items()},
                    }
                )
            # Emitted before retrieval so the panel can draw the decision while the
            # RAG is still working — the wait becomes the product.
            yield {"type": "decision", "decision": decision}

            top_k = self._settings.orchestrator.top_k
            # In parallel, because a compound message asks two independent stores
            # and they do not wait on each other. For the usual single-route
            # message this is one coroutine and behaves exactly as it always did.
            asks = decision.asks or ((question, decision.route),)
            gathered = await asyncio.gather(
                *(self._retrieve(route, asked, top_k) for asked, route in asks)
            )
            found = [
                (route, results)
                for (_, route), results in zip(asks, gathered, strict=True)
                if results
            ]

            # A store that finds nothing hands the question over instead of giving
            # up. The same fallback the ingestor uses, for the same reason: the
            # router judges what a question means, and can be wrong about which
            # store holds the answer. Measured on the held-out set, a misrouted
            # question was answered with "nao encontrei nada" while the passage sat
            # in the vector store. Deterministic and bounded -- one extra lookup,
            # only when every consulted store returned nothing, and never a loop back.
            if not found and FALLBACK_ROUTE not in decision.routes:
                fallback = await self._retrieve(
                    FALLBACK_ROUTE, question, top_k, fallback_from=decision.route
                )
                if fallback:
                    found = [(FALLBACK_ROUTE, fallback)]

            sources = [item for _, results in found for item in results]
            yield {"type": "sources", "sources": sources}

            if not found:
                yield {"type": "token", "text": "Não encontrei nada nas bases sobre isso."}
                yield {"type": "done"}
                return

            prompt = self._settings.orchestrator.answer_prompt.format(
                context=_join_context(found),
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
                self._cache.store(vector, question, answer)
            yield {"type": "done"}
