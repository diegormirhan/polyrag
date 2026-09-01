from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, chat, embed
from app.core.vectors import normalize
from app.pipeline.cache import SemanticCache
from app.pipeline.router import RouteDecision, Router
from app.rags.base import RAGBase
from app.rags.graph import GraphRAG
from app.rags.relational import RelationalRAG
from app.rags.vectorial import VectorialRAG


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
        rags: dict[str, RAGBase] = {
            "relational": RelationalRAG(clients, settings),
            "vectorial": await VectorialRAG.create(clients, settings),
            "graph": GraphRAG.load(clients, settings),
        }
        # A disabled cache is simply no cache: one source of truth, instead of an
        # `enabled` flag checked again at every call site.
        cache = SemanticCache(settings) if settings.cache.enabled else None
        router = await Router.create(clients, settings)
        return cls(clients, router, cache, rags, settings)

    async def answer(self, question: str) -> ChatResult:
        vector = normalize((await embed(self._clients.embeddings, [question]))[0])

        if self._cache is not None:
            cached = self._cache.lookup(vector)
            if cached is not None:
                return ChatResult(cached, cache_hit=True, decision=None, sources=[])

        # On a miss the router embeds the question a second time. Left as is: this
        # path already costs seconds of SQL/LLM work, so one extra embedding call is
        # noise, and keeping route() self-contained is worth more than the microopt.
        decision = await self._router.route(question)
        results = await self._rags[decision.route].query(
            question, top_k=self._settings.orchestrator.top_k
        )
        if not results:
            return ChatResult("Não encontrei nada nas bases sobre isso.", False, decision, [])

        prompt = self._settings.orchestrator.answer_prompt.format(
            context=_format_context(decision.route, results),
            question=question,
        )
        answer = await chat(
            self._clients.llm,
            [{"role": "user", "content": prompt}],
            temperature=self._settings.llama.llm.temperature,
        )

        if self._cache is not None:
            self._cache.store(vector, answer)
        return ChatResult(answer, False, decision, results)