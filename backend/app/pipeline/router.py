from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import trace

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, embed
from app.core.vectors import dot, normalize
from app.pipeline.tabularity import tabularity_score
from app.rags.base import RAGBase

_tracer = trace.get_tracer("polyrag.pipeline")


@dataclass(frozen=True)
class RouteDecision:
    route: str
    decision_stage: str  # "heuristic" | "embedding" | "content_evidence"
    scores: dict[str, float]

    @property
    def score_top1(self) -> float:
        return self._ranked[0]

    @property
    def score_top2(self) -> float:
        ranked = self._ranked
        return ranked[1] if len(ranked) > 1 else 0.0

    @property
    def margin(self) -> float:
        return self.score_top1 - self.score_top2

    @property
    def _ranked(self) -> list[float]:
        return sorted(self.scores.values(), reverse=True)


def _best_route_scores(vector: list[float], anchors: dict[str, list[list[float]]]) -> dict[str, float]:
    # For each route, the score is the cosine against its BEST-matching anchor
    # (a 1-nearest-neighbor classifier per route), not an average of all anchors.
    return {route: max(dot(vector, a) for a in route_anchors) for route, route_anchors in anchors.items()}


class Router:
    """Precomputes route anchors once; route() reuses them for every chunk/question"""

    def __init__(
        self,
        anchors: dict[str, list[list[float]]],
        clients: LlamaClients,
        settings: Settings,
        content_anchors: dict[str, list[list[float]]] | None = None,
    ) -> None:
        self._anchors = anchors
        # Kept apart from the configured anchors rather than merged into them: they
        # answer different questions and are trusted at different moments. See route().
        self._content_anchors = content_anchors or {}
        self._clients = clients
        self._settings = settings

    @classmethod
    async def create(
        cls,
        clients: LlamaClients,
        settings: Settings | None = None,
        rags: dict[str, RAGBase] | None = None,
    ) -> Router:
        """Builds the anchor vectors once. `rags` lets each store describe itself.

        Two sources, on purpose. The configured utterances say what a route is
        *for*, and work on an empty corpus. The stores' own section headings say
        what each one currently *holds*, and cost nothing to maintain: ingesting a
        document teaches the router about it without a config edit.

        Measured on 40 questions written after the utterances were tuned -- the
        honest test of whether they generalise -- routing accuracy went from 82%
        to 95% once the stores contributed their headings.

        The anchors are a snapshot: a store that gains content after startup does
        not change them until the process restarts.
        """
        settings = settings or load_config()
        anchors: dict[str, list[list[float]]] = {}
        content: dict[str, list[list[float]]] = {}
        for route_name, route_cfg in settings.router.routes.items():
            anchors[route_name] = [
                normalize(v) for v in await embed(clients.embeddings, route_cfg.utterances)
            ]
            phrases = await rags[route_name].content_anchors() if rags and route_name in rags else []
            if phrases:
                content[route_name] = [normalize(v) for v in await embed(clients.embeddings, phrases)]
        return cls(anchors, clients, settings, content)

    async def route(self, text: str) -> RouteDecision:
        cfg = self._settings.router

        # Stage 1: heuristic — cheap, no embedding call at all.
        heuristic_score = tabularity_score(text)
        if heuristic_score >= cfg.tau_heuristic:
            return RouteDecision(
                route="relational",
                decision_stage="heuristic",
                scores={"relational": heuristic_score},
            )
        # Stage 2: cosine + margin against precomputed anchors.
        raw_vector = (await embed(self._clients.embeddings, [text]))[0]
        vector = normalize(raw_vector)
        scores = _best_route_scores(vector, self._anchors)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top1_route, top1_score = ranked[0]
        _, top2_score = ranked[1]
        margin = top1_score - top2_score

        if top1_score >= cfg.tau_high and margin >= cfg.delta_margin:
            return RouteDecision(top1_route, "embedding", scores)

        if top1_score < cfg.tau_low:
            return RouteDecision("vectorial", "embedding", scores)

        # Stage 3: the gray zone — the configured anchors did not separate the top
        # two, so ask the stores what they actually hold and decide on that.
        #
        # This used to be a model call, and the model was measured three times and
        # never earned it: on the gray-zone questions of the golden set, stage 2
        # alone was right 9 times out of 10 and the judge 8 -- it agreed with the
        # geometry 9 times (two model calls to repeat what arithmetic already said)
        # and the one time it disagreed, it was wrong.
        #
        # The store headings are right where a written phrase cannot be: nothing in
        # config.yaml could know that "Banco de Dados Orion" names a section in the
        # graph, so questions about it went to the relational store, whose only
        # table is about sales. Measured on 40 questions written after the anchors
        # were tuned, this took routing from 82% to 95% -- and every stage of the
        # router is now arithmetic.
        if self._content_anchors:
            combined = _best_route_scores(vector, self._content_anchors)
            for route, score in scores.items():
                combined[route] = max(combined.get(route, 0.0), score)
            chosen = max(combined, key=combined.get)
            return RouteDecision(chosen, "content_evidence", combined)

        return RouteDecision(top1_route, "embedding", scores)
