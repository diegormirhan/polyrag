from __future__ import annotations
from dataclasses import dataclass

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, chat, embed
from app.core.vectors import dot, normalize
from app.pipeline.tabularity import tabularity_score

@dataclass(frozen=True)
class RouteDecision:
    route: str
    score_top1: float
    score_top2: float
    margin: float
    decision_stage: str # "heuristic" | "embedding" | "llm_judge"

def _best_route_scores(vector: list[float], anchors: dict[str, list[list[float]]]) -> dict[str, float]:
    # For each route, the score is the cosine against its BEST-matching anchor
    # (a 1-nearest-neighbor classifier per route), not an average of all anchors.
    return {route: max(dot(vector, a) for a in route_anchors) for route, route_anchors in anchors.items()}

async def _ask_judge(clients: LlamaClients, text: str, route_a: str, route_b: str, settings: Settings) -> str:
    prompt = settings.router.llm_judge_prompt.format(
        route_a=route_a,
        desc_a=settings.router.routes[route_a].description,
        route_b=route_b,
        desc_b=settings.router.routes[route_b].description,
        text=text,
    )
    response = await chat(clients.judge, [{"role": "user", "content": prompt}], temperature=0)
    choice = response.strip()
    return choice if choice in (route_a, route_b) else route_a


async def _llm_judge(clients: LlamaClients, text: str, route_a: str, route_b: str, settings: Settings) -> str:
    # Position bias check: LLM judges (especially small/lightweight ones) tend to
    # favor whichever option is listed FIRST in the prompt, regardless of merit.
    # Ask twice with the order swapped; only trust the verdict if both agree.
    first = await _ask_judge(clients, text, route_a, route_b, settings)
    second = await _ask_judge(clients, text, route_b, route_a, settings)

    if first == second:
        return first

    # Disagreement between orderings signals position bias — don't trust either
    # potentially-biased answer, fall back to the deterministic Stage 2 winner.
    return route_a

class Router:
    """Precomputes route anchors once; route() reuses them for every chunk/question"""

    def __init__(self, anchors: dict[str, list[list[float]]], clients: LlamaClients, settings: Settings) -> None:
        self._anchors = anchors
        self._clients = clients
        self._settings = settings

    @classmethod
    async def create(cls, clients: LlamaClients, settings: Settings | None = None) -> "Router":
        settings = settings or load_config()
        anchors: dict[str, list[list[float]]] = {}
        for route_name, route_cfg in settings.router.routes.items():
            raw_vectors = await embed(clients.embeddings, route_cfg.utterances)
            anchors[route_name] = [normalize(v) for v in raw_vectors]
        return cls(anchors, clients, settings)

    async def route(self, text: str) -> RouteDecision:
        cfg = self._settings.router

        # Stage 1: heuristic — cheap, no embedding call at all.
        heuristic_score = tabularity_score(text)
        if heuristic_score >= cfg.tau_heuristic:
            return RouteDecision(
                route="relational",
                score_top1=heuristic_score,
                score_top2=0.0,  # not meaningful at this stage — no competing route was scored
                margin=heuristic_score,
                decision_stage="heuristic",
            )

        # Stage 2: cosine + margin against precomputed anchors.
        raw_vector = (await embed(self._clients.embeddings, [text]))[0]
        vector = normalize(raw_vector)
        scores = _best_route_scores(vector, self._anchors)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top1_route, top1_score = ranked[0]
        top2_route, top2_score = ranked[1]
        margin = top1_score - top2_score

        if top1_score >= cfg.tau_high and margin >= cfg.delta_margin:
            return RouteDecision(top1_route, top1_score, top2_score, margin, "embedding")

        if top1_score < cfg.tau_low:
            return RouteDecision("vectorial", top1_score, top2_score, margin, "embedding")

        # Stage 3: gray zone — ask the LLM to pick between the top 2 candidates.
        if not cfg.llm_judge_enabled:
            return RouteDecision(top1_route, top1_score, top2_score, margin, "embedding")

        chosen = await _llm_judge(self._clients, text, top1_route, top2_route, self._settings)
        return RouteDecision(chosen, top1_score, top2_score, margin, "llm_judge")