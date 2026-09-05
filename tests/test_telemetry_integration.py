"""Runs a question through the pipeline and prints the span tree it produced.

This is the terminal preview of the Day 5 observability panel: the same payloads
the WebSocket will stream, rendered as text. Run directly (not pytest) with the
llama-servers and Qdrant up, against bases that already have data ingested.
"""

import asyncio

from app.core.config import load_config
from app.core.llama_client import LlamaClients
from app.core.telemetry import LiveSpanExporter, setup_telemetry
from app.pipeline.orchestrator import Orchestrator

QUESTIONS = [
    "Uma compra de oitenta mil reais do Fornecedor Vega precisa de qual aprovacao?",
    "Se o Banco de Dados Orion entrar em manutencao, qual norma de dados e impactada?",
]

# Attributes worth putting on the timeline row itself; everything else is noise there.
HIGHLIGHTS = (
    "cache.hit",
    "router.route",
    "router.decision_stage",
    "router.margin",
    "rag.results",
    "embed.count",
    "llm.completion_tokens",
)


def _render(spans: list[dict], parent: str | None = None, depth: int = 0) -> None:
    children = sorted(
        (s for s in spans if s["parent_span_id"] == parent),
        key=lambda s: s["start_ms"],
    )
    for span in children:
        shown = " ".join(
            f"{key.split('.')[-1]}={span['attributes'][key]}"
            for key in HIGHLIGHTS
            if key in span["attributes"]
        )
        indent = "  " * depth
        print(f"  {indent}{span['name']:<{34 - 2 * depth}} {span['duration_ms']:8.1f}ms  {shown}")
        _render(spans, span["span_id"], depth + 1)


async def main() -> None:
    settings = load_config()
    exporter: LiveSpanExporter = setup_telemetry(settings)
    orchestrator = await Orchestrator.create(LlamaClients(), settings)

    for question in QUESTIONS:
        exporter.recent.clear()
        result = await orchestrator.answer(question)

        print(f"\npergunta: {question}")
        spans = list(exporter.recent)
        _render(spans)
        print(f"  -> {result.answer}")

    print(f"\n===== {len(exporter.recent)} spans no ultimo trace =====")


if __name__ == "__main__":
    asyncio.run(main())
