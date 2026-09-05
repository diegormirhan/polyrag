"""End-to-end check of the search path: Cache -> Router -> RAG -> LLM.

Run directly (not through pytest): it needs the four llama-servers and Qdrant up.

Each case carries the route it must take and the exact strings its answer must
contain. The assertions are deliberately crude — figures and proper nouns only,
never paraphrasable prose — because those are the parts an LLM cannot legitimately
reword. Real evaluation (recall@k, LLM-as-judge on a golden set) is Day 11.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.core.llama_client import LlamaClients
from app.pipeline.orchestrator import ChatResult, Orchestrator
from app.rags.graph import GraphRAG
from app.rags.relational import RelationalRAG
from app.rags.vectorial import VectorialRAG


@dataclass(frozen=True)
class Case:
    question: str
    route: str
    must_contain: list[str]
    why: str


SALES = pd.DataFrame(
    {
        "produto": ["Notebook", "Mouse", "Teclado", "Monitor"],
        "regiao": ["Sudeste", "Sul", "Nordeste", "Sudeste"],
        "vendas": [120, 80, 45, 200],
    }
)

CHUNKS = [
    "Buracos negros sao regioes do espaco onde a gravidade e tao intensa que nem a luz escapa.",
    "A Fabrica Beta fornece o Motor X para a Montadora Alfa.",
    "A Montadora Alfa usa o Motor X na linha de producao do Sedan Z.",
]

CASES = [
    Case(
        question="Qual o total de vendas da regiao Sudeste?",
        route="relational",
        must_contain=["320"],
        why="120 + 200 — SQL aggregates it, the LLM must not do the arithmetic itself",
    ),
    Case(
        question="o que acontece perto de um buraco negro?",
        route="vectorial",
        must_contain=["luz"],
        why="the single fact stated in the ingested chunk",
    ),
    Case(
        question="se a Fabrica Beta atrasar, o que e afetado?",
        route="graph",
        must_contain=["Motor X", "Sedan Z"],
        why="2 hops — no chunk names Fabrica Beta and Sedan Z together, "
        "so vector search alone cannot reach it",
    ),
]

PARAPHRASE = "o que ocorre proximo a um buraco negro?"


async def seed(clients: LlamaClients) -> None:
    await RelationalRAG(clients).ingest((SALES, "vendas"))
    vectorial = await VectorialRAG.create(clients)
    graph = GraphRAG.load(clients)
    for chunk in CHUNKS:
        await vectorial.ingest(chunk)
        await graph.ingest(chunk)
    print(f"bases populadas: {len(SALES)} linhas, {len(CHUNKS)} chunks")


def _format_source(item: Any) -> str:
    # Vectorial and graph return {text, score}; relational returns raw SQL rows.
    if isinstance(item, dict) and "text" in item:
        return f"score={item['score']:.4f} | {item['text'][:75]}"
    return str(item)


def _report(result: ChatResult, elapsed: float, failures: list[str]) -> None:
    if result.decision is None:
        print(f"  CACHE HIT  ({elapsed * 1000:.0f}ms)")
    else:
        d = result.decision
        ranked = sorted(d.scores.items(), key=lambda kv: kv[1], reverse=True)
        print(f"  [{'  '.join(f'{route}={score:.3f}' for route, score in ranked)}]")
        print(f"  rota={d.route} estagio={d.decision_stage} margem={d.margin:.3f} ({elapsed:.2f}s)")

    print(f"  fontes ({len(result.sources)}):")
    for source in result.sources:
        print(f"    {_format_source(source)}")
    print(f"  resposta: {result.answer}")
    print("  OK" if not failures else f"  FALHOU: {' ; '.join(failures)}")


async def run_case(orchestrator: Orchestrator, case: Case) -> tuple[bool, str]:
    started = time.perf_counter()
    result = await orchestrator.answer(case.question)
    elapsed = time.perf_counter() - started

    route = result.decision.route if result.decision else "cache"
    failures = []
    if route != case.route:
        failures.append(f"rota={route}, esperada={case.route}")
    missing = [text for text in case.must_contain if text.lower() not in result.answer.lower()]
    if missing:
        failures.append(f"resposta nao menciona {missing}")

    print(f"\npergunta: {case.question}")
    print(f"  esperado: rota={case.route} + {case.must_contain}  ({case.why})")
    _report(result, elapsed, failures)
    return not failures, result.answer


async def run_cache_case(orchestrator: Orchestrator, question: str, reference: str, label: str) -> bool:
    started = time.perf_counter()
    result = await orchestrator.answer(question)
    elapsed = time.perf_counter() - started

    failures = []
    if result.decision is not None:
        failures.append("nao foi cache hit — o pipeline inteiro rodou de novo")
    elif result.answer != reference:
        failures.append("resposta difere da que foi guardada")

    print(f"\npergunta ({label}): {question}")
    _report(result, elapsed, failures)
    return not failures


async def main() -> None:
    clients = LlamaClients()
    await seed(clients)
    orchestrator = await Orchestrator.create(clients)

    results = []
    answers = {}
    for case in CASES:
        passed, answer = await run_case(orchestrator, case)
        results.append(passed)
        answers[case.question] = answer

    print("\n===== CAG: mesma pergunta e uma parafrase =====")
    reference = answers[CASES[1].question]
    results.append(await run_cache_case(orchestrator, CASES[1].question, reference, "repetida"))
    results.append(await run_cache_case(orchestrator, PARAPHRASE, reference, "parafrase"))

    print(f"\n===== {sum(results)}/{len(results)} verificacoes OK =====")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
