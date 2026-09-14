"""Runs the golden set and reports what the pipeline actually achieves.

    uv run python scripts/evaluate.py

Needs the servers up and the demo corpus ingested:

    uv run python scripts/start_servers.py
    uv run python scripts/load_demo.py --reset

Two metrics answer two different questions, and both are needed to read a
failure. `recall@k` asks whether the right passage even reached the model;
`answer_facts` asks whether the right answer reached the user. Low recall with a
wrong answer means retrieval failed. High recall with a wrong answer means the
model had what it needed and still got it wrong. Reporting only one of them
turns every failure into a guess about which half broke.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import load_demo  # noqa: E402  (same directory; re-uses its reset and staging)

from app.core.config import load_config  # noqa: E402
from app.core.llama_client import LlamaClients  # noqa: E402
from app.pipeline.ingestor import Ingestor  # noqa: E402
from app.pipeline.orchestrator import Orchestrator  # noqa: E402

EVAL_DIR = ROOT / "eval"
DEFAULT_SET = EVAL_DIR / "golden_set.yaml"
RETRIEVE_K = 10
CUTOFFS = (1, 5, 10)


# A number written as a word and the same number in digits are the same answer.
# The corpus spells them out ("cerca de quatrocentas pessoas", "retencao minima de
# sessenta meses") and the model usually replies in digits, so without this the
# metric scores orthography: "A empresa opera em 5 regioes" was counted wrong
# against a gold fact of "cinco". Only the values these corpora actually use --
# a general parser for Portuguese numerals would be more code than the thing it
# checks, and every entry here is one that appears in a source document.
_NUMBER_WORDS = {
    "um": "1",
    "uma": "1",
    "dois": "2",
    "duas": "2",
    "tres": "3",
    "quatro": "4",
    "cinco": "5",
    "seis": "6",
    "sete": "7",
    "oito": "8",
    "nove": "9",
    "dez": "10",
    "doze": "12",
    "vinte e quatro": "24",
    "quarenta": "40",
    "cinquenta": "50",
    "sessenta": "60",
    "cem": "100",
    "quatrocentas": "400",
    "quatrocentos": "400",
}
# Longest first, so "vinte e quatro" is replaced before "quatro" can eat part of it.
# The word boundaries are not decoration: without them "um" matches inside "algum"
# and "seis" inside "seiscentos", which would corrupt every comparison silently.
_NUMBER_PATTERN = re.compile(r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b")


def normalize(text: str) -> str:
    """Strips what is formatting rather than content, on both sides of a comparison.

    Case, accents and thousands separators, so the metric does not measure whether
    the model wrote "4.988.300" or "4988300"; and number words, so it does not
    measure whether it wrote "cinco" or "5".
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    flattened = stripped.lower().replace(".", "").replace(",", "")
    return _NUMBER_PATTERN.sub(lambda m: _NUMBER_WORDS[m.group(1)], flattened)


@dataclass
class Outcome:
    question: str
    expected_route: str
    actual_route: str | None
    route_ok: bool
    recall: dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float | None = None
    retrieved: int | None = None
    facts_ok: bool | None = None
    missing_facts: list[str] = field(default_factory=list)
    # Kept because a missing fact cannot be diagnosed without it: the answer
    # is the only place that says whether the model refused, hedged, or was
    # simply wrong.
    answer: str = ""
    seconds: float = 0.0


def _score_retrieval(chunks: list[str], markers: list[str]) -> tuple[dict[int, float], float]:
    """recall@k per cutoff, plus the reciprocal rank of the first useful chunk.

    MRR rather than precision@k. With one or two relevant chunks per question,
    precision@10 is capped at 10-20% by arithmetic alone, so reporting it reads as
    a failure of something that was never possible. What matters here is how far
    down the list the first useful passage sits.
    """
    normalized = [normalize(chunk) for chunk in chunks]
    wanted = [normalize(marker) for marker in markers]

    recall = {}
    for k in CUTOFFS:
        top = normalized[:k]
        found = sum(1 for marker in wanted if any(marker in chunk for chunk in top))
        recall[k] = found / len(wanted)

    reciprocal_rank = 0.0
    for position, chunk in enumerate(normalized[:RETRIEVE_K], start=1):
        if any(marker in chunk for marker in wanted):
            reciprocal_rank = 1 / position
            break
    return recall, reciprocal_rank


async def _evaluate(orchestrator: Orchestrator, case: dict[str, Any]) -> Outcome:
    question = case["question"]
    # `routes` (plural) is how a compound question says that several stores must
    # all be consulted. For the single-question sets the two spellings are the
    # same statement, so their numbers do not move.
    expected = case.get("routes") or [case["route"]]
    started = time.perf_counter()

    # Cleared per question so a paraphrase of an earlier one cannot be served from
    # cache: a cache hit skips routing and retrieval, and would silently remove
    # the question from every metric except the clock.
    orchestrator.clear_cache()
    result = await orchestrator.answer(question)
    route = result.decision.route if result.decision else None
    consulted = set(result.decision.routes) if result.decision else set()

    outcome = Outcome(
        question=question,
        expected_route="+".join(expected),
        actual_route="+".join(sorted(consulted)) or None,
        # Every expected store had to be consulted. With one expected store and
        # one consulted this is `route == expected`, unchanged.
        route_ok=set(expected) <= consulted,
        answer=result.answer,
        seconds=time.perf_counter() - started,
    )

    if facts := case.get("facts"):
        answer = normalize(result.answer)
        outcome.missing_facts = [f for f in facts if normalize(f) not in answer]
        outcome.facts_ok = not outcome.missing_facts

    # A second retrieval, deeper than the one that produced the answer. The
    # orchestrator serves config's top_k; recall@10 needs ten. Measuring the
    # answer through the product's own path and the ranking through a deeper
    # query is worth one extra call per question.
    # Only for single-store questions. A compound message retrieves each half from
    # its own store with its own text, so there is no one ranked list to score --
    # `facts` is what says whether both halves were answered.
    if (markers := case.get("relevant")) and len(expected) == 1:
        store = orchestrator._rags[route] if route else None
        hits = await store.query(question, top_k=RETRIEVE_K) if store else []
        # Only chunk-shaped results carry text. A question misrouted to the
        # relational store comes back as SQL rows, and the right passages were
        # therefore not retrieved — recall 0 is the honest end-to-end number, and
        # it is the routing failure this evaluation exists to surface rather than
        # a crash to work around.
        chunks = [hit["text"] for hit in hits if isinstance(hit, dict) and "text" in hit]
        outcome.retrieved = len(chunks)
        outcome.recall, outcome.reciprocal_rank = _score_retrieval(chunks, markers)

    return outcome


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _report(outcomes: list[Outcome]) -> dict[str, Any]:
    by_route: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        by_route.setdefault(outcome.expected_route, []).append(outcome)

    def summarise(group: list[Outcome]) -> dict[str, Any]:
        retrieved = [o for o in group if o.recall]
        judged = [o for o in group if o.facts_ok is not None]
        return {
            "questions": len(group),
            "router_accuracy": _mean([float(o.route_ok) for o in group]),
            # None, not zero: the relational route returns SQL rows rather than
            # chunks, so recall@k does not apply there. Reporting 0% would read as
            # a failure of something that was never attempted.
            **{
                f"recall@{k}": _mean([o.recall[k] for o in retrieved]) if retrieved else None for k in CUTOFFS
            },
            "mrr": _mean([o.reciprocal_rank or 0.0 for o in retrieved]) if retrieved else None,
            # The number that explains the recall column: entity-seeded graph search
            # returns NOTHING when named-entity extraction finds no node to start
            # from. Those are not badly ranked results, they are absent ones, and no
            # amount of reranking would touch them.
            "empty_retrieval": _mean([float(o.retrieved == 0) for o in retrieved]) if retrieved else None,
            "answer_facts": _mean([float(o.facts_ok) for o in judged]) if judged else None,
            "seconds_avg": _mean([o.seconds for o in group]),
        }

    summary = {"overall": summarise(outcomes)}
    for route, group in sorted(by_route.items()):
        summary[route] = summarise(group)

    header = (
        f"{'':<14}{'n':>4}{'router':>9}{'r@1':>8}{'r@5':>8}{'r@10':>8}"
        f"{'mrr':>8}{'vazio':>8}{'facts':>8}{'s':>7}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for name, row in summary.items():

        def pct(value: float | None) -> str:
            return "—" if value is None else f"{value:.0%}"

        cells = "".join(f"{pct(row[f'recall@{k}']):>8}" for k in CUTOFFS)
        print(
            f"{name:<14}{row['questions']:>4}{row['router_accuracy']:>9.0%}"
            f"{cells}{pct(row['mrr']):>8}{pct(row['empty_retrieval']):>8}"
            f"{pct(row['answer_facts']):>8}{row['seconds_avg']:>7.1f}"
        )

    misrouted = [o for o in outcomes if not o.route_ok]
    wrong = [o for o in outcomes if o.facts_ok is False]
    if misrouted:
        print(f"\nrota errada ({len(misrouted)}):")
        for o in misrouted:
            print(f"  {o.actual_route or 'cache'} != {o.expected_route}  | {o.question}")
    if wrong:
        print(f"\nfato ausente na resposta ({len(wrong)}):")
        for o in wrong:
            detail = ""
            if o.recall:
                detail = " SEM RESULTADO" if o.retrieved == 0 else f" r@10={o.recall[10]:.0%}"
            print(f"  faltou {o.missing_facts}{detail}  | {o.question}")

    return summary


async def _rebuild_corpus(clients: LlamaClients) -> None:
    """Empties the stores and re-ingests demo/, in this process.

    Measuring against whatever the stores happen to hold is how a number outlives
    the code that produced it. It happened here: the router's anchors were
    rewritten, the corpus was never re-ingested, and two published figures
    described a chunk distribution the code no longer produces.

    Refuses to run while the backend is up. Its watcher polls the same folder, and
    two ingesters racing over data_drop/ has already corrupted a measurement once
    in this project -- both wrote graph.json and the loser's chunks vanished.
    """
    settings = load_config()
    try:
        health = f"http://{settings.server.host}:{settings.server.port}/api/v1/health"
        reachable = httpx.get(health, timeout=2.0)
    except httpx.HTTPError:
        reachable = None
    if reachable is not None:
        raise SystemExit(
            "o backend esta de pe e o watcher dele competiria por data_drop/. "
            "Pare o backend antes de rodar com --reset."
        )

    load_demo.reset(settings)
    print()
    load_demo.stage(settings)

    ingestor = await Ingestor.create(clients)
    print()
    while reports := await ingestor.ingest_pending():
        for report in reports:
            counts = dict(Counter(report.routes))
            print(f"  {report.path.name:<28} {len(report.routes):>2} chunks -> {counts}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="wipe the stores and re-ingest demo/ before measuring, so the numbers "
        "describe a corpus this code actually produces",
    )
    parser.add_argument(
        "--set",
        default=str(DEFAULT_SET),
        help="question file to run (default: the golden set; eval/holdout_set.yaml is the "
        "validation set that no threshold was chosen against)",
    )
    args = parser.parse_args()

    questions = Path(args.set)
    results_path = EVAL_DIR / f"{questions.stem.replace('_set', '')}_results.json"
    cases = yaml.safe_load(questions.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    clients = LlamaClients()
    if args.reset:
        await _rebuild_corpus(clients)
    # Built after the ingestion on purpose: the stores are read at construction, so
    # an orchestrator created first would hold the graph as it was before.
    orchestrator = await Orchestrator.create(clients)
    outcomes = []
    try:
        for index, case in enumerate(cases, start=1):
            outcome = await _evaluate(orchestrator, case)
            mark = "ok" if outcome.route_ok and outcome.facts_ok is not False else "!!"
            print(f"  {index:>3}/{len(cases)} {mark} {outcome.question[:66]}")
            outcomes.append(outcome)
    finally:
        await clients.aclose()

    summary = _report(outcomes)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {"summary": summary, "questions": [vars(o) for o in outcomes]},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\ndetalhe por pergunta em {results_path.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
