"""Measures the latency numbers the README claims, with enough repetitions to report a spread.

    uv run python scripts/benchmark.py                 # 8 runs per question
    uv run python scripts/benchmark.py --repeat 20
    uv run python scripts/benchmark.py --warmup 3

Needs the servers and the backend up, and the demo corpus ingested:

    uv run python scripts/start_servers.py
    uv run uvicorn app.main:app --app-dir backend --port 8000
    uv run python scripts/load_demo.py --reset

It drives the HTTP and WebSocket API rather than the orchestrator directly, so
what it reports is what a client actually waits for — the wire and the
serialisation included. Per-stage figures come from the OpenTelemetry spans of
each request, matched by trace id, so the breakdown is the pipeline's own
account of itself rather than a second set of timers written for the benchmark.

A single number cannot describe a latency: the first call after a model loads
pays for a cold KV cache, and the tail is what a user actually notices. Every
row is therefore p50 and p95 over n runs, after discarding warm-up runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "benchmark.json"
BASE = "http://127.0.0.1:8000"

# One question per route, so the report separates a slow store from a slow
# pipeline. Written against the demo/ corpus.
QUESTIONS = [
    ("relational", "Qual foi a receita total da regiao Sudeste?"),
    ("vectorial", "O que motivou a criacao do Sistema Atlas?"),
    ("graph", "Os pedidos processados pelo Sistema Atlas seguem qual politica de aprovacao?"),
]

CACHE_QUESTION = "Qual foi a receita total da regiao Sudeste?"
# Not a repeat of the same string: an exact repeat would also hit any lookup keyed
# on the text. This one has to be recognised by cosine alone, which is the claim.
CACHE_PARAPHRASE = "Qual e a receita somada do Sudeste?"

# Spans that contain the whole pipeline rather than one step of it. Listing them
# as stages would show a row equal to the total next to its own components.
STAGE_TOTALS = {"pipeline.chat"}


@dataclass
class Samples:
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(value)

    @property
    def p50(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    @property
    def p95(self) -> float:
        if not self.values:
            return 0.0
        # Nearest-rank rather than interpolation: with 8-20 samples an interpolated
        # p95 invents a value between two measurements that never happened.
        ordered = sorted(self.values)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def _rows(title: str, samples: dict[str, Samples]) -> None:
    if not samples:
        return
    width = max(len(name) for name in samples)
    print(f"\n{title}")
    print(f"{'':<{width}}{'n':>5}{'p50 ms':>10}{'p95 ms':>10}")
    print("-" * (width + 25))
    for name, sample in samples.items():
        print(f"{name:<{width}}{len(sample.values):>5}{sample.p50:>10.1f}{sample.p95:>10.1f}")


async def _spans_for(client: httpx.AsyncClient, trace_id: str) -> list[dict]:
    traces = (await client.get(f"{BASE}/api/v1/telemetry/traces")).json()
    return [span for span in traces if span["trace_id"] == trace_id]


async def _time_to_first_token(question: str) -> tuple[float, float]:
    """Returns (ms to the routing decision, ms to the first token) over the socket.

    Both are what the UI waits for: the decision paints the route panel, the first
    token starts the answer. Measured on the wire because that is where the claim
    lives — an in-process timer would omit the serialisation and the socket.
    """
    started = time.perf_counter()
    decision_ms = 0.0
    async with websockets.connect("ws://127.0.0.1:8000/api/v1/chat/stream") as socket:
        await socket.send(json.dumps({"message": question}))
        while True:
            event = json.loads(await socket.recv())
            if event["type"] == "decision" and not decision_ms:
                decision_ms = (time.perf_counter() - started) * 1000
            if event["type"] == "token":
                return decision_ms, (time.perf_counter() - started) * 1000
            if event["type"] == "done":
                # A cache hit answers without streaming tokens; it is measured
                # separately and should not silently become a TTFT sample.
                return decision_ms, 0.0


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=8, help="measured runs per question")
    parser.add_argument("--warmup", type=int, default=2, help="runs discarded before measuring")
    args = parser.parse_args()

    end_to_end: dict[str, Samples] = {}
    stages: dict[str, Samples] = {}
    wire: dict[str, Samples] = {}
    model_share = Samples()

    async with httpx.AsyncClient(timeout=180.0) as client:
        if (await client.get(f"{BASE}/api/v1/health")).status_code != 200:
            raise SystemExit("backend nao responde em :8000")

        total = len(QUESTIONS) * (args.warmup + args.repeat)
        done = 0
        for route, question in QUESTIONS:
            for run in range(args.warmup + args.repeat):
                # Cleared every run: a hit would skip routing and retrieval, and the
                # sample would measure the cache instead of the pipeline.
                await client.delete(f"{BASE}/api/v1/cache")

                started = time.perf_counter()
                response = (await client.post(f"{BASE}/api/v1/chat", json={"message": question})).json()
                elapsed = (time.perf_counter() - started) * 1000

                done += 1
                measuring = run >= args.warmup
                tag = "" if measuring else "  (warmup)"
                print(f"  {done:>3}/{total} {route:<11} {elapsed:>8.0f} ms{tag}")
                if not measuring:
                    continue

                end_to_end.setdefault(route, Samples()).add(elapsed)

                spans = await _spans_for(client, response["trace_id"])
                for span in spans:
                    # pipeline.chat and the ASGI request span both wrap everything,
                    # so they would show up as a stage containing all the others.
                    if span["name"] in STAGE_TOTALS:
                        continue
                    if span["name"].startswith(("pipeline.", "rag.", "router.")):
                        stages.setdefault(span["name"], Samples()).add(span["duration_ms"])
                # llama.* is every call that leaves for a model server; the rest is
                # our own code. Nested spans are excluded above, so this cannot
                # double-count a call inside its own stage.
                llama_ms = sum(s["duration_ms"] for s in spans if s["name"].startswith("llama."))
                if elapsed:
                    model_share.add(100 * llama_ms / elapsed)

        print("\n  medindo cache e streaming...")
        for label, follow_up in (
            ("cache hit (same question)", CACHE_QUESTION),
            ("cache hit (paraphrase)", CACHE_PARAPHRASE),
        ):
            misses = 0
            for _ in range(args.repeat):
                await client.delete(f"{BASE}/api/v1/cache")
                await client.post(f"{BASE}/api/v1/chat", json={"message": CACHE_QUESTION})

                started = time.perf_counter()
                hit = (await client.post(f"{BASE}/api/v1/chat", json={"message": follow_up})).json()
                elapsed = (time.perf_counter() - started) * 1000
                # A miss is not a slow hit. Averaging the two would hide the only
                # thing worth knowing about a cache: whether it recognised the
                # question at all.
                if hit["cache_hit"]:
                    wire.setdefault(label, Samples()).add(elapsed)
                else:
                    misses += 1
            if misses:
                print(f"  {label}: {misses}/{args.repeat} nao bateram no threshold")

        for _ in range(args.repeat):
            await client.delete(f"{BASE}/api/v1/cache")
            decision_ms, ttft_ms = await _time_to_first_token(QUESTIONS[2][1])
            wire.setdefault("route decision on the wire", Samples()).add(decision_ms)
            if ttft_ms:
                wire.setdefault("time to first token", Samples()).add(ttft_ms)

    _rows("end to end, POST /api/v1/chat", end_to_end)
    _rows("streaming and cache, over the socket", wire)
    _rows("per stage, from the spans of each request", dict(sorted(stages.items())))
    if model_share.values:
        print(f"\nshare of the request spent in model calls: p50 {model_share.p50:.0f}%")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "runs_per_question": args.repeat,
                "warmup_discarded": args.warmup,
                **{
                    title: {
                        name: {"n": len(s.values), "p50": s.p50, "p95": s.p95} for name, s in group.items()
                    }
                    for title, group in (
                        ("end_to_end", end_to_end),
                        ("wire", wire),
                        ("stages", stages),
                    )
                },
                "model_share_pct_p50": model_share.p50,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ndetalhe em {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
