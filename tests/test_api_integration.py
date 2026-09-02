"""Exercises the /api/v1 surface against a running backend.

    uv run uvicorn --app-dir backend app.main:app --port 8000

Run directly (not pytest), with the llama-servers and Qdrant up and data already
ingested. Checks the pieces the frontend depends on: routing metadata on an
answer, the cache round trip, and spans arriving live on the WebSocket.
"""

import asyncio
import json

import httpx
import websockets

BASE = "http://127.0.0.1:8000/api/v1"
STREAM = "ws://127.0.0.1:8000/api/v1/telemetry/stream"

QUESTION = "Uma compra de oitenta mil reais do Fornecedor Vega precisa de qual aprovacao?"


async def with_live_spans(coro):
    """Runs `coro` while a WebSocket collects every span it emits."""
    async with websockets.connect(STREAM) as socket:
        task = asyncio.create_task(coro)
        spans = []
        while True:
            try:
                spans.append(json.loads(await asyncio.wait_for(socket.recv(), timeout=2.0)))
            except asyncio.TimeoutError:
                if task.done():
                    break
        return await task, spans


async def ask(client: httpx.AsyncClient) -> dict:
    response = await client.post(f"{BASE}/chat", json={"message": QUESTION}, timeout=120.0)
    response.raise_for_status()
    return response.json()


def report(label: str, body: dict) -> None:
    stage = f"{body['route']}/{body['decision_stage']}" if body["route"] else "-"
    print(f"  {label:<22} cache_hit={str(body['cache_hit']):<5} rota={stage:<20} trace={body['trace_id'][:12]}")


async def main() -> None:
    async with httpx.AsyncClient() as client:
        health = (await client.get(f"{BASE}/health")).json()
        print("\n== /health ==")
        for service in health["services"]:
            state = "dormindo" if service.get("sleeping") else ""
            print(f"  {service['name']:<12} up={service['up']}  {state}")

        print("\n== /chat ==")
        first, spans = await with_live_spans(ask(client))
        report("miss (pipeline)", first)
        second = await ask(client)
        report("repetida (cache)", second)

        print(f"\n== spans recebidos ao vivo pelo WebSocket: {len(spans)} ==")
        for span in sorted(spans, key=lambda s: s["start_ms"]):
            print(f"  {span['name']:<34} {span['duration_ms']:8.1f}ms")

        traces = (await client.get(f"{BASE}/telemetry/traces")).json()
        print(f"\n== /telemetry/traces: {len(traces)} spans no ring buffer ==")

        await client.delete(f"{BASE}/cache")
        third = await ask(client)
        print("\n== apos DELETE /cache ==")
        report("deve ser miss", third)

        checks = {
            "health ok": health["ok"],
            "1a chamada foi miss": not first["cache_hit"],
            "2a chamada foi hit": second["cache_hit"],
            "hit devolveu a mesma resposta": second["answer"] == first["answer"],
            "websocket entregou spans": len(spans) > 0,
            "ring buffer tem spans": len(traces) > 0,
            "cache limpo volta a ser miss": not third["cache_hit"],
        }
        print()
        for name, ok in checks.items():
            print(f"  {'OK  ' if ok else 'FALHOU'} {name}")
        print(f"\n===== {sum(checks.values())}/{len(checks)} verificacoes OK =====")
        if not all(checks.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
