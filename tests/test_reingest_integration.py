"""Correct a file, drop it back under the same name, and check the old text is gone.

Run directly (not pytest): it needs the llama-servers and Qdrant up.

    uv run python -u tests/test_reingest_integration.py

Proves the whole path that tests/test_watcher_reingest.py only proves the front of.
The watcher deciding a changed file is new is worth nothing if the previous
version's chunks stay in the stores: retrieval would then return whichever of the
two ranked higher, and a corrected document would be indistinguishable from a
corrupted one.

Cleans up after itself -- the scratch file's chunks and its manifest entry are
removed at the end -- so the demo corpus is exactly as it was.
"""

import asyncio
import json
from pathlib import Path

import _bootstrap  # noqa: F401  # puts backend/ on sys.path; must precede `app`

from app.core.config import load_config
from app.core.llama_client import LlamaClients
from app.ingest.watcher import Watcher
from app.pipeline.ingestor import Ingestor

NAME = "reingest_scratch.md"
FIRST = """# Nota de teste de reingestao

## Politica PZ-99

A Politica PZ-99 exige aprovacao do Conselho Fiscal para qualquer contrato.
A Politica PZ-99 foi publicada em janeiro.
"""
SECOND = """# Nota de teste de reingestao

## Politica PZ-99

A Politica PZ-99 exige aprovacao do Comite Tecnico para qualquer contrato.
A Politica PZ-99 foi revisada em agosto.
"""


async def _stored_texts(rags) -> list[str]:
    """Every chunk of text the two text-holding stores currently contain."""
    settings = load_config()
    points, _ = await rags["vectorial"]._client.scroll(
        collection_name=settings.qdrant.collection, limit=10_000, with_payload=True
    )
    graph = rags["graph"]._graph
    return [point.payload["text"] for point in points] + [
        data["text"] for _, data in graph.nodes(data=True) if data.get("kind") == "chunk"
    ]


async def main() -> None:
    settings = load_config()
    drop = Path(settings.paths.data_drop)
    manifest_path = Path(settings.paths.ingest_manifest)
    ingestor = await Ingestor.create(LlamaClients())

    def write(body: str) -> None:
        (drop / NAME).write_text(body, encoding="utf-8")

    async def ingest_once() -> list[str]:
        pending = [f for f in await Watcher(settings).poll_once() if f.path.name == NAME]
        if not pending:
            raise SystemExit(f"{NAME} nao foi oferecido pelo watcher")
        return (await ingestor.ingest_file(pending[0])).routes

    try:
        print("1. primeira ingestao")
        write(FIRST)
        print(f"   rotas: {await ingest_once()}")
        texts = await _stored_texts(ingestor._rags)
        assert any("Conselho Fiscal" in text for text in texts), "a primeira versao nao foi guardada"
        print("   'Conselho Fiscal' esta nas bases")

        print("2. mesmo arquivo, mesmos bytes")
        write(FIRST)
        offered = [f.path.name for f in await Watcher(settings).poll_once()]
        assert NAME not in offered, "os mesmos bytes foram oferecidos de novo"
        print("   o watcher ignorou (digest ja conhecido)")

        print("3. arquivo corrigido, MESMO nome")
        write(SECOND)
        print(f"   rotas: {await ingest_once()}")
        texts = await _stored_texts(ingestor._rags)
        assert any("Comite Tecnico" in text for text in texts), "a correcao nao foi guardada"
        assert not any("Conselho Fiscal" in text for text in texts), (
            "a versao antiga continua nas bases -- as duas responderiam a mesma pergunta"
        )
        print("   'Comite Tecnico' entrou e 'Conselho Fiscal' saiu")

        print("\n===== a reingestao substitui, nao duplica =====")
    finally:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = entries.pop(NAME, None)
        if entry:
            by_route: dict[str, list[str]] = {}
            for route, chunk_id in entry["stored"]:
                by_route.setdefault(route, []).append(chunk_id)
            for route, ids in by_route.items():
                await ingestor._rags[route].forget(ids)
            manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        (drop / NAME).unlink(missing_ok=True)
        (Path(settings.paths.processed) / NAME).unlink(missing_ok=True)
        print("limpeza: o corpus de demonstracao ficou como estava")


if __name__ == "__main__":
    asyncio.run(main())
