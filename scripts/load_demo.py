"""Puts the demo corpus into the hot folder, optionally on empty stores.

    uv run python scripts/load_demo.py            # copy demo/ into data_drop/
    uv run python scripts/load_demo.py --reset    # wipe the stores first

Nothing here is special-cased: the files land in data_drop/ and are ingested by
the same watcher that handles anything else dropped there. A demo that took a
private path would prove nothing about the pipeline.

--reset exists because a demo inherits whatever the last test run left behind,
and stale chunks are exactly what makes a demonstration confusing. It is opt-in
and destructive, so it says what it removed.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings, load_config  # noqa: E402

DEMO = ROOT / "demo"


def _reset(settings: Settings) -> None:
    paths = settings.paths
    removed: list[str] = []

    for path in (ROOT / paths.sqlite_db, ROOT / paths.graph_store, ROOT / paths.ingest_manifest):
        if path.exists():
            path.unlink()
            removed.append(path.name)

    processed = ROOT / paths.processed
    archived = [p for p in processed.glob("*") if p.is_file()] if processed.exists() else []
    for path in archived:
        path.unlink()
    if archived:
        removed.append(f"{len(archived)} arquivo(s) em {paths.processed}")

    qdrant = settings.qdrant
    url = f"http://{qdrant.host}:{qdrant.port}/collections/{qdrant.collection}"
    try:
        if httpx.delete(url, timeout=5.0).status_code == 200:
            removed.append(f"coleção {qdrant.collection}")
    except httpx.HTTPError:
        print(f"  Qdrant não respondeu em {qdrant.host}:{qdrant.port} — coleção intacta")

    print(f"  removido: {', '.join(removed) if removed else 'nada, já estava vazio'}")


def main() -> None:
    settings = load_config()

    if "--reset" in sys.argv:
        _reset(settings)
        print()

    data_drop = ROOT / settings.paths.data_drop
    data_drop.mkdir(parents=True, exist_ok=True)
    processed = {p.name for p in (ROOT / settings.paths.processed).glob("*")}

    for source in sorted(DEMO.glob("*")):
        # A README belongs to the reader, not to the corpus.
        if not source.is_file() or source.name.lower() == "readme.md":
            continue
        if source.name in processed:
            print(f"  {source.name:<28} já ingerido (use --reset para refazer)")
            continue
        shutil.copy2(source, data_drop / source.name)
        print(f"  {source.name:<28} copiado")

    print(
        "\nO watcher ingere em segundos com o backend de pé; sem ele, rode tests/test_ingest_integration.py."
    )


if __name__ == "__main__":
    main()
