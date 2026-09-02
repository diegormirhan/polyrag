"""Drop files into data_drop/ and watch each one land in the right base.

This is the first line of the Definition of Done. Run directly (not pytest): it
needs the llama-servers and Qdrant up.

Destructive on purpose: ingested files are MOVED into data/processed/, which is
exactly how the watcher knows they are done. Run it twice and the second pass
finds nothing — that is the expected behaviour, not a failure.
"""

import asyncio
from collections import Counter

from app.core.llama_client import LlamaClients
from app.ingest.watcher import Watcher
from app.pipeline.ingestor import Ingestor


async def main() -> None:
    pending = await Watcher().poll_once()
    if not pending:
        print("data_drop/ vazio (ou tudo ja processado) — nada a fazer")
        return

    print(f"{len(pending)} arquivo(s): {[f.path.name for f in pending]}\n")
    ingestor = await Ingestor.create(LlamaClients())

    totals = Counter()
    for ingest_file in pending:
        report = await ingestor.ingest_file(ingest_file)
        totals.update(report.routes)
        print(f"{report.path.name}  ({ingest_file.kind.value})")
        print(f"  {len(report.routes)} chunk(s) -> {report.routes}\n")

    print(f"===== destino final: {dict(totals)} =====")


if __name__ == "__main__":
    asyncio.run(main())
