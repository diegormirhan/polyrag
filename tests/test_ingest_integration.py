import asyncio

from app.core.llama_client import LlamaClients
from app.ingest import chunking, loaders, ocr
from app.ingest.watcher import FileKind, Watcher


async def main() -> None:
    watcher = Watcher()
    files = await watcher.poll_once()
    print(f"Encontrados {len(files)} arquivos: {[f.path.name for f in files]}")

    clients = LlamaClients()

    for f in files:
        print(f"\n=== {f.path.name} ({f.kind.value}) ===")

        if f.kind == FileKind.VISION:
            text = await ocr.extract_text(f.path, clients)
            print(f"Texto extraído ({len(text)} chars):\n{text}")
            pieces = await chunking.chunks(text, clients)
            print(f"-> {len(pieces)} chunks")
            for i, p in enumerate(pieces):
                print(f"--- chunk {i} ({len(p)} chars) ---\n{p}")

        elif f.kind == FileKind.TABLE:
            df = loaders.load(f)
            print(df)

        else:
            text = loaders.load(f)
            print(f"Texto ({len(text)} chars):\n{text}")
            pieces = await chunking.chunks(text, clients)
            print(f"-> {len(pieces)} chunks")
            for i, p in enumerate(pieces):
                print(f"--- chunk {i} ({len(p)} chars) ---\n{p}")


if __name__ == "__main__":
    asyncio.run(main())
