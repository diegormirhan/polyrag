"""Downloads the binaries and models the backend needs.

    uv run python scripts/fetch_runtimes.py

Roughly 5 GB, mostly models. Anything already present is skipped, so re-running
this only fills in what is missing.

Sources are pinned here; destinations come from config.yaml. That split is
deliberate — a URL is setup metadata, but a path is something the running app
also reads, and hardcoding it twice is how a downloader ends up putting a file
somewhere the app never looks.
"""

from __future__ import annotations

import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings, load_config  # noqa: E402

LLAMA_BUILD = "b10287"
QDRANT_VERSION = "v1.19.0"


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    target: Path
    # A zip is unpacked into `target`'s directory; anything else is saved as-is.
    unzip: bool = False

    @property
    def done(self) -> bool:
        return self.target.exists()


def _downloads(settings: Settings) -> list[Download]:
    llama, qdrant = settings.llama, settings.qdrant
    hf = "https://huggingface.co/{repo}/resolve/main/{file}"

    def model(repo: str, path: Path) -> Download:
        return Download(path.name, hf.format(repo=repo, file=path.name), ROOT / path)

    return [
        Download(
            f"llama.cpp {LLAMA_BUILD} (Vulkan)",
            f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}"
            f"/llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip",
            ROOT / llama.bin_path,
            unzip=True,
        ),
        Download(
            f"Qdrant {QDRANT_VERSION}",
            f"https://github.com/qdrant/qdrant/releases/download/{QDRANT_VERSION}"
            f"/qdrant-x86_64-pc-windows-msvc.zip",
            ROOT / qdrant.bin_path,
            unzip=True,
        ),
        model("unsloth/Qwen3.5-4B-GGUF", llama.llm.model_path),
        model("ggml-org/GLM-OCR-GGUF", llama.ocr.model_path),
        model("ggml-org/GLM-OCR-GGUF", llama.ocr.mmproj_path),
        model("gpustack/bge-m3-GGUF", llama.embeddings.model_path),
    ]


def _fetch(client: httpx.Client, item: Download) -> None:
    destination = item.target if not item.unzip else item.target.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with client.stream("GET", item.url) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as out:
            for block in response.iter_bytes(1024 * 1024):
                out.write(block)
                if total:
                    share = response.num_bytes_downloaded / total
                    print(f"\r  {item.name:<34} {share:6.1%}", end="", flush=True)
    print(f"\r  {item.name:<34} pronto ")

    if item.unzip:
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(item.target.parent)
        destination.unlink()


def main() -> None:
    settings = load_config()
    items = _downloads(settings)
    pending = [item for item in items if not item.done]

    for item in items:
        if item.done:
            print(f"  {item.name:<34} ja existe")

    if not pending:
        print("\ntudo no lugar")
        return

    print()
    with httpx.Client(follow_redirects=True, timeout=None) as client:
        for item in pending:
            _fetch(client, item)

    missing = [item.name for item in items if not item.done]
    if missing:
        print(f"\nnao chegou ao destino esperado: {missing}")
        raise SystemExit(1)
    print("\ntudo no lugar")


if __name__ == "__main__":
    main()
