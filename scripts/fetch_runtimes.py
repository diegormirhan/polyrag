"""Downloads the binaries and models the backend needs.

    uv run python scripts/fetch_runtimes.py

Roughly 4.9 GB, almost all of it models. Anything already present is skipped, so
re-running this only fills in what is missing — including after an interrupted
download, since each file is checked on its own.

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

HF = "https://huggingface.co/{repo}/resolve/main/{file}"
HF_TREE = "https://huggingface.co/api/models/{repo}/tree/main/{folder}"

# The llama.cpp Vulkan build, mirrored as loose files rather than taken from the
# upstream release zip: it pins one known-good build instead of tracking whatever
# the latest tag happens to contain.
LLAMA_REPO = "diegomirhan/voice-assistant-binaries"
LLAMA_FOLDER = "llama-server"

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


def _static(settings: Settings) -> list[Download]:
    llama, qdrant = settings.llama, settings.qdrant

    def model(repo: str, path: Path) -> Download:
        return Download(path.name, HF.format(repo=repo, file=path.name), ROOT / path)

    return [
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


def _llama_binaries(client: httpx.Client, settings: Settings) -> list[Download]:
    """The llama-server executable and every DLL beside it.

    The file list is read from the mirror rather than pinned here: the set of
    DLLs a llama.cpp build ships changes between versions, and a list copied into
    this script would quietly go stale and produce a binary that cannot start.
    """
    response = client.get(HF_TREE.format(repo=LLAMA_REPO, folder=LLAMA_FOLDER))
    response.raise_for_status()
    destination = (ROOT / settings.llama.bin_path).parent
    return [
        Download(
            f"{LLAMA_FOLDER}/{Path(entry['path']).name}",
            HF.format(repo=LLAMA_REPO, file=entry["path"]),
            destination / Path(entry["path"]).name,
        )
        for entry in response.json()
        if entry["type"] == "file"
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

    with httpx.Client(follow_redirects=True, timeout=None) as client:
        items = _static(settings)
        # Only ask the mirror what it holds when the binary is actually missing,
        # so a complete install needs no network at all.
        if not (ROOT / settings.llama.bin_path).exists():
            items = _llama_binaries(client, settings) + items

        for item in items:
            if item.done:
                print(f"  {item.name:<34} ja existe")

        pending = [item for item in items if not item.done]
        if pending:
            print()
            for item in pending:
                _fetch(client, item)

    missing = [item.name for item in items if not item.done]
    if missing:
        print(f"\nnao chegou ao destino esperado: {missing}")
        raise SystemExit(1)
    print("\ntudo no lugar")


if __name__ == "__main__":
    main()
