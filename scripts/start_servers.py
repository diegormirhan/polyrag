"""Starts everything the backend needs: the llama-servers and Qdrant.

    uv run python scripts/start_servers.py

Ports, model paths and flags all come from config.yaml, so this script and the
backend can never disagree about them. Each server opens its own console window
so its logs stay visible — closing that window stops the server. A service whose
readiness probe already answers is left alone, so re-running only fills in what
is down.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings, load_config  # noqa: E402

STARTUP_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class Service:
    name: str
    probe: str  # URL that answers 200 once the service is ready
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)


def _is_up(probe: str) -> bool:
    try:
        return httpx.get(probe, timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def _sleep_flag(cfg) -> list[str]:
    # A sleeping server hands its VRAM back while idle and reloads it on the next
    # request (measured: GLM-OCR returns 2.08GB). Only worth it for models used in
    # bursts — OCR during image ingestion, the judge in the router's gray zone.
    if cfg.sleep_idle_seconds <= 0:
        return []
    return ["--sleep-idle-seconds", str(cfg.sleep_idle_seconds)]


def _llama(settings: Settings, name: str, cfg, extra: list[str]) -> Service:
    return Service(
        name=name,
        probe=f"http://{cfg.host}:{cfg.port}/health",
        command=[
            str(ROOT / settings.llama.bin_path),
            "-m", str(cfg.model_path),
            "--host", cfg.host,
            "--port", str(cfg.port),
            "-c", str(cfg.ctx_size),
            "-ngl", str(cfg.n_gpu_layers),
            *extra,
        ],
    )


def _services(settings: Settings) -> list[Service]:
    llama, qdrant = settings.llama, settings.qdrant
    return [
        # --parallel 1: `-c` is the TOTAL context split across slots (measured — the
        # flag changes VRAM by only 0.07GB), so with the default 4 slots a ctx_size
        # of 8192 would leave 2048 tokens per request, under what the answer path needs.
        _llama(settings, "llm (Qwen3.5-4B)", llama.llm, ["--parallel", "1"]),
        _llama(settings, "ocr (GLM-OCR)", llama.ocr,
               ["--mmproj", str(llama.ocr.mmproj_path), *_sleep_flag(llama.ocr)]),
        _llama(settings, "embeddings (BGE-M3)", llama.embeddings, ["--embedding"]),
        # The judge shares the llm's port, so it is the llm — nothing extra to start.
        # A judge configured on its own port would get its own server again.
        *(
            [_llama(settings, "judge", llama.judge, _sleep_flag(llama.judge))]
            if llama.judge.port != llama.llm.port
            else []
        ),
        Service(
            name="qdrant",
            probe=f"http://{qdrant.host}:{qdrant.port}/healthz",
            command=[str(ROOT / qdrant.bin_path)],
            # Qdrant takes its storage location from the environment, not a flag.
            env={"QDRANT__STORAGE__STORAGE_PATH": str(ROOT / qdrant.storage_path)},
        ),
    ]


def _wait_until_up(service: Service) -> bool:
    started = time.perf_counter()
    while time.perf_counter() - started < STARTUP_TIMEOUT_S:
        if _is_up(service.probe):
            print(f"  {service.name:<22} pronto em {time.perf_counter() - started:.0f}s")
            return True
        time.sleep(1.0)
    print(f"  {service.name:<22} NAO respondeu em {STARTUP_TIMEOUT_S:.0f}s — veja a janela dele")
    return False


def main() -> None:
    pending = []
    for service in _services(load_config()):
        if _is_up(service.probe):
            print(f"  {service.name:<22} ja esta de pe")
            continue
        subprocess.Popen(
            service.command,
            cwd=ROOT,  # paths in config.yaml are relative to the project root
            env={**os.environ, **service.env},
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        print(f"  {service.name:<22} subindo...")
        pending.append(service)

    if pending:
        print()
        if not all([_wait_until_up(service) for service in pending]):
            raise SystemExit(1)
    print("\ntudo respondendo")


if __name__ == "__main__":
    main()
