"""Starts one llama-server per model, with ports, paths and flags from config.yaml.

    uv run python scripts/start_llm_servers.py

Each server opens its own console window so its logs stay visible. A server whose
port already answers is left running, so re-running this only fills in what is down.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_config  # noqa: E402

STARTUP_TIMEOUT_S = 300.0


def _is_up(host: str, port: int) -> bool:
    try:
        return httpx.get(f"http://{host}:{port}/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def _sleep_flag(cfg) -> list[str]:
    # A sleeping server hands its VRAM back while idle and reloads it on the next
    # request (measured: GLM-OCR returns 2.08GB). Only worth it for models used in
    # bursts — OCR during image ingestion, the judge in the router's gray zone.
    if cfg.sleep_idle_seconds <= 0:
        return []
    return ["--sleep-idle-seconds", str(cfg.sleep_idle_seconds)]


def _command(bin_path: Path, cfg, extra: list[str]) -> list[str]:
    return [
        str(bin_path),
        "-m", str(cfg.model_path),
        "--host", cfg.host,
        "--port", str(cfg.port),
        "-c", str(cfg.ctx_size),
        "-ngl", str(cfg.n_gpu_layers),
        *extra,
    ]


def _wait_until_up(name: str, cfg) -> bool:
    started = time.perf_counter()
    while time.perf_counter() - started < STARTUP_TIMEOUT_S:
        if _is_up(cfg.host, cfg.port):
            print(f"  {name:<22} pronto em {time.perf_counter() - started:.0f}s")
            return True
        time.sleep(1.0)
    print(f"  {name:<22} NAO respondeu em {STARTUP_TIMEOUT_S:.0f}s — veja a janela do servidor")
    return False


def main() -> None:
    settings = load_config()
    llama = settings.llama
    bin_path = ROOT / llama.bin_path

    servers = {
        # --parallel 1: `-c` is the TOTAL context, split across slots (measured: 4 slots
        # by default). With one slot the whole ctx_size goes to the single request the
        # pipeline ever makes at a time, and the extra slots cost VRAM for nothing.
        "llm (Qwen3-8B)": (llama.llm, ["--parallel", "1"]),
        "ocr (GLM-OCR)": (llama.ocr, ["--mmproj", str(llama.ocr.mmproj_path), *_sleep_flag(llama.ocr)]),
        "embeddings (BGE-M3)": (llama.embeddings, ["--embedding"]),
        "judge (Prometheus 2)": (llama.judge, _sleep_flag(llama.judge)),
    }

    pending = []
    for name, (cfg, extra) in servers.items():
        if _is_up(cfg.host, cfg.port):
            print(f"  {name:<22} :{cfg.port}  ja esta de pe")
            continue
        subprocess.Popen(
            _command(bin_path, cfg, extra),
            cwd=ROOT,  # model paths in config.yaml are relative to the project root
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        print(f"  {name:<22} :{cfg.port}  subindo...")
        pending.append((name, cfg))

    if pending:
        print()
        if not all([_wait_until_up(name, cfg) for name, cfg in pending]):
            raise SystemExit(1)
    print("\ntodos os servidores respondendo")


if __name__ == "__main__":
    main()
