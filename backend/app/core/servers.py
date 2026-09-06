"""Starting and stopping the llama-servers and Qdrant.

One definition of what a service is, shared by the CLI launcher and the API, so
the two can never disagree about a port, a flag or a model path.

Servers run windowless and write to `data/run/<name>.log`. A visible console per
server was fine while there were two of them and everything was started by hand;
with the frontend offering start/stop it would put four windows on the user's
desktop that must not be closed. The log file is the console, kept.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.core.config import Settings

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "data" / "run"

# CREATE_NO_WINDOW exists only on Windows; elsewhere the flag is simply absent.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class Service:
    name: str  # the id the API and the pid file use
    label: str  # what the model is, for the UI
    probe: str  # URL that answers 200 once the service is ready
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)

    @property
    def pid_file(self) -> Path:
        return RUN_DIR / f"{self.name}.pid"

    @property
    def log_file(self) -> Path:
        return RUN_DIR / f"{self.name}.log"


def _llama(settings: Settings, name: str, label: str, cfg, extra: list[str]) -> Service:
    return Service(
        name=name,
        label=label,
        probe=f"http://{cfg.host}:{cfg.port}/health",
        command=[
            str(ROOT / settings.llama.bin_path),
            "-m",
            str(cfg.model_path),
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
            "-c",
            str(cfg.ctx_size),
            "-ngl",
            str(cfg.n_gpu_layers),
            *extra,
        ],
    )


def _sleep_flag(cfg) -> list[str]:
    # A sleeping server hands its VRAM back while idle and reloads it on the next
    # request (measured: GLM-OCR returns 2.08GB). Only worth it for models used in
    # bursts — OCR during image ingestion.
    if cfg.sleep_idle_seconds <= 0:
        return []
    return ["--sleep-idle-seconds", str(cfg.sleep_idle_seconds)]


def services(settings: Settings) -> list[Service]:
    llama, qdrant = settings.llama, settings.qdrant
    return [
        # --parallel 1: `-c` is the TOTAL context split across slots (measured — the
        # flag changes VRAM by only 0.07GB), so with the default 4 slots a ctx_size
        # of 8192 would leave 2048 tokens per request, under what the answer path needs.
        _llama(settings, "llm", "Qwen3.5-4B", llama.llm, ["--parallel", "1"]),
        _llama(
            settings,
            "ocr",
            "GLM-OCR",
            llama.ocr,
            ["--mmproj", str(llama.ocr.mmproj_path), *_sleep_flag(llama.ocr)],
        ),
        _llama(settings, "embeddings", "BGE-M3", llama.embeddings, ["--embedding"]),
        # The judge shares the llm's port, so it is the llm — nothing extra to start.
        # A judge configured on its own port would get its own server again.
        *(
            [_llama(settings, "judge", "judge", llama.judge, _sleep_flag(llama.judge))]
            if llama.judge.port != llama.llm.port
            else []
        ),
        Service(
            name="qdrant",
            label="Qdrant",
            probe=f"http://{qdrant.host}:{qdrant.port}/healthz",
            command=[str(ROOT / qdrant.bin_path)],
            # Qdrant takes its storage location from the environment, not a flag.
            env={"QDRANT__STORAGE__STORAGE_PATH": str(ROOT / qdrant.storage_path)},
        ),
    ]


def find(settings: Settings, name: str) -> Service | None:
    return next((s for s in services(settings) if s.name == name), None)


def is_up(probe: str, timeout: float = 1.0) -> bool:
    try:
        return httpx.get(probe, timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def read_pid(service: Service) -> int | None:
    try:
        return int(service.pid_file.read_text())
    except (OSError, ValueError):
        return None


def start(service: Service) -> None:
    """Launch the service windowless, appending its output to its log file."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # Appended, not truncated: after a restart the log of the run that failed is
    # the only thing left to read.
    log = service.log_file.open("a", encoding="utf-8", errors="replace")
    try:
        process = subprocess.Popen(
            service.command,
            cwd=ROOT,  # paths in config.yaml are relative to the project root
            env={**os.environ, **service.env},
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
    finally:
        # The child holds its own handle to the file; ours is not needed and would
        # otherwise leak one descriptor per start.
        log.close()
    service.pid_file.write_text(str(process.pid))


def stop(service: Service) -> str:
    """Terminate the service. Returns what happened, for the caller to report."""
    pid = read_pid(service)
    if pid is None:
        # Reachable but unmanaged: started outside the app, so there is no pid to
        # trust. Guessing one from the port would risk killing the wrong process.
        return "unmanaged" if is_up(service.probe) else "already stopped"

    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        # The pid died on its own, or the number was recycled and now belongs to a
        # process we may not touch. Either way the file is stale.
        service.pid_file.unlink(missing_ok=True)
        return "already stopped"

    service.pid_file.unlink(missing_ok=True)
    return "stopped"
