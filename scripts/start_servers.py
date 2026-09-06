"""Starts everything the backend needs: the llama-servers and Qdrant.

    uv run python scripts/start_servers.py           # start what is down
    uv run python scripts/start_servers.py --stop    # stop what this started

Ports, model paths and flags come from config.yaml through app.core.servers, the
same module the API uses — so the command line and the frontend cannot disagree
about what a service is. Servers run windowless; their output goes to
data/run/<name>.log, and the frontend's Servers tab can start and stop them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core import servers  # noqa: E402
from app.core.config import load_config  # noqa: E402

STARTUP_TIMEOUT_S = 300.0


def _wait_until_up(service: servers.Service) -> bool:
    started = time.perf_counter()
    while time.perf_counter() - started < STARTUP_TIMEOUT_S:
        if servers.is_up(service.probe):
            print(f"  {service.name:<12} pronto em {time.perf_counter() - started:.0f}s")
            return True
        time.sleep(1.0)
    print(f"  {service.name:<12} NAO respondeu em {STARTUP_TIMEOUT_S:.0f}s — veja {service.log_file}")
    return False


def _start_all(all_services: list[servers.Service]) -> None:
    pending = []
    for service in all_services:
        if servers.is_up(service.probe):
            print(f"  {service.name:<12} ja esta de pe")
            continue
        servers.start(service)
        print(f"  {service.name:<12} subindo... (log: {service.log_file.relative_to(ROOT)})")
        pending.append(service)

    if pending:
        print()
        # A list, not a generator: all() short-circuits, and a generator would
        # stop waiting for the remaining services after the first failure.
        if not all([_wait_until_up(service) for service in pending]):  # noqa: C419
            raise SystemExit(1)
    print("\ntudo respondendo")


def _stop_all(all_services: list[servers.Service]) -> None:
    for service in all_services:
        print(f"  {service.name:<12} {servers.stop(service)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", action="store_true", help="stop the servers instead of starting them")
    args = parser.parse_args()

    all_services = servers.services(load_config())
    _stop_all(all_services) if args.stop else _start_all(all_services)


if __name__ == "__main__":
    main()
