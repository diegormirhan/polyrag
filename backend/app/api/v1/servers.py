from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.deps import get_settings
from app.core import servers
from app.core.config import Settings
from app.schemas import ServerAction, ServerState

router = APIRouter(prefix="/servers")


async def _state(client: httpx.AsyncClient, service: servers.Service) -> ServerState:
    base = service.probe.removesuffix("/health").removesuffix("/healthz")
    sleeping = None
    try:
        response = await client.get(service.probe, timeout=2.0)
        up = response.status_code == 200
        if up and service.name != "qdrant":
            # /props is the only endpoint that reports sleep state, and a sleeping
            # server answers it without waking (verified).
            props = await client.get(f"{base}/props", timeout=2.0)
            sleeping = props.json().get("is_sleeping")
    except (httpx.HTTPError, ValueError):
        up = False
    return ServerState(
        name=service.name,
        label=service.label,
        url=base,
        up=up,
        sleeping=sleeping,
        managed=servers.read_pid(service) is not None,
    )


@router.get("", response_model=list[ServerState])
async def list_servers(settings: Settings = Depends(get_settings)) -> list[ServerState]:
    async with httpx.AsyncClient() as client:
        # Probed together: sequentially, four services that are all down would cost
        # four timeouts before the page could render.
        return list(await asyncio.gather(*(_state(client, s) for s in servers.services(settings))))


def _service(settings: Settings, name: str) -> servers.Service:
    service = servers.find(settings, name)
    if service is None:
        raise HTTPException(status_code=404, detail=f"no server named {name!r}")
    return service


@router.post("/{name}/start", response_model=ServerAction)
async def start_server(name: str, settings: Settings = Depends(get_settings)) -> ServerAction:
    service = _service(settings, name)
    if servers.is_up(service.probe):
        return ServerAction(name=name, result="already running")
    servers.start(service)
    # Returns as soon as the process exists. Loading a model into VRAM takes tens
    # of seconds, and holding the request open for it would stall the UI; the
    # frontend polls the list and watches the state change.
    return ServerAction(name=name, result="started")


@router.post("/{name}/stop", response_model=ServerAction)
async def stop_server(name: str, settings: Settings = Depends(get_settings)) -> ServerAction:
    return ServerAction(name=name, result=servers.stop(_service(settings, name)))
