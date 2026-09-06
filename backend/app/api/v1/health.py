from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Depends

from app.api.v1.deps import get_settings
from app.core.config import Settings
from app.schemas import HealthResponse, ServiceHealth

router = APIRouter()


async def _probe_llama(client: httpx.AsyncClient, name: str, host: str, port: int) -> ServiceHealth:
    url = f"http://{host}:{port}"
    try:
        # /props rather than /health: a server that freed its VRAM to sleep still
        # answers both without waking (verified), but only /props reports whether it
        # is asleep — which the panel should show rather than hide.
        response = await client.get(f"{url}/props", timeout=2.0)
        return ServiceHealth(name=name, url=url, up=True, sleeping=response.json().get("is_sleeping"))
    except (httpx.HTTPError, ValueError):
        return ServiceHealth(name=name, url=url, up=False)


async def _probe_qdrant(client: httpx.AsyncClient, host: str, port: int) -> ServiceHealth:
    url = f"http://{host}:{port}"
    try:
        up = (await client.get(f"{url}/healthz", timeout=2.0)).status_code == 200
    except httpx.HTTPError:
        up = False
    return ServiceHealth(name="qdrant", url=url, up=up)


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    llama = settings.llama
    async with httpx.AsyncClient() as client:
        # Probed together: sequentially, five services that are all down would take
        # five timeouts to report.
        services = await asyncio.gather(
            *(
                _probe_llama(client, name, cfg.host, cfg.port)
                for name, cfg in (
                    ("llm", llama.llm),
                    ("ocr", llama.ocr),
                    ("embeddings", llama.embeddings),
                )
            ),
            _probe_qdrant(client, settings.qdrant.host, settings.qdrant.port),
        )
    return HealthResponse(ok=all(s.up for s in services), services=list(services))
