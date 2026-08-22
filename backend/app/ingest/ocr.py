from __future__ import annotations

import base64
from pathlib import Path
from app.core.config import load_config
from app.core.llama_client import LlamaClients, chat

_MIME_OVERRIDES = {"jpg": "jpeg"}

async def extract_text(path: Path, clients: LlamaClients) -> str:
    settings = load_config()
    image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    extension = path.suffix.lstrip(".").lower()
    mime = f"image/{_MIME_OVERRIDES.get(extension, extension)}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": settings.ingest.ocr_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
            ]
        }
    ]
    return await chat(clients.ocr, messages)
