from __future__ import annotations

import base64
from io import StringIO
from pathlib import Path
import re
import pandas as pd
from app.core.config import load_config
from app.core.llama_client import LlamaClients, chat

_MIME_OVERRIDES = {"jpg": "jpeg"}
_TABLE_PATTERN = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)

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
    # temperature=0: OCR is extraction, not generation — sampling only risks
    # hallucinating a digit/character instead of reading exactly what's on the image.
    raw_text = await chat(clients.ocr, messages, temperature=0)
    return _html_tables_to_markdown(raw_text)

def _html_tables_to_markdown(text: str) -> str:
    def replace(match: re.Match) -> str:
        table = pd.read_html(StringIO(match.group(0)))[0]
        return table.to_markdown(index=False)
    return _TABLE_PATTERN.sub(replace, text)