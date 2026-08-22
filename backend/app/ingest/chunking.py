from __future__ import annotations
from app.core.config import Settings, load_config

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def chunks(text: str, settings: Settings | None = None) -> list[str]:
    settings = settings or load_config()
    return chunk_text(text, settings.ingest.chunk_size, settings.ingest.chunk_overlap)