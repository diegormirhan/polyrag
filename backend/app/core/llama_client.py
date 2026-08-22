from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import Settings, load_config


def _client(host: str, port: int) -> AsyncOpenAI:
    # llama-server ignores the API key, but the OpenAI SDK requires a non-empty string
    return AsyncOpenAI(base_url=f"http://{host}:{port}/v1", api_key="not-needed")


class LlamaClients:
    """One AsyncOpenAI client per llama-server instance (llm, ocr, embeddings)."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or load_config()
        self.llm = _client(settings.llama.llm.host, settings.llama.llm.port)
        self.ocr = _client(settings.llama.ocr.host, settings.llama.ocr.port)
        self.embeddings = _client(settings.llama.embeddings.host, settings.llama.embeddings.port)


async def chat(client: AsyncOpenAI, messages: list[dict], temperature: float = 0.2) -> str:
    response = await client.chat.completions.create(
        model="local",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


async def embed(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    response = await client.embeddings.create(model="local", input=texts)
    return [item.embedding for item in response.data]
