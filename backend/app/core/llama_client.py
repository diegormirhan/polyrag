from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import Settings, load_config


def _client(host: str, port: int) -> AsyncOpenAI:
    # llama-server ignores the API key, but the OpenAI SDK requires a non-empty string
    return AsyncOpenAI(base_url=f"http://{host}:{port}/v1", api_key="not-needed")


class LlamaClients:
    """One AsyncOpenAI client per llama-server instance (llm, ocr, embeddings, judge)."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or load_config()
        self.llm = _client(settings.llama.llm.host, settings.llama.llm.port)
        self.ocr = _client(settings.llama.ocr.host, settings.llama.ocr.port)
        self.embeddings = _client(settings.llama.embeddings.host, settings.llama.embeddings.port)
        self.judge = _client(settings.llama.judge.host, settings.llama.judge.port)


async def chat(
    client: AsyncOpenAI,
    messages: list[dict],
    temperature: float = 0.2,
    enable_thinking: bool = False,
) -> str:
    # Qwen3 ships with thinking mode ON by default: it emits a full reasoning block
    # (billed as generated tokens) before every answer — measured 3.06s vs 0.62s for
    # a trivial prompt. Structured tasks (SQL, OpenIE, NER, routing) gain nothing
    # from it, so it is off unless a caller explicitly asks for it.
    response = await client.chat.completions.create(
        model="local",
        messages=messages,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    return response.choices[0].message.content or ""


async def embed(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    response = await client.embeddings.create(model="local", input=texts)
    return [item.embedding for item in response.data]
