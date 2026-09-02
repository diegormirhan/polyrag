from __future__ import annotations

from openai import AsyncOpenAI
from opentelemetry import trace

from app.core.config import Settings, load_config

# Every call to a model goes through this module, and that is where the pipeline's
# time actually goes — a single OpenIE call once hid 98% of an ingestion's latency.
# Spanning here means no LLM call can ever be invisible again.
_tracer = trace.get_tracer("polyrag.llama")


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
    with _tracer.start_as_current_span("llama.chat") as span:
        response = await client.chat.completions.create(
            model="local",
            messages=messages,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        content = response.choices[0].message.content or ""
        if response.usage is not None:
            span.set_attributes({
                "llm.prompt_tokens": response.usage.prompt_tokens,
                "llm.completion_tokens": response.usage.completion_tokens,
            })
        return content


async def embed(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    with _tracer.start_as_current_span("llama.embed") as span:
        span.set_attribute("embed.count", len(texts))
        response = await client.embeddings.create(model="local", input=texts)
        return [item.embedding for item in response.data]
