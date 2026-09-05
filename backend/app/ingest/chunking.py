from __future__ import annotations

import re

from opentelemetry import trace

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, embed
from app.core.vectors import dot, normalize

_tracer = trace.get_tracer("polyrag.ingest")

TABLE_BLOCK = re.compile(r"(?:^\|.*\|[ \t]*\n?)+", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_HEADING = re.compile(r"^(#{1,6}) +(.+)$", re.MULTILINE)


def _centroid(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    return normalize(mean)


async def semantic_chunk_text(
    text: str, clients: LlamaClients, threshold: float, min_chunk_chars: int
) -> list[str]:
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 1:
        return sentences

    raw_vectors = await embed(clients.embeddings, sentences)
    vectors = [normalize(v) for v in raw_vectors]

    groups: list[list[str]] = [[sentences[0]]]
    group_vectors: list[list[list[float]]] = [[vectors[0]]]

    for sentence, vector in zip(sentences[1:], vectors[1:], strict=True):
        current_len = sum(len(s) for s in groups[-1])
        # Compare the next sentence against the AVERAGE of every vector already
        # accumulated in the current group — not just the single previous sentence.
        # A one-sentence anchor is noisy; the running average is more stable and
        # represents "what this chunk has been about so far" more reliably.
        centroid = _centroid(group_vectors[-1])

        # Same rule as _paragraphs, one level down: a fragment on either side joins
        # rather than becoming a chunk of its own. "(Questão Aberta)" was being split
        # off its question by the sentence splitter and left alone as a 16-char chunk.
        if (
            current_len < min_chunk_chars
            or len(sentence) < min_chunk_chars
            or dot(centroid, vector) >= threshold
        ):
            groups[-1].append(sentence)
            group_vectors[-1].append(vector)
        else:
            groups.append([sentence])
            group_vectors.append([vector])

    return [" ".join(group) for group in groups]


def _paragraphs(text: str, min_chars: int) -> list[str]:
    """Blank-line separated blocks, with runts merged into the next one.

    A blank line is the author declaring a topic boundary — a signal that is free
    and more reliable than cosine. Measured on a 4-topic document: the real
    boundaries scored 0.541/0.534/0.584 against the centroid while the lowest
    within-topic score was 0.500, so the ranges overlap and no single threshold
    separates them. The blank lines separate all four exactly.

    A block shorter than min_chars is a fragment, not a topic — a heading on its
    own line, most often — so it joins what follows instead of becoming a chunk.
    """
    blocks = [block.strip() for block in _PARAGRAPH_SPLIT.split(text) if block.strip()]
    merged: list[str] = []
    for block in blocks:
        # A heading belongs to what follows it, never to what precedes it. Letting
        # the fragment rule absorb one backwards tagged the introduction with the
        # heading of the section that started after it.
        opens_section = bool(_HEADING.fullmatch(block))
        # Otherwise either side being a fragment merges them: a short block that
        # FOLLOWS a long one is just as orphaned as one that precedes it, and
        # merging only forward left "(Questão Aberta)" alone as its own chunk.
        if merged and not opens_section and (len(merged[-1]) < min_chars or len(block) < min_chars):
            merged[-1] = f"{merged[-1]}\n{block}"
        else:
            merged.append(block)
    return merged


def _track_headings(block: str, trail: dict[int, str]) -> None:
    """Updates the heading trail in place from any headings inside `block`."""
    for match in _HEADING.finditer(block):
        level = len(match.group(1))
        # A new heading closes every deeper one: "## Section 3" ends whatever
        # "### Subsection" was still open under Section 2.
        for deeper in [key for key in trail if key > level]:
            del trail[deeper]
        trail[level] = match.group(2).strip()


def _breadcrumb(trail: dict[int, str]) -> str:
    return " › ".join(trail[level] for level in sorted(trail))  # noqa: RUF001


def _split_table_blocks(text: str) -> list[tuple[str, bool]]:
    """Splits text into (segment, is_table) pairs, preserving order."""
    segments: list[tuple[str, bool]] = []
    pos = 0
    for match in TABLE_BLOCK.finditer(text):
        if match.start() > pos:
            segments.append((text[pos : match.start()], False))
        segments.append((match.group(), True))
        pos = match.end()
    if pos < len(text):
        segments.append((text[pos:], False))
    return segments


async def chunks(text: str, clients: LlamaClients, settings: Settings | None = None) -> list[str]:
    settings = settings or load_config()
    with _tracer.start_as_current_span("pipeline.chunking") as span:
        span.set_attribute("chunking.chars", len(text))
        pieces = await _chunk(text, clients, settings)
        span.set_attribute("chunking.chunks", len(pieces))
        return pieces


async def _chunk(text: str, clients: LlamaClients, settings: Settings) -> list[str]:
    pieces: list[str] = []
    # Only the first chunk of a section used to contain its heading, so every chunk
    # after it was orphaned. A question from "Section 4" and one from "Section 2"
    # then reached the model looking identical, and it answered with both — the
    # model was not wrong, the context was. A chunk has to carry where it came from.
    trail: dict[int, str] = {}
    for segment, is_table in _split_table_blocks(text):
        if is_table:
            pieces.append(segment)
            continue
        # Structure first, meaning second: paragraphs are hard boundaries, and the
        # semantic split only runs *inside* one — for the long paragraph that drifts
        # across topics on its own.
        for paragraph in _paragraphs(segment, settings.ingest.min_chunk_chars):
            _track_headings(paragraph, trail)
            context = _breadcrumb(trail)
            for piece in await semantic_chunk_text(
                paragraph,
                clients,
                settings.ingest.semantic_threshold,
                settings.ingest.min_chunk_chars,
            ):
                pieces.append(piece if not context or context in piece else f"{context}\n{piece}")
    return pieces
