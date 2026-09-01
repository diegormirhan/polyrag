from __future__ import annotations
import re

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, embed
from app.core.vectors import dot, normalize

TABLE_BLOCK = re.compile(r"(?:^\|.*\|[ \t]*\n?)+", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

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

    for sentence, vector in zip(sentences[1:], vectors[1:]):
        current_len = sum(len(s) for s in groups[-1])
        # Compare the next sentence against the AVERAGE of every vector already
        # accumulated in the current group — not just the single previous sentence.
        # A one-sentence anchor is noisy; the running average is more stable and
        # represents "what this chunk has been about so far" more reliably.
        centroid = _centroid(group_vectors[-1])

        if current_len < min_chunk_chars or dot(centroid, vector) >= threshold:
            groups[-1].append(sentence)
            group_vectors[-1].append(vector)
        else:
            groups.append([sentence])
            group_vectors.append([vector])

    return [" ".join(group) for group in groups]

def _split_table_blocks(text: str) -> list[tuple[str, bool]]:
    """Splits text into (segment, is_table) pairs, preserving order."""
    segments: list[tuple[str, bool]] = []
    pos = 0
    for match in TABLE_BLOCK.finditer(text):
        if match.start() > pos:
            segments.append((text[pos:match.start()], False))
        segments.append((match.group(), True))
        pos = match.end()
    if pos < len(text):
        segments.append((text[pos:], False))
    return segments

async def chunks(text: str, clients: LlamaClients, settings: Settings | None = None) -> list[str]:
    settings = settings or load_config()
    pieces: list[str] = []
    for segment, is_table in _split_table_blocks(text):
        if is_table:
            pieces.append(segment)
        elif segment.strip():
            pieces.extend(
                await semantic_chunk_text(
                    segment, clients, settings.ingest.semantic_threshold, settings.ingest.min_chunk_chars
                )
            )
    return pieces

