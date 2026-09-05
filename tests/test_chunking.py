"""Chunking is where most of this project's real bugs came from.

Every case here is a regression: a document that was actually ingested and split
wrongly, not a hypothetical. The functions are pure (or made pure by a fake
embedder), so none of this needs a GPU, a model server or a network.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.ingest import chunking
from app.ingest.chunking import (
    TRAIL_SEPARATOR,
    _breadcrumb,
    _paragraphs,
    _split_table_blocks,
    _track_headings,
    semantic_chunk_text,
)

MIN_CHARS = 80
LONG = "Uma frase suficientemente longa para passar do piso de oitenta caracteres sem ajuda."
LONGER = "Outra frase igualmente longa, que tambem passa do piso e pode virar seu proprio bloco."


# --------------------------------------------------------------------------- #
# _paragraphs — blank lines are the author declaring a topic boundary
# --------------------------------------------------------------------------- #


def test_splits_on_blank_lines():
    assert _paragraphs(f"{LONG}\n\n{LONGER}", MIN_CHARS) == [LONG, LONGER]


def test_a_single_block_survives_untouched():
    assert _paragraphs(LONG, MIN_CHARS) == [LONG]


def test_whitespace_only_input_yields_nothing():
    assert _paragraphs("\n\n   \n\n", MIN_CHARS) == []


def test_a_short_block_absorbs_the_next_one():
    """The original rule: a fragment cannot stand alone, so it takes what follows."""
    assert _paragraphs(f"Curto demais\n\n{LONG}", MIN_CHARS) == [f"Curto demais\n{LONG}"]


def test_a_short_block_joins_the_previous_one():
    """The rule that was missing: "(Questão Aberta)" stood alone as a 16-char chunk
    because merging only looked forward."""
    assert _paragraphs(f"{LONG}\n\n(Questao Aberta)", MIN_CHARS) == [f"{LONG}\n(Questao Aberta)"]


def test_a_heading_is_never_absorbed_backwards():
    """A heading belongs to what follows it. Absorbing it backwards tagged a
    document's introduction with the heading of the section that started after it."""
    blocks = _paragraphs(f"{LONG}\n\n## Secao 2\n\n{LONGER}", MIN_CHARS)
    assert blocks[0] == LONG
    assert blocks[1].startswith("## Secao 2")


def test_a_heading_still_takes_the_content_under_it():
    """It is short, so the ordinary fragment rule pulls the next block into it —
    which is what keeps a section's first chunk labelled."""
    assert _paragraphs(f"## Secao 2\n\n{LONG}", MIN_CHARS) == [f"## Secao 2\n{LONG}"]


# --------------------------------------------------------------------------- #
# _track_headings / _breadcrumb — a chunk has to state where it came from
# --------------------------------------------------------------------------- #


def test_trail_is_empty_without_headings():
    trail: dict[int, str] = {}
    _track_headings(LONG, trail)
    assert trail == {}
    assert _breadcrumb(trail) == ""


def test_trail_accumulates_across_levels():
    trail: dict[int, str] = {}
    _track_headings("# Questionario", trail)
    _track_headings("## Secao 2: Acesso", trail)
    assert _breadcrumb(trail) == TRAIL_SEPARATOR.join(["Questionario", "Secao 2: Acesso"])


def test_a_new_heading_replaces_its_own_level():
    trail: dict[int, str] = {}
    _track_headings("# Doc", trail)
    _track_headings("## Secao 2", trail)
    _track_headings("## Secao 4", trail)
    assert _breadcrumb(trail) == TRAIL_SEPARATOR.join(["Doc", "Secao 4"])


def test_a_shallower_heading_closes_the_deeper_ones():
    """Otherwise a subsection stays attached to sections it does not belong to."""
    trail: dict[int, str] = {}
    _track_headings("# Doc", trail)
    _track_headings("## Secao 2", trail)
    _track_headings("### Detalhe", trail)
    _track_headings("## Secao 3", trail)
    assert _breadcrumb(trail) == TRAIL_SEPARATOR.join(["Doc", "Secao 3"])


def test_several_headings_in_one_block_leave_the_last():
    trail: dict[int, str] = {}
    _track_headings("## Primeira\n\ntexto\n\n## Segunda", trail)
    assert _breadcrumb(trail) == "Segunda"


def test_a_hash_that_is_not_a_heading_is_ignored():
    trail: dict[int, str] = {}
    _track_headings("custa #5 reais e #tag nao e titulo", trail)
    assert trail == {}


# --------------------------------------------------------------------------- #
# _split_table_blocks — a table must never be cut in half
# --------------------------------------------------------------------------- #

TABLE = "| item | qtd |\n|------|-----|\n| cafe | 2 |"


def test_text_without_a_table_is_one_segment():
    assert _split_table_blocks(LONG) == [(LONG, False)]


def test_a_table_is_kept_whole_and_flagged():
    segments = _split_table_blocks(TABLE)
    assert len(segments) == 1
    assert segments[0][1] is True
    assert "| cafe | 2 |" in segments[0][0]


def test_text_around_a_table_keeps_its_order():
    kinds = [is_table for _, is_table in _split_table_blocks(f"antes\n{TABLE}\ndepois")]
    assert kinds == [False, True, False]


# --------------------------------------------------------------------------- #
# semantic_chunk_text — the grouping decision, with a fake embedder
#
# Each sentence is embedded as a unit axis chosen by its first letter, so cosine
# between two sentences is exactly 1.0 (same letter) or 0.0 (different). That
# turns "did it group these?" into an assertion instead of a judgement call.
# --------------------------------------------------------------------------- #

AXES = {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "C": [0.0, 0.0, 1.0]}


@pytest.fixture
def fake_embedder(monkeypatch):
    async def embed(_client, texts):
        return [AXES[text.strip()[0]] for text in texts]

    monkeypatch.setattr(chunking, "embed", embed)


def chunk(text: str, threshold: float = 0.5, min_chars: int = MIN_CHARS) -> list[str]:
    # A stand-in for LlamaClients: the fake embedder ignores the client, but the
    # attribute is read before it is called, so it has to exist.
    clients = SimpleNamespace(embeddings=None)
    # asyncio.run rather than pytest-asyncio: five async tests do not justify a
    # dependency, and this keeps the suite runnable with plain pytest.
    return asyncio.run(semantic_chunk_text(text, clients, threshold, min_chars))  # type: ignore[arg-type]


def test_a_single_sentence_is_returned_untouched(fake_embedder):
    assert chunk("A frase unica.") == ["A frase unica."]


def test_identical_meaning_stays_in_one_chunk(fake_embedder):
    text = f"A{LONG} A{LONGER}"
    assert len(chunk(text)) == 1


def test_unrelated_long_sentences_are_split(fake_embedder):
    text = f"A{LONG} B{LONGER}"
    assert len(chunk(text)) == 2


def test_the_minimum_size_floor_beats_a_low_score(fake_embedder):
    """Below the floor the group keeps growing even across a topic change —
    otherwise a two-word opening line becomes a chunk nobody can retrieve."""
    text = f"A curta. B{LONG}"
    assert len(chunk(text)) == 1


def test_a_short_sentence_joins_even_when_unrelated(fake_embedder):
    """The runt rule, one level below _paragraphs: the group is already long
    enough to stand alone, but the incoming fragment is not."""
    text = f"A{LONG} B curta."
    assert len(chunk(text)) == 1


def test_grouping_compares_against_the_centroid_not_the_previous_sentence(fake_embedder):
    """The Day 2 design decision, pinned.

    The group ends up holding one A and one B, so its centroid sits at 45° from
    each — cosine 0.707. A third sentence B matches the previous sentence exactly
    (1.0) but only 0.707 against the centroid. With the threshold at 0.8 the two
    rules disagree, and the split proves the centroid is what decides.
    """
    text = f"A curta. B{LONG} B{LONGER}"
    assert len(chunk(text, threshold=0.8)) == 2
