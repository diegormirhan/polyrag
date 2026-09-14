"""How the retrieved passages are laid out before the model reads them.

Pure list and string work, so none of this needs a server.
"""

from app.pipeline.orchestrator import _edges_first, _format_context, _join_context


def _chunks(count: int) -> list[dict]:
    return [{"text": f"chunk {index}"} for index in range(1, count + 1)]


def test_best_and_second_best_land_at_the_two_ends():
    order = [item["text"] for item in _edges_first(_chunks(5))]
    assert order == ["chunk 1", "chunk 3", "chunk 5", "chunk 4", "chunk 2"]
    # The point of the reordering: rank 1 opens and rank 2 closes.
    assert order[0] == "chunk 1"
    assert order[-1] == "chunk 2"


def test_reordering_keeps_every_passage():
    assert sorted(item["text"] for item in _edges_first(_chunks(7))) == sorted(
        item["text"] for item in _chunks(7)
    )


def test_a_single_passage_is_unchanged():
    assert _edges_first(_chunks(1)) == _chunks(1)


def test_one_store_is_not_labelled():
    """The single-route prompt must stay byte-identical to the measured one."""
    context = _join_context([("vectorial", _chunks(2))])
    assert context == _format_context("vectorial", _chunks(2))
    assert "[TEXT]" not in context


def test_several_stores_are_labelled_by_origin():
    context = _join_context([("vectorial", _chunks(2)), ("graph", _chunks(2))])
    assert context.startswith("[TEXT]")
    assert "[GRAPH]" in context
