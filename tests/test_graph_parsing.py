from app.rags.graph import parse_triples


def test_parses_plain_json_array():
    raw = '[{"subject": "Fornecedor A", "relation": "supplies", "object": "Fabrica B"}]'
    assert parse_triples(raw) == [("Fornecedor A", "supplies", "Fabrica B")]


def test_strips_markdown_fences():
    raw = '```json\n[{"subject": "A", "relation": "depends_on", "object": "B"}]\n```'
    assert parse_triples(raw) == [("A", "depends_on", "B")]


def test_empty_array_yields_no_triples():
    assert parse_triples("[]") == []


def test_malformed_json_yields_no_triples():
    # A bad reply must not crash the whole ingest — the chunk is just skipped.
    assert parse_triples("I could not find any relationships.") == []


def test_skips_items_missing_required_keys():
    raw = '[{"subject": "A", "relation": "r", "object": "B"}, {"subject": "C"}]'
    assert parse_triples(raw) == [("A", "r", "B")]
