from app.rags.graph import normalize_entity, parse_triples


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


# --------------------------------------------------------------------------- #
# normalize_entity — the same thing must always become the same node
#
# Without this the graph fragmented into disconnected islands and multi-hop
# retrieval silently stopped working: "A Fabrica Beta" in one chunk and
# "Fabrica_Beta" in the next became two nodes with no edge between them.
# --------------------------------------------------------------------------- #


def test_a_leading_article_is_dropped():
    assert normalize_entity("A Fabrica Beta") == "Fabrica Beta"
    assert normalize_entity("os Contratos") == "Contratos"


def test_underscores_become_spaces():
    assert normalize_entity("Contrato_Gama") == "Contrato Gama"


def test_an_article_inside_the_name_survives():
    """Only a LEADING article is noise. "Banco de Dados Orion" must not lose its
    middle, or two different entities collapse into one."""
    assert normalize_entity("Banco de Dados Orion") == "Banco de Dados Orion"


def test_a_word_that_merely_starts_with_a_is_untouched():
    """The rule needs whitespace after the article, or "Atlas" becomes "tlas"."""
    assert normalize_entity("Atlas") == "Atlas"
    assert normalize_entity("A2 Sistema") == "A2 Sistema"


def test_the_two_spellings_of_one_entity_converge():
    assert normalize_entity("A Fabrica_Beta ") == normalize_entity("Fabrica Beta")


# --------------------------------------------------------------------------- #
# max_entity_words — a clause is not an entity
#
# "anonimizacao antes de qualquer exportacao para terceiros" became a graph node
# that nothing else would ever match. Enforced here rather than trusted to the
# prompt: a 4B model ignores "at most N words" often enough to matter.
# --------------------------------------------------------------------------- #


def _triple(subject: str, obj: str) -> str:
    return f'[{{"subject": "{subject}", "relation": "requires", "object": "{obj}"}}]'


def test_a_long_object_drops_the_whole_triple():
    raw = _triple("Norma NC-11", "anonimizacao antes de qualquer exportacao para terceiros")
    assert parse_triples(raw, max_entity_words=5) == []


def test_a_long_subject_drops_the_whole_triple():
    raw = _triple("Compras entre dez mil e cinquenta mil reais", "aprovacao")
    assert parse_triples(raw, max_entity_words=5) == []


def test_a_four_word_proper_noun_survives():
    """Portuguese names run long through "de" — the limit has to clear
    "Banco de Dados Orion" or it throws away real entities."""
    raw = _triple("Sistema Atlas", "Banco de Dados Orion")
    assert parse_triples(raw, max_entity_words=5) == [("Sistema Atlas", "requires", "Banco de Dados Orion")]


def test_an_empty_name_is_not_an_entity():
    assert parse_triples(_triple("", "Norma NC-11")) == []


def test_the_limit_is_configurable():
    raw = _triple("Sistema Atlas", "Banco de Dados Orion")
    assert parse_triples(raw, max_entity_words=3) == []
    assert len(parse_triples(raw, max_entity_words=4)) == 1
