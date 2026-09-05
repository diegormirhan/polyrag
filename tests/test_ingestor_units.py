"""The two pure helpers the ingestion path depends on.

Both come from real ingestion failures: a chunk the router sent to the relational
store that held no table at all, and file names that are not valid SQL
identifiers.
"""

from pathlib import Path

from app.pipeline.ingestor import _markdown_table_to_frame, _table_name

TABLE = "| Item | Qtd |\n|------|:---:|\n| Cafe | 2   |\n| Pao  | 10  |"


# --------------------------------------------------------------------------- #
# _markdown_table_to_frame — an empty frame is the signal, not an exception
# --------------------------------------------------------------------------- #


def test_parses_a_markdown_table():
    frame = _markdown_table_to_frame(TABLE)
    assert list(frame.columns) == ["Item", "Qtd"]
    assert frame.to_dict("records") == [{"Item": "Cafe", "Qtd": "2"}, {"Item": "Pao", "Qtd": "10"}]


def test_the_alignment_row_is_layout_not_data():
    """`|---|:---:|` describes how to render, and must never become a row."""
    assert len(_markdown_table_to_frame(TABLE)) == 2


def test_text_without_a_table_yields_an_empty_frame():
    """This is the case that crashed ingestion. The router can send prose to the
    relational route — the embedding stage never inspects structure — so this has
    to return something the caller can test, not raise."""
    assert _markdown_table_to_frame("Total 12,50 Cafe 3,00 Data 01/09/2026").empty


def test_a_header_with_no_rows_yields_an_empty_frame():
    assert _markdown_table_to_frame("| a | b |\n|---|---|").empty


def test_an_empty_string_yields_an_empty_frame():
    assert _markdown_table_to_frame("").empty


# --------------------------------------------------------------------------- #
# _table_name — a file name is not a SQL identifier
# --------------------------------------------------------------------------- #


def test_spaces_and_punctuation_become_underscores():
    assert _table_name(Path("Vendas 2026-Q1.csv")) == "vendas_2026_q1"


def test_a_leading_digit_gets_a_prefix():
    """SQLite rejects an identifier that starts with a digit."""
    assert _table_name(Path("2026 vendas.csv")) == "t_2026_vendas"


def test_accents_are_preserved():
    """They are word characters, and mangling them would make the table name
    unrecognisable next to the file it came from."""
    assert _table_name(Path("Questionário.csv")) == "questionário"


def test_an_index_distinguishes_tables_from_one_file():
    """An image can yield more than one table, and without the index the second
    would overwrite the first."""
    assert _table_name(Path("nota.png"), 0) == "nota_0"
    assert _table_name(Path("nota.png"), 1) == "nota_1"


def test_a_name_with_nothing_usable_falls_back():
    assert _table_name(Path("___.csv")) == "table"


def test_the_same_file_always_gives_the_same_name():
    """Re-ingesting replaces the table rather than creating a second one."""
    assert _table_name(Path("a/b/vendas.csv")) == _table_name(Path("c/vendas.csv"))
