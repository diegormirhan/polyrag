"""Stage 1 of the router: is this chunk a table?

The threshold that consumes this score is tau_heuristic = 0.8, so the tests pin
where real content actually lands relative to it — which turned out not to be
where the code comments assumed.
"""

from app.pipeline.tabularity import tabularity_score

TAU_HEURISTIC = 0.8  # mirrors config.yaml; the score only matters against it

MARKDOWN_TABLE = "| item | qtd |\n|------|-----|\n| cafe | 2 |\n| pao | 10 |"


def test_empty_text_scores_zero():
    assert tabularity_score("") == 0.0
    assert tabularity_score("   \n  ") == 0.0


def test_a_complete_markdown_table_scores_one():
    assert tabularity_score(MARKDOWN_TABLE) == 1.0


def test_a_table_followed_by_prose_does_not_score_one():
    """Only a chunk that is ENTIRELY a table gets the shortcut. Otherwise the
    trailing prose would be stored as if it were rows."""
    assert tabularity_score(f"{MARKDOWN_TABLE}\n\nO total acima refere-se ao mes de marco.") < 1.0


def test_prose_scores_near_zero():
    prose = "A Politica PC-04 estabelece os limites de alcada para aquisicoes da empresa."
    assert tabularity_score(prose) < 0.1


def test_dense_numeric_rows_clear_the_threshold():
    """What the heuristic is actually for: comma-and-digit dense content with no
    Markdown pipes, such as a raw CSV row."""
    assert tabularity_score("1,2,3,4,5,6,7,8,9,10,11,12") >= TAU_HEURISTIC


def test_a_receipt_does_not_clear_the_threshold():
    """Pinned because a code comment claimed the opposite.

    A receipt reached the relational route during development, and the comment
    explaining the storage-time guard blamed this heuristic. Measured, it scores
    ~0.53 — well under 0.8. The receipt arrived through the embedding stage,
    which never inspects structure at all. The guard is still needed; the reason
    written next to it was wrong.
    """
    receipt = (
        "SUPERMERCADO ABC LTDA CNPJ 12.345.678/0001-90 "
        "CAFE 500G 2 UN 15,90 31,80 ARROZ 5KG 1 UN 24,50 TOTAL 56,30"
    )
    assert tabularity_score(receipt) < TAU_HEURISTIC


def test_a_csv_header_alone_is_not_enough():
    """Commas without digits only reach half the score, by construction: the two
    signals are weighted equally and neither can carry the decision alone."""
    assert tabularity_score("produto,regiao,vendas,trimestre,meta,realizado") < TAU_HEURISTIC


def test_the_score_never_leaves_the_unit_interval():
    for text in ["", MARKDOWN_TABLE, ",,,,,,,,,,", "1111111111", "a"]:
        assert 0.0 <= tabularity_score(text) <= 1.0
