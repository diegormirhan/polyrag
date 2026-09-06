import pytest

from app.rags.relational import is_fabricated, is_table_dump, validate_read_only


def test_accepts_plain_select():
    # Not `SELECT * FROM vendas`: a projection of everything with no clause is a
    # table dump, which validate_read_only now rejects on its own.
    sql = "SELECT receita FROM vendas WHERE regiao = 'Sul'"
    assert validate_read_only(sql) == sql


def test_accepts_cte():
    sql = "WITH t AS (SELECT 1) SELECT * FROM t"
    assert validate_read_only(sql) == sql


def test_strips_markdown_fences():
    assert validate_read_only("```sql\nSELECT 1\n```") == "SELECT 1"


def test_strips_trailing_semicolon():
    assert validate_read_only("SELECT 1;") == "SELECT 1"


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE vendas",
        "DELETE FROM vendas",
        "UPDATE vendas SET total = 0",
        "INSERT INTO vendas VALUES (1)",
        "ALTER TABLE vendas ADD COLUMN x INT",
    ],
)
def test_rejects_write_statements(sql):
    with pytest.raises(ValueError):
        validate_read_only(sql)


def test_rejects_piggybacked_statement():
    # The classic injection shape: a valid SELECT followed by a destructive one.
    with pytest.raises(ValueError):
        validate_read_only("SELECT 1; DROP TABLE vendas")


# --------------------------------------------------------------------------- #
# Fabricated queries: valid, read-only, and still a hallucination.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        # The real one, produced when the sales table was asked about a database
        # replication schedule it knows nothing about.
        "SELECT 'Banco de Dados Orion' AS banco, 'tempo real' AS frequencia FROM vendas_2026_q1",
        "SELECT 'Sim' AS resposta FROM vendas_2026_q1",
        "select 'a' as x, 'b' as y from t",
    ],
)
def test_rejects_query_that_invents_its_own_answer(sql):
    assert is_fabricated(sql)
    with pytest.raises(ValueError, match="invents"):
        validate_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT regiao, SUM(receita) FROM vendas_2026_q1 GROUP BY regiao",
        # A literal beside something real is a label, not a fabrication.
        "SELECT 'total' AS rotulo, SUM(receita) FROM vendas_2026_q1",
        "SELECT COUNT(DISTINCT regiao) FROM vendas_2026_q1",
        # A literal in the WHERE clause is a filter over real rows.
        "SELECT produto FROM vendas_2026_q1 WHERE regiao = 'Norte'",
        "WITH t AS (SELECT regiao FROM v) SELECT * FROM t",
    ],
)
def test_accepts_queries_that_read_the_table(sql):
    assert not is_fabricated(sql)
    assert validate_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    ["SELECT * FROM vendas_2026_q1", 'SELECT * FROM "vendas_2026_q1"', "select  *  from t"],
)
def test_rejects_a_bare_table_dump(sql):
    """What the model writes when the table cannot answer the question at all."""
    assert is_table_dump(sql)
    with pytest.raises(ValueError, match="dumps the table"):
        validate_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t WHERE regiao = 'Norte'",
        "SELECT SUM(receita) FROM t",
        "SELECT regiao FROM t ORDER BY receita DESC LIMIT 1",
        "SELECT t.* FROM t JOIN u ON t.id = u.id",
        "SELECT produto, SUM(receita) FROM t GROUP BY produto",
    ],
)
def test_accepts_a_query_that_computes_something(sql):
    assert not is_table_dump(sql)
    assert validate_read_only(sql)
