import pytest

from app.rags.relational import validate_read_only


def test_accepts_plain_select():
    assert validate_read_only("SELECT * FROM vendas") == "SELECT * FROM vendas"


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
