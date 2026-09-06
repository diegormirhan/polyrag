from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd
from opentelemetry import trace

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, chat
from app.rags.base import RAGBase

_tracer = trace.get_tracer("polyrag.rag")

# Only a single leading SELECT (or a CTE that feeds one) is allowed through.
_SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# Statement separator: anything after a ";" would be a second, unvalidated query.
_MULTI_STATEMENT = re.compile(r";\s*\S")
# The projection list: everything between SELECT and the FROM that closes it.
_SELECT_LIST = re.compile(r"\bselect\b(.*?)\bfrom\b", re.IGNORECASE | re.DOTALL)
_STRING_LITERAL = re.compile(r"'[^']*'")
# A bare word that is not SQL punctuation, an alias keyword, or a number.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ALIAS = re.compile(r"\s+as\s+[A-Za-z_][A-Za-z0-9_]*", re.IGNORECASE)
# `SELECT *` or `SELECT t.*` and nothing else projected.
_STAR_ONLY = re.compile(r"^[ ]*([A-Za-z_][A-Za-z0-9_]*[.])?[*][ ]*$")
# Any clause that makes the query compute rather than dump.
_COMPUTES_SOMETHING = re.compile(r"\b(where|group[ ]+by|having|order[ ]+by|join|union)\b", re.IGNORECASE)
_SQL_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(sql: str) -> str:
    # Models often wrap SQL in markdown fences despite being told not to.
    return _SQL_FENCE.sub("", sql).strip()


def is_fabricated(sql: str) -> bool:
    """True when the query answers from itself instead of from the table.

    Asked "com que frequencia o Banco de Dados Orion replica" -- a question the
    sales table cannot answer -- the model produced:

        SELECT 'Banco de Dados Orion' AS banco,
               'Sincronizacao em tempo real' AS frequencia FROM vendas_2026_q1

    Every output column is a string literal it invented. The query is valid, it
    is read-only, it returns a row, and the row is a hallucination wearing SQL
    syntax. That defeats the reason this route exists: a figure from a generated
    SELECT is trustworthy because it came from the data.

    Detected by structure rather than by prompting against it: a SELECT list with
    no reference to any column and no aggregate is not reading anything.
    """
    select_list = _SELECT_LIST.search(sql)
    if not select_list:
        return False
    projected = select_list.group(1)
    if not _STRING_LITERAL.search(projected):
        return False
    # Strip the literals AND the aliases naming them: in `'Orion' AS banco` the
    # alias is a word the model chose, not a column it read, so leaving it in
    # would make every fabricated projection look like it referenced something.
    remainder = _ALIAS.sub("", _STRING_LITERAL.sub("", projected))
    # A quoted literal is only fabrication when nothing real is projected beside
    # it: `SELECT 'total' AS label, SUM(receita) ...` is a legitimate query.
    return not _IDENTIFIER.search(remainder)


def is_table_dump(sql: str) -> bool:
    """True when the query selects everything and computes nothing.

    Asked a question this table cannot answer, the model does not refuse -- it
    writes `SELECT * FROM vendas_2026_q1` and hands back five arbitrary rows of
    sales data, which the answer step then treats as context. The result is a
    confident answer about the wrong subject.

    A question that belongs here always asks the table to DO something: filter,
    aggregate, group or order. Measured across the 24 relational questions of
    both evaluation sets, every one produced a WHERE, a SUM or an ORDER BY, and
    none produced a bare dump; all four misrouted questions produced exactly a
    bare dump. Separation is clean, so this is structure rather than a threshold.

    Returning no rows for it lets the orchestrator fall back to the vector store.
    """
    select_list = _SELECT_LIST.search(sql)
    if select_list is None or _STAR_ONLY.match(select_list.group(1)) is None:
        return False
    return _COMPUTES_SOMETHING.search(sql) is None


def validate_read_only(sql: str) -> str:
    """Raises unless `sql` is a single read-only statement. Returns the cleaned SQL."""
    cleaned = _strip_fences(sql)
    if not _SELECT_ONLY.match(cleaned):
        raise ValueError(f"only SELECT queries are allowed, got: {cleaned[:60]}")
    if _MULTI_STATEMENT.search(cleaned):
        raise ValueError("only a single statement is allowed")
    if is_table_dump(cleaned):
        raise ValueError(f"query dumps the table instead of answering: {cleaned[:80]}")
    if is_fabricated(cleaned):
        raise ValueError(f"query invents its own answer instead of reading the table: {cleaned[:80]}")
    return cleaned.rstrip(";")


class RelationalRAG(RAGBase):
    """RAG 1 — tabular data in SQLite, queried through generated SQL."""

    def __init__(self, clients: LlamaClients, settings: Settings | None = None) -> None:
        self._settings = settings or load_config()
        self._clients = clients
        self._db_path = Path(self._settings.paths.sqlite_db)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async def ingest(self, content: tuple[pd.DataFrame, str]) -> None:
        df, table_name = content
        with sqlite3.connect(self._db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

    def schema(self) -> str:
        # The generated CREATE TABLE statements ARE the schema context the LLM needs.
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
            ).fetchall()
        return "\n".join(row[0] for row in rows)

    async def stats(self) -> dict:
        with sqlite3.connect(self._db_path) as conn:
            names = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            tables = [
                # Table names come from sqlite_master, not from a request, so they
                # cannot be attacker-controlled here; quoting still guards odd names.
                {"name": name, "rows": conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]}
                for name in names
            ]
        return {"tables": tables}

    async def generate_sql(self, question: str) -> str:
        prompt = self._settings.rags.relational.text_to_sql_prompt.format(
            schema=self.schema(),
            question=question,
        )
        with _tracer.start_as_current_span("rag.relational.generate_sql") as span:
            raw = await chat(self._clients.llm, [{"role": "user", "content": prompt}], temperature=0)
            sql = validate_read_only(raw)
            # The generated query on the span is the audit trail for a Text-to-SQL
            # answer: a wrong number is explained by the SQL, not by the prose.
            span.set_attribute("sql.query", sql)
            return sql

    async def query(self, question: str, top_k: int = 5) -> list[dict]:
        try:
            sql = await self.generate_sql(question)
        except (ValueError, sqlite3.Error) as error:
            # A rejected or unrunnable query means this store has no answer, not
            # that the request failed. Returning empty lets the orchestrator fall
            # back to the vector store, which is the right behaviour when a
            # question about prose was routed here by mistake.
            with _tracer.start_as_current_span("rag.relational.rejected") as span:
                span.set_attribute("sql.rejected_reason", str(error)[:200])
            return []

        with _tracer.start_as_current_span("rag.relational.execute") as span:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(sql).fetchmany(top_k)
            except sqlite3.Error as error:
                # Generated SQL can name a column that does not exist. Same
                # reasoning as above: no answer here, let the fallback try.
                span.set_attribute("sql.error", str(error)[:200])
                return []
            span.set_attribute("sql.rows", len(rows))
            if not rows:
                return []
            # The query leads the results because the question's intent lives in it,
            # not in the rows. Measured: `ORDER BY receita ASC LIMIT 1` returned the
            # correct single row and the model answered "only a list of regions,
            # impossible to determine" -- it could not know the row was the minimum.
            # It is also the honest thing to show in the Sources panel.
            return [{"sql": sql}, *(dict(row) for row in rows)]
