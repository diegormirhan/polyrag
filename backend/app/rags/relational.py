from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, chat
from app.rags.base import RAGBase
from opentelemetry import trace

_tracer = trace.get_tracer("polyrag.rag")

# Only a single leading SELECT (or a CTE that feeds one) is allowed through.
_SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# Statement separator: anything after a ";" would be a second, unvalidated query.
_MULTI_STATEMENT = re.compile(r";\s*\S")
_SQL_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(sql: str) -> str:
    # Models often wrap SQL in markdown fences despite being told not to.
    return _SQL_FENCE.sub("", sql).strip()


def validate_read_only(sql: str) -> str:
    """Raises unless `sql` is a single read-only statement. Returns the cleaned SQL."""
    cleaned = _strip_fences(sql)
    if not _SELECT_ONLY.match(cleaned):
        raise ValueError(f"only SELECT queries are allowed, got: {cleaned[:60]}")
    if _MULTI_STATEMENT.search(cleaned):
        raise ValueError("only a single statement is allowed")
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
        sql = await self.generate_sql(question)
        with _tracer.start_as_current_span("rag.relational.execute") as span:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql).fetchmany(top_k)
            span.set_attribute("sql.rows", len(rows))
            return [dict(row) for row in rows]