from __future__ import annotations
from app.ingest.chunking import TABLE_BLOCK

def tabularity_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0

    match = TABLE_BLOCK.match(stripped)
    if match and match.end() >= len(stripped):
        return 1.0

    comma_density = stripped.count(",") / len(stripped)
    digit_ratio = sum(c.isdigit() for c in stripped) / len(stripped)
    return 0.5 * min(comma_density * 10, 1.0) + 0.5 * min(digit_ratio * 2, 1.0)