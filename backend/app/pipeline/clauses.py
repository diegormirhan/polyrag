"""Splits a message into the separate questions it asks, if it asks more than one.

A user does not always ask one thing at a time. "Qual foi a receita de marco e
qual norma regula a retencao desses registros?" needs the sales table *and* the
compliance graph, and a router that picks one store answers half the message.

The obvious detector was a threshold on the router's second-best score, and it
was measured and rejected. Compound questions score 0.365 to 0.511 there and
single ones 0.303 to 0.476: the ranges overlap almost completely, and the single
question "Como uma reclamacao de cliente deve ser tratada?" outscores five of the
seven compound ones. The reason is mechanical -- an embedding of two topics lands
near their average, so a second subject dilutes both scores instead of lifting the
second one. Same shape of failure as the cache threshold, and no cutoff exists.

What does separate them is structure, not magnitude: a compound message contains
two questions. Measured on 9 compound messages (including one in English and one
written as two sentences) and the 80 questions of both evaluation sets: 9/9
detected, 0/80 false positives.
"""

from __future__ import annotations

import re
import unicodedata

# Where one clause can end and another begin. `\be\b` is the Portuguese "and";
# it also matches the verb "e" (é written without its accent, which this corpus
# does), and that ambiguity is what the interrogative test below exists to settle.
_BOUNDARY = re.compile(r"\?|;|\be\b|\btambem\b|\band\b|\balso\b", re.IGNORECASE)

# Words that make a fragment a question of its own, in either language.
_ASKS = re.compile(
    r"\b(qual|quais|quem|quando|onde|como|quanto|quantos|quanta|quantas|que|porque|"
    r"liste|descreva|explique|compare|"
    r"what|which|who|whom|when|where|how|why|list|describe|explain)\b",
    re.IGNORECASE,
)

# Below this a fragment is a stray word, not a question. "e" between two short
# noun phrases splits into pieces this short.
_MIN_CHARS = 18


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def question_clauses(message: str) -> list[str]:
    """The separate questions in `message`, or `[message]` when there is only one.

    Every surviving fragment must read as a question by itself. Without that test
    two of the 80 evaluation questions split wrongly: "Uma exportacao sem
    anonimizacao e reportada para quem?", where the "e" is the verb, and "Qual a
    diferenca entre um atraso comunicado e um atraso descoberto?", where it joins
    two noun phrases rather than two questions. In both, one half carries no
    interrogative at all, which is exactly what marks it as not a question.
    """
    fragments = [fragment.strip(" ,.") for fragment in _BOUNDARY.split(_fold(message))]
    kept = [fragment for fragment in fragments if len(fragment) >= _MIN_CHARS]
    if len(kept) < 2 or not all(_ASKS.search(fragment) for fragment in kept):
        return [message]
    return kept
