from __future__ import annotations

import re
import unicodedata

import faiss
import numpy as np

from app.core.config import Settings
from app.pipeline.clauses import question_clauses

# A proper noun (capitalised, including accented capitals) or anything carrying a
# digit. `À-ÖØ-Þ` rather than `À-Þ` because U+00D7 in the middle of that range is
# the multiplication sign, not a letter.
_DISCRIMINATIVE = re.compile(r"[A-ZÀ-ÖØ-Þ]\w*|\w*\d\w*")


def cache_key(question: str) -> tuple[int, frozenset[str]]:
    """What a paraphrase is not allowed to change: how many questions, and about what.

    The names and numbers, because "receita do Sudeste" and "receita do Nordeste"
    are different questions and no rewording turns one into the other, while "soma
    das vendas" and "total das vendas" are the same one. Accents and case are
    folded, so "São Paulo" and "SAO PAULO" are one token. The interrogative that
    opens the sentence is kept rather than special-cased: keeping it can only cost
    a hit, and dropping it could cost a correct answer for a question that opens
    with the name it is about.

    And the number of questions, because a message that asks two of them is not
    the same message as one that asks the first alone. Measured: "qual foi a
    receita do Sudeste e quem aprova uma compra desse valor" and "qual foi a
    receita do Sudeste" carry the same names, sit well above the threshold, and
    have different answers -- the shorter one was served the longer one's reply.
    """
    folded = unicodedata.normalize("NFKD", question)
    without_accents = "".join(c for c in folded if not unicodedata.combining(c))
    names = frozenset(match.lower() for match in _DISCRIMINATIVE.findall(without_accents))
    return len(question_clauses(question)), names


class SemanticCache:
    """CAG — a repeated (or nearly repeated) question is answered straight from RAM.

    Takes the vector *and* the question text. The vector alone was measured
    serving wrong answers: "qual a receita do Sudeste" and "qual a receita do
    Nordeste" differ by one word in a long sentence and score 0.917 against each
    other, comfortably above the 0.80 threshold, so the second question was
    answered with the first one's figure. Raising the threshold does not separate
    them -- legitimate paraphrases of one question score 0.771 to 0.911, so the
    two ranges overlap and no single cutoff exists.

    So a hit needs two independent things to agree: the embedding says the
    questions mean the same, and the names and numbers in them are identical.
    Geometry is good at "same topic" and bad at "same entity"; exact token
    comparison is the opposite. Neither is asked to do the other's job.

    Fails closed when there is nothing to check: a question with no capitalised
    word and no digit produces an empty set, and an empty set trivially equals
    another empty set. "qual a receita do sudeste" and "qual a receita do
    nordeste", both typed lowercase, would otherwise share that empty key and
    reintroduce the exact bug this class exists to fix. Refusing the hit costs a
    cache miss on a question that genuinely names nothing; serving it wrong costs
    a fabricated number that looks like a real answer.

    Still pure arithmetic and still unit-testable with handmade vectors, no
    llama-server involved.
    """

    def __init__(self, settings: Settings) -> None:
        self._threshold = settings.cache.similarity_threshold
        self._max_entries = settings.cache.max_entries
        # IndexFlatIP = exact brute-force search by inner product. Vectors arrive
        # already normalized, so the inner product IS the cosine similarity —
        # no division by norms, same trick used in the router.
        self._index = faiss.IndexFlatIP(settings.llama.embeddings.dimensions)
        self._answers: list[str] = []
        self._keys: list[tuple[int, frozenset[str]]] = []

    def lookup(self, vector: list[float], question: str) -> str | None:
        """Answer stored for a question that is both near enough and about the same things."""
        if self._index.ntotal == 0:
            return None
        wanted = cache_key(question)
        if not wanted[1]:
            # Nothing to pin the question to: an empty set matches any other
            # empty set, which is the token check silently turning itself off.
            return None
        # More than one neighbour, because the nearest is not necessarily the one
        # that passes the token check: the Nordeste question sits closer to the
        # Sudeste one than to its own earlier paraphrase.
        depth = min(self._index.ntotal, _NEIGHBOURS)
        scores, ids = self._index.search(_as_batch(vector), depth)
        for score, index in zip(scores[0], ids[0], strict=True):
            if score < self._threshold:
                break  # search returns them sorted, so nothing further can qualify
            if self._keys[index] == wanted:
                return self._answers[index]
        return None

    def store(self, vector: list[float], question: str, answer: str) -> None:
        # Full cache simply stops accepting new entries. No eviction policy until
        # a real workload proves one is needed.
        if self._index.ntotal >= self._max_entries:
            return
        self._index.add(_as_batch(vector))
        self._answers.append(answer)
        self._keys.append(cache_key(question))

    def clear(self) -> None:
        self._index.reset()
        self._answers.clear()
        self._keys.clear()


# How far down the neighbour list to look for one that also matches on names and
# numbers. Small on purpose: past a handful the similarity is no longer saying
# "the same question", and the cost of the check is paid on every question.
_NEIGHBOURS = 5


def _as_batch(vector: list[float]) -> np.ndarray:
    # FAISS always works on batches: a single vector goes in as a 1-row matrix.
    return np.array([vector], dtype="float32")
