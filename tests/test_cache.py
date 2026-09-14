from app.core.config import load_config
from app.core.vectors import normalize
from app.pipeline.cache import SemanticCache, cache_key

DIMENSIONS = 1024


def _unit_vector(position: int) -> list[float]:
    v = [0.0] * DIMENSIONS
    v[position] = 1.0
    return normalize(v)


def _near(position: int, similarity: float) -> list[float]:
    """A vector at a chosen cosine from `_unit_vector(position)`.

    Two orthogonal axes combined as cos*e_i + sin*e_j: the dot product with e_i is
    exactly `similarity`, which lets a test name the score it is exercising instead
    of hoping a handmade vector lands on the right side of the threshold.
    """
    v = [0.0] * DIMENSIONS
    v[position] = similarity
    v[position + 500] = (1 - similarity**2) ** 0.5
    return v


def test_empty_cache_never_hits():
    assert SemanticCache(load_config()).lookup(_unit_vector(0), "qualquer") is None


def test_identical_vector_hits():
    cache = SemanticCache(load_config())
    cache.store(_unit_vector(0), "qual o total de vendas", "resposta A")
    assert cache.lookup(_unit_vector(0), "qual o total de vendas") == "resposta A"


def test_orthogonal_vector_misses():
    # Cosine between two different unit axes is exactly 0 — far below any threshold.
    cache = SemanticCache(load_config())
    cache.store(_unit_vector(0), "qual o total de vendas", "resposta A")
    assert cache.lookup(_unit_vector(1), "qual o total de vendas") is None


def test_clear_empties_the_cache():
    cache = SemanticCache(load_config())
    cache.store(_unit_vector(0), "qual o total de vendas", "resposta A")
    cache.clear()
    assert cache.lookup(_unit_vector(0), "qual o total de vendas") is None


def test_same_topic_different_region_misses():
    """The measured bug: 0.917 similarity, and the two questions are not the same one.

    Without the token check this returned the Sudeste figure for the Nordeste
    question — the failure the README documented.
    """
    cache = SemanticCache(load_config())
    cache.store(_unit_vector(0), "qual foi a receita total da regiao Sudeste", "R$ 4.988.300")
    assert cache.lookup(_near(0, 0.917), "qual foi a receita total da regiao Nordeste") is None


def test_paraphrase_without_names_still_hits():
    """The thing the check must not break: same question, different wording."""
    cache = SemanticCache(load_config())
    cache.store(_unit_vector(0), "qual o total de vendas", "R$ 12.400")
    assert cache.lookup(_near(0, 0.88), "qual a soma das vendas") == "R$ 12.400"


def test_a_closer_wrong_neighbour_does_not_hide_the_right_one():
    """Why the lookup scans several neighbours instead of only the nearest.

    The Nordeste question is nearer to the Sudeste one than to its own earlier
    paraphrase, so checking only the top hit would report a miss for a question
    that is genuinely cached.
    """
    cache = SemanticCache(load_config())
    cache.store(_near(0, 0.90), "receita da regiao Nordeste no ano", "R$ 2.100.000")
    cache.store(_unit_vector(0), "qual foi a receita total da regiao Sudeste", "R$ 4.988.300")
    # Scores 0.990 against the Sudeste entry and 0.953 against the Nordeste one.
    assert cache.lookup(_near(0, 0.99), "qual a receita da regiao Nordeste") == "R$ 2.100.000"


def test_the_key_is_names_and_numbers_only():
    assert cache_key("qual a receita do Sudeste em 2026") == (1, frozenset({"sudeste", "2026"}))
    # Accents fold, so the same name written two ways is one token.
    assert cache_key("vendas em Sao Paulo") == cache_key("vendas em São Paulo")
    # Nothing to pin the question to: the cache falls back to geometry alone.
    assert cache_key("qual o total de vendas") == (1, frozenset())


def test_a_compound_question_is_not_its_own_first_half():
    """Measured: the half-question was served the compound question's answer.

    Both carry the name Sudeste and both open with "qual", so the names alone do
    not separate them -- the number of questions asked does.
    """
    cache = SemanticCache(load_config())
    cache.store(
        _unit_vector(0),
        "Qual foi a receita do Sudeste e quem aprova uma compra desse valor?",
        "duas respostas",
    )
    assert cache.lookup(_near(0, 0.93), "Qual foi a receita do Sudeste?") is None
