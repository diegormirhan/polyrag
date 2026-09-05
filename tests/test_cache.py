from app.core.config import load_config
from app.core.vectors import normalize
from app.pipeline.cache import SemanticCache

DIMENSIONS = 1024


def _unit_vector(position: int) -> list[float]:
    v = [0.0] * DIMENSIONS
    v[position] = 1.0
    return normalize(v)


def test_empty_cache_never_hits():
    assert SemanticCache(load_config()).lookup(_unit_vector(0)) is None


def test_identical_vector_hits():
    cache = SemanticCache(load_config())
    cache.store(_unit_vector(0), "resposta A")
    assert cache.lookup(_unit_vector(0)) == "resposta A"


def test_orthogonal_vector_misses():
    # Cosine between two different unit axes is exactly 0 — far below any threshold.
    cache = SemanticCache(load_config())
    cache.store(_unit_vector(0), "resposta A")
    assert cache.lookup(_unit_vector(1)) is None


def test_clear_empties_the_cache():
    cache = SemanticCache(load_config())
    cache.store(_unit_vector(0), "resposta A")
    cache.clear()
    assert cache.lookup(_unit_vector(0)) is None
