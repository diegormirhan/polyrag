import math

from app.core.vectors import dot, normalize
from app.pipeline.router import _best_route_scores


def test_normalize_produces_unit_length_vector():
    v = normalize([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0)


def test_dot_of_identical_normalized_vectors_is_one():
    v = normalize([1.0, 2.0, 3.0])
    assert math.isclose(dot(v, v), 1.0)


def test_dot_of_orthogonal_vectors_is_zero():
    a = normalize([1.0, 0.0])
    b = normalize([0.0, 1.0])
    assert math.isclose(dot(a, b), 0.0, abs_tol=1e-9)


def test_best_route_scores_picks_max_anchor_not_average():
    vector = [1.0, 0.0]
    anchors = {
        "route_a": [[1.0, 0.0], [-1.0, 0.0]],  # one identical anchor, one opposite
    }
    scores = _best_route_scores(vector, anchors)
    assert math.isclose(scores["route_a"], 1.0)  # takes the best match, not the average


def test_best_route_scores_ranks_routes_correctly():
    vector = [1.0, 0.0]
    anchors = {
        "route_a": [[1.0, 0.0]],        # identical -> cosine 1.0
        "route_b": [[0.0, 1.0]],        # orthogonal -> cosine 0.0
        "route_c": [[0.7071, 0.7071]],  # 45 degrees -> cosine ~0.7071
    }
    scores = _best_route_scores(vector, anchors)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    assert [route for route, _ in ranked] == ["route_a", "route_c", "route_b"]