import math

from app.core.vectors import dot, normalize
from app.pipeline.router import RouteDecision, _best_route_scores


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
        "route_a": [[1.0, 0.0]],  # identical -> cosine 1.0
        "route_b": [[0.0, 1.0]],  # orthogonal -> cosine 0.0
        "route_c": [[0.7071, 0.7071]],  # 45 degrees -> cosine ~0.7071
    }
    scores = _best_route_scores(vector, anchors)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    assert [route for route, _ in ranked] == ["route_a", "route_c", "route_b"]


# --------------------------------------------------------------------------- #
# RouteDecision — the derived numbers ARE the decision, so they get pinned
#
# score_top1/top2/margin are properties over `scores` rather than stored fields,
# so a span's numbers can never disagree with the numbers that made the call.
# --------------------------------------------------------------------------- #


def decision(scores: dict[str, float]) -> RouteDecision:
    return RouteDecision(route="graph", decision_stage="embedding", scores=scores)


def test_margin_is_the_gap_between_the_top_two():
    d = decision({"graph": 0.488, "relational": 0.381, "vectorial": 0.316})
    assert math.isclose(d.score_top1, 0.488)
    assert math.isclose(d.score_top2, 0.381)
    assert math.isclose(d.margin, 0.107)


def test_ranking_ignores_the_order_the_scores_arrived_in():
    ascending = decision({"vectorial": 0.316, "relational": 0.381, "graph": 0.488})
    descending = decision({"graph": 0.488, "relational": 0.381, "vectorial": 0.316})
    assert ascending.margin == descending.margin


def test_a_single_route_has_no_runner_up():
    """The heuristic stage decides without scoring the other routes, so there is
    no second place. Reporting 0.0 keeps margin equal to the winning score
    instead of raising on an empty slot."""
    d = decision({"relational": 0.92})
    assert d.score_top2 == 0.0
    assert math.isclose(d.margin, 0.92)


def test_a_perfect_tie_has_no_margin():
    """This is exactly the gray zone: zero margin is what sends the decision to
    the model instead of letting arithmetic pick arbitrarily."""
    assert decision({"graph": 0.5, "vectorial": 0.5}).margin == 0.0
