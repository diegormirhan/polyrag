from __future__ import annotations

import math


def normalize(v: list[float]) -> list[float]:
    # L2 norm (length) of the vector — square root of the sum of squares.
    # Ex: vector [3, 4] has norm sqrt(3² + 4²) = sqrt(25) = 5.
    norm = math.sqrt(sum(x * x for x in v))
    # Divide every position by the norm -> the resulting vector has length exactly 1.
    return [x / norm for x in v]


def dot(a: list[float], b: list[float]) -> float:
    # Dot product — sum of (position i of vector a) * (position i of vector b).
    # When both vectors are already normalized (length 1), this IS the cosine
    # similarity directly — no division by norms needed here anymore.
    # strict: two vectors of different dimensions is a bug, and zip would
    # silently return the dot product of the shorter prefix instead.
    return sum(x * y for x, y in zip(a, b, strict=True))
