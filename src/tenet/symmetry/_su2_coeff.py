"""Exact SU(2) recoupling coefficients over doubled spins (``dj = 2j``).

Everything is computed in exact integer arithmetic: ``w3j``/``w6j``/``cg`` return a
sqrt-rational ``(sign, radicand)`` whose value is ``sign * sqrt(radicand)``, so the
gauge is exact and float rounding happens only at the final conversion.

Conventions follow racah / TensorKitSectors: 3j and CG in Condon-Shortley phase; see
``SU2_GAUGE`` in :mod:`tenet.symmetry.su2` for the full fingerprint. F-symbols,
R-symbols and Frobenius-Schur signs are Milestone 4 and deliberately absent.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache
from math import factorial, sqrt

import numpy as np

__all__ = ["cg", "cg_tensor", "triangle", "value", "w3j", "w6j"]

SqrtRational = tuple[int, Fraction]
"""``(sign, radicand)`` encoding the exact value ``sign * sqrt(radicand)``."""

_ZERO: SqrtRational = (0, Fraction(0))


def triangle(dj1: int, dj2: int, dj3: int) -> bool:
    """Triangle rule on doubled spins: ``|j1-j2| <= j3 <= j1+j2`` with integer steps."""
    return abs(dj1 - dj2) <= dj3 <= dj1 + dj2 and (dj1 + dj2 + dj3) % 2 == 0


def _delta(dj1: int, dj2: int, dj3: int) -> Fraction:
    """Squared Racah triangle coefficient ``Delta(j1,j2,j3)**2``, exact."""
    return Fraction(
        factorial((dj1 + dj2 - dj3) // 2)
        * factorial((dj1 - dj2 + dj3) // 2)
        * factorial((-dj1 + dj2 + dj3) // 2),
        factorial((dj1 + dj2 + dj3) // 2 + 1),
    )


@cache
def w3j(dj1: int, dj2: int, dj3: int, dm1: int, dm2: int, dm3: int) -> SqrtRational:
    """Wigner 3j symbol ``(j1 j2 j3; m1 m2 m3)`` as an exact sqrt-rational."""
    if dm1 + dm2 + dm3 != 0 or not triangle(dj1, dj2, dj3):
        return _ZERO
    for dj, dm in ((dj1, dm1), (dj2, dm2), (dj3, dm3)):
        if abs(dm) > dj or (dj - dm) % 2:
            return _ZERO
    t_lo = max(0, (dj2 - dj3 - dm1) // 2, (dj1 - dj3 + dm2) // 2)
    t_hi = min((dj1 + dj2 - dj3) // 2, (dj1 - dm1) // 2, (dj2 + dm2) // 2)
    total = Fraction(0)
    for t in range(t_lo, t_hi + 1):
        denom = (
            factorial(t)
            * factorial((dj1 + dj2 - dj3) // 2 - t)
            * factorial((dj1 - dm1) // 2 - t)
            * factorial((dj2 + dm2) // 2 - t)
            * factorial((dj3 - dj2 + dm1) // 2 + t)
            * factorial((dj3 - dj1 - dm2) // 2 + t)
        )
        total += Fraction((-1) ** t, denom)
    if total == 0:
        return _ZERO
    pre = _delta(dj1, dj2, dj3)
    for dj, dm in ((dj1, dm1), (dj2, dm2), (dj3, dm3)):
        pre *= factorial((dj + dm) // 2) * factorial((dj - dm) // 2)
    sign = (-1) ** ((dj1 - dj2 - dm3) // 2) * (1 if total > 0 else -1)
    return (sign, pre * total * total)


@cache
def w6j(dj1: int, dj2: int, dj3: int, dj4: int, dj5: int, dj6: int) -> SqrtRational:
    """Wigner 6j symbol ``{j1 j2 j3; j4 j5 j6}`` as an exact sqrt-rational."""
    tris = ((dj1, dj2, dj3), (dj1, dj5, dj6), (dj4, dj2, dj6), (dj4, dj5, dj3))
    if not all(triangle(*t) for t in tris):
        return _ZERO
    pre = Fraction(1)
    for t in tris:
        pre *= _delta(*t)
    sums = [sum(t) // 2 for t in tris]
    quads = [
        (dj1 + dj2 + dj4 + dj5) // 2,
        (dj2 + dj3 + dj5 + dj6) // 2,
        (dj3 + dj1 + dj6 + dj4) // 2,
    ]
    total = Fraction(0)
    for t in range(max(sums), min(quads) + 1):
        denom = 1
        for s in sums:
            denom *= factorial(t - s)
        for q in quads:
            denom *= factorial(q - t)
        total += Fraction((-1) ** t * factorial(t + 1), denom)
    if total == 0:
        return _ZERO
    return (1 if total > 0 else -1, pre * total * total)


@cache
def cg(dj1: int, dm1: int, dj2: int, dm2: int, dj3: int, dm3: int) -> SqrtRational:
    """Clebsch-Gordan ``<j1 m1; j2 m2 | j3 m3>`` (Condon-Shortley), exact sqrt-rational."""
    sign, radicand = w3j(dj1, dj2, dj3, dm1, dm2, -dm3)
    if sign == 0:
        return _ZERO
    return (sign * (-1) ** ((dj2 - dj1 - dm3) // 2), radicand * (dj3 + 1))


def value(pair: SqrtRational) -> float:
    """Collapse a ``(sign, radicand)`` pair to a float."""
    sign, radicand = pair
    return sign * sqrt(float(radicand))


@cache
def cg_tensor(dj1: int, dj2: int, dj3: int) -> np.ndarray:
    """CG tensor of shape ``(dj1+1, dj2+1, dj3+1)``, magnetic index **descending**.

    Index 0 of each axis is ``m = +j`` (TensorKit convention), i.e. axis entry ``i``
    carries ``dm = dj - 2*i``. Returned array is read-only and cached.
    """
    out = np.zeros((dj1 + 1, dj2 + 1, dj3 + 1))
    for i1 in range(dj1 + 1):
        dm1 = dj1 - 2 * i1
        for i2 in range(dj2 + 1):
            dm2 = dj2 - 2 * i2
            dm3 = dm1 + dm2
            i3 = (dj3 - dm3) // 2
            if 0 <= i3 <= dj3:
                out[i1, i2, i3] = value(cg(dj1, dm1, dj2, dm2, dj3, dm3))
    out.flags.writeable = False
    return out
