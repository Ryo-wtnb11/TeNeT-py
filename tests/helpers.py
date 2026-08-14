"""Test helpers shared between test modules. Not part of the library.

``supersign`` is the dense-side Koszul sign of an axis permutation, written for
issue #39 and promoted here unchanged when #51 needed the same oracle for
``tensordot``: the fermionic correctness of a contraction is *inherited* from
``transpose``, so the two suites must weigh it on the same scale.
"""

import numpy as np

from tenet.space import GradedSpace

__all__ = ["parity_vector", "supersign"]


def parity_vector(space: GradedSpace) -> np.ndarray:
    """Parity of each dense index of ``space``, in canonical sector order."""
    return np.concatenate([np.full(m, a.parity) for a, m in space.sectors])


def supersign(legs, p: tuple[int, ...], *, per_side: bool) -> np.ndarray:
    """Dense-side Koszul sign array, shaped like ``np.transpose(dense, p)``.

    ``per_side=False`` counts every inversion of ``p`` (correct when every leg
    lives on one side); ``per_side=True`` counts only inversions between two axes
    of the same side, which is TeNeT-py's stated convention.
    """
    pars = [parity_vector(legs[ax].space) for ax in p]
    sides = [legs[ax].side for ax in p]
    sign = np.ones(tuple(len(v) for v in pars))
    n = len(p)
    for j in range(n):
        for k in range(j + 1, n):
            if p[j] <= p[k] or (per_side and sides[j] is not sides[k]):
                continue
            shape_j = [1] * n
            shape_j[j] = len(pars[j])
            shape_k = [1] * n
            shape_k[k] = len(pars[k])
            product = pars[j].reshape(shape_j) * pars[k].reshape(shape_k)
            sign = sign * (-1.0) ** product
    return sign
