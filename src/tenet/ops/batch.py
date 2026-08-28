"""Batching a plan's terms into array operations.

``(perm, terms)`` is the one shape every plan in ``tenet.ops`` reduces to: transpose each
source block by a single permutation, then scatter-add the results with per-term
coefficients. Walked term by term that is a Python loop whose length is the term count —
447,752 iterations for one SU(2) rank-8 ``tensordot`` — while the arrays it moves are 64
elements each. This module turns the same plan into a few hundred array operations,
by grouping terms that can run as one.

The grouping is a pure function of the plan, so it is computed once and cached with it,
and it holds no numerical value: the arrays here say which block goes where, never what is
in one. Coefficients ride along as data, in the dtype the per-term multiply would have
produced (see [cast_coefficients][tenet.ops.batch.cast_coefficients]) — which is what
makes the batched execution bit-identical to the loop rather than merely close to it.

Nothing here dispatches on a backend. A segment sum is where a batched scatter usually
picks up a per-library primitive — NumPy has ``add.reduceat``, JAX has ``segment_sum``,
PyTorch has ``index_add`` and no two agree — so the grouping is arranged to avoid needing
one: bucketing by *multiplicity* as well as by shape makes every bucket a rectangle, and a
rectangle is summed by indexing and adding, which every backend spells the same way.
"""

from functools import cache
from typing import Any

import autoray as ar
import numpy as np

from tenet.structure import TensorStructure

__all__ = ["batch_plan", "cast_coefficients"]


MIN_BATCH_ROWS = 4
"""Destinations a multiplicity bucket needs before an array operation beats the loop.

Batching a bucket costs one gather, one multiply and ``width - 1`` adds whatever its
height; the loop costs one Python iteration per term, i.e. ``height * width`` of them.
Below a handful of destinations the fixed cost wins.

A bucket of ``width`` 1 is left to the loop whatever its height, and that is not a
tuning constant but the shape of the work: one term per destination has no addition to
fuse, so batching would stack and unstack — both ``O(destinations)`` in Python, which is
what it is trying to remove — to save nothing. Measured, batching those anyway costs 2x
on a U(1) rank-5 transpose, where every bucket is one term wide.
"""


def cast_coefficients(coeff: np.ndarray, block: Any) -> Any:
    """``coeff`` in the dtype and backend a per-term scalar would have promoted to.

    The looped path multiplies a block by a Python scalar, which NumPy and both array
    libraries treat as weak: a ``complex64`` block times a real coefficient stays
    ``complex64``. An array of coefficients is not weak, so a ``float64`` one would
    promote the whole tensor to ``complex128``. Casting first is what keeps the two paths
    bit-identical *and* keeps a strict-``complex64`` consumer in ``complex64``.

    Parameters
    ----------
    coeff : numpy.ndarray
        The bucket's coefficients, already shaped to broadcast against ``block``.
    block : array
        The blocks the coefficients multiply; supplies backend and dtype.

    Returns
    -------
    array
        ``coeff`` as a backend-native array of the promoted dtype.
    """
    dtype = np.dtype(ar.get_dtype_name(block))
    if dtype.kind in "fc":
        real = np.empty(0, dtype=dtype).real.dtype
        dtype = real if coeff.dtype.kind == "f" else np.result_type(real, np.complex64)
    else:
        dtype = np.result_type(dtype, coeff.dtype)
    return ar.do("array", coeff.astype(dtype, copy=False), like=block)


@cache
def batch_plan(
    structure: TensorStructure,
    perm: tuple[int, ...],
    terms: tuple[tuple[int, int, complex], ...],
) -> tuple[
    tuple[tuple[tuple[int, ...], tuple[tuple[Any, Any, int, tuple[int, ...]], ...]], ...],
    tuple[tuple[int, int, complex], ...],
]:
    """``terms`` regrouped into array operations, plus the terms left for the loop.

    Two nested groupings, both pure functions of the plan and so computed once and cached
    with it. The outer one is by **destination block shape**: ``perm`` fixes a source's
    shape from its destination's, so a shape bucket's sources stack into one array and
    transpose once. The inner one is by **multiplicity** — how many terms a destination
    receives — so that each bucket is a rectangle ``(destinations, width)`` that gathers,
    scales and sums with no padding and no scatter primitive, which is what makes the
    execution identical on NumPy, JAX and PyTorch.

    Terms keep their plan order inside a destination, and a destination's terms never
    straddle two buckets, so the additions happen in the order the loop would do them.

    Parameters
    ----------
    structure : TensorStructure
        The plan's ``new_structure``; read for its block shapes.
    perm : tuple of int
        The plan's single per-block axis permutation.
    terms : tuple of (int, int, complex)
        ``(source block, target block, coefficient)``.

    Returns
    -------
    groups : tuple
        One ``(sources, buckets)`` per shape group, where ``buckets`` is a tuple of
        ``(take, coeff, width, destinations)``: ``take`` indexes the stacked sources,
        ``coeff`` broadcasts against a moved block, and reshaping to
        ``(len(destinations), width, ...)`` puts each destination's terms on one row.
    loose : tuple of (int, int, complex)
        The terms whose bucket was not worth batching. Regrouped, not in plan order — a
        destination's terms stay together and in their plan order, which is all the
        accumulation depends on.

    Notes
    -----
    Cached on the plan, and a plan's index arrays are smaller than the plan: two
    ``intp``/float columns of one entry per term against a tuple of three-element tuples.
    """
    shapes = structure.block_shapes
    ndim = len(perm)
    order = sorted(range(len(terms)), key=lambda i: (shapes[terms[i][1]], terms[i][1]))

    groups: list[Any] = []
    loose: list[tuple[int, int, complex]] = []
    start = 0
    while start < len(order):
        shape = shapes[terms[order[start]][1]]
        stop = start
        while stop < len(order) and shapes[terms[order[stop]][1]] == shape:
            stop += 1

        runs: dict[int, list[list[int]]] = {}
        run: list[int] = []
        for i in (*order[start:stop], None):  # type: ignore[misc]
            if run and (i is None or terms[i][1] != terms[run[0]][1]):
                runs.setdefault(len(run), []).append(run)
                run = []
            if i is not None:
                run.append(i)

        buckets = []
        sources: dict[int, int] = {}
        for width, rows in sorted(runs.items()):
            if width < 2 or len(rows) < MIN_BATCH_ROWS:
                loose.extend(terms[i] for row in rows for i in row)
                continue
            flat = [i for row in rows for i in row]
            for i in flat:
                sources.setdefault(terms[i][0], len(sources))
            take = np.array([sources[terms[i][0]] for i in flat], dtype=np.intp)
            coeff = np.array([terms[i][2] for i in flat])
            if not coeff.imag.any():
                coeff = coeff.real
            buckets.append(
                (
                    take,
                    coeff.reshape((-1, *(1,) * ndim)),
                    width,
                    tuple(terms[r[0]][1] for r in rows),
                )
            )
        if buckets:
            groups.append((tuple(sources), tuple(buckets)))
        start = stop

    return tuple(groups), tuple(loose)
