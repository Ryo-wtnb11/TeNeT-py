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

from tenet.cache import plan_cache
from tenet.map_view import Rect, _pools, _rects
from tenet.structure import TensorStructure

__all__ = ["batch_plan", "cast_coefficients", "coefficient_dtype", "recouple_plan"]


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
    dtype = coefficient_dtype(coeff.dtype, ar.get_dtype_name(block))
    return ar.do("array", coeff.astype(dtype, copy=False), like=block)


@cache
def coefficient_dtype(coeff: np.dtype, name: str) -> np.dtype:
    """The dtype [cast_coefficients][tenet.ops.batch.cast_coefficients] casts to.

    Parameters
    ----------
    coeff : numpy.dtype
        The coefficient column's own dtype.
    name : str
        The block dtype's autoray name.

    Returns
    -------
    numpy.dtype
        What the column has to be in for the multiply to promote the way a per-term
        Python scalar would.

    Notes
    -----
    Split out and cached because the answer depends on two dtypes and nothing else,
    while the caller asks it once per bucket: a recoupling of an SU(2) rank-6 bend at
    ragged degeneracies has 2,773 of them, and the ``np.result_type`` walk behind it
    costs more than the cast it decides.
    """
    dtype = np.dtype(name)
    if dtype.kind in "fc":
        real = np.empty(0, dtype=dtype).real.dtype
        return real if coeff.kind == "f" else np.result_type(real, np.complex64)
    return np.result_type(dtype, coeff)


@plan_cache(cost=lambda r: sum(len(t) for _, bs in r[0] for t, _, _, _ in bs) + len(r[1]))
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


Bucket = tuple[int, Any, Any, int, int]
"""``(pool, source rows, coefficients or None, terms per cell, cells)``."""


@plan_cache(
    cost=lambda plan: 1 if plan is None else sum(len(b[1]) for _, _, bs, _ in plan for b in bs)
)
def recouple_plan(
    source: TensorStructure,
    structure: TensorStructure,
    perm: tuple[int, ...],
    terms: tuple[tuple[int, int, complex], ...],
) -> tuple[tuple[int, Rect, tuple[Bucket, ...], Any], ...] | None:
    """``(perm, terms)`` regrouped so that one destination *rectangle* is built at a time.

    Parameters
    ----------
    source : TensorStructure
        The structure the plan reads; supplies the pools the gathers index.
    structure : TensorStructure
        The plan's ``new_structure``.
    perm : tuple of int
        The plan's single per-block axis permutation.
    terms : tuple of (int, int, complex)
        ``(source block, target block, coefficient)``.

    Returns
    -------
    tuple or None
        Per destination rectangle, ``(sector, rectangle, buckets, reorder)``: the
        [Bucket][tenet.map_view.Bucket]s that build its cells and the index array that
        puts their results back into cell order, or ``None`` where that is already the
        order. ``None`` for the whole plan where some target block receives no term at
        all -- the caller's fill check is then the one that reports it.

    Notes
    -----
    [batch_plan][tenet.ops.batch.batch_plan] groups by destination *shape*; this groups
    by destination *rectangle*, which is finer and is what makes the write side one
    strided store per rectangle instead of one reshape and one store per block. Within a
    rectangle the cells are grouped by multiplicity for the reason ``batch_plan`` gives
    -- a rectangle of equal multiplicities needs no padding and no segment-sum primitive
    -- and the groups are concatenated and reordered back, which is two array calls
    against the alternative of padding the sum with zeros and losing bit-identity.

    A destination's terms keep their plan order, and each is accumulated in that order,
    so the arithmetic is the loop's arithmetic and not merely close to it.
    """
    _, rows, pool_of = _pools(source)
    by_dst: dict[int, list[tuple[int, complex]]] = {}
    for src, dst, coeff in terms:
        by_dst.setdefault(dst, []).append((src, coeff))

    plan: list[tuple[int, Rect, tuple[Bucket, ...], Any]] = []
    for si, (_, rects) in enumerate(_rects(structure)):
        for rect in rects:
            cells = rect[7]
            widths: dict[int, list[int]] = {}
            for p, block in enumerate(cells):
                got = by_dst.get(block)
                if got is None:
                    return None  # a target block no term reaches; the caller says so
                widths.setdefault(len(got), []).append(p)

            buckets: list[Bucket] = []
            places: list[int] = []
            for width in sorted(widths):
                at = widths[width]
                places.extend(at)
                flat = [term for p in at for term in by_dst[cells[p]]]
                coeff = np.array([c for _, c in flat])
                if not coeff.imag.any():
                    coeff = coeff.real
                buckets.append(
                    (
                        pool_of[flat[0][0]],
                        np.fromiter((rows[s] for s, _ in flat), np.intp, len(flat)),
                        None if (coeff == 1).all() else coeff.reshape((-1, *(1,) * len(perm))),
                        width,
                        len(at),
                    )
                )
            plan.append(
                (
                    si,
                    rect,
                    tuple(buckets),
                    None if len(buckets) == 1 else np.argsort(np.array(places), kind="stable"),
                )
            )
    return tuple(plan)
