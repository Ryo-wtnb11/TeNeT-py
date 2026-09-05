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

from collections.abc import Callable
from typing import Any

import autoray as ar
import numpy as np

from tenet.cache import plan_cache
from tenet.structure import TensorStructure

__all__ = ["band_runs", "band_scale", "batch_plan", "cast_coefficients", "scale_bands"]


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
    device = {"device": block.device} if ar.infer_backend(block) == "torch" else {}
    return ar.do("asarray", coeff.astype(dtype, copy=False), like=block, **device)


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


def _block_positions(structure: TensorStructure) -> list[np.ndarray]:
    """Per block, where each of its public-order elements sits in the flat matrices.

    ``map_view.views`` reaches a block as ``transpose(reshape(mat[rows, cols], shape),
    inverse)``; this is that same cut expressed as indices instead of as a view, so a
    caller that wants *all* the blocks at once can take them in one gather.
    """
    from tenet.map_view import _slots, _tables, map_layout

    layout = map_layout(structure)
    _, shapes = _tables(structure)
    inverse = tuple(sorted(range(structure.ndim), key=layout.axes_order.__getitem__))
    bases, base = [], 0
    for rows, cols in layout.shapes:
        bases.append((base, cols))
        base += rows * cols
    out = []
    for i, (pos, ro, dr, co, dc) in enumerate(_slots(structure)):
        start, ncols = bases[pos]
        rows = start + (ro + np.arange(dr, dtype=np.intp)) * ncols
        flat = (rows[:, None] + (co + np.arange(dc, dtype=np.intp))[None, :]).reshape(shapes[i])
        out.append(np.transpose(flat, inverse))
    return out


def _indexed_cost(plan: Any) -> int:
    """Index/coefficient bytes in the cache's 156-byte term units."""
    if plan is None:
        return 1
    groups, order = plan
    size = order.nbytes + sum(i.nbytes + (0 if c is None else c.nbytes) for i, c in groups)
    return max(1, (size + 155) // 156)


@plan_cache(cost=_indexed_cost)
def _indexed_plan(
    source: TensorStructure,
    target: TensorStructure,
    perm: tuple[int, ...],
    terms: tuple[tuple[int, int, complex], ...],
) -> tuple[tuple[Any, ...], np.ndarray] | None:
    """Flat gather indices, coefficients and placement for a complete linear plan.

    Group by the number of summands, independent of block shape: every destination
    element in a group has the same number of sources. Term order is preserved.
    These arrays are a numerical side table, never fields of TensorStructure.

    The table grows with *elements times multiplicity*. Decline before allocating
    if it would exceed this cache's byte budget; the shape-bucket executor then
    keeps indices proportional to blocks instead. This also avoids rebuilding an
    oversized, uncacheable table on every call.
    """
    from tenet.map_view import _normalized, _slots, map_layout

    normalized = _normalized(terms)
    if normalized is None:
        return None
    terms = normalized
    dst = map_layout(target)
    dst_slots = _slots(target)
    size = sum(r * c for r, c in dst.shapes)
    elements = sum(dst_slots[d][2] * dst_slots[d][4] for _, d, _ in terms)
    if 16 * elements + 8 * size > 156 * _indexed_plan.budget:
        return None
    axes = tuple(perm[i] for i in dst.axes_order)
    positions = _block_positions(source)
    sources = {i: positions[i].transpose(axes).ravel() for i in {s for s, _, _ in terms}}
    target_positions = _block_positions(target)
    destinations: dict[int, list[tuple[int, complex]]] = {}
    for i, d, coeff in terms:
        destinations.setdefault(d, []).append((i, coeff))
    if len(destinations) != target.num_blocks:
        raise ValueError(
            f"lower_plan: the plan fills {len(destinations)} of {target.num_blocks} target "
            "blocks -- provider coefficients dropped terms"
        )
    by_width: dict[int, list[Any]] = {}
    for d, row in destinations.items():
        by_width.setdefault(len(row), []).append((d, row))
    groups, writes = [], []
    for rows in by_width.values():
        reads, coefficients = [], []
        for d, row in rows:
            take = np.stack([sources[i] for i, _ in row])
            reads.append(take)
            coefficients.append(np.broadcast_to(np.array([c for _, c in row])[:, None], take.shape))
            writes.append(target_positions[d].transpose(dst.axes_order).ravel())
        coeff = np.concatenate(coefficients, axis=1)
        groups.append((np.concatenate(reads, axis=1), None if np.all(coeff == 1) else coeff))
    order = np.empty(size, dtype=np.intp)
    if writes:
        order[np.concatenate(writes)] = np.arange(order.size)
    return tuple(groups), order


@plan_cache(cost=lambda vec: len(vec))
def _repeated(runs: tuple[tuple[complex, int], ...]) -> np.ndarray:
    """``(coefficient, extent)`` per band, expanded to one coefficient per index."""
    values = [coeff for coeff, _ in runs]
    band = (
        np.array([coeff.real for coeff in values], dtype=float)
        if all(coeff.imag == 0 for coeff in values)
        else np.array(values, dtype=complex)
    )
    return np.repeat(band, [extent for _, extent in runs])


def band_scale(runs: tuple[tuple[complex, int], ...], mat: Any, axis: int) -> Any:
    """One coefficient per band of a coupled-sector matrix, ready to multiply ``mat`` by.

    Parameters
    ----------
    runs : tuple of (complex, int)
        ``(coefficient, extent)`` per band of ``axis``, in band order; the extents sum
        to ``mat``'s length along it.
    mat : array
        The coupled-sector matrix the result multiplies; supplies backend and dtype.
    axis : int
        0 to scale rows, 1 to scale columns.

    Returns
    -------
    array
        A backend-native vector shaped to broadcast against ``mat`` along ``axis``.

    Notes
    -----
    The numerical half of a *band* plan, kept here for the reason the index arrays above
    are: a plan stays array-free Python metadata, and the arrays that execute it are a
    cached side table hanging off it. The expansion is cached because it is a pure
    function of the runs, which are per structure and not per tensor; the cast is not,
    because it depends on the operand's dtype.

    A diagonal scaling is the shape every *relabelling* operation reduces to -- a scalar
    per fusion tree, which is a scalar per band -- and multiplying the matrix by this is
    what keeps such an operation from cutting its operand into blocks (invariant 8).
    Real where it can be, so a real tensor stays real, which is
    [cast_coefficients][tenet.ops.batch.cast_coefficients]'s rule and not a new one.
    """
    vec = _repeated(runs)
    return cast_coefficients(vec[:, None] if axis == 0 else vec[None, :], mat)


Runs = tuple[tuple[complex, int], ...]
"""``(coefficient, extent)`` per band of one side of one coupled-sector matrix."""

Scales = tuple[tuple[Runs | None, Runs | None], ...]
"""One ``(row runs, column runs)`` pair per coupled sector, in ``sectors`` order."""


def band_runs(
    bands: tuple[tuple[Any, int, int], ...], coefficient: Callable[[Any], complex]
) -> Runs | None:
    """``coefficient`` of each band's fusion tree, paired with the band's extent.

    Parameters
    ----------
    bands : tuple of (FusionTree, int, int)
        One side's ``(tree, offset, extent)`` bands, as
        [MapLayout.row_bands][tenet.MapLayout.row_bands] gives them.
    coefficient : callable
        The scalar this operation pays on a block whose tree on this side is the given
        one. Called once per band, on static metadata, never on an array.

    Returns
    -------
    tuple of (complex, int), or None
        The runs [band_scale][tenet.ops.batch.band_scale] expands, or ``None`` when
        every coefficient is ``1`` and the side needs no scaling at all.

    Notes
    -----
    ``None`` rather than a tuple of ones is what lets a bosonic grading skip the
    multiply entirely and keep the operand's own arrays, so those paths stay
    bit-identical to the ones that never scaled anything.
    """
    runs = tuple((coefficient(tree), extent) for tree, _, extent in bands)
    return None if all(coeff == 1 for coeff, _ in runs) else runs


def scale_bands(data: tuple[Any, ...], scales: Scales) -> tuple[Any, ...]:
    """``data`` with each coupled-sector matrix's row and column bands scaled.

    Parameters
    ----------
    data : tuple of array
        The coupled-sector matrices, in ``map_layout(structure).sectors`` order.
    scales : tuple of (Runs or None, Runs or None)
        One pair per matrix, from [band_runs][tenet.ops.batch.band_runs].

    Returns
    -------
    tuple of array
        The scaled matrices; a matrix whose pair is ``(None, None)`` comes back
        unchanged and uncopied.

    Notes
    -----
    The execution half of every operation whose whole content is one scalar per fusion
    block -- ``twist`` and ``flip_dual`` today. Such a scalar always *factorizes*: a leg
    on the OUT side reads the block's output tree only and a leg on the IN side its input
    tree only, so a per-block grid of scalars is a row scaling times a column scaling,
    and a diagonal scaling of a matrix is one array operation. That is what lets these
    operations read ``data`` instead of cutting every block out to multiply it by a
    number (invariant 8).
    """
    out = []
    for mat, (rows, cols) in zip(data, scales, strict=True):
        if rows is not None:
            mat = ar.do("multiply", mat, band_scale(rows, mat, 0))
        if cols is not None:
            mat = ar.do("multiply", mat, band_scale(cols, mat, 1))
        out.append(mat)
    return tuple(out)
