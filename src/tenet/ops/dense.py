"""The dense boundary — ``to_dense`` and ``from_dense``.

``T = Σ_τ A^(τ) ⊗ C^(τ)``, in both directions, with the categorical half
precomputed once per :class:`~tenet.structure.TensorStructure` and the numerical
half done entirely through ``ar.do`` — so a JAX-backed tensor densifies *inside*
``jit`` and differentiates like any other composition of ``einsum``, ``reshape``
and ``concatenate``.

Layout contract, depended on by twenty test modules: axis
``i`` has length ``legs[i].space.dim``; sectors occupy contiguous slabs in the
space's canonical order; within sector ``a``'s slab the index is ``alpha * d_a + m``.

**Mechanism.** The dense array is a *grid of slabs* indexed by one space sector
per public axis, and the whole grid is walked *at once*: :func:`_expansion` resolves
it to three static index/coefficient vectors, and ``to_dense`` applies them as one
gather out of the tensor's own matrices, one multiply by the Clebsch-Gordan
coefficients, one sum per fusion multiplicity, and one gather into dense position.
A *gather*, not a scatter, so there is still no ``zeros`` + write and no ``at[].add``
(JAX-only spelling), and no data-dependent skip of an all-zero block (invariant 9).
Cells that carry no fusion channel are not built at all: they read one shared
leading zero, which is where the deleted ``block.any()`` skip's saving really lived.

It used to be one ``ar.do`` per grid cell per axis level — a cell built by
``einsum`` and the grid glued by nested ``concatenate``, symmray's
``_to_dense_abelian`` design. That is a cheap call per cell on NumPy, but on JAX
every distinct shape is a fresh XLA module and a symmetric tensor's cells are all
different shapes: a five-leg SU(2) tensor with three sectors per leg (117 occupied
cells, 207 blocks) compiled **420 executables in 3.4 s**, against **39 in 0.38 s**
here, and NumPy went 5.7 ms to 4.1 ms per call rather than paying for the change.
The number of backend calls is now set by the number of *distinct fusion
multiplicities*, which is a handful, instead of by the size of the grid.

The index arithmetic is layout-only — charges, degeneracies, slab offsets — so it
is done once in NumPy and cached on the structure. Reading the blocks out of the
matrices is part of it, so ``to_dense`` no longer goes through
[blocks][tenet.SymmetricTensor.blocks]: that cut is per block and pays the same
per-shape compile toll on JAX, and expressed as indices it folds into the gather
that was happening anyway.

**NumPy.** This is the one module in ``ops/`` besides ``map.py``'s ``eye`` that
imports NumPy, and correctly so: the :class:`~tenet.symmetry.base.ClebschGordanData`
protocol *returns* ``np.ndarray`` (``cgc``, ``z_matrix``), so the coefficient
data is NumPy by contract. The blocks never are — they are only ever touched by
``ar.do``, and the plan's arrays reach the backend per call through
``ar.do("asarray", ..., like=block)`` (constant-folded under ``jit``; caching
per-backend copies would pin device buffers alive behind an uncleared cache).
``asarray`` rather than ``array`` because the expansion vectors are the size of the
dense result and are only ever read, so NumPy must not copy them.

**The adjoint is a pseudo-inverse, not a conjugate transpose.** Stacking a
cell's ``K`` CG tensors into ``C`` of shape ``(K, D)``, the Gram matrix
``C Cᴴ`` is *diagonal* with diagonal ``qdim(c_k)``: the fusion basis is
orthogonal but not orthonormal. That is the same quantum-dimension weight
``ops.basic.norm`` carries, seen from the other side of the identity
``‖T‖ = ‖to_dense(T)‖_F``. So the adjoint is ``Cᴴ diag(1/qdim)`` — spelled
``np.linalg.pinv(C)``, which agrees with the closed form for every shipped
provider (pinned by ``tests/ops/test_dense.py``) and stays correct for a future
provider whose gauge is not orthogonal.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Protocol

import autoray as ar
import numpy as np

from tenet.cache import plan_cache
from tenet.fusion_tree import FusionTree
from tenet.leg import Leg
from tenet.ops.batch import _block_positions
from tenet.structure import TensorStructure
from tenet.symmetry.base import (
    CapabilityError,
    ClebschGordanData,
    DualBasis,
    Sector,
    _DualFusionRules,
    requires,
)

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["PROJECT", "Cell", "DensePlan", "dense_plan", "from_dense", "to_dense"]

PROJECT: float = math.inf
"""The ``atol`` that means "project onto the symmetric subspace, do not check".

Pass it wherever an ``atol`` is accepted — [from_dense][tenet.SymmetricTensor.from_dense],
[restrict][tenet.restrict], [to_symmetry][tenet.to_symmetry] — to skip the concrete-value
check those functions otherwise run, which is what makes them traceable:

>>> import math, tenet
>>> tenet.PROJECT is math.inf
True

It **is** ``math.inf``, not a distinct object: infinity is the limit of a tolerance, "any
residual acceptable", so the mode and the tolerance value coincide and every existing
``atol=math.inf`` call site keeps working unchanged. The name exists because a call site
reading ``atol=tenet.PROJECT`` says which of the two operations is happening, while one
reading ``atol=math.inf`` requires the reader to know the idiom.
"""

Array = Any


@dataclass(frozen=True, slots=True, eq=False)
class Cell:
    """One sector tuple ``(a_0, ..., a_{N-1})`` of the dense grid.

    ``matrix`` is the ``(K, D)`` stack of this cell's CG tensors flattened over
    the dense (irrep) indices; ``cgts`` are the same rows reshaped to
    ``(d_0, ..., d_{N-1})`` (views, not copies) and ``adjoint`` is ``pinv(matrix)``.
    """

    sectors: tuple[Sector, ...]
    offsets: tuple[int, ...]
    degens: tuple[int, ...]  # m_i
    dims: tuple[int, ...]  # d_i
    block_indices: tuple[int, ...]
    cgts: tuple[np.ndarray, ...]
    matrix: np.ndarray
    adjoint: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        """``(m_0 d_0, ..., m_{N-1} d_{N-1})`` — the cell's dense extent."""
        return tuple(m * d for m, d in zip(self.degens, self.dims, strict=True))

    @property
    def slabs(self) -> tuple[slice, ...]:
        """Where the cell sits in the dense array, as basic slices."""
        return tuple(slice(o, o + s) for o, s in zip(self.offsets, self.shape, strict=True))


@dataclass(frozen=True, slots=True, eq=False)
class DensePlan:
    """The categorical half of densification: NumPy CG arrays plus offsets.

    NOT a field of :class:`~tenet.structure.TensorStructure` — invariant 8 keeps
    arrays out of structural metadata. It lives in the module-level
    :func:`dense_plan` cache, exactly as ``permutation_plan``, ``map_layout``
    and ``fusion_plan`` do.
    """

    structure: TensorStructure
    axis_sectors: tuple[tuple[Sector, ...], ...]
    """Per axis, the space's sectors in canonical order — the grid's axes."""
    axis_sizes: tuple[tuple[int, ...], ...]
    """Per axis, ``m_a * d_a`` for each of those sectors."""
    cells: tuple[Cell, ...]
    """One per *occupied* sector tuple; every other grid cell is exact zeros."""

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(sum(sizes) for sizes in self.axis_sizes)

    def cell_map(self) -> dict[tuple[Sector, ...], Cell]:
        return {c.sectors: c for c in self.cells}


def _refuse_dual(provider: _DualFusionRules, axis: int) -> None:
    """Turn the bare ``DualBasis`` failure into a message a user can act on."""
    try:
        requires(provider, DualBasis)
    except CapabilityError as exc:
        raise CapabilityError(
            f"to_dense: axis {axis} has dual=True, and provider {provider.name} does not "
            "implement DualBasis. Expanding a dual leg needs the Z-isomorphism "
            "V_a -> V_a^* in the dense basis, which carries the Frobenius-Schur "
            "sign of the sector. Trivial and U(1) supply it (one-dimensional irreps, "
            "Z = [[1]]); SU(2) supplies the antidiagonal (-1)^(j-m) matrix"
        ) from exc


class _DenseCapable(_DualFusionRules, ClebschGordanData, DualBasis, Protocol):
    """What :func:`_tree_cgt` actually calls: fusion data plus CG tensors plus
    the Z-isomorphism. ``ClebschGordanData`` alone understated it — the dual-leg
    branch reads ``dual`` and ``z_matrix`` (runtime-guarded by
    ``_check_capabilities``/``_refuse_dual`` before any tree is expanded)."""


def _tree_cgt(provider: _DenseCapable, tree: FusionTree, duals: tuple[bool, ...]) -> np.ndarray:
    """A tree's CG tensor, shape ``(d_u0, ..., d_u{N-1}, d_coupled)``.

    Left-associated contraction along the spine. Rank 0 is the unit's ``(1,)``
    (so an empty side contracts like any other) and rank 1 is the identity.

    ``duals[i]`` says whether the leg feeding uncoupled line ``i`` is ``dual``.
    Where it is, the tree's label is ``dual(a)`` but the dense axis must run over
    ``V_a``, so the provider's ``Z_a: V_a -> V_a^*`` is contracted onto that axis.

    The ``d_a > 1`` path is covered: SU(2)'s ``Z`` is the antidiagonal
    ``(-1)**i``, so a dual leg's dense slab is the direct one with the magnetic
    index reversed and alternately signed, and the placement (together with the
    ``.conj()`` the caller applies to the input tree) is pinned by the cup/cap
    oracle in ``tests/symmetry/test_su2_dual.py``.
    """
    dim = provider.irrep_dim
    if tree.rank == 0:
        return np.ones((dim(tree.coupled),))
    x = np.eye(dim(tree.uncoupled[0]))
    for e, u, f, mu in tree.vertices():
        x = np.tensordot(x, provider.cgc(e, u, f)[..., mu], axes=([-1], [0]))
    for i, is_dual in enumerate(duals):
        if is_dual:
            z = provider.z_matrix(provider.dual(tree.uncoupled[i]))
            x = np.moveaxis(np.tensordot(x, z, axes=([i], [1])), -1, i)
    return x


def _check_capabilities(structure: TensorStructure) -> None:
    """``ClebschGordanData`` for every provider, ``DualBasis`` for a ``dual`` leg.

    Raised before any block or dense element is read, and identically inside and
    outside ``jit``: these are questions about the legs, not about the numbers.
    """
    provider = structure.provider
    requires(provider, ClebschGordanData)
    for axis, leg in enumerate(structure.legs):
        if leg.dual:
            _refuse_dual(provider, axis)
            break


def _dense_cost(plan: DensePlan) -> int:
    """The plan's Clebsch-Gordan bytes in ``tenet.cache`` units (156 bytes per term).

    ``cgts`` are views into ``matrix``, so only ``matrix`` and ``adjoint`` are counted.
    This is the one cost function whose values are floats rather than Python tuples,
    which is why it converts through bytes instead of counting entries.
    """
    return sum(cell.matrix.nbytes + cell.adjoint.nbytes for cell in plan.cells) // 156


@plan_cache(cost=_dense_cost)
def dense_plan(structure: TensorStructure) -> DensePlan:
    """The dense grid of ``structure``: cells, offsets and NumPy CG tensors.

    Cached on the structure alone. No ``provider_key`` component is needed and
    one would be dead weight: the structure holds ``Leg``s holding
    ``GradedSpace``s holding the frozen provider *value*, so provider identity —
    and with it the gauge — is already in the key.

    """
    _check_capabilities(structure)
    provider = structure.provider
    legs = structure.legs
    n = structure.ndim
    duals = tuple(leg.dual for leg in legs)
    out_axes, in_axes = structure.out_axes, structure.in_axes
    order = (*out_axes, *in_axes)
    # `_tree_cgt` builds axes in (out..., in...) order; store them in public axis
    # order instead, so every downstream index list is just `range(n)`.
    to_public = tuple(sorted(range(n), key=order.__getitem__))

    grouped: dict[tuple[Sector, ...], list[tuple[int, np.ndarray]]] = {}
    for i, key in enumerate(structure.block_order):
        # _check_capabilities(structure) above proved ClebschGordanData (and
        # DualBasis for any dual leg); a raise-based check does not narrow
        xout = _tree_cgt(provider, key.output_tree, tuple(duals[a] for a in out_axes))  # ty: ignore[invalid-argument-type]
        xin = _tree_cgt(provider, key.input_tree, tuple(duals[a] for a in in_axes))  # ty: ignore[invalid-argument-type]
        cgt = np.tensordot(xout, xin.conj(), axes=([-1], [-1]))
        grouped.setdefault(structure.axis_sectors(key), []).append(
            (i, np.transpose(cgt, to_public))
        )

    cells = []
    for sectors, entries in grouped.items():
        degens = tuple(leg.degeneracy(a) for leg, a in zip(legs, sectors, strict=True))
        dims = tuple(provider.irrep_dim(a) for a in sectors)  # ty: ignore[unresolved-attribute]  # see above
        offsets = tuple(leg.space.sector_offset(a) for leg, a in zip(legs, sectors, strict=True))
        matrix = np.stack([cgt.reshape(-1) for _, cgt in entries])
        cells.append(
            Cell(
                sectors=sectors,
                offsets=offsets,
                degens=degens,
                dims=dims,
                block_indices=tuple(i for i, _ in entries),
                cgts=tuple(row.reshape(dims) for row in matrix),
                matrix=matrix,
                adjoint=np.linalg.pinv(matrix),
            )
        )

    axis_sectors = tuple(leg.sectors for leg in legs)
    axis_sizes = tuple(
        tuple(leg.degeneracy(a) * provider.irrep_dim(a) for a in sectors)  # ty: ignore[unresolved-attribute]
        for leg, sectors in zip(legs, axis_sectors, strict=True)
    )
    return DensePlan(
        structure=structure,
        axis_sectors=axis_sectors,
        axis_sizes=axis_sizes,
        cells=tuple(cells),
    )


@dataclass(frozen=True, slots=True, eq=False)
class _Expansion:
    """Densification as three static vectors and a table — the grid walk, precomputed.

    ``source`` and ``weights`` are equally long: term ``k`` of the expansion is
    ``matrices_flat[source[k]] * weights[k]``, where ``matrices_flat`` is
    ``concatenate([m.reshape(-1) for m in t.data])``. The terms run cell by cell
    with the cells grouped by fusion multiplicity, so ``groups`` — one
    ``(multiplicity, elements)`` pair per group, in order — cuts the term vector
    into ``(multiplicity, elements)`` rectangles whose rows are summed to give the
    cells' dense values. ``placement`` then reads the dense array out of
    ``concatenate((zeros(1), *those sums))``, a symmetry-forbidden element taking
    the leading zero.
    """

    source: np.ndarray
    weights: np.ndarray
    groups: tuple[tuple[int, int], ...]
    placement: np.ndarray


def _expansion_cost(exp: _Expansion) -> int:
    """The expansion's bytes in ``tenet.cache`` units, as ``_dense_cost`` counts them."""
    return (exp.source.nbytes + exp.weights.nbytes + exp.placement.nbytes) // 156


@plan_cache(cost=_expansion_cost)
def _expansion(structure: TensorStructure) -> _Expansion:
    """The dense grid walk of ``structure``, resolved to indices once.

    Parameters
    ----------
    structure : TensorStructure
        The structure to expand.

    Returns
    -------
    _Expansion
        The vectors [to_dense][tenet.ops.dense.to_dense] applies.

    Notes
    -----
    Pure layout arithmetic — charges, degeneracies, slab offsets and the
    Clebsch-Gordan coefficients the plan already holds — so it is NumPy, and it is
    cached on the structure exactly as [dense_plan][tenet.ops.dense.dense_plan] is.
    Separately from it, because ``from_dense`` shares the plan and does not want
    this: it is the size of the dense array, and building it is the price of the
    walk that ``to_dense`` no longer takes per cell.

    The cells are grouped by multiplicity rather than padded to the largest, so the
    term vector is exactly as long as the sum over cells of ``multiplicity × cell
    size`` — the work that is actually there — and the number of backend calls is
    set by the number of *distinct* multiplicities, which is a handful.
    """
    plan = dense_plan(structure)
    positions = _block_positions(structure)
    cells = sorted(plan.cells, key=lambda cell: len(cell.block_indices))

    source: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    groups: list[tuple[int, int]] = []
    # np.intp throughout: NumPy converts any narrower index type on every gather,
    # and JAX narrows to int32 itself when x64 is off
    index = np.zeros(plan.shape, dtype=np.intp)
    base = 1
    start = 0
    while start < len(cells):
        mult = len(cells[start].block_indices)
        stop = start
        while stop < len(cells) and len(cells[stop].block_indices) == mult:
            stop += 1
        run = cells[start:stop]
        start = stop
        groups.append((mult, sum(math.prod(cell.shape) for cell in run)))
        for j in range(mult):
            for cell in run:
                # the cell's dense index is alpha * d + m per axis: broadcasting the
                # block over the irrep axes and the CG tensor over the degeneracy axes
                # spells that interleave without an einsum
                interleaved = tuple(
                    x for pair in zip(cell.degens, cell.dims, strict=True) for x in pair
                )
                block = positions[cell.block_indices[j]].reshape(
                    tuple(x for m in cell.degens for x in (m, 1))
                )
                cgt = cell.cgts[j].reshape(tuple(x for d in cell.dims for x in (1, d)))
                source.append(np.broadcast_to(block, interleaved).reshape(-1))
                weights.append(np.broadcast_to(cgt, interleaved).reshape(-1))
        for cell in run:
            n = math.prod(cell.shape)
            index[cell.slabs] = np.arange(base, base + n, dtype=index.dtype).reshape(cell.shape)
            base += n

    return _Expansion(
        source=np.concatenate(source),
        weights=np.concatenate(weights),
        groups=tuple(groups),
        placement=index.reshape(-1),
    )


def to_dense(t: "SymmetricTensor") -> Array:
    """``T = Σ_τ A^(τ) ⊗ C^(τ)`` expanded into a plain dense array of ``t``'s backend.

    Explicit by design (invariant 9). Requires ``ClebschGordanData``; a leg with
    ``dual=True`` additionally requires ``DualBasis``, the provider's
    ``V_a -> V_a^*`` isomorphism in the dense basis.

    The dtype is whatever the backend's own promotion makes of ``block × CG``:
    on NumPy that is ``result_type(block.dtype, float64)``; on JAX without
    ``jax_enable_x64`` the CG array arrives as
    ``float32``, the same stance ``to_backend`` documents.
    """
    plan = dense_plan(t.structure)
    if not t.data:
        return np.zeros(plan.shape, dtype=np.float64)
    ref = t.data[0]
    exp = _expansion(t.structure)

    # every term of Σ_τ A^(τ) ⊗ C^(τ), for the whole grid at once: one gather out of
    # the tensor's own matrices, times the Clebsch-Gordan factors the plan holds.
    # ``asarray``, not ``array``: the two index arrays and the coefficient vector are
    # the size of the dense result, and only ever read, so NumPy must not copy them
    flat = ar.do("concatenate", tuple(ar.do("reshape", m, (-1,)) for m in t.data))
    terms = ar.do("take", flat, ar.do("asarray", exp.source, like=flat), axis=0) * ar.do(
        "asarray", exp.weights, like=ref
    )

    # a group's terms are (multiplicity, elements); summing the rows in order is the
    # `acc = acc + full` the per-cell walk did, on every cell of the group at once
    values = [ar.do("zeros", (1,), like=ref)]
    at = 0
    for mult, size in exp.groups:
        rows = ar.do("reshape", terms[at : at + mult * size], (mult, size))
        acc = rows[0]
        for j in range(1, mult):
            acc = acc + rows[j]
        values.append(acc)
        at += mult * size

    cells = ar.do("concatenate", tuple(values))
    index = ar.do("asarray", exp.placement, like=cells)
    return ar.do("reshape", ar.do("take", cells, index, axis=0), plan.shape)


def from_dense(
    dense: Array, legs: Sequence[Leg], *, atol: float | None = None
) -> "SymmetricTensor":
    """Project a dense carrier-basis array onto the symmetric subspace of ``legs``.

    The exact inverse of :func:`to_dense`: every cell of the dense grid is sliced
    out and contracted against the cached pseudo-inverse of its CG stack.

    ``legs`` cannot be inferred — a dense array carries no categorical
    information at all (which sector an index belongs to, which axes are
    ``dual``, which are IN) — so it is given.

    Input that is not symmetric to ``atol`` is **refused**, never silently
    projected: the residual is ``sqrt(‖dense‖² − ‖reproduced‖²)`` accumulated
    over the cells *plus* the entire mass of the symmetry-forbidden cells, which
    is the part a per-cell check alone would miss. ``atol`` defaults to
    ``sqrt(eps(dtype)) * ‖dense‖`` — relative, because an absolute tolerance is
    meaningless for an unnormalized tensor.

    The check is a concrete-value question, so it raises JAX's own
    ``ConcretizationTypeError`` under a trace. The projection itself is pure
    slicing, ``reshape``, ``transpose`` and one matmul, hence traceable and
    differentiable: ``atol=tenet.PROJECT`` is the spelling for "project, don't
    check", and it is the form that goes inside ``jit``. [PROJECT][tenet.PROJECT]
    is exactly ``math.inf``, so ``atol=math.inf`` is the same call.
    """
    from tenet.tensor import SymmetricTensor

    structure = TensorStructure(tuple(legs))
    _check_capabilities(structure)
    plan = dense_plan(structure)
    if tuple(ar.do("shape", dense)) != plan.shape:
        raise ValueError(
            f"from_dense: array of shape {tuple(ar.do('shape', dense))} does not match "
            f"the legs' dense shape {plan.shape}"
        )

    n = structure.ndim
    # (m_0, d_0, m_1, d_1, ...) -> (m_0, m_1, ..., d_0, d_1, ...)
    interleave = tuple(range(0, 2 * n, 2)) + tuple(range(1, 2 * n, 2))

    blocks: list[Array] = [None] * structure.num_blocks  # type: ignore[list-item]
    residuals: dict[tuple[Sector, ...], Array] = {}
    for cell in plan.cells:
        piece = dense[cell.slabs]  # basic slicing: every backend spells it alike
        interleaved = tuple(x for pair in zip(cell.degens, cell.dims, strict=True) for x in pair)
        piece = ar.do("reshape", piece, interleaved)
        piece = ar.do("transpose", piece, interleave)
        mat = ar.do("reshape", piece, (math.prod(cell.degens), math.prod(cell.dims)))
        coeffs = mat @ ar.do("array", cell.adjoint, like=mat)
        for k, i in enumerate(cell.block_indices):
            blocks[i] = ar.do("reshape", coeffs[:, k], cell.degens)
        if atol != PROJECT:
            # what the cell's CG span could not reproduce, taken as a difference
            # rather than as ‖mat‖² − ‖repro‖²: Pythagoras holds exactly in real
            # arithmetic but cancels to sqrt(eps) in floating point, which is the
            # same size as the default tolerance it would be compared against
            residuals[cell.sectors] = mat - coeffs @ ar.do("array", cell.matrix, like=mat)

    if atol != PROJECT:
        _check_symmetric(dense, plan, residuals, atol)
    return SymmetricTensor(structure, tuple(blocks))


def _sum_squares(x: Array) -> float:
    return float(ar.do("sum", ar.do("abs", x) ** 2))


def _check_symmetric(
    dense: Array,
    plan: DensePlan,
    residuals: dict[tuple[Sector, ...], Array],
    atol: float | None,
) -> None:
    """Refuse a dense array that is not symmetric, naming the worst cell.

    Every grid cell contributes: an occupied one its projection residual, a
    symmetry-forbidden one its whole mass (there is no channel it could live in).
    Concrete values throughout — under a JAX trace this is where
    ``ConcretizationTypeError`` comes from, correctly and in JAX's own voice.
    """
    total = _sum_squares(dense)
    cells = plan.cell_map()
    worst: tuple[float, tuple[Sector, ...]] = (0.0, ())
    residual_sq = 0.0
    for index in product(*(range(len(s)) for s in plan.axis_sectors)):
        sectors = tuple(plan.axis_sectors[ax][i] for ax, i in enumerate(index))
        cell = cells.get(sectors)
        if cell is None:  # forbidden: no channel it could live in, so all of it
            slabs = tuple(
                slice(sum(plan.axis_sizes[ax][:i]), sum(plan.axis_sizes[ax][: i + 1]))
                for ax, i in enumerate(index)
            )
            here = _sum_squares(dense[slabs])
        else:
            here = _sum_squares(residuals[sectors])
        residual_sq += here
        if here > worst[0]:
            worst = (here, sectors)

    residual = math.sqrt(residual_sq)
    if atol is None:
        # ar.get_dtype_name, not `.dtype`: a torch tensor's dtype is not a NumPy
        # one and `np.promote_types(torch.float64, ...)` raises (#95). Same
        # spelling as ops/map.py::compose.
        dtype = np.promote_types(ar.get_dtype_name(dense), np.float32)
        atol = math.sqrt(float(np.finfo(dtype).eps)) * math.sqrt(total)
    if residual > atol:
        raise ValueError(
            f"from_dense: array is not symmetric — residual {residual:.6g} exceeds "
            f"atol {atol:.6g}; the worst sector tuple is {worst[1]} "
            f"(mass {math.sqrt(worst[0]):.6g}). Pass a larger atol to accept it, or "
            "atol=tenet.PROJECT (== math.inf) to project without checking"
        )
