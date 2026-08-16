"""The dense boundary — ``to_dense`` and ``from_dense`` (issue #82).

``T = Σ_τ A^(τ) ⊗ C^(τ)``, in both directions, with the categorical half
precomputed once per :class:`~tenet.structure.TensorStructure` and the numerical
half done entirely through ``ar.do`` — so a JAX-backed tensor densifies *inside*
``jit`` and differentiates like any other composition of ``einsum``, ``reshape``
and ``concatenate``.

Layout contract, unchanged from #9 and depended on by twenty test modules: axis
``i`` has length ``legs[i].space.dim``; sectors occupy contiguous slabs in the
space's canonical order; within sector ``a``'s slab the index is ``alpha * d_a + m``.

**Mechanism.** The dense array is a *grid of slabs* indexed by one space sector
per public axis. Every cell is built independently and the grid is glued by
nested ``concatenate`` — symmray's ``_to_dense_abelian`` design. No ``zeros`` +
scatter, no ``at[].add`` (JAX-only spelling), no data-dependent skip of an
all-zero block (invariant 9). Cells that carry no fusion channel are exact
``zeros`` *by construction*, which is where the deleted ``block.any()`` skip's
saving really lived.

**NumPy.** This is the one module in ``ops/`` besides ``map.py``'s ``eye`` that
imports NumPy, and correctly so: the :class:`~tenet.symmetry.base.ClebschGordan`
protocol *returns* ``np.ndarray`` (``cgc``, ``z_matrix``), so the coefficient
data is NumPy by contract. The blocks never are — they are only ever touched by
``ar.do``, and the plan's arrays reach the backend per call through
``ar.do("array", cgt, like=block)`` (constant-folded under ``jit``; caching
per-backend copies would pin device buffers alive behind an uncleared cache).

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
import string
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from itertools import product
from typing import TYPE_CHECKING, Any

import autoray as ar
import numpy as np

from tenet.fusion_tree import FusionTree
from tenet.leg import Leg
from tenet.structure import TensorStructure
from tenet.symmetry.base import (
    CapabilityError,
    ClebschGordan,
    DualBasis,
    FusionProvider,
    Sector,
    requires,
)

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["Cell", "DensePlan", "dense_plan", "from_dense", "to_dense"]

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
    :func:`dense_plan` cache, exactly as ``permutation_plan`` (#21),
    ``map_layout`` (#29) and ``fusion_plan`` (#22) do.
    """

    structure: TensorStructure
    axis_sectors: tuple[tuple[Sector, ...], ...]
    """Per axis, the space's sectors in canonical order — the grid's axes."""
    axis_sizes: tuple[tuple[int, ...], ...]
    """Per axis, ``m_a * d_a`` for each of those sectors."""
    cells: tuple[Cell, ...]
    """One per *occupied* sector tuple; every other grid cell is exact zeros."""
    subscripts: str
    """The ``einsum`` equation gluing one block to one CG tensor, built once."""

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(sum(sizes) for sizes in self.axis_sizes)

    def cell_map(self) -> dict[tuple[Sector, ...], Cell]:
        return {c.sectors: c for c in self.cells}


def _refuse_dual(provider: FusionProvider, axis: int) -> None:
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


def _tree_cgt(provider: ClebschGordan, tree: FusionTree, duals: tuple[bool, ...]) -> np.ndarray:
    """A tree's CG tensor, shape ``(d_u0, ..., d_u{N-1}, d_coupled)``.

    Left-associated contraction along the spine. Rank 0 is the unit's ``(1,)``
    (so an empty side contracts like any other) and rank 1 is the identity.

    ``duals[i]`` says whether the leg feeding uncoupled line ``i`` is ``dual``.
    Where it is, the tree's label is ``dual(a)`` but the dense axis must run over
    ``V_a``, so the provider's ``Z_a: V_a -> V_a^*`` is contracted onto that axis.

    The ``d_a > 1`` path is covered as of #37: SU(2)'s ``Z`` is the antidiagonal
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
    """``ClebschGordan`` for every provider, ``DualBasis`` for a ``dual`` leg.

    Raised before any block or dense element is read, and identically inside and
    outside ``jit``: these are questions about the legs, not about the numbers.
    """
    provider = structure.provider
    requires(provider, ClebschGordan)
    for axis, leg in enumerate(structure.legs):
        if leg.dual:
            _refuse_dual(provider, axis)
            break


@cache
def dense_plan(structure: TensorStructure) -> DensePlan:
    """The dense grid of ``structure``: cells, offsets and NumPy CG tensors.

    Cached on the structure alone. No ``provider_key`` component is needed and
    one would be dead weight (#53): the structure holds ``Leg``s holding
    ``GradedSpace``s holding the frozen provider *value*, so provider identity —
    and with it the gauge — is already in the key.

    Simplification: unbounded ``functools.cache``, as every other plan cache. The
    ceiling is larger here than for the array-free plans — a plan holds
    ``Σ_cells K·Π d_i`` floats, i.e. every CG tensor of the structure at once —
    so swap in ``lru_cache(maxsize=...)`` if a workload ever densifies many
    distinct large SU(2) structures.
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
        xout = _tree_cgt(provider, key.output_tree, tuple(duals[a] for a in out_axes))
        xin = _tree_cgt(provider, key.input_tree, tuple(duals[a] for a in in_axes))
        cgt = np.tensordot(xout, xin.conj(), axes=([-1], [-1]))
        grouped.setdefault(structure.axis_sectors(key), []).append(
            (i, np.transpose(cgt, to_public))
        )

    cells = []
    for sectors, entries in grouped.items():
        degens = tuple(leg.degeneracy(a) for leg, a in zip(legs, sectors, strict=True))
        dims = tuple(provider.irrep_dim(a) for a in sectors)
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
        tuple(leg.degeneracy(a) * provider.irrep_dim(a) for a in sectors)
        for leg, sectors in zip(legs, axis_sectors, strict=True)
    )
    # block indices "abc", CG indices "def", output interleaved "adbecf": the
    # within-slab index alpha * d_a + m falls out of the reshape that follows.
    a_sub = string.ascii_letters[:n]
    c_sub = string.ascii_letters[n : 2 * n]
    out_sub = "".join(a + c for a, c in zip(a_sub, c_sub, strict=True))
    return DensePlan(
        structure=structure,
        axis_sectors=axis_sectors,
        axis_sizes=axis_sizes,
        cells=tuple(cells),
        subscripts=f"{a_sub},{c_sub}->{out_sub}",
    )


def to_dense(t: "SymmetricTensor") -> Array:
    """``T = Σ_τ A^(τ) ⊗ C^(τ)`` expanded into a plain dense array of ``t``'s backend.

    Explicit by design (invariant 9). Requires ``ClebschGordan``; a leg with
    ``dual=True`` additionally requires ``DualBasis``, the provider's
    ``V_a -> V_a^*`` isomorphism in the dense basis.

    The dtype is whatever the backend's own promotion makes of ``block × CG``:
    on NumPy that is ``result_type(block.dtype, float64)``, the pre-#82 rule
    exactly; on JAX without ``jax_enable_x64`` the CG array arrives as
    ``float32``, the same stance ``to_backend`` documents.
    """
    plan = dense_plan(t.structure)
    if not t.blocks:
        return np.zeros(plan.shape, dtype=np.float64)
    ref = t.blocks[0]
    cells = plan.cell_map()

    def build(sectors: tuple[Sector, ...], index: tuple[int, ...]) -> Array:
        cell = cells.get(sectors)
        if cell is None:  # symmetry-forbidden: exact zeros, no block read
            shape = tuple(plan.axis_sizes[ax][i] for ax, i in enumerate(index))
            return ar.do("zeros", shape, like=ref)
        acc = None
        for i, cgt in zip(cell.block_indices, cell.cgts, strict=True):
            g = ar.do("array", cgt, like=ref)  # backend + dtype policy in one call
            full = ar.do("einsum", plan.subscripts, t.blocks[i], g)
            full = ar.do("reshape", full, cell.shape)
            acc = full if acc is None else acc + full
        return acc

    def glue(sectors: tuple[Sector, ...], index: tuple[int, ...]) -> Array:
        axis = len(index)
        if axis == t.ndim:
            return build(sectors, index)
        parts = tuple(
            glue((*sectors, a), (*index, i)) for i, a in enumerate(plan.axis_sectors[axis])
        )
        return parts[0] if len(parts) == 1 else ar.do("concatenate", parts, axis=axis)

    return glue((), ())


def from_dense(
    dense: Array, legs: Sequence[Leg], *, atol: float | None = None
) -> "SymmetricTensor":
    """Project a dense carrier-basis array onto the symmetric subspace of ``legs``.

    The exact inverse of :func:`to_dense`: every cell of the dense grid is sliced
    out and contracted against the cached pseudo-inverse of its CG stack.

    ``legs`` cannot be inferred — a dense array carries no categorical
    information at all (which sector an index belongs to, which axes are
    ``dual``, which are IN) — so it is given, as ``from_legs`` spells it.

    Input that is not symmetric to ``atol`` is **refused**, never silently
    projected: the residual is ``sqrt(‖dense‖² − ‖reproduced‖²)`` accumulated
    over the cells *plus* the entire mass of the symmetry-forbidden cells, which
    is the part a per-cell check alone would miss. ``atol`` defaults to
    ``sqrt(eps(dtype)) * ‖dense‖`` — relative, because an absolute tolerance is
    meaningless for an unnormalized tensor.

    The check is a concrete-value question, so it raises JAX's own
    ``ConcretizationTypeError`` under a trace. The projection itself is pure
    slicing, ``reshape``, ``transpose`` and one matmul, hence traceable and
    differentiable: ``atol=math.inf`` is the documented spelling for "project,
    don't check", and it is the form that goes inside ``jit``.
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
        if atol != math.inf:
            # what the cell's CG span could not reproduce, taken as a difference
            # rather than as ‖mat‖² − ‖repro‖²: Pythagoras holds exactly in real
            # arithmetic but cancels to sqrt(eps) in floating point, which is the
            # same size as the default tolerance it would be compared against
            residuals[cell.sectors] = mat - coeffs @ ar.do("array", cell.matrix, like=mat)

    if atol != math.inf:
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
            "atol=math.inf to project without checking"
        )
