"""The coupled-sector matrix lowering ``T ≃ ⊕_c B_c ⊗ id_c`` and its inverse.

This is the "temporary map lowering" arrow of README "Reduced blocks follow
public ndarray axes": public-axis-ordered reduced blocks in, one dense matrix
per coupled sector out, exactly invertible. Nothing about the public axis order
changes and nothing is cached in storage — :func:`to_matrices` allocates
temporaries and hands them back.

Conventions fixed here once, because Milestone 7's truncation code has to invert
them:

* **Rows are OUT, columns are IN.** The row index of ``B_c`` is
  ``(output tree, degeneracy multi-index over the OUT axes in public order)``,
  the column index likewise over the IN axes. ``axes_order`` is
  ``(*out_axes, *in_axes)`` — the one transpose that brings a block to matrix
  form.
* **Degeneracy multi-indices are flattened in C order** over each side's public
  axis order. That is not a choice to maintain but a consequence of using the
  backend's own row-major ``reshape`` right after that transpose.
* **Bands are indexed by trees, not by external sectors.** For SU(2) two
  distinct output trees can carry identical external sectors, and they occupy
  two distinct row bands.
* **Band order is ``block_order`` restricted**, never a fresh enumeration, so two
  tensors whose OUT (resp. IN) legs agree as ordered legs get *identical* row
  (resp. column) orderings. Composition in #30 is then a plain matmul.
* **No ``sqrt(qdim)`` is folded into ``B_c``.** The norm identity is
  ``‖T‖² = Σ_c qdim(c)·‖B_c‖²_F`` — i.e. :func:`tenet.norm` regrouped — and
  folding the weight in would silently change the convention ``norm`` and
  ``to_dense`` already agree on.

The ``(output tree, input tree)`` grid of a coupled sector is *complete*
(``_block_order`` pairs the two sides' trees by cross product), so ``B_c``
assembles by pure concatenation: no zeros, no scatter, no in-place writes, and
therefore nothing that would refuse to trace under JAX. Blocks move only through
``ar.do("transpose"/"reshape"/"concatenate")`` plus basic slicing; there is no
NumPy call, no ``to_dense`` and no provider branching in this module.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.fusion_tree import FusionTree
from tenet.space import ProductSpace
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry.base import Sector

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "MapLayout",
    "TensorMapView",
    "as_map",
    "from_matrices",
    "map_layout",
    "to_matrices",
]

Band = tuple[Sector, FusionTree, int, int]
"""``(coupled sector, tree, offset, extent)`` — the offset/extent style of ``ops.fusion``."""


@dataclass(frozen=True, slots=True)
class MapLayout:
    """Where every block sits inside its coupled-sector matrix. Array-free, hashable.

    The block for ``(ot, it)`` occupies rows ``[row offset, + Π m_out)`` and
    columns ``[col offset, + Π m_in)`` of ``B_c`` — an outer reshape, no data
    motion beyond one transpose and one reshape per block.
    """

    structure: TensorStructure
    axes_order: tuple[int, ...]
    sectors: tuple[Sector, ...]
    rows: tuple[Band, ...]
    cols: tuple[Band, ...]
    grid: tuple[tuple[Sector, tuple[tuple[int, ...], ...]], ...]
    """Per coupled sector, block indices in row-major (row band × column band) order."""

    def row_bands(self, c: Sector) -> tuple[tuple[FusionTree, int, int], ...]:
        """``(output tree, offset, extent)`` for ``c``, in row order."""
        return tuple(band[1:] for band in self.rows if band[0] == c)

    def col_bands(self, c: Sector) -> tuple[tuple[FusionTree, int, int], ...]:
        """``(input tree, offset, extent)`` for ``c``, in column order."""
        return tuple(band[1:] for band in self.cols if band[0] == c)

    def shape(self, c: Sector) -> tuple[int, int]:
        """``(Σ_ot Π m_out, Σ_it Π m_in)`` — the shape of ``B_c``."""
        return (
            sum(extent for _, _, extent in self.row_bands(c)),
            sum(extent for _, _, extent in self.col_bands(c)),
        )


@cache
def map_layout(structure: TensorStructure) -> MapLayout:
    """The lowering plan for ``structure``. Cached: repeat calls return one object."""
    out_axes, in_axes = structure.out_axes, structure.in_axes
    # dict insertion order == block_order order, which is sorted: the row bands
    # come out in tree order for free, and so do the column bands within the
    # first row band (which, the grid being complete, already lists them all).
    row_trees: dict[Sector, dict[FusionTree, None]] = {}
    col_trees: dict[Sector, dict[FusionTree, None]] = {}
    for key in structure.block_order:
        row_trees.setdefault(key.coupled, {})[key.output_tree] = None
        col_trees.setdefault(key.coupled, {})[key.input_tree] = None

    sectors = tuple(sorted(row_trees))
    rows: list[Band] = []
    cols: list[Band] = []
    grid: list[tuple[Sector, tuple[tuple[int, ...], ...]]] = []
    for c in sectors:
        rbands = _bands(structure, out_axes, c, row_trees[c])
        cbands = _bands(structure, in_axes, c, col_trees[c])
        rows.extend(rbands)
        cols.extend(cbands)
        grid.append(
            (
                c,
                tuple(
                    tuple(structure.index_of(FusionBlockKey(ot, it)) for _, it, _, _ in cbands)
                    for _, ot, _, _ in rbands
                ),
            )
        )

    # The grid is complete by construction of `_block_order`; assert it rather
    # than trust it. index_of() above already raises on a missing cell, so a
    # matching total is enough to make every cell distinct and every block used.
    covered = sum(len(cells) * len(cells[0]) for _, cells in grid)
    if covered != structure.num_blocks:
        raise RuntimeError(
            f"map_layout: the (output tree, input tree) grid covers {covered} of "
            f"{structure.num_blocks} blocks; block_order is not a per-sector cross product"
        )
    return MapLayout(
        structure, (*out_axes, *in_axes), sectors, tuple(rows), tuple(cols), tuple(grid)
    )


def _bands(
    structure: TensorStructure, axes: tuple[int, ...], c: Sector, trees: Mapping[FusionTree, None]
) -> tuple[Band, ...]:
    """Lay one side's trees out contiguously; extent is ``Π`` of its degeneracies."""
    bands: list[Band] = []
    offset = 0
    for tree in trees:
        extent = 1
        for ax, u in zip(axes, tree.uncoupled, strict=True):
            leg = structure.legs[ax]
            extent *= leg.degeneracy(leg.space_sector(u))
        bands.append((c, tree, offset, extent))
        offset += extent
    return tuple(bands)


def to_matrices(t: "SymmetricTensor") -> dict[Sector, Any]:
    """``{c: B_c}``, one dense backend matrix per coupled sector. ``t`` is untouched."""
    layout = map_layout(t.structure)
    identity = tuple(range(t.ndim))
    mats: dict[Sector, Any] = {}
    for c, cells in layout.grid:
        cbands = layout.col_bands(c)
        rows = []
        for (_, _, dr), indices in zip(layout.row_bands(c), cells, strict=True):
            parts = []
            for i, (_, _, dc) in zip(indices, cbands, strict=True):
                block = t.blocks[i]
                if layout.axes_order != identity:
                    block = ar.do("transpose", block, layout.axes_order)
                parts.append(ar.do("reshape", block, (dr, dc)))
            rows.append(parts[0] if len(parts) == 1 else ar.do("concatenate", parts, axis=1))
        mats[c] = rows[0] if len(rows) == 1 else ar.do("concatenate", rows, axis=0)
    return mats


def from_matrices(structure: TensorStructure, mats: Mapping[Sector, Any]) -> "SymmetricTensor":
    """Inverse of :func:`to_matrices` against the same ``structure``. Exact round-trip.

    ``mats`` must have exactly the coupled sectors of ``structure`` — a missing
    one, an unknown one or a wrong shape is a ``ValueError`` (invariant 11); the
    matrices carry no categorical information, so the structure has to be given.
    """
    from tenet.tensor import SymmetricTensor

    layout = map_layout(structure)
    _check(layout, mats)

    order = layout.axes_order
    identity = tuple(range(structure.ndim))
    inverse = tuple(sorted(identity, key=order.__getitem__))
    blocks: list[Any] = [None] * structure.num_blocks
    for c, cells in layout.grid:
        mat = mats[c]
        cbands = layout.col_bands(c)
        for (_, ro, dr), indices in zip(layout.row_bands(c), cells, strict=True):
            for i, (_, co, dc) in zip(indices, cbands, strict=True):
                # basic slicing, not an ar.do: every backend spells a contiguous
                # 2-D slice the same way (as in ops.fusion._unapply)
                piece = mat[ro : ro + dr, co : co + dc]
                shape = structure.block_shape(structure.block_order[i])
                piece = ar.do("reshape", piece, tuple(shape[a] for a in order))
                blocks[i] = piece if order == identity else ar.do("transpose", piece, inverse)
    return SymmetricTensor(structure, tuple(blocks))


@dataclass(frozen=True, slots=True)
class TensorMapView:
    """A *semantic* view of a tensor as a morphism. Nothing is moved or materialized.

    ``as_map`` allocates nothing: the view holds the tensor and every property is
    derived from the ``side`` metadata the legs already carry (README "TensorMap
    views"). Materializing an ``(out..., in...)`` reordering here would violate
    invariant 3 and fork ``T.as_map().svd()`` from ``svd(T, axes=...)``.
    """

    tensor: "SymmetricTensor"

    @property
    def codomain(self) -> ProductSpace:
        """OUT legs, in public axis order."""
        return ProductSpace(self.tensor.codomain)

    @property
    def domain(self) -> ProductSpace:
        """IN legs, in public axis order."""
        return ProductSpace(self.tensor.domain)

    def matrices(self) -> dict[Sector, Any]:
        """``{c: B_c}`` — :func:`to_matrices` of the underlying tensor."""
        return to_matrices(self.tensor)

    def compose(self, other: "TensorMapView | SymmetricTensor") -> "SymmetricTensor":
        """``self ∘ other``. See :func:`tenet.compose`."""
        from tenet.ops.map import compose

        return compose(self.tensor, other.tensor if isinstance(other, TensorMapView) else other)

    def adjoint(self) -> "SymmetricTensor":
        """``T†`` in ``Hom(codomain, domain)``. See :func:`tenet.adjoint`."""
        from tenet.ops.map import adjoint

        return adjoint(self.tensor)

    def svd(self) -> tuple["SymmetricTensor", "SymmetricTensor", "SymmetricTensor"]:
        """``U, S, Vh`` for the current partition. See :func:`tenet.linalg.svd`."""
        from tenet.ops.linalg import svd

        return svd(self.tensor)

    def svd_truncated(
        self, **kwargs
    ) -> tuple["SymmetricTensor", "SymmetricTensor", "SymmetricTensor"]:
        """Truncated ``U, S, Vh`` for the current partition. Not jittable.

        See :func:`tenet.linalg.svd_truncated` for every keyword.
        """
        from tenet.ops.linalg import svd_truncated

        return svd_truncated(self.tensor, **kwargs)

    def qr(self) -> tuple["SymmetricTensor", "SymmetricTensor"]:
        """``Q, R`` for the current partition. See :func:`tenet.linalg.qr`."""
        from tenet.ops.linalg import qr

        return qr(self.tensor)

    def eigh(self) -> tuple["SymmetricTensor", "SymmetricTensor"]:
        """``W, V`` for the current partition. See :func:`tenet.linalg.eigh`."""
        from tenet.ops.linalg import eigh

        return eigh(self.tensor)

    def polar(self, side: str = "left") -> tuple["SymmetricTensor", "SymmetricTensor"]:
        """``W, P`` for the current partition. See :func:`tenet.linalg.polar`."""
        from tenet.ops.linalg import polar

        return polar(self.tensor, side=side)

    def lq(self) -> tuple["SymmetricTensor", "SymmetricTensor"]:
        """``L, Q`` for the current partition. See :func:`tenet.linalg.lq`."""
        from tenet.ops.linalg import lq

        return lq(self.tensor)


def as_map(t: "SymmetricTensor") -> TensorMapView:
    """View ``t`` as a morphism. Zero-copy: no block is read, moved or allocated."""
    return TensorMapView(t)


def _check(layout: MapLayout, mats: Mapping[Sector, Any]) -> None:
    expected = set(layout.sectors)
    for c in mats:
        if c not in expected:
            raise ValueError(
                f"from_matrices: sector {c!r} is not a coupled sector of this structure; "
                f"expected {sorted(expected)}"
            )
    for c in layout.sectors:
        if c not in mats:
            raise ValueError(
                f"from_matrices: no matrix for coupled sector {c!r}, "
                f"expected one of shape {layout.shape(c)}"
            )
        got = tuple(mats[c].shape)
        if got != layout.shape(c):
            raise ValueError(
                f"from_matrices: matrix for sector {c!r} has shape {got}, "
                f"expected {layout.shape(c)}"
            )
