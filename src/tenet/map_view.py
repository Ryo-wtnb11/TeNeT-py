"""The coupled-sector matrix lowering ``T ≃ ⊕_c B_c ⊗ id_c`` and its inverse.

This is the "temporary map lowering" arrow of docs/design.md "Reduced blocks follow
public ndarray axes": public-axis-ordered reduced blocks in, one dense matrix
per coupled sector out, exactly invertible. Nothing about the public axis order
changes and nothing is cached in storage — [to_matrices][tenet.to_matrices] allocates
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
  ``‖T‖² = Σ_c qdim(c)·‖B_c‖²_F`` — i.e. [tenet.norm][] regrouped — and
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
from tenet.leg import Leg
from tenet.space import GradedSpace, ProductSpace
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry.base import Sector

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "MapLayout",
    "TensorMapView",
    "as_map",
    "check_square",
    "from_matrices",
    "map_layout",
    "to_matrices",
]

Band = tuple[Sector, FusionTree, int, int]
"""``(coupled sector, tree, offset, extent)`` — the offset/extent style of ``ops.fusion``."""


@dataclass(frozen=True, slots=True)
class MapLayout:
    """Where every block sits inside its coupled-sector matrix. Array-free, hashable.

    Parameters
    ----------
    structure : TensorStructure
        The structure this layout was computed from.
    axes_order : tuple of int
        ``(*out_axes, *in_axes)`` — the one transpose to matrix form.
    sectors : tuple of Sector
        The coupled sectors, sorted; one matrix ``B_c`` per entry.
    rows : tuple of Band
        ``(coupled sector, output tree, offset, extent)`` row bands, in order.
    cols : tuple of Band
        ``(coupled sector, input tree, offset, extent)`` column bands, in order.
    grid : tuple
        Per coupled sector, block indices in row-major (row band × column band)
        order.

    Notes
    -----
    Built only by [map_layout][tenet.map_layout], never by hand. The block for
    ``(ot, it)`` occupies rows ``[row offset, + Π m_out)`` and
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
        """``(output tree, offset, extent)`` for ``c``, in row order.

        Parameters
        ----------
        c : Sector
            A coupled sector of the layout.

        Returns
        -------
        tuple of (FusionTree, int, int)
            The row bands of ``B_c``.
        """
        return tuple(band[1:] for band in self.rows if band[0] == c)

    def col_bands(self, c: Sector) -> tuple[tuple[FusionTree, int, int], ...]:
        """``(input tree, offset, extent)`` for ``c``, in column order.

        Parameters
        ----------
        c : Sector
            A coupled sector of the layout.

        Returns
        -------
        tuple of (FusionTree, int, int)
            The column bands of ``B_c``.
        """
        return tuple(band[1:] for band in self.cols if band[0] == c)

    def shape(self, c: Sector) -> tuple[int, int]:
        """``(Σ_ot Π m_out, Σ_it Π m_in)`` — the shape of ``B_c``.

        Parameters
        ----------
        c : Sector
            A coupled sector of the layout.

        Returns
        -------
        tuple of int
            The ``(rows, columns)`` shape of ``B_c``.
        """
        return (
            sum(extent for _, _, extent in self.row_bands(c)),
            sum(extent for _, _, extent in self.col_bands(c)),
        )


@cache
def map_layout(structure: TensorStructure) -> MapLayout:
    """The lowering plan for ``structure``. Cached: repeat calls return one object.

    Parameters
    ----------
    structure : TensorStructure
        The structure to lay out.

    Returns
    -------
    MapLayout
        The per-sector band and grid tables.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, TensorStructure, map_layout
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> layout = map_layout(TensorStructure((Leg(V, OUT), Leg(V, IN))))
    >>> layout.sectors
    (U1Sector(charge=0), U1Sector(charge=1))
    >>> layout.shape(U1Sector(0))
    (2, 2)
    """
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


@cache
def _tables(
    structure: TensorStructure,
) -> tuple[tuple[tuple[tuple[FusionTree, int, int], ...], ...], tuple[tuple[int, ...], ...]]:
    """Per-sector ``(row bands, column bands)`` and per-block shapes in ``axes_order``.

    Parameters
    ----------
    structure : TensorStructure
        The structure to tabulate, in the layout's own sector order.

    Returns
    -------
    bands : tuple
        ``(row_bands(c), col_bands(c))`` per coupled sector, aligned with
        ``map_layout(structure).grid``.
    shapes : tuple of tuple of int
        Per block of ``block_order``, its degeneracies permuted into
        ``axes_order`` — the shape ``from_matrices`` reshapes a slice to.

    Notes
    -----
    Both are pure functions of ``structure``, so they are cached beside
    [map_layout][tenet.map_layout] rather than rebuilt inside the assembly loops.
    They used to be: the band tuples once per coupled sector per call (each an
    ``O(bands)`` filter of ``rows``/``cols``, so ``O(bands²)`` for the call), and
    the shapes once per block per call (a ``FusionBlockKey`` hash and a dict lookup
    through ``block_shape``). On a many-small-blocks structure that re-derivation is
    most of what an assembly costs, and it is what makes ``from_matrices`` — which
    moves no bytes, every piece being a view — the more expensive of the two
    directions there (docs/design.md "M69").
    """
    layout = map_layout(structure)
    order = layout.axes_order
    return (
        tuple((layout.row_bands(c), layout.col_bands(c)) for c in layout.sectors),
        tuple(tuple(structure.block_shape(key)[a] for a in order) for key in structure.block_order),
    )


def to_matrices(t: "SymmetricTensor") -> dict[Sector, Any]:
    """``{c: B_c}``, one dense backend matrix per coupled sector. ``t`` is untouched.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor to lower.

    Returns
    -------
    dict of Sector to array
        One matrix per coupled sector, laid out per
        [map_layout][tenet.map_layout].

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, to_matrices
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> mats = to_matrices(t)
    >>> sorted(mats)
    [U1Sector(charge=0), U1Sector(charge=1)]
    >>> mats[U1Sector(0)].shape
    (2, 2)
    """
    layout = map_layout(t.structure)
    bands, _ = _tables(t.structure)
    order = layout.axes_order
    permuted = order != tuple(range(t.ndim))
    mats: dict[Sector, Any] = {}
    for (c, cells), (rbands, cbands) in zip(layout.grid, bands, strict=True):
        rows = []
        for (_, _, dr), indices in zip(rbands, cells, strict=True):
            parts = []
            for i, (_, _, dc) in zip(indices, cbands, strict=True):
                block = t.blocks[i]
                if permuted:
                    block = ar.do("transpose", block, order)
                parts.append(ar.do("reshape", block, (dr, dc)))
            rows.append(parts[0] if len(parts) == 1 else ar.do("concatenate", parts, axis=1))
        mats[c] = rows[0] if len(rows) == 1 else ar.do("concatenate", rows, axis=0)
    return mats


def from_matrices(structure: TensorStructure, mats: Mapping[Sector, Any]) -> "SymmetricTensor":
    """Inverse of [to_matrices][tenet.to_matrices] against the same ``structure``. Exact round-trip.

    Parameters
    ----------
    structure : TensorStructure
        The structure the matrices belong to; the matrices carry no categorical
        information, so the structure has to be given.
    mats : Mapping[Sector, array]
        Exactly the coupled sectors of ``structure``, each with the shape
        ``map_layout(structure).shape(c)``.

    Returns
    -------
    SymmetricTensor
        The tensor whose lowering is ``mats``.

    Raises
    ------
    ValueError
        If a sector is missing, unknown, or its matrix has the wrong shape
        (invariant 11).

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet import from_matrices, to_matrices
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> from_matrices(t.structure, to_matrices(t)) == t
    True
    """
    from tenet.tensor import SymmetricTensor

    layout = map_layout(structure)
    _check(layout, mats)

    bands, shapes = _tables(structure)
    order = layout.axes_order
    identity = tuple(range(structure.ndim))
    inverse = tuple(sorted(identity, key=order.__getitem__))
    permuted = order != identity
    blocks: list[Any] = [None] * structure.num_blocks
    for (c, cells), (rbands, cbands) in zip(layout.grid, bands, strict=True):
        mat = mats[c]
        for (_, ro, dr), indices in zip(rbands, cells, strict=True):
            for i, (_, co, dc) in zip(indices, cbands, strict=True):
                # basic slicing, not an ar.do: every backend spells a contiguous
                # 2-D slice the same way (as in ops.fusion._unapply)
                piece = ar.do("reshape", mat[ro : ro + dr, co : co + dc], shapes[i])
                blocks[i] = ar.do("transpose", piece, inverse) if permuted else piece
    return SymmetricTensor(structure, tuple(blocks))


@dataclass(frozen=True, slots=True)
class TensorMapView:
    """A *semantic* view of a tensor as a morphism. Nothing is moved or materialized.

    Parameters
    ----------
    tensor : SymmetricTensor
        The tensor being viewed. Held, not copied.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> m = t.as_map()
    >>> m.codomain.reduced_dim, m.domain.reduced_dim
    (3, 3)
    >>> u, s, vh = m.svd()
    >>> s.ndim
    2

    Notes
    -----
    ``as_map`` allocates nothing: the view holds the tensor and every property is
    derived from the ``side`` metadata the legs already carry (docs/design.md "TensorMap
    views"). Materializing an ``(out..., in...)`` reordering here would violate
    invariant 3 and fork ``T.as_map().svd()`` from ``svd(T, axes=...)``.
    """

    tensor: "SymmetricTensor"

    @property
    def codomain(self) -> ProductSpace:
        """OUT legs, in public axis order.

        Returns
        -------
        ProductSpace
            The codomain factor.
        """
        return ProductSpace(self.tensor.codomain)

    @property
    def domain(self) -> ProductSpace:
        """IN legs, in public axis order.

        Returns
        -------
        ProductSpace
            The domain factor.
        """
        return ProductSpace(self.tensor.domain)

    def matrices(self) -> dict[Sector, Any]:
        """``{c: B_c}`` — [to_matrices][tenet.to_matrices] of the underlying tensor.

        Returns
        -------
        dict of Sector to array
            One matrix per coupled sector.
        """
        return to_matrices(self.tensor)

    def compose(self, other: "TensorMapView | SymmetricTensor") -> "SymmetricTensor":
        """``self ∘ other``. See [tenet.compose][].

        Parameters
        ----------
        other : TensorMapView or SymmetricTensor
            The morphism applied first; its codomain must match this map's
            domain.

        Returns
        -------
        SymmetricTensor
            The composition, as a tensor.
        """
        from tenet.ops.map import compose

        return compose(self.tensor, other.tensor if isinstance(other, TensorMapView) else other)

    def adjoint(self) -> "SymmetricTensor":
        """``T†`` in ``Hom(codomain, domain)``. See [tenet.adjoint][].

        Returns
        -------
        SymmetricTensor
            The adjoint tensor.
        """
        from tenet.ops.map import adjoint

        return adjoint(self.tensor)

    def svd(
        self, *, bond: GradedSpace | None = None
    ) -> tuple["SymmetricTensor", "SymmetricTensor", "SymmetricTensor"]:
        """``U, S, Vh`` for the current partition. See [svd][tenet.ops.linalg.svd].

        Parameters
        ----------
        bond : GradedSpace or None, optional
            ``bond=<GradedSpace>`` projects onto a pre-decided bond space, exactly
            as the free function does — still shape-static, still traceable.
            Default ``None``.

        Returns
        -------
        U, S, Vh : SymmetricTensor
            The factors, as for [svd][tenet.ops.linalg.svd].
        """
        from tenet.ops.linalg import svd

        return svd(self.tensor, bond=bond)

    def svd_truncated(
        self, **kwargs: Any
    ) -> tuple["SymmetricTensor", "SymmetricTensor", "SymmetricTensor"]:
        """Truncated ``U, S, Vh`` for the current partition. Not jittable.

        Parameters
        ----------
        **kwargs
            Every keyword of [svd_truncated][tenet.ops.linalg.svd_truncated],
            forwarded unchanged.

        Returns
        -------
        U, S, Vh : SymmetricTensor
            The truncated factors.
        """
        from tenet.ops.linalg import svd_truncated

        return svd_truncated(self.tensor, **kwargs)

    def qr(self) -> tuple["SymmetricTensor", "SymmetricTensor"]:
        """``Q, R`` for the current partition. See [qr][tenet.ops.linalg.qr].

        Returns
        -------
        Q, R : SymmetricTensor
            The factors.
        """
        from tenet.ops.linalg import qr

        return qr(self.tensor)

    def eigh(self) -> tuple["SymmetricTensor", "SymmetricTensor"]:
        """``W, V`` for the current partition. See [eigh][tenet.ops.linalg.eigh].

        Returns
        -------
        W, V : SymmetricTensor
            Eigenvalues and eigenvectors.
        """
        from tenet.ops.linalg import eigh

        return eigh(self.tensor)

    def eig(self) -> tuple["SymmetricTensor", "SymmetricTensor"]:
        """``W, V`` for the current partition. See [eig][tenet.ops.linalg.eig].

        Returns
        -------
        W, V : SymmetricTensor
            Eigenvalues and eigenvectors.
        """
        from tenet.ops.linalg import eig

        return eig(self.tensor)

    def eigvals(self) -> "SymmetricTensor":
        """``W`` for the current partition. See [eigvals][tenet.ops.linalg.eigvals].

        Returns
        -------
        SymmetricTensor
            The eigenvalues.
        """
        from tenet.ops.linalg import eigvals

        return eigvals(self.tensor)

    def expm(self, *, alpha: Any = 1.0) -> "SymmetricTensor":
        """``exp(alpha * T)`` for the current partition. See [expm][tenet.ops.linalg.expm].

        Parameters
        ----------
        alpha : scalar, optional
            The prefactor inside the exponential. Default ``1.0``.

        Returns
        -------
        SymmetricTensor
            The matrix exponential.
        """
        from tenet.ops.linalg import expm

        return expm(self.tensor, alpha=alpha)

    def polar(self, side: str = "left") -> tuple["SymmetricTensor", "SymmetricTensor"]:
        """``W, P`` for the current partition. See [polar][tenet.ops.linalg.polar].

        Parameters
        ----------
        side : str, optional
            ``"left"`` (the default) or ``"right"``.

        Returns
        -------
        W, P : SymmetricTensor
            The isometric and positive factors.
        """
        from tenet.ops.linalg import polar

        return polar(self.tensor, side=side)

    def left_null(self) -> "SymmetricTensor":
        """``N`` with ``N† T = 0`` for the current partition. See
        [left_null][tenet.ops.linalg.left_null].

        Returns
        -------
        SymmetricTensor
            The left null space.
        """
        from tenet.ops.linalg import left_null

        return left_null(self.tensor)

    def right_null(self) -> "SymmetricTensor":
        """``N`` with ``T N† = 0`` for the current partition. See
        [right_null][tenet.ops.linalg.right_null].

        Returns
        -------
        SymmetricTensor
            The right null space.
        """
        from tenet.ops.linalg import right_null

        return right_null(self.tensor)

    def lq(self) -> tuple["SymmetricTensor", "SymmetricTensor"]:
        """``L, Q`` for the current partition. See [lq][tenet.ops.linalg.lq].

        Returns
        -------
        L, Q : SymmetricTensor
            The factors.
        """
        from tenet.ops.linalg import lq

        return lq(self.tensor)


def as_map(t: "SymmetricTensor") -> TensorMapView:
    """View ``t`` as a morphism. Zero-copy: no block is read, moved or allocated.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor to view.

    Returns
    -------
    TensorMapView
        The semantic view.
    """
    return TensorMapView(t)


def check_square(m: "SymmetricTensor", caller: str) -> None:
    """Refuse a map whose domain is not its codomain, as ``(space, dual)`` in order.

    Parameters
    ----------
    m : SymmetricTensor
        The map to check.
    caller : str
        The name the message opens with.

    Raises
    ------
    ValueError
        If the map is not square; the message names the first offending
        position and both legs.

    Notes
    -----
    Not a new checker: the predicate is ``ProductSpace.matches``, the identical
    call ``ops.map._check_composable`` makes — ``m`` composability-checked against
    itself. Only the message is new. Structural and total; the *numbers* are never
    inspected (see [eigh][tenet.ops.linalg.eigh]).

    Five square-map operations share this paragraph (``eigh``, ``expm``, ``eig``,
    ``eigvals``, ``full_trace``); a copy per caller is a paragraph that would need
    editing five times. It lives here, next to [as_map][tenet.as_map], because
    the predicate is pure map-view metadata — ``ops`` used to hold it privately
    and its fifth caller is in another ``ops`` module (#126).
    """
    codomain, domain = as_map(m).codomain, as_map(m).domain
    i = codomain.matches(domain)
    if i is None:
        return

    def at(axes: tuple[int, ...], legs: tuple[Leg, ...]) -> str:
        return f"public axis {axes[i]}: {legs[i]!r}" if i < len(legs) else "no leg"

    raise ValueError(
        f"{caller}: the map is not square at position {i} "
        f"(codomain {at(m.structure.out_axes, codomain.legs)}; "
        f"domain {at(m.structure.in_axes, domain.legs)}). "
        f"{caller} requires the domain to be the codomain as (space, dual) in the same order — "
        "side is not compared and name is ignored, and dimensions alone are never enough "
        "(a charge-reversed U(1) partner has the same dimension and the wrong space). "
        "Matching only up to a reordering would be a within-side transpose, i.e. a braid; "
        "use tenet.transpose, or repartition (#32), to build a square map first."
    )


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
