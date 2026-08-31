"""The coupled-sector matrix lowering ``T ≃ ⊕_c B_c ⊗ id_c`` -- the storage, and its cut.

The matrices are not a temporary: a [SymmetricTensor][tenet.SymmetricTensor] *is* one
dense matrix per coupled sector, and this module is where their layout is decided.
[to_matrices][tenet.to_matrices] hands the stored arrays back keyed by sector,
[from_matrices][tenet.from_matrices] wraps a set of them as a tensor, and
[assemble][tenet.map_view.assemble] / [views][tenet.map_view.views] are the two halves of
the boundary a caller crosses when it speaks in fusion-tree blocks instead.

Conventions fixed here once, because the truncation code has to invert them:

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
* **Band order is configuration order**: the bands are laid out one *irrep
  configuration* -- one choice of uncoupled irrep per leg on that side -- at a time,
  configurations ordered by ``(degeneracies, uncoupled labels)`` and the trees inside a
  configuration in the order they already arrive in. So the row index is
  ``offset(configuration) + i_tree * extent + i_degeneracy``, the structural axis slow
  and the degeneracy axis fast, which is the order both TensorKit and frostspin use.
  The order is a pure function of *this side's* ordered legs, never a fresh enumeration
  of its own, so two tensors whose OUT (resp. IN) legs agree as ordered legs get
  *identical* row (resp. column) orderings and composition is a plain matmul. Grouping
  by configuration is what makes a run of bands share an extent *and* a shape, which
  merely equal extents do not: ``(1, 2, 3)`` and ``(3, 2, 1)`` are both six rows tall
  and are two different cell shapes.
* **No ``sqrt(qdim)`` is folded into ``B_c``.** The norm identity is
  ``‖T‖² = Σ_c qdim(c)·‖B_c‖²_F`` — i.e. [tenet.norm][] regrouped — and
  folding the weight in would silently change the convention ``norm`` and
  ``to_dense`` already agree on.

The structural half of that layout -- which configurations exist and how many trees each
carries -- is [tree_structure][tenet.map_view.tree_structure], cached on the structure with its
*degeneracies stripped*, so one entry serves every bond dimension. The offsets and
extents are [map_layout][tenet.map_layout]'s and are keyed on the full structure. That is
TensorKit's ``sectorstructure``/``degeneracystructure`` split, and frostspin's.

The ``(output tree, input tree)`` grid of a coupled sector is *complete*
(``_block_order`` pairs the two sides' trees by cross product), so every cell of
``B_c`` is written exactly once and none is left zero. That completeness admits
two assemblies, and [assemble][tenet.map_view.assemble] chooses between them on the
blocks' **backend**, never on their shapes:

* On an **immutable** backend -- JAX -- ``B_c`` is built by pure concatenation:
  no ``zeros``, no scatter, no in-place write, nothing that would refuse to
  trace. This is the reference path and it is what defines the values.
* On a **mutable** backend -- NumPy, PyTorch -- ``B_c`` is one ``empty`` per
  coupled sector into which each block is copied once. The *destination* slice is
  reshaped, never the source: splitting a 2-D slice's two axes into the block's
  axes only subdivides strides, so it is always a view, and the assignment is one
  strided copy per block. The concatenating path costs three passes (materialise
  the transposed view, join the row band, join the sector), and on a
  bandwidth-bound block geometry that is most of what an assembly costs.

The two produce bit-identical matrices -- the mutable path writes exactly the
cells the concatenating path would place, in the same layout -- and the tests run
them against each other on every provider. The split is over array *mutability*,
a property of the backend resolved once per call by ``ar.infer_backend``; it is
not a size heuristic and it never branches on a traced value.

[views][tenet.map_view.views] cuts the matrices back into blocks, and the grid gives it a
cheaper walk than cell by cell. Group the row bands by degeneracies and the column bands
likewise; because the bands are laid out configuration by configuration, every such pair
is one contiguous rectangle of ``B_c``, and splitting that rectangle's two axes into
``(rows, extent, columns, extent)``, swapping the two middle axes and splitting the
extents into the cell's own axes exposes every cell of it as a view -- three array calls
for the whole rectangle instead of two per cell. Every cell of a rectangle has the same
shape by construction, so no cell is ever reshaped on its own.

Blocks move only through ``ar.do("transpose"/"reshape"/"concatenate"/"empty")``
plus basic slicing; there is no ``to_dense`` and no *symmetry provider* branching in
this module.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.backend import lib_fn
from tenet.cache import plan_cache
from tenet.fusion_tree import FusionTree
from tenet.leg import Leg
from tenet.space import GradedSpace, ProductSpace
from tenet.structure import FusionBlockKey, TensorStructure, _pattern
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
    "tree_structure",
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
    shapes : tuple of (int, int)
        The ``(rows, columns)`` shape of each ``B_c``, aligned with ``sectors``.

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
    shapes: tuple[tuple[int, int], ...]
    """The shape of each ``B_c``, aligned with ``sectors``."""

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
        return self.shapes[self.sectors.index(c)]


def _layout_cost(layout: MapLayout) -> int:
    """One band, and one grid cell, per term -- the tables' size in ``tenet.cache`` units."""
    return (
        len(layout.rows)
        + len(layout.cols)
        + sum(len(cells) * len(cells[0]) for _, cells in layout.grid if cells)
    )


@plan_cache(cost=_layout_cost)
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
    table = tree_structure(structure)
    sectors = tuple(c for c, _, _ in table)
    rows: list[Band] = []
    cols: list[Band] = []
    shapes: list[tuple[int, int]] = []
    grid: list[tuple[Sector, tuple[tuple[int, ...], ...]]] = []
    for c, out_configs, in_configs in table:
        rbands = _bands(structure, out_axes, c, out_configs)
        cbands = _bands(structure, in_axes, c, in_configs)
        rows.extend(rbands)
        cols.extend(cbands)
        shapes.append(
            (
                sum(extent for _, _, _, extent in rbands),
                sum(extent for _, _, _, extent in cbands),
            )
        )
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
        structure,
        (*out_axes, *in_axes),
        sectors,
        tuple(rows),
        tuple(cols),
        tuple(grid),
        tuple(shapes),
    )


Config = tuple[tuple[Sector, ...], tuple[FusionTree, ...]]
"""One irrep configuration: its uncoupled labels in that side's public axis order, and
the fusion trees carrying them -- the structural multiplicity of the configuration."""


@plan_cache(cost=lambda table: sum(len(t) for _, o, i in table for _, t in (*o, *i)))
def tree_structure(
    structure: TensorStructure,
) -> tuple[tuple[Sector, tuple[Config, ...], tuple[Config, ...]], ...]:
    """Per coupled sector, each side's irrep configurations and the trees in each.

    Parameters
    ----------
    structure : TensorStructure
        The structure to tabulate.

    Returns
    -------
    tuple
        ``(coupled sector, output configurations, input configurations)`` per coupled
        sector, sorted by sector; each configuration is
        [Config][tenet.map_view.Config], sorted by its uncoupled labels, and its trees
        arrive in ``block_order``'s own order.

    Notes
    -----
    **Degeneracy-independent, deliberately.** The key is ``_pattern(structure)`` -- the
    same legs with every degeneracy 1 -- so one entry serves every bond dimension a
    sweep passes through, which is the only way a structure-keyed cache survives an
    optimization that moves the degeneracies at every bond. What lives here is the part
    of the layout that a degeneracy cannot change: which configurations exist and how
    many trees each carries. The offsets and extents, which a degeneracy does change,
    are [map_layout][tenet.map_layout]'s and are keyed on the full structure.

    That split is not ours: TensorKit keys its ``sectorstructure`` on sector content
    alone and its ``degeneracystructure`` per space, and frostspin keys its structural
    table on irrep labels with the degeneracies stripped. The measurement agrees --
    the structural count is 1,967 at SU(2) rank 6 whether the degeneracies are ``2`` or
    ``j+1``.
    """
    if (pattern := _pattern(structure)) is not structure:
        return tree_structure(pattern)
    # dict insertion order == block_order order, which is sorted: the trees of a
    # configuration come out in block order for free, and so do the column trees within
    # the first row tree (which, the grid being complete, already lists them all).
    out_trees: dict[Sector, dict[tuple[Sector, ...], dict[FusionTree, None]]] = {}
    in_trees: dict[Sector, dict[tuple[Sector, ...], dict[FusionTree, None]]] = {}
    for key in structure.block_order:
        ot, it = key.output_tree, key.input_tree
        out_trees.setdefault(key.coupled, {}).setdefault(ot.uncoupled, {})[ot] = None
        in_trees.setdefault(key.coupled, {}).setdefault(it.uncoupled, {})[it] = None
    return tuple((c, _configs(out_trees[c]), _configs(in_trees[c])) for c in sorted(out_trees))


def _configs(trees: Mapping[tuple[Sector, ...], Mapping[FusionTree, None]]) -> tuple[Config, ...]:
    """One side's configurations, sorted by their uncoupled labels."""
    return tuple((u, tuple(trees[u])) for u in sorted(trees))


def _dims(
    structure: TensorStructure, axes: tuple[int, ...], uncoupled: tuple[Sector, ...]
) -> tuple[int, ...]:
    """One configuration's degeneracy per axis, in that side's public order."""
    return tuple(
        structure.legs[ax].degeneracy(structure.legs[ax].space_sector(u))
        for ax, u in zip(axes, uncoupled, strict=True)
    )


def _degeneracies(
    structure: TensorStructure, axes: tuple[int, ...], tree: FusionTree
) -> tuple[int, ...]:
    """One tree's degeneracy per axis, in that side's public order."""
    return _dims(structure, axes, tree.uncoupled)


def _bands(
    structure: TensorStructure, axes: tuple[int, ...], c: Sector, configs: tuple[Config, ...]
) -> tuple[Band, ...]:
    """Lay one side's trees out contiguously, configuration by configuration.

    Ordered by ``(degeneracies, uncoupled labels)`` over the configurations, with the
    trees of a configuration in the order they already arrive in. A configuration's
    bands therefore share an extent *and* a shape by construction, which merely equal
    extents do not -- ``(1, 2, 3)`` and ``(3, 2, 1)`` are both six rows tall and are
    two different cell shapes. That is a function of this side's ordered legs alone,
    which is what composition needs: two tensors whose OUT (resp. IN) legs agree as
    ordered legs compute the same row (resp. column) ordering, so composing them is a
    plain matmul.
    """
    dims = {u: _dims(structure, axes, u) for u, _ in configs}
    bands: list[Band] = []
    offset = 0
    for u, trees in sorted(configs, key=lambda cfg: (dims[cfg[0]], cfg[0])):
        extent = math.prod(dims[u])
        for tree in trees:
            bands.append((c, tree, offset, extent))
            offset += extent
    return tuple(bands)


@plan_cache(cost=lambda t: sum(len(r) + len(c) for r, c in t[0]) + len(t[1]))
def _tables(
    structure: TensorStructure,
) -> tuple[
    tuple[
        tuple[tuple[tuple[FusionTree, int, int], ...], tuple[tuple[FusionTree, int, int], ...]],
        ...,
    ],
    tuple[tuple[int, ...], ...],
]:
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
    Rebuilding them inside the loops would cost the band tuples once per coupled
    sector per call (each an ``O(bands)`` filter of ``rows``/``cols``, so
    ``O(bands²)`` for the call) and the shapes once per block per call (a
    ``FusionBlockKey`` hash and a dict lookup through ``block_shape``). On a
    many-small-blocks structure that re-derivation is most of what an assembly
    costs.
    """
    layout = map_layout(structure)
    order = layout.axes_order
    return (
        tuple((layout.row_bands(c), layout.col_bands(c)) for c in layout.sectors),
        tuple(tuple(structure.block_shape(key)[a] for a in order) for key in structure.block_order),
    )


Rect = tuple[int, int, int, int, int, int, tuple[int, ...], tuple[int, ...]]
"""``(row offset, rows, row extent, column offset, columns, column extent, split, cells)``."""


@plan_cache(cost=lambda rects: sum(len(rs) for _, rs in rects))
def _rects(structure: TensorStructure) -> tuple[tuple[Sector, tuple[Rect, ...]], ...]:
    """Each coupled sector's grid, cut into rectangles of equally sized cells.

    Parameters
    ----------
    structure : TensorStructure
        The structure to cut up.

    Returns
    -------
    tuple
        Per coupled sector, one [Rect][tenet.map_view.Rect] per pair of stripes: where
        the rectangle starts, how many bands it spans each way, the shape its cells
        split into, and the block indices of its cells in row-major order.

    Notes
    -----
    A pure function of ``structure``, cached beside [map_layout][tenet.map_layout] for
    the reason [_tables][tenet.map_view._tables] is: the cut is the same for every
    tensor of that structure, and rebuilding it inside the assembly would cost what it
    saves. It holds indices and extents, never a value and never an array.

    The rectangles are contiguous only because ``_bands`` lays the bands out
    configuration by configuration; the two facts are one design, and separating them
    would leave this walking a scattered set of rows, which no slice can express. Every
    band of a stripe carries the same degeneracies, so every cell of a rectangle has the
    same shape and the split is always defined.
    """
    layout = map_layout(structure)
    cells = dict(layout.grid)
    out: list[tuple[Sector, tuple[Rect, ...]]] = []
    for c in layout.sectors:
        rows = _stripes(structure, structure.out_axes, layout.row_bands(c))
        cols = _stripes(structure, structure.in_axes, layout.col_bands(c))
        grid = cells[c]
        out.append(
            (
                c,
                tuple(
                    (
                        ro,
                        len(ri),
                        math.prod(sr),
                        co,
                        len(ci),
                        math.prod(sc),
                        (len(ri), len(ci), *sr, *sc),
                        tuple(grid[i][j] for i in ri for j in ci),
                    )
                    for ro, ri, sr in rows
                    for co, ci, sc in cols
                ),
            )
        )
    return tuple(out)


def _stripes(
    structure: TensorStructure,
    axes: tuple[int, ...],
    bands: tuple[tuple[FusionTree, int, int], ...],
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    """One side's bands run-grouped by degeneracies: ``(offset, band indices, shape)``.

    A run is one or more whole configurations laid side by side, so its bands agree on
    their *individual* degeneracies and not merely on the product: a cell of the stripe
    always has the stripe's shape. Adjacent configurations that happen to share a shape
    -- every configuration, at uniform degeneracies -- merge into one run, which is what
    keeps the rectangle count at the number of distinct shapes rather than at the number
    of configurations.
    """
    stripes: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for i, (tree, offset, _) in enumerate(bands):
        dims = _degeneracies(structure, axes, tree)
        if stripes and stripes[-1][2] == dims:
            start, members, shape = stripes[-1]
            stripes[-1] = (start, (*members, i), shape)
        else:
            stripes.append((offset, (i,), dims))
    return tuple(stripes)


_MUTABLE = frozenset({"numpy", "torch"})
"""Backends whose arrays accept an in-place write; every other one concatenates."""


def _concatenated(
    blocks: tuple[Any, ...],
    layout: MapLayout,
    bands: tuple[
        tuple[tuple[tuple[FusionTree, int, int], ...], tuple[tuple[FusionTree, int, int], ...]],
        ...,
    ],
    order: tuple[int, ...],
    permuted: bool,
) -> tuple[Any, ...]:
    """Assemble by pure concatenation -- the immutable-backend path, and the reference.

    Parameters
    ----------
    blocks : tuple of array
        The reduced blocks, in ``block_order``.
    layout : MapLayout
        Their layout, already looked up.
    bands : tuple
        The per-sector band tables from [_tables][tenet.map_view._tables].
    order : tuple of int
        ``layout.axes_order``.
    permuted : bool
        Whether ``order`` is anything but the identity.

    Returns
    -------
    tuple of array
        One matrix per coupled sector, in ``layout.sectors`` order.
    """
    if not blocks:  # no block, hence no backend to resolve against and nothing to move
        return ()

    # resolved once, called once per block: see tenet.backend
    backend = ar.infer_backend(blocks[0])
    transpose = lib_fn(backend, "transpose")
    reshape = lib_fn(backend, "reshape")
    concatenate = lib_fn(backend, "concatenate")

    mats: list[Any] = []
    for (_, cells), (rbands, cbands) in zip(layout.grid, bands, strict=True):
        rows = []
        for (_, _, dr), indices in zip(rbands, cells, strict=True):
            parts = []
            for i, (_, _, dc) in zip(indices, cbands, strict=True):
                block = blocks[i]
                if permuted:
                    block = transpose(block, order)
                parts.append(reshape(block, (dr, dc)))
            rows.append(parts[0] if len(parts) == 1 else concatenate(parts, axis=1))
        mats.append(rows[0] if len(rows) == 1 else concatenate(rows, axis=0))
    return tuple(mats)


def gather(
    structure: TensorStructure, blocks: tuple[Any, ...]
) -> tuple[tuple[Any, ...] | None, tuple[Any, ...]]:
    """[assemble][tenet.map_view.assemble], plus the blocks of the result where they are free.

    Parameters
    ----------
    structure : TensorStructure
        The structure the blocks belong to.
    blocks : tuple of array
        One reduced block per key of ``structure.block_order``, in public axis order.

    Returns
    -------
    mats : tuple of array or None
        One matrix per coupled sector, in ``map_layout(structure).sectors`` order, or
        ``None`` where the gather is deferred -- see
        [data][tenet.SymmetricTensor.data] for which backend defers and why.
    views : tuple of array
        The blocks of the result: views **into** ``mats`` where they were gathered, and
        the given blocks themselves where they were not.

    Notes
    -----
    The in-place assembly reshapes each block's *destination* slice to write through it,
    so it has already built every block of the result as a view into the matrix it is
    writing -- the second half of what ``views`` would later recompute. Handing them back
    turns a tensor built from blocks and then read as blocks from two passes over the
    grid into one, which is the shape of every plan applier: it is given blocks and its
    result is read as blocks. The concatenating path builds no such destination and
    returns ``None``.

    These are the same views ``views`` cuts -- the same elements of the same buffer with
    the same strides -- so a tensor's blocks do not depend on which door it came in
    through. The tests assert that against the ``views`` walk, block for block.
    """
    if not blocks:  # no block, hence no matrix and no backend to ask about either
        return (), ()
    if ar.infer_backend(blocks[0]) not in _MUTABLE:
        # Nothing to gather *now*. The blocks handed in are the blocks of the result,
        # and on an immutable backend that is the whole of what ``blocks`` promises:
        # there is no memory to alias, so "a view into the storage" and "the same
        # values" are the same statement. Building the matrices here would be pure
        # concatenation -- under a JAX trace, one graph node per block with a ``pad`` in
        # its backward pass -- for a tensor that may never be asked for one. See
        # [data][tenet.SymmetricTensor.data].
        return None, blocks

    layout = map_layout(structure)
    bands, shapes = _tables(structure)
    order = layout.axes_order
    identity = tuple(range(structure.ndim))
    permuted = order != identity

    ref = blocks[0]
    backend = ar.infer_backend(ref)
    dtype = ar.to_backend_dtype(ar.get_dtype_name(ref), like=ref)
    transpose = lib_fn(backend, "transpose")
    inverse = tuple(sorted(identity, key=order.__getitem__))
    cut: list[Any] = [None] * structure.num_blocks
    mats: list[Any] = []
    for (c, cells), (rbands, cbands) in zip(layout.grid, bands, strict=True):
        out = ar.do("empty", layout.shape(c), dtype=dtype, like=ref)
        for (_, ro, dr), indices in zip(rbands, cells, strict=True):
            for i, (_, co, dc) in zip(indices, cbands, strict=True):
                block = blocks[i]
                if permuted:
                    block = transpose(block, order)
                # The *destination* is reshaped, never the source. Splitting the
                # slice's two axes into the block's only subdivides strides, so this
                # ``.reshape`` is a view on NumPy and on PyTorch alike (asserted with
                # ``shares_memory`` in the tests) and the write lands in ``out``.
                # Reshaping the source instead is the copy this path exists to avoid.
                dest = out[ro : ro + dr, co : co + dc].reshape(shapes[i])
                dest[...] = block
                cut[i] = transpose(dest, inverse) if permuted else dest
        mats.append(out)
    return tuple(mats), tuple(cut)


def assemble(structure: TensorStructure, blocks: tuple[Any, ...]) -> tuple[Any, ...]:
    """The reduced blocks of ``structure`` gathered into its coupled-sector matrices.

    Parameters
    ----------
    structure : TensorStructure
        The structure the blocks belong to.
    blocks : tuple of array
        One reduced block per key of ``structure.block_order``, in public axis order.

    Returns
    -------
    tuple of array
        One matrix per coupled sector, in ``map_layout(structure).sectors`` order --
        the storage a [SymmetricTensor][tenet.SymmetricTensor] holds.

    Notes
    -----
    The trust boundary's other half: a caller who hands the constructor per-fusion-tree
    blocks gets them gathered here, once, and every later operation reads the matrices.
    Which of the two assemblies runs is decided on the blocks' **backend** and never on
    their shapes -- see the module docstring. [gather][tenet.map_view.gather] is this
    with the result's own blocks kept as well, and the gather itself deferred where it is
    not free; that is what the constructor takes, and this is what forces it.
    """
    layout = map_layout(structure)
    bands, _ = _tables(structure)
    order = layout.axes_order
    mats, _ = gather(structure, blocks)
    if mats is not None:
        return mats
    return _concatenated(blocks, layout, bands, order, order != tuple(range(structure.ndim)))


def views(structure: TensorStructure, data: tuple[Any, ...]) -> tuple[Any, ...]:
    """The coupled-sector matrices of ``structure`` cut back into reduced blocks.

    Parameters
    ----------
    structure : TensorStructure
        The structure the matrices belong to.
    data : tuple of array
        One matrix per coupled sector, in ``map_layout(structure).sectors`` order.

    Returns
    -------
    tuple of array
        One block per key of ``structure.block_order``, in public axis order. Every one
        is a **view** into its matrix on either path; nothing is copied and nothing is
        allocated but the tuple.

    Notes
    -----
    The inverse of [assemble][tenet.map_view.assemble], and what
    [blocks][tenet.SymmetricTensor.blocks] answers with.

    On NumPy the grid is walked a rectangle at a time -- see the module docstring for
    the cut -- so the number of array calls is the number of distinct cell shapes rather
    than the number of blocks. Every other backend takes the cell-by-cell walk, on the
    same gate and for the same reason the plan applier does: whether trading Python
    iterations for array operations pays is a property of the backend and not of the
    size of the work, and it has been measured only on NumPy. The two walks take the
    same views of the same matrices, so they agree bit for bit.
    """
    if not data:  # no coupled sector, hence no block
        return ()

    layout = map_layout(structure)
    bands, shapes = _tables(structure)
    order = layout.axes_order
    identity = tuple(range(structure.ndim))
    inverse = tuple(sorted(identity, key=order.__getitem__))
    permuted = order != identity

    backend = ar.infer_backend(data[0])
    reshape = lib_fn(backend, "reshape")
    transpose = lib_fn(backend, "transpose")

    blocks: list[Any] = [None] * structure.num_blocks
    if backend == "numpy":
        # axis 2 + inverse[a] of a split rectangle is public axis a
        public = (0, 1, *(2 + inverse[a] for a in range(structure.ndim)))
        for mat, (_, rects) in zip(data, _rects(structure), strict=True):
            for ro, nr, dr, co, nc, dc, split, indices in rects:
                if len(indices) == 1:
                    # a rectangle of one cell has nothing to group -- splitting it and
                    # swapping its axes would build the cell the plain slice already is
                    i = indices[0]
                    piece = reshape(mat[ro : ro + dr, co : co + dc], shapes[i])
                    blocks[i] = transpose(piece, inverse) if permuted else piece
                    continue
                cut = reshape(mat[ro : ro + nr * dr, co : co + nc * dc], (nr, dr, nc, dc))
                cut = reshape(transpose(cut, (0, 2, 1, 3)), split)
                if permuted:
                    cut = transpose(cut, public)
                cursor = 0
                for row in cut:
                    for cell in row:
                        blocks[indices[cursor]] = cell
                        cursor += 1
        return tuple(blocks)

    for mat, ((_, cells), (rbands, cbands)) in zip(
        data, zip(layout.grid, bands, strict=True), strict=True
    ):
        for (_, ro, dr), indices in zip(rbands, cells, strict=True):
            for i, (_, co, dc) in zip(indices, cbands, strict=True):
                # basic slicing, not a dispatched call: every backend spells a
                # contiguous 2-D slice the same way (as in ops.fusion._unapply)
                piece = reshape(mat[ro : ro + dr, co : co + dc], shapes[i])
                blocks[i] = transpose(piece, inverse) if permuted else piece
    return tuple(blocks)


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

    Notes
    -----
    The tensor's own storage, keyed by sector rather than positionally: the matrices are
    the arrays ``t`` holds, not copies of them, so writing into one writes into ``t``.
    """
    return dict(zip(map_layout(t.structure).sectors, t.data, strict=True))


@plan_cache(cost=len)
def _slots(structure: TensorStructure) -> tuple[tuple[int, int, int, int, int], ...]:
    """Per block of ``block_order``, the cell it occupies: ``(c, row, rows, col, cols)``.

    Parameters
    ----------
    structure : TensorStructure
        The structure to tabulate.

    Returns
    -------
    tuple
        ``(sector position, row offset, row extent, column offset, column extent)``
        indexed by block. The sector is given as its index into
        ``map_layout(structure).sectors``, which is the order the matrices are held in.

    Notes
    -----
    The grid walk [to_matrices][tenet.to_matrices] does inline, inverted so that a
    *term* -- which knows its destination block, not its position in the grid -- can
    find its slot in one lookup. Cached beside [map_layout][tenet.map_layout] for the
    same reason [_tables][tenet.map_view._tables] is.

    **The sector is a position, not a label.** The tensor holds its matrices as a tuple
    in that order, so a term reaches its matrix by index; keying on the sector would hash
    a sector object once per term, which on a bend is one hash per block of both operands
    and was measurable next to the array work it stands in front of.
    """
    layout = map_layout(structure)
    bands, _ = _tables(structure)
    slots: list[Any] = [None] * structure.num_blocks
    for pos, ((_c, cells), (rbands, cbands)) in enumerate(zip(layout.grid, bands, strict=True)):
        for (_, ro, dr), indices in zip(rbands, cells, strict=True):
            for i, (_, co, dc) in zip(indices, cbands, strict=True):
                slots[i] = (pos, ro, dr, co, dc)
    return tuple(slots)


def _real(coeff: complex) -> Any:
    """A coefficient with no imaginary part, as a real scalar -- the plan layers' rule."""
    return coeff.real if getattr(coeff, "imag", 0) == 0 else coeff


@plan_cache(cost=lambda terms: len(terms) if terms else 1)
def _normalized(
    terms: tuple[tuple[int, int, complex], ...],
) -> tuple[tuple[int, int, complex], ...] | None:
    """``terms`` with every zero-imaginary coefficient made real, or ``None`` if one is not.

    Parameters
    ----------
    terms : tuple of (int, int, complex)
        ``(source block, target block, coefficient)``.

    Returns
    -------
    tuple of (int, int, complex) or None
        The same terms with real coefficients where the imaginary part is zero, so that a
        real tensor stays real; ``None`` where a coefficient is genuinely complex and
        [lower_plan][tenet.map_view.lower_plan] therefore declines.

    Notes
    -----
    Cached with the plan, not recomputed per call: it is one pass over the terms, and
    ``lower_plan`` is called twice per contraction on plans whose terms number in the
    hundreds of thousands. On the belief-propagation workload that pass alone was 10% of
    the run.
    """
    out = tuple((src, dst, _real(coeff)) for src, dst, coeff in terms)
    return None if any(isinstance(coeff, complex) for _, _, coeff in out) else out


def scaled(block: Any, coeff: Any) -> Any:
    """``block * coeff`` as a temporary, in one named place.

    The plan appliers reach this only for a term that both accumulates into an
    already-written destination and carries a coefficient: ``out=`` scales or
    accumulates, never both. Every other scaled term rides the ``out=`` of a write that
    was happening anyway. Named rather than spelled inline so the extra pass is
    countable -- the operator form is invisible to an ``autoray`` spy.
    """
    return block * coeff


@plan_cache(cost=lambda _: 1)
def is_identity_plan(
    structure: TensorStructure,
    perm: tuple[int, ...],
    terms: tuple[tuple[int, int, complex], ...],
) -> bool:
    """Whether the plan asks for nothing: every block back where it was, unscaled.

    Parameters
    ----------
    structure : TensorStructure
        The structure the plan builds.
    perm : tuple of int
        The plan's single per-block axis permutation.
    terms : tuple of (int, int, complex)
        ``(source block, target block, coefficient)``.

    Returns
    -------
    bool
        True when ``perm`` is the identity and ``terms`` is the identity map with unit
        coefficients, so applying the plan would rebuild the tensor it read.

    Notes
    -----
    Not a micro-optimization: a contraction whose axes need no bend composes a restore
    that *is* the identity, and it is one term per block. On an SU(2) rank-8
    intermediate that is 613,468 terms walked to hand back the tensor they were read
    from, measured at 0.4 s of a 0.47 s ``tensordot``. The predicate costs one pass over
    the terms and is cached with them, so it is paid once per distinct plan against a
    saving paid on every call.

    It lives here rather than beside the plan appliers because both of them ask it: the
    applier to hand its tensor straight back, and [lower_plan][tenet.map_view.lower_plan]
    to hand the tensor's own coupled-sector matrices straight back.
    """
    if perm != tuple(range(len(perm))) or len(terms) != structure.num_blocks:
        return False
    # every block once, in place, unscaled -- the destination set is checked too, so a
    # plan that repeats one block and drops another cannot pass for the identity
    return (
        all(src == dst and coeff == 1 for src, dst, coeff in terms)
        and len({dst for _, dst, _ in terms}) == structure.num_blocks
    )


def lower_plan(
    t: "SymmetricTensor",
    structure: TensorStructure,
    perm: tuple[int, ...],
    terms: tuple[tuple[int, int, complex], ...],
) -> dict[Sector, Any] | None:
    """``to_matrices`` of the tensor ``(perm, terms)`` would build, assembled from ``t``.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor the plan reads, *before* the plan is applied.
    structure : TensorStructure
        The plan's ``new_structure`` -- the structure of the tensor it would build.
    perm : tuple of int
        The plan's single per-block axis permutation.
    terms : tuple of (int, int, complex)
        ``(source block, target block, coefficient)``, the plan's own terms.

    Returns
    -------
    dict of Sector to array or None
        One matrix per coupled sector, bit-identical to ``to_matrices`` of the tensor
        the plan builds; ``None`` where this route does not apply and the caller must
        take the ordinary one.

    Notes
    -----
    The fusion of "apply the plan" and "lower the result", and the one place the two
    happen at once. Applying a plan on its own writes one array per term -- a transposed
    view, materialised by the scalar multiply when the coefficient is not 1 -- and
    lowering then copies every one of them into its sector matrix, so a block crosses
    memory twice for one pass' worth of movement. Composing the plan's permutation with
    ``axes_order`` gives the one transpose that takes a *source* block straight to matrix
    form. YASTN fuses the same two steps for the same reason, its meta carrying order and
    destination slot together (its NumPy backend's ``transpose_and_merge``).

    **The source is read out of ``t.data``, never out of ``t.blocks``.** A source cell is
    one 2-D slice of the source matrix split into the block's axes -- a view, by the same
    stride argument the assembly uses -- so only the sources a term names are cut, and the
    cut of the *whole* tensor that ``blocks`` performs never happens. Composing the source
    layout's ``axes_order`` into the plan's permutation is what makes that possible: the
    resulting transpose takes a cell of ``t``'s storage straight to the destination's
    matrix order, and it is the identity whenever the two layouts agree, which is a bend
    that moves the partition without reordering the legs. Reading ``blocks`` here is what
    a small tensor cannot afford: an MPS tensor lowers a plan for every operation, and a
    lowering that starts by cutting every block of its operand pays the assembly the
    matrices were made storage to remove.

    The terms are not walked one at a time. On NumPy they go through
    [batch_plan][tenet.ops.batch.batch_plan] -- the grouping ``apply_plan`` already used,
    by destination shape and then by multiplicity -- so a whole bucket is gathered,
    scaled and summed as array operations and only its result is written into the
    destination matrix. That is what makes the Python work proportional to the buckets
    rather than to the terms: 2,457 dispatches for 296,953 terms on a rank-8 SU(2) bend.
    What the grouping declines is the tail, and it keeps the term walk, with one
    transposed view per distinct source rather than one per term.

    **A bucket is one allocation**, and that is the half of the grouping that decides
    what it costs on a big plan. The gather ``stacked[take]`` is already a buffer of the
    bucket's own -- an index array never returns a view -- so the scaling and the
    accumulation are written back into it. Out of place each is another array the size of
    the bucket, and a bucket is a multiple of the *tensor*: on a rank-6 SU(2) intermediate
    whose one group holds 2,141 of its 2,407 terms the multiply's own temporary was 70 MB
    against a 32 MB result, and writing that one and the accumulations back into the
    gather took the lowering from 20.7 ms to 14.1 ms, with the same array calls carrying
    the same operands in the same order -- bit for bit. Grouping trades Python
    iterations for array operations, so its cost is the traffic those operations move,
    and a bucket wide enough to be worth grouping is wide enough for a spare pass over
    it to be what the call costs.

    ``None`` is returned where the route does not apply: no blocks, an immutable backend
    (which has no ``out=``, and whose ``to_matrices`` concatenates), or a genuinely
    complex coefficient, which would have to promote the destination's dtype.
    Simplification: every provider shipped here has real permutation and bending
    coefficients, so the promotion case is left to the ordinary route rather than given
    a dtype rule of its own.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, to_matrices
    >>> from tenet.map_view import lower_plan
    >>> from tenet.ops.permutation import permutation_plan
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> plan = permutation_plan(t.structure, (2, 0, 1))
    >>> got = lower_plan(t, plan.new_structure, plan.axes, plan.terms)
    >>> want = to_matrices(t.transpose((2, 0, 1)))
    >>> all(got[c].tobytes() == want[c].tobytes() for c in want)
    True
    """
    # ``t.backend`` and not ``ar.infer_backend(t.data[0])``: on an immutable backend the
    # matrices are not built until something asks for one, and declining the route is not
    # asking. Reading the backend off ``data`` would gather every block of the source --
    # under a trace, a graph node each -- to discover that this function cannot use them.
    if not t.structure.num_blocks or t.backend not in _MUTABLE:
        return None
    # ``out=`` is what makes this one pass, and torch refuses it under autograd
    # ("functions with out=... arguments don't support automatic differentiation").
    # A watched tensor is not mutable in the sense ``_MUTABLE`` means, so the route
    # declines and the ordinary applier -- which builds a temporary per term, and
    # which is what torch always took before -- runs instead.
    if getattr(t._first_block(), "requires_grad", False):
        return None
    if structure == t.structure and is_identity_plan(structure, perm, terms):
        # the plan rebuilds what it reads, and the tensor already holds the matrices
        return to_matrices(t)
    normalized = _normalized(terms)
    if normalized is None:
        return None
    # the same grouping ``apply_plan`` batches with, on the same backend gate, and the
    # same reason: the trade of Python iterations for array operations has been measured
    # only on NumPy. Here the group's result is written straight into the destination
    # matrix, so the blocks the applier would have built never exist.
    from tenet.ops.batch import batch_plan, cast_coefficients

    groups, loose = (
        batch_plan(structure, perm, normalized)
        if ar.infer_backend(t.data[0]) == "numpy"
        else ((), normalized)
    )

    layout = map_layout(structure)
    _, shapes = _tables(structure)
    slots = _slots(structure)
    order = tuple(perm[i] for i in layout.axes_order)

    source = map_layout(t.structure)
    _, src_shapes = _tables(t.structure)
    src_slots = _slots(t.structure)
    src_mats = t.data
    # ``order`` reads a block in *public* axis order; a source cell arrives in the source
    # layout's own, so composing the two is the one transpose that takes a cell of ``t``'s
    # storage straight to the destination's -- and it is the identity whenever the two
    # layouts agree, which is most of a bend.
    src_order = source.axes_order
    inverse = tuple(sorted(range(len(src_order)), key=src_order.__getitem__))
    composed = tuple(inverse[a] for a in order)
    permuted = composed != tuple(range(len(composed)))

    ref = t.data[0]

    def cell(s: int) -> Any:
        """Source block ``s`` as a view into ``t``'s storage, in *source matrix* order."""
        c, ro, dr, co, dc = src_slots[s]
        # the mirror of the destination write below, and a view for the same reason:
        # splitting a 2-D slice's two axes into the block's only subdivides strides
        return src_mats[c][ro : ro + dr, co : co + dc].reshape(src_shapes[s])

    dtype = ar.to_backend_dtype(ar.get_dtype_name(ref), like=ref)
    mats = [ar.do("empty", shape, dtype=dtype, like=ref) for shape in layout.shapes]

    transpose = lib_fn(t.backend, "transpose")
    multiply = lib_fn(t.backend, "multiply")
    add = lib_fn(t.backend, "add")

    written: set[int] = set()
    for srcs, buckets in groups:
        stacked = ar.do("stack", tuple(cell(s) for s in srcs))
        stacked = ar.do("transpose", stacked, (0, *(ax + 1 for ax in composed)))
        for take, coeff, width, dsts in buckets:
            # ``take`` is an index array, so the gather is already a fresh buffer of the
            # bucket's own: scaling it and summing it *in place* is what keeps a bucket
            # to one allocation. Out of place each is another array the size of the
            # bucket -- the one the multiply returns and one per accumulation step --
            # which on a plan whose buckets hold most of the tensor is the dominant
            # traffic, not the Python that issues it.
            rows = stacked[take]
            multiply(rows, cast_coefficients(coeff, stacked), out=rows)
            rows = ar.do("reshape", rows, (len(dsts), width, *ar.shape(rows)[1:]))
            acc = rows[:, 0]
            for i in range(1, width):
                # ``acc`` is a slice of ``rows`` and the summands are its other slices,
                # so accumulating into it neither aliases a summand nor outlives them
                add(acc, rows[:, i], out=acc)
            for p, dst in enumerate(dsts):
                written.add(dst)
                c, ro, dr, co, dc = slots[dst]
                mats[c][ro : ro + dr, co : co + dc].reshape(shapes[dst])[...] = acc[p]

    # one cut and one transpose per distinct source, not per term -- ``composed`` is
    # per-plan and only the coefficient is per-term, so terms sharing a source would
    # otherwise recompute a byte-identical view (the rule ``ops.repartition._looped``
    # follows for the same reason: 2.87 terms per source at SU(2), exactly 1.00 at U(1))
    moved = {}
    for src in {s for s, _, _ in loose}:
        view = cell(src)
        moved[src] = transpose(view, composed) if permuted else view
    for src, dst, coeff in loose:
        block = moved[src]
        c, ro, dr, co, dc = slots[dst]
        # the *destination* is reshaped, never the source -- see ``to_matrices``
        dest = mats[c][ro : ro + dr, co : co + dc].reshape(shapes[dst])
        if dst not in written:
            written.add(dst)
            if coeff == 1:
                dest[...] = block
            else:
                multiply(block, coeff, out=dest)
        elif coeff == 1:
            add(dest, block, out=dest)
        else:
            # a second source summing into one destination *with* a coefficient is the
            # one term that still needs a temporary: ``out=`` scales or accumulates, not
            # both. Only a multi-term (non-Abelian) expansion produces these, and that is
            # also why ordering the group so a coefficient-carrying term writes first buys
            # nothing: such a group's terms *all* carry coefficients, so the first write
            # already scales through its ``out=`` and every later one still needs this.
            add(dest, scaled(block, coeff), out=dest)

    if len(written) != structure.num_blocks:
        raise ValueError(
            f"lower_plan: the plan fills {len(written)} of {structure.num_blocks} target "
            f"blocks -- {t.provider.name}'s coefficients dropped terms"
        )
    return dict(zip(layout.sectors, mats, strict=True))


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
        If a sector is missing, unknown, its matrix has the wrong shape
        (invariant 11), or the matrices do not share one dtype.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet import from_matrices, to_matrices
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> from_matrices(t.structure, to_matrices(t)) == t
    True

    Notes
    -----
    Zero-copy, and now trivially so: the matrices *are* the storage, so this checks them
    against the layout and hands them to the tensor. Nothing is allocated but the tuple,
    and the blocks are cut out of them only if someone asks for
    [blocks][tenet.SymmetricTensor.blocks].

    The refusals above are the whole check, and they are deliberately spelled over the
    *matrices* rather than over the blocks that come out of them. The matrices are the
    untrusted input; the blocks are views cut to the shapes ``structure`` dictates, so
    re-reading those shapes back off them would be one pass per block -- 613,468 of them
    on a rank-8 SU(2) intermediate, a fifth of the contraction that builds it -- to
    confirm the reshape that has not even happened yet. One touch per coupled sector says
    the same thing (#328).
    """
    from tenet.tensor import SymmetricTensor

    layout = map_layout(structure)
    _check(layout, mats)
    return SymmetricTensor.from_data(structure, tuple(mats[c] for c in layout.sectors))


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
    derived from the ``side`` metadata the legs already carry. Materializing an
    ``(out..., in...)`` reordering here would violate
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
    the predicate is pure map-view metadata and its callers span more than one
    ``ops`` module.
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
    # The dtype half of the tensor constructor's check, moved to where it belongs: every
    # block is a view of one of these matrices, so one touch per coupled sector decides
    # what one touch per block used to (#328).
    dtypes = {mats[c].dtype for c in layout.sectors}
    if len(dtypes) > 1:
        raise ValueError(
            f"from_matrices: matrices must share one dtype, got {sorted(map(str, dtypes))}"
        )
