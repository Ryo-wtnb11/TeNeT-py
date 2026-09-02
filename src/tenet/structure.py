"""Tensor-level structure: ordered legs, the fusion basis, and block shapes.

A [TensorStructure][tenet.TensorStructure] owns *all* categorical metadata of a tensor and no
numbers: it is frozen, hashable and array-free (invariant 8) so it can serve as
a JAX treedef while the blocks are the leaves.

A block is labelled by a [FusionBlockKey][tenet.FusionBlockKey] — a *pair* of fusion trees
(output side, input side) sharing one coupled sector. A tuple of one sector per
axis is **not** enough: for SU(2) two distinct trees can carry identical
external sectors (see ``tenet.fusion_tree``). External sectors are recovered
*from* a key via [axis_sectors][tenet.TensorStructure.axis_sectors], never the other way.

``block_order`` is a sorted tuple and it is the storage contract: ``blocks[i]``
belongs to ``block_order[i]``. It is a pure function of the structure, stable
across processes, resting on the ordering chain ``Sector`` → ``FusionTree`` →
``FusionBlockKey`` (all ``order=True``).

Block *shapes* follow **public axis order** (invariant 7): legs ``(C1, D1, C2,
D2)`` give shape ``(m_c1, m_d1, m_c2, m_d2)``, not the ``(out..., in...)``
regrouping. ``out_axes``/``in_axes`` exist so the coupled-sector matrix
lowering ([tenet.to_matrices][]) is written without touching storage.

Derived values are cached by module-level ``functools.cache`` functions
keyed on the structure, not by ``cached_property``: the dataclass stays
genuinely frozen and the cache is shared between equal structures.
"""

from dataclasses import dataclass
from dataclasses import replace as _replace
from functools import cache
from itertools import product

from tenet.cache import plan_cache
from tenet.fusion_tree import FusionTree, coupled_sectors, fusion_trees
from tenet.leg import OUT, Leg
from tenet.space import GradedSpace
from tenet.symmetry.base import Sector, _DualFusionRules, _HashMemo

__all__ = ["FusionBlockKey", "TensorStructure"]


@dataclass(frozen=True, slots=True, order=True)
class FusionBlockKey(_HashMemo):
    """An output/input fusion-tree pair with a shared coupled sector.

    Parameters
    ----------
    output_tree : FusionTree
        The tree over the OUT legs.
    input_tree : FusionTree
        The tree over the IN legs; must couple to the same sector.

    Notes
    -----
    Both trees list their uncoupled sectors in **public axis order** restricted
    to their side, already dual-resolved (``Leg.fused_sector``). Field order is
    load-bearing: it defines the canonical sort used by ``block_order``.
    """

    output_tree: FusionTree
    input_tree: FusionTree

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.output_tree, self.input_tree)))

    def __hash__(self) -> int:
        return self._hash

    @property
    def coupled(self) -> Sector:
        """The shared coupled sector. Equality of the two is a ``validate`` check.

        Returns
        -------
        Sector
            ``output_tree.coupled``.
        """
        return self.output_tree.coupled


@dataclass(frozen=True)
class TensorStructure(_HashMemo):
    """Ordered legs plus everything derivable from them. Immutable and hashable.

    Parameters
    ----------
    legs : tuple of Leg
        The tensor's legs, in public axis order. At least one.

    Raises
    ------
    ValueError
        If ``legs`` is empty — a leg-less scalar has no provider to enumerate
        against.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, TensorStructure
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> s = TensorStructure((Leg(V, OUT), Leg(V, IN)))
    >>> s.ndim, s.num_blocks
    (2, 2)
    >>> s.out_axes, s.in_axes
    ((0,), (1,))
    >>> s.block_shape(s.block_order[0])
    (2, 2)
    """

    legs: tuple[Leg, ...]

    def __post_init__(self) -> None:
        if not self.legs:
            # Simplification: a leg-less scalar has no provider to enumerate against.
            # Give TensorStructure an explicit provider field if scalars ever matter.
            raise ValueError("TensorStructure needs at least one leg")
        object.__setattr__(self, "legs", tuple(self.legs))
        object.__setattr__(self, "_hash", hash((self.legs,)))

    def __hash__(self) -> int:
        return self._hash

    @property
    def provider(self) -> _DualFusionRules:
        """The shared provider. ``validate()`` is what checks the legs agree.

        Returns
        -------
        provider
            The first leg's provider.
        """
        return self.legs[0].provider

    @property
    def ndim(self) -> int:
        """Number of legs.

        Returns
        -------
        int
            ``len(self.legs)``.
        """
        return len(self.legs)

    @property
    def out_axes(self) -> tuple[int, ...]:
        """Public axis indices with ``side is OUT``, ascending.

        Returns
        -------
        tuple of int
            The OUT axes.
        """
        return _axes(self)[0]

    @property
    def in_axes(self) -> tuple[int, ...]:
        """Public axis indices with ``side is IN``, ascending.

        Returns
        -------
        tuple of int
            The IN axes.
        """
        return _axes(self)[1]

    @property
    def block_order(self) -> tuple[FusionBlockKey, ...]:
        """Every structurally allowed key, sorted. The storage contract.

        Returns
        -------
        tuple of FusionBlockKey
            The canonical block order: ``blocks[i]`` belongs to
            ``block_order[i]``.
        """
        return _block_order(self)

    @property
    def num_blocks(self) -> int:
        """Number of structurally allowed blocks.

        Returns
        -------
        int
            ``len(self.block_order)``.

        Notes
        -----
        Counted off the cross product, so asking how many blocks there are does
        not build them: the count is the reason the layout never has to.
        """
        return _num_blocks(self)

    def index_of(self, key: FusionBlockKey) -> int:
        """Position of ``key`` in [block_order][tenet.TensorStructure.block_order].

        Parameters
        ----------
        key : FusionBlockKey
            A key of this structure.

        Returns
        -------
        int
            The index of ``key``.

        Raises
        ------
        KeyError
            If ``key`` is foreign to this structure.
        """
        return _index_map(self)[key]

    def axis_sectors(self, key: FusionBlockKey) -> tuple[Sector, ...]:
        """One **space** sector per public axis, de-dualized via ``Leg.space_sector``.

        Parameters
        ----------
        key : FusionBlockKey
            A key of this structure.

        Returns
        -------
        tuple of Sector
            One space sector per public axis.

        Raises
        ------
        KeyError
            If ``key`` is foreign to this structure.

        Notes
        -----
        A lookup into the per-structure table built once by
        ``_axis_sectors_table``; foreign keys raise ``KeyError`` as before.
        """
        return _axis_sectors_table(self)[self.index_of(key)]

    def block_shape(self, key: FusionBlockKey) -> tuple[int, ...]:
        """Degeneracies in **public** axis order (invariant 7).

        Parameters
        ----------
        key : FusionBlockKey
            A key of this structure.

        Returns
        -------
        tuple of int
            The block's shape: one degeneracy per public axis.

        Raises
        ------
        KeyError
            If ``key`` is foreign to this structure.
        """
        return _block_shape_table(self)[self.index_of(key)]

    @property
    def block_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Every block's shape, aligned with [block_order][tenet.TensorStructure.block_order].

        The whole-table companion to
        [block_shape][tenet.TensorStructure.block_shape], for a caller walking
        ``block_order`` rather than asking about one key. Same information; the
        difference is cost, and it is not small.

        ``block_shape(key)`` enters two structure-keyed caches -- one to turn the key
        into an index, one for the table -- and each entry pays a hash of this
        ``TensorStructure``, which reaches through its legs to their spaces and sectors.
        Called once per block in a loop, a tensor with a thousand blocks pays two
        thousand deep hashes to read a tuple that was already built. This property pays
        one. Hoisting it out of ``SymmetricTensor.__post_init__`` alone took 17% off a
        cold SU(2) plan workload (#307).

        Returns
        -------
        tuple of tuple of int
            One shape per key of ``block_order``, in that order.

        Examples
        --------
        >>> from tenet import GradedSpace, IN, OUT, Leg, TensorStructure
        >>> from tenet.symmetry import U1, U1Sector
        >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
        >>> s = TensorStructure((Leg(V, OUT), Leg(V, IN)))
        >>> s.block_shapes == tuple(s.block_shape(k) for k in s.block_order)
        True

        """
        return _block_shape_table(self)

    def validate(self, key: FusionBlockKey | None = None) -> None:
        """Check the legs, and either every key in ``block_order`` or just ``key``.

        Parameters
        ----------
        key : FusionBlockKey or None, optional
            The single key to check; ``None`` (the default) checks every key in
            ``block_order``.

        Raises
        ------
        ValueError
            If the legs disagree on the provider, a tree's rank does not match
            its side's leg count, a tree fails ``FusionTree.validate``, an
            uncoupled label does not map back into its leg's space, or the two
            coupled sectors disagree.

        Notes
        -----
        Explicit and total (invariant 11): one provider for all legs; each tree's
        rank matches its side's leg count and passes ``FusionTree.validate``; every
        uncoupled label maps back into the corresponding leg's space; the two
        coupled sectors agree.
        """
        provider = self.provider
        for i, leg in enumerate(self.legs):
            if leg.provider != provider:
                raise ValueError(
                    f"leg {i} has provider {leg.provider.name}, "
                    f"but leg 0 has provider {provider.name}"
                )
        if key is None:
            for k in self.block_order:
                self.validate(k)
            return

        if key.output_tree.coupled != key.input_tree.coupled:
            raise ValueError(
                f"coupled sectors disagree: output {key.output_tree.coupled!r} "
                f"vs input {key.input_tree.coupled!r}"
            )
        for side, tree, axes in (
            ("output", key.output_tree, self.out_axes),
            ("input", key.input_tree, self.in_axes),
        ):
            if tree.rank != len(axes):
                raise ValueError(
                    f"{side}_tree has rank {tree.rank}, but there are {len(axes)} {side} legs"
                )
            tree.validate(provider)
            for u, ax in zip(tree.uncoupled, axes, strict=True):
                leg = self.legs[ax]
                a = leg.space_sector(u)
                if a not in leg.space:
                    raise ValueError(f"axis {ax}: {a!r} is not a sector of the leg's space")


# --- derived values: module-level caches keyed on the (frozen, hashable) structure ---
# Simplification: unbounded caches, deliberately — every one below except
# ``_block_shape_table`` is either keyed on ``_pattern`` (so a growing bond adds no
# entry) or holds two small tuples. ``tenet.cache`` states that distinction once, and
# bounds ``_block_shape_table``, which is the one here that reads the degeneracies.
#
# Which key each one takes is load-bearing (#248). A ``GradedSpace`` hashes its
# degeneracies, so a structure whose bond degeneracies moved is a new key even though its
# sector set did not — and a DMRG sweep moves them at every bond, every sweep, until the
# state settles. Everything below that is a function of the *sectors* alone
# (``_side_trees``, ``_block_cross``, ``_num_blocks``, ``_block_order`` and the two
# tables aligned with it) is therefore keyed on the structure's ``_pattern``, the same
# legs with every degeneracy 1; only ``_block_shape_table``, which reads the
# degeneracies, is keyed on the structure itself.


@cache
def _flat(space: GradedSpace) -> GradedSpace:
    """``space`` with every degeneracy 1 — the part of it a *block set* depends on."""
    return GradedSpace(space.provider, tuple((a, 1) for a, _ in space.sectors))


@cache
def _pattern(s: TensorStructure) -> TensorStructure:
    """``s`` with every degeneracy 1, or ``s`` itself when it already is that.

    Two structures share a pattern exactly when they have the same sectors, sides and
    duals per leg — which is exactly when they have the same ``block_order``.
    """
    legs = tuple(_replace(leg, space=_flat(leg.space)) for leg in s.legs)
    return s if legs == s.legs else TensorStructure(legs)


@cache
def _axes(s: TensorStructure) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(i for i, leg in enumerate(s.legs) if leg.side is OUT),
        tuple(i for i, leg in enumerate(s.legs) if leg.side is not OUT),
    )


@cache
def _side_trees(s: TensorStructure, axes: tuple[int, ...]) -> dict[Sector, tuple[FusionTree, ...]]:
    """One side's fusion trees, sorted, grouped by the sector they couple to.

    The empty side falls out for free: the empty product yields one empty
    assignment, whose only tree is the rank-0 tree at the unit.
    """
    if (p := _pattern(s)) is not s:
        return _side_trees(p, axes)
    legs = tuple(s.legs[i] for i in axes)
    by_coupled: dict[Sector, list[FusionTree]] = {}
    for assignment in product(*(leg.sectors for leg in legs)):
        u = tuple(leg.fused_sector(a) for leg, a in zip(legs, assignment, strict=True))
        for c in coupled_sectors(s.provider, u):
            by_coupled.setdefault(c, []).extend(fusion_trees(s.provider, u, c))
    return {c: tuple(sorted(trees)) for c, trees in by_coupled.items()}


@cache
def _block_cross(
    s: TensorStructure,
) -> tuple[dict[FusionTree, int], dict[Sector, tuple[FusionTree, ...]]]:
    """``block_order`` as the per-sector cross product it is, without the keys.

    Returns where each output tree's run of blocks starts in ``block_order`` —
    in that order, so the keys of the dict *are* the output trees — and each
    coupled sector's input trees in the order a run lists them. An index is
    therefore a base plus the input tree's position in its sector, which is how
    [map_layout][tenet.map_layout] fills its grid with no lookup per block.

    The two sides are independent given the coupled sector, so they are
    enumerated apart: ``S^n_out + S^n_in`` leg assignments rather than the
    ``S^ndim`` joint walk, which is the difference between a sum and a product
    of exponentials. Sorting likewise falls on the *trees* — a few thousand —
    and not on the finished keys, of which there is one per block.
    """
    if (p := _pattern(s)) is not s:
        return _block_cross(p)
    out_by_c, in_by_c = _side_trees(s, s.out_axes), _side_trees(s, s.in_axes)
    bases: dict[FusionTree, int] = {}
    n = 0
    for ot in sorted(t for c, trees in out_by_c.items() if c in in_by_c for t in trees):
        bases[ot] = n
        n += len(in_by_c[ot.coupled])
    return bases, in_by_c


@cache
def _num_blocks(s: TensorStructure) -> int:
    """``len(_block_order(s))``, from the cross product's sizes alone."""
    bases, in_by_c = _block_cross(s)
    return sum(len(in_by_c[ot.coupled]) for ot in bases)


@cache
def _block_order(s: TensorStructure) -> tuple[FusionBlockKey, ...]:
    """Pair up output- and input-side trees, per shared coupled sector.

    Sorted by construction: a key orders on ``(output_tree, input_tree)``, the
    output trees are walked in order and each one's partners are already sorted.
    """
    bases, in_by_c = _block_cross(s)
    return tuple(FusionBlockKey(ot, it) for ot in bases for it in in_by_c[ot.coupled])


@cache
def _index_map(s: TensorStructure) -> dict[FusionBlockKey, int]:
    if (p := _pattern(s)) is not s:
        return _index_map(p)
    return {k: i for i, k in enumerate(_block_order(s))}


@cache
def _axis_sectors_table(s: TensorStructure) -> tuple[tuple[Sector, ...], ...]:
    """Space sectors per axis for every key, aligned with ``block_order``.

    The duality convention is inverted here and nowhere else (``Leg.space_sector``).
    """
    if (p := _pattern(s)) is not s:
        return _axis_sectors_table(p)
    out_axes, in_axes = _axes(s)

    def row(key: FusionBlockKey) -> tuple[Sector, ...]:
        by_axis = dict(zip(out_axes, key.output_tree.uncoupled, strict=True))
        by_axis |= dict(zip(in_axes, key.input_tree.uncoupled, strict=True))
        return tuple(leg.space_sector(by_axis[ax]) for ax, leg in enumerate(s.legs))

    return tuple(row(k) for k in _block_order(s))


@plan_cache(cost=len)
def _block_shape_table(s: TensorStructure) -> tuple[tuple[int, ...], ...]:
    """Block shapes in public axis order for every key, aligned with ``block_order``."""
    return tuple(
        tuple(leg.degeneracy(a) for leg, a in zip(s.legs, sectors, strict=True))
        for sectors in _axis_sectors_table(s)
    )
