"""Tensor-level structure: ordered legs, the fusion basis, and block shapes.

A :class:`TensorStructure` owns *all* categorical metadata of a tensor and no
numbers: it is frozen, hashable and array-free (invariant 8) so it can serve as
a JAX treedef while the blocks are the leaves.

A block is labelled by a :class:`FusionBlockKey` — a *pair* of fusion trees
(output side, input side) sharing one coupled sector. A tuple of one sector per
axis is **not** enough: for SU(2) two distinct trees can carry identical
external sectors (see :mod:`tenet.fusion_tree`). External sectors are recovered
*from* a key via :meth:`TensorStructure.axis_sectors`, never the other way.

``block_order`` is a sorted tuple and it is the storage contract: ``blocks[i]``
belongs to ``block_order[i]``. It is a pure function of the structure, stable
across processes, resting on the ordering chain ``Sector`` → ``FusionTree`` →
``FusionBlockKey`` (all ``order=True``).

Block *shapes* follow **public axis order** (invariant 7): legs ``(C1, D1, C2,
D2)`` give shape ``(m_c1, m_d1, m_c2, m_d2)``, not the ``(out..., in...)``
regrouping. ``out_axes``/``in_axes`` exist so the coupled-sector matrix
lowering can be written later (Milestone 3) without touching storage.

Derived values are cached by module-level :func:`functools.cache` functions
keyed on the structure, not by ``cached_property``: the dataclass stays
genuinely frozen and the cache is shared between equal structures.
"""

from dataclasses import dataclass
from functools import cache
from itertools import product

from tenet.fusion_tree import FusionTree, coupled_sectors, fusion_trees
from tenet.leg import OUT, Leg
from tenet.symmetry.base import FusionProvider, Sector, _HashMemo

__all__ = ["FusionBlockKey", "TensorStructure"]


@dataclass(frozen=True, slots=True, order=True)
class FusionBlockKey(_HashMemo):
    """An output/input fusion-tree pair with a shared coupled sector.

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
        """The shared coupled sector. Equality of the two is a ``validate`` check."""
        return self.output_tree.coupled


@dataclass(frozen=True)
class TensorStructure(_HashMemo):
    """Ordered legs plus everything derivable from them. Immutable and hashable."""

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
    def provider(self) -> FusionProvider:
        """The shared provider. ``validate()`` is what checks the legs agree."""
        return self.legs[0].provider

    @property
    def ndim(self) -> int:
        return len(self.legs)

    @property
    def out_axes(self) -> tuple[int, ...]:
        """Public axis indices with ``side is OUT``, ascending."""
        return _axes(self)[0]

    @property
    def in_axes(self) -> tuple[int, ...]:
        """Public axis indices with ``side is IN``, ascending."""
        return _axes(self)[1]

    @property
    def block_order(self) -> tuple[FusionBlockKey, ...]:
        """Every structurally allowed key, sorted. The storage contract."""
        return _block_order(self)

    @property
    def num_blocks(self) -> int:
        return len(_block_order(self))

    def index_of(self, key: FusionBlockKey) -> int:
        """Position of ``key`` in :attr:`block_order`; ``KeyError`` if foreign."""
        return _index_map(self)[key]

    def axis_sectors(self, key: FusionBlockKey) -> tuple[Sector, ...]:
        """One **space** sector per public axis, de-dualized via ``Leg.space_sector``.

        A lookup into the per-structure table built once by
        :func:`_axis_sectors_table`; foreign keys raise ``KeyError`` as before.
        """
        return _axis_sectors_table(self)[self.index_of(key)]

    def block_shape(self, key: FusionBlockKey) -> tuple[int, ...]:
        """Degeneracies in **public** axis order (invariant 7)."""
        return _block_shape_table(self)[self.index_of(key)]

    def validate(self, key: FusionBlockKey | None = None) -> None:
        """Check the legs, and either every key in ``block_order`` or just ``key``.

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
# Simplification: unbounded caches, deliberately — structures are small, few and long-lived,
# and sharing across equal structures is the point. Swap for lru_cache if that changes.


@cache
def _axes(s: TensorStructure) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(i for i, leg in enumerate(s.legs) if leg.side is OUT),
        tuple(i for i, leg in enumerate(s.legs) if leg.side is not OUT),
    )


@cache
def _block_order(s: TensorStructure) -> tuple[FusionBlockKey, ...]:
    """Pair up output- and input-side trees over every sector assignment.

    The empty side falls out for free: ``fusion_trees(p, (), c)`` is the single
    rank-0 tree when ``c is p.unit`` and nothing otherwise, so a tensor with no
    IN legs keeps exactly the unit-coupled blocks.
    """
    provider, out_axes, in_axes = s.provider, s.out_axes, s.in_axes
    keys: set[FusionBlockKey] = set()
    for assignment in product(*(leg.sectors for leg in s.legs)):
        uncoupled = tuple(leg.fused_sector(a) for leg, a in zip(s.legs, assignment, strict=True))
        out_u = tuple(uncoupled[i] for i in out_axes)
        in_u = tuple(uncoupled[i] for i in in_axes)
        for c in coupled_sectors(provider, out_u):
            keys.update(
                FusionBlockKey(ot, it)
                for ot in fusion_trees(provider, out_u, c)
                for it in fusion_trees(provider, in_u, c)
            )
    return tuple(sorted(keys))


@cache
def _index_map(s: TensorStructure) -> dict[FusionBlockKey, int]:
    return {k: i for i, k in enumerate(_block_order(s))}


@cache
def _axis_sectors_table(s: TensorStructure) -> tuple[tuple[Sector, ...], ...]:
    """Space sectors per axis for every key, aligned with ``block_order``.

    The duality convention is inverted here and nowhere else (``Leg.space_sector``).
    """
    out_axes, in_axes = _axes(s)

    def row(key: FusionBlockKey) -> tuple[Sector, ...]:
        by_axis = dict(zip(out_axes, key.output_tree.uncoupled, strict=True))
        by_axis |= dict(zip(in_axes, key.input_tree.uncoupled, strict=True))
        return tuple(leg.space_sector(by_axis[ax]) for ax, leg in enumerate(s.legs))

    return tuple(row(k) for k in _block_order(s))


@cache
def _block_shape_table(s: TensorStructure) -> tuple[tuple[int, ...], ...]:
    """Block shapes in public axis order for every key, aligned with ``block_order``."""
    return tuple(
        tuple(leg.degeneracy(a) for leg, a in zip(s.legs, sectors, strict=True))
        for sectors in _axis_sectors_table(s)
    )
