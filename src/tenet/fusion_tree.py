"""Fusion trees: the categorical basis label for a list of sectors.

External sectors do **not** determine the fusion basis once the symmetry is
non-Abelian: ``(1/2, 1/2, 1/2) -> 1/2`` has two independent trees, told apart by
their internal line. A :class:`FusionTree` carries the uncoupled sectors, the
internal lines, one multiplicity label per vertex, and the coupled sector.

Conventions, stated once and enforced everywhere:

* **Left-associated only.** ``e_0 = u_0``; ``e_{k+1} in fusion(e_k, u_{k+1})``;
  ``inner == (e_1, ..., e_{N-2})`` and ``coupled == e_{N-1}``. Vertex ``k``
  carries ``mu_k in range(n_symbol(e_k, u_{k+1}, e_{k+1}))``. Other tree shapes
  are related by F-moves (Milestone 4); there is deliberately no shape field.
* **No ``isdual``.** Duality is per-leg metadata (invariant 2/4), so
  ``uncoupled`` holds already dual-resolved labels and one tree is shareable
  between tensors with different duality patterns.
* **``multiplicities`` is always length ``N-1``**, even for multiplicity-free
  providers where every entry is ``0``, so downstream code cannot be written in
  a way that only works multiplicity-free.
* **Field order is load-bearing.** ``order=True`` gives block keys a canonical
  total order (hence stable JAX treedefs); do not reorder the fields.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from tenet.symmetry.base import FusionRules, Sector, _HashMemo

__all__ = ["FusionTree", "coupled_sectors", "fusion_trees"]


@dataclass(frozen=True, slots=True, order=True)
class FusionTree(_HashMemo):
    """A left-associated fusion tree. Frozen, hashable, totally ordered."""

    uncoupled: tuple[Sector, ...]
    inner: tuple[Sector, ...]
    multiplicities: tuple[int, ...]
    coupled: Sector

    def __post_init__(self) -> None:
        n = len(self.uncoupled)
        if len(self.inner) != max(n - 2, 0):
            raise ValueError(
                f"rank-{n} tree needs {max(n - 2, 0)} inner lines, got {len(self.inner)}"
            )
        if len(self.multiplicities) != max(n - 1, 0):
            raise ValueError(
                f"rank-{n} tree needs {max(n - 1, 0)} multiplicity labels, "
                f"got {len(self.multiplicities)}"
            )
        object.__setattr__(
            self, "_hash", hash((self.uncoupled, self.inner, self.multiplicities, self.coupled))
        )

    def __hash__(self) -> int:
        return self._hash

    @property
    def rank(self) -> int:
        """``N``, the number of uncoupled sectors."""
        return len(self.uncoupled)

    def lines(self) -> tuple[Sector, ...]:
        """The left spine ``(e_0, ..., e_{N-1})``; ``()`` for ``N == 0``."""
        if self.rank == 0:
            return ()
        if self.rank == 1:
            return (self.coupled,)
        return (self.uncoupled[0], *self.inner, self.coupled)

    def vertices(self) -> tuple[tuple[Sector, Sector, Sector, int], ...]:
        """``((e_k, u_{k+1}, e_{k+1}, mu_k), ...)`` for ``k = 0 .. N-2``."""
        spine = self.lines()
        return tuple(
            zip(spine[:-1], self.uncoupled[1:], spine[1:], self.multiplicities, strict=True)
        )

    def validate(self, provider: FusionRules) -> None:
        """Raise if the tree is not a valid basis label for ``provider``."""
        if self.rank == 0:
            if self.coupled != provider.unit:
                raise ValueError(
                    f"rank-0 tree must couple to {provider.unit!r}, got {self.coupled!r}"
                )
            return
        if self.rank == 1 and self.coupled != self.uncoupled[0]:
            raise ValueError(
                f"rank-1 tree must couple to {self.uncoupled[0]!r}, got {self.coupled!r}"
            )
        for k, (a, b, c, mu) in enumerate(self.vertices()):
            n = provider.n_symbol(a, b, c)
            if n == 0:
                raise ValueError(f"vertex {k}: fusion rule forbids {a!r} x {b!r} -> {c!r}")
            if not 0 <= mu < n:
                raise ValueError(
                    f"vertex {k}: multiplicity label {mu} outside range({n}) "
                    f"for {a!r} x {b!r} -> {c!r}"
                )


def fusion_trees(
    provider: FusionRules,
    uncoupled: Sequence[Sector],
    coupled: Sector,
) -> tuple[FusionTree, ...]:
    """All valid left-associated trees for ``uncoupled -> coupled``, sorted."""
    # The ignores in this module: providers are frozen dataclasses, hashable by
    # contract (symmetry.base module docstring); ``__hash__`` is deliberately
    # not a protocol member — the member set is pinned by
    # ``test_fusion_provider_is_a_protocol`` — so the ``@cache`` wrappers see a
    # non-Hashable protocol.
    return _fusion_trees(provider, tuple(uncoupled), coupled)  # ty: ignore[invalid-argument-type]


def coupled_sectors(
    provider: FusionRules,
    uncoupled: Sequence[Sector],
) -> tuple[Sector, ...]:
    """Every sector reachable from ``uncoupled``, canonically sorted."""
    # hashable by provider contract; see the ignore rationale at _fusion_trees
    return _coupled_sectors(provider, tuple(uncoupled))  # ty: ignore[invalid-argument-type]


@cache
def _fusion_trees(
    provider: FusionRules, uncoupled: tuple[Sector, ...], coupled: Sector
) -> tuple[FusionTree, ...]:
    # hashable by provider contract; see the ignore rationale at fusion_trees
    trees = _all_trees(provider, uncoupled)  # ty: ignore[invalid-argument-type]
    return tuple(t for t in trees if t.coupled == coupled)


@cache
def _coupled_sectors(provider: FusionRules, uncoupled: tuple[Sector, ...]) -> tuple[Sector, ...]:
    # hashable by provider contract; see the ignore rationale at fusion_trees
    trees = _all_trees(provider, uncoupled)  # ty: ignore[invalid-argument-type]
    return tuple(sorted({t.coupled for t in trees}))


@cache
def _all_trees(provider: FusionRules, uncoupled: tuple[Sector, ...]) -> tuple[FusionTree, ...]:
    """Every tree over ``uncoupled``, any coupled sector, sorted."""
    if not uncoupled:
        return (FusionTree((), (), (), provider.unit),)
    if len(uncoupled) == 1:
        return (FusionTree(uncoupled, (), (), uncoupled[0]),)
    u = uncoupled[-1]
    out = []
    # hashable by provider contract; see the ignore rationale at _fusion_trees
    for tree in _all_trees(provider, uncoupled[:-1]):  # ty: ignore[invalid-argument-type]
        e = tree.coupled
        inner = (*tree.inner, e) if tree.rank >= 2 else ()
        for c in provider.fusion(e, u):
            for mu in range(provider.n_symbol(e, u, c)):
                out.append(FusionTree(uncoupled, inner, (*tree.multiplicities, mu), c))
    return tuple(sorted(out))
