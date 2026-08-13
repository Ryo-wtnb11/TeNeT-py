"""Graded representation spaces: ``V = ⊕_a C^{m_a} ⊗ V_a``.

A :class:`GradedSpace` is a provider plus a canonical sector→degeneracy mapping.
It is the only place degeneracies live, and it is immutable, hashable and
array-free so it can sit inside ``TensorStructure`` (invariant 8).

It carries **no** ``dual`` flag: ``side`` and ``dual`` are per-leg metadata
(invariant 2), so the same ``GradedSpace`` can be shared by a dual and a
non-dual leg. There is deliberately no ``dual()`` method here — ask the ``Leg``.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from tenet.symmetry.base import ClebschGordan, FusionProvider, Sector, requires

__all__ = ["GradedSpace"]


@dataclass(frozen=True, slots=True)
class GradedSpace:
    """Immutable graded space: ``sectors`` is sorted by sector, all ``m >= 1``.

    Use :meth:`new` to build one from a mapping; the raw constructor takes an
    already-canonical tuple and does not validate.
    """

    provider: FusionProvider
    sectors: tuple[tuple[Sector, int], ...]

    @classmethod
    def new(
        cls,
        provider: FusionProvider,
        sectors: Mapping[Sector, int] | Iterable[tuple[Sector, int]],
    ) -> "GradedSpace":
        """Normalizing constructor: sorts, rejects duplicates and ``m <= 0``.

        All sectors must share one type, and that type must be the provider's
        own sector type (taken as ``type(provider.unit)``) — the ``FusionProvider``
        protocol exposes no sector-type attribute, so ``unit`` is the proxy.
        """
        pairs = list(sectors.items() if isinstance(sectors, Mapping) else sectors)
        expected = type(provider.unit)
        seen: set[Sector] = set()
        for a, m in pairs:
            if type(a) is not expected:
                raise TypeError(
                    f"sector {a!r} of type {type(a).__name__} does not belong to "
                    f"provider {provider.name} (expects {expected.__name__})"
                )
            if m <= 0:
                raise ValueError(f"degeneracy of {a!r} must be positive, got {m}")
            if a in seen:
                raise ValueError(f"duplicate sector {a!r}")
            seen.add(a)
        return cls(provider, tuple(sorted(pairs, key=lambda p: p[0])))

    def degeneracy(self, a: Sector) -> int:
        """``m_a``, or ``0`` if ``a`` is absent (so filtering reads as a predicate)."""
        # ponytail: linear scan; spaces hold a handful of sectors. Add a cached
        # dict if profiling ever shows this hot.
        for b, m in self.sectors:
            if b == a:
                return m
        return 0

    def __contains__(self, a: Sector) -> bool:
        return self.degeneracy(a) > 0

    def __iter__(self) -> Iterator[Sector]:
        """Sectors in canonical order."""
        return (a for a, _ in self.sectors)

    def __len__(self) -> int:
        """Number of sectors — *not* a dimension."""
        return len(self.sectors)

    @property
    def reduced_dim(self) -> int:
        """``Σ m_a``: what reduced ndarray blocks are made of. Any provider."""
        return sum(m for _, m in self.sectors)

    @property
    def dim(self) -> int:
        """``Σ m_a d_a``: the full dense dimension. Requires ``ClebschGordan``."""
        requires(self.provider, ClebschGordan)
        return sum(m * self.provider.irrep_dim(a) for a, m in self.sectors)

    def sector_offset(self, a: Sector) -> int:
        """Start of ``a``'s slab in the dense layout. Requires ``ClebschGordan``.

        Sectors are laid out in canonical order, each contributing a contiguous
        slab of ``m_a * d_a``; the within-slab index is ``alpha * d_a + m``.
        """
        requires(self.provider, ClebschGordan)
        offset = 0
        for b, m in self.sectors:
            if b == a:
                return offset
            offset += m * self.provider.irrep_dim(b)
        raise KeyError(a)
