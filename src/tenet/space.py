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
from math import prod
from typing import TYPE_CHECKING

from tenet.symmetry.base import ClebschGordanData, Sector, _DualFusionRules, _HashMemo, requires

if TYPE_CHECKING:
    from tenet.leg import Leg

__all__ = ["GradedSpace", "ProductSpace"]


@dataclass(frozen=True, slots=True)
class GradedSpace(_HashMemo):
    """Immutable graded space: ``sectors`` is sorted by sector, all ``m >= 1``.

    Use :meth:`new` to build one from a mapping; the raw constructor takes an
    already-canonical tuple and does not validate.
    """

    provider: _DualFusionRules
    sectors: tuple[tuple[Sector, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.provider, self.sectors)))

    def __hash__(self) -> int:
        return self._hash

    @classmethod
    def new(
        cls,
        provider: _DualFusionRules,
        sectors: Mapping[Sector, int] | Iterable[tuple[Sector, int]],
    ) -> "GradedSpace":
        """Normalizing constructor: sorts, rejects duplicates and ``m <= 0``.

        All sectors must share one type, and that type must be the provider's
        own sector type (taken as ``type(provider.unit)``) — the provider
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
        # Sector is an order=True dataclass, so the key is comparable; ty does
        # not synthesize dataclass ordering methods here
        return cls(
            provider,
            tuple(sorted(pairs, key=lambda p: p[0])),  # ty: ignore[no-matching-overload]
        )

    def direct_sum(self, other: "GradedSpace") -> "GradedSpace":
        """``self ⊕ other`` at the label level: union of sectors, degeneracies added.

        This says nothing about ``dual`` — the flag lives on the ``Leg``
        (invariant 2), not on the space, so whether two legs' conventions are
        summable is checked where the legs are, by the tensor-level
        :func:`tenet.direct_sum`; ``tenet.flip`` is the numerical route for
        normalising a dual convention beforehand. TensorKit's ``⊕`` must compare
        ``isdual`` here precisely because its flag lives on the space.
        """
        if other.provider != self.provider:
            raise TypeError(
                f"cannot direct-sum a space over provider {other.provider.name} "
                f"with one over {self.provider.name}; a direct sum never casts "
                "between symmetries"
            )
        merged = dict(self.sectors)
        for a, m in other.sectors:
            merged[a] = merged.get(a, 0) + m
        return GradedSpace.new(self.provider, merged)

    def degeneracy(self, a: Sector) -> int:
        """``m_a``, or ``0`` if ``a`` is absent (so filtering reads as a predicate)."""
        # Simplification: linear scan; spaces hold a handful of sectors. Add a cached
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
        """``Σ m_a d_a``: the full dense dimension. Requires ``ClebschGordanData``."""
        requires(self.provider, ClebschGordanData)
        # requires() above; raise-based check does not narrow
        return sum(m * self.provider.irrep_dim(a) for a, m in self.sectors)  # ty: ignore[unresolved-attribute]

    def sector_offset(self, a: Sector) -> int:
        """Start of ``a``'s slab in the dense layout. Requires ``ClebschGordanData``.

        Sectors are laid out in canonical order, each contributing a contiguous
        slab of ``m_a * d_a``; the within-slab index is ``alpha * d_a + m``.
        """
        requires(self.provider, ClebschGordanData)
        offset = 0
        for b, m in self.sectors:
            if b == a:
                return offset
            offset += m * self.provider.irrep_dim(b)  # ty: ignore[unresolved-attribute]  # see above
        raise KeyError(a)


@dataclass(frozen=True, slots=True)
class ProductSpace:
    """An ordered tuple of legs, viewed as one factor of ``Hom(domain, codomain)``.

    A TensorMap-level value type (docs/design.md "``ProductSpace``"): nobody constructs one
    to make an ordinary five-axis tensor, and ``SymmetricTensor.codomain`` keeps
    returning plain legs. It appears on :class:`~tenet.map_view.TensorMapView`.

    Frozen, hashable, array-free; equal iff its ordered legs are equal (``name``
    included, since ``Leg`` compares it). Composition, however, ignores ``name`` —
    see :meth:`matches`.
    """

    legs: tuple["Leg", ...]

    @property
    def provider(self) -> _DualFusionRules:
        if not self.legs:
            raise ValueError("an empty ProductSpace has no provider")
        return self.legs[0].provider

    @property
    def dim(self) -> int:
        """``Π leg.space.dim`` — the full dense dimension. Requires ``ClebschGordanData``."""
        return prod(leg.space.dim for leg in self.legs)

    @property
    def reduced_dim(self) -> int:
        """``Π leg.space.reduced_dim``. Any provider."""
        return prod(leg.space.reduced_dim for leg in self.legs)

    def matches(self, other: "ProductSpace") -> int | None:
        """``None`` if composable against ``other``, else the first offending position.

        Composability is ``(space, dual, order)``, exactly (invariant 2): the
        effective categorical objects ``V`` or ``V*`` must agree, in order. ``side``
        is not compared (OUT meets IN by construction) and neither is ``name``,
        which is user bookkeeping. Dimensions are never compared on their own — a
        charge-reversed U(1) partner has the same dimension and the wrong space.

        Differing lengths report the first position past the shorter tuple; callers
        that can say something better about leg counts should check them first.
        """
        for i, (x, y) in enumerate(zip(self.legs, other.legs, strict=False)):
            if x.space != y.space or x.dual != y.dual:
                return i
        if len(self.legs) != len(other.legs):
            return min(len(self.legs), len(other.legs))
        return None
