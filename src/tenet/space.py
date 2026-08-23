"""Graded representation spaces: ``V = ⊕_a C^{m_a} ⊗ V_a``.

A [GradedSpace][tenet.GradedSpace] is a provider plus a canonical sector→degeneracy mapping.
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

    Use [new][tenet.GradedSpace.new] to build one from a mapping; the raw
    constructor takes an already-canonical tuple and does not validate.

    Parameters
    ----------
    provider : provider
        The symmetry provider whose sectors label the grading.
    sectors : tuple of (Sector, int)
        Already-canonical ``(sector, degeneracy)`` pairs, sorted by sector —
        not validated here; go through [new][tenet.GradedSpace.new] instead.

    Examples
    --------
    >>> from tenet import GradedSpace
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(1): 1, U1Sector(0): 2})
    >>> tuple(V)
    (U1Sector(charge=0), U1Sector(charge=1))
    >>> V.reduced_dim, V.dim
    (3, 3)
    >>> V.degeneracy(U1Sector(0))
    2
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

        Parameters
        ----------
        provider : provider
            The symmetry provider the sectors must belong to.
        sectors : Mapping[Sector, int] or iterable of (Sector, int)
            Sector → degeneracy, in any order.

        Returns
        -------
        GradedSpace
            The canonical space: pairs sorted by sector.

        Raises
        ------
        TypeError
            If a sector is not of the provider's own sector type.
        ValueError
            If a degeneracy is ``<= 0``, or a sector appears twice.

        Notes
        -----
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

        Parameters
        ----------
        other : GradedSpace
            The other summand; must be over the same provider.

        Returns
        -------
        GradedSpace
            The label-level direct sum, canonically sorted.

        Raises
        ------
        TypeError
            If ``other`` is over a different provider — a direct sum never
            casts between symmetries.

        Notes
        -----
        This says nothing about ``dual`` — the flag lives on the ``Leg``
        (invariant 2), not on the space, so whether two legs' conventions are
        summable is checked where the legs are, by the tensor-level
        [tenet.direct_sum][]; ``tenet.flip_dual`` is the numerical route for
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
        """``m_a``, or ``0`` if ``a`` is absent (so filtering reads as a predicate).

        Parameters
        ----------
        a : Sector
            The sector to look up.

        Returns
        -------
        int
            The degeneracy of ``a``, or ``0`` when ``a`` is not a sector of
            this space.
        """
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
        """``Σ m_a``: what reduced ndarray blocks are made of. Any provider.

        Returns
        -------
        int
            The total degeneracy dimension.
        """
        return sum(m for _, m in self.sectors)

    @property
    def dim(self) -> int:
        """``Σ m_a d_a``: the full dense dimension.

        Returns
        -------
        int
            The dense carrier-space dimension.

        Raises
        ------
        CapabilityError
            If the provider lacks ``ClebschGordanData`` — without integer irrep
            dimensions there is no dense dimension.
        """
        requires(self.provider, ClebschGordanData)
        # requires() above; raise-based check does not narrow
        return sum(m * self.provider.irrep_dim(a) for a, m in self.sectors)  # ty: ignore[unresolved-attribute]

    def sector_offset(self, a: Sector) -> int:
        """Start of ``a``'s slab in the dense layout.

        Parameters
        ----------
        a : Sector
            A sector of this space.

        Returns
        -------
        int
            The offset of ``a``'s contiguous slab in the dense layout.

        Raises
        ------
        CapabilityError
            If the provider lacks ``ClebschGordanData``.
        KeyError
            If ``a`` is not a sector of this space.

        Notes
        -----
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

    Parameters
    ----------
    legs : tuple of Leg
        The ordered legs making up this factor.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, ProductSpace
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> out = ProductSpace((Leg(V, OUT),))
    >>> out.reduced_dim
    3
    >>> out.matches(ProductSpace((Leg(V, IN),))) is None  # side is not compared
    True
    >>> out.matches(ProductSpace((Leg(V, IN, dual=True),)))  # dual is
    0

    Notes
    -----
    A TensorMap-level value type: nobody constructs one
    to make an ordinary five-axis tensor, and ``SymmetricTensor.codomain`` keeps
    returning plain legs. It appears on [TensorMapView][tenet.TensorMapView].

    Frozen, hashable, array-free; equal iff its ordered legs are equal (``name``
    included, since ``Leg`` compares it). Composition, however, ignores ``name`` —
    see [matches][tenet.ProductSpace.matches].
    """

    legs: tuple["Leg", ...]

    @property
    def provider(self) -> _DualFusionRules:
        """The legs' shared provider.

        Returns
        -------
        provider
            The first leg's provider.

        Raises
        ------
        ValueError
            If the ``ProductSpace`` is empty — it then has no provider.
        """
        if not self.legs:
            raise ValueError("an empty ProductSpace has no provider")
        return self.legs[0].provider

    @property
    def dim(self) -> int:
        """``Π leg.space.dim`` — the full dense dimension.

        Returns
        -------
        int
            The product of the legs' dense dimensions.

        Raises
        ------
        CapabilityError
            If the provider lacks ``ClebschGordanData`` (via ``GradedSpace.dim``).
        """
        return prod(leg.space.dim for leg in self.legs)

    @property
    def reduced_dim(self) -> int:
        """``Π leg.space.reduced_dim``. Any provider.

        Returns
        -------
        int
            The product of the legs' degeneracy dimensions.
        """
        return prod(leg.space.reduced_dim for leg in self.legs)

    def matches(self, other: "ProductSpace") -> int | None:
        """``None`` if composable against ``other``, else the first offending position.

        Parameters
        ----------
        other : ProductSpace
            The factor to compare against, leg by leg.

        Returns
        -------
        int or None
            ``None`` when composable; otherwise the first position where the
            two disagree as ``(space, dual)``.

        Notes
        -----
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
