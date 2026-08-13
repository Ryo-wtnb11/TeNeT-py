"""Symmetry foundation: sector labels, fusion provider protocol, Trivial provider.

Sectors are pure labels; all symmetry behaviour (dual, fusion, dimensions) lives
on the provider. Providers are frozen, hashable, array-free values so they can
sit inside ``TensorStructure`` (invariant 8).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CapabilityError",
    "ClebschGordan",
    "FusionProvider",
    "PermutationCoefficients",
    "QuantumDimension",
    "Sector",
    "Trivial",
    "TrivialProvider",
    "TrivialSector",
    "requires",
]

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree


@dataclass(frozen=True, slots=True, order=True)
class Sector:
    """Marker base for sector labels.

    Subclasses are frozen, slotted, ordered dataclasses, so hashing and canonical
    sorting come for free. Comparison is only defined within one sector type;
    comparing different sector types raises ``TypeError``.
    """


class FusionProvider(Protocol):
    """Minimal fusion-category data every symmetry must supply."""

    @property
    def name(self) -> str: ...

    @property
    def unit(self) -> Sector: ...

    def dual(self, a: Sector) -> Sector: ...

    def fusion(self, a: Sector, b: Sector) -> tuple[Sector, ...]: ...

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        """Multiplicity ``N^c_ab``. Multiplicity-free providers return 0 or 1."""
        ...


@runtime_checkable
class QuantumDimension(Protocol):
    """Providers that define a quantum dimension."""

    def qdim(self, a: Sector) -> float: ...


@runtime_checkable
class ClebschGordan(Protocol):
    """Providers that can supply explicit Clebsch-Gordan tensors."""

    def irrep_dim(self, a: Sector) -> int: ...

    def cgc(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        """Shape ``(d_a, d_b, d_c, N^c_ab)``; last axis is the multiplicity label mu."""
        ...


@runtime_checkable
class PermutationCoefficients(Protocol):
    """Providers that can expand a fusion tree with permuted uncoupled lines."""

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """``((tree', coeff), ...)`` with the permuted tree equal to ``Σ coeff · tree'``.

        ``perm[j]`` is the *old* uncoupled position that becomes position ``j``
        (the same convention as ``transpose``'s ``axes``). Expansion is over
        canonical left-associated trees. Scalar coefficients are a
        multiplicity-free assumption: a provider with ``n_symbol > 1`` must raise
        rather than truncate a matrix-valued coefficient (Milestone 4).
        """
        ...


class CapabilityError(TypeError):
    """Raised when a provider lacks a capability an operation requires."""


def requires(provider: FusionProvider, capability: type) -> None:
    """Raise :class:`CapabilityError` unless ``provider`` implements ``capability``."""
    if not isinstance(provider, capability):
        raise CapabilityError(
            f"{type(provider).__name__} does not provide capability {capability.__name__}"
        )


def permute_unique_tree(
    provider: FusionProvider, tree: "FusionTree", perm: tuple[int, ...]
) -> tuple[tuple["FusionTree", complex], ...]:
    """``permute_tree`` for providers whose fusion is unique and whose F/R are 1.

    Shared by Trivial and U(1): permuting the uncoupled labels leaves exactly one
    left-associated tree with the same coupled sector, so the whole expansion is
    that tree with coefficient 1. The spine is *recomputed* rather than permuted,
    which is what makes this correct rather than a relabelling.

    Not a capability check: a provider must opt in by defining ``permute_tree``
    (fermionic parity also has unique multiplicity-free fusion and still carries a
    sign, so uniqueness alone must never be taken as permission — see #21).
    """
    from tenet.fusion_tree import fusion_trees

    uncoupled = tuple(tree.uncoupled[i] for i in perm)
    trees = fusion_trees(provider, uncoupled, tree.coupled)
    if len(trees) != 1:
        raise CapabilityError(
            f"{provider.name}: permuting {tree.uncoupled} by {perm} gives {len(trees)} trees "
            f"coupling to {tree.coupled!r}; permute_unique_tree needs exactly one"
        )
    return ((trees[0], 1.0),)


@dataclass(frozen=True, slots=True, order=True)
class TrivialSector(Sector):
    """The single sector of the trivial symmetry."""


@dataclass(frozen=True, slots=True)
class TrivialProvider:
    """Degenerate reference provider: one sector, everything trivial."""

    name: str = "Trivial"

    @property
    def unit(self) -> TrivialSector:
        return TrivialSector()

    def dual(self, a: Sector) -> Sector:
        return a

    def fusion(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        return (TrivialSector(),)

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1.0

    def irrep_dim(self, a: Sector) -> int:
        return 1

    def cgc(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return np.ones((1, 1, 1, 1))

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """One term, coefficient 1: there is a single sector and ``F = R = 1``."""
        return permute_unique_tree(self, tree, perm)


Trivial = TrivialProvider()
"""Module-level singleton, used as ``GradedSpace(provider=Trivial, ...)``."""
