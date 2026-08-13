"""Symmetry foundation: sector labels, fusion provider protocol, Trivial provider.

Sectors are pure labels; all symmetry behaviour (dual, fusion, dimensions) lives
on the provider. Providers are frozen, hashable, array-free values so they can
sit inside ``TensorStructure`` (invariant 8).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CapabilityError",
    "ClebschGordan",
    "FusionProvider",
    "QuantumDimension",
    "Sector",
    "Trivial",
    "TrivialProvider",
    "TrivialSector",
    "requires",
]


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


class CapabilityError(TypeError):
    """Raised when a provider lacks a capability an operation requires."""


def requires(provider: FusionProvider, capability: type) -> None:
    """Raise :class:`CapabilityError` unless ``provider`` implements ``capability``."""
    if not isinstance(provider, capability):
        raise CapabilityError(
            f"{type(provider).__name__} does not provide capability {capability.__name__}"
        )


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


Trivial = TrivialProvider()
"""Module-level singleton, used as ``GradedSpace(provider=Trivial, ...)``."""
