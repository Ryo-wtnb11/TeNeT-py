"""U(1) symmetry: integer-charge sectors, unique fusion, non-trivial dual.

Charges are plain ``int``; product symmetries (U(1)xU(1), U(1)xSU(2)) are a
later product-sector concern, not a reason to make every charge a tuple.
"""

from dataclasses import dataclass

import numpy as np

from tenet.symmetry.base import Sector

__all__ = ["U1", "U1Provider", "U1Sector"]


@dataclass(frozen=True, slots=True, order=True)
class U1Sector(Sector):
    """A U(1) irrep, labelled by its integer charge."""

    charge: int


@dataclass(frozen=True, slots=True)
class U1Provider:
    """U(1): abelian, multiplicity-free, one-dimensional irreps."""

    name: str = "U1"

    @property
    def unit(self) -> U1Sector:
        return U1Sector(0)

    def dual(self, a: U1Sector) -> U1Sector:
        return U1Sector(-a.charge)

    def fusion(self, a: U1Sector, b: U1Sector) -> tuple[U1Sector, ...]:
        return (U1Sector(a.charge + b.charge),)

    def n_symbol(self, a: U1Sector, b: U1Sector, c: U1Sector) -> int:
        return int(c.charge == a.charge + b.charge)

    def qdim(self, a: U1Sector) -> float:
        return 1.0

    def irrep_dim(self, a: U1Sector) -> int:
        return 1

    def cgc(self, a: U1Sector, b: U1Sector, c: U1Sector) -> np.ndarray:
        """Shape ``(1, 1, 1, 1)``, read-only. Raises on a fusion-forbidden triple."""
        if not self.n_symbol(a, b, c):
            raise ValueError(f"U(1) fusion forbids {a} x {b} -> {c}")
        return _CGC


_CGC = np.ones((1, 1, 1, 1))
_CGC.flags.writeable = False

U1 = U1Provider()
"""Module-level singleton, used as ``GradedSpace(provider=U1, ...)``."""
