"""SU(2) fusion provider: triangle-rule fusion, self-duality, Clebsch-Gordan tensors.

Sectors are labelled by the doubled spin ``two_j`` so labels stay exact integers:
``SU2Sector(0)`` is the singlet, ``SU2Sector(1)`` spin-1/2, ``SU2Sector(2)`` spin-1.

``dual(a) == a`` because SU(2) is self-dual as a *label* map. The Z-isomorphism
``V_j -> V_j^*`` carrying the Frobenius-Schur sign ``(-1)^(2j)`` is **not** the
identity and is not implemented here (Milestone 4), so ``leg.dual`` must never be
treated as a no-op (invariant 2). F-symbols, R-symbols and B-symbols are likewise
Milestone 4.
"""

from dataclasses import dataclass

import numpy as np

from tenet.symmetry._su2_coeff import cg_tensor, triangle
from tenet.symmetry.base import Sector

__all__ = ["SU2", "SU2_GAUGE", "SU2Provider", "SU2Sector"]

SU2_GAUGE = "3j=condon-shortley;cg=condon-shortley;f=tks-su2irrep;r=tks-su2irrep;fs=tks-su2irrep"
"""Gauge fingerprint (racah / TensorKitSectors conventions); belongs in plan-cache keys."""


@dataclass(frozen=True, slots=True, order=True)
class SU2Sector(Sector):
    """An SU(2) irrep labelled by ``two_j = 2j >= 0``."""

    two_j: int

    def __post_init__(self) -> None:
        if not isinstance(self.two_j, int) or isinstance(self.two_j, bool):
            raise TypeError(f"two_j must be an int, got {type(self.two_j).__name__}")
        if self.two_j < 0:
            raise ValueError(f"two_j must be non-negative, got {self.two_j}")


@dataclass(frozen=True, slots=True)
class SU2Provider:
    """SU(2) provider. Array-free and hashable; CG arrays live in a module-level cache."""

    name: str = "SU2"

    @property
    def unit(self) -> SU2Sector:
        return SU2Sector(0)

    def dual(self, a: SU2Sector) -> SU2Sector:
        return a

    def fusion(self, a: SU2Sector, b: SU2Sector) -> tuple[SU2Sector, ...]:
        """``|ja-jb| <= jc <= ja+jb`` in integer steps, ascending in ``two_j``."""
        lo, hi = abs(a.two_j - b.two_j), a.two_j + b.two_j
        return tuple(SU2Sector(two_j) for two_j in range(lo, hi + 1, 2))

    def n_symbol(self, a: SU2Sector, b: SU2Sector, c: SU2Sector) -> int:
        return int(triangle(a.two_j, b.two_j, c.two_j))

    def qdim(self, a: SU2Sector) -> float:
        return float(a.two_j + 1)

    def irrep_dim(self, a: SU2Sector) -> int:
        return a.two_j + 1

    def cgc(self, a: SU2Sector, b: SU2Sector, c: SU2Sector) -> np.ndarray:
        """Shape ``(d_a, d_b, d_c, 1)``, read-only; magnetic indices descending.

        The trailing axis is the multiplicity label ``mu``; SU(2) is multiplicity-free
        so it always has size 1, but consumers must still index it.
        """
        if not triangle(a.two_j, b.two_j, c.two_j):
            raise ValueError(f"{c} does not appear in the fusion of {a} and {b}")
        return cg_tensor(a.two_j, b.two_j, c.two_j)[..., np.newaxis]


SU2 = SU2Provider()
"""Module-level singleton, used as ``GradedSpace(provider=SU2, ...)``."""
