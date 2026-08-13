"""SU(2) fusion provider: triangle-rule fusion, self-duality, Clebsch-Gordan tensors.

Sectors are labelled by the doubled spin ``two_j`` so labels stay exact integers:
``SU2Sector(0)`` is the singlet, ``SU2Sector(1)`` spin-1/2, ``SU2Sector(2)`` spin-1.

``dual(a) == a`` because SU(2) is self-dual as a *label* map. The Z-isomorphism
``V_j -> V_j^*`` carrying the Frobenius-Schur sign ``(-1)^(2j)`` is **not** the
identity and is not implemented here (Milestone 4), so ``leg.dual`` must never be
treated as a no-op (invariant 2).

F-, R- and B-symbols and Frobenius-Schur signs *are* available, through the
:class:`~tenet.symmetry.base.RecouplingData` capability; their gauge is pinned to
the vendored TensorKitSectors fixtures (see ``SU2_GAUGE``). ``permute_tree`` is
built on them, so ``transpose`` is total for SU(2). Still absent: the dense
``z_matrix`` of the Z-isomorphism and ``bend_right``/``bend_left``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tenet.symmetry import _su2_coeff
from tenet.symmetry._su2_coeff import cg_tensor, triangle
from tenet.symmetry.base import Sector, permute_braided_tree

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree

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

    def f_symbol(
        self,
        a: SU2Sector,
        b: SU2Sector,
        c: SU2Sector,
        d: SU2Sector,
        e: SU2Sector,
        f: SU2Sector,
    ) -> float:
        """``[F^{abc}_d]_{e,f}``; ``e`` is the inner line of ``((ab)c)``, ``f`` of ``(a(bc))``.

        Real in this gauge, so downstream conjugation of domain-side coefficients
        is a no-op. Raises if any vertex has ``n_symbol > 1`` (unreachable for
        SU(2), asserted so the scalar-valued contract is not merely documentation).
        """
        for x, y, z in ((a, b, e), (e, c, d), (b, c, f), (a, f, d)):
            if self.n_symbol(x, y, z) > 1:
                raise ValueError(
                    f"{self.name}: f_symbol is scalar-valued but "
                    f"N^{z!r}_{{{x!r},{y!r}}} > 1; matrix-valued F is not supported"
                )
        return _su2_coeff.f_symbol(a.two_j, b.two_j, c.two_j, d.two_j, e.two_j, f.two_j)

    def r_symbol(self, a: SU2Sector, b: SU2Sector, c: SU2Sector) -> int:
        """``R^{ab}_c = (-1)^(ja + jb - jc)``, exactly ``+-1``; SU(2) braiding is symmetric."""
        return _su2_coeff.r_symbol(a.two_j, b.two_j, c.two_j)

    def b_symbol(self, a: SU2Sector, b: SU2Sector, c: SU2Sector) -> float:
        """``B^{ab}_c``, derived from :meth:`f_symbol`; real, of unit modulus."""
        return _su2_coeff.b_symbol(a.two_j, b.two_j, c.two_j)

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """Artin-braid expansion; SU(2) is symmetric, so ``perm`` fixes the braid."""
        return permute_braided_tree(self, tree, perm)

    def frobenius_schur(self, a: SU2Sector) -> int:
        """``chi_a = (-1)^(2j)``: ``+1`` for integer spin, ``-1`` for half-integer."""
        return _su2_coeff.frobenius_schur(a.two_j)


SU2 = SU2Provider()
"""Module-level singleton, used as ``GradedSpace(provider=SU2, ...)``."""
