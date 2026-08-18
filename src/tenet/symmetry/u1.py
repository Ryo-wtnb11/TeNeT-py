"""U(1) symmetry: integer-charge sectors, unique fusion, non-trivial dual.

Charges are plain ``int``; product symmetries (U(1)xU(1), U(1)xSU(2)) are a
later product-sector concern, not a reason to make every charge a tuple.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tenet.symmetry.base import Sector, bend_unique, permute_unique_tree

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree
    from tenet.structure import FusionBlockKey

__all__ = ["U1", "U1Provider", "U1Sector"]


@dataclass(frozen=True, slots=True, order=True)
class U1Sector(Sector):  # ty: ignore[subclass-of-dataclass-with-order]  # deliberate, see Sector
    """A U(1) irrep, labelled by its integer charge.

    Parameters
    ----------
    charge : int
        The conserved charge; negation is duality.

    Examples
    --------
    >>> from tenet.symmetry import U1, U1Sector
    >>> U1.fusion(U1Sector(1), U1Sector(2))
    (U1Sector(charge=3),)
    >>> U1.dual(U1Sector(1))
    U1Sector(charge=-1)
    """

    charge: int


@dataclass(frozen=True, slots=True)
class U1Provider:
    """U(1): abelian, multiplicity-free, one-dimensional irreps.

    Use the module-level singleton [U1][tenet.symmetry.U1] rather than
    constructing one; ``name`` is an identity label that participates in
    equality, not a configuration knob. The capability contract of every
    method is documented once, on the protocols in ``tenet.symmetry`` —
    only behaviour that *differs* from a protocol carries a docstring here.

    Examples
    --------
    >>> from tenet.symmetry import U1, U1Sector
    >>> U1.fusion(U1Sector(-1), U1Sector(1))
    (U1Sector(charge=0),)
    >>> U1.n_symbol(U1Sector(1), U1Sector(1), U1Sector(2))
    1
    >>> U1.irrep_dim(U1Sector(5))
    1
    """

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

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """One term, coefficient 1: charge addition is commutative and ``F = R = 1``."""
        # ``self``'s Sector parameters are narrowed to this symmetry's own sector
        # type; deliberate per-symmetry specialization the unparameterized
        # protocol cannot express, so the checker misreads the conformance.
        return permute_unique_tree(self, tree, perm)  # ty: ignore[invalid-argument-type]

    def bend_right(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        """One term, coefficient 1: for an Abelian irrep ``B = N``, ``dim = 1``, ``FS = 1``."""
        # ``self``'s Sector parameters are narrowed to this symmetry's own sector
        # type; deliberate per-symmetry specialization the unparameterized
        # protocol cannot express, so the checker misreads the conformance.
        return bend_unique(self, key, right=True, dual=dual)  # ty: ignore[invalid-argument-type]

    def bend_left(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        # ``self``'s Sector parameters are narrowed to this symmetry's own sector
        # type; deliberate per-symmetry specialization the unparameterized
        # protocol cannot express, so the checker misreads the conformance.
        return bend_unique(self, key, right=False, dual=dual)  # ty: ignore[invalid-argument-type]

    def z_matrix(self, a: U1Sector) -> np.ndarray:
        """``Z = [[1]]``, read-only: ``V_q`` is one-dimensional and the FS phase is 1."""
        return _Z

    def frobenius_schur(self, a: U1Sector) -> float:
        """``chi_a = 1``: every U(1) irrep is one-dimensional with a real pairing."""
        return 1.0

    def twist(self, a: U1Sector) -> float:
        """``theta_a = 1``: bosonic Abelian, trivial twist."""
        return 1.0


_CGC = np.ones((1, 1, 1, 1))
_CGC.flags.writeable = False

_Z = np.ones((1, 1))
_Z.flags.writeable = False

U1 = U1Provider()
"""Module-level singleton, used as ``GradedSpace(provider=U1, ...)``."""
