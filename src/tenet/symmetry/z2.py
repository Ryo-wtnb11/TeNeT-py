"""Z2: bosonic parity — Z2 fusion with *symmetric* braiding (Vect_Z2).

Two sectors, XOR fusion, self-dual labels, ``qdim == irrep_dim == 1``, all-ones
CGC, ``F ≡ 1``, ``B = N``, ``FS = +1``. Every one of those is shared verbatim with
``fz2.py``, the fermion-parity provider sitting one file over; the *only* thing
that differs is the braiding, and therefore the only method body that differs is
[Z2Provider.permute_tree][tenet.symmetry.Z2Provider.permute_tree].

**Why [permute_unique_tree][tenet.symmetry.permute_unique_tree] is permitted
here.** Its docstring carries a
standing warning: a provider must opt in by defining ``permute_tree``, because
fermion parity has the identical unique multiplicity-free fusion rule and still
carries a Koszul sign — *uniqueness of fusion is never permission* (#21). This
provider trips that warning head-on and answers it from somewhere else entirely:
``Z2Provider`` is **defined** to be ``Vect_Z2`` with symmetric bosonic braiding,
``R^{ab}_c ≡ +1`` for every triple, so the permutation coefficient is 1 *by the
braiding*, not by the fusion. ``fz2.py`` is the counterexample that makes the
distinction real: same fusion, ``R = -1`` for odd–odd, hence a sign. Read the two
``permute_tree`` bodies side by side and the whole bosonic/fermionic difference
in this repository is those two lines.

No gauge fingerprint, deliberately. A gauge string pins a *choice*, and there is
no choice here: every F, R, B and FS is ``+1``, which is why ``U1Provider`` and
``TrivialProvider`` carry no gauge string and do not appear in
``serialize._GAUGES`` either. The twist is ``+1`` for both sectors, so there is
nothing to record in prose either.

The bosonic Z2 whose data this reproduces is TensorKitSectors' ``Z2Irrep``, which
sits next to ``FermionParity`` there for exactly the reason two files sit next to
each other here: same fusion rules, different ``BraidingStyle``.
https://github.com/QuantumKitHub/TensorKitSectors.jl
"""

# Simplification: mod-2 is hardcoded. ``Z_N`` for ``N > 2`` is a real generalization
# (non-self-dual sectors, ``dual(a) = N - a``); the upgrade path is a separate
# ``ZNProvider`` with an ``n`` field, not a flag on this one.

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tenet.symmetry.base import Sector, bend_unique, permute_unique_tree

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree
    from tenet.structure import FusionBlockKey

__all__ = ["Z2", "Z2Provider", "Z2Sector"]


@dataclass(frozen=True, slots=True, order=True)
class Z2Sector(Sector):  # ty: ignore[subclass-of-dataclass-with-order]  # deliberate, see Sector
    """A Z2 charge: ``parity in {0, 1}``. 0 is even (the unit), 1 is odd.

    Parameters
    ----------
    parity : int
        ``0`` or ``1``. A ``bool`` is refused even though it is an ``int``
        subclass, so ``Z2Sector(True) != Z2Sector(1)`` can never become a live
        question.

    Raises
    ------
    TypeError
        If ``parity`` is not an ``int`` (``bool`` included).
    ValueError
        If ``parity`` is not ``0`` or ``1``.
    """

    parity: int

    def __post_init__(self) -> None:
        # bool is an int subclass; reject it so `Z2Sector(True) != Z2Sector(1)` can
        # never become a live question (SU2Sector does the same for two_j).
        if not isinstance(self.parity, int) or isinstance(self.parity, bool):
            raise TypeError(f"parity must be an int, got {self.parity!r}")
        if self.parity not in (0, 1):
            raise ValueError(f"parity must be 0 or 1, got {self.parity}")


@dataclass(frozen=True, slots=True)
class Z2Provider:
    """Z2 with *bosonic* braiding: Vect_Z2. Abelian, multiplicity-free, d = 1, R = +1.

    Use the module-level singleton [Z2][tenet.symmetry.Z2] rather than
    constructing one; ``name`` is an identity label that participates in
    equality, not a configuration knob. The capability contract of every
    method is documented once, on the protocols in ``tenet.symmetry`` —
    only behaviour that *differs* from a protocol carries a docstring here.

    Examples
    --------
    >>> from tenet.symmetry import Z2, Z2Sector
    >>> Z2.fusion(Z2Sector(1), Z2Sector(1))
    (Z2Sector(parity=0),)
    >>> int(Z2.twist(Z2Sector(1)))  # bosonic: no sign, unlike fermion parity
    1
    """

    name: str = "Z2"

    @property
    def unit(self) -> Z2Sector:
        return Z2Sector(0)

    def dual(self, a: Z2Sector) -> Z2Sector:
        """Both sectors are self-dual (``dual(q) = q``)."""
        return a

    def fusion(self, a: Z2Sector, b: Z2Sector) -> tuple[Z2Sector, ...]:
        return (Z2Sector(a.parity ^ b.parity),)

    def n_symbol(self, a: Z2Sector, b: Z2Sector, c: Z2Sector) -> int:
        return int(c.parity == a.parity ^ b.parity)

    def qdim(self, a: Z2Sector) -> float:
        return 1.0

    def irrep_dim(self, a: Z2Sector) -> int:
        return 1

    def cgc(self, a: Z2Sector, b: Z2Sector, c: Z2Sector) -> np.ndarray:
        """Shape ``(1, 1, 1, 1)``, read-only. Raises on a fusion-forbidden triple."""
        if not self.n_symbol(a, b, c):
            raise ValueError(f"Z2 fusion forbids {a} x {b} -> {c}")
        return _CGC

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """One term, coefficient 1 — the braiding is symmetric and ``F = R = +1``.

        This is the one method that separates this provider from the fermionic one
        in ``fz2.py``, which returns the same tree with a Koszul sign. See the
        module docstring: the permission to delegate is the braiding, never the
        uniqueness of the fusion.
        """
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

    def z_matrix(self, a: Z2Sector) -> np.ndarray:
        """``Z = [[1]]``, read-only: ``V_q`` is one-dimensional and the FS phase is 1."""
        return _Z

    def frobenius_schur(self, a: Z2Sector) -> float:
        """``chi_a = 1``: every Z2 irrep is one-dimensional with a real pairing."""
        return 1.0

    def twist(self, a: Z2Sector) -> float:
        """``theta_a = 1``: bosonic Abelian, trivial twist."""
        return 1.0


_CGC = np.ones((1, 1, 1, 1))
_CGC.flags.writeable = False

_Z = np.ones((1, 1))
_Z.flags.writeable = False

Z2 = Z2Provider()
"""Module-level singleton, used as ``GradedSpace(provider=Z2, ...)``."""
