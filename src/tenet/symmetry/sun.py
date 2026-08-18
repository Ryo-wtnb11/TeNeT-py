"""SU(N) fusion provider, backed by the ``racah`` coefficient crate.

Sectors are labelled by their Dynkin label: for SU(3), ``SUNSector((0, 0))`` is
the singlet, ``(1, 0)`` the fundamental ``3``, ``(0, 1)`` its conjugate ``3bar``
and ``(1, 1)`` the adjoint ``8``. A provider carries its own ``n`` and refuses a
sector of the wrong rank, so ``SUNProvider(3)`` and ``SUNProvider(4)`` never mix
silently.

SU(3) is the first symmetry tenet ships with ``N^c_ab > 1`` (``8 x 8 -> 8`` has
``N = 2``) and the first where ``dual(a) != a`` on an irrep of dimension ``> 1``.
Both are served through the matrix-valued [FMatrixData][tenet.symmetry.FMatrixData]
/ [RMatrixData][tenet.symmetry.RMatrixData] / [BMatrixData][tenet.symmetry.BMatrixData] and
[DualBasis][tenet.symmetry.DualBasis], so ``transpose``, ``repartition`` and
``to_dense`` are total for SU(N).

This module needs the optional ``racah-py`` wheel: ``pip install 'tenet-py[sun]'``.
The refusal is categorical, not a fallback — see ``tenet.symmetry._sun_coeff``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tenet.symmetry import _sun_coeff
from tenet.symmetry.base import Sector, bend_braided, permute_braided_tree

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree
    from tenet.structure import FusionBlockKey

__all__ = ["SUNProvider", "SUNSector"]

_SUN_GAUGE = _sun_coeff.GAUGE
"""Internal gauge fingerprint, taken verbatim from ``racah.sun_authority_fingerprint()``.

Written into a saved file's header by [tenet.save][] and compared on
[tenet.load][], which refuses a file recorded under a different convention.
"""

# Unlike ``_SU2_GAUGE`` this is not a constant: it is whatever the installed racah build
# reports. That is the point — racah's contract makes any coefficient-affecting change a
# breaking change, so upgrading the wheel is exactly the event that must invalidate old
# files. It still does not belong in a cache key: ``SUNProvider`` implements one
# convention per process, so provider identity already pins it one-for-one.


@dataclass(frozen=True, slots=True, order=True)
class SUNSector(Sector):  # ty: ignore[subclass-of-dataclass-with-order]  # deliberate, see Sector
    """An SU(N) irrep labelled by its Dynkin label ``(a_1, ..., a_{N-1})``.

    Parameters
    ----------
    dynkin : tuple of int
        The Dynkin label, one non-negative entry per node of the ``A_{N-1}``
        diagram. A list is accepted and stored as a tuple, so a sector
        survives a JSON round trip through [tenet.save][] / [tenet.load][]
        unchanged.

    Raises
    ------
    TypeError
        If ``dynkin`` is not a sequence of ``int`` entries (``bool`` included).
    ValueError
        If ``dynkin`` is empty or carries a negative entry.

    Examples
    --------
    >>> from tenet.symmetry.sun import SUNSector
    >>> SUNSector((1, 0))       # the fundamental 3 of SU(3)
    SUNSector(dynkin=(1, 0))
    """

    dynkin: tuple[int, ...]

    def __post_init__(self) -> None:
        try:
            dynkin = tuple(self.dynkin)
        except TypeError:
            raise TypeError(f"dynkin must be a sequence of ints, got {self.dynkin!r}") from None
        if not dynkin:
            raise ValueError("dynkin must have at least one entry (SU(N) with N >= 2)")
        for x in dynkin:
            if not isinstance(x, int) or isinstance(x, bool):
                raise TypeError(f"dynkin entries must be ints, got {x!r}")
            if x < 0:
                raise ValueError(f"dynkin entries must be non-negative, got {x}")
        object.__setattr__(self, "dynkin", dynkin)


@dataclass(frozen=True, slots=True)
class SUNProvider:
    """SU(N) provider for ``n >= 2``. Array-free and hashable; arrays come from racah.

    ``SUNProvider(3)`` is SU(3). Every sector-taking method validates that the
    Dynkin label has ``n - 1`` entries, so feeding an SU(4) sector to an SU(3)
    provider fails at the first query rather than producing nonsense.

    The capability contract of every method is documented once, on the
    protocols in ``tenet.symmetry`` — only behaviour that *differs* from a
    protocol carries a docstring here. Every coefficient (fusion, CG, F/R/B
    matrices, FS) is delegated to the ``racah`` crate through
    ``tenet.symmetry._sun_coeff``, whose fingerprint is the gauge.

    Parameters
    ----------
    n : int
        The ``N`` of SU(N); at least 2.
    name : str, optional
        An identity label that participates in equality, not a configuration
        knob; leave it at the default.

    Raises
    ------
    TypeError
        If ``n`` is not an ``int`` (``bool`` included).
    ValueError
        If ``n < 2``.

    Examples
    --------
    >>> from tenet.symmetry.sun import SUNProvider, SUNSector
    >>> su3 = SUNProvider(3)
    >>> f, fbar = SUNSector((1, 0)), SUNSector((0, 1))
    >>> su3.fusion(f, fbar)                          # 3 x 3bar = 1 + 8
    (SUNSector(dynkin=(0, 0)), SUNSector(dynkin=(1, 1)))
    >>> adj = SUNSector((1, 1))
    >>> su3.n_symbol(adj, adj, adj)                  # 8 x 8 -> 8 has multiplicity 2
    2
    >>> su3.dual(f)                                  # the conjugate irrep
    SUNSector(dynkin=(0, 1))
    """

    n: int
    name: str = "SUN"

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool):
            raise TypeError(f"n must be an int, got {type(self.n).__name__}")
        if self.n < 2:
            raise ValueError(f"n must be at least 2, got {self.n}")

    def _d(self, a: Sector) -> tuple[int, ...]:
        if not isinstance(a, SUNSector):
            raise ValueError(f"{self.name}: {a!r} is not an SU({self.n}) sector")
        if len(a.dynkin) != self.n - 1:
            raise ValueError(
                f"{self.name}: {a!r} is an SU({len(a.dynkin) + 1}) label, "
                f"but this provider is SU({self.n})"
            )
        return a.dynkin

    @property
    def unit(self) -> SUNSector:
        return SUNSector((0,) * (self.n - 1))

    def dual(self, a: SUNSector) -> SUNSector:
        """The conjugate irrep: the Dynkin label reversed. Not the identity for ``N > 2``."""
        return SUNSector(_sun_coeff.dual(self._d(a)))

    def fusion(self, a: SUNSector, b: SUNSector) -> tuple[SUNSector, ...]:
        """The Littlewood-Richardson decomposition of ``a x b``, ascending in Dynkin order."""
        return tuple(SUNSector(c) for c in _sun_coeff.fusion(self._d(a), self._d(b)))

    def n_symbol(self, a: SUNSector, b: SUNSector, c: SUNSector) -> int:
        return _sun_coeff.n_symbol(self._d(a), self._d(b), self._d(c))

    def qdim(self, a: SUNSector) -> float:
        return float(_sun_coeff.dim(self._d(a)))

    def irrep_dim(self, a: SUNSector) -> int:
        return _sun_coeff.dim(self._d(a))

    def cgc(self, a: SUNSector, b: SUNSector, c: SUNSector) -> np.ndarray:
        """Shape ``(d_a, d_b, d_c, N^c_ab)``, read-only; last axis is the multiplicity label."""
        da, db, dc = self._d(a), self._d(b), self._d(c)
        if not _sun_coeff.n_symbol(da, db, dc):
            raise ValueError(f"{c} does not appear in the fusion of {a} and {b}")
        return _sun_coeff.cgc(da, db, dc)

    # --- FMatrixData / RMatrixData / BMatrixData --------------------------------

    def f_matrix(
        self,
        a: SUNSector,
        b: SUNSector,
        c: SUNSector,
        d: SUNSector,
        e: SUNSector,
        f: SUNSector,
    ) -> np.ndarray:
        """``[F^{abc}_d]_{e,f}``, shape ``(N^e_ab, N^d_ec, N^f_bc, N^d_af)``, read-only."""
        return _sun_coeff.f_matrix(*(self._d(x) for x in (a, b, c, d, e, f)))

    def r_matrix(self, a: SUNSector, b: SUNSector, c: SUNSector) -> np.ndarray:
        """``R^{ab}_c``, shape ``(N^c_ab, N^c_ba)``, read-only."""
        return _sun_coeff.r_matrix(self._d(a), self._d(b), self._d(c))

    def b_matrix(self, a: SUNSector, b: SUNSector, c: SUNSector) -> np.ndarray:
        """``B^{ab}_c``, shape ``(N^c_ab, N^a_{c,dual(b)})``."""
        return _sun_coeff.b_matrix(self._d(a), self._d(b), self._d(c))

    # --- scalar F/R/B and the FS indicator --------------------------------------

    def f_symbol(
        self,
        a: SUNSector,
        b: SUNSector,
        c: SUNSector,
        d: SUNSector,
        e: SUNSector,
        f: SUNSector,
    ) -> float:
        """``[F^{abc}_d]_{e,f}``. Raises if any vertex has ``n_symbol > 1``."""
        return _scalar(self.f_matrix(a, b, c, d, e, f), "F")

    def r_symbol(self, a: SUNSector, b: SUNSector, c: SUNSector) -> float:
        """``R^{ab}_c``. Raises if ``N^c_ab > 1``."""
        return _scalar(self.r_matrix(a, b, c), "R")

    def b_symbol(self, a: SUNSector, b: SUNSector, c: SUNSector) -> float:
        """``B^{ab}_c``. Raises if any vertex has ``n_symbol > 1``."""
        return _scalar(self.b_matrix(a, b, c), "B")

    def frobenius_schur(self, a: SUNSector) -> int:
        """``chi_a = sign([F^{a dual(a) a}_a]_{1,1})``, the TensorKit definition."""
        return _sun_coeff.frobenius_schur(self._d(a))

    def twist(self, a: SUNSector) -> int:
        """``theta_a = 1``: SU(N) braiding is symmetric, so the twist is trivial."""
        self._d(a)
        return 1

    # --- DualBasis --------------------------------------------------------------

    def z_matrix(self, a: SUNSector) -> np.ndarray:
        """``Z_a: V_a -> V_a^*``, shape ``(d_a, d_dual(a))``; read-only.

        Derived from the CG singlet of ``V_a (x) V_dual(a)``, so it is in the same
        gauge as ``cgc`` by construction. Never the identity for ``d_a > 1``,
        and its two axes index *different* sectors whenever ``dual(a) != a``.
        """
        return _sun_coeff.z_matrix(self._d(a))

    # --- Permutation / Bending --------------------------------------------------

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """Artin-braid expansion; SU(N) is symmetric, so ``perm`` fixes the braid."""
        return permute_braided_tree(self, tree, perm)

    def bend_right(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        """``sqrt(qdim(c)/qdim(a))·B(a,b,c)``, expanded over the new vertex's multiplicity."""
        # ``self``'s Sector parameters are narrowed to this symmetry's own sector
        # type; deliberate per-symmetry specialization the unparameterized
        # protocol cannot express, so the checker misreads the conformance.
        return bend_braided(self, key, right=True, dual=dual)  # ty: ignore[invalid-argument-type]

    def bend_left(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        # ``self``'s Sector parameters are narrowed to this symmetry's own sector
        # type; deliberate per-symmetry specialization the unparameterized
        # protocol cannot express, so the checker misreads the conformance.
        return bend_braided(self, key, right=False, dual=dual)  # ty: ignore[invalid-argument-type]


def _scalar(block: np.ndarray, which: str) -> float:
    if block.size != 1:
        raise ValueError(
            f"SUN: {which}-symbol is scalar-valued but this vertex has multiplicity "
            f"{block.shape}; use the matrix-valued f_matrix/r_matrix/b_matrix method instead"
        )
    return float(block.reshape(()))
