"""SU(2) fusion provider: triangle-rule fusion, self-duality, Clebsch-Gordan tensors.

Sectors are labelled by the doubled spin ``two_j`` so labels stay exact integers:
``SU2Sector(0)`` is the singlet, ``SU2Sector(1)`` spin-1/2, ``SU2Sector(2)`` spin-1.

``dual(a) == a`` because SU(2) is self-dual as a *label* map. The Z-isomorphism
``V_j -> V_j^*`` carrying the Frobenius-Schur sign ``(-1)^(2j)`` is **not** the
identity, so ``leg.dual`` must never be treated as a no-op (invariant 2). It is
now available in the dense basis through :meth:`SU2Provider.z_matrix` (the
:class:`~tenet.symmetry.base.DualBasis` capability), so ``to_dense`` works on
dual SU(2) legs.

F-, R- and B-symbols and Frobenius-Schur signs are available through the
:class:`~tenet.symmetry.base.RecouplingData` capability; their gauge is pinned to
the vendored TensorKitSectors fixtures (see ``_SU2_GAUGE``). ``permute_tree`` is
built on them, so ``transpose`` is total for SU(2), and ``bend_right`` /
``bend_left`` are built on the B-symbol and the Frobenius-Schur sign, so
``repartition`` is total for SU(2) too (#38).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tenet.symmetry import _su2_coeff
from tenet.symmetry._su2_coeff import cg_tensor, triangle
from tenet.symmetry.base import (
    CapabilityError,
    FusionProvider,
    Sector,
    bend_braided,
    permute_braided_tree,
)
from tenet.symmetry.u1 import U1Provider, U1Sector

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree
    from tenet.structure import FusionBlockKey

__all__ = ["SU2", "SU2Provider", "SU2Sector"]

_SU2_GAUGE = "3j=condon-shortley;cg=condon-shortley;f=tks-su2irrep;r=tks-su2irrep;fs=tks-su2irrep"
"""Internal gauge fingerprint (racah / TensorKitSectors conventions).

Keys: ``3j``/``cg`` — the 3j and Clebsch-Gordan phase convention (Condon-Shortley,
magnetic indices descending); ``f``/``r``/``fs`` — F-symbols, R-symbols and
Frobenius-Schur signs, all matching TensorKitSectors' ``SU2Irrep``. Written into a saved
file's header by :func:`tenet.save` and compared on :func:`tenet.load`, which refuses a
file recorded under a different convention.

Cross-validated against froSTspin (2026-08-14; sympy Condon-Shortley lineage,
descending-m, dimension-labeled irreps — an implementation lineage independent
of racah/TensorKitSectors): all 215 admissible CG triples with ``dj <= 8``
agree entrywise at 1.1e-16 with no phase correction, and froSTspin's
``SU2SymmetricTensor.from_array`` (which asserts SU(2) invariance internally)
round-trips our ``to_dense`` output at exactly 0.0.
"""

# Documentation, not a cache key. Every plan cache (``permutation_plan``, ``bend_plan``,
# ``fusion_plan``, ``map_layout``, ``contraction_plan``) is a ``functools.cache`` keyed on
# ``TensorStructure``, which contains ``Leg``s -> ``GradedSpace``s -> the provider *value*.
# ``SU2Provider`` implements exactly this one hard-coded convention, so provider identity
# already pins the gauge one-for-one: appending this string to a key would be a constant on
# every key and could never change a lookup outcome.
# Simplification: the upgrade path for a provider that can carry two gauges is a ``gauge``
# field on the provider itself, which then enters every existing cache key for free with
# zero cache-code changes — and which knowingly flips
# ``tests/symmetry/test_fz2.py::test_gauge_is_a_module_string_not_a_field``. One line, when
# and only when a second gauge exists.


@dataclass(frozen=True, slots=True, order=True)
class SU2Sector(Sector):  # ty: ignore[subclass-of-dataclass-with-order]  # deliberate, see Sector
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

    def bend_right(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        """One term: ``sqrt(qdim(c)/qdim(a))·B(a,b,c)``, times ``chi`` if already dual."""
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

    def frobenius_schur(self, a: SU2Sector) -> int:
        """``chi_a = (-1)^(2j)``: ``+1`` for integer spin, ``-1`` for half-integer."""
        return _su2_coeff.frobenius_schur(a.two_j)

    def flip_phase(self, a: SU2Sector) -> int:
        """``chi_a * theta_a = chi_a``: SU(2) braiding is symmetric, so the twist is 1."""
        return self.frobenius_schur(a)

    def z_matrix(self, a: SU2Sector) -> np.ndarray:
        """``Z_a: V_a -> V_a^*``, shape ``(d_a, d_dual(a)) == (d_a, d_a)``; read-only.

        ``Z[i, d_a - 1 - i] = (-1)**i`` in the descending-m basis of :meth:`cgc`;
        see :func:`tenet.symmetry._su2_coeff.z_matrix`. Not the identity for
        ``two_j >= 1``, even though ``dual(a) == a``.
        """
        if not isinstance(a, SU2Sector):
            raise ValueError(f"{self.name}: {a!r} is not an SU(2) sector")
        return _su2_coeff.z_matrix(a.two_j)

    def branch(self, target: FusionProvider, a: SU2Sector) -> tuple[Sector, ...]:
        """SU(2) -> U(1): the magnetic quantum numbers, doubled, descending.

        :meth:`cgc`'s magnetic indices run descending, so index ``k`` of ``V_j``
        carries ``S_z = j - k`` and charge ``2 S_z = two_j - 2 k``. Identical to
        froSTspin's ``np.arange(irr - 1, -irr - 1, -2)``.

        Dual-agnostic: on a ``dual`` leg ``to_dense`` applies :meth:`z_matrix`,
        an antidiagonal ``±1`` signed permutation, so reversing the magnetic
        order and negating the weight are the same operation and cancel.
        """
        if not isinstance(target, U1Provider):
            raise CapabilityError(
                f"{self.name}: cannot branch to {getattr(target, 'name', target)!r}; "
                "the only target SU2Provider can restrict to is U1"
            )
        return tuple(U1Sector(a.two_j - 2 * k) for k in range(a.two_j + 1))


SU2 = SU2Provider()
"""Module-level singleton, used as ``GradedSpace(provider=SU2, ...)``."""
