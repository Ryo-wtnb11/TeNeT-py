"""A test-only Fibonacci provider — the acceptance gate of the M24a decomposition.

Fibonacci is the smallest theory where the assumptions the seven shipped
providers share all fail at once: the braiding is chiral (``R != R**-1``), the
twist is a genuine phase (``theta_tau = e^{4 pi i / 5}``), the quantum
dimension is irrational (``phi``), and there is **no dense expansion at all** —
``tau`` is not a representation of anything, so this provider deliberately has
no ``cgc``, no ``irrep_dim`` and no ``z_matrix``. A decomposition validated
only against providers that are all symmetric, spherical, dense-expandable and
``theta = 1`` has validated nothing; this fixture is the negative half of the
test.

Data is the standard Fibonacci UMTC in TensorKitSectors' conventions
(``FibonacciAnyon``): ``tau x tau = 1 + tau``, multiplicity-free;
``F^{tau tau tau}_tau = [[phi^-1, phi^-1/2], [phi^-1/2, -phi^-1]]``;
``R^{tau tau}_1 = e^{-4 pi i / 5}``, ``R^{tau tau}_tau = e^{3 pi i / 5}``;
``chi_tau = +1``; ``qdim(tau) = phi``. ``B`` is derived from ``F`` exactly as
SU(2)'s and the SU(3) fixture's are (``Bsymbol_from_Fsymbol``).

Follows ``_su3_fixture.py``'s precedent: a test-only provider, no ``src/``
footprint, no registry name.
"""

from cmath import exp
from dataclasses import dataclass
from math import pi, sqrt

from tenet.structure import FusionBlockKey
from tenet.symmetry.base import Sector, bend_braided

PHI = (1 + sqrt(5)) / 2


@dataclass(frozen=True, slots=True, order=True)
class FibSector(Sector):
    """A Fibonacci sector: ``tau in {0, 1}``. 0 is the unit ``1``, 1 is ``tau``."""

    tau: int

    def __post_init__(self) -> None:
        if self.tau not in (0, 1):
            raise ValueError(f"tau must be 0 or 1, got {self.tau}")


ONE = FibSector(0)
TAU = FibSector(1)

# F^{tau tau tau}_tau over inner lines (1, tau) x (1, tau)
_F_TAU = {
    (0, 0): 1 / PHI,
    (0, 1): 1 / sqrt(PHI),
    (1, 0): 1 / sqrt(PHI),
    (1, 1): -1 / PHI,
}
_R = {ONE: exp(-4j * pi / 5), TAU: exp(3j * pi / 5)}


@dataclass(frozen=True, slots=True)
class FibonacciProvider:
    """Fibonacci anyons. Frozen, hashable, array-free; chiral and non-integral."""

    name: str = "Fibonacci"

    @property
    def unit(self) -> FibSector:
        return ONE

    def dual(self, a: FibSector) -> FibSector:
        """Both sectors are self-dual."""
        return a

    def fusion(self, a: FibSector, b: FibSector) -> tuple[FibSector, ...]:
        if a.tau and b.tau:
            return (ONE, TAU)
        return (TAU,) if a.tau or b.tau else (ONE,)

    def n_symbol(self, a: FibSector, b: FibSector, c: FibSector) -> int:
        return int(c in self.fusion(a, b))

    def qdim(self, a: FibSector) -> float:
        return PHI if a.tau else 1.0

    def all_sectors(self) -> tuple[FibSector, ...]:
        """The complete (finite) sector set — Fibonacci has exactly two."""
        return (ONE, TAU)

    def f_symbol(
        self, a: FibSector, b: FibSector, c: FibSector, d: FibSector, e: FibSector, f: FibSector
    ) -> float:
        """``[F^{abc}_d]_{e,f}``; the only non-trivial block is ``F^{tau tau tau}_tau``."""
        admissible = (
            self.n_symbol(a, b, e)
            and self.n_symbol(e, c, d)
            and self.n_symbol(b, c, f)
            and self.n_symbol(a, f, d)
        )
        if not admissible:
            return 0.0
        if a.tau and b.tau and c.tau and d.tau:
            return _F_TAU[(e.tau, f.tau)]
        return 1.0

    def r_symbol(self, a: FibSector, b: FibSector, c: FibSector) -> complex:
        """``R^{ab}_c``: chiral — ``R**2 != 1`` on the ``tau x tau`` channels."""
        if not self.n_symbol(a, b, c):
            return 0.0
        if a.tau and b.tau:
            return _R[c]
        return 1.0

    def b_symbol(self, a: FibSector, b: FibSector, c: FibSector) -> float:
        """``B^{ab}_c = sqrt(d_a d_b / d_c) [F^{a b dual(b)}_a]_{c, 1}``."""
        return sqrt(self.qdim(a) * self.qdim(b) / self.qdim(c)) * self.f_symbol(
            a, b, self.dual(b), a, c, self.unit
        )

    def frobenius_schur(self, a: FibSector) -> float:
        """``chi = +1`` for both sectors."""
        return 1.0

    def twist(self, a: FibSector) -> complex:
        """``theta_tau = e^{4 pi i / 5}`` — a genuine ribbon phase."""
        return exp(4j * pi / 5) if a.tau else 1.0

    def bend_right(
        self, key: FusionBlockKey, *, dual: bool
    ) -> tuple[tuple[FusionBlockKey, complex], ...]:
        return bend_braided(self, key, right=True, dual=dual)

    def bend_left(
        self, key: FusionBlockKey, *, dual: bool
    ) -> tuple[tuple[FusionBlockKey, complex], ...]:
        return bend_braided(self, key, right=False, dual=dual)

    # deliberately NO cgc, NO irrep_dim, NO z_matrix (no dense expansion exists)
    # and NO permute_tree (the braiding is chiral, so ``axes`` underdetermine a
    # braid and transpose must refuse rather than route through the symmetric
    # helper).


Fib = FibonacciProvider()
