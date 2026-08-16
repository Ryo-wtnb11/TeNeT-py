"""SU(N) coefficients from the ``racah`` crate, through the ``racah-py`` wheel.

Every array this module returns is read-only and indexed by Dynkin *tuples*, not
by :class:`~tenet.symmetry.sun.SUNSector`, so the provider stays array-free.

Importing this module — or :mod:`tenet.symmetry.sun` — without ``racah`` raises
``ImportError`` naming ``pip install 'tenet-py[sun]'``. There is no pure-Python
fallback and there will not be one: a second implementation of these coefficients
would be a second gauge.
"""

from functools import cache
from math import sqrt

import numpy as np

try:
    import racah
except ImportError as exc:  # pragma: no cover - exercised by the packaging test
    raise ImportError(
        "tenet.symmetry.sun needs the racah coefficient backend: "
        "pip install 'tenet-py[sun]'. There is no pure-Python fallback by design - "
        "a second implementation would be a second gauge (see tenet.symmetry.su2)."
    ) from exc

__all__ = [
    "GAUGE",
    "b_matrix",
    "cgc",
    "dim",
    "dual",
    "f_matrix",
    "fusion",
    "n_symbol",
    "r_matrix",
    "z_matrix",
]

Dynkin = tuple[int, ...]

GAUGE = f"sun=racah;{racah.sun_authority_fingerprint()}"
"""Gauge fingerprint of the running ``racah`` build, verbatim from the crate.

racah's contract makes any change that can alter a coefficient value a breaking
change, so this string is exactly what must trip :func:`tenet.load`'s gauge check.
"""

# No tenet-side coefficient cache: racah owns tiered caches with its own budgets, and a
# second one here would be a second eviction policy. Only ``Irrep`` construction is
# memoized, which is object churn rather than coefficients (#127 M12b).


@cache
def _irrep(a: Dynkin) -> "racah.Irrep":
    return racah.Irrep(a)


def _frozen(x: np.ndarray) -> np.ndarray:
    x.flags.writeable = False
    return x


def dim(a: Dynkin) -> int:
    """Dimension of the irrep with Dynkin label ``a``."""
    return _irrep(a).dim()


def dual(a: Dynkin) -> Dynkin:
    """Dynkin label of the conjugate irrep."""
    return tuple(_irrep(a).dual().dynkin)


def fusion(a: Dynkin, b: Dynkin) -> tuple[Dynkin, ...]:
    """Dynkin labels appearing in ``a x b``, ascending; multiplicities dropped."""
    return tuple(sorted(tuple(c.dynkin) for c, _ in racah.fusion(_irrep(a), _irrep(b))))


def n_symbol(a: Dynkin, b: Dynkin, c: Dynkin) -> int:
    """``N^c_ab``, the outer multiplicity."""
    return racah.fusion_multiplicity(_irrep(a), _irrep(b), _irrep(c))


def cgc(a: Dynkin, b: Dynkin, c: Dynkin) -> np.ndarray:
    """Clebsch-Gordan tensor, shape ``(d_a, d_b, d_c, N^c_ab)``, read-only."""
    return _frozen(racah.clebsch_gordan(_irrep(a), _irrep(b), _irrep(c)))


def f_matrix(a: Dynkin, b: Dynkin, c: Dynkin, d: Dynkin, e: Dynkin, f: Dynkin) -> np.ndarray:
    """``[F^{abc}_d]_{e,f}``, shape ``(N^e_ab, N^d_ec, N^f_bc, N^d_af)``, read-only."""
    return _frozen(racah.f_symbol(*(_irrep(x) for x in (a, b, c, d, e, f))))


def r_matrix(a: Dynkin, b: Dynkin, c: Dynkin) -> np.ndarray:
    """``R^{ab}_c``, shape ``(N^c_ab, N^c_ba)``, read-only."""
    return _frozen(racah.r_symbol(_irrep(a), _irrep(b), _irrep(c)))


def b_matrix(a: Dynkin, b: Dynkin, c: Dynkin) -> np.ndarray:
    """``B^{ab}_c``, shape ``(N^c_ab, N^a_{c,dual(b)})``.

    ``sqrt(d_a d_b / d_c) [F^{a b dual(b)}_a]_{c,1}``, the same
    ``Bsymbol_from_Fsymbol`` derivation SU(2) uses; the two one-dimensional
    vertices ``(b, dual(b), 1)`` and ``(a, 1, a)`` are dropped, which leaves
    exactly the shape above.
    """
    unit = (0,) * len(a)
    scale = sqrt(dim(a) * dim(b) / dim(c))
    return scale * f_matrix(a, b, dual(b), a, c, unit)[:, :, 0, 0]


def frobenius_schur(a: Dynkin) -> int:
    """``chi_a = sign([F^{a dual(a) a}_a]_{1,1})``, the TensorKit definition."""
    unit = (0,) * len(a)
    return 1 if f_matrix(a, dual(a), a, a, unit, unit)[0, 0, 0, 0] > 0 else -1


def z_matrix(a: Dynkin) -> np.ndarray:
    """``Z_a: V_a -> V_a^*``, shape ``(d_a, d_dual(a))``, read-only.

    Derived exactly as SU(2)'s is: the singlet in ``V_a (x) V_dual(a)`` *is* the
    duality pairing, so it cannot drift out of gauge with the CG tensors
    ``to_dense`` contracts it against.
    """
    unit = (0,) * len(a)
    return _frozen(sqrt(dim(a)) * cgc(a, dual(a), unit)[:, :, 0, 0])
