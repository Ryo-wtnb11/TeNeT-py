"""Exact SU(2) recoupling coefficients, computed by ``racah``'s closed-form engine.

The twin of [tenet.symmetry._sun_coeff][] for the one symmetry that has a closed form.
Both are ``racah``; they are different *tiers* of it, and the split is deliberate:

* **F, R, B and the Frobenius-Schur indicator come from here.** They are gauge-invariant
  combinations, so the exact tier and the generated SU(N) tier return the same numbers --
  measured over 55 R triangles and 76 F classes at worst 1.3e-15 apart, pinned in
  ``racah``'s own ``racah-py/tests/test_su2_exact.py``. Taking them from the closed form
  is therefore a cost change and not a value change: ``racah.su2_r_symbol`` is a sign,
  where the generated path builds a Clebsch-Gordan tensor by SVD nullspace, least-squares
  descent and QR gauge fixing to reach the same number (#307: 3.8 ms per cold call).

* **Clebsch-Gordan tensors and the duality map do not come from here.** They are gauge
  *data*, not gauge-invariant, and the two tiers fix that gauge differently: the
  generated tier's Gelfand-Tsetlin basis against this one's Condon-Shortley convention.
  Their coefficients differ by exactly one sign per fusion channel, uniform in the
  magnetic indices and equal to ``R^{ab}_c`` itself. So ``cgc`` and ``z_matrix`` stay on
  ``_sun_coeff``, where every dense expansion this library has ever written was produced,
  and no stored tensor's ``to_dense`` moves.

Mixing the two is sound for exactly the reason the split is drawn there: an F-symbol is
independent of which CGC gauge computed it, so an exact F above a generated CGC describes
the same category. That is what the agreement measurement says, and it is what makes this
module a performance change with no gauge consequence.
"""

import racah

__all__ = ["GAUGE", "b_symbol", "f_symbol", "frobenius_schur", "r_symbol"]

GAUGE = f"su2=racah;{racah.su2_authority_fingerprint()}"
"""Gauge fingerprint of the exact tier, verbatim from the crate.

Distinct from [tenet.symmetry._sun_coeff.GAUGE][] on purpose: numerical agreement on F
and R is not authority identity, and a file that recorded one string while its values
came from the other tier would validate against the wrong contract.
"""


def f_symbol(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
    """``[F^{abc}_d]_{e,f}`` for doubled spins. Exactly ``0.0`` on an empty vertex."""
    return racah.su2_f_symbol(a, b, c, d, e, f)


def r_symbol(a: int, b: int, c: int) -> float:
    """``R^{ab}_c = (-1)^{(a + b - c)/2}``, exactly ``+-1.0``; ``0.0`` off the triangle."""
    return racah.su2_r_symbol(a, b, c)


def b_symbol(a: int, b: int, c: int) -> float:
    """``B^{ab}_c = sqrt(d_a d_b / d_c) [F^{a b b}_a]_{c,0}``.

    The same ``Bsymbol_from_Fsymbol`` derivation ``_sun_coeff.b_matrix`` performs, with
    the two one-dimensional vertices dropped. SU(2) is self-dual, so ``dual(b) == b`` and
    the unit sector is ``0``.
    """
    return ((a + 1) * (b + 1) / (c + 1)) ** 0.5 * f_symbol(a, b, b, a, c, 0)


def frobenius_schur(a: int) -> int:
    """``chi_a = (-1)^{2j}``, from the crate's own indicator."""
    return int(racah.su2_frobenius_schur(a))
