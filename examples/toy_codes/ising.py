"""The classical 2D Ising model: its partition-function bulk tensor and Onsager's free energy.

The physics half of ``examples/toy_codes/ctmrg.py`` (#268 split it out): what the CTMRG in
``ctmrg.py`` contracts, and the closed form it is judged against. ``ctmrg.py`` imports
both names; nothing here imports ``ctmrg.py``.

**The Ising half is Z2-graded**, for the reason YASTN's CTMRG Ising example passes
``sym='Z2'``: it stops a finite-chi environment from breaking the symmetry spuriously in
the ordered phase, which is what lets this file run at ``beta > beta_c`` against Onsager
at all. Two further things the grading buys: zero magnetization is *structural* -- a spin
insertion is a Z2-odd tensor, which no invariant ``SymmetricTensor`` can hold -- and the
ordered-phase corner spectrum acquires **exact** two-fold degeneracy across the parity
sectors. Because that doubling is *cross*-sector and ``tenet.ad`` broadens *per coupled
sector*, the graded run never hands one SVD a degenerate pair.
"""

import jax.numpy as jnp
import numpy as np

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.symmetry import Z2, Z2Sector

BETA_C = 0.4406867935097714  # ln(1 + sqrt(2)) / 2


def ising_bulk(beta):
    """Classical 2D Ising partition-function tensor, legs ``(l OUT, u OUT, r IN, d IN)``.

    ``a[l,u,r,d] = sum_s W[s,l] W[s,u] W[s,r] W[s,d]`` with ``W W^T`` the bond Boltzmann
    matrix ``[[e^b, e^-b], [e^-b, e^b]]``, i.e. the symmetric splitting
    ``W = [[sqrt cosh b, sqrt sinh b], [sqrt cosh b, -sqrt sinh b]]``. That ``W`` *is
    already the parity basis*: ``W[s, 0]`` does not depend on ``s`` and ``W[s, 1]`` is odd
    under ``s -> -s``, so summing over ``s`` doubles every term with an even number of odd
    legs and annihilates the rest.

    The ``Z2`` legs are therefore not a claim checked afterwards, they are the statement:
    the blocks the grading allows are exactly the surviving entries, and each one is
    ``2 W[0,l] W[0,u] W[0,r] W[0,d]``. The eight structurally zero entries have no block to
    live in and are never built.

    ``beta`` may be a *traced scalar*, so the block values are built with ``jax.numpy``.
    """
    c, s = jnp.sqrt(jnp.cosh(beta)), jnp.sqrt(jnp.sinh(beta))
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    legs = (Leg(space, OUT), Leg(space, OUT), Leg(space, IN), Leg(space, IN))
    structure = TensorStructure(legs)
    blocks = {}
    for key in structure.block_order:  # the key names (l, u) and (r, d)
        w = [c if sector.parity == 0 else s for sector in key.output_tree.uncoupled]
        w += [c if sector.parity == 0 else s for sector in key.input_tree.uncoupled]
        blocks[key] = jnp.full((1, 1, 1, 1), 2.0 * (w[0] * w[1] * w[2] * w[3]))
    return SymmetricTensor.from_blocks(legs, blocks)


def onsager(beta: float, points: int = 200_001) -> float:
    """``beta f`` from Onsager's closed form, by direct quadrature. NumPy, no ``scipy``.

    ``-beta f = ln 2 + (1/2pi) int_0^pi dtheta ln[cosh^2(2b) + (1/k) sqrt(1 + k^2 - 2k cos
    2theta)]``, ``k = 1/sinh^2(2b)``. The equivalent elliptic form is cross-checked in
    ``tests/integration/test_ctmrg.py`` before this is used to judge anything.
    """
    kk = 1.0 / np.sinh(2.0 * beta) ** 2
    theta = np.linspace(0.0, np.pi, points)
    integrand = np.log(
        np.cosh(2.0 * beta) ** 2 + np.sqrt(1.0 + kk**2 - 2.0 * kk * np.cos(2.0 * theta)) / kk
    )
    return -(np.log(2.0) / 2.0 + np.trapezoid(integrand, theta) / (2.0 * np.pi))
