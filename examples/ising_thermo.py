"""Thermodynamics of the 2D classical Ising model by differentiating a CTMRG contraction.

Run it standalone::

    uv run --extra jax python examples/ising_thermo.py

``examples/ising2d.py`` contracts the Boltzmann network and reads off ``beta f``. This
file differentiates that same contraction with respect to ``beta``, which turns one
number into three::

    beta f = -(1/N) ln Z(beta)          the free energy       (ising2d.py already)
    u      =  d(beta f)/d beta          the internal energy per site
    c_V    = -beta^2 d^2(beta f)/d beta^2   the specific heat per site

All three have closed forms from Onsager, so every derivative here is judged against an
oracle rather than against itself. The chain the file demonstrates is

    beta -> bulk tensor a(beta) -> CTM environment -> ln kappa -> beta f -> d/d beta

with ``jax.grad`` applied to the whole of it, the bulk tensor included: ``beta`` enters
only through the block *values* of ``a(beta)``, so its grading, its block shapes and the
environment bond are all structure that ``jax`` never sees.

**What is differentiated, exactly.** ``EnvCTMc4v.iterate_`` re-decides the environment
bond every sweep from measured singular values, so it cannot run under a trace and does
not: it is called **once, outside**, and hands over two things — a converged corner/edge
pair, which enters the traced region as a *constant initial condition*, and a
``GradedSpace`` bond, which enters as a static cache key. Inside, exactly ``K`` calls to
``EnvCTMc4v.update_(bond=...)`` run at that frozen bond. So this is **truncated backprop
through K unrolled CTMRG moves, not an implicit fixed-point derivative** (PRX 9, 031041
Sec. III C). The difference is measurable and is measured: :func:`k_scan` prints ``c_V``
against ``K``, and the first derivative is converged at ``K = 2`` while the second still
moves until ``K = 8``.

**Why the second derivative works at all.** ``tenet.ad``'s broadened SVD VJP is a
``jax.custom_vjp`` over ordinary ``jnp`` operations, so JAX differentiates the backward
pass again for free. No second rule is registered, and none is needed.

The Z2 grading is what puts an oracle on both sides of ``beta_c``: an ungraded
finite-``chi`` environment may break the symmetry spuriously in the ordered phase.
``examples/ising2d.py``'s docstring has that argument; this file inherits the bulk tensor
from it and only re-states it in a form ``beta`` can be traced through.
"""

import jax
import jax.numpy as jnp
from ising2d import BETA_C, log_kappa, onsager  # noqa: F401  (BETA_C is a re-export)

import tenet
from tenet import OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.network import EnvCTMc4v, Peps, SquareLattice
from tenet.symmetry import Z2, Z2Sector

#: Environment bond. The free energy is already at float64 noise here off criticality,
#: and the derivatives are limited by :data:`K`, not by this -- measured in
#: ``tests/test_examples.py``.
CHI = 16

#: Unrolled moves inside the traced region. 8 puts ``c_V`` within 1e-4 of Onsager;
#: :func:`k_scan` is the evidence rather than the assertion.
K = 8


def traced_bulk(beta):
    """``ising2d.ising_bulk`` with a *traced* ``beta``: blocks named, not projected.

    Same tensor, same four identical ``Z2`` legs, same numbers -- asserted against
    ``ising_bulk`` in ``tests/test_examples.py``. It is spelled differently because
    ``from_dense`` asks a concrete-value question ("is this array symmetric to within
    ``atol``?") that a tracer cannot answer, while
    [from_blocks][tenet.SymmetricTensor.from_blocks] asks none: it is handed the value of
    each allowed block.

    The allowed blocks *are* the statement of the model. ``W = [[sqrt cosh b, sqrt sinh
    b], [sqrt cosh b, -sqrt sinh b]]`` splits the bond weight ``W W^T = [[e^b, e^-b],
    [e^-b, e^b]]`` symmetrically across the two sites it joins, and that ``W`` is already
    the parity basis: column 0 does not depend on the spin ``s``, column 1 is odd under
    ``s -> -s``. So ``a[t,l,b,r] = sum_s W[s,t] W[s,l] W[s,b] W[s,r]`` doubles every entry
    whose four parities multiply to even and annihilates the other eight -- and those
    eight are exactly the keys ``Z2`` refuses to enumerate.
    """
    c, s = jnp.sqrt(jnp.cosh(beta)), jnp.sqrt(jnp.sinh(beta))
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    legs = (Leg(space, OUT),) * 4
    blocks = {}
    for key in TensorStructure(legs).block_order:
        # The key names one allowed parity assignment to the four legs; reading the
        # parities off its two trees picks the four columns of W. Every sector has
        # degeneracy 1, so a block is a single number.
        w = [c if sector.parity == 0 else s for sector in key.output_tree.uncoupled]
        w += [c if sector.parity == 0 else s for sector in key.input_tree.uncoupled]
        blocks[key] = jnp.full((1, 1, 1, 1), 2.0 * (w[0] * w[1] * w[2] * w[3]))
    return SymmetricTensor.from_blocks(legs, blocks)


def warm(beta: float, chi: int = CHI):
    """**Outside** the trace. ``(seed, bond)`` from a converged C4v environment.

    ``seed`` is the converged ``(corner, edge)`` pair and ``bond`` the ``GradedSpace``
    the sweep settled on. Building it on :func:`traced_bulk` rather than on
    ``ising_bulk`` is not about tracing -- ``beta`` is a plain float here -- but so that
    the blocks are already JAX arrays when the traced region receives them.
    """
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_bulk(beta)))
    out = env.iterate_(max_bond=chi, max_sweeps=200, corner_tol=1e-10)
    local = env[0, 0]
    return (local.tl, local.t), local.tl.legs[0].space, out


def beta_free_energy(beta, seed, bond, k: int = K):
    """**Inside** ``jax.grad``. ``beta f = -ln kappa`` after exactly ``k`` frozen moves.

    ``kappa = Z(L+1,L+1) Z(L,L) / Z(L+1,L) Z(L,L+1)`` is Baxter's corner-transfer
    telescoping: the three patches differ by exactly one site's worth of partition
    function, so every environment tensor and every gauge factor cancels and
    ``ln kappa = (1/N) ln Z`` survives. ``ising2d.log_kappa`` is that contraction; the
    only thing this function adds is that the bulk tensor is rebuilt from ``beta``
    *inside* the region, which is where the derivative enters.

    ``seed`` carries no gradient -- it is a constant of the trace -- so the derivative is
    carried entirely by the ``k`` moves. That is the truncated-backprop statement, and
    :func:`k_scan` measures what it costs.
    """
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_bulk(beta)), init=None)
    env.env[0, 0].tl, env.env[0, 0].t = seed
    for _ in range(k):
        # update_(bond=...) is the traceable move: shape-static, one trace, no singular
        # value ever compared against a threshold. iterate_ would raise here.
        env.update_(bond=bond)
    return -log_kappa(env)


def thermodynamics(beta: float, chi: int = CHI, k: int = K):
    """``(beta f, u, c_V)`` at one ``beta``, the last two by automatic differentiation."""
    seed, bond, _ = warm(beta, chi)
    bf = float(beta_free_energy(beta, seed, bond, k))
    # u = d(beta f)/d beta: one reverse-mode pass through the k unrolled moves.
    u = float(jax.grad(beta_free_energy)(beta, seed, bond, k))
    # c_V = -beta^2 d^2(beta f)/d beta^2: grad of grad, no second rule installed.
    d2 = float(jax.grad(jax.grad(beta_free_energy))(beta, seed, bond, k))
    return bf, u, -(beta**2) * d2


def onsager_derivatives(beta: float):
    """``(u, c_V)`` from central differences of :func:`onsager`, the reference values.

    Onsager's ``beta f`` is a smooth quadrature accurate to ~1e-12, which fixes the two
    step sizes: a central *first* difference has truncation ``O(h^2)`` and roundoff
    ``O(1e-12/h)``, balanced near ``h = 1e-4``; a central *second* difference has
    roundoff ``O(1e-12/h^2)`` and needs the larger ``h = 1e-3``. Both leave the oracle an
    order of magnitude sharper than the CTM error this file measures, which is what makes
    it an oracle rather than a second estimate.
    """
    h1 = 1e-4
    u = (onsager(beta + h1) - onsager(beta - h1)) / (2 * h1)
    h2 = 1e-3
    d2 = (onsager(beta + h2) - 2 * onsager(beta) + onsager(beta - h2)) / h2**2
    return u, -(beta**2) * d2


def k_scan(beta: float, ks=(2, 4, 8), chi: int = CHI):
    """``c_V`` against the number of unrolled moves: the truncated backprop, measured.

    The environment is at its fixed point when the traced region starts, so ``beta f``
    itself does not depend on ``k`` at all and the first derivative barely does. The
    *second* derivative is where a finite unrolling shows: the ``k`` moves have to carry
    the second-order response of the environment to ``beta``, and two of them do not.
    """
    seed, bond, _ = warm(beta, chi)
    return [
        (k, -(beta**2) * float(jax.grad(jax.grad(beta_free_energy))(beta, seed, bond, k)))
        for k in ks
    ]


def main(chi: int = CHI, k: int = K):
    """Free energy, internal energy and specific heat at two betas, against Onsager."""
    tenet.enable_jax(ad=True)  # pytrees + the broadened SVD VJP the CTM spectra need
    results = {}
    # One below beta_c and one above it. Criticality is left out on purpose: there the
    # correlation length outruns any finite chi, so the CTM error would swamp the
    # derivative error this file is about.
    for beta in (0.3, 0.5):
        bf, u, cv = thermodynamics(beta, chi, k)
        u_ref, cv_ref = onsager_derivatives(beta)
        print(
            f"beta={beta:.2f}  beta*f={bf:+.10f} ({onsager(beta):+.10f})  "
            f"u={u:+.8f} ({u_ref:+.8f})  c_V={cv:+.6f} ({cv_ref:+.6f})"
        )
        results[beta] = (bf, u, cv, u_ref, cv_ref)
    scan = k_scan(0.5, chi=chi)
    print("c_V at beta=0.5 vs unrolled moves: " + "  ".join(f"K={k}:{v:+.6f}" for k, v in scan))
    return results, scan


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)  # tests/conftest.py does this for the suite
    main()
