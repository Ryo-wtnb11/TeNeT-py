"""Differentiable CTMRG: classical Ising against Onsager, then a U(1)/SU(2) iPEPS gradient.

Run it standalone::

    uv run --extra jax python examples/ctmrg.py

Two physical problems, **one** CTMRG core -- and since #114 that core is the library's.
:mod:`tenet.network.ctmrg` owns ``Absorb``, ``single_layer``/``double_layer``, ``move``,
``ctmrg`` and ``ctmrg_unrolled``, with the ``svd_truncated``-outside / ``svd(bond=)``-inside
pairing (#77), the leg conventions and the four environment ceilings (truncated backprop,
no checkpointing, no pre-QR, ``svd`` rather than ``eigh``) now in its docstrings. What
stayed here is what the library must **not** decide: which bulk tensor
(:func:`ising_bulk`), which ansatz (:func:`c4v`, an ansatz constraint -- a library that
symmetrized its input would be silently editing the user's state), and what to measure
(:func:`_halves`/:func:`energy` are a C4v-and-1x1-and-2x1 reduced density matrix with one
geometry and one caller, i.e. a measurement API; :func:`log_kappa`'s Baxter telescoping is
classical-partition-function physics with no meaning for an iPEPS).

* the classical 2D Ising partition function, whose free energy per site has a closed form
  (Onsager) and whose internal energy ``d(beta f)/d beta`` is therefore an oracle for
  ``jax.grad`` through the unrolled sweeps;
* a single-site U(1) (or SU(2)) iPEPS with a random symmetric two-site ``h``, which
  exercises graded truncation, ``svd(bond=)`` across sectors and multiplet degeneracies.

**The Ising half is Z2-graded** (#104), for the reason YASTN's CTMRG Ising example passes
``sym='Z2'``: it stops a finite-chi environment from breaking the symmetry spuriously in
the ordered phase, which is what lets this file run at ``beta > beta_c`` against Onsager at
all. Two further things the grading buys, both asserted in
``tests/integration/test_ctmrg.py``: zero magnetization becomes *structural* -- a spin
insertion is a Z2-odd tensor, which no invariant ``SymmetricTensor`` can hold, so
``from_dense`` refuses it and the refusal is the statement -- and the ordered-phase corner
spectrum acquires **exact** two-fold degeneracy across the parity sectors. Because that
doubling is *cross*-sector and ``tenet.ad`` broadens *per coupled sector*, the graded run
never hands one SVD a degenerate pair: grading removes the ``NaN``, it does not create it.
It changed no arithmetic either -- see :func:`ising_bulk`.

**The iPEPS half is a plumbing result, not a physics result, and cannot be otherwise with a
one-site unit cell**, so it makes **no benchmark-energy claim**. Liao et al. get a
single-site AFM Heisenberg cell by rotating one sublattice by pi about y, which turns
``S^x S^x - S^y S^y`` into ``(S^+S^+ + S^-S^-)/2`` -- an operator that changes ``S^z_tot``
by +-2 and so *destroys the U(1) the ansatz is graded by*. The alternatives are a two-site
unit cell (out of scope) or dropping the symmetry (which deletes the reason this half
exists). So it follows ``examples/vmc_mps.py``: random symmetric ``h``, no comparison
against ``-0.669437(5)``, said out loud right here.

ponytail: **``tenet.cast`` (#92) is mentioned and not used.** Building an SU(2) ansatz and
casting it to U(1) is a third concept in a file that already has two models; the SU(2)
provider is instead run through the *same* iPEPS path via a ``provider`` parameter.
"""

import math

import autoray as ar

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import (
    ctmrg,
    ctmrg_unrolled,
    double_layer,
    double_layer_ctm,
    layers,
    ring,
    scalar,
    single_layer,
    single_layer_ctm,
)
from tenet.symmetry import SU2, U1, Z2, SU2Sector, U1Sector, Z2Sector

BETA_C = 0.4406867935097714  # ln(1 + sqrt(2)) / 2

# ``opt_einsum``'s greedy path, for the ring contractions only. ponytail: greedy, not
# "auto" -- at ten-plus operands "auto"'s dynamic-programming *search* costs an order of
# magnitude more than the contraction it plans (4.5 s against 0.4 s for the two-site
# energy). Upgrade path: an explicit path, or cotengra.
PATH = "greedy"

# Environment dimension for the iPEPS half, per provider. ponytail: not one number.
# ``max_bond`` bounds the *dense* bond, which for SU(2) is ``sum_c (2j+1) m_c``: a budget
# of 4 stops in the middle of the second multiplet -- the split Francuz-Schuch-Vanhecke's
# Appendix C warns about, and slower to converge and to differentiate than the 6 that
# closes it. U(1) has no multiplets and 4 is plenty.
CHI_IPEPS = {"u1": 4, "su2": 6}

# Physical and virtual spaces, per provider, as ``vmc_mps.SPACES`` does. The virtual space
# must contain the unit sector or a spin-1/2 site tensor has no allowed block at all.
SPACES = {
    "u1": (
        GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1}),
        GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1}),
    ),
    "su2": (
        GradedSpace.new(SU2, {SU2Sector(1): 1}),
        GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 1}),
    ),
}


# --- the two bulk tensors ----------------------------------------------------------


def ising_bulk(beta):
    """Classical 2D Ising partition-function tensor, legs ``(l OUT, u OUT, r IN, d IN)``.

    ``a[l,u,r,d] = sum_s W[s,l] W[s,u] W[s,r] W[s,d]`` with ``W W^T`` the bond Boltzmann
    matrix ``[[e^b, e^-b], [e^-b, e^b]]``, i.e. the symmetric splitting
    ``W = [[sqrt cosh b, sqrt sinh b], [sqrt cosh b, -sqrt sinh b]]``. That ``W`` is
    *already the parity basis* -- ``W[s, 0]`` does not depend on ``s`` and ``W[s, 1]`` is
    odd under ``s -> -s`` -- so eight of the sixteen entries are *structurally* zero, the
    ``Z2`` legs are what stops us storing them, and the grading declared a symmetry the
    example always had rather than changing any arithmetic.

    ``beta`` may be a *traced scalar*: the block is built through ``autoray``, so
    ``jax.grad`` has something to differentiate. Hence ``atol=math.inf`` -- #82's "project,
    don't check" spelling, because a symmetry check is a concrete-value question and would
    raise under a trace. The check is *moved*, not lost: an untraced test runs the same
    array through ``from_dense`` at the **default** relative ``atol``.

    ponytail: dense-then-gather at setup on a 16-element array, ceiling ``prod dim_i``.
    """
    c, s = ar.do("sqrt", ar.do("cosh", beta)), ar.do("sqrt", ar.do("sinh", beta))
    w = ar.do("stack", (ar.do("stack", (c, s)), ar.do("stack", (c, -s))))
    block = ar.do("einsum", "sl,su,sr,sd->lurd", w, w, w, w)
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    legs = (Leg(space, OUT), Leg(space, OUT), Leg(space, IN), Leg(space, IN))
    return SymmetricTensor.from_dense(block, legs, atol=math.inf)


def c4v(a: SymmetricTensor) -> SymmetricTensor:
    """Symmetrize an iPEPS tensor under the C4v diagonal mirror ``l <-> u``, ``r <-> d``.

    An **ansatz constraint**, which is why it is here rather than in the library: one
    corner and one edge describe the environment only if the bulk is mirror-symmetric, a
    random ansatz is not, and symmetrizing the caller's state is not the environment's
    business -- ``double_layer_ctm`` documents it as a precondition and never enforces it.
    Because ``l``/``u`` share a side and ``r``/``d`` share one, the mirror is the plain
    transpose ``(0, 2, 1, 4, 3)``: no bend, no bending coefficient, and linear, so it
    differentiates for free.

    ponytail: **one C4v move, not four directional ones** -- symmetrizing the *ansatz*
    instead. A general single-site iPEPS needs the four-move environment, which is the same
    upgrade the multi-site unit cell needs, and what buys the rotated Heisenberg energy.
    """
    return (a + tenet.transpose(a, (0, 2, 1, 4, 3))) / 2


def build_ipeps(provider: str = "u1", seed: int = 1) -> SymmetricTensor:
    """A random single-site iPEPS, legs ``(P OUT, l OUT, u OUT, r IN, d IN)``."""
    phys, virt = SPACES[provider]
    legs = (Leg(phys, OUT), Leg(virt, OUT), Leg(virt, OUT), Leg(virt, IN), Leg(virt, IN))
    return SymmetricTensor.random(legs, seed=seed).to_backend("jax")


def build_h(provider: str = "u1", seed: int = 100) -> SymmetricTensor:
    """A random two-site operator on ``(P OUT, P OUT, P IN, P IN)``: symmetric by
    construction, hence ``Sz``-conserving for U(1) and a scalar under SU(2). A plumbing
    operator, exactly as ``vmc_mps.build_h`` is."""
    phys = SPACES[provider][0]
    legs = (Leg(phys, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(phys, IN))
    return SymmetricTensor.random(legs, seed=seed).to_backend("jax")


# --- observables -------------------------------------------------------------------


def log_kappa(beta, env, k: int = 4):
    """``ln`` of the partition function per site, from ``k`` unrolled moves at ``beta``.

    ``kappa = Z(L+1,L+1) Z(L,L) / Z(L+1,L) Z(L,L+1)``, Baxter's corner-transfer
    telescoping: four corners cover an ``L x L`` patch, adding four edges and one bulk
    tensor covers ``(L+1) x (L+1)``, and adding only the left and right edges covers
    ``L x (L+1)``. Every leg closes except one bond, which ``scalar`` traces. ``env`` is
    the ``CTMEnv`` from ``ctmrg``: the truncated backprop's *initial condition*, which
    carries no gradient, while the ``k`` moves inside do.
    """
    c0, e0, bond = env
    bulk = ising_bulk(beta)
    c, e = ctmrg_unrolled(c0, e0, single_layer(bulk), bond, k=k)
    cc, ca, ec, ea = ring(c, e)
    z_c = scalar(tenet.einsum("ab,ac,dc,eb->de", cc, ca, cc, ca))
    z_h = scalar(tenet.einsum("ab,ac,dcf,ed,eg,ghf->hb", cc, ca, ea, cc, ca, ec, optimize=PATH))
    z_a = scalar(
        tenet.einsum(
            "ab,acp,cd,edq,fe,gfr,gh,hks,spqr->kb",
            cc,
            ec,
            ca,
            ea,
            cc,
            ea,
            ca,
            ec,
            bulk,
            optimize=PATH,
        )
    )
    return ar.do("log", z_a * z_c / z_h**2)


def free_energy(beta, env, k: int = 4):
    """``-ln(kappa)/beta``, the free energy per site. Compare :func:`onsager`."""
    return -log_kappa(beta, env, k=k) / beta


def beta_free_energy(beta, env, k: int = 4):
    """``beta f = -ln kappa``. This is the function differentiated: ``d(beta f)/d beta`` is
    the internal energy per site, and the Onsager oracle has it in closed form."""
    return -log_kappa(beta, env, k=k)


def onsager(beta: float, points: int = 200_001) -> float:
    """``beta f`` from Onsager's closed form, by direct quadrature. NumPy, no ``scipy``.

    ``-beta f = ln 2 + (1/2pi) int_0^pi dtheta ln[cosh^2(2b) + (1/k) sqrt(1 + k^2 - 2k cos
    2theta)]``, ``k = 1/sinh^2(2b)``. The equivalent elliptic form is cross-checked in
    ``tests/integration/test_ctmrg.py`` before this is used to judge anything.
    """
    import numpy as np

    kk = 1.0 / np.sinh(2.0 * beta) ** 2
    theta = np.linspace(0.0, np.pi, points)
    integrand = np.log(
        np.cosh(2.0 * beta) ** 2 + np.sqrt(1.0 + kk**2 - 2.0 * kk * np.cos(2.0 * theta)) / kk
    )
    return -(np.log(2.0) / 2.0 + np.trapezoid(integrand, theta) / (2.0 * np.pi))


def _halves(r, ket, bra, phys1: str = "", phys2: str = ""):
    """The 2x1 environment, split down the middle into two halves.

    ``left`` is the bottom-left corner, the left edge, the top-left corner, the first top
    and bottom edges and the first site: legs ``(*phys1, b, c, k, h, r, R)`` where
    ``b``/``k`` are the ring's one open bond, ``c``/``h`` the cut through the top and
    bottom rows and ``r``/``R`` the bonds between the two sites; ``right`` is the mirror
    image. ``phys`` is ``""`` (physical legs closed, the denominator) or ``"Ww"`` -- bra
    first, then ket -- for the numerator. Each half is built the way ``double_layer`` builds
    a corner (environment, ket, bra), so the peak is rank 7 -- rank 8 with the physical legs
    open, froSTspin ``rdm.py``:30-69, ``a*d*chi**2*D**4`` -- and no double layer is formed.

    ponytail: two hand-written halves instead of one twelve-operand equation. The
    contraction is identical; what changes is that the intermediates are rank 5 and rank 3
    by construction rather than by whatever path ``opt_einsum`` picks from *physical* leg
    sizes -- which for an unevenly filled graded tensor it picks badly and unpredictably:
    the same network measured 0.7 s and 3.6 s for two SU(2) environments differing only in
    how ``chi`` split across sectors. Upgrade path: a path planner that costs a graded
    network by its *blocks*, which is M9.
    """
    cc, ca, ec, ea = r
    k1, b1 = (phys1[1], phys1[0]) if phys1 else ("s", "s")
    k2, b2 = (phys2[1], phys2[0]) if phys2 else ("s", "s")
    left = tenet.einsum("ij,jklL,ihdD->khlLdD", ca, ec, ea)
    left = tenet.einsum(f"khlLdD,{k1}lurd->khLD{k1}ur", left, ket)
    left = tenet.einsum(f"khLD{k1}ur,LU{b1}RD->{phys1}khuUrR", left, bra)
    left = tenet.einsum(f"{phys1}khuUrR,ab,acuU->{phys1}bckhrR", left, cc, ec, optimize=PATH)
    right = tenet.einsum("cduU,de,ferR->cfuUrR", ec, ca, ea)
    right = tenet.einsum(f"cfuUrR,{k2}lurd->cfUR{k2}ld", right, ket)
    right = tenet.einsum(f"cfUR{k2}ld,LU{b2}RD->{phys2}cflLdD", right, bra)
    right = tenet.einsum(f"{phys2}cflLdD,gf,hgdD->{phys2}chlL", right, cc, ea, optimize=PATH)
    return left, right


def energy(a: SymmetricTensor, h: SymmetricTensor, env, k: int = 4):
    """``<h> / <1>`` on a 2x1 patch, from ``k`` unrolled moves at the current ``a``.

    The ring is four corners, two top edges, two bottom edges and one edge on each side;
    each site enters as a ket and a bra absorbed one after the other, physical legs left
    **open** in the numerator so ``h`` closes them and closed against each other in the
    denominator. One bond stays open for ``scalar``. With :func:`_halves` this is a
    reduced-density-matrix API at one geometry, which is why it stayed out of the library.

    ponytail: ``h`` closes two open physical legs (froSTspin ``contract_open_corner``)
    rather than being inserted into the ket (YASTN's ``DoublePepsTensor(op=...)``), whose
    route is cheaper only for a *one-site* operator: ``h`` is two-site here, so inserting it
    means an SVD of ``h``, a new bond space and a truncation decision -- a third
    factorization in a file that already teaches two.
    """
    c0, e0, bond = env
    ket, bra = layers(c4v(a))
    r = ring(*ctmrg_unrolled(c0, e0, double_layer(ket, bra), bond, k=k))
    left, right = _halves(r, ket, bra, "Ww", "Xx")
    numerator = scalar(tenet.einsum("WwbckhrR,XxchrR,WXwx->kb", left, right, h, optimize=PATH))
    left, right = _halves(r, ket, bra)
    denominator = scalar(tenet.einsum("bckhrR,chrR->kb", left, right))
    return numerator / denominator


def step(a: SymmetricTensor, h: SymmetricTensor, env, lr: float, k: int = 4):
    """One plain SGD step on ``a``, ``vmc_mps.step``-style. ``optax`` would slot in here."""
    import jax

    value, grad = jax.value_and_grad(energy)(a, h, env, k)
    return jax.tree.map(lambda p, g: p - lr * g, a, grad), value


def main(chi_ising: int = 16, chi_ipeps: dict | None = None, k: int = 4, steps: int = 3):
    """Print both halves: Ising against Onsager with its gradient, then the iPEPS trace."""
    import jax

    import tenet.ad
    import tenet.pytree  # noqa: F401  # registration is the import's side effect

    tenet.ad.install()

    for beta in (0.3, 0.4, 0.5):
        env = ctmrg(*single_layer_ctm(ising_bulk(beta)), chi=chi_ising)[0]
        bf = float(beta_free_energy(beta, env, k=k))
        grad = float(jax.grad(beta_free_energy)(beta, env, k))
        print(
            f"ising beta={beta:.2f}  beta*f={bf:+.10f}  onsager={onsager(beta):+.10f}  "
            f"rel={abs(bf / onsager(beta) - 1):.2e}  d(beta f)/dbeta={grad:+.8f}"
        )

    for provider in ("u1", "su2"):
        a, h = build_ipeps(provider), build_h(provider)
        env = ctmrg(*double_layer_ctm(c4v(a)), chi=(chi_ipeps or CHI_IPEPS)[provider])[0]
        trace = []
        for _ in range(steps):
            a, value = step(a, h, env, lr=0.01, k=k)
            trace.append(float(value))
        print(f"ipeps {provider}: " + " ".join(f"{v:+.8f}" for v in trace))


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)  # tests/conftest.py does this for the suite
    main()
