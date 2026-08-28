"""Differentiable CTMRG against an oracle rather than against itself.

This is the test in the repository that judges a *physics* result against a closed form:
a CTM environment for the classical 2D Ising model is converged on
[EnvCTMc4v][tenet.network.EnvCTMc4v], its free energy per site is compared with Onsager's
-- which is itself pinned here by two independent closed forms before it is used to judge
anything -- and ``k`` fixed-bond moves are then differentiated and checked against the
*internal energy* of the same oracle. The variational half of the same lane, where there
is no closed form to answer to, is ``test_ctmrg_ipeps.py``.

**The bulk is Z2-graded**, and that is what lets it enter the ordered phase at all:
``beta in {0.5, 0.6}`` are oracle points here. Three things the grading buys and one it
does not, each asserted below -- Onsager on both sides of ``beta_c``; ``<s> = 0`` proved by
a ``from_dense`` *refusal* rather than by a small float; exactly two-fold *cross-sector*
degeneracy of the ordered corner spectrum, which even ``chi`` provably never splits; and
**not** a ``NaN`` criterion, which :func:`test_the_retired_nan_criterion` retires with its
reasoning.

The Boltzmann tensor sits on four *identical* legs -- the C4v ansatz's signature -- and is
symmetric under every permutation of them, so it carries the whole point group and one
corner and one edge describe its environment exactly.

Like ``tests/integration/test_vmc.py`` it adds nothing to ``src/tenet``; the only code of
its own is the second Onsager form, the observables -- Baxter's telescoping, which is
geometry-specific and therefore not the library's -- and the tolerances.

x64 is enabled process-globally in ``tests/conftest.py``; every tolerance here depends on
it.

**Runtime** is the teaching lane's own ``main()``, a measured 52 s warm; everything else
here is under 20 s, because the environments are converged once per module and shared and
no test spends a gradient on a claim the oracle comparisons do not already make.
``jax.jit`` around the traced region stays rejected: compiling the gradient costs more
than running the few eager ones this module needs, and it would hide the point that
``iterate_`` sits outside the trace and ``update_(bond=B)`` inside it.
"""

import contextlib
import io
import math
import pathlib
import sys
from functools import partial

import autoray as ar
import numpy as np
import pytest
from helpers import check_example_page

import tenet
from tenet import OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import EnvCTMc4v, Peps, SquareLattice, flip
from tenet.structure import TensorStructure
from tenet.symmetry import Z2, Z2Sector

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import tenet.ad  # noqa: E402
import tenet.pytree  # noqa: E402, F401  # registration is the import's side effect

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples" / "toy_codes"))
import ctmrg  # noqa: E402  # the teaching lane, which writes its own CTMRG out
from ising import onsager  # noqa: E402  # the closed form, borrowed rather than copied

# ``opt_einsum``'s greedy path, for the patch contractions only. Simplification: greedy,
# not "auto" -- at seven operands "auto"'s dynamic-programming *search* costs an order of
# magnitude more than the contraction it plans, and it is re-run on every call.
PATH = "greedy"

CHI, K = 16, 4
CHI_IPEPS, K_IPEPS = 6, 2
PROVIDERS = ["u1", "su2"]

BETA_C = 0.4406867935097714  # ln(1 + sqrt(2)) / 2

# The ordered phase asks a sharper question than the disordered one -- whether two singular
# values in *different* parity blocks are equal to the last bits -- so its environments are
# swept to the float64 floor rather than to the default 1e-10.
TOL_ORDERED, SWEEPS_ORDERED = 1e-14, 300

# Converging an environment is the expensive part, so each is built once per module.
# Simplification: a plain dict, not a fixture per beta -- the key is a tuple of floats and
# ints, and nothing below mutates a stored environment (every traced run seeds a fresh one).
_ENVS: dict = {}


# --- the classical model, its observable and its two closed forms -------------------


def ising(beta: float) -> SymmetricTensor:
    """The Boltzmann tensor on four identical Z2 legs, ``a[t,l,b,r] = sum_s prod W[s,.]``.

    ``W = [[sqrt cosh b, sqrt sinh b], [sqrt cosh b, -sqrt sinh b]]`` *is* the parity
    basis, so summing over ``s`` annihilates every entry with an odd number of odd legs:
    the grading is the statement, not a claim checked afterwards.
    """
    c, s = math.sqrt(math.cosh(beta)), math.sqrt(math.sinh(beta))
    w = np.array([[c, s], [c, -s]])
    block = np.einsum("st,sl,sb,sr->tlbr", w, w, w, w)
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    return SymmetricTensor.from_dense(block, (Leg(space, OUT),) * 4)


def traced_ising(beta):
    """:func:`ising` with a *traced* ``beta``: blocks through ``jax.numpy``, no
    ``from_dense``, because a symmetry check is a concrete-value question."""
    c, s = jnp.sqrt(jnp.cosh(beta)), jnp.sqrt(jnp.sinh(beta))
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    legs = (Leg(space, OUT),) * 4
    blocks = {}
    for key in TensorStructure(legs).block_order:
        w = [c if sector.parity == 0 else s for sector in key.output_tree.uncoupled]
        w += [c if sector.parity == 0 else s for sector in key.input_tree.uncoupled]
        blocks[key] = jnp.full((1, 1, 1, 1), 2.0 * (w[0] * w[1] * w[2] * w[3]))
    return SymmetricTensor.from_blocks(legs, blocks)


def onsager_elliptic(beta: float, points: int = 200_001) -> float:
    """``beta f`` from the elliptic form: ``-beta f = ln(2 cosh 2b) + (1/2pi) int_0^pi dphi
    ln[(1 + sqrt(1 - kappa^2 sin^2 phi))/2]``, ``kappa = 2 sinh(2b)/cosh^2(2b)``.

    Algebraically the same number as ``examples/toy_codes/ising.py``'s ``onsager`` and
    numerically an independent route to it: different integrand, different singularity
    structure, same grid rule.
    """
    kappa = 2.0 * np.sinh(2.0 * beta) / np.cosh(2.0 * beta) ** 2
    phi = np.linspace(0.0, np.pi, points)
    integrand = np.log((1.0 + np.sqrt(1.0 - (kappa * np.sin(phi)) ** 2)) / 2.0)
    return -(np.log(2.0 * np.cosh(2.0 * beta)) + np.trapezoid(integrand, phi) / (2.0 * np.pi))


def log_kappa(env, site=(0, 0)):
    """``ln`` of the partition function per site, Baxter's corner-transfer telescoping.

    ``kappa = z_a z_c / z_h**2``: four corners cover an ``L x L`` patch, four corners with
    four edges and the bulk tensor cover ``(L + 1) x (L + 1)``, and four corners with two
    edges cover ``L x (L + 1)``. The four-corner and four-corner-two-edge objects put two
    corners next to each other, which crosses a sublattice boundary, so one of each pair
    enters flipped; the eight-tensor ring around a site does not, because corners and edges
    already alternate around it.
    """
    e, a = env[site], env.psi[site]
    c, cf, t, tf = e.tl, flip(e.tl), e.t, flip(e.t)
    z_c = tenet.full_trace(tenet.einsum("ab,ac,dc,eb->de", c, cf, c, cf))
    z_h = tenet.full_trace(tenet.einsum("ab,ac,dfc,ed,eg,gfh->hb", c, cf, tf, cf, c, t))
    z_a = tenet.full_trace(
        tenet.einsum("ab,apc,cd,eqd,fe,grf,gh,hsk,spqr->kb", c, t, c, t, c, t, c, t, a)
    )
    return ar.do("log", z_a * z_c / z_h**2)


def beta_free_energy(env, site=(0, 0)):
    """``beta f = -ln kappa``. Differentiating it in ``beta`` gives the internal energy per
    site, which Onsager has in closed form."""
    return -log_kappa(env, site)


def converged(beta: float, chi: int = CHI, tol: float = 1e-10, max_sweeps: int = 100):
    """``(env, CTMRG_out)``, memoized on the full set of knobs that decides it."""
    key = (beta, chi, tol, max_sweeps)
    if key not in _ENVS:
        env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising(beta)))
        _ENVS[key] = (env, env.iterate_(max_bond=chi, max_sweeps=max_sweeps, corner_tol=tol))
    return _ENVS[key]


def ordered(beta: float, chi: int = CHI):
    return converged(beta, chi, TOL_ORDERED, SWEEPS_ORDERED)


def spectrum_by_sector(corner: SymmetricTensor) -> dict[int, list[float]]:
    """The corner spectrum split by Z2 charge, each half descending.

    [spectrum][tenet.network.spectrum] deliberately throws the sector labels away and every
    ordered-phase criterion below is *about* those labels, so they are recovered here. The
    renormalized corner is ``V^dagger U S`` rather than a diagonal, so the spectrum is an
    SVD of it and not a block diagonal.
    """
    s = tenet.linalg.svd(corner, ((0,), (1,)))[1]
    return {
        sector.parity: sorted((float(v) for v in ar.do("diag", m)), reverse=True)
        for sector, m in tenet.to_matrices(s).items()
    }


@pytest.fixture(scope="module", autouse=True)
def broadened_svd():
    """``tenet.ad`` is process-global, so this module installs it and puts it back.

    Every gradient here runs through a factorization of a symmetric fixed point, which is
    where degenerate singular values live; ``test_the_retired_nan_criterion`` is the one
    test that deliberately runs both ways.
    """
    tenet.ad.install()
    yield
    tenet.ad.uninstall()


# --- 1. the oracle, pinned before it is used --------------------------------------


@pytest.mark.parametrize("beta", [0.2, 0.3, 0.4, 0.5])
def test_the_two_onsager_closed_forms_agree(beta):
    assert onsager(beta) == pytest.approx(onsager_elliptic(beta), abs=1e-12)


# --- 2. CTMRG against the oracle ---------------------------------------------------


@pytest.mark.parametrize("beta", [0.3, 0.4, 0.5])
def test_free_energy_matches_onsager(beta):
    env, _ = converged(beta)
    assert float(beta_free_energy(env)) == pytest.approx(onsager(beta), rel=1e-6)


@pytest.mark.parametrize("beta", [0.5, 0.6])
def test_free_energy_matches_onsager_in_the_ordered_phase(beta):
    """Onsager's closed form is valid on both sides of ``beta_c``, but an *ungraded*
    finite-chi environment may break the Z2 symmetry spuriously in the ordered phase --
    YASTN passes ``sym='Z2'`` to its CTMRG Ising example precisely to stop that -- so the
    grading is what puts an oracle here at all."""
    assert beta > BETA_C
    env, _ = ordered(beta)
    assert float(beta_free_energy(env)) == pytest.approx(onsager(beta), rel=1e-6)


def test_the_sweep_stops_on_the_tolerance():
    """``iterate_`` exits because the corner spectrum stopped moving, not because it ran
    out of sweeps -- which is what every oracle comparison above is entitled to assume."""
    _, out = converged(0.4)
    assert out.converged


# --- 3. the gradient against the oracle's derivative -------------------------------


def unrolled_beta_f(beta, seed, bond, k=K):
    """``beta f`` after exactly ``k`` fixed-bond moves from ``seed``.

    ``seed`` is the converged ``(corner, edge)``: the truncated backprop's *initial
    condition*, which carries no gradient, while the ``k`` moves inside do.
    """
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_ising(beta)), init=None)
    env.env[0, 0].tl, env.env[0, 0].t = seed
    for _ in range(k):
        env.update_(bond=bond)
    return beta_free_energy(env)


def warm(beta: float, chi: int = CHI, tol: float = 1e-10, max_sweeps: int = 100):
    """``(seed, bond)`` for :func:`unrolled_beta_f`, from a converged environment built on
    the *traced* bulk builder so that the blocks are already JAX arrays."""
    key = ("warm", beta, chi, tol, max_sweeps)
    if key not in _ENVS:
        env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_ising(beta)))
        env.iterate_(max_bond=chi, max_sweeps=max_sweeps, corner_tol=tol)
        local = env[0, 0]
        _ENVS[key] = ((local.tl, local.t), local.tl.legs[0].space)
    return _ENVS[key]


def test_grad_matches_the_onsager_internal_energy_and_central_differences():
    beta, delta = 0.4, 1e-5
    seed, bond = warm(beta)
    got = float(jax.grad(unrolled_beta_f)(beta, seed, bond, K))
    oracle = (onsager(beta + delta) - onsager(beta - delta)) / (2 * delta)
    assert got == pytest.approx(oracle, rel=1e-4)

    fd = (
        float(unrolled_beta_f(beta + delta, seed, bond, K))
        - float(unrolled_beta_f(beta - delta, seed, bond, K))
    ) / (2 * delta)
    assert got == pytest.approx(fd, rel=1e-6)


def test_grad_matches_the_onsager_internal_energy_in_the_ordered_phase():
    """``beta = 0.6``, the same oracle on the other side of ``beta_c``. The corner spectrum
    there is exactly doubled across the parity sectors, and the gradient is finite anyway
    -- see :func:`test_the_retired_nan_criterion`."""
    beta, delta = 0.6, 1e-5
    seed, bond = warm(beta, CHI, TOL_ORDERED, SWEEPS_ORDERED)
    got = float(jax.grad(unrolled_beta_f)(beta, seed, bond, K))
    oracle = (onsager(beta + delta) - onsager(beta - delta)) / (2 * delta)
    assert got == pytest.approx(oracle, rel=1e-4)


# --- 4. the outside-decide / inside-differentiate boundary -------------------------


def test_the_traced_region_traces_once_and_the_frozen_bond_is_static():
    beta = 0.4
    seed, bond = warm(beta)
    count = 0

    @partial(jax.jit, static_argnums=(2, 3))
    def objective(seed, beta, bond, k):
        nonlocal count
        count += 1  # a Python side effect: trace time only
        env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_ising(beta)), init=None)
        env.env[0, 0].tl, env.env[0, 0].t = seed
        for _ in range(k):
            env.update_(bond=bond)
        corner = env[0, 0].tl
        dagger = tenet.adjoint(corner)
        # `norm(corner)` would be the constant 1, since every move renormalizes, and a
        # constant has nothing to differentiate
        return tenet.full_trace(tenet.einsum("ab,ac,dc,eb->de", corner, dagger, corner, dagger))

    grad = jax.grad(objective, argnums=1)
    a = grad(seed, beta, bond, K)
    b = grad(seed, beta + 0.01, bond, K)
    assert count == 1  # different block values, same structure, one trace
    assert not np.isclose(float(a), float(b))

    smaller, _ = converged(beta, 8)
    smaller_bond = smaller[0, 0].tl.legs[0].space
    assert smaller_bond != bond
    grad(seed, beta, bond, K)
    assert count == 1  # the same frozen bond does not retrace ...
    grad(seed, beta, smaller_bond, K)
    assert count == 2  # ... a different one does: the GradedSpace is part of the key


def test_deciding_a_bond_is_refused_under_jit():
    """The boundary is structural, not a convention, and it has two independent halves.

    ``update_(max_bond=...)`` decides a bond space from the singular *values*, which is
    ``tenet.StructureChangingError`` by construction, and ``iterate_`` runs that same
    sweep, so it raises there first. Its *loop exit* is a second, separate obstacle:
    ``corner_spectra`` reads a spectrum to decide when to stop, and a data-dependent loop
    exit is not a tracing edge case -- it is the thing the outside/inside split exists to
    keep outside. It is shown on a traceable ``update_(bond=B)`` move, where the
    truncation is no longer in the way.
    """
    seed, bond = warm(0.4)

    def decide(beta):
        env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_ising(beta)))
        env.update_(max_bond=4)
        return log_kappa(env)

    with pytest.raises(tenet.StructureChangingError, match="tenet.linalg.svd"):
        jax.jit(decide)(0.4)

    def exit_criterion(beta):
        env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_ising(beta)), init=None)
        env.env[0, 0].tl, env.env[0, 0].t = seed
        env.update_(bond=bond)
        return env.corner_spectra()

    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(exit_criterion)(0.4)


@pytest.mark.parametrize(("beta", "chi"), [(0.44, CHI), (0.6, CHI)])
def test_the_retired_nan_criterion(beta, chi):
    """A ``NaN``-without-``tenet.ad`` criterion, **retired** -- with the reason.

    The criterion expected ``NaN`` from JAX's stock SVD VJP near ``beta_c`` and a finite
    gradient from ``tenet.ad``. The Z2 grading *does* produce exact degeneracies in the
    ordered-phase corner, and the ``NaN`` still does not appear. That is not a near miss;
    it is structural:

    **every exact degeneracy in the Z2 Ising corner is *across* parity sectors**, because
    the two partners differ by the global spin flip, which *is* the Z2 charge; and a graded
    factorization runs one SVD **per coupled sector**, exactly as ``tenet.ad`` broadens per
    coupled sector. So the blocks separate the partners before any VJP sees them, and every
    within-sector gap stays healthy -- pinned by
    :func:`test_no_exact_within_sector_degeneracy_anywhere`. **Grading is what removes the
    ``NaN``, not what creates it.**

    Where the ``NaN`` would live is the *ungraded* ordered-phase run, whose single SVD sees
    the cross-sector doublet as one degenerate pair. That configuration is precisely the one
    a finite-chi ordered run is fenced away from for YASTN's spurious-symmetry-breaking
    reason, so it is measured and reported in the PR body, never asserted on.

    What is pinned here: the gradient is finite **with and without** ``tenet.ad.install()``
    on both sides of ``beta_c``, and the two agree to ``1e-9``, so the broadening does not
    perturb an answer whose per-sector gaps are healthy.
    """
    tol, sweeps = (TOL_ORDERED, SWEEPS_ORDERED) if beta > BETA_C else (1e-10, 60)
    seed, bond = warm(beta, chi, tol, sweeps)

    broadened = float(jax.grad(unrolled_beta_f)(beta, seed, bond, K))
    assert np.isfinite(broadened)
    tenet.ad.uninstall()
    try:
        stock = float(jax.grad(unrolled_beta_f)(beta, seed, bond, K))
    finally:
        tenet.ad.install()
    assert np.isfinite(stock)
    assert stock == pytest.approx(broadened, rel=1e-9)


# --- 4b. what the Z2 grading buys --------------------------------------------------


@pytest.mark.parametrize("beta", [0.3, 0.44, 0.6])
def test_the_graded_bulk_is_the_same_numbers(beta):
    """The Z2-block weight is the dense ``sum_s W W W W`` tensor, and the traced builder
    -- which cannot run ``from_dense``, because a symmetry check is a concrete-value
    question -- is the same tensor again.

    Summing over ``s`` annihilates every entry with an odd number of odd legs: eight of
    sixteen are exactly zero and have no block to live in.
    """
    c, s = math.sqrt(math.cosh(beta)), math.sqrt(math.sinh(beta))
    w = np.array([[c, s], [c, -s]])
    dense = np.einsum("st,sl,sb,sr->tlbr", w, w, w, w)
    np.testing.assert_allclose(np.asarray(ising(beta).to_dense()), dense, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(traced_ising(beta).to_dense()), dense, atol=1e-14)
    odd = [idx for idx in np.ndindex(dense.shape) if sum(idx) % 2]
    assert len(odd) == 8
    assert all(dense[idx] == 0.0 for idx in odd)


@pytest.mark.parametrize("beta", [0.3, 0.44, 0.6])
def test_zero_magnetization_is_structural_not_numerical(beta):
    """``<s> = 0`` proved by a refusal, YASTN's "zero magnetization by symmetry".

    The spin-insertion impurity ``sum_s s * prod_i W[s,i]`` is nonzero exactly when the
    number of *odd* legs is odd -- it is a Z2-**odd** tensor, and no invariant
    ``SymmetricTensor`` can hold one. ``from_dense`` therefore refuses it with a residual
    naming an offending sector tuple, and *that refusal is the statement*: exact and
    structural, where an ungraded run could only offer a small float.

    Simplification: measuring a genuine ``<s>`` (rather than proving it zero) wants a dummy
    leg in the odd sector; nothing here needs one.
    """
    c, s = math.sqrt(math.cosh(beta)), math.sqrt(math.sinh(beta))
    w = np.array([[c, s], [c, -s]])
    impurity = np.einsum("s,st,sl,sb,sr->tlbr", np.array([1.0, -1.0]), w, w, w, w)
    assert np.abs(impurity).max() > 0.1  # the impurity is not the zero array
    with pytest.raises(ValueError) as excinfo:
        SymmetricTensor.from_dense(impurity, ising(beta).legs)
    message = str(excinfo.value)
    assert "not symmetric" in message
    assert "Z2Sector(parity=1)" in message  # the odd sector is named


@pytest.mark.parametrize("beta", [0.5, 0.6])
def test_ordered_phase_spectrum_is_an_exact_cross_sector_doublet(beta):
    """Spontaneous-symmetry-breaking doubling, and it is cross-sector by construction: the
    two partners differ by the global spin flip, which is the Z2 charge.

    The assertion is at ``1e-12``, four orders below anything the disordered phase produces
    and eleven below the ``2.5e-3`` a phase-blind test would accept.
    """
    assert beta > BETA_C
    env, _ = ordered(beta)
    halves = spectrum_by_sector(env[0, 0].tl)
    assert len(halves[0]) == len(halves[1])
    top = halves[0][0]
    deviations = [abs(x - y) / top for x, y in zip(halves[0], halves[1], strict=True)]
    assert max(deviations) < 1e-12, deviations


@pytest.mark.parametrize(("beta", "chi"), [(0.3, 8), (0.44, 8), (0.44, CHI)])
def test_disordered_phase_has_no_such_pairing(beta, chi):
    """The other half of the criterion: it distinguishes the phases rather than merely
    passing. Below ``beta_c`` the two sectors' spectra are unrelated, and the full spectrum
    has no degeneracy at all."""
    assert beta < BETA_C
    env, _ = converged(beta, chi, 1e-10, 200)
    halves = spectrum_by_sector(env[0, 0].tl)
    top = max(halves[0][0], halves[1][0])
    n = min(len(halves[0]), len(halves[1]))
    closest = min(abs(halves[0][i] - halves[1][i]) / top for i in range(n))
    assert closest > 1e-5, closest
    every = sorted(halves[0] + halves[1], reverse=True)
    assert min(abs(x - y) / top for x, y in zip(every, every[1:], strict=False)) > 1e-5


@pytest.mark.parametrize(
    ("beta", "chi"), [(0.3, 8), (0.44, 8), (0.44, CHI), (0.4, CHI), (0.5, CHI), (0.6, CHI)]
)
def test_no_exact_within_sector_degeneracy_anywhere(beta, chi):
    """The fact that decides the ``NaN`` question, since ``tenet.ad`` broadens per coupled
    sector: nothing is ever exactly degenerate *within* a sector, on either side of
    ``beta_c``."""
    env, _ = ordered(beta, chi) if beta > BETA_C else converged(beta, chi, 1e-10, 200)
    halves = spectrum_by_sector(env[0, 0].tl)
    top = max(halves[0][0], halves[1][0])
    gaps = [
        abs(x - y) / top
        for values in halves.values()
        for x, y in zip(values, values[1:], strict=False)
    ]
    assert min(gaps) > 1e-12, min(gaps)


@pytest.mark.parametrize("chi", [8, 16])
def test_even_chi_never_splits_a_doublet(chi):
    """``svd_truncated(max_bond=chi)`` ranks singular values *globally* across sectors, and
    the doublet partners are exactly equal and therefore adjacent in that ranking -- so any
    even ``chi`` keeps both. This is the bosonic analogue of the SU(2) multiplet-splitting
    hazard (Francuz, Schuch, Vanhecke, PRR 7, 013237 (2025), Appendix C: *"be careful not
    to split multiplets when converging the original CTM"*) -- cheaper here, because the
    multiplet size is 2 and known. Asserted, not assumed.
    """
    assert chi % 2 == 0
    env, _ = ordered(0.6, chi)
    halves = spectrum_by_sector(env[0, 0].tl)
    assert len(halves[0]) == len(halves[1]) == chi // 2  # equal degeneracy in both sectors
    top = halves[0][0]
    assert max(abs(x - y) / top for x, y in zip(halves[0], halves[1], strict=True)) < 1e-12


def test_main_runs_both_halves():
    """The teaching lane's standalone entry point, which writes its own CTMRG out and
    imports nothing from ``tenet.network``. ``steps=1`` stays: raising to ``main()``'s
    default 3 costs a measured 17.4 s warm, against this module's 100 s budget."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ctmrg.main(chi_ising=CHI, k=K, steps=1)
    check_example_page("toy-ctmrg.md", buf.getvalue())
