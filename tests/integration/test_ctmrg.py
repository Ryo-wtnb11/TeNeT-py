"""Issues #102/#104 — differentiable CTMRG: the first end-to-end algorithm, against an oracle.

This is the first test in the repository that judges a *physics* result against a closed
form rather than against a self-consistency check: ``examples/ctmrg.py`` converges a CTMRG
environment for the classical 2D Ising model and its free energy per site is compared with
Onsager's, which is itself pinned here by two independent closed forms before it is used to
judge anything. It then differentiates ``k`` unrolled fixed-structure sweeps and checks the
gradient against the *internal energy* of the same oracle.

**The Ising half is Z2-graded since #104**, and that is what lets it enter the ordered
phase at all: ``beta in {0.5, 0.6}`` are oracle points here, where #102 had to stop at
``beta < beta_c``. Three things the grading buys and one it does not, each asserted below —
Onsager on both sides of ``beta_c``; ``<s> = 0`` proved by a ``from_dense`` *refusal*
rather than by a small float; exactly two-fold *cross-sector* degeneracy of the ordered
corner spectrum, which even ``chi`` provably never splits; and **not** #102's ``NaN``
criterion, which :func:`test_the_retired_nan_criterion` retires with its reasoning rather
than deferring it a third time.

Like ``tests/integration/test_vmc.py`` (#69) it adds nothing to ``src/tenet`` and nothing to
the example: it imports ``examples/ctmrg.py`` and runs it, so the example cannot rot. The
only code of its own is the second Onsager form and the tolerances.

x64 is enabled process-globally in ``tests/conftest.py``; every tolerance here depends on it.

**Runtime, measured and over budget.** #102 asked for 30 s for this module against
``test_vmc.py``'s 2.92 s. The Ising half -- every test with an oracle behind it -- was 5.7 s
ungraded and is **11.0 s** graded, of which about 1.5 s is #104's new tests and the rest is
the grading making the *existing* gradient tests slower. #104 asked for ``chi`` to come down
before the budget went up; measured, ``chi`` is not the lever. The cost is the **first**
gradient through each distinct bond structure -- the fusion-tree enumeration and per-block
plan build -- and it is 3.28 s at ``chi=8`` against 2.64 s at ``chi=16``, while every
subsequent gradient through an already-planned structure is 0.37 s at either. So the saving
came from *reusing* structures instead of shrinking them: the retired-``NaN`` test moved
from ``chi=8`` to ``chi=16`` (13.5 s -> 11.0 s), which costs nothing -- the near-critical
within-sector gap is ``6.6e-3`` there against ``5.1e-2`` at ``chi=8``, the same statement.
What would actually move this number is the graded plan cache, which is M9's work, not a
tolerance here.

The iPEPS half is the other 100 s, and it is not ``chi`` or ``K``: those are already at 4/6
and 2, and lowering them further makes SU(2) *slower*, not faster. It is the double-layer
tensor. Building ``adjoint(a) . a`` with the physical legs open goes through a rank-10
intermediate whose fusion-tree enumeration and per-block plan construction is thousands of
Python-level blocks, and reverse mode walks all of it again: for SU(2) one forward energy
costs 0.7 s and one gradient 6 s, of which 3 s is that one tensor. Nothing in this file
fixes that; what would is a cheaper double-layer construction in ``src/tenet`` (fusing
before contracting, or a graded contraction planner that costs a network by its blocks
rather than by its dense shapes), which is M9's contraction-path work and its own issue.
"""

import math
import pathlib
import sys
from functools import partial

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import SymmetricTensor

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import tenet.ad  # noqa: E402
import tenet.pytree  # noqa: E402, F401  # registration is the import's side effect

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples"))
import ctmrg  # noqa: E402

CHI, K = 16, 4
PROVIDERS = ["u1", "su2"]

K_IPEPS = 2

# The ordered phase asks a sharper question than the disordered one -- whether two singular
# values in *different* parity blocks are equal to the last bits -- so its environments are
# swept to the float64 floor rather than to the default 1e-10. Measured: 134 sweeps at
# beta=0.5 and 66 at beta=0.6, each well under a tenth of a second at chi=16.
TOL_ORDERED, SWEEPS_ORDERED = 1e-14, 200

# Converging an environment is the expensive part and nothing here mutates one, so each is
# built once per module. ponytail: a plain dict, not a fixture per beta -- the key is a
# tuple of floats and ints and the values are immutable.
_ENVS: dict = {}


def converged(beta: float, chi: int = CHI, tol: float = 1e-10, max_sweeps: int = 100):
    """``(c, e, bond, history)``, memoized on the full set of knobs that decides it."""
    key = (beta, chi, tol, max_sweeps)
    if key not in _ENVS:
        _ENVS[key] = ctmrg.converge(ctmrg.ising_bulk(beta), chi=chi, tol=tol, max_sweeps=max_sweeps)
    return _ENVS[key]


def env(beta: float, chi: int = CHI, tol: float = 1e-10, max_sweeps: int = 100):
    return converged(beta, chi, tol, max_sweeps)[:3]


def ordered_env(beta: float, chi: int = CHI):
    return converged(beta, chi, TOL_ORDERED, SWEEPS_ORDERED)


def spectrum_by_sector(c: SymmetricTensor) -> dict[int, list[float]]:
    """The corner spectrum split by Z2 charge, each half descending.

    ``ctmrg.spectrum`` deliberately throws the sector labels away; every ordered-phase
    criterion below is *about* those labels, so they are recovered here rather than in the
    example. The corner is diagonal by construction, so this reads block diagonals.
    """
    return {
        sector.parity: sorted((float(v) for v in ar.do("diag", m)), reverse=True)
        for sector, m in tenet.to_matrices(c).items()
    }


def ising_block(beta: float) -> np.ndarray:
    """The dense ``(2,2,2,2)`` Ising bulk, written here independently of the example so
    "the regrade changed no numbers" is a claim against a second source."""
    c, s = math.sqrt(math.cosh(beta)), math.sqrt(math.sinh(beta))
    w = np.array([[c, s], [c, -s]])
    return np.einsum("sl,su,sr,sd->lurd", w, w, w, w)


def ipeps_env(provider: str):
    if provider not in _ENVS:
        a, h = ctmrg.build_ipeps(provider), ctmrg.build_h(provider)
        chi = ctmrg.CHI_IPEPS[provider]
        _ENVS[provider] = (a, h, ctmrg.converge(ctmrg.ipeps_bulk(a), chi=chi)[:3])
    return _ENVS[provider]


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


def onsager_elliptic(beta: float, points: int = 200_001) -> float:
    """``beta f`` from the elliptic form: ``-beta f = ln(2 cosh 2b) + (1/2pi) int_0^pi dphi
    ln[(1 + sqrt(1 - kappa^2 sin^2 phi))/2]``, ``kappa = 2 sinh(2b)/cosh^2(2b)``.

    Algebraically the same number as ``ctmrg.onsager``'s form and numerically an independent
    route to it: different integrand, different singularity structure, same grid rule.
    """
    kappa = 2.0 * np.sinh(2.0 * beta) / np.cosh(2.0 * beta) ** 2
    phi = np.linspace(0.0, np.pi, points)
    integrand = np.log((1.0 + np.sqrt(1.0 - (kappa * np.sin(phi)) ** 2)) / 2.0)
    return -(np.log(2.0 * np.cosh(2.0 * beta)) + np.trapezoid(integrand, phi) / (2.0 * np.pi))


@pytest.mark.parametrize("beta", [0.2, 0.3, 0.4, 0.5])
def test_the_two_onsager_closed_forms_agree(beta):
    assert ctmrg.onsager(beta) == pytest.approx(onsager_elliptic(beta), abs=1e-12)


# --- 2. CTMRG against the oracle ---------------------------------------------------


@pytest.mark.parametrize("beta", [0.3, 0.4, 0.5])
def test_free_energy_matches_onsager(beta):
    got = float(ctmrg.beta_free_energy(beta, env(beta), k=K))
    assert got == pytest.approx(ctmrg.onsager(beta), rel=1e-6)


@pytest.mark.parametrize("beta", [0.5, 0.6])
def test_free_energy_matches_onsager_in_the_ordered_phase(beta):
    """New with #104's grading, and the reason it exists.

    Onsager's closed form is valid on both sides of ``beta_c``, but an *ungraded*
    finite-chi environment may break the Z2 symmetry spuriously in the ordered phase --
    YASTN passes ``sym='Z2'`` to its CTMRG Ising example precisely to stop that -- so #102
    fenced this file into ``beta < beta_c`` and had no oracle here. Measured relative
    deviations: ``5.5e-14`` at ``beta=0.5`` and ``5.6e-16`` at ``beta=0.6``.
    """
    assert beta > ctmrg.BETA_C
    got = float(ctmrg.beta_free_energy(beta, ordered_env(beta)[:3], k=K))
    assert got == pytest.approx(ctmrg.onsager(beta), rel=1e-6)


def test_convergence_is_monotone_and_terminating():
    """Asserted, not assumed: the corner-spectrum change reaches ``tol`` inside
    ``max_sweeps``, decreases over the run, and takes a pinned number of sweeps."""
    history = converged(0.4)[3]
    assert history[-1] < 1e-10
    assert len(history) < 100  # terminated on the tolerance, not on max_sweeps
    assert 40 <= len(history) <= 120  # 72 as measured under the Z2 grading; the range is the pin
    tail = history[len(history) // 2 :]
    assert all(b < a for a, b in zip(tail, tail[1:], strict=False)), tail


# --- 3. the gradient against the oracle's derivative -------------------------------


def test_grad_matches_the_onsager_internal_energy_and_central_differences():
    beta, delta = 0.4, 1e-5
    got = float(jax.grad(ctmrg.beta_free_energy)(beta, env(beta), K))
    oracle = (ctmrg.onsager(beta + delta) - ctmrg.onsager(beta - delta)) / (2 * delta)
    assert got == pytest.approx(oracle, rel=1e-4)

    fd = (
        float(ctmrg.beta_free_energy(beta + delta, env(beta), k=K))
        - float(ctmrg.beta_free_energy(beta - delta, env(beta), k=K))
    ) / (2 * delta)
    assert got == pytest.approx(fd, rel=1e-6)


def test_grad_matches_the_onsager_internal_energy_in_the_ordered_phase():
    """``beta = 0.6``, the same oracle on the other side of ``beta_c``. Measured relative
    deviation ``8.6e-11``; the corner spectrum there is exactly doubled across the parity
    sectors, and the gradient is finite anyway -- see :func:`test_the_retired_nan_criterion`."""
    beta, delta = 0.6, 1e-5
    got = float(jax.grad(ctmrg.beta_free_energy)(beta, ordered_env(beta)[:3], K))
    oracle = (ctmrg.onsager(beta + delta) - ctmrg.onsager(beta - delta)) / (2 * delta)
    assert got == pytest.approx(oracle, rel=1e-4)


def test_k_dependence_is_measured_not_assumed():
    """Two statements, because a converged environment makes the interesting one invisible.

    At the default ``tol=1e-10`` the four gradients agree to float64 noise: the environment
    is *at* its fixed point when the traced region starts, so the truncation is irrelevant
    and no ratio computed from those differences means anything. Deliberately stopping the
    sweep early (``tol=1e-6``) puts the K-dependence above the noise floor, and there the
    truncated backprop is visibly converging.
    """
    converged = {k: float(jax.grad(ctmrg.beta_free_energy)(0.4, env(0.4), k)) for k in (1, 2, 4, 8)}
    assert max(abs(g / converged[8] - 1) for g in converged.values()) < 1e-11

    beta = 0.25
    loose = ctmrg.converge(ctmrg.ising_bulk(beta), chi=CHI, tol=1e-6)[:3]
    gradients = {k: float(jax.grad(ctmrg.beta_free_energy)(beta, loose, k)) for k in (1, 2, 4, 8)}
    first = abs(gradients[1] - gradients[2])
    last = abs(gradients[4] - gradients[8])
    assert last < first / 10, gradients


# --- 4. the outside-decide / inside-differentiate boundary -------------------------


def test_unrolled_traces_once_and_the_frozen_bond_is_static():
    beta = 0.4
    c, e, bond = env(beta)
    count = 0

    @partial(jax.jit, static_argnums=(3, 4))
    def objective(c, e, bulk, bond, k):
        nonlocal count
        count += 1  # a Python side effect: trace time only
        new_c, _ = ctmrg.unrolled(c, e, bulk, bond, k=k)
        dagger = tenet.adjoint(new_c)
        # the four-corner ring of `log_kappa`; `norm(new_c)` would be the constant 1,
        # since every move renormalizes, and a constant has nothing to differentiate
        return ctmrg.scalar(tenet.einsum("ab,ac,dc,eb->de", new_c, dagger, new_c, dagger))

    grad = jax.grad(objective, argnums=2)
    a = grad(c, e, ctmrg.ising_bulk(beta), bond, K)
    b = grad(c, e, ctmrg.ising_bulk(beta + 0.01), bond, K)
    assert count == 1  # different block values, same structure, one trace
    assert any(
        not np.allclose(np.asarray(x), np.asarray(y))
        for x, y in zip(a.blocks, b.blocks, strict=True)
    )

    smaller = ctmrg.converge(ctmrg.ising_bulk(beta), chi=8)[2]
    assert smaller != bond
    grad(c, e, ctmrg.ising_bulk(beta), bond, K)
    assert count == 1  # the same frozen bond does not retrace ...
    grad(c, e, ctmrg.ising_bulk(beta), smaller, K)
    assert count == 2  # ... a different one does: the GradedSpace is part of the key


def test_svd_truncated_is_refused_under_jit():
    """The boundary in this example is structural, not a convention.

    Two refusals, one per reason. ``move(chi=...)`` decides a bond space from the singular
    *values*, which is ``tenet.StructureChangingError`` by construction (#64). ``converge``
    additionally *reads* the corner spectrum to decide when to stop, so it cannot be traced
    even before it gets there -- a data-dependent loop exit is not a tracing edge case, it
    is the thing the outside/inside split exists to keep outside.
    """
    bulk = ctmrg.ising_bulk(0.4)
    c, e = ctmrg.init_env(bulk)
    with pytest.raises(tenet.StructureChangingError, match="tenet.linalg.svd"):
        jax.jit(partial(ctmrg.move, chi=4))(c, e, bulk)
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(partial(ctmrg.converge, chi=4, max_sweeps=1))(bulk)


@pytest.mark.parametrize(("beta", "chi"), [(0.44, CHI), (0.6, CHI)])
def test_the_retired_nan_criterion(beta, chi):
    """#102's ``NaN``-without-``tenet.ad`` criterion, **retired** — with the reason.

    The criterion expected ``NaN`` from JAX's stock SVD VJP near ``beta_c`` and a finite
    gradient from ``tenet.ad``. #103 found it unreproducible on the *ungraded* corner,
    whose spectrum has no exactly degenerate singular values, and deferred it to the
    bosonic Z2 grading. The grading is here, it *does* produce exact degeneracies — the
    smallest cross-sector splitting at ``beta=0.6, chi=16`` is ``3.4e-20`` relative — and
    the ``NaN`` still does not appear. That is not a near miss; it is structural, and it
    is why the criterion is retired rather than deferred a third time:

    **every exact degeneracy in the Z2 Ising corner is *across* parity sectors**, because
    the two partners differ by the global spin flip, which *is* the Z2 charge; and a graded
    factorization runs one SVD **per coupled sector**, exactly as ``tenet.ad`` broadens per
    coupled sector (#76). So the blocks separate the partners before any VJP sees them, and
    every within-sector gap stays healthy — measured ``6.6e-3`` at ``beta=0.44, chi=16``
    (``5.1e-2`` at ``chi=8``) and ``3.6e-10`` at ``beta=0.6, chi=16``, pinned by
    :func:`test_no_exact_within_sector_degeneracy_anywhere`. **Grading is what removes the
    ``NaN``, not what creates it.**

    Where the ``NaN`` actually lives is the *ungraded* ordered-phase run, whose single SVD
    sees the cross-sector doublet as one degenerate pair: measured min relative gap
    ``2.8e-17`` at ``beta=0.6, chi=16``. That configuration is precisely the one #102
    forbade for YASTN's spurious-symmetry-breaking reason, so it is measured and reported
    in the PR body, never asserted on.

    What is pinned here: the gradient is finite **with and without** ``tenet.ad.install()``
    on both sides of ``beta_c``, and the two agree to ``1e-9`` (measured: to the last bit,
    ``2.2e-16``), so the broadening does not perturb an answer whose per-sector gaps are
    healthy.
    """
    if beta > ctmrg.BETA_C:
        c, e, bond = ordered_env(beta, chi)[:3]
    else:
        c, e, bond = converged(beta, chi, 1e-10, 60)[:3]

    broadened = float(jax.grad(ctmrg.beta_free_energy)(beta, (c, e, bond), K))
    assert np.isfinite(broadened)
    tenet.ad.uninstall()
    try:
        stock = float(jax.grad(ctmrg.beta_free_energy)(beta, (c, e, bond), K))
    finally:
        tenet.ad.install()
    assert np.isfinite(stock)
    assert stock == pytest.approx(broadened, rel=1e-9)


# --- 4b. what the Z2 grading buys (#104) -------------------------------------------


@pytest.mark.parametrize("beta", [0.3, 0.44, 0.6])
def test_the_graded_bulk_is_the_same_numbers(beta):
    """The regrade changed the legs and no arithmetic, asserted at ``0.0``.

    ``W = [[sqrt cosh b, sqrt sinh b], [sqrt cosh b, -sqrt sinh b]]`` *is* the parity
    change of basis, so summing over ``s`` already annihilated every entry with an odd
    number of odd legs: eight of sixteen are exactly zero in the file we always shipped.
    The example was not missing a symmetry, it was declining to declare one.
    """
    dense = ising_block(beta)
    np.testing.assert_array_equal(np.asarray(ctmrg.ising_bulk(beta).to_dense()), dense)
    odd = [idx for idx in np.ndindex(dense.shape) if sum(idx) % 2]
    assert len(odd) == 8
    assert all(dense[idx] == 0.0 for idx in odd)


@pytest.mark.parametrize("beta", [0.3, 0.44, 0.6])
def test_the_bulk_is_z2_symmetric_by_from_dense_accepting_it(beta):
    """The example builds with ``atol=math.inf`` because ``beta`` is a traced scalar and
    the symmetry check is a concrete-value question (#82). The check is not lost, it is
    moved here: untraced, at the **default** relative ``atol``, ``from_dense`` accepts the
    array — which is the library certifying the grading rather than a comment claiming it.
    """
    dense = ising_block(beta)
    t = SymmetricTensor.from_dense(dense, ctmrg.ising_bulk(beta).legs)
    np.testing.assert_allclose(np.asarray(t.to_dense()), dense, atol=1e-14)


@pytest.mark.parametrize("beta", [0.3, 0.44, 0.6])
def test_zero_magnetization_is_structural_not_numerical(beta):
    """``<s> = 0`` proved by a refusal, YASTN's "zero magnetization by symmetry".

    The spin-insertion impurity ``sum_s s * prod_i W[s,i]`` is nonzero exactly when the
    number of *odd* legs is odd — it is a Z2-**odd** tensor, and no invariant
    ``SymmetricTensor`` can hold one. ``from_dense`` therefore refuses it with a residual
    naming an offending sector tuple, and *that refusal is the statement*: exact and
    structural, where the ungraded run could only offer a small float.

    ponytail: measuring a genuine ``<s>`` (rather than proving it zero) wants a dummy leg
    in the odd sector; nothing here needs one.
    """
    c, s = math.sqrt(math.cosh(beta)), math.sqrt(math.sinh(beta))
    w = np.array([[c, s], [c, -s]])
    impurity = np.einsum("s,sl,su,sr,sd->lurd", np.array([1.0, -1.0]), w, w, w, w)
    assert np.abs(impurity).max() > 0.1  # the impurity is not the zero array
    with pytest.raises(ValueError) as excinfo:
        SymmetricTensor.from_dense(impurity, ctmrg.ising_bulk(beta).legs)
    message = str(excinfo.value)
    assert "not symmetric" in message
    assert "Z2Sector(parity=1)" in message  # the odd sector is named


@pytest.mark.parametrize("beta", [0.5, 0.6])
def test_ordered_phase_spectrum_is_an_exact_cross_sector_doublet(beta):
    """Spontaneous-symmetry-breaking doubling, and it is cross-sector by construction: the
    two partners differ by the global spin flip, which is the Z2 charge.

    Measured maximum relative deviation over the pairing: ``1.0e-13`` at ``beta=0.5`` and
    ``1.9e-15`` at ``beta=0.6``, both at ``chi=16``. (The issue's standalone probe — a
    different CTMRG, ``eigh`` rather than ``svd`` — reached ``1.7e-17`` and ``2.0e-20``;
    its last digits are not predictions of this one's, so the assertion is at ``1e-12``,
    four orders below anything the disordered phase produces and thirteen below the
    ``2.5e-3`` a phase-blind test would accept.)
    """
    assert beta > ctmrg.BETA_C
    halves = spectrum_by_sector(ordered_env(beta)[0])
    assert len(halves[0]) == len(halves[1])
    top = halves[0][0]
    deviations = [abs(x - y) / top for x, y in zip(halves[0], halves[1], strict=True)]
    assert max(deviations) < 1e-12, deviations


@pytest.mark.parametrize(("beta", "chi"), [(0.3, 8), (0.44, 8), (0.44, CHI)])
def test_disordered_phase_has_no_such_pairing(beta, chi):
    """The other half of the criterion: it distinguishes the phases rather than merely
    passing. Below ``beta_c`` the two sectors' spectra are unrelated — the *closest*
    cross-sector approach is ``2.6e-4`` at ``beta=0.3, chi=8``, fourteen orders above the
    ordered phase — and the full spectrum has no degeneracy at all.
    """
    assert beta < ctmrg.BETA_C
    halves = spectrum_by_sector(converged(beta, chi, 1e-10, 200)[0])
    top = max(halves[0][0], halves[1][0])
    n = min(len(halves[0]), len(halves[1]))
    closest = min(abs(halves[0][i] - halves[1][i]) / top for i in range(n))
    assert closest > 1e-5, closest
    every = sorted(halves[0] + halves[1], reverse=True)
    assert min(abs(x - y) / top for x, y in zip(every, every[1:], strict=False)) > 1e-5


@pytest.mark.parametrize(
    ("beta", "chi", "tol"),
    [(0.3, 8, 1e-10), (0.44, 8, 1e-10), (0.44, CHI, 1e-10), (0.4, CHI, 1e-10)],
)
def test_no_exact_within_sector_degeneracy_anywhere(beta, chi, tol):
    """The fact that decides the ``NaN`` question, since ``tenet.ad`` broadens per coupled
    sector: nothing is ever exactly degenerate *within* a sector. Measured smallest
    within-sector relative gaps — ``1.1e-3`` (0.3/8), ``5.1e-2`` (0.44/8), ``6.6e-3``
    (0.44/16), ``1.6e-7`` (0.4/16), and in the ordered phase ``5.1e-7`` (0.5/16) and
    ``3.6e-10`` (0.6/16), the closest approach anywhere measured."""
    halves = spectrum_by_sector(converged(beta, chi, tol, 200)[0])
    top = max(halves[0][0], halves[1][0])
    gaps = [
        abs(x - y) / top
        for values in halves.values()
        for x, y in zip(values, values[1:], strict=False)
    ]
    assert min(gaps) > 1e-12, min(gaps)


@pytest.mark.parametrize("beta", [0.5, 0.6])
def test_no_exact_within_sector_degeneracy_in_the_ordered_phase(beta):
    halves = spectrum_by_sector(ordered_env(beta)[0])
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
    the doublet partners are exactly equal and therefore adjacent in that ranking — so any
    even ``chi`` keeps both. This is the bosonic analogue of the SU(2) multiplet-splitting
    hazard ``ctmrg.CHI_IPEPS`` already documents (Francuz, Schuch, Vanhecke, PRR 7, 013237
    (2025), Appendix C: *"be careful not to split multiplets when converging the original
    CTM"*) — cheaper here, because the multiplet size is 2 and known. Asserted, not assumed.
    """
    assert chi % 2 == 0
    halves = spectrum_by_sector(ordered_env(0.6, chi)[0])
    assert len(halves[0]) == len(halves[1]) == chi // 2  # equal degeneracy in both sectors
    top = halves[0][0]
    assert max(abs(x - y) / top for x, y in zip(halves[0], halves[1], strict=True)) < 1e-12


# --- 5. the U(1) / SU(2) iPEPS half ------------------------------------------------


@pytest.fixture(params=PROVIDERS)
def provider(request):
    return request.param


def ipeps_grad(provider):
    """One gradient per provider, reused. The backward pass through a double-layer bulk is
    the single most expensive thing in this module; three tests want the same one."""
    key = f"grad-{provider}"
    if key not in _ENVS:
        a, h, e = ipeps_env(provider)
        _ENVS[key] = jax.value_and_grad(ctmrg.energy)(a, h, e, K_IPEPS)
    return _ENVS[key]


def test_ipeps_energy_is_real(provider):
    value, _ = ipeps_grad(provider)
    assert abs(complex(value).imag) < 1e-12


def test_ipeps_grad_matches_central_differences_on_one_block(provider):
    a, h, e = ipeps_env(provider)
    grads = ipeps_grad(provider)[1]
    blk, delta = 0, 1e-5

    def shifted(idx, d):
        blocks = list(a.blocks)
        blocks[blk] = blocks[blk].at[idx].add(d)
        return float(ctmrg.energy(SymmetricTensor(a.structure, tuple(blocks)), h, e, K_IPEPS))

    block = a.blocks[blk]
    want = np.zeros(block.shape)
    for idx in np.ndindex(block.shape):
        want[idx] = (shifted(idx, delta) - shifted(idx, -delta)) / (2 * delta)
    np.testing.assert_allclose(np.asarray(grads.blocks[blk]), want, rtol=1e-5, atol=1e-8)


def test_ipeps_sgd_decreases_the_energy(provider):
    """Three plain SGD steps, ``vmc_mps``-style: the first one reuses the cached gradient."""
    a, h, e = ipeps_env(provider)
    value, grad = ipeps_grad(provider)
    trace = [float(value)]
    a = jax.tree.map(lambda p, g: p - 0.05 * g, a, grad)
    for _ in range(2):
        a, value = ctmrg.step(a, h, e, lr=0.05, k=K_IPEPS)
        trace.append(float(value))
    assert all(b < x for x, b in zip(trace, trace[1:], strict=False)), trace


def test_structures_survive_the_traced_region(provider):
    """#69's trust boundary, re-run: nothing in the differentiated region rewrites a
    ``TensorStructure``, so the originals are still ``is``-identical afterwards."""
    a, h, e = ipeps_env(provider)
    c, edge, bond = e
    before = (a.structure, c.structure, edge.structure)
    ipeps_grad(provider)  # the traced region has run by now
    bulk = tenet.trace(ctmrg.ipeps_bulk_open(a), (0, 1))
    out_c, out_e = ctmrg.unrolled(c, edge, bulk, bond, k=K_IPEPS)
    assert all(
        x is y for x, y in zip((a.structure, c.structure, edge.structure), before, strict=True)
    )
    # the traced region is structure-preserving: the moves come back on the frozen bond
    assert out_c.structure == c.structure and out_e.structure == edge.structure
    for t in (a, c, edge, out_c, out_e):
        SymmetricTensor(t.structure, t.blocks)  # the trust boundary, re-run


def test_main_runs_both_halves():
    """The standalone entry point, at the module's own chi and k so that every plan it
    needs is already cached."""
    ctmrg.main(chi_ising=CHI, k=K, steps=1)
