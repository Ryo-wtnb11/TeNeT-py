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

**Since #114 the environment machinery is ``tenet.network``** -- ``Absorb``, both absorbers
(under their model-free names ``single_layer``/``double_layer``), ``move``, ``ctmrg``
and ``ctmrg_unrolled`` -- so this module resolves those names there and the example's own names
are the physics: the bulk tensor, the C4v ansatz constraint and the observables. Nothing
else changed: every number below is bit-identical to the pre-promotion run.

x64 is enabled process-globally in ``tests/conftest.py``; every tolerance here depends on it.

**Runtime: the budget is 100 s, and #102's 30 s is withdrawn with arithmetic** (#105, #107).
#102 asked for 30 s for this module against
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

The iPEPS half is the rest, and since #107 the old diagnosis in this paragraph is
**retired rather than restated**, on four points:

1. **The rank-10 double layer no longer exists.** ``ipeps_bulk``, ``ipeps_bulk_open`` and
   their four fused ``(D_ket, D_bra)`` bonds are deleted; the environment edge carries the
   ket bond and its conjugate as two rank-4 legs and the site enters as a ket and then a
   bra (froSTspin ``ctm_contract.py``:42,53, YASTN ``_env_contractions.py``:221-224). So
   the 841-block intermediate that every earlier version of this paragraph was about is
   not slow any more, it is *absent*, and #105's 4 808 ``repartition_plan`` terms with it.
2. **The peak is now rank 6 / 51 blocks** at SU(2) chi=6, against rank 10 / 841 --
   froSTspin's ``2*a*d*chi**2*D**4`` instead of ``d**2 D**8``. Per enlarged corner: 4 852 ->
   1 839 ``ar.do`` calls and 95 -> 52 distinct program keys at SU(2); for the open corner
   the physics energy needs, 18 546 -> 2 315 calls. The edge, measured on its own rather
   than assumed to follow the corner, goes the other way on calls and the right way on
   keys: 906 -> 1 836 ``ar.do`` calls but 133 -> 52 program keys at SU(2) (599 -> 1 026
   and 78 -> 50 at U(1)), because it no longer absorbs a pre-built double layer. The
   forward ``energy`` is 0.48 s -> 0.36 s at SU(2). The *warm* gradient is not faster --
   4.84 s -> 5.42 s at SU(2), 1.22 s -> 2.01 s at U(1) -- because unfusing trades a few
   large blocks for many small ones and eager per-block dispatch is charged per block.
   That is the same mechanism as (3) seen from the other end, and it is the honest half
   of the trade.
3. **What remains is one-off XLA compiles**, and that is where the redesign pays: the
   *cold* SU(2) ``value_and_grad(energy)`` is 37.5 s -> 23.8 s and
   ``test_ipeps_energy_is_real[su2]`` 36.8 s -> 27.0 s, because the number of distinct
   ``(primitive, aval, params)`` programs falls with the number of distinct block shapes.
   The U(1) cold case does not move at all (22.8 s -> 22.8 s): U(1) has no multiplets, so
   there was no fusion blow-up to remove, and its warm regression is not bought back. #105
   measured the floor itself -- ~2 640 compiles at ~9 ms is ~60 s -- and roughly 47 s of
   this module is still exactly that. #74's block bucketing is its fix; nothing in
   ``examples/`` substitutes for it.
4. **#102's 30 s is still not reached and #74 is the only thing that reaches it.** A
   contraction order changes how many primitives run and how many distinct *shapes* they
   see; it cannot make a distinct shape stop implying a distinct program. **The budget is
   100 s** (95.8-98.1 s measured over two runs, plus load headroom), from 110 s: the whole
   module is where it was -- 97.8 s for 48 tests before #107, 95.8-98.1 s for 52 after,
   the four new ones being the ~4.2 s migration criteria -- because the ~10 s the SU(2)
   compile path gives back is spent again on warm dispatch. #107's win is a *structural*
   one (no rank-10, a peak that scales as ``chi**2 D**4``) plus a cold-path one; it was
   never going to be a wall-clock one on a module whose floor is compilation. Deleting the
   SU(2) parametrization -- the only alternative -- would delete the reason the iPEPS half
   exists, and ``jax.jit`` stays rejected (#105: compiling the SU(2) gradient costs more
   than running the three eager ones this module needs, and it would hide the point that
   ``ctmrg`` sits outside the trace and ``ctmrg_unrolled`` inside it).
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
    """The :class:`CTMRG_out`, memoized on the full set of knobs that decides it."""
    key = (beta, chi, tol, max_sweeps)
    if key not in _ENVS:
        bulk = ctmrg.ising_bulk(beta)
        _ENVS[key] = tenet.network.ctmrg(
            *tenet.network.single_layer_ctm(bulk), chi=chi, tol=tol, max_sweeps=max_sweeps
        )
    return _ENVS[key]


def env(beta: float, chi: int = CHI, tol: float = 1e-10, max_sweeps: int = 100):
    return converged(beta, chi, tol, max_sweeps).env


def ordered_env(beta: float, chi: int = CHI):
    return converged(beta, chi, TOL_ORDERED, SWEEPS_ORDERED)


def spectrum_by_sector(c: SymmetricTensor) -> dict[int, list[float]]:
    """The corner spectrum split by Z2 charge, each half descending.

    ``tenet.network.spectrum`` deliberately throws the sector labels away; every ordered-phase
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
        env = tenet.network.ctmrg(*tenet.network.double_layer_ctm(ctmrg.c4v(a)), chi=chi)
        _ENVS[provider] = (a, h, env.env)
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
    got = float(ctmrg.beta_free_energy(beta, ordered_env(beta).env, k=K))
    assert got == pytest.approx(ctmrg.onsager(beta), rel=1e-6)


def test_convergence_is_monotone_and_terminating():
    """Asserted, not assumed: the corner-spectrum change reaches ``tol`` inside
    ``max_sweeps``, decreases over the run, and takes a pinned number of sweeps."""
    out = converged(0.4)
    history = out.history
    assert out.converged  # the field, not a re-derivation: terminated on the tolerance
    assert out.sweeps < 100
    assert 40 <= out.sweeps <= 120  # 72 as measured under the Z2 grading; the range is the pin
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
    got = float(jax.grad(ctmrg.beta_free_energy)(beta, ordered_env(beta).env, K))
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
    loose = tenet.network.ctmrg(
        *tenet.network.single_layer_ctm(ctmrg.ising_bulk(beta)), chi=CHI, tol=1e-6
    ).env
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
        # the absorber is built inside, from the traced bulk: it is two closures, so the
        # gradient w.r.t. `bulk` still flows -- the closure is captured under the trace
        new_c, _ = tenet.network.ctmrg_unrolled(c, e, tenet.network.single_layer(bulk), bond, k=k)
        dagger = tenet.adjoint(new_c)
        # the four-corner ring of `log_kappa`; `norm(new_c)` would be the constant 1,
        # since every move renormalizes, and a constant has nothing to differentiate
        return tenet.network.scalar(tenet.einsum("ab,ac,dc,eb->de", new_c, dagger, new_c, dagger))

    grad = jax.grad(objective, argnums=2)
    a = grad(c, e, ctmrg.ising_bulk(beta), bond, K)
    b = grad(c, e, ctmrg.ising_bulk(beta + 0.01), bond, K)
    assert count == 1  # different block values, same structure, one trace
    assert any(
        not np.allclose(np.asarray(x), np.asarray(y))
        for x, y in zip(a.blocks, b.blocks, strict=True)
    )

    smaller_env = tenet.network.ctmrg(
        *tenet.network.single_layer_ctm(ctmrg.ising_bulk(beta)), chi=8
    ).env
    smaller = smaller_env.bond
    assert smaller != bond
    grad(c, e, ctmrg.ising_bulk(beta), bond, K)
    assert count == 1  # the same frozen bond does not retrace ...
    grad(c, e, ctmrg.ising_bulk(beta), smaller, K)
    assert count == 2  # ... a different one does: the GradedSpace is part of the key


def test_svd_truncated_is_refused_under_jit():
    """The boundary in this example is structural, not a convention.

    Two refusals, one per reason. ``move(chi=...)`` decides a bond space from the singular
    *values*, which is ``tenet.StructureChangingError`` by construction (#64). ``ctmrg``
    additionally *reads* the corner spectrum to decide when to stop, so it cannot be traced
    even before it gets there -- a data-dependent loop exit is not a tracing edge case, it
    is the thing the outside/inside split exists to keep outside.
    """
    absorb, c, e = tenet.network.single_layer_ctm(ctmrg.ising_bulk(0.4))
    with pytest.raises(tenet.StructureChangingError, match="tenet.linalg.svd"):
        jax.jit(partial(tenet.network.move, chi=4), static_argnums=(2,))(c, e, absorb)
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(partial(tenet.network.ctmrg, chi=4, max_sweeps=1), static_argnums=(0,))(
            absorb, c, e
        )


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
        c, e, bond = ordered_env(beta, chi).env
    else:
        c, e, bond = converged(beta, chi, 1e-10, 60).env

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
    halves = spectrum_by_sector(ordered_env(beta).env.c)
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
    halves = spectrum_by_sector(converged(beta, chi, 1e-10, 200).env.c)
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
    halves = spectrum_by_sector(converged(beta, chi, tol, 200).env.c)
    top = max(halves[0][0], halves[1][0])
    gaps = [
        abs(x - y) / top
        for values in halves.values()
        for x, y in zip(values, values[1:], strict=False)
    ]
    assert min(gaps) > 1e-12, min(gaps)


@pytest.mark.parametrize("beta", [0.5, 0.6])
def test_no_exact_within_sector_degeneracy_in_the_ordered_phase(beta):
    halves = spectrum_by_sector(ordered_env(beta).env.c)
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
    halves = spectrum_by_sector(ordered_env(0.6, chi).env.c)
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


# The energies the *fused double-layer* formulation reported, before #107 replaced it with
# the env->ket->bra absorption order. Ten significant figures, which is what the two
# formulations being the same bilinear form entitles us to; measured agreement is 5.3e-15
# (su2) and 8.9e-16 (u1) relative, i.e. the last bits.
ENERGY_BASELINE = {"su2": -0.038475159359, "u1": -0.310993394006}


def test_ipeps_energy_matches_the_pre_redesign_baseline(provider):
    """#107's migration criterion, second half: same environment, same energy.

    The environment is now converged by the *new* code and the energy computed through the
    open-corner route; the numbers are the ones the deleted ``ipeps_bulk``/
    ``ipeps_bulk_open`` produced. If a truncation tie ever broke differently the two would
    separate at the truncation level rather than at 1e-15 -- that has not happened at
    either provider, at ``CHI_IPEPS`` 4 and 6.
    """
    value, _ = ipeps_grad(provider)
    assert float(value) == pytest.approx(ENERGY_BASELINE[provider], rel=1e-10)


def _fuse_pair(t, i, j):
    """``ctmrg._fuse_pair``, deleted by #107 and kept here for the migration test alone."""
    order = (i, j, *(k for k in range(t.ndim) if k not in (i, j)))
    return tenet.fuse(tenet.transpose(t, order), (0, 1))


def legacy_double_layer(a):
    """``ctmrg.ipeps_bulk``, deleted by #107: the fused rank-4 double layer, built through
    the rank-10 intermediate (841 blocks at SU(2) chi=6) this redesign removed."""
    ket, bra = tenet.network.layers(ctmrg.c4v(a))
    dl = tenet.einsum("LUsRD,slurd->lLuUrRdD", bra, ket)
    dl = tenet.repartition(dl, (0, 1, 2, 3), (4, 5, 6, 7))
    for i in range(4):
        dl = _fuse_pair(dl, i, i + 1)
    return tenet.transpose(dl, (3, 2, 1, 0))


def test_the_unfused_corner_gives_the_old_projector(provider):
    """#107's migration criterion, first half — and the thing that licenses the redesign.

    The enlarged corner is *for* the projector and for nothing else, so the two
    formulations agree iff their truncated spectra do. On one converged environment, the
    rank-6 corner (env -> ket -> bra, two ``D`` bonds per edge) is compared against the
    rank-4 one (one fused ``(D_ket, D_bra)`` bond, one double layer) by fusing the very
    same edge back down: same tensor, seen through a unitary refusal-to-fuse.

    Measured maximum absolute deviation over the spectrum: ``1.7e-16`` at SU(2) chi=6 and
    ``5.6e-17`` at U(1) chi=4, with equal ``tenet.norm`` to the last bit.

    Fusing the converged edge is also where the old convention's hidden M4 dependency
    shows: ``tenet.fuse`` wants its pair to lead the side, and ``tenet.unfuse`` -- the
    direction the old code would have needed to get here -- refuses a non-leading leg
    outright ("splitting a non-leading leg needs an F-move, which is Milestone 4"). The
    redesigned example never fuses, so it never asks.
    """
    a, _, (c, e, _) = ipeps_env(provider)
    chi = ctmrg.CHI_IPEPS[provider]

    fused = tenet.transpose(_fuse_pair(e, 2, 3), (1, 2, 0))  # (X IN, X OUT, V IN)
    old = tenet.einsum("ab,ace,fbg,gehi->chfi", c, fused, fused, legacy_double_layer(a))
    new = tenet.network.double_layer(*tenet.network.layers(ctmrg.c4v(a))).corner(c, e)
    assert (old.ndim, new.ndim) == (4, 6)

    old_s = tenet.network.spectrum(
        tenet.linalg.svd_truncated(old, ((0, 1), (2, 3)), max_bond=chi)[1]
    )
    new_s = tenet.network.spectrum(
        tenet.linalg.svd_truncated(new, ((0, 1, 2), (3, 4, 5)), max_bond=chi)[1]
    )
    np.testing.assert_allclose(new_s, old_s, atol=1e-14)
    assert float(tenet.norm(new)) == pytest.approx(float(tenet.norm(old)), abs=1e-15)


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
    absorb = tenet.network.double_layer(*tenet.network.layers(ctmrg.c4v(a)))
    out_c, out_e = tenet.network.ctmrg_unrolled(c, edge, absorb, bond, k=K_IPEPS)
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
