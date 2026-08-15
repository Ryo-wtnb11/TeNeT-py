"""Issue #102 — differentiable CTMRG: the first end-to-end algorithm, against an oracle.

This is the first test in the repository that judges a *physics* result against a closed
form rather than against a self-consistency check: ``examples/ctmrg.py`` converges a CTMRG
environment for the classical 2D Ising model and its free energy per site is compared with
Onsager's, which is itself pinned here by two independent closed forms before it is used to
judge anything. It then differentiates ``k`` unrolled fixed-structure sweeps and checks the
gradient against the *internal energy* of the same oracle.

Like ``tests/integration/test_vmc.py`` (#69) it adds nothing to ``src/tenet`` and nothing to
the example: it imports ``examples/ctmrg.py`` and runs it, so the example cannot rot. The
only code of its own is the second Onsager form and the tolerances.

x64 is enabled process-globally in ``tests/conftest.py``; every tolerance here depends on it.

**Runtime, measured and over budget.** #102 asked for 30 s for this module against
``test_vmc.py``'s 2.92 s. The Ising half -- every test with an oracle behind it -- is 5 s.
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

import pathlib
import sys
from functools import partial

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

# Converging an environment is the expensive part and nothing here mutates one, so each is
# built once per module. ponytail: a plain dict, not a fixture per beta -- the key is a
# float and the values are immutable.
_ENVS: dict = {}


def env(beta: float, chi: int = CHI):
    if (beta, chi) not in _ENVS:
        _ENVS[beta, chi] = ctmrg.converge(ctmrg.ising_bulk(beta), chi=chi)
    return _ENVS[beta, chi][:3]


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
    where degenerate singular values live; ``test_near_criticality`` is the one test that
    deliberately runs both ways.
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


def test_convergence_is_monotone_and_terminating():
    """Asserted, not assumed: the corner-spectrum change reaches ``tol`` inside
    ``max_sweeps``, decreases over the run, and takes a pinned number of sweeps."""
    _, _, _, history = _ENVS.setdefault((0.4, CHI), ctmrg.converge(ctmrg.ising_bulk(0.4), chi=CHI))
    assert history[-1] < 1e-10
    assert len(history) < 100  # terminated on the tolerance, not on max_sweeps
    assert 40 <= len(history) <= 120  # 82 as measured; the range is the pin
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


def test_near_criticality_gap_and_the_broadened_gradient():
    """``beta = 0.44``, just below ``beta_c``, at ``chi = 8`` -- and a finding.

    The criterion this test was written for expected ``NaN`` from JAX's stock SVD VJP and a
    finite gradient from ``tenet.ad``. What the measurement says is that an *ungraded* Ising
    corner spectrum has no exactly degenerate singular values to produce that ``NaN``: the
    smallest relative gap here is ``1.6e-4``, and even deep in the ordered phase it only
    reaches ``2e-13``. ``1/(sigma_i - sigma_j)`` is then enormous but finite, and the two
    VJPs agree. The exactly-degenerate multiplets #76 was built for are what a *bosonic Z2*
    grading of this bulk tensor would create -- the follow-up this example already cites --
    so what is pinned here is the gap, the finiteness with the broadening installed, and the
    fact that the broadening does not perturb the answer where the gap is healthy.
    """
    beta, chi = 0.44, 8
    c, e, bond, _ = _ENVS.setdefault(
        (beta, chi), ctmrg.converge(ctmrg.ising_bulk(beta), chi=chi, max_sweeps=60)
    )
    kept = ctmrg.spectrum(c)
    gaps = [abs(x - y) / kept[0] for x, y in zip(kept, kept[1:], strict=False)]
    assert min(gaps) < 1e-2  # near-degenerate, but not degenerate

    broadened = float(jax.grad(ctmrg.beta_free_energy)(beta, (c, e, bond), K))
    assert np.isfinite(broadened)
    tenet.ad.uninstall()
    try:
        stock = float(jax.grad(ctmrg.beta_free_energy)(beta, (c, e, bond), K))
    finally:
        tenet.ad.install()
    assert stock == pytest.approx(broadened, rel=1e-9)


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
