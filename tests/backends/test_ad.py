"""Tests for ``tenet.ad`` — Lorentzian-broadened ``svd``/``eigh`` VJPs (issue #76).

Every degeneracy here is *constructed*, not hoped for: the tensors are built by
replacing each coupled-sector matrix with an explicit one (:func:`refill`), so
"two identical singular values inside one coupled sector" is a fact of the fixture
rather than a property of a random seed. The gradients are compared on
**gauge-invariant** objectives — reconstruction, the singular values, ``V W V†`` —
except where the point is precisely that a gauge-*dependent* one is meaningless.

x64 is enabled process-globally in ``tests/conftest.py``; the finite-difference
tolerances here depend on it, as in ``tests/backends/test_pytree.py``.
"""

import pathlib
import subprocess
import sys
import textwrap

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import from_matrices, to_matrices
from tenet.symmetry import SU2, U1, FZ2Sector, SU2Sector, U1Sector, fZ2

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import tenet.ad  # noqa: E402
import tenet.pytree  # noqa: E402, F401  # registration is the import's side effect

# Square maps: one OUT leg and its IN partner, so every coupled-sector matrix is
# m_c x m_c and both svd and eigh accept the tensor. Degeneracies of 3 and 2 give a
# sector big enough for *two* zero singular values and one that is not.
Q = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(1): 2})
V = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2})  # qdim weights 1 and 3
F = GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 2})
SQUARE = {
    "u1": (Leg(Q, OUT), Leg(Q, IN)),
    "su2": (Leg(V, OUT), Leg(V, IN)),
    "fz2": (Leg(F, OUT), Leg(F, IN)),
}
PROVIDERS = list(SQUARE)

# Rectangular in both directions: (3, 2) in the c = 0 sector and (2, 3) in c = 1, so
# both ``rows > k`` and ``cols > k`` complement terms are exercised.
QA = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(1): 2})
QB = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})
RECT = (Leg(QA, OUT), Leg(QB, IN))

# A bigger structure for the jit trace-count test: same provider, different degeneracy.
Q_BIG = GradedSpace.new(U1, {U1Sector(0): 4, U1Sector(1): 2})
SQUARE_BIG = (Leg(Q_BIG, OUT), Leg(Q_BIG, IN))


@pytest.fixture(autouse=True)
def stock_dispatch():
    """Every test starts from autoray's stock bindings and leaves them behind."""
    tenet.ad.uninstall()
    yield
    tenet.ad.uninstall()


# --- fixtures -------------------------------------------------------------------


def jt(legs, seed=0, dtype=None):
    """A random JAX-backed tensor, optionally complex."""
    t = SymmetricTensor.random(legs, seed=seed).to_backend("jax")
    if dtype is None:
        return t
    other = SymmetricTensor.random(legs, seed=seed + 1000).to_backend("jax")
    return SymmetricTensor(
        t.structure,
        tuple((a + 1j * b).astype(dtype) for a, b in zip(t.blocks, other.blocks, strict=True)),
    )


def refill(t, fill):
    """``t`` with every coupled-sector matrix replaced by ``fill(rows, cols)``."""
    mats = {c: jnp.asarray(fill(*m.shape)) for c, m in to_matrices(t).items()}
    return from_matrices(t.structure, mats)


def hermitian(t):
    """``t`` with every coupled-sector matrix Hermitized. Input for ``eigh``."""
    return from_matrices(
        t.structure, {c: (m + jnp.conj(m).T) / 2 for c, m in to_matrices(t).items()}
    )


def degenerate(legs, seed=0):
    """Every singular value / eigenvalue equal to 1 in every sector."""
    return refill(jt(legs, seed), lambda r, c: np.eye(r, c))


def diagonal_fill(values):
    """``fill(rows, cols)`` planting ``values(k)`` on the diagonal, zeros elsewhere."""

    def fill(r, c):
        k = min(r, c)
        m = np.zeros((r, c))
        m[:k, :k] = np.diag(values(k))
        return m

    return fill


def with_zeros(legs, n_zero, seed=0):
    """Distinct nonzero singular values, then ``n_zero`` **exact** zeros, per sector.

    Exact is the operative word: an all-ones block has a numerically tiny rather than
    an exactly zero tail, and stock JAX survives that. ``_lower``'s ``min(rows, cols)``
    bond rule is what puts genuine zeros in ``S``.
    """

    def values(k):
        d = np.arange(1.0, k + 1)
        d[k - n_zero :] = 0.0
        return d

    return refill(jt(legs, seed), diagonal_fill(values))


def near_degenerate(legs, seed=0, split=1e-6):
    """Singular values ``1, 1 + split, ...`` — a gap well inside the broadening width.

    ``eps`` is invisible at *exact* degeneracy (``safe_inverse(0) == 0`` whatever ``eps``
    is), so the neighbourhood is where the knob actually bites and where the gauge
    ambiguity actually shows up as an ``eps``-dependent number.
    """
    return refill(jt(legs, seed), diagonal_fill(lambda k: 1 + split * np.arange(k)))


def has_nan(t):
    return any(bool(np.isnan(np.asarray(b)).any()) for b in t.blocks)


def blocks(t):
    return [np.asarray(b) for b in t.blocks]


# --- objectives (gauge-invariant unless the name says otherwise) -----------------


def recon(t):
    u, s, vh = tenet.linalg.svd(t)
    return tenet.norm(u @ s @ vh)


def spectrum(t):
    return tenet.norm(tenet.linalg.svd(t)[1])


def eig_recon(t):
    w, v = tenet.linalg.eigh(t)
    return tenet.norm(v @ w @ tenet.adjoint(v))


def qr_recon(t):
    q, r = tenet.linalg.qr(t)
    return tenet.norm(q @ r)


def lq_recon(t):
    lo, q = tenet.linalg.lq(t)
    return tenet.norm(lo @ q)


def polar_recon(t):
    w, p = tenet.linalg.polar(t)
    return tenet.norm(w @ p)


def gauge_dependent(t):
    """Sum of the first column of every ``U`` — **not** gauge-invariant.

    Inside a degenerate multiplet the singular vectors are only defined up to a
    unitary, so this quantity has no derivative at all; the broadened rule returns an
    ``eps``-suppressed arbitrary value for it. That is the documented precondition of
    :mod:`tenet.ad`, pinned below rather than merely written down.
    """
    u = tenet.linalg.svd(t)[0]
    return sum(jnp.real(jnp.sum(m[:, 0])) for m in to_matrices(u).values())


# --- the module itself ----------------------------------------------------------


def test_install_and_uninstall_are_idempotent():
    tenet.ad.install()
    tenet.ad.install(epsilon=1e-10)
    assert not has_nan(jax.grad(recon)(degenerate(SQUARE["u1"])))
    tenet.ad.uninstall()
    tenet.ad.uninstall()
    assert has_nan(jax.grad(recon)(degenerate(SQUARE["u1"])))


def test_bare_import_changes_no_dispatch():
    """Installation is an explicit call, not an import side effect — in a fresh process."""
    script = textwrap.dedent("""
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp, numpy as np, autoray as ar
        import tenet.ad                       # the import under test

        def f(x):
            u, s, vh = ar.do("linalg.svd", x, full_matrices=False)
            return jnp.sum(u @ jnp.diag(s) @ vh)

        b = jnp.eye(3)                        # every singular value 1: degenerate
        assert np.isnan(np.asarray(jax.grad(f)(b))).any(), "bare import changed dispatch"
        tenet.ad.install()
        assert not np.isnan(np.asarray(jax.grad(f)(b))).any(), "install() did nothing"
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "OK"


def test_import_error_without_jax_names_extra():
    """Same style as ``tenet.pytree``'s, verified with a meta-path blocker."""
    script = textwrap.dedent("""
        import importlib.abc, sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "jax" or name.startswith("jax."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, Blocker())
        import tenet                      # core must survive a JAX-less environment
        assert "jax" not in sys.modules
        try:
            import tenet.ad
        except ImportError as exc:
            msg = str(exc)
            assert "JAX" in msg, msg
            assert "tenet-py[jax]" in msg, msg
        else:
            raise AssertionError("tenet.ad imported without jax")
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "OK"


def test_not_exported_from_tenet_namespace():
    assert "ad" not in tenet.__all__
    assert tenet.ad.__all__ == []


# --- degeneracy: the whole point -------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_svd_degenerate_is_nan_before_and_finite_after(name):
    t = degenerate(SQUARE[name])
    assert has_nan(jax.grad(recon)(t))
    tenet.ad.install()
    g = jax.grad(recon)(t)
    assert not has_nan(g)
    assert all(np.isfinite(b).all() for b in blocks(g))


@pytest.mark.parametrize("name", PROVIDERS)
def test_eigh_degenerate_is_nan_before_and_finite_after(name):
    t = degenerate(SQUARE[name])  # every eigenvalue 1, in every sector
    assert has_nan(jax.grad(eig_recon)(t))
    tenet.ad.install()
    assert not has_nan(jax.grad(eig_recon)(t))


def test_polar_inherits_the_fix():
    """No code in ``polar`` changed: it calls ``ar.do("linalg.svd")`` and gets this."""
    t = degenerate(SQUARE["u1"])
    assert has_nan(jax.grad(polar_recon)(t))
    tenet.ad.install()
    assert not has_nan(jax.grad(polar_recon)(t))


@pytest.mark.parametrize("objective", [recon, eig_recon], ids=["svd", "eigh"])
def test_away_from_degeneracy_it_agrees_with_stock_jax(objective):
    """Stock and broadened in one process — the reason ``uninstall`` exists."""
    t = hermitian(jt(SQUARE["u1"], seed=3))
    stock = blocks(jax.grad(objective)(t))
    tenet.ad.install()
    broadened = blocks(jax.grad(objective)(t))
    for a, b in zip(stock, broadened, strict=True):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-8)


# --- finite differences ----------------------------------------------------------


def central_differences(f, t, h=1e-5):
    """df/d(every entry of every block), by central differences. As in test_pytree."""
    out = []
    for i, block in enumerate(t.blocks):
        d = np.zeros(block.shape)
        for idx in np.ndindex(block.shape):

            def shifted(delta, i=i, idx=idx):
                bs = list(t.blocks)
                bs[i] = bs[i].at[idx].add(delta)
                return float(f(SymmetricTensor(t.structure, tuple(bs))))

            d[idx] = (shifted(h) - shifted(-h)) / (2 * h)
        out.append(d)
    return out


@pytest.mark.parametrize("name", ["u1", "su2"])
@pytest.mark.parametrize("objective", [recon, spectrum], ids=["recon", "spectrum"])
def test_finite_differences_svd(name, objective):
    tenet.ad.install()
    t = jt(SQUARE[name], seed=7)  # random: a well-separated spectrum
    g = jax.grad(objective)(t)
    for got, want in zip(blocks(g), central_differences(objective, t), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_finite_differences_eigh(name):
    tenet.ad.install()
    t = hermitian(jt(SQUARE[name], seed=8))
    g = jax.grad(eig_recon)(t)
    for got, want in zip(blocks(g), central_differences(eig_recon, t), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_finite_differences_rectangular_sectors():
    """Both complement terms: the c = 0 sector is (3, 2) and c = 1 is (2, 3)."""
    tenet.ad.install()
    t = jt(RECT, seed=9)
    assert sorted(m.shape for m in to_matrices(t).values()) == [(2, 3), (3, 2)]
    g = jax.grad(recon)(t)
    for got, want in zip(blocks(g), central_differences(recon, t), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


# --- zero singular values --------------------------------------------------------


def test_one_zero_singular_value_is_finite_after_install():
    """A single exact zero, no degeneracy — and the rectangular case is *not* survivable.

    Stock JAX copes with one zero in a square sector, as the issue measured. In a
    rectangular one it does not: the ``(I - U U†) dU diag(1/sigma)`` complement divides
    by that zero, and only ``safe_inverse`` covers it. Same formula, no rank branch.
    """
    square, rect = with_zeros(SQUARE["u1"], 1, seed=10), with_zeros(RECT, 1, seed=10)
    assert not has_nan(jax.grad(recon)(square))
    assert has_nan(jax.grad(recon)(rect))
    tenet.ad.install()
    assert not has_nan(jax.grad(recon)(square))
    assert not has_nan(jax.grad(recon)(rect))


def test_two_zero_singular_values_are_nan_before_and_finite_after():
    """Two exact zeros is a degeneracy *at* zero, and one formula covers both."""
    t = with_zeros(SQUARE["u1"], 2, seed=11)
    assert has_nan(jax.grad(recon)(t))
    tenet.ad.install()
    assert not has_nan(jax.grad(recon)(t))


# --- complex ---------------------------------------------------------------------


def complex_central_differences(f, t, h=1e-5):
    """``df/dx - i df/dy`` per entry — JAX's gradient convention for real-valued ``f``.

    The conjugate one, i.e. ``jax.grad(lambda z: abs(z)**2)(1 + 2j) == 2 - 4j``.
    """
    real = central_differences(f, t, h)
    imag = []
    for i, block in enumerate(t.blocks):
        d = np.zeros(block.shape)
        for idx in np.ndindex(block.shape):

            def shifted(delta, i=i, idx=idx):
                bs = list(t.blocks)
                bs[i] = bs[i].at[idx].add(1j * delta)
                return float(f(SymmetricTensor(t.structure, tuple(bs))))

            d[idx] = (shifted(h) - shifted(-h)) / (2 * h)
        imag.append(d)
    return [a - 1j * b for a, b in zip(real, imag, strict=True)]


def test_complex_gradient_matches_finite_differences():
    tenet.ad.install()
    t = jt(SQUARE["u1"], seed=12, dtype="complex128")
    g = jax.grad(recon)(t)
    for got, want in zip(blocks(g), complex_central_differences(recon, t), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def _bwd_without_the_phase_diagonal(res, ct):
    """``_svd_bwd`` minus the ``1/(sigma_i + sigma_j)`` diagonal, i.e. tensorgrad's rule."""
    u, s, vh = res
    (da,) = tenet.ad._svd_bwd(res, ct)
    du, _, dvh = (jnp.conj(x) for x in ct)
    v = tenet.ad._dag(vh)
    dv = tenet.ad._dag(dvh)
    a = tenet.ad._antiherm(tenet.ad._dag(u) @ du) - tenet.ad._antiherm(tenet.ad._dag(v) @ dv)
    drop = jnp.diag(jnp.diag(a) * tenet.ad._safe_inverse(2 * s, tenet.ad._EPSILON))
    return (da - jnp.conj(u @ drop @ vh),)


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_the_phase_diagonal_is_exercised_for_complex_input_only(dtype):
    """Dropping the ``1/(sigma_i + sigma_j)`` diagonal must change a *complex* answer.

    It vanishes identically for real input, which is why tensorgrad — real-only — can
    drop it and why a real test could never see it.
    """
    tenet.ad.install()

    @jax.custom_vjp
    def svd_no_phase(b):
        return jnp.linalg.svd(b, full_matrices=False)

    svd_no_phase.defvjp(tenet.ad._svd_fwd, _bwd_without_the_phase_diagonal)

    b = jt(SQUARE["u1"], seed=13, dtype="complex128").blocks[0]
    if dtype == "float64":
        b = jnp.real(b)

    def objective(f, x):
        u, s, vh = f(x)
        return jnp.real(jnp.sum(u @ jnp.diag(s**2) @ vh)) + jnp.real(jnp.sum(u @ vh))

    full = np.asarray(jax.grad(lambda x: objective(tenet.ad._svd, x))(b))
    dropped = np.asarray(jax.grad(lambda x: objective(svd_no_phase, x))(b))
    if dtype == "float64":
        np.testing.assert_allclose(dropped, full, rtol=0, atol=1e-12)
    else:
        assert np.abs(dropped - full).max() > 1e-3


# --- jit -------------------------------------------------------------------------


@pytest.mark.parametrize("objective", [recon, eig_recon], ids=["svd", "eigh"])
def test_jit_of_grad_matches_the_unjitted_values(objective):
    tenet.ad.install()
    t = hermitian(jt(SQUARE["u1"], seed=14))
    plain = blocks(jax.grad(objective)(t))
    jitted = blocks(jax.jit(jax.grad(objective))(t))
    for a, b in zip(plain, jitted, strict=True):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-12)


def test_jit_still_traces_once_per_structure():
    """The trace count is the structure's, unchanged: the VJP is not a new cache key."""
    tenet.ad.install()
    count = 0

    @jax.jit
    def f(t):
        nonlocal count
        count += 1  # a Python side effect: it runs at trace time only
        return recon(t)

    f(jt(SQUARE["u1"], seed=15))
    f(jt(SQUARE["u1"], seed=16))
    assert count == 1
    f(jt(SQUARE_BIG, seed=17))  # different degeneracy -> different treedef
    assert count == 2
    f(jt(SQUARE["u1"], seed=18))  # back to the first structure: cache hit
    assert count == 2


# --- what the registration must NOT touch ----------------------------------------


@pytest.mark.parametrize("objective", [qr_recon, lq_recon], ids=["qr", "lq"])
def test_qr_and_lq_gradients_are_untouched(objective):
    t = jt(RECT, seed=19)
    before = blocks(jax.grad(objective)(t))
    tenet.ad.install()
    after = blocks(jax.grad(objective)(t))
    for a, b in zip(before, after, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("objective", [recon, eig_recon], ids=["svd", "eigh"])
def test_numpy_backend_is_unaffected(objective):
    t = hermitian(jt(SQUARE["u1"], seed=20)).to_backend("numpy")
    before = float(objective(t))
    tenet.ad.install()
    assert float(objective(t)) == before
    u, s, vh = ar.do("linalg.svd", np.eye(3), full_matrices=False)
    assert isinstance(s, np.ndarray)
    np.testing.assert_array_equal(s, np.linalg.svd(np.eye(3), full_matrices=False)[1])


# --- epsilon and the gauge precondition ------------------------------------------


def test_epsilon_is_live():
    """``O(eps)`` away from degeneracy, ``O(1)`` on a gauge-dependent objective near it."""
    away = jt(SQUARE["u1"], seed=21)
    at = near_degenerate(SQUARE["u1"], seed=22)  # see near_degenerate: exact is eps-blind

    tenet.ad.install(epsilon=1e-12)
    small_away, small_at = blocks(jax.grad(recon)(away)), blocks(jax.grad(gauge_dependent)(at))
    tenet.ad.install(epsilon=1e-11)
    big_away, big_at = blocks(jax.grad(recon)(away)), blocks(jax.grad(gauge_dependent)(at))

    for a, b in zip(small_away, big_away, strict=True):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-8)
    assert max(np.abs(a - b).max() for a, b in zip(small_at, big_at, strict=True)) > 1e-3


def test_a_gauge_dependent_objective_is_meaningless_at_degeneracy():
    """The documented precondition, pinned: this answer is a function of ``eps``.

    ``dU/dA`` does not exist inside a degenerate multiplet. Gauge-invariant objectives
    (reconstruction, the singular values, a multiplet projector) are the supported ones;
    this one is knowingly wrong, and is here so that "knowingly" is a test.
    """
    t = near_degenerate(SQUARE["u1"], seed=23)
    tenet.ad.install(epsilon=1e-12)
    a = blocks(jax.grad(gauge_dependent)(t))
    tenet.ad.install(epsilon=1e-6)
    b = blocks(jax.grad(gauge_dependent)(t))
    assert max(np.abs(x - y).max() for x, y in zip(a, b, strict=True)) > 1e-3


# --- the example still runs ------------------------------------------------------

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples"))
import vmc_mps  # noqa: E402


@pytest.mark.parametrize("provider", ["u1", "su2"])
def test_vmc_example_is_unchanged(provider):
    """Unchanged to 1e-10 without ``install()``, and to 1e-8 with it."""
    reference = vmc_mps.main(provider=provider)
    np.testing.assert_allclose(vmc_mps.main(provider=provider), reference, rtol=0, atol=1e-10)
    tenet.ad.install()
    np.testing.assert_allclose(vmc_mps.main(provider=provider), reference, rtol=0, atol=1e-8)
