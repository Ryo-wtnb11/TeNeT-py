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

import itertools
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

# The vendored SU(3) fixture provider lives beside the tests that built it; pytest
# puts each test file's own directory on sys.path, so from here it needs the same
# explicit insert the examples import at the bottom of this file uses. No racah-py:
# the coefficients are read from tests/fixtures/su3_*.txt.
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "symmetry"))
from _su3_fixture import EIGHT, ONE, SU3  # noqa: E402

# Square maps: one OUT leg and its IN partner, so every coupled-sector matrix is
# m_c x m_c and both svd and eigh accept the tensor. Degeneracies of 3 and 2 give a
# sector big enough for *two* zero singular values and one that is not.
Q = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(1): 2})
V = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2})  # qdim weights 1 and 3
F = GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 2})
# The multiplicity-bearing provider: the adjoint is self-dual and 8 x 8 -> 8 has
# N = 2, so plans over this space cross the matrix-valued F/R/B branch that SU(2)
# structurally cannot reach. Same space on both legs keeps every coupled-sector
# matrix square, as the SQUARE comment above requires.
W = GradedSpace.new(SU3, {ONE: 3, EIGHT: 2})
SQUARE = {
    "u1": (Leg(Q, OUT), Leg(Q, IN)),
    "su2": (Leg(V, OUT), Leg(V, IN)),
    "fz2": (Leg(F, OUT), Leg(F, IN)),
    "su3": (Leg(W, OUT), Leg(W, IN)),
}
PROVIDERS = list(SQUARE)

# Rectangular in both directions: (3, 2) in the c = 0 sector and (2, 3) in c = 1, so
# both ``rows > k`` and ``cols > k`` complement terms are exercised.
QA = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(1): 2})
QB = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})
RECT = (Leg(QA, OUT), Leg(QB, IN))

# The same both-directions shape for the other two providers, so #80's wide-qr / tall-lq
# coverage is not U(1)-only.
VB = GradedSpace.new(SU2, {SU2Sector(0): 3, SU2Sector(1): 1})
FB = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 3})
WB = GradedSpace.new(SU3, {ONE: 2, EIGHT: 3})  # (3, 2) and (2, 3), as for u1
RECT_BY_PROVIDER = {
    "u1": RECT,
    "su2": (Leg(V, OUT), Leg(VB, IN)),
    "fz2": (Leg(F, OUT), Leg(FB, IN)),
    "su3": (Leg(W, OUT), Leg(WB, IN)),
}

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
            assert "symtenet[jax]" in msg, msg
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


@pytest.mark.parametrize("shape", ["square", "rect"])
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("objective", [qr_recon, lq_recon], ids=["qr", "lq"])
def test_qr_and_lq_gradients_are_untouched(objective, name, shape):
    """``install()`` registers ``svd``/``eigh`` and nothing else, so these are bit-identical.

    ``RECT``'s sectors are ``(3, 2)`` and ``(2, 3)`` (and the ``su2``/``fz2`` mirrors the
    same), so the ``rect`` case silently exercises *wide* ``qr`` and *tall* ``lq`` — the
    shapes that need JAX's wide-QR JVP (>= 0.10, see #80). It would have caught the
    version gap had ``pyproject.toml``'s floor been honest.
    """
    t = jt(SQUARE[name] if shape == "square" else RECT_BY_PROVIDER[name], seed=19)
    before = blocks(jax.grad(objective)(t))
    tenet.ad.install()
    after = blocks(jax.grad(objective)(t))
    for a, b in zip(before, after, strict=True):
        assert np.array_equal(a, b)


def test_ad_registers_svd_and_eigh_only():
    tenet.ad.install()
    assert tenet.ad._NAMES == ("linalg.svd", "linalg.eigh")

    # ``_FUNCS`` is also autoray's resolution *cache*, so a key's mere presence proves
    # nothing; what proves it is whose function sits behind it.
    def owner(name):
        return ar.get_lib_fn("jax", name).__module__

    assert owner("linalg.svd").startswith("tenet.")
    assert owner("linalg.eigh").startswith("tenet.")
    assert not owner("linalg.qr").startswith("tenet.")


def test_numpy_backed_qr_and_lq_are_unaffected():
    t = jt(RECT, seed=41).to_backend("numpy")
    before = (float(qr_recon(t)), float(lq_recon(t)))
    tenet.ad.install()
    assert (float(qr_recon(t)), float(lq_recon(t))) == before
    q, r = ar.do("linalg.qr", np.eye(3))
    assert isinstance(q, np.ndarray) and isinstance(r, np.ndarray)
    want_q, want_r = np.linalg.qr(np.eye(3))
    np.testing.assert_array_equal(q, want_q)
    np.testing.assert_array_equal(r, want_r)


def test_uninstall_restores_qr_and_lq_gradients_in_the_same_process():
    t = jt(RECT, seed=42)
    stock = blocks(jax.grad(qr_recon)(t)) + blocks(jax.grad(lq_recon)(t))
    tenet.ad.install()
    tenet.ad.uninstall()
    again = blocks(jax.grad(qr_recon)(t)) + blocks(jax.grad(lq_recon)(t))
    for a, b in zip(stock, again, strict=True):
        assert np.array_equal(a, b)


# --- qr / lq: JAX's own rule, measured (#80) -------------------------------------
#
# No custom VJP exists for these: the wide case is JAX >= 0.10's ``qr_jvp_rule``
# (Roberts-Roberts Eqs. (9)-(10)), the square/tall one Liao-Liu-Wang-Xiang Eq. (5).
# What is pinned here is that the floor in ``pyproject.toml`` is high enough, on every
# sector shape, and that the exactly-rank-deficient corner stays *unsupported*.


def q_only(factorize, which):
    """``sum(sin(Q))`` / ``sum(cos(R))`` over the coupled-sector matrices — gauge-*dependent*.

    Unlike ``svd``, at full rank ``Q`` and ``R`` are each individually well defined, so
    there is no gauge precondition to hide behind and finite differences must agree.
    """

    def objective(t):
        f = jnp.sin if which == 0 else jnp.cos
        m = factorize(t)[which]
        return sum(jnp.real(jnp.sum(f(b))) for b in to_matrices(m).values())

    return objective


qr_q = q_only(tenet.linalg.qr, 0)
qr_r = q_only(tenet.linalg.qr, 1)
lq_l = q_only(tenet.linalg.lq, 0)
lq_q = q_only(tenet.linalg.lq, 1)

QR_OBJECTIVES = {
    "qr_recon": qr_recon,
    "lq_recon": lq_recon,
    "qr_q": qr_q,
    "qr_r": qr_r,
    "lq_l": lq_l,
    "lq_q": lq_q,
}


@pytest.mark.parametrize("shape", ["square", "rect"])
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("objective", [qr_recon, lq_recon], ids=["qr", "lq"])
def test_qr_and_lq_gradients_are_finite_on_every_sector_shape(objective, name, shape):
    """Square, tall and wide, for U(1), SU(2) and fZ2 — the JAX floor, as a test.

    On JAX < 0.10 the ``rect`` case raises ``NotImplementedError`` out of
    ``qr_jvp_rule``: wide for ``qr``, and for ``lq`` the *tall* sector, since
    ``ops/linalg.py::lq`` factorizes ``B†``.
    """
    legs = SQUARE[name] if shape == "square" else RECT_BY_PROVIDER[name]
    t = jt(legs, seed=43)
    if shape == "rect":  # both directions really are present
        assert {m.shape[0] > m.shape[1] for m in to_matrices(t).values()} == {True, False}
    g = jax.grad(objective)(t)
    assert all(np.isfinite(b).all() for b in blocks(g))


@pytest.mark.parametrize("shape", ["square", "rect"])
@pytest.mark.parametrize("name", ["u1", "su2"])
@pytest.mark.parametrize("label", list(QR_OBJECTIVES))
def test_finite_differences_qr_and_lq(label, name, shape):
    objective = QR_OBJECTIVES[label]
    legs = SQUARE[name] if shape == "square" else RECT_BY_PROVIDER[name]
    t = jt(legs, seed=44)
    g = jax.grad(objective)(t)
    for got, want in zip(blocks(g), central_differences(objective, t), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=label)


@pytest.mark.parametrize("objective", [qr_recon, lq_recon], ids=["qr", "lq"])
def test_complex_qr_and_lq_gradients_match_finite_differences(objective):
    """JAX's complex diagonal correction, through the wide path."""
    t = jt(RECT, seed=45, dtype="complex128")
    g = jax.grad(objective)(t)
    for got, want in zip(blocks(g), complex_central_differences(objective, t), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("objective", [qr_recon, lq_recon], ids=["qr", "lq"])
def test_rank_deficient_r_is_nan_before_and_after_install(objective):
    """The documented non-support, as an executable claim.

    An **exact** zero on ``R``'s diagonal makes the backward's triangular solve
    back-substitute through a zero pivot. It is not stabilized, before or after
    ``install()``: at exact rank deficiency the QR itself is non-unique, so there is no
    value to broaden towards (unlike ``svd``, whose limit exists and is what #76
    returns). If a future change accidentally stabilizes this, revisit the docstrings in
    ``ops/linalg.py`` deliberately rather than deleting this test.
    """
    t = with_zeros(SQUARE["u1"], 1, seed=46)  # refill + diagonal_fill: the zero is a fact
    assert has_nan(jax.grad(objective)(t))
    tenet.ad.install()
    assert has_nan(jax.grad(objective)(t))


@pytest.mark.parametrize("objective", [qr_recon, lq_recon], ids=["qr", "lq"])
def test_jit_of_qr_and_lq_grad_matches_the_unjitted_values(objective):
    t = jt(RECT, seed=47)  # wide (2, 3) and tall (3, 2) sectors
    plain = blocks(jax.grad(objective)(t))
    jitted = blocks(jax.jit(jax.grad(objective))(t))
    for a, b in zip(plain, jitted, strict=True):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-12)


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


# --- svd(bond=...): differentiating through a truncation (#77) -------------------


def with_spectrum(legs, values, seed=0):
    """A tensor with a *designed* spectrum per sector, in random orthogonal bases.

    A diagonal fill would make ``U`` and ``V`` permutations and hide half the backward;
    the random bases put real content in the ``dU``/``dV`` cotangents, which is exactly
    where a truncated backward can be wrong.
    """
    counter = itertools.count()

    def fill(r, c):
        rng = np.random.default_rng(seed * 97 + next(counter))
        k = min(r, c)
        m = np.zeros((r, c))
        m[:k, :k] = np.diag(values(k))
        q1 = np.linalg.qr(rng.standard_normal((r, r)))[0]
        q2 = np.linalg.qr(rng.standard_normal((c, c)))[0]
        return q1 @ m @ q2

    return refill(jt(legs, seed), fill)


def clean_gap(k):
    """``1.0, 0.9, ...`` with the last value a factor ``1e-3`` below the one above it."""
    d = 1.0 - 0.1 * np.arange(k)
    d[-1] = d[-2] * 1e-3
    return d


def drop_one(t):
    """The full ``min`` bond with one singular value taken off every sector."""
    full = tenet.linalg.svd(t)[0].legs[-1].space
    return GradedSpace.new(full.provider, {c: m - 1 for c, m in full.sectors if m > 1})


def full_bond(t):
    return tenet.linalg.svd(t)[0].legs[-1].space


def overlap(t, bond, w):
    """``<W, U S Vh>`` — gauge-invariant, and the cotangent reaches ``U`` and ``V``.

    ``norm(U @ S @ Vh)`` and ``norm(S)`` depend on the kept singular *values* alone, so
    their cotangents for ``U`` and ``Vh`` are symbolic zeros and a wrong truncated
    backward is invisible in them. This one is the discriminating objective.
    """
    u, s, vh = tenet.linalg.svd(t, bond=bond)
    x = u @ s @ vh
    return sum(jnp.real(jnp.sum(a * b)) for a, b in zip(x.blocks, w.blocks, strict=True))


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_finite_differences_through_a_truncation_clean_gap(name):
    """Clean gap at the cut (``sigma_perp/sigma_min == 1e-3``), bond decided outside."""
    tenet.ad.install()
    legs = SQUARE[name]
    t = with_spectrum(legs, clean_gap, seed=30)
    bond = drop_one(t)
    w = jt(legs, seed=31)
    assert bond != full_bond(t)  # the truncation is not vacuous

    objectives = {
        "recon": lambda x: tenet.norm(
            tenet.linalg.svd(x, bond=bond)[0]
            @ tenet.linalg.svd(x, bond=bond)[1]
            @ tenet.linalg.svd(x, bond=bond)[2]
        ),
        "spectrum": lambda x: tenet.norm(tenet.linalg.svd(x, bond=bond)[1]),
        "overlap": lambda x: overlap(x, bond, w),
    }
    for label, f in objectives.items():
        got = blocks(jax.grad(f)(t))
        want = central_differences(f, t)
        for a, b in zip(got, want, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, err_msg=label)


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_full_bond_gradient_is_the_bond_none_gradient(name):
    """The truncated backward degenerates to the compact one, as it must."""
    tenet.ad.install()
    legs = SQUARE[name]
    t = jt(legs, seed=32)
    w = jt(legs, seed=33)
    bond = full_bond(t)

    def plain(x):
        u, s, vh = tenet.linalg.svd(x)
        y = u @ s @ vh
        return sum(jnp.real(jnp.sum(a * b)) for a, b in zip(y.blocks, w.blocks, strict=True))

    for a, b in zip(
        blocks(jax.grad(lambda x: overlap(x, bond, w))(t)), blocks(jax.grad(plain)(t)), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


# --- the approximation, measured rather than assumed ------------------------------


def _zeroth_order_truncated_svd(k):
    """The rule #77 set out to ship: #76's VJP formed against the **kept** factors only.

    Francuz, Schuch and Vanhecke (Phys. Rev. Research 7, 013237 (2025)) Eqs. (14)-(15)
    write the exact truncated pullback as

        (1 - U U†) dA V   = (1 - U U†)(dU S - A dV)
        (1 - V V†) dA† U  = (1 - V V†)(dV S - A† dU)

    whose underlined ``A dV`` / ``A† dU`` terms have the *discarded* block
    ``A_perp = A - U S V†`` as their kernel; substituting ``A -> U S V†`` drops them and
    leaves exactly the formula below. This function exists to measure what that costs.
    """

    @jax.custom_vjp
    def f(b):
        u, s, vh = jnp.linalg.svd(b, full_matrices=False)
        return u[:, :k], s[:k], vh[:k, :]

    def fwd(b):
        u, s, vh = jnp.linalg.svd(b, full_matrices=False)
        out = (u[:, :k], s[:k], vh[:k, :])
        return out, out

    f.defvjp(fwd, tenet.ad._svd_bwd)  # the kept factors are the only residuals it sees
    return f


def test_the_truncated_backward_is_exact_and_the_zeroth_order_one_is_first_order():
    """What ``bond=`` actually computes, measured against the rule #77 planned to ship.

    The finding, and the reason the Francuz-Schuch-Vanhecke follow-up is **not** needed:
    ``ops/linalg.py`` computes the *full* compact SVD (``k = min(rows_c, cols_c)``, #57's
    bond rule) and only then slices, so the discarded space never leaves the
    factorization. JAX zero-pads the slice's cotangent, and the resulting cross block of
    #76's broadened ``F`` matrix -- rows ``i > k`` against columns ``j <= k``, weighted by
    ``1/(sigma_j - sigma_i)`` -- *is* the ``A_perp dV`` correction. So the composition is
    the exact truncated pullback, not the zeroth-order one, and it stays exact as the gap
    at the cut closes.

    Measured relative deviation from central differences, ``sigma_perp/sigma_min = r``,
    3x3 block, ``k = 1`` (this test's numbers, float64, ``h = 1e-6``):

    ====== ==================== ==========================
    ``r``  ``svd(bond=)``       zeroth-order (Eqs. 14-15
                                without the underlined term)
    ====== ==================== ==========================
    0.1    ~2e-10 (FD noise)    1.1e-1
    0.3    ~4e-10 (FD noise)    3.1e-1
    0.5    ~3e-10 (FD noise)    4.9e-1
    0.7    ~2e-10 (FD noise)    6.6e-1
    0.9    ~2e-10 (FD noise)    8.4e-1
    ====== ==================== ==========================

    So the zeroth-order error is ``O(sigma_perp/sigma_min)`` -- **first** order, log-log
    slope 0.97 measured down to ``r = 3e-3``, not the second order the issue conjectured
    from MatrixAlgebraKit's ``(A_perp A_perp†)/S**2`` doubling iteration -- and ours is
    flat at finite-difference noise across the whole sweep.
    """
    tenet.ad.install()
    rng = np.random.default_rng(9)
    w = rng.standard_normal((3, 3))
    zeroth = _zeroth_order_truncated_svd(1)

    def exact(b):
        u, s, vh = tenet.ad._svd(b)
        return u[:, :1], s[:1], vh[:1, :]

    def objective(f, b):
        u, s, vh = f(b)
        return jnp.real(jnp.sum(w * (u @ jnp.diag(s) @ vh)))

    def block(r, seed=3):
        gen = np.random.default_rng(seed)
        q1 = np.linalg.qr(gen.standard_normal((3, 3)))[0]
        q2 = np.linalg.qr(gen.standard_normal((3, 3)))[0]
        return jnp.asarray(q1 @ np.diag([1.0, r, 0.9 * r]) @ q2)

    def differences(f, b, h=1e-6):
        g = np.zeros(b.shape)
        for idx in np.ndindex(b.shape):
            plus = float(objective(f, b.at[idx].add(h)))
            minus = float(objective(f, b.at[idx].add(-h)))
            g[idx] = (plus - minus) / (2 * h)
        return g

    ratios = (0.1, 0.3, 0.5, 0.7, 0.9)
    ours, theirs = [], []
    for r in ratios:
        b = block(r)
        reference = differences(exact, b)
        scale = np.abs(reference).max()
        ours.append(
            np.abs(np.asarray(jax.grad(lambda x: objective(exact, x))(b)) - reference).max() / scale
        )
        theirs.append(
            np.abs(np.asarray(jax.grad(lambda x: objective(zeroth, x))(b)) - reference).max()
            / scale
        )

    # ours: exact, and flat -- no growth as the gap closes
    assert max(ours) < 1e-7, ours
    assert max(ours) / min(ours) < 50, ours
    # the zeroth-order rule: linear in the gap ratio, and already 10% at r = 0.1
    assert theirs[0] > 0.05, theirs
    slope = float(np.polyfit(np.log(ratios), np.log(theirs), 1)[0])
    assert 0.85 < slope < 1.25, (slope, theirs)


# --- degeneracy, carried through the projection ----------------------------------


def kept_degenerate(k):
    """``1, 1, ...`` with a tail a factor ``1e-3`` down: the pair sits inside the cut."""
    d = np.ones(k)
    d[-1] = 1e-3
    return d


def exactly(legs, values, seed):
    """:func:`with_spectrum` without the random bases — because *exact* is the point.

    ``q1 @ diag(sigma) @ q2`` reproduces ``sigma`` only to rounding, and a degeneracy
    that holds to 1e-16 is not a degeneracy: stock JAX survives it. The diagonal fill
    is what puts two bit-identical singular values in one coupled sector.
    """
    return refill(jt(legs, seed), diagonal_fill(values))


def straddling(k):
    """A degenerate pair split by the cut: ``1, 0.5, 0.5`` where ``k == 3``."""
    return np.array([1.0, 0.5, 0.5]) if k == 3 else np.array([0.9, 0.4])


def test_a_degeneracy_inside_a_kept_sector_survives_the_projection():
    """#76's broadening carrying through the slice, asserted rather than assumed."""
    legs = SQUARE["u1"]  # sector 0 is 3x3: two equal values kept, one tail dropped
    t = exactly(legs, kept_degenerate, seed=34)
    bond = drop_one(t)
    w = jt(legs, seed=35)
    sigma = np.diagonal(np.asarray(to_matrices(tenet.linalg.svd(t)[1])[U1Sector(0)]))
    assert sigma[0] == pytest.approx(sigma[1], abs=1e-12)  # the degeneracy is real
    assert bond.degeneracy(U1Sector(0)) == 2  # and both members are kept

    assert has_nan(jax.grad(lambda x: overlap(x, bond, w))(t))
    tenet.ad.install()
    g = jax.grad(lambda x: overlap(x, bond, w))(t)
    assert not has_nan(g)
    assert all(np.isfinite(b).all() for b in blocks(g))


def test_a_degenerate_pair_straddling_the_cut():
    """Finite with ``install()``, ``NaN`` without — and the tie-break is svd_truncated's.

    A multiplet split by the cut makes the kept subspace gauge-*dependent*, so the
    number is meaningless (tenet.ad's documented precondition); what is asserted here is
    that it is finite and that the structural decision is reproducible. Francuz et al.
    say the same from the physics side: "one should be careful not to split multiplets".
    """
    legs = SQUARE["u1"]
    t = exactly(legs, straddling, seed=36)
    w = jt(legs, seed=37)

    # global descending order: 1.0 (c0, 0), 0.9 (c1, 0), 0.5 (c0, 1), 0.5 (c0, 2), 0.4
    bond = tenet.linalg.svd_truncated(t, max_bond=3)[0].legs[-1].space
    assert bond.sectors == ((U1Sector(0), 2), (U1Sector(1), 1))
    # the tie went to the lower index: the kept set is a prefix, deterministically
    kept = np.diagonal(np.asarray(to_matrices(tenet.linalg.svd(t, bond=bond)[1])[U1Sector(0)]))
    np.testing.assert_allclose(kept, [1.0, 0.5], rtol=0, atol=1e-12)

    assert has_nan(jax.grad(lambda x: overlap(x, bond, w))(t))
    tenet.ad.install()
    assert not has_nan(jax.grad(lambda x: overlap(x, bond, w))(t))


# --- the shape of the use case ---------------------------------------------------


def test_a_mini_differentiable_ctmrg_step():
    """Three power-iteration steps, each projecting onto one bond decided outside.

    The shape of Liao-Liu-Wang-Xiang Fig. 2(b): a truncated SVD *inside* the region
    being differentiated, with the kept subspace frozen across it. One
    ``jax.jit(jax.grad(...))``, one trace.
    """
    tenet.ad.install()
    legs = SQUARE["u1"]
    t = with_spectrum(legs, clean_gap, seed=38)
    bond, w = drop_one(t), jt(legs, seed=39)
    traces = []

    def power(x0):
        traces.append(1)  # a Python side effect: it runs at trace time only
        x = x0
        for _ in range(3):
            u, s, vh = tenet.linalg.svd(x @ x0, bond=bond)
            x = u @ s @ vh
        return sum(jnp.real(jnp.sum(a * b)) for a, b in zip(x.blocks, w.blocks, strict=True))

    differentiate = jax.jit(jax.grad(power))
    g = differentiate(t)
    differentiate(with_spectrum(legs, clean_gap, seed=40))  # same structure: cache hit
    assert len(traces) == 1
    assert not has_nan(g)
    for a, b in zip(blocks(g), central_differences(power, t), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


# --- the example still runs ------------------------------------------------------

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples" / "toy_codes"))
import vmc_mps  # noqa: E402

# The first trace per provider, memoized so the docs page test below reuses the run
# ``test_vmc_example_is_unchanged`` already pays for (#164).
_VMC_TRACES: dict[str, list] = {}


def _vmc_trace(provider):
    if provider not in _VMC_TRACES:
        _VMC_TRACES[provider] = vmc_mps.main(provider=provider)
    return _VMC_TRACES[provider]


@pytest.mark.parametrize("provider", ["u1", "su2"])
def test_vmc_example_is_unchanged(provider):
    """Unchanged to 1e-10 without ``install()``, and to 1e-8 with it."""
    reference = _vmc_trace(provider)
    np.testing.assert_allclose(vmc_mps.main(provider=provider), reference, rtol=0, atol=1e-10)
    tenet.ad.install()
    np.testing.assert_allclose(vmc_mps.main(provider=provider), reference, rtol=0, atol=1e-8)


def test_vmc_page_output_is_current():
    """The docs page's two lines, from the traces the parametrized test above computed.

    ``main()`` returns the trace and prints nothing; the formatting lives in the file's
    ``__main__`` guard, and the toy codes are frozen content (#148).
    """
    from helpers import check_example_page

    # Simplification: duplicates the guard's format string from
    # examples/toy_codes/vmc_mps.py; move the print into main() if the freeze is lifted.
    lines = [
        f"{provider}: " + " ".join(f"{e:+.6f}" for e in _vmc_trace(provider))
        for provider in ("u1", "su2")
    ]
    check_example_page("toy-vmc-mps.md", "".join(line + "\n" for line in lines))


@pytest.mark.parametrize("provider", ["u1", "su2"])
def test_vmc_example_pairs_compress_with_project(provider):
    """``compress`` decides the bond outside; ``project`` differentiates onto it inside."""
    tenet.ad.install()
    t = vmc_mps.build_mps(4, provider=provider, seed=5)[1]
    bond = vmc_mps.compress(t, max_bond=2)[0].legs[-1].space

    with pytest.raises(tenet.StructureChangingError):
        jax.jit(lambda x: vmc_mps.compress(x)[0])(t)

    g = jax.jit(jax.grad(lambda x: tenet.norm(vmc_mps.project(x, bond)[1])))(t)
    assert not has_nan(g)
    assert vmc_mps.project(t, bond)[0].legs[-1].space == bond
