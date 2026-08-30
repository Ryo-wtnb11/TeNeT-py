"""Tests for ``expm`` (#86) and ``eig``/``eigvals`` (#87).

Extends ``tests/ops/test_linalg.py`` and ``tests/ops/test_linalg_more.py`` rather
than forking them: the spaces, the square/flat leg patterns, the ``hermitian``
fixture and the jit helpers are imported from there.

Two criteria here are stated on *non*-gauge-invariant quantities on purpose, and
both are negatives that must not be "fixed" later: ``V`` from :func:`eig` is
asserted **not** to be an isometry, and the ``eigh``-route exponential is asserted
to **disagree** with :func:`expm` off the Hermitian locus. Both are the reason the
functions are shaped the way they are.
"""

import sys

import numpy as np
import pytest
import scipy.linalg
from test_linalg import SPLIT, dense_matrix, tensor, to_jax, use_jax
from test_linalg_more import (
    FLAT_LEGS,
    PROVIDERS,
    SQUARE_LEGS,
    eigenvalues,
    hermitian,
    with_domain_names,
)

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import from_matrices, map_layout, to_matrices
from tenet.symmetry import U1, U1Sector

REPARTITION_MODULE = sys.modules["tenet.ops.repartition"]

ALPHAS = [1.0, -0.3j]


def square(name: str, seed: int = 0) -> SymmetricTensor:
    """A random — hence genuinely non-Hermitian — square map, no bend needed."""
    return SymmetricTensor.random(FLAT_LEGS[name], seed=seed)


def matrices(t: SymmetricTensor) -> dict:
    return {c: np.asarray(b) for c, b in to_matrices(t).items()}


def sort_complex(values: np.ndarray) -> np.ndarray:
    """Sorted by ``(Re, Im)`` — *rounded* first, or a conjugate pair whose real parts
    differ in the last bit sorts one way here and the other way in the oracle."""
    keys = np.round(np.stack([values.real, values.imag]), 9)
    return values[np.lexsort((keys[1], keys[0]))]


def spectrum(w: SymmetricTensor, provider) -> np.ndarray:
    """Every eigenvalue of ``W``, repeated ``irrep_dim(c)`` times, sorted by (Re, Im)."""
    return sort_complex(
        np.concatenate([np.repeat(v, provider.irrep_dim(c)) for c, v in eigenvalues(w).items()])
    )


# --- expm: structure and the map spelling --------------------------------------------


def crossing(name: str, seed: int = 0) -> SymmetricTensor:
    """``SQUARE_LEGS`` — square only *after* ``SPLIT`` bends axes 1 and 2 (#63's case)."""
    return SymmetricTensor.random(SQUARE_LEGS[name], seed=seed)


@pytest.mark.parametrize("name", PROVIDERS)
def test_expm_carries_the_repartitioned_structure_exactly(name):
    """No bond leg, no leg change — ``polar``'s ``W`` is the only other one like this."""
    t = crossing(name, seed=1)
    m = tenet.repartition(t, *SPLIT)
    got = tenet.linalg.expm(t, axes=SPLIT, alpha=0.7)
    assert got.structure == m.structure
    assert got.legs == m.legs


@pytest.mark.parametrize("name", PROVIDERS)
def test_as_map_expm_matches_the_axes_spelling_and_bends_nothing(name, monkeypatch):
    t = square(name, seed=2)
    axes = (t.structure.out_axes, t.structure.in_axes)

    def no_bends(*args, **kwargs):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("as_map().expm() must perform zero bends")

    monkeypatch.setattr(REPARTITION_MODULE, "bend", no_bends)
    got = t.as_map().expm(alpha=-0.3j)
    want = tenet.linalg.expm(t, axes=axes, alpha=-0.3j)
    assert got.structure == want.structure
    for a, b in zip(got.blocks, want.blocks, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# --- expm: the dense oracle and the group law -----------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_expm_dense_oracle(name, alpha):
    """``FLAT_LEGS`` needs no bend, so no dual leg enters the dense expansion."""
    t = square(name, seed=3)
    got = dense_matrix(tenet.linalg.expm(t, alpha=alpha))
    want = scipy.linalg.expm(alpha * dense_matrix(t))
    # relative to the exponential's own scale: exp of an O(1) matrix is not O(1)
    np.testing.assert_allclose(got, want, atol=1e-13 * max(1.0, float(np.abs(want).max())))


@pytest.mark.parametrize("name", PROVIDERS)
def test_expm_at_alpha_zero_is_the_identity(name):
    t = square(name, seed=4)
    got = tenet.linalg.expm(t, alpha=0.0)
    # ``identity`` names its IN legs after the codomain; ``t``'s domain names its own
    want = with_domain_names(tenet.identity(t.codomain), t)
    assert tenet.allclose(got, want, rtol=0, atol=1e-14)


@pytest.mark.parametrize("name", PROVIDERS)
def test_expm_is_a_one_parameter_group(name):
    """``exp(aT) exp(bT) == exp((a+b)T)``: no per-sector bookkeeping bug survives it."""
    t = square(name, seed=5)
    a, b = 0.4, -0.25j
    product = tenet.linalg.expm(t, alpha=a) @ tenet.linalg.expm(t, alpha=b)
    assert tenet.allclose(product, tenet.linalg.expm(t, alpha=a + b), rtol=0, atol=1e-12)


# --- expm: the Trotter gate ------------------------------------------------------------


@pytest.mark.parametrize("name", ["su2", "u1"])
def test_expm_of_a_hermitian_generator_is_unitary(name):
    """``u = exp(-i dt h)``: ``u†u == id`` and ``u`` preserves the norm. #86's motivation."""
    h = hermitian(FLAT_LEGS[name], seed=6)
    u = tenet.linalg.expm(h, alpha=-1j * 0.35)
    assert tenet.allclose(tenet.adjoint(u) @ u, tenet.identity(h.domain), rtol=0, atol=1e-13)
    x = square(name, seed=7)
    assert float(tenet.norm(u @ x)) == pytest.approx(float(tenet.norm(x)), abs=1e-12)


# --- expm: the eigh route, agreed with and then disagreed with --------------------------


def eigh_route(t: SymmetricTensor, alpha) -> dict:
    """``V diag(exp(alpha w)) V†`` per coupled sector, built from ``tenet.linalg.eigh``.

    This is what ``expm`` would be if it inherited ``eigh``'s "the caller promised"
    clause — correct on the Hermitian locus and quietly wrong off it, which is
    exactly why ``expm`` does not go through ``eigh``. See :func:`tenet.linalg.expm`.
    """
    w, v = tenet.linalg.eigh(t)
    vectors, values = matrices(v), eigenvalues(w)
    return {
        c: vec @ np.diag(np.exp(alpha * values[c])) @ vec.conj().T for c, vec in vectors.items()
    }


@pytest.mark.parametrize("alpha", ALPHAS)
def test_expm_agrees_with_the_eigh_route_on_a_hermitian_tensor(alpha):
    h = hermitian(FLAT_LEGS["su2"], seed=8)
    got, want = matrices(tenet.linalg.expm(h, alpha=alpha)), eigh_route(h, alpha)
    for c in got:
        np.testing.assert_allclose(got[c], want[c], atol=1e-14, err_msg=repr(c))


def test_expm_disagrees_with_the_eigh_route_off_the_hermitian_locus():
    """A deliberate negative: the eigh route reads one triangle and is *wrong* here."""
    t = square("su2", seed=9)
    assert max(float(np.linalg.norm(b - b.conj().T)) for b in matrices(t).values()) > 1e-3
    got, want = matrices(tenet.linalg.expm(t, alpha=0.2)), eigh_route(t, 0.2)
    assert max(float(np.max(np.abs(got[c] - want[c]))) for c in got) > 1e-2


# --- expm: dtype promotion --------------------------------------------------------------


def test_a_complex_alpha_promotes_real_blocks():
    t = square("su2", seed=10)
    assert not np.issubdtype(np.asarray(t.blocks[0]).dtype, np.complexfloating)
    u = tenet.linalg.expm(t, alpha=-1j * 0.2)
    assert np.asarray(u.blocks[0]).dtype == np.complex128


def test_a_complex_tensor_exponentiates_against_the_dense_oracle():
    t = square("su2", seed=11)
    t = SymmetricTensor(t.structure, tuple(b + 0.5j * b[::-1] for b in t.blocks))
    u = tenet.linalg.expm(t, alpha=0.3)
    assert np.asarray(u.blocks[0]).dtype == np.complex128
    np.testing.assert_allclose(
        dense_matrix(u), scipy.linalg.expm(0.3 * dense_matrix(t)), atol=1e-13
    )


# --- expm: the square-map refusal -------------------------------------------------------


def test_expm_refuses_a_non_square_map_naming_itself():
    with pytest.raises(ValueError) as excinfo:
        tenet.linalg.expm(tensor("u1"), axes=SPLIT)
    message = str(excinfo.value)
    assert message.startswith("expm: the map is not square at position 0")
    assert "eigh" not in message
    assert "public axis 0" in message and "public axis 2" in message
    assert "expm requires the domain to be the codomain" in message


@pytest.mark.parametrize(
    "fn",
    [tenet.linalg.expm, tenet.linalg.eig, tenet.linalg.eigvals],
    ids=["expm", "eig", "eigvals"],
)
def test_a_charge_reversed_u1_partner_of_the_right_dimension_is_refused(fn):
    up = GradedSpace.new(U1, {U1Sector(1): 2, U1Sector(-1): 3})
    down = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(1): 3})
    assert up.dim == down.dim
    t = SymmetricTensor.random((Leg(up, OUT, name="a"), Leg(down, IN, name="b")), seed=12)
    with pytest.raises(ValueError, match=f"{fn.__name__}: the map is not square at position 0"):
        fn(t)


def test_the_square_refusal_reads_no_block_value_and_is_the_same_under_jit():
    jax = use_jax()
    t = tensor("u1", seed=13)
    nan = SymmetricTensor(t.structure, tuple(np.full_like(b, np.nan) for b in t.blocks))
    with pytest.raises(ValueError) as eager:
        tenet.linalg.expm(nan, axes=SPLIT)
    with pytest.raises(ValueError) as traced:
        jax.jit(lambda x: tenet.linalg.expm(x, axes=SPLIT))(to_jax(t))
    assert str(traced.value) == str(eager.value)


# --- expm: SciPy is not a core dependency (#86, maintainer decision) ----------------------


def test_expm_on_numpy_without_scipy_names_pip_install_scipy(monkeypatch):
    """The alternative the maintainer took: no core SciPy, an actionable ImportError.

    autoray resolves NumPy's ``linalg.expm`` to ``scipy.linalg.expm``; blocking the
    import is the only way to see what a bare ``pip install symtenet`` would see.
    """
    import autoray as ar

    for key in [k for k in sys.modules if k == "scipy" or k.startswith("scipy.")]:
        monkeypatch.setitem(sys.modules, key, None)
    saved = ar.autoray._FUNCS.pop(("numpy", "linalg.expm"), None)
    ar.autoray._NAMESPACE_CACHE.clear()
    try:
        with pytest.raises(ImportError) as excinfo:
            tenet.linalg.expm(square("u1", seed=14))
    finally:
        if saved is not None:
            ar.autoray._FUNCS[("numpy", "linalg.expm")] = saved
        ar.autoray._NAMESPACE_CACHE.clear()
    message = str(excinfo.value)
    assert "pip install scipy" in message
    assert "not a dependency" in message and "JAX backend" in message


# --- expm: jit, grad and the max_squarings ceiling ----------------------------------------


def test_jit_expm_traces_once_and_alpha_stays_a_traced_scalar():
    """``alpha`` is a *traced* argument here: changing its value must not retrace."""
    jax = use_jax()
    traces = []

    @jax.jit
    def gate(x, alpha):
        traces.append(1)
        return tenet.norm(tenet.linalg.expm(x, alpha=alpha))

    t = to_jax(square("su2", seed=15))
    import jax.numpy as jnp

    got = gate(t, jnp.asarray(-0.1j))
    gate(to_jax(square("su2", seed=16)), jnp.asarray(-0.1j))
    gate(t, jnp.asarray(-0.2j))  # a different value of a traced scalar: still one trace
    assert len(traces) == 1
    want = tenet.norm(tenet.linalg.expm(square("su2", seed=15), alpha=-0.1j))
    assert float(got) == pytest.approx(float(want), abs=1e-10)


def central_differences(t: SymmetricTensor, objective, indices, h: float = 1e-5) -> np.ndarray:
    """``d objective / d t.blocks[0][idx]`` by central differences, in float64."""
    out = []
    for idx in indices:
        shifted = []
        for step in (h, -h):
            blocks = [np.asarray(b).copy() for b in t.blocks]
            blocks[0][idx] += step
            shifted.append(float(objective(SymmetricTensor(t.structure, tuple(blocks)))))
        out.append((shifted[0] - shifted[1]) / (2 * h))
    return np.asarray(out)


def first_block_indices(t: SymmetricTensor):
    return list(np.ndindex(np.asarray(t.blocks[0]).shape))


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_expm_grad_matches_finite_differences_without_tenet_ad(name):
    """Explicitly *before* ``tenet.ad.install()``: this route needs no broadening."""
    jax = use_jax()
    t = square(name, seed=17)

    def objective(x):
        return tenet.norm(tenet.linalg.expm(x, alpha=-0.3))

    g = jax.grad(objective)(to_jax(t))
    assert all(np.isfinite(np.asarray(b)).all() for b in g.blocks)
    indices = first_block_indices(t)
    np.testing.assert_allclose(
        np.asarray([np.asarray(g.blocks[0])[i] for i in indices]),
        central_differences(t, objective, indices),
        atol=1e-6,
    )


def test_expm_is_finite_at_full_degeneracy_where_the_eigh_route_is_nan():
    """The measurement that decided #86, asserted in one place, under *stock* JAX."""
    jax = use_jax()
    t = square("su2", seed=18)
    identity = to_jax(with_domain_names(tenet.identity(t.codomain), t))

    def by_expm(x):
        return tenet.norm(tenet.linalg.expm(x, alpha=-0.3))

    def by_eigh(x):
        w, v = tenet.linalg.eigh(x)
        return tenet.norm(v @ w @ tenet.adjoint(v))

    good = jax.grad(by_expm)(identity)
    assert all(np.isfinite(np.asarray(b)).all() for b in good.blocks)
    bad = jax.grad(by_eigh)(identity)
    assert not all(np.isfinite(np.asarray(b)).all() for b in bad.blocks)


def test_tenet_ad_does_not_grow_and_leaves_expm_gradients_untouched():
    jax = use_jax()
    import tenet.ad

    assert tenet.ad._NAMES == ("linalg.svd", "linalg.eigh")
    t = to_jax(square("su2", seed=19))

    def objective(x):
        return tenet.norm(tenet.linalg.expm(x, alpha=-0.3))

    before = jax.grad(objective)(t)
    tenet.ad.install()
    try:
        after = jax.grad(objective)(t)
    finally:
        tenet.ad.uninstall()
    for a, b in zip(before.blocks, after.blocks, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_max_squarings_ceiling_is_nan_and_not_an_exception():
    """``jax/_src/scipy/linalg.py``: ``lax.cond(n_squarings > max_squarings, _nan, ...)``.

    ``NaN``, not an exception and not an overflow ``inf``: past ``max_squarings=16``
    JAX substitutes ``_nan`` for the whole result. Not wrapped, because a norm
    comparison is a data-dependent branch and could not run inside a trace
    (invariant 9); the caller's escape is ``exp(aH) = exp(aH/n)**n``, which is the
    caller's to apply — at *this* ``alpha`` the true exponential overflows float64
    anyway, so there is nothing here for the library to rescue.
    """
    use_jax()
    t = to_jax(square("su2", seed=20))
    blown = tenet.linalg.expm(t, alpha=1e7)
    assert any(np.isnan(np.asarray(b)).any() for b in blown.blocks)
    assert all(np.isfinite(np.asarray(b)).all() for b in tenet.linalg.expm(t, alpha=0.5).blocks)


# --- eig / eigvals: legs, conventions and the map spelling ---------------------------------


@pytest.mark.parametrize("name", ["u1", "su2", "fz2"])
def test_eig_factor_legs_are_eighs(name):
    t = square(name, seed=21)
    m = tenet.repartition(t, *SPLIT)
    w, v = tenet.linalg.eig(t, axes=SPLIT)

    bond = v.legs[-1].space
    assert w.legs == (Leg(bond, OUT), Leg(bond, IN))
    assert v.legs == (*m.codomain, Leg(bond, IN))
    assert w.legs[0].dual is False and w.legs[1].dual is False  # the identity mirror
    assert v.legs[-1] == Leg(w.legs[0].space, IN)

    layout = map_layout(m.structure)
    for c in layout.sectors:
        rows, cols = layout.shape(c)
        assert rows == cols  # square, so ``_lower``'s min is vacuous — asserted
        assert bond.degeneracy(c) == rows
    assert tuple(bond) == layout.sectors


@pytest.mark.parametrize("name", PROVIDERS)
def test_eigvals_equals_eigs_first_output_exactly(name):
    t = square(name, seed=22)
    w = tenet.linalg.eig(t)[0]
    got = tenet.linalg.eigvals(t)
    assert got.structure == w.structure
    for a, b in zip(got.blocks, w.blocks, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("name", PROVIDERS)
def test_as_map_eig_and_eigvals_match_the_free_functions_and_bend_nothing(name, monkeypatch):
    t = square(name, seed=23)
    axes = (t.structure.out_axes, t.structure.in_axes)

    def no_bends(*args, **kwargs):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("as_map().eig()/.eigvals() must perform zero bends")

    monkeypatch.setattr(REPARTITION_MODULE, "bend", no_bends)
    for got, want in zip(t.as_map().eig(), tenet.linalg.eig(t, axes=axes), strict=True):
        assert got.structure == want.structure
        for a, b in zip(got.blocks, want.blocks, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    got, want = t.as_map().eigvals(), tenet.linalg.eigvals(t, axes=axes)
    assert got.structure == want.structure
    for a, b in zip(got.blocks, want.blocks, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# --- eig: the residual, and the isometry that is deliberately absent -------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_eigen_residual_is_zero_on_a_non_hermitian_map(name):
    """``T V - V W``: the reconstruction criterion that needs no ``inv``."""
    t = square(name, seed=24)
    assert max(float(np.linalg.norm(b - b.conj().T)) for b in matrices(t).values()) > 1e-3
    w, v = tenet.linalg.eig(t)
    residual = float(tenet.norm(t @ v - v @ w))
    assert residual <= 1e-12 * float(tenet.norm(t))


def test_v_is_deliberately_not_an_isometry():
    """A negative on purpose, so that nobody later "fixes" it.

    Right eigenvectors of a non-normal matrix are not orthogonal — see the second
    paragraph of :func:`tenet.linalg.eig`'s docstring.
    """
    t = square("su2", seed=25)
    _, v = tenet.linalg.eig(t)
    bond = tenet.identity((Leg(v.legs[-1].space, OUT),))
    overlap = tenet.adjoint(v) @ v
    assert not tenet.allclose(overlap, bond, rtol=0, atol=1e-3)


# --- eig: the dense oracle, and agreement with eigh where both are defined --------------------


@pytest.mark.parametrize("name", ["u1", "su2", "fz2"])
def test_dense_oracle_eigenvalues_carry_the_qdim_weight(name):
    t = square(name, seed=26)
    got = spectrum(tenet.linalg.eig(t)[0], t.provider)
    want = sort_complex(np.linalg.eigvals(dense_matrix(t)))
    assert len(got) == len(want)
    np.testing.assert_allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_eigvals_agrees_with_eigh_on_a_hermitian_fixture(name):
    h = hermitian(FLAT_LEGS[name], seed=27)
    got = spectrum(tenet.linalg.eigvals(h), h.provider)
    np.testing.assert_allclose(got.imag, 0.0, atol=1e-12)
    want = spectrum(tenet.linalg.eigh(h)[0], h.provider)
    np.testing.assert_allclose(got.real, want.real, atol=1e-10)


def test_eig_survives_the_crossing_fz2_split_with_its_signs():
    t = crossing("fz2", seed=28)
    m = tenet.repartition(t, *SPLIT)
    w, v = tenet.linalg.eig(t, axes=SPLIT)
    assert float(tenet.norm(m @ v - v @ w)) <= 1e-12 * float(tenet.norm(m))


# --- eig: dtypes -------------------------------------------------------------------------------


def test_real_input_gives_complex_outputs_on_numpy_and_on_jax():
    t = square("su2", seed=29)
    assert not np.issubdtype(np.asarray(t.blocks[0]).dtype, np.complexfloating)
    w, v = tenet.linalg.eig(t)
    assert np.asarray(w.blocks[0]).dtype == np.complex128
    assert np.asarray(v.blocks[0]).dtype == np.complex128
    assert np.asarray(tenet.linalg.eigvals(t).blocks[0]).dtype == np.complex128

    use_jax()
    w, v = tenet.linalg.eig(to_jax(t))
    assert np.asarray(w.blocks[0]).dtype == np.complex128
    assert np.asarray(v.blocks[0]).dtype == np.complex128


def test_a_complex_input_round_trips_with_the_residual_criterion():
    t = square("su2", seed=30)
    t = SymmetricTensor(t.structure, tuple(b + 0.5j * b[::-1] for b in t.blocks))
    w, v = tenet.linalg.eig(t)
    assert float(tenet.norm(t @ v - v @ w)) <= 1e-12 * float(tenet.norm(t))


# --- eig / eigvals: jit and the gradient boundary ------------------------------------------------


def test_eigvals_jits_and_grads_while_eig_refuses_to_differentiate():
    """The whole boundary of #87 on one screen, next to ``svd_truncated``'s refusal."""
    jax = use_jax()
    traces = []

    def objective(x):
        return tenet.norm(tenet.linalg.eigvals(x))

    @jax.jit
    def jitted(x):
        traces.append(1)
        return objective(x)

    t = square("su2", seed=31)
    got = jitted(to_jax(t))
    jitted(to_jax(square("su2", seed=32)))
    assert len(traces) == 1  # same structure, same treedef: no retrace
    assert float(got) == pytest.approx(float(objective(t)), abs=1e-10)

    g = jax.grad(objective)(to_jax(t))
    assert all(np.isfinite(np.asarray(b)).all() for b in g.blocks)

    with pytest.raises(NotImplementedError) as excinfo:
        jax.grad(lambda x: tenet.norm(tenet.linalg.eig(x)[1]))(to_jax(t))
    assert "enable_eigvec_derivs" in str(excinfo.value)


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_eigvals_grad_matches_finite_differences(name):
    jax = use_jax()
    t = square(name, seed=33)

    def objective(x):
        return tenet.norm(tenet.linalg.eigvals(x))

    g = jax.grad(objective)(to_jax(t))
    indices = first_block_indices(t)
    np.testing.assert_allclose(
        np.asarray([np.asarray(g.blocks[0])[i] for i in indices]),
        central_differences(t, objective, indices),
        atol=1e-6,
    )


def jordan(like: SymmetricTensor) -> SymmetricTensor:
    """A per-sector exact Jordan block ``I + N``: defective, so the eigenvectors are
    deficient and the eigenvector derivative genuinely does not exist.

    ``I +`` rather than the bare nilpotent ``N``: at ``N`` every eigenvalue is zero,
    so the *objective* below is ``sqrt(0)`` and its own gradient is ``NaN`` — that
    would pin the norm's behaviour at zero, not ``eigvals``'.
    """
    mats = {}
    for c, b in matrices(like).items():
        rows = min(b.shape)
        block = np.eye(rows, dtype=float)
        block[np.arange(rows - 1), np.arange(1, rows)] = 1.0
        mats[c] = block
    return from_matrices(like.structure, mats)


def test_the_eigvals_gradient_stays_finite_on_defective_and_degenerate_blocks():
    """Finite is not correct here, and that is the point of the comment.

    At a defective matrix the *eigenvector* derivative genuinely does not exist —
    which is what JAX refuses in :func:`tenet.linalg.eig` — while the values-only
    JVP still returns a number. It is finite; whether it means anything is the
    caller's problem, and there is no gauge-invariant limit to broaden towards, so
    ``tenet.ad`` does not grow.
    """
    jax = use_jax()

    def objective(x):
        return tenet.norm(tenet.linalg.eigvals(x))

    t = square("su2", seed=34)
    for fixture in (jordan(t), with_domain_names(tenet.identity(t.codomain), t)):
        g = jax.grad(objective)(to_jax(fixture))
        assert all(np.isfinite(np.asarray(b)).all() for b in g.blocks)


def test_install_neither_registers_eig_nor_changes_its_results():
    use_jax()
    import autoray as ar

    import tenet.ad

    t = to_jax(square("su2", seed=35))
    before = tenet.linalg.eig(t)
    registered = {
        key: ar.autoray._FUNCS.get(key)
        for key in (("jax", "linalg.eig"), ("jax", "linalg.eigvals"))
    }
    tenet.ad.install()
    try:
        assert tenet.ad._NAMES == ("linalg.svd", "linalg.eigh")
        # install() rebinds exactly ``_NAMES``; autoray's own eig/eigvals entries are
        # left as they were shipped, which is the assertion the issue was after
        for key in (("jax", "linalg.eig"), ("jax", "linalg.eigvals")):
            assert ar.autoray._FUNCS.get(key) is registered[key]
        after = tenet.linalg.eig(t)
    finally:
        tenet.ad.uninstall()
    for x, y in zip(before, after, strict=True):
        for a, b in zip(x.blocks, y.blocks, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# --- module hygiene -------------------------------------------------------------------------


def test_the_new_functions_did_not_touch_the_closed_lists():
    import ast
    import pathlib

    import autoray as ar

    src = pathlib.Path(tenet.ops.linalg.__file__).read_text()
    assert "import numpy" not in src and "np." not in src
    assert "to_dense(" not in src and "import jax" not in src
    assert "if provider ==" not in src and "isinstance(provider" not in src
    for name in ("expm", "eig", "eigvals"):
        with pytest.raises(Exception):  # noqa: B017  the exception type is autoray's business
            ar.do(name, square("su2", seed=36))
        assert not hasattr(tenet, name)
    assert ast.parse(src) is not None
