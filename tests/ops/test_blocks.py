"""Elementwise maps over the reduced blocks — issue #93.

The load-bearing test in this file is the *boundary*: ``apply_blocks`` works in
coefficient space, and for a non-Abelian provider that is a different operation
from the dense elementwise one. Both directions are asserted — exact agreement for
the all-ones-CG providers, a **lower bound** on the disagreement for SU(2) — so a
change that accidentally made them agree fails the suite and gets read.
"""

import pathlib

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops import blocks as blocks_module
from tenet.symmetry import (
    SU2,
    U1,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

# --- fixtures -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stock_ad_dispatch():
    """Leave autoray's stock jax bindings behind: two tests here call ad.install(),
    and a leaked registration flips other modules' stock-JAX NaN assertions."""
    yield
    try:
        from tenet import ad
    except ImportError:
        return
    ad.uninstall()


# Four-leg cases with interleaved sides, so the (0, 1) | (2, 3) split of `svd`
# genuinely bends axes 1 and 2 (same shape as tests/ops/test_linalg.py).

SPLIT = ((0, 1), (2, 3))

T_TRIV = tuple(GradedSpace.new(Trivial, {TrivialSector(): m}) for m in (2, 3, 4, 3))
Q1 = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
Q2 = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})
V = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 3})
W = GradedSpace.new(SU2, {SU2Sector(1): 2, SU2Sector(2): 1})
X = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(2): 2})
F1 = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 3})
F2 = GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 1})

UU = ProductProvider((U1, U1))


def uu(a: int, b: int) -> ProductSector:
    return ProductSector((U1Sector(a), U1Sector(b)))


P1 = GradedSpace.new(UU, {uu(0, 0): 2, uu(1, 0): 2, uu(0, 1): 1})
P2 = GradedSpace.new(UU, {uu(0, 0): 2, uu(1, 0): 1})

LEGS = {
    "trivial": tuple(
        Leg(s, side, name=f"l{i}")
        for i, (s, side) in enumerate(zip(T_TRIV, (OUT, IN, OUT, IN), strict=True))
    ),
    "u1": (Leg(Q1, OUT, name="a"), Leg(Q2, IN, name="b"), Leg(Q2, OUT), Leg(Q1, IN)),
    "su2": (Leg(V, OUT, name="a"), Leg(W, IN, name="b"), Leg(X, OUT), Leg(V, IN)),
    "fz2": (Leg(F1, OUT, name="a"), Leg(F2, IN, name="b"), Leg(F2, OUT), Leg(F1, IN)),
    # A product provider forwards no BendingCoefficients (#40), so its case keeps
    # both OUT legs in the codomain and splits on the existing partition.
    "product": (Leg(P1, OUT, name="a"), Leg(P2, OUT), Leg(P2, IN), Leg(P1, IN)),
}
PROVIDERS = ["trivial", "u1", "su2", "fz2", "product"]
AXES = {name: (None if name == "product" else SPLIT) for name in PROVIDERS}


def tensor(name: str, seed: int = 0) -> SymmetricTensor:
    return SymmetricTensor.random(LEGS[name], seed=seed)


def positive(t: SymmetricTensor) -> SymmetricTensor:
    """Blocks made strictly positive, so ``sqrt`` and ``power`` are real everywhere."""
    return tenet.apply_blocks(t, lambda b: np.abs(b) + 0.5)


# The rank-3 pair the blockwise/dense boundary is measured on: two sectors per leg,
# legs (OUT, OUT, IN), positive blocks. Both dense images have 64 entries.
Q3 = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2})
V3 = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 1})
RANK3 = {
    "u1": (Leg(Q3, OUT), Leg(Q3, OUT), Leg(Q3, IN)),
    "su2": (Leg(V3, OUT), Leg(V3, OUT), Leg(V3, IN)),
}


def rank3(name: str, seed: int = 0) -> SymmetricTensor:
    return positive(SymmetricTensor.random(RANK3[name], seed=seed))


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  registration is the import's side effect

    return jax


# --- apply_blocks is the constructor, literally ---------------------------------


def test_apply_blocks_keeps_the_same_structure_object():
    t = tensor("su2")
    out = tenet.apply_blocks(t, lambda b: 2 * b)
    assert out.structure is t.structure
    assert out.legs == t.legs


def test_apply_blocks_with_identity_returns_the_identical_block_objects():
    t = tensor("su2")
    out = tenet.apply_blocks(t, lambda b: b)
    assert all(a is b for a, b in zip(out.blocks, t.blocks, strict=True))


def test_apply_blocks_is_set_params_of_get_params():
    """quimb's ``Tensor.apply_to_arrays``, over the protocol we already ship."""
    t = tensor("u1")
    fn = np.exp
    assert tenet.apply_blocks(t, fn) == t.set_params([fn(b) for b in t.get_params()])


def test_method_delegations():
    t = positive(tensor("su2"))
    assert t.sqrt() == tenet.sqrt(t)
    assert t.power(0.5) == tenet.power(t, 0.5)
    assert t.apply_blocks(np.exp) == tenet.apply_blocks(t, np.exp)


def test_exports():
    for name in ("apply_blocks", "sqrt", "power"):
        assert getattr(tenet, name) is getattr(tenet.ops, name)


# --- the motivating case: splitting S across the two factors --------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_svd_sqrt_round_trip(name):
    """``u @ sqrt(s) @ (sqrt(s) @ vh) == t`` — measured 3.3e-14 / 22.66 on SU(2)."""
    t = tensor(name)
    u, s, vh = tenet.linalg.svd(t, AXES[name])
    rs = tenet.sqrt(s)
    left, right = u @ rs, rs @ vh
    ref = tenet.linalg.svd(t, AXES[name])[0] @ s @ vh  # t in the svd's own partition
    assert float(tenet.norm(left @ right - ref)) / float(tenet.norm(ref)) < 1e-14


@pytest.mark.parametrize("name", PROVIDERS)
def test_sqrt_of_s_is_the_matrix_square_root(name):
    """``sqrt(S) @ sqrt(S) == S`` (measured 2.5e-15) — a fact about ``S``, not ``sqrt``.

    ``svd`` builds ``S``'s reduced blocks with ``ar.do("diag", sigma[:k])``, so each
    block is diagonal, and elementwise and matrix square roots coincide on a
    diagonal matrix. See the sibling test below for the general failure.
    """
    _, s, _ = tenet.linalg.svd(tensor(name), AXES[name])
    rs = tenet.sqrt(s)
    assert float(tenet.norm(rs @ rs - s)) < 1e-14


def test_sqrt_of_a_non_diagonal_tensor_is_not_the_matrix_square_root():
    """The coincidence above does not generalize; conflating the two is the trap."""
    m = positive(SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=3))
    rm = tenet.sqrt(m)
    assert float(tenet.norm(rm @ rm - m)) > 1.0


# --- the blockwise / dense boundary, in both directions -------------------------


@pytest.mark.parametrize("name", ["trivial", "u1", "fz2"])
def test_blockwise_sqrt_equals_dense_sqrt_for_all_ones_cg(name):
    """Trivial / U(1) / fZ2: all-ones CG and ``d_a == 1``, so the two coincide to 0.0."""
    legs = RANK3["u1"] if name == "u1" else LEGS[name][:3]
    t = positive(SymmetricTensor.random(legs, seed=0))
    dense = t.to_dense()
    assert dense.min() >= 0.0  # positive blocks + all-ones CG: no negative entries
    assert np.max(np.abs(tenet.sqrt(t).to_dense() - np.sqrt(dense))) == 0.0


def test_blockwise_sqrt_differs_from_dense_sqrt_for_su2():
    """SU(2): a *different operation*, off by a sizeable fraction of the array's scale.

    The issue's own measurement on this structure is ``1.673`` against a dense
    scale of ``3.82``; the block filling here is this file's, so the number below
    is ours. It is asserted as a **lower bound** on purpose: a future change that
    accidentally made blockwise and dense agree must fail here and be read
    deliberately, not silently pass a tightened tolerance.

    The dense ``sqrt`` is taken on the complex branch so that it is defined on the
    whole array: SU(2)'s Clebsch-Gordan coefficients are signed, so a rank-3 dense
    image has negative entries even for strictly positive blocks — which is itself
    part of why ``sqrt`` on the dense image is not the operation anyone wanted.
    """
    t = rank3("su2")
    dense = t.to_dense()
    assert dense.min() < 0.0
    diff = np.max(np.abs(tenet.sqrt(t).to_dense() - np.sqrt(dense.astype(np.complex128))))
    assert diff > 1.0  # measured 1.85 here, 1.673 on the issue's filling
    scale = np.max(np.abs(dense))  # 2.00 here, 3.82 on the issue's
    assert diff > 0.4 * scale  # a large fraction of the array's own scale, not noise


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_forbidden_dense_entries_stay_zero_even_when_f_of_zero_is_not_zero(name):
    """``exp`` blockwise is the worst case (``exp(0) == 1``) and sparsity survives it.

    Measured: 40 of 64 forbidden entries for the U(1) structure, 44 of 64 for the
    SU(2) one, max magnitude 0.0 in both.
    """
    t = rank3(name)
    # The forbidden mask, built without trusting any single filling: intersect the
    # zero sets of two independent random fillings of the same structure.
    a = SymmetricTensor.random(RANK3[name], seed=1).to_dense()
    b = SymmetricTensor.random(RANK3[name], seed=2).to_dense()
    forbidden = (a == 0.0) & (b == 0.0)
    assert forbidden.sum() == {"u1": 40, "su2": 44}[name]
    assert forbidden.size == 64
    assert np.max(np.abs(tenet.apply_blocks(t, np.exp).to_dense()[forbidden])) == 0.0


# --- power ----------------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_power_half_agrees_with_sqrt(name):
    t = positive(tensor(name))
    for a, b in zip(tenet.power(t, 0.5).blocks, tenet.sqrt(t).blocks, strict=True):
        assert np.allclose(a, b, rtol=0, atol=1e-15)


@pytest.mark.parametrize("name", PROVIDERS)
def test_power_one_is_the_identity(name):
    t = tensor(name)
    assert tenet.power(t, 1) == t


def test_power_minus_one_is_the_reciprocal_per_block():
    t = positive(tensor("su2"))
    for a, b in zip(tenet.power(t, -1).blocks, t.blocks, strict=True):
        assert np.allclose(a, 1 / b, rtol=0, atol=1e-15)


# --- dispatch stays closed ------------------------------------------------------


@pytest.mark.parametrize("call", [lambda t: ar.do("sqrt", t), lambda t: ar.do("power", t, 0.5)])
def test_ar_do_sqrt_and_power_still_raise(call):
    with pytest.raises(ValueError, match="not defined for a symmetric tensor"):
        call(tensor("su2"))


# --- backends -------------------------------------------------------------------


def test_jax_blocks_stay_jax():
    use_jax()
    t = positive(tensor("su2")).to_backend("jax")
    out = tenet.sqrt(t)
    assert ar.infer_backend(out.blocks[0]) == "jax"
    ref = tenet.sqrt(t.to_backend("numpy"))
    assert np.allclose(np.asarray(out.blocks[0]), ref.blocks[0])


def test_jit_traces_once_per_structure():
    jax = use_jax()
    count = 0

    @jax.jit
    def f(t):
        nonlocal count
        count += 1
        return tenet.norm(tenet.sqrt(t))

    a = positive(tensor("su2", 4)).to_backend("jax")
    b = positive(tensor("su2", 5)).to_backend("jax")
    ra, rb = f(a), f(b)
    assert count == 1
    assert not np.isclose(float(ra), float(rb))
    f(positive(tensor("u1", 6)).to_backend("jax"))  # different structure: retrace
    assert count == 2
    f(positive(tensor("su2", 7)).to_backend("jax"))
    assert count == 2


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_grad_through_svd_and_sqrt_matches_finite_differences(name):
    jax = use_jax()
    from tenet import ad

    ad.install()
    t = tensor(name, seed=8).to_backend("jax")
    structure = t.structure

    def f(params):
        u, s, _ = tenet.linalg.svd(SymmetricTensor(structure, tuple(params)), AXES[name])
        return tenet.norm(u @ tenet.sqrt(s))

    grads = jax.grad(f)(list(t.blocks))
    assert all(bool(np.all(np.isfinite(np.asarray(g)))) for g in grads)
    jitted = jax.jit(jax.grad(f))(list(t.blocks))
    for g, h in zip(grads, jitted, strict=True):
        assert np.allclose(np.asarray(g), np.asarray(h), rtol=1e-10, atol=1e-12)

    # central differences on a handful of entries, float64 (conftest enables x64)
    h = 1e-6
    for i in (0, len(grads) - 1):
        idx = tuple(0 for _ in t.blocks[i].shape)
        bumped = list(np.asarray(b, dtype=np.float64) for b in t.blocks)
        bumped[i] = bumped[i].copy()
        bumped[i][idx] += h
        plus = float(f([np.asarray(b) for b in bumped]))
        bumped[i][idx] -= 2 * h
        minus = float(f([np.asarray(b) for b in bumped]))
        assert (plus - minus) / (2 * h) == pytest.approx(float(np.asarray(grads[i])[idx]), abs=1e-6)


# --- dtypes and the backend's own nan -------------------------------------------


def test_complex_sqrt_and_power_follow_the_backend_branch_cut():
    t = tensor("su2")
    c = tenet.apply_blocks(t, lambda b: b.astype(np.complex128) + 1j)
    out = tenet.sqrt(c)
    assert out.dtype == np.complex128
    for a, b in zip(out.blocks, c.blocks, strict=True):
        assert np.array_equal(a, ar.do("sqrt", b))
    for a, b in zip(tenet.power(c, 0.5).blocks, c.blocks, strict=True):
        assert np.array_equal(a, ar.do("power", b, 0.5))


def test_negative_entries_give_the_backend_nan_before_and_after_ad_install():
    t = positive(tensor("su2"))
    blocks = [np.array(b, copy=True) for b in t.blocks]
    blocks[0] = blocks[0].copy()
    blocks[0].flat[0] = -1.0
    t = t.set_params(blocks)

    def check():
        out = tenet.sqrt(t)
        assert np.isnan(np.asarray(out.blocks[0]).flat[0])
        assert np.isfinite(np.asarray(out.blocks[0]).flat[1:]).all()
        assert all(np.isfinite(np.asarray(b)).all() for b in out.blocks[1:])

    check()
    pytest.importorskip("jax")
    from tenet import ad

    ad.install()
    check()


# --- module hygiene -------------------------------------------------------------


def test_module_source_has_no_backend_import_and_no_data_dependent_branch():
    src = pathlib.Path(blocks_module.__file__).read_text()
    for banned in ("import jax", "import torch", "import numpy"):
        assert banned not in src
    for banned in (".any()", ".all()", "bool(", "functools.cache", "try:"):
        assert banned not in src
