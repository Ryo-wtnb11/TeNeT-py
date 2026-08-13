"""Tests for the map view, composition and identity — issue #30."""

import dataclasses
import pathlib

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, ProductSpace, SymmetricTensor, TensorMapView, as_map
from tenet.map_view import map_layout, to_matrices
from tenet.symmetry import SU2, U1, SU2Sector, Trivial, TrivialSector, U1Sector

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 2})
W = GradedSpace.new(SU2, {HALF: 2, ONE: 1})
U = GradedSpace.new(SU2, {ZERO: 1, HALF: 2})
X = GradedSpace.new(SU2, {ZERO: 2, ONE: 1})
TRIV = GradedSpace.new(Trivial, {TrivialSector(): 3})
TRIV2 = GradedSpace.new(Trivial, {TrivialSector(): 2})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
P = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2})

# a: interleaved sides, so the map view has to be a view and not a reordering
A_LEGS = (Leg(V, OUT), Leg(W, IN), Leg(U, OUT), Leg(X, IN))
B_LEGS = (Leg(W, OUT), Leg(U, IN), Leg(X, OUT))
U1_A_LEGS = (Leg(Q, OUT), Leg(P, IN), Leg(Q, IN))
U1_B_LEGS = (Leg(P, OUT), Leg(Q, OUT), Leg(P, IN))
TRIV_A_LEGS = (Leg(TRIV, OUT), Leg(TRIV2, IN))
TRIV_B_LEGS = (Leg(TRIV2, OUT), Leg(TRIV, IN))
OP_A_LEGS = (Leg(V, OUT), Leg(W, IN))
OP_B_LEGS = (Leg(W, OUT), Leg(U, IN))

PAIRS = [
    pytest.param(A_LEGS, B_LEGS, id="su2"),
    pytest.param(U1_A_LEGS, U1_B_LEGS, id="u1"),
    pytest.param(TRIV_A_LEGS, TRIV_B_LEGS, id="trivial"),
    pytest.param(OP_A_LEGS, OP_B_LEGS, id="su2-operator"),
]


def use_jax():
    jax = pytest.importorskip("jax")
    # ponytail: global x64, same as tests/test_map_view.py — float32 would make
    # the tolerances below meaningless.
    jax.config.update("jax_enable_x64", True)
    return jax


def pair(a_legs=A_LEGS, b_legs=B_LEGS, seeds=(0, 1)):
    return (
        SymmetricTensor.random(a_legs, seed=seeds[0]),
        SymmetricTensor.random(b_legs, seed=seeds[1]),
    )


def dense_compose(a, b):
    """The oracle: contract a's IN physical axes against b's OUT physical axes."""
    return np.tensordot(
        a.to_dense(), b.to_dense(), axes=(a.structure.in_axes, b.structure.out_axes)
    )


# --- the view is a view ---------------------------------------------------------


def test_as_map_moves_nothing(monkeypatch):
    t = SymmetricTensor.random(A_LEGS, seed=0)

    def boom(*args, **kw):
        raise AssertionError("as_map called into the backend")

    monkeypatch.setattr(ar, "do", boom)
    m = t.as_map()
    assert m.tensor is t
    assert as_map(t).tensor is t
    assert all(x is y for x, y in zip(m.tensor.blocks, t.blocks, strict=True))
    assert m.codomain.legs and m.domain.legs  # properties are derived, not computed


def test_view_sides_follow_public_axis_order_through_the_interleaving():
    t = SymmetricTensor.random(A_LEGS, seed=0)
    m = t.as_map()
    assert [leg.side for leg in t.legs] == [OUT, IN, OUT, IN]
    assert m.codomain.legs == t.codomain == (A_LEGS[0], A_LEGS[2])
    assert m.domain.legs == t.domain == (A_LEGS[1], A_LEGS[3])


def test_codomain_domain_on_the_tensor_still_return_legs():
    t = SymmetricTensor.random(A_LEGS, seed=0)
    assert isinstance(t.codomain, tuple) and isinstance(t.codomain[0], Leg)
    assert isinstance(t.as_map().codomain, ProductSpace)


def test_view_matrices_are_to_matrices():
    t = SymmetricTensor.random(A_LEGS, seed=0)
    got, want = t.as_map().matrices(), to_matrices(t)
    assert set(got) == set(want)
    for c in got:
        np.testing.assert_array_equal(got[c], want[c])


# --- ProductSpace ---------------------------------------------------------------


def test_product_space_is_frozen_hashable_and_array_free():
    ps = ProductSpace((Leg(V, OUT), Leg(U, OUT)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ps.legs = ()
    assert hash(ps) == hash(ProductSpace((Leg(V, OUT), Leg(U, OUT))))
    for field in dataclasses.fields(ps):
        assert not hasattr(getattr(ps, field.name), "shape")


def test_product_space_equality_is_ordered_leg_equality():
    x, y = Leg(V, OUT), Leg(U, OUT)
    assert ProductSpace((x, y)) == ProductSpace((x, y))
    assert ProductSpace((x, y)) != ProductSpace((y, x))
    assert ProductSpace((x, y)) != ProductSpace((x,))


def test_product_space_dims():
    ps = ProductSpace((Leg(V, OUT), Leg(W, IN)))
    assert ps.reduced_dim == V.reduced_dim * W.reduced_dim
    assert ps.dim == V.dim * W.dim
    assert ProductSpace(()).dim == 1 and ProductSpace(()).reduced_dim == 1
    assert ps.provider is SU2


def test_product_space_matches():
    assert ProductSpace((Leg(V, IN),)).matches(ProductSpace((Leg(V, OUT),))) is None
    assert ProductSpace((Leg(V, IN),)).matches(ProductSpace((Leg(V, OUT, name="x"),))) is None
    assert ProductSpace((Leg(V, IN),)).matches(ProductSpace((Leg(U, OUT),))) == 0
    assert ProductSpace((Leg(V, IN),)).matches(ProductSpace((Leg(V, OUT, dual=True),))) == 0
    assert ProductSpace((Leg(V, IN), Leg(U, IN))).matches(ProductSpace((Leg(V, OUT),))) == 1


# --- composition, against the dense oracle --------------------------------------


def test_su2_composition_matches_the_dense_oracle():
    a, b = pair()
    shared = set(map_layout(a.structure).sectors) & set(map_layout(b.structure).sectors)
    assert len({SU2.qdim(c) for c in shared}) >= 2  # genuinely non-Abelian

    c = a @ b
    assert c.legs == (*a.codomain, *b.domain)
    np.testing.assert_allclose(c.to_dense(), dense_compose(a, b), atol=1e-10)


@pytest.mark.parametrize(("a_legs", "b_legs"), PAIRS)
def test_composition_matches_the_dense_oracle(a_legs, b_legs):
    a, b = pair(a_legs, b_legs, seeds=(2, 3))
    c = a @ b
    assert c.legs == (*a.codomain, *b.domain)
    np.testing.assert_allclose(c.to_dense(), dense_compose(a, b), atol=1e-10)


def test_operator_case_is_plain_matrix_multiplication():
    a, b = pair(OP_A_LEGS, OP_B_LEGS, seeds=(4, 5))
    np.testing.assert_allclose((a @ b).to_dense(), a.to_dense() @ b.to_dense(), atol=1e-10)


def test_the_three_spellings_agree():
    a, b = pair()
    assert a @ b == tenet.compose(a, b)
    assert a @ b == a.as_map().compose(b)
    assert a @ b == a.as_map().compose(b.as_map())


def test_associativity():
    a, b = pair()
    c = SymmetricTensor.random((Leg(U, OUT), Leg(V, IN)), seed=6)
    assert tenet.allclose((a @ b) @ c, a @ (b @ c))


def test_linearity():
    a, b = pair()
    b2 = SymmetricTensor.random(B_LEGS, seed=7)
    assert tenet.allclose(a @ (b + b2), (a @ b) + (a @ b2), atol=1e-12)
    assert tenet.allclose((2.5 * a) @ b, 2.5 * (a @ b), atol=1e-12)


def test_a_sector_on_only_one_side_gives_zero_blocks_not_a_key_error():
    shared = GradedSpace.new(SU2, {HALF: 2})
    va = GradedSpace.new(SU2, {ZERO: 1, HALF: 1})
    a = SymmetricTensor.random((Leg(va, OUT), Leg(shared, IN)), seed=8)
    b = SymmetricTensor.random((Leg(shared, OUT), Leg(va, IN)), seed=9)
    assert ZERO not in map_layout(a.structure).sectors
    c = a @ b
    assert ZERO in map_layout(c.structure).sectors  # declared by the result structure
    for key, block in c.items():
        if key.coupled == ZERO:
            assert not np.any(block)
    np.testing.assert_allclose(c.to_dense(), dense_compose(a, b), atol=1e-10)


# --- compatibility refusals -----------------------------------------------------


def refuse(a, b):
    with pytest.raises(ValueError) as e:
        a @ b
    return str(e.value)


def test_refuses_a_different_space():
    a, _ = pair()
    b = SymmetricTensor.random((Leg(V, OUT), Leg(U, IN), Leg(X, OUT)), seed=1)
    message = refuse(a, b)
    assert "axis 1 of a" in message and "axis 0 of b" in message
    assert repr(a.legs[1]) in message and repr(b.legs[0]) in message


def test_refuses_a_different_dual_flag():
    a, _ = pair()
    b = SymmetricTensor.random((Leg(W, OUT, dual=True), Leg(U, IN), Leg(X, OUT)), seed=1)
    message = refuse(a, b)
    assert "axis 1 of a" in message and "axis 0 of b" in message
    assert "dual=True" in message


def test_refuses_the_same_legs_in_another_order_and_names_transpose():
    a, _ = pair()
    b = SymmetricTensor.random((Leg(X, OUT), Leg(U, IN), Leg(W, OUT)), seed=1)
    # same multiset of (space, dual) on the shared side, only the order differs
    assert {leg.space for leg in a.domain} == {leg.space for leg in b.codomain}
    message = refuse(a, b)
    assert "position 0" in message and "transpose" in message


def test_refuses_a_wrong_leg_count():
    a, _ = pair()
    b = SymmetricTensor.random((Leg(W, OUT), Leg(U, IN)), seed=1)
    message = refuse(a, b)
    assert "2 IN legs" in message and "1 OUT" in message


def test_refuses_a_different_provider_and_names_both():
    a, _ = pair(OP_A_LEGS, OP_B_LEGS)
    b = SymmetricTensor.random((Leg(Q, OUT), Leg(P, IN)), seed=1)
    message = refuse(a, b)
    assert "SU2" in message and "U1" in message


def test_no_refusal_case_silently_returns_a_tensor():
    a, _ = pair()
    bad = [
        (Leg(V, OUT), Leg(U, IN), Leg(X, OUT)),
        (Leg(W, OUT, dual=True), Leg(U, IN), Leg(X, OUT)),
        (Leg(X, OUT), Leg(U, IN), Leg(W, OUT)),
        (Leg(W, OUT), Leg(U, IN)),
    ]
    for legs in bad:
        b = SymmetricTensor.random(legs, seed=1)
        with pytest.raises(ValueError):
            a @ b


def test_a_name_difference_composes_fine():
    a = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN, name="bond")), seed=0)
    b = SymmetricTensor.random((Leg(W, OUT, name="other"), Leg(U, IN)), seed=1)
    np.testing.assert_allclose((a @ b).to_dense(), dense_compose(a, b), atol=1e-10)


# --- identity -------------------------------------------------------------------


def test_identity_axis_order_and_matrices():
    legs = (Leg(V, OUT), Leg(W, OUT))
    i = tenet.identity(legs)
    assert [leg.side for leg in i.legs] == [OUT, OUT, IN, IN]
    assert [leg.space for leg in i.legs] == [V, W, V, W]
    layout = map_layout(i.structure)
    assert len({SU2.qdim(c) for c in layout.sectors}) >= 2
    for c, m in to_matrices(i).items():
        n = layout.shape(c)[0]
        np.testing.assert_array_equal(m, np.eye(n))


def test_identity_mirrors_dual_rather_than_dualizing():
    i = tenet.identity((Leg(V, OUT, dual=True),))
    assert [leg.dual for leg in i.legs] == [True, True]


def test_identity_is_neutral_for_composition_block_for_block():
    legs = (Leg(V, OUT), Leg(U, OUT), Leg(W, IN))  # already map-ordered
    t = SymmetricTensor.random(legs, seed=10)
    left = tenet.identity(t.codomain) @ t
    right = t @ tenet.identity(t.domain)
    assert left.structure == t.structure and right.structure == t.structure
    for x, y, z in zip(left.blocks, right.blocks, t.blocks, strict=True):
        np.testing.assert_allclose(x, z, atol=1e-12)
        np.testing.assert_allclose(y, z, atol=1e-12)


def test_identity_composed_with_itself_is_itself():
    i = tenet.identity((Leg(V, OUT), Leg(W, OUT)))
    assert tenet.allclose(i @ i, i)


def test_identity_is_dense_identity():
    legs = (Leg(V, OUT), Leg(W, OUT))
    i = tenet.identity(legs)
    n = V.dim * W.dim
    dense = i.to_dense().reshape(n, n)
    np.testing.assert_allclose(dense, np.eye(n), atol=1e-12)


def test_identity_dtype():
    i = tenet.identity((Leg(V, OUT),), dtype=np.complex128)
    assert i.dtype == np.complex128


# --- refusals that are ergonomics, not math -------------------------------------


def test_svd_names_milestone_7_and_to_matrices():
    t = SymmetricTensor.random(A_LEGS, seed=0)
    with pytest.raises(NotImplementedError) as e:
        t.as_map().svd()
    assert "Milestone 7" in str(e.value) and "to_matrices" in str(e.value)


def test_view_adjoint_is_the_tensor_adjoint():
    # #31: the view's adjoint is tenet.adjoint of the underlying tensor.
    t = SymmetricTensor.random(A_LEGS, seed=0)
    assert t.as_map().adjoint() == t.adjoint()


def test_matmul_with_a_scalar_and_mul_with_a_tensor_both_explain_themselves():
    a, b = pair()
    with pytest.raises(TypeError) as e:
        a @ 3
    assert "composes" in str(e.value) and "t * s" in str(e.value)

    with pytest.raises(TypeError) as e:
        a * b
    assert "tensordot" in str(e.value) and "@" in str(e.value)


# --- backend agnosticism --------------------------------------------------------


def test_jax_composition_matches_numpy():
    use_jax()
    a, b = pair()
    want = a @ b
    got = a.to_backend("jax") @ b.to_backend("jax")
    assert got.backend == "jax"
    assert tenet.allclose(got, want, atol=1e-12)


def test_jax_composition_matches_the_dense_oracle():
    use_jax()
    a, b = pair()
    got = a.to_backend("jax") @ b.to_backend("jax")
    numpy_back = SymmetricTensor(got.structure, tuple(np.asarray(x) for x in got.blocks))
    np.testing.assert_allclose(numpy_back.to_dense(), dense_compose(a, b), atol=1e-10)


def test_jax_identity_composition():
    use_jax()
    legs = (Leg(V, OUT), Leg(U, OUT), Leg(W, IN))
    t = SymmetricTensor.random(legs, seed=10).to_backend("jax")
    i = tenet.identity(t.codomain).to_backend("jax")
    assert tenet.allclose(i @ t, t, atol=1e-12)


# --- module hygiene -------------------------------------------------------------


def test_ops_map_has_no_dense_expansion_and_no_provider_branching():
    src = pathlib.Path(tenet.ops.map.__file__).read_text()
    assert "to_dense(" not in src
    assert "if provider ==" not in src
    assert "isinstance(provider" not in src
    # numpy appears exactly once, as identity's default dtype
    assert src.count("np.") == 1 and "np.float64" in src


def test_view_and_helpers_are_exported():
    for name in ("ProductSpace", "TensorMapView", "as_map", "compose", "identity"):
        assert name in tenet.__all__
        assert getattr(tenet, name) is not None
    assert isinstance(as_map(SymmetricTensor.random(A_LEGS, seed=0)), TensorMapView)
