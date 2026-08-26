"""Tests for tenet.tensordot / tenet.trace — one test per acceptance criterion of #51.

The oracle is ``np.tensordot`` of the dense expansions, with **no ``z_matrix``
insertions**: ``to_dense`` already contracts ``Z`` onto every dual axis (#37) and
the bending coefficient is that same isomorphism in the fusion basis (#38), so
the two cancel. The one provider that needs more is fZ2, where the oracle is the
dense contraction of the *graded-transposed* operands, the sign coming from the
shared :func:`helpers.supersign` (#39) and never from this module's own idea of
what the sign should be.
"""

import ast
import itertools
import pathlib

import autoray as ar
import numpy as np
import pytest
from helpers import NoBendProvider, supersign

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.contraction import contractible, outward_dual
from tenet.ops.map import compose
from tenet.ops.repartition import repartition
from tenet.symmetry import (
    SU2,
    U1,
    CapabilityError,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

# --- spaces ----------------------------------------------------------------------

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 2, ONE: 1})  # three sectors, qdim 1/2/3
W = GradedSpace.new(SU2, {ZERO: 1, HALF: 2})
U = GradedSpace.new(SU2, {HALF: 1, ONE: 2})

Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
P = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2})
# same dimension as P, opposite charges: the "dimensions are never compared" trap
P_REVERSED = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(-1): 2})

T3 = GradedSpace.new(Trivial, {TrivialSector(): 3})
T2 = GradedSpace.new(Trivial, {TrivialSector(): 2})

EVEN, ODD = FZ2Sector(0), FZ2Sector(1)
FP = GradedSpace.new(fZ2, {EVEN: 2, ODD: 3})
FQ = GradedSpace.new(fZ2, {EVEN: 1, ODD: 2})
FR = GradedSpace.new(fZ2, {EVEN: 2, ODD: 1})
FS = GradedSpace.new(fZ2, {EVEN: 1, ODD: 1})

UF = ProductProvider((U1, fZ2))


def uf(charge: int, parity: int) -> ProductSector:
    return ProductSector((U1Sector(charge), FZ2Sector(parity)))


S1 = GradedSpace.new(UF, {uf(0, 0): 2, uf(1, 1): 1, uf(-1, 1): 1})
S2 = GradedSpace.new(UF, {uf(0, 0): 1, uf(1, 1): 2})

# A provider with the product's sectors and *no* bending: #312 forwarded
# ``BendingCoefficients`` through products, so ``UF`` no longer refuses a bend and the
# refusal cases below need a vehicle that still does. ``helpers.NoBendProvider`` withholds
# exactly that capability and delegates the rest, so the layout, the fZ2 signs and the
# sector pattern are unchanged -- only the bend is out of contract.
NB = NoBendProvider(UF)
NB_S1 = GradedSpace.new(NB, {uf(0, 0): 2, uf(1, 1): 1, uf(-1, 1): 1})
NB_S2 = GradedSpace.new(NB, {uf(0, 0): 1, uf(1, 1): 2})


def pair(a_legs, b_legs, seeds=(0, 1)):
    return (
        SymmetricTensor.random(a_legs, seed=seeds[0]),
        SymmetricTensor.random(b_legs, seed=seeds[1]),
    )


def free_legs(a, b, axes):
    return tuple(a.legs[i] for i in range(a.ndim) if i not in axes[0]) + tuple(
        b.legs[j] for j in range(b.ndim) if j not in axes[1]
    )


def check_oracle(a_legs, b_legs, axes, seeds=(0, 1), atol=1e-10):
    """``tensordot`` against ``np.tensordot`` of the dense expansions."""
    a, b = pair(a_legs, b_legs, seeds)
    c = tenet.tensordot(a, b, axes)
    assert c.legs == free_legs(a, b, axes)
    np.testing.assert_allclose(
        c.to_dense(), np.tensordot(a.to_dense(), b.to_dense(), axes), atol=atol
    )
    return c


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  registration is the import's side effect

    return jax


def to_numpy(t: SymmetricTensor) -> SymmetricTensor:
    return SymmetricTensor(t.structure, tuple(np.asarray(x) for x in t.blocks))


# --- outward_dual and contractible -------------------------------------------------

COMBOS = tuple(itertools.product((OUT, IN), (False, True)))


def test_outward_dual_over_every_side_dual_combination():
    assert outward_dual(Leg(V, OUT, dual=False)) is False  # V, read outwards
    assert outward_dual(Leg(V, OUT, dual=True)) is True
    assert outward_dual(Leg(V, IN, dual=False)) is True  # an IN leg contributes X*
    assert outward_dual(Leg(V, IN, dual=True)) is False


@pytest.mark.parametrize("x_combo", COMBOS)
@pytest.mark.parametrize("y_combo", COMBOS)
def test_contractible_is_symmetric_and_pairs_opposite_outward_duals(x_combo, y_combo):
    x, y = Leg(V, *x_combo), Leg(V, *y_combo)
    assert contractible(x, y) == contractible(y, x)
    assert contractible(x, y) == (outward_dual(x) != outward_dual(y))


@pytest.mark.parametrize("x_combo", COMBOS)
@pytest.mark.parametrize("y_combo", COMBOS)
def test_contractible_is_false_whenever_the_spaces_differ(x_combo, y_combo):
    assert not contractible(Leg(V, *x_combo), Leg(W, *y_combo))
    # equal dimension, opposite charges: invariant 2, never a dimension comparison
    assert P.dim == P_REVERSED.dim
    assert not contractible(Leg(P, *x_combo), Leg(P_REVERSED, *y_combo))


@pytest.mark.parametrize("y_combo", COMBOS)
def test_contractible_is_invariant_under_bending_either_leg(y_combo):
    """Not asserted from the formula — asserted by actually calling ``repartition``."""
    y = Leg(V, *y_combo)
    for axis, legs in ((1, (Leg(W, OUT), Leg(V, IN))), (1, (Leg(W, OUT), Leg(V, IN, dual=True)))):
        t = SymmetricTensor.random(legs, seed=0)
        bent = repartition(t, (0, 1), ())  # axis 1 crosses into the codomain
        assert bent.legs[axis].side is not t.legs[axis].side
        assert bent.legs[axis].dual is not t.legs[axis].dual
        assert contractible(t.legs[axis], y) == contractible(bent.legs[axis], y)


# --- output structure ---------------------------------------------------------------


def test_output_legs_are_the_free_legs_unchanged_including_name():
    """Both operands have free legs on both sides, so both repartitions do real work
    in both directions and the round trip is what is under test."""
    a_legs = (Leg(V, OUT, name="a0"), Leg(W, OUT, name="a1"), Leg(U, IN, name="a2"))
    b_legs = (Leg(W, IN, name="b0"), Leg(U, OUT, name="b1"), Leg(V, IN, name="b2"))
    a, b = pair(a_legs, b_legs)
    c = tenet.tensordot(a, b, ((1,), (0,)))
    assert c.legs == (a_legs[0], a_legs[2], b_legs[1], b_legs[2])
    assert [leg.name for leg in c.legs] == ["a0", "a2", "b1", "b2"]
    assert [leg.side for leg in c.legs] == [OUT, IN, OUT, IN]


# --- dense oracles ------------------------------------------------------------------


def test_dense_oracle_trivial():
    check_oracle(
        (Leg(T3, OUT), Leg(T2, IN)),
        (Leg(T2, OUT), Leg(T3, IN), Leg(T2, OUT)),
        ((1,), (0,)),
    )


U1_CASES = [
    pytest.param(
        (Leg(Q, OUT), Leg(P, IN, dual=True)),
        (Leg(P, OUT, dual=True), Leg(Q, IN)),
        ((1,), (0,)),
        id="composition-shaped",
    ),
    pytest.param(
        (Leg(Q, OUT), Leg(P, OUT, dual=True), Leg(Q, IN)),
        (Leg(P, IN, dual=True), Leg(Q, OUT)),
        ((1,), (0,)),
        id="bends-the-contracted-legs",
    ),
    pytest.param(
        (Leg(Q, OUT), Leg(P, IN), Leg(Q, IN, dual=True)),
        (Leg(P, IN), Leg(Q, OUT, dual=True)),
        ((2,), (1,)),
        id="bends-free-legs",
    ),
]


@pytest.mark.parametrize(("a_legs", "b_legs", "axes"), U1_CASES)
def test_dense_oracle_u1(a_legs, b_legs, axes):
    assert any(leg.dual for leg in a_legs) and any(leg.dual for leg in b_legs)
    check_oracle(a_legs, b_legs, axes)


SU2_CASES = [
    pytest.param(
        (Leg(V, OUT), Leg(W, IN), Leg(U, IN)),
        (Leg(U, OUT), Leg(W, OUT), Leg(V, IN)),
        ((2, 1), (0, 1)),
        id="within-side-transpose",
    ),
    pytest.param(
        (Leg(V, OUT), Leg(W, OUT), Leg(U, IN)),
        (Leg(W, IN), Leg(U, OUT), Leg(V, IN)),
        ((1,), (0,)),
        id="bends-on-both-operands",
    ),
    pytest.param(
        (Leg(V, OUT), Leg(W, OUT)),
        (Leg(W, OUT, dual=True), Leg(U, IN)),
        ((1,), (0,)),
        id="out-against-out-dual",
    ),
]


@pytest.mark.parametrize(("a_legs", "b_legs", "axes"), SU2_CASES)
def test_dense_oracle_su2(a_legs, b_legs, axes):
    """SU(2) carries three sectors of differing quantum dimension on ``V``."""
    assert len({SU2.qdim(s) for s in V}) == 3
    check_oracle(a_legs, b_legs, axes)


# fZ2: the contracted axes are paired in the reversed order, so the lowering
# transposes two IN lines past each other — a genuine Koszul sign.
FZ2_A_LEGS = (Leg(FP, OUT), Leg(FQ, IN), Leg(FR, IN))
FZ2_B_LEGS = (Leg(FR, OUT), Leg(FQ, OUT), Leg(FS, IN))
FZ2_AXES = ((2, 1), (0, 1))
FZ2_PERM = (0, 2, 1)


def fz2_signed_oracle(a, b):
    """``np.tensordot`` of the *graded*-transposed operands: the sign is
    :func:`helpers.supersign`'s, computed from the axis pattern alone."""
    sign = supersign(FZ2_A_LEGS, FZ2_PERM, per_side=True)
    return sign, np.tensordot(
        sign * np.transpose(a.to_dense(), FZ2_PERM), b.to_dense(), axes=((1, 2), (0, 1))
    )


def test_dense_oracle_fz2_with_a_nontrivial_supersign():
    a, b = pair(FZ2_A_LEGS, FZ2_B_LEGS, seeds=(3, 4))
    sign, want = fz2_signed_oracle(a, b)
    assert np.any(sign != 1.0), "a test whose sign is identically +1 proves nothing"
    np.testing.assert_allclose(tenet.tensordot(a, b, FZ2_AXES).to_dense(), want, atol=1e-12)


def test_plain_tensordot_disagrees_for_fz2_so_the_sign_has_teeth():
    a, b = pair(FZ2_A_LEGS, FZ2_B_LEGS, seeds=(3, 4))
    got = tenet.tensordot(a, b, FZ2_AXES).to_dense()
    plain = np.tensordot(a.to_dense(), b.to_dense(), FZ2_AXES)
    assert not np.allclose(got, plain, atol=1e-8)


def test_dense_oracle_product_provider_composition_shaped():
    check_oracle((Leg(S1, OUT), Leg(S2, IN)), (Leg(S2, OUT), Leg(S1, IN)), ((1,), (0,)))


def test_a_provider_without_bending_refuses_and_names_the_way_out():
    a, b = pair((Leg(NB_S1, OUT), Leg(NB_S2, IN)), (Leg(NB_S2, IN, dual=True), Leg(NB_S1, OUT)))
    with pytest.raises(CapabilityError) as excinfo:
        tenet.tensordot(a, b, ((1,), (0,)))
    message = str(excinfo.value)
    assert "BendingCoefficients" in message
    assert "axis 0 of b" in message  # the caller's numbering
    assert "composition-shaped" in message
    assert repr((a.structure.in_axes, b.structure.out_axes)) in message


# --- negative companions ------------------------------------------------------------


NEGATIVE_CASE = (
    (Leg(V, OUT), Leg(W, OUT), Leg(U, IN)),
    (Leg(W, IN), Leg(U, OUT), Leg(V, IN)),
    ((1,), (0,)),
)


def test_dropping_the_final_side_restoration_returns_the_wrong_legs():
    """ "Correctly shaped" is not "correct": the composition leaves ``a``'s free IN
    leg in the codomain with its ``dual`` flipped, so the output legs would depend
    on the lowering and ``tensordot`` would not compose with itself predictably.

    Note what this test does *not* claim. Stopping here does not break the dense
    oracle, and it cannot: a bend is invisible to ``to_dense`` for exactly the
    reason #38 recorded (``to_dense`` contracts ``Z`` onto every dual axis and the
    bending coefficient is that same isomorphism, so the two cancel). The damage
    is entirely in the metadata, which is why the issue's promise is about legs
    and why the two companions below carry the numerical half.
    """
    a_legs, b_legs, axes = NEGATIVE_CASE
    a, b = pair(a_legs, b_legs)
    truncated = compose(repartition(a, (0, 2), (1,)), repartition(b, (0,), (1, 2)))
    want_legs = free_legs(a, b, axes)
    assert truncated.legs != want_legs
    assert [leg.side for leg in truncated.legs] == [OUT, OUT, IN, IN]
    assert [leg.dual for leg in truncated.legs] != [leg.dual for leg in want_legs]
    assert tenet.tensordot(a, b, axes).legs == want_legs


def test_dropping_the_final_transpose_breaks_the_oracle():
    """The side restoration delivers ``(*outputs, *inputs)``; the transpose that
    puts the free legs back in ``(a free..., b free...)`` order is a separate step,
    and skipping it is wrong at the same shape."""
    a_legs, b_legs, axes = NEGATIVE_CASE
    a, b = pair(a_legs, b_legs)
    c = compose(repartition(a, (0, 2), (1,)), repartition(b, (0,), (1, 2)))
    untransposed = repartition(c, (0, 2), (1, 3))  # legs (a0, b1, a2, b2)
    want = np.tensordot(a.to_dense(), b.to_dense(), axes)
    assert untransposed.to_dense().shape == want.shape
    assert not np.allclose(untransposed.to_dense(), want, atol=1e-8)
    np.testing.assert_allclose(tenet.tensordot(a, b, axes).to_dense(), want, atol=1e-10)


def test_pairing_the_contracted_axes_in_the_wrong_order_breaks_the_oracle():
    a_legs = (Leg(V, OUT), Leg(W, IN), Leg(W, IN))
    b_legs = (Leg(W, OUT), Leg(W, OUT), Leg(U, IN))
    a, b = pair(a_legs, b_legs)
    want = np.tensordot(a.to_dense(), b.to_dense(), ((1, 2), (0, 1)))
    np.testing.assert_allclose(tenet.tensordot(a, b, ((1, 2), (0, 1))).to_dense(), want, atol=1e-10)
    swapped = tenet.tensordot(a, b, ((2, 1), (0, 1))).to_dense()
    assert not np.allclose(swapped, want, atol=1e-8)


# --- outer product ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a_legs", "b_legs"),
    [
        pytest.param((Leg(Q, OUT), Leg(P, IN)), (Leg(P, OUT),), id="u1"),
        pytest.param((Leg(V, OUT), Leg(W, IN)), (Leg(U, OUT), Leg(W, IN)), id="su2"),
    ],
)
def test_outer_product_falls_out_of_the_lowering(a_legs, b_legs):
    a, b = pair(a_legs, b_legs)
    c = tenet.tensordot(a, b, ((), ()))
    assert c.legs == (*a_legs, *b_legs)
    np.testing.assert_allclose(
        c.to_dense(), np.tensordot(a.to_dense(), b.to_dense(), axes=0), atol=1e-10
    )


# --- trace --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("legs", "axes"),
    [
        pytest.param((Leg(V, OUT), Leg(W, OUT), Leg(V, IN)), (0, 2), id="su2-opposite-sides"),
        pytest.param((Leg(V, OUT), Leg(V, OUT, dual=True), Leg(W, IN)), (0, 1), id="su2-same-side"),
        pytest.param((Leg(Q, OUT), Leg(P, IN), Leg(Q, IN)), (0, 2), id="u1-opposite-sides"),
        pytest.param((Leg(Q, OUT), Leg(Q, OUT, dual=True), Leg(P, IN)), (0, 1), id="u1-same-side"),
    ],
)
def test_trace_matches_numpy_trace(legs, axes):
    t = SymmetricTensor.random(legs, seed=5)
    got = tenet.trace(t, axes)
    assert got.legs == tuple(leg for i, leg in enumerate(legs) if i not in axes)
    want = np.trace(t.to_dense(), axis1=axes[0], axis2=axes[1])
    np.testing.assert_allclose(got.to_dense(), want, atol=1e-12)


def test_trace_refuses_a_non_contractible_pair_through_tensordot():
    t = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT), Leg(V, IN)), seed=5)
    with pytest.raises(ValueError, match="do not contract"):
        tenet.trace(t, (0, 1))


# --- norm and adjoint ----------------------------------------------------------------


@pytest.mark.parametrize(
    "legs",
    [
        pytest.param((Leg(V, OUT), Leg(W, IN)), id="su2"),
        pytest.param((Leg(Q, OUT), Leg(P, IN)), id="u1"),
        pytest.param((Leg(T3, OUT), Leg(T2, IN)), id="trivial"),
    ],
)
def test_norm_is_unchanged_by_a_composition_shaped_identity(legs):
    t = SymmetricTensor.random(legs, seed=7)
    c = tenet.tensordot(t, tenet.identity((t.legs[1],)), ((1,), (0,)))
    assert c.legs == t.legs
    assert float(tenet.norm(c)) == pytest.approx(float(tenet.norm(t)), abs=1e-12)


def test_adjoint_swaps_the_operands_up_to_the_documented_permutation():
    a_legs = (Leg(V, OUT), Leg(W, IN), Leg(U, IN))
    b_legs = (Leg(U, OUT), Leg(W, OUT), Leg(V, IN))
    a, b = pair(a_legs, b_legs)
    axes = ((2, 1), (0, 1))
    lhs = tenet.tensordot(a, b, axes).adjoint()
    rhs = tenet.tensordot(b.adjoint(), a.adjoint(), (axes[1], axes[0]))
    # lhs is (a free..., b free...); rhs is (b free..., a free...)
    n_b = b.ndim - len(axes[1])
    roll = (*range(n_b, rhs.ndim), *range(n_b))
    assert tenet.allclose(lhs, rhs.transpose(roll), atol=1e-10)


# --- refusals -------------------------------------------------------------------------


def test_refuses_a_pair_whose_spaces_differ():
    a, b = pair((Leg(V, OUT), Leg(W, IN)), (Leg(U, OUT), Leg(V, IN)))
    with pytest.raises(ValueError) as excinfo:
        tenet.tensordot(a, b, ((1,), (0,)))
    message = str(excinfo.value)
    assert "axis 1 of a" in message and "axis 0 of b" in message
    assert "spaces differ" in message


def test_refuses_a_pair_pointing_the_same_way():
    a, b = pair((Leg(V, OUT), Leg(W, IN)), (Leg(W, OUT, dual=True), Leg(V, IN)))
    with pytest.raises(ValueError) as excinfo:
        tenet.tensordot(a, b, ((1,), (0,)))
    message = str(excinfo.value)
    assert "axis 1 of a" in message and "axis 0 of b" in message
    assert "both ends of the wire point the same way" in message


def test_refuses_mismatched_axis_counts():
    a, b = pair((Leg(V, OUT), Leg(W, IN)), (Leg(W, OUT), Leg(V, IN)))
    with pytest.raises(ValueError, match="different lengths"):
        tenet.tensordot(a, b, ((0, 1), (0,)))


def test_refuses_a_repeated_axis_inside_one_operand():
    a, b = pair((Leg(V, OUT), Leg(W, IN)), (Leg(W, OUT), Leg(W, OUT), Leg(V, IN)))
    with pytest.raises(ValueError) as excinfo:
        tenet.tensordot(a, b, ((1, 1), (0, 1)))
    assert "axis 1 appears twice in a's axes" in str(excinfo.value)
    assert "diagonal" in str(excinfo.value)


@pytest.mark.parametrize("bad", [((-1,), (0,)), ((5,), (0,)), ((0,), (-2,)), ((0,), (9,))])
def test_refuses_out_of_range_and_negative_axes(bad):
    a, b = pair((Leg(V, OUT), Leg(W, IN)), (Leg(W, OUT), Leg(V, IN)))
    with pytest.raises(ValueError) as excinfo:
        tenet.tensordot(a, b, bad)
    assert "out of range" in str(excinfo.value)
    assert "negative axes are not accepted" in str(excinfo.value)


def test_refuses_a_negative_integer_axes_form():
    a, b = pair((Leg(V, OUT), Leg(W, IN)), (Leg(W, OUT), Leg(V, IN)))
    with pytest.raises(ValueError, match="negative"):
        tenet.tensordot(a, b, -1)
    with pytest.raises(ValueError, match="there are not that many"):
        tenet.tensordot(a, b, 3)


def test_refuses_different_providers():
    a, b = pair((Leg(V, OUT), Leg(W, IN)), (Leg(Q, OUT), Leg(P, IN)))
    with pytest.raises(ValueError) as excinfo:
        tenet.tensordot(a, b, ((1,), (0,)))
    assert "SU2" in str(excinfo.value) and "U1" in str(excinfo.value)


def test_refuses_a_full_contraction_and_cites_the_minimum_one_leg_rule():
    a, b = pair((Leg(V, OUT), Leg(W, IN)), (Leg(W, OUT), Leg(V, IN)))
    with pytest.raises(ValueError) as excinfo:
        tenet.tensordot(a, b, ((0, 1), (1, 0)))
    message = str(excinfo.value)
    assert "no free leg" in message and "TensorStructure" in message
    assert "tenet.norm" in message


def test_nothing_moves_before_a_refusal(monkeypatch):
    """Refuse-before-move: not one ``ar.do`` call happens on the way out."""
    a, b = pair((Leg(NB_S1, OUT), Leg(NB_S2, IN)), (Leg(NB_S2, IN, dual=True), Leg(NB_S1, OUT)))

    def boom(*args, **kwargs):
        raise AssertionError("tensordot moved a block before refusing")

    monkeypatch.setattr(ar, "do", boom)
    with pytest.raises(CapabilityError):
        tenet.tensordot(a, b, ((1,), (0,)))
    with pytest.raises(ValueError):
        tenet.tensordot(a, b, ((1,), (1,)))


# --- the NumPy integer axes form and autoray dispatch ---------------------------------


def test_integer_axes_form_matches_the_explicit_pair():
    a_legs = (Leg(V, OUT), Leg(W, IN), Leg(U, IN))
    b_legs = (Leg(W, OUT), Leg(U, OUT), Leg(V, IN))
    a, b = pair(a_legs, b_legs)
    assert tenet.allclose(tenet.tensordot(a, b, 2), tenet.tensordot(a, b, ((1, 2), (0, 1))))
    np.testing.assert_allclose(
        tenet.tensordot(a, b, 2).to_dense(),
        np.tensordot(a.to_dense(), b.to_dense(), 2),
        atol=1e-10,
    )


def test_ar_do_tensordot_is_the_direct_call():
    a_legs = (Leg(V, OUT), Leg(W, IN), Leg(U, IN))
    b_legs = (Leg(U, OUT), Leg(W, OUT), Leg(V, IN))
    a, b = pair(a_legs, b_legs)
    axes = ((2,), (0,))
    assert ar.do("tensordot", a, b, axes=axes) == tenet.tensordot(a, b, axes)


def test_ar_do_trace_is_the_direct_call():
    t = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT), Leg(V, IN)), seed=5)
    assert ar.do("trace", t, axes=(0, 2)) == tenet.trace(t, (0, 2))


# --- jit and grad ----------------------------------------------------------------------

JIT_CASES = [
    pytest.param(
        (Leg(V, OUT), Leg(W, OUT), Leg(U, IN)),
        (Leg(W, IN), Leg(U, OUT), Leg(V, IN)),
        ((1,), (0,)),
        id="su2-bending",
    ),
    pytest.param(
        (Leg(Q, OUT), Leg(P, OUT), Leg(Q, IN)),
        (Leg(P, IN), Leg(Q, OUT), Leg(Q, IN)),
        ((1,), (0,)),
        id="u1-bending",
    ),
]


@pytest.mark.parametrize(("a_legs", "b_legs", "axes"), JIT_CASES)
def test_jit_matches_the_numpy_backend(a_legs, b_legs, axes):
    jax = use_jax()
    a, b = pair(a_legs, b_legs)
    want = tenet.tensordot(a, b, axes)
    f = jax.jit(lambda x, y: tenet.tensordot(x, y, axes))
    got = f(a.to_backend("jax"), b.to_backend("jax"))
    assert got.backend == "jax"
    assert tenet.allclose(to_numpy(got), want, atol=1e-12)


def test_recompilation_happens_iff_the_treedef_changes():
    jax = use_jax()
    a_legs, b_legs, axes = JIT_CASES[0].values
    a, b = pair(a_legs, b_legs)
    traces = []

    @jax.jit
    def f(x, y):
        traces.append(1)
        return tenet.tensordot(x, y, axes)

    ja, jb = a.to_backend("jax"), b.to_backend("jax")
    f(ja, jb)
    f(SymmetricTensor.random(a_legs, seed=11).to_backend("jax"), jb)
    assert len(traces) == 1  # same treedef, same shapes: no retrace
    other = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT), Leg(U, IN, name="x")), seed=12)
    f(other.to_backend("jax"), jb)
    assert len(traces) == 2  # a different structure is a different treedef


def test_grad_differentiates_through_the_bend_coefficients():
    """SU(2) with a bending pattern, against a central difference over the blocks."""
    jax = use_jax()
    a_legs, b_legs, axes = JIT_CASES[0].values
    a, b = pair(a_legs, b_legs)
    jb = b.to_backend("jax")

    def loss(t):
        return tenet.norm(tenet.tensordot(t, jb, axes))

    grad = jax.grad(loss)(a.to_backend("jax"))
    h = 1e-6
    checked = 0
    for i, block in enumerate(a.blocks):
        if not block.size:
            continue
        for k in (0, block.size - 1):
            shifted = []
            for sign in (+1, -1):
                blocks = [np.array(x, copy=True) for x in a.blocks]
                blocks[i].reshape(-1)[k] += sign * h
                shifted.append(float(loss(SymmetricTensor(a.structure, tuple(blocks)))))
            fd = (shifted[0] - shifted[1]) / (2 * h)
            assert fd == pytest.approx(float(np.asarray(grad.blocks[i]).reshape(-1)[k]), abs=1e-6)
            checked += 1
    assert checked


# --- module hygiene ---------------------------------------------------------------------


def test_contraction_module_is_dense_free_numpy_free_and_provider_agnostic():
    """Read the file, like #41's ``test_only_pytree_module_imports_jax``. Docstrings
    are stripped first: this module's prose cites ``np.tensordot`` and ``to_dense``
    on purpose, and the invariant is about the code."""
    tree = ast.parse(pathlib.Path(tenet.ops.contraction.__file__).read_text())
    for node in ast.walk(tree):
        holder = isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef)
        if holder and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)
    assert "to_dense" not in code
    assert "numpy" not in code and "np." not in code
    assert "if provider is" not in code
    assert "isinstance(provider" not in code


def test_tensordot_and_trace_are_exported():
    for name in ("tensordot", "trace"):
        assert name in tenet.__all__
        assert name in tenet.ops.__all__
        assert getattr(tenet, name) is getattr(tenet.ops, name)
