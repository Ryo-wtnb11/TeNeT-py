"""Tests for line bending and repartitioning — issue #32.

The dense oracle is the cup/cap statement in its simplest true form: for Trivial
and U(1) the duality cup is the identity, so a bend changes no dense entry and
``to_dense(T.repartition(o, i)) == np.transpose(to_dense(T), (*o, *i))``. That is
weak on the coefficient axis on purpose, which is why the block-level "no
arithmetic happened" and round-trip criteria sit next to it.
"""

import dataclasses
import pathlib
import sys

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.repartition import BendPlan, bend, bend_plan, repartition
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry import (
    SU2,
    U1,
    BendingCoefficients,
    CapabilityError,
    DualBasis,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
)

REPARTITION_MODULE = sys.modules["tenet.ops.repartition"]

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 2, ONE: 3})
W = GradedSpace.new(SU2, {SINGLET: 2, HALF: 3})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
Q2 = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})

# interleaved sides throughout: OUT at axes 0, 2 and IN at axes 1, 3
SU2_LEGS = (Leg(V, OUT, name="a"), Leg(W, IN, name="b"), Leg(W, OUT), Leg(V, IN))
U1_LEGS = (Leg(Q, OUT, name="a"), Leg(Q2, IN, name="b"), Leg(Q2, OUT), Leg(Q, IN))
TRIV_LEGS = tuple(
    Leg(GradedSpace.new(Trivial, {TrivialSector(): m}), side, name=f"l{i}")
    for i, (m, side) in enumerate(((2, OUT), (3, IN), (4, OUT), (5, IN)))
)

# every (outputs, inputs) split of range(4) that keeps each side ascending;
# the within-side order is left alone so SU(2) can be asked the same questions
SPLITS = (
    ((0, 1, 2), (3,)),
    ((0, 2), (1, 3)),
    ((0,), (1, 2, 3)),
    ((1, 3), (0, 2)),
    ((0, 1, 2, 3), ()),
    ((), (0, 1, 2, 3)),
    ((2, 3), (0, 1)),
    ((0, 1), (2, 3)),
)


def u1() -> SymmetricTensor:
    return SymmetricTensor.random(U1_LEGS, seed=11)


def triv() -> SymmetricTensor:
    return SymmetricTensor.random(TRIV_LEGS, seed=13)


def su2() -> SymmetricTensor:
    return SymmetricTensor.random(SU2_LEGS, seed=7)


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def restore(ndim: int, axis: int) -> tuple[int, ...]:
    """Transpose axes that undo ``bend``'s "move the leg to the end"."""
    return (*range(axis), ndim - 1, *range(axis, ndim - 1))


# --- the capabilities ----------------------------------------------------------


def test_bending_coefficients_is_runtime_checkable():
    assert isinstance(Trivial, BendingCoefficients)
    assert isinstance(U1, BendingCoefficients)
    assert not isinstance(SU2, BendingCoefficients)


def test_dual_basis_is_runtime_checkable_and_is_the_identity_for_abelian():
    assert isinstance(Trivial, DualBasis)
    assert isinstance(U1, DualBasis)
    assert not isinstance(SU2, DualBasis)
    for provider, a in ((Trivial, TrivialSector()), (U1, U1Sector(2))):
        z = provider.z_matrix(a)
        np.testing.assert_array_equal(z, np.ones((1, 1)))
        assert not z.flags.writeable


# --- SU(2): the refusal, and the bend-free repartition that still works ---------


@pytest.mark.parametrize(("outputs", "inputs"), [((0, 1, 2), (3,)), ((0,), (1, 2, 3))])
def test_su2_repartition_that_moves_a_leg_is_refused(outputs, inputs):
    t = su2()
    with pytest.raises(CapabilityError) as excinfo:
        t.repartition(outputs, inputs)
    message = str(excinfo.value)
    assert "SU2" in message
    assert "BendingCoefficients" in message
    assert "B-symbol" in message or "B(a,b,c)" in message
    assert "Milestone 4" in message
    moved = next(ax for ax in range(4) if (t.legs[ax].side is OUT) != (ax in outputs))
    assert f"axis {moved}" in message


def test_su2_refusal_is_not_a_silent_result():
    t = su2()
    results = []
    for outputs, inputs in SPLITS:
        try:
            results.append(t.repartition(outputs, inputs))
        except CapabilityError:
            pass
    # only the split that matches the existing sides survives
    assert len(results) == 1
    assert results[0].legs == tuple(t.legs[i] for i in (0, 2, 1, 3))


def test_su2_no_bend_repartition_succeeds_and_reduces_to_transpose():
    t = su2()
    r = t.repartition(t.structure.out_axes, t.structure.in_axes)
    assert r == t.transpose((0, 2, 1, 3))
    assert [leg.dual for leg in r.legs] == [leg.dual for leg in t.legs]
    np.testing.assert_allclose(r.to_dense(), np.transpose(t.to_dense(), (0, 2, 1, 3)), atol=1e-12)


@pytest.mark.parametrize("legs", [SU2_LEGS, U1_LEGS, TRIV_LEGS])
def test_identity_repartition_moves_nothing(legs):
    t = SymmetricTensor.random(legs, seed=2)
    r = t.repartition(t.structure.out_axes, t.structure.in_axes)
    order = (*t.structure.out_axes, *t.structure.in_axes)
    assert r == t.transpose(order)
    assert r.legs == tuple(t.legs[i] for i in order)


# --- bend: the primitive's precondition and its leg bookkeeping -----------------


@pytest.mark.parametrize("axis", [0, 1])
def test_bend_refuses_a_non_last_leg_of_its_side(axis):
    t = u1()  # OUT axes (0, 2), IN axes (1, 3): neither 0 nor 1 is last on its side
    with pytest.raises(ValueError, match=f"axis {axis}"):
        bend(t, axis)


def test_bend_rejects_an_out_of_range_axis():
    with pytest.raises(ValueError, match="axis 4"):
        bend(u1(), 4)


@pytest.mark.parametrize("axis", [2, 3])
def test_bend_leg_bookkeeping(axis):
    t = u1()
    r = bend(t, axis)
    old = t.legs[axis]
    moved = r.legs[-1]
    assert moved.space == old.space
    assert moved.side is not old.side
    assert moved.dual is not old.dual
    assert moved.name == old.name
    assert r.legs[:-1] == tuple(leg for i, leg in enumerate(t.legs) if i != axis)
    assert sorted(r.reduced_shape) == sorted(t.reduced_shape)


def test_bend_never_needed_by_repartition_for_any_valid_argument():
    t = u1()
    for outputs, inputs in SPLITS:
        t.repartition(outputs, inputs)  # no ValueError from bend's precondition


# --- U(1) tree bookkeeping, key by key -----------------------------------------


def test_u1_tree_bookkeeping_key_by_key():
    t = u1()
    axis = 2  # last OUT leg: a bend_right
    plan = bend_plan(t.structure, axis)
    assert len(plan.terms) == t.structure.num_blocks
    for src, dst, _ in plan.terms:
        old = t.structure.block_order[src]
        new = plan.new_structure.block_order[dst]
        q = old.output_tree.uncoupled[-1]
        # the moved charge arrives negated, at the END of the input tree
        assert new.input_tree.uncoupled[-1] == U1Sector(-q.charge)
        assert new.input_tree.uncoupled[:-1] == old.input_tree.uncoupled
        # the output tree is the old one with its last line dropped ...
        assert new.output_tree.uncoupled == old.output_tree.uncoupled[:-1]
        # ... and the new coupled sector is the old output tree's last inner line
        assert new.coupled == old.output_tree.lines()[-2]
        assert new.coupled == U1Sector(old.coupled.charge - q.charge)


def test_u1_bend_left_tree_bookkeeping_key_by_key():
    t = u1()
    axis = 3  # last IN leg: a bend_left
    plan = bend_plan(t.structure, axis)
    for src, dst, _ in plan.terms:
        old = t.structure.block_order[src]
        new = plan.new_structure.block_order[dst]
        q = old.input_tree.uncoupled[-1]
        assert new.output_tree.uncoupled[-1] == U1Sector(-q.charge)
        assert new.output_tree.uncoupled[:-1] == old.output_tree.uncoupled
        assert new.input_tree.uncoupled == old.input_tree.uncoupled[:-1]
        assert new.coupled == old.input_tree.lines()[-2]


def test_rank_one_side_bends_to_the_unit():
    """Bending the only OUT leg empties the output tree; its coupled sector is the unit."""
    legs = (Leg(Q, OUT), Leg(Q2, IN))
    t = SymmetricTensor.random(legs, seed=3)
    r = bend(t, 0)
    assert r.structure.out_axes == ()
    for key in r.structure.block_order:
        assert key.output_tree.rank == 0
        assert key.coupled == U1.unit
    r.structure.validate()


def test_spines_are_recomputed_not_relabelled():
    """Every key of a rank-3 bend validates; naive relabelling gives invalid spines."""
    legs = (Leg(Q, OUT), Leg(Q2, OUT), Leg(Q, OUT), Leg(Q2, IN))
    t = SymmetricTensor.random(legs, seed=5)
    r = bend(t, 2)
    r.structure.validate()  # raises on any bad inner line, coupled sector or vertex
    for key in r.structure.block_order:
        key.output_tree.validate(U1)
        key.input_tree.validate(U1)
        # the surviving inner line is genuinely recomputed from the new uncoupled tuple
        assert key.input_tree.lines()[-1] == key.output_tree.coupled


def test_bend_unique_returns_a_key_for_the_shared_helper_on_both_providers():
    for provider, legs in ((U1, U1_LEGS), (Trivial, TRIV_LEGS)):
        s = TensorStructure(legs)
        key = s.block_order[0]
        for direction in ("bend_right", "bend_left"):
            terms = getattr(provider, direction)(key, dual=False)
            assert len(terms) == 1
            new_key, coeff = terms[0]
            assert isinstance(new_key, FusionBlockKey)
            assert coeff == 1.0


# --- the coefficient is exactly 1: no arithmetic touches a block ----------------


@pytest.mark.parametrize("legs", [U1_LEGS, TRIV_LEGS])
@pytest.mark.parametrize("axis", [2, 3])
def test_coefficients_are_exactly_one_and_blocks_are_only_reindexed(legs, axis):
    t = SymmetricTensor.random(legs, seed=17)
    plan = bend_plan(t.structure, axis)
    assert all(coeff == 1.0 for _, _, coeff in plan.terms)
    r = bend(t, axis)
    perm = (*(i for i in range(t.ndim) if i != axis), axis)
    for src, dst, _ in plan.terms:
        # exactly equal, not merely close: a pure transpose, no arithmetic
        np.testing.assert_array_equal(r.blocks[dst], np.transpose(t.blocks[src], perm))


@pytest.mark.parametrize("legs", [U1_LEGS, TRIV_LEGS])
@pytest.mark.parametrize("axis", [2, 3])
def test_bend_is_a_bijection_on_blocks(legs, axis):
    s = TensorStructure(legs)
    plan = bend_plan(s, axis)
    n = s.num_blocks
    assert sorted(src for src, _, _ in plan.terms) == list(range(n))
    assert sorted(dst for _, dst, _ in plan.terms) == list(range(n))
    assert plan.new_structure.num_blocks == n


# --- round trips and the norm ---------------------------------------------------


@pytest.mark.parametrize("legs", [U1_LEGS, TRIV_LEGS])
@pytest.mark.parametrize("axis", [2, 3])
def test_bend_round_trip_is_exact(legs, axis):
    t = SymmetricTensor.random(legs, seed=19)
    there = bend(t, axis)
    back = bend(there, there.ndim - 1)
    # the bend moved the leg to the end, so come back to the original axis order
    assert back.transpose(restore(t.ndim, axis)) == t


@pytest.mark.parametrize(
    ("legs", "axis"),
    [
        ((Leg(Q, OUT), Leg(Q2, IN), Leg(Q2, OUT), Leg(Q, IN)), 2),  # interleaved sides
        ((Leg(Q, OUT), Leg(Q2, IN), Leg(Q2, OUT), Leg(Q, IN)), 3),  # interleaved sides
        ((Leg(Q, OUT), Leg(Q2, OUT), Leg(Q2, IN)), 1),
        ((Leg(Q2, IN), Leg(Q, OUT)), 0),  # a rank-1 side, bent away entirely
    ],
)
def test_round_trip_from_four_starting_axes(legs, axis):
    t = SymmetricTensor.random(legs, seed=23)
    there = bend(t, axis)
    back = bend(there, there.ndim - 1)
    assert back.transpose(restore(t.ndim, axis)) == t


@pytest.mark.parametrize("legs", [U1_LEGS, TRIV_LEGS])
@pytest.mark.parametrize(("outputs", "inputs"), SPLITS)
def test_repartition_round_trip(legs, outputs, inputs):
    t = SymmetricTensor.random(legs, seed=29)
    r = t.repartition(outputs, inputs)
    # r's axes are numbered in its own public order; send every axis back
    where = {a: i for i, a in enumerate((*outputs, *inputs))}
    back = r.repartition(
        tuple(where[a] for a in t.structure.out_axes),
        tuple(where[a] for a in t.structure.in_axes),
    )
    order = (*t.structure.out_axes, *t.structure.in_axes)
    assert back.legs == tuple(t.legs[i] for i in order)
    assert tenet.allclose(back, t.transpose(order))


@pytest.mark.parametrize("legs", [U1_LEGS, TRIV_LEGS])
@pytest.mark.parametrize(("outputs", "inputs"), SPLITS)
def test_norm_is_preserved_exactly(legs, outputs, inputs):
    t = SymmetricTensor.random(legs, seed=31)
    assert abs(tenet.norm(t.repartition(outputs, inputs)) - tenet.norm(t)) < 1e-12


@pytest.mark.parametrize("axis", [2, 3])
def test_bend_preserves_norm(axis):
    t = u1()
    assert abs(tenet.norm(bend(t, axis)) - tenet.norm(t)) < 1e-12


# --- the dense oracle ------------------------------------------------------------


@pytest.mark.parametrize("legs", [U1_LEGS, TRIV_LEGS])
@pytest.mark.parametrize(("outputs", "inputs"), SPLITS)
def test_dense_oracle(legs, outputs, inputs):
    t = SymmetricTensor.random(legs, seed=37)
    np.testing.assert_allclose(
        t.repartition(outputs, inputs).to_dense(),
        np.transpose(t.to_dense(), (*outputs, *inputs)),
        atol=1e-12,
    )


@pytest.mark.parametrize("axis", [2, 3])
def test_bend_dense_oracle(axis):
    t = u1()
    perm = (*(i for i in range(t.ndim) if i != axis), axis)
    np.testing.assert_allclose(
        bend(t, axis).to_dense(), np.transpose(t.to_dense(), perm), atol=1e-12
    )


# --- to_dense's narrowed dual guard ----------------------------------------------


@pytest.mark.parametrize("provider_legs", [U1_LEGS, TRIV_LEGS])
def test_to_dense_now_accepts_a_dual_leg_for_abelian_providers(provider_legs):
    legs = tuple(
        Leg(leg.space, leg.side, dual=(i == 1), name=leg.name)
        for i, leg in enumerate(provider_legs)
    )
    t = SymmetricTensor.random(legs, seed=41)
    dense = t.to_dense()
    assert dense.shape == t.shape
    assert abs(float(np.linalg.norm(dense)) - tenet.norm(t)) < 1e-12


def test_to_dense_still_refuses_a_dual_su2_leg_by_capability():
    legs = (Leg(V, OUT), Leg(W, IN, dual=True))
    with pytest.raises(CapabilityError) as excinfo:
        SymmetricTensor.zeros(legs).to_dense()
    message = str(excinfo.value)
    assert "DualBasis" in message
    assert "axis 1" in message
    assert "Z-isomorphism" in message
    assert "Frobenius-Schur" in message
    assert "Milestone 4" in message


def test_dual_leg_dense_expansion_matches_the_non_dual_one_for_u1():
    """Z = [[1]] means the dense entries are unchanged; only the labels move."""
    plain = (Leg(Q, OUT), Leg(Q2, IN))
    t = SymmetricTensor.random(plain, seed=43)
    bent = bend(t, 0)
    assert bent.legs[-1].dual
    np.testing.assert_allclose(bent.to_dense(), np.transpose(t.to_dense(), (1, 0)), atol=1e-12)


# --- map-level and composition consistency (U(1)) ---------------------------------


def test_map_level_shapes_follow_the_new_layout():
    t = u1()
    r = t.repartition((0, 1, 2), (3,))
    layout = tenet.map_layout(r.structure)
    mats = tenet.to_matrices(r)
    assert set(mats) == set(layout.sectors)
    for c, m in mats.items():
        assert tuple(m.shape) == layout.shape(c)
    assert tenet.allclose(tenet.from_matrices(r.structure, mats), r)


def test_map_level_empty_side_gives_column_and_row_vectors():
    t = u1()
    column = t.repartition((0, 1, 2, 3), ())  # no IN legs: B_c is a column vector
    layout = tenet.map_layout(column.structure)
    assert layout.sectors == (U1.unit,)
    mats = tenet.to_matrices(column)
    assert mats[U1.unit].shape[1] == 1

    row = t.repartition((), (0, 1, 2, 3))  # no OUT legs: B_c is a row vector
    mats = tenet.to_matrices(row)
    assert mats[U1.unit].shape[0] == 1
    assert abs(tenet.norm(column) - tenet.norm(row)) < 1e-12


def test_composition_agrees_with_bending_on_the_shared_index():
    """Repartition A and B in matching ways; the dense contraction is unchanged.

    ``a`` (axes ``x, y, z``) meets ``b`` (axes ``z, y``) at ``z``. Bending ``a``'s
    ``y`` into its domain and ``b``'s ``y`` into its codomain keeps them composable
    — both arrive dualized — and the composite now contracts ``y`` as well. If the
    bend got the ``dual`` flip or the charge negation wrong, either the legs would
    stop matching or the dense answer would move.
    """
    a = SymmetricTensor.random((Leg(Q, OUT), Leg(Q2, OUT), Leg(Q, IN)), seed=47)
    b = SymmetricTensor.random((Leg(Q, OUT), Leg(Q2, IN)), seed=53)
    da, db = a.to_dense(), b.to_dense()
    np.testing.assert_allclose((a @ b).to_dense(), np.tensordot(da, db, ([2], [0])), atol=1e-10)

    a2 = a.repartition((0,), (1, 2))
    b2 = b.repartition((1, 0), ())
    # composable: same (space, dual) in the same order — side is what differs
    assert a2.as_map().domain.matches(b2.as_map().codomain) is None
    assert all(leg.dual for leg in (a2.legs[1], b2.legs[0]))

    composed = a2 @ b2
    assert composed.legs == (a.legs[0],)
    np.testing.assert_allclose(
        composed.to_dense(), np.tensordot(da, db, ([1, 2], [1, 0])), atol=1e-10
    )


# --- validation -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outputs", "inputs", "needle"),
    [
        ((0, 1), (1, 2, 3), "axis 1"),  # overlap
        ((0, 1), (2,), "axis 3"),  # omission
        ((0, 0), (1, 2, 3), "axis 0"),  # repetition
        ((0, 9), (1, 2, 3), "axis 9"),  # out of range
        ((0, 1, 2), (-1,), "axis -1"),  # negative
    ],
)
def test_repartition_validates_its_arguments(outputs, inputs, needle):
    with pytest.raises(ValueError, match=needle):
        u1().repartition(outputs, inputs)


def test_repartition_rejects_non_integer_axes():
    with pytest.raises(ValueError, match="integers"):
        u1().repartition(("0", 1, 2), (3,))


# --- the plan ---------------------------------------------------------------------


def test_bend_plan_is_cached_frozen_hashable_and_array_free():
    s = u1().structure
    plan = bend_plan(s, 2)
    assert bend_plan(s, 2) is plan
    assert hash(plan) == hash(bend_plan(s, 2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.terms = ()
    for field in dataclasses.fields(BendPlan):
        assert not hasattr(getattr(plan, field.name), "shape"), field.name


# --- module hygiene ---------------------------------------------------------------


def test_module_is_numpy_free_and_provider_agnostic():
    src = pathlib.Path(REPARTITION_MODULE.__file__).read_text()
    assert "np." not in src
    assert "import numpy" not in src
    assert "to_dense()" not in src
    assert "U1Provider" not in src and "SU2Provider" not in src
    assert "if provider ==" not in src
    assert "isinstance(provider" not in src
    assert 'ar.do("transpose"' in src
    assert "ponytail:" in src  # the rejected SU(2) rank-1 shortcut is recorded


def test_public_exports_and_readme_spelling():
    assert tenet.repartition is repartition
    assert tenet.bend is bend
    t = u1()
    # the README's keyword spelling, and the free-function form
    assert t.repartition(outputs=(0, 1, 2), inputs=(3,)) == repartition(t, (0, 1, 2), (3,))


# --- backends ----------------------------------------------------------------------


@pytest.mark.parametrize(("outputs", "inputs"), SPLITS)
def test_jax_repartition_matches_numpy(outputs, inputs):
    use_jax()
    t = u1()
    r = t.to_backend("jax").repartition(outputs, inputs)
    assert r.backend == "jax"
    np.testing.assert_allclose(
        r.to_backend("numpy").to_dense(),
        np.transpose(t.to_dense(), (*outputs, *inputs)),
        atol=1e-12,
    )


@pytest.mark.parametrize("axis", [2, 3])
def test_jax_bend_round_trip(axis):
    use_jax()
    t = u1().to_backend("jax")
    there = bend(t, axis)
    back = bend(there, there.ndim - 1)
    assert back.backend == "jax"
    assert tenet.allclose(back.transpose(restore(t.ndim, axis)).to_backend("numpy"), u1())
