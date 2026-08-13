"""Tests for SU(2)'s ``BendingCoefficients`` — one per acceptance criterion of #38.

Unlike the Abelian bend (#32), every factor here is a real number that can be
individually wrong while shapes and sector bookkeeping stay plausible, so three
independent oracles are used and each is shown to have teeth by a negative
companion built with :func:`bent_with` — the real structural plan, hand-modified
coefficients:

* **norm** ``sqrt(Σ qdim(c)‖A‖²)`` is preserved because ``qdim(a)·qdim(c)/qdim(a)
  == qdim(c)`` and ``|B| == 1``. Dropping the ``sqrt(qdim(c)/qdim(a))`` prefactor
  breaks it; dropping ``B`` does not (it has unit modulus), which is exactly why
  the dense oracle is also needed.
* **dense** ``to_dense(T.repartition(o, i)) == np.transpose(to_dense(T), (*o, *i))``,
  *with no Z insertion*. See :func:`dense_oracle` for why that is the sharp
  statement here and not the weak one it was for U(1): ``to_dense`` already
  inserts ``z_matrix`` on every ``dual`` axis (#37), and the bend's own duality
  coefficient is what cancels it. Dropping ``B``, the prefactor **or** the
  Frobenius-Schur factor all move dense entries.
* **round trip** ``bend_left ∘ bend_right == id``, which pins ``bend_left``'s
  coefficient as the exact inverse rather than a reciprocal guess.
"""

import dataclasses
import pathlib
from math import sqrt

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.repartition import bend, bend_plan
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry import SU2, U1, BendingCoefficients, SU2Sector, Trivial, U1Sector
from tenet.symmetry.base import bend_braided
from tenet.symmetry.su2 import SU2Provider

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)

# half-integer *and* integer sectors on every space, with differing qdim (1, 2, 3)
V = GradedSpace.new(SU2, {HALF: 2, ONE: 3})
W = GradedSpace.new(SU2, {SINGLET: 2, HALF: 3})
U = GradedSpace.new(SU2, {SINGLET: 1, HALF: 2, ONE: 2})

# interleaved sides: OUT at axes 0, 2 and IN at axes 1, 3 (the #32 layout)
LEGS = (Leg(V, OUT, name="a"), Leg(W, IN, name="b"), Leg(W, OUT), Leg(V, IN))
# axes 0, 2, 3 start dual=True; 2 and 3 are the last leg of their side, so they are
# the ones ``bend`` can act on directly (axis 0 crosses via ``repartition``)
DUAL_LEGS = (
    Leg(V, OUT, dual=True, name="a"),
    Leg(W, IN),
    Leg(W, OUT, dual=True),
    Leg(V, IN, dual=True),
)

SPLITS = (
    ((0, 1, 2), (3,)),
    ((0, 2), (1, 3)),  # the bend-free one
    ((0,), (1, 2, 3)),
    ((1, 3), (0, 2)),
    ((0, 1, 2, 3), ()),
    ((), (0, 1, 2, 3)),
    ((2, 3), (0, 1)),
    ((0, 1), (2, 3)),
)
MOVING = tuple(s for s in SPLITS if s != ((0, 2), (1, 3)))


def su2(legs=LEGS, seed=7) -> SymmetricTensor:
    return SymmetricTensor.random(legs, seed=seed)


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def restore(ndim: int, axis: int) -> tuple[int, ...]:
    """Transpose axes that undo ``bend``'s "move the leg to the end"."""
    return (*range(axis), ndim - 1, *range(axis, ndim - 1))


def moved_sector(t: SymmetricTensor, axis: int, key: FusionBlockKey) -> SU2Sector:
    """``b``: the label the bent leg contributes to its own tree, for ``key``."""
    tree = key.output_tree if t.legs[axis].side is OUT else key.input_tree
    return tree.uncoupled[-1]


def bent_with(t: SymmetricTensor, axis: int, tweak) -> SymmetricTensor:
    """:func:`bend` with ``tweak(key, coeff) -> coeff'`` applied to every coefficient.

    Uses the real, unmodified :func:`bend_plan`, so only the *arithmetic* differs
    from the shipped bend — which is what makes the negative companions honest.
    """
    plan = bend_plan(t.structure, axis)
    perm = (*(i for i in range(t.ndim) if i != axis), axis)
    blocks = {
        dst: np.transpose(t.blocks[src], perm) * tweak(t.structure.block_order[src], coeff)
        for src, dst, coeff in plan.terms
    }
    return SymmetricTensor(
        plan.new_structure, tuple(blocks[i] for i in range(plan.new_structure.num_blocks))
    )


def dense_oracle(t: SymmetricTensor, outputs, inputs) -> np.ndarray:
    """The dense array ``t.repartition(outputs, inputs)`` must equal.

    **The Z convention.** A bend moves a leg to the other side and flips its
    ``dual`` flag, and ``to_dense`` expands a ``dual`` axis by contracting the
    provider's ``Z_a: V_a -> V_a^*`` onto it (#37, ``_tree_cgt``). The bending
    coefficient ``sqrt(qdim(c)/qdim(a))·B(a,b,c)·conj(chi)`` *is* that same
    duality isomorphism written in the fusion basis, so the two cancel exactly
    and the dense array carries no leftover ``Z``: the oracle is the plain
    transposed array. It is therefore not the weak U(1) statement re-used — for
    U(1) it held because ``Z == [[1]]`` and the coefficient was ``1``; here both
    sides are non-trivial and their cancellation is the thing under test. The
    negative companions below show each factor is needed for it to hold.
    """
    return np.transpose(np.asarray(t.to_dense()), (*outputs, *inputs))


# --- the capability -------------------------------------------------------------


def test_su2_is_a_bending_coefficients_provider():
    assert isinstance(SU2, BendingCoefficients)
    _: BendingCoefficients = SU2  # static conformance


def test_su2_repartition_is_total_and_the_refusal_is_gone():
    t = su2()
    for outputs, inputs in SPLITS:
        r = t.repartition(outputs, inputs)  # no CapabilityError
        assert r.legs == tuple(
            dataclasses.replace(
                t.legs[a],
                side=OUT if a in outputs else IN,
                dual=t.legs[a].dual ^ ((t.legs[a].side is OUT) != (a in outputs)),
            )
            for a in (*outputs, *inputs)
        )


# --- leg and tree bookkeeping ----------------------------------------------------


@pytest.mark.parametrize("axis", [2, 3])
@pytest.mark.parametrize("legs", [LEGS, DUAL_LEGS])
def test_leg_bookkeeping(legs, axis):
    t = su2(legs)
    r = bend(t, axis)
    old, moved = t.legs[axis], r.legs[-1]
    assert moved.space == old.space
    assert moved.name == old.name
    assert moved.side is not old.side
    assert moved.dual is not old.dual
    assert r.legs[:-1] == tuple(leg for i, leg in enumerate(t.legs) if i != axis)
    assert sorted(r.reduced_shape) == sorted(t.reduced_shape)


def test_tree_bookkeeping_key_by_key_bend_right():
    t = su2()
    axis = 2  # last OUT leg
    plan = bend_plan(t.structure, axis)
    assert len(plan.terms) == t.structure.num_blocks
    for src, dst, _ in plan.terms:
        old = t.structure.block_order[src]
        new = plan.new_structure.block_order[dst]
        b = old.output_tree.uncoupled[-1]
        a = old.output_tree.lines()[-2] if old.output_tree.rank >= 2 else SU2.unit
        assert new.output_tree.uncoupled == old.output_tree.uncoupled[:-1]
        assert new.coupled == a
        assert new.input_tree.uncoupled[-1] == SU2.dual(b)
        assert new.input_tree.uncoupled[:-1] == old.input_tree.uncoupled
        # the destination's old coupled sector became its new last inner line
        assert new.input_tree.lines()[-2] == old.coupled
    plan.new_structure.validate()


def test_tree_bookkeeping_key_by_key_bend_left():
    t = su2()
    plan = bend_plan(t.structure, 3)  # last IN leg
    for src, dst, _ in plan.terms:
        old = t.structure.block_order[src]
        new = plan.new_structure.block_order[dst]
        b = old.input_tree.uncoupled[-1]
        assert new.input_tree.uncoupled == old.input_tree.uncoupled[:-1]
        assert new.output_tree.uncoupled[-1] == SU2.dual(b)
        assert new.output_tree.lines()[-2] == old.coupled
    plan.new_structure.validate()


def test_spines_are_recomputed_not_relabelled():
    """A rank-3 SU(2) side: naive relabelling of the spine gives invalid trees."""
    legs = (Leg(U, OUT), Leg(U, OUT), Leg(U, OUT), Leg(U, IN))
    t = SymmetricTensor.random(legs, seed=5)
    r = bend(t, 2)
    r.structure.validate()
    relabelled = 0
    for src, dst, _ in bend_plan(t.structure, 2).terms:
        old = t.structure.block_order[src]
        new = r.structure.block_order[dst]
        new.output_tree.validate(SU2)
        new.input_tree.validate(SU2)
        # the surviving output inner line is the old spine truncated, and the new
        # input spine ends on it — not on the old coupled sector
        assert new.output_tree.coupled == old.output_tree.lines()[-2]
        relabelled += new.coupled == old.coupled
    assert relabelled < len(bend_plan(t.structure, 2).terms)  # a relabel would be wrong


def test_bend_is_a_bijection_on_blocks():
    for legs in (LEGS, DUAL_LEGS):
        s = TensorStructure(legs)
        for axis in (2, 3):
            plan = bend_plan(s, axis)
            n = s.num_blocks
            assert sorted(src for src, _, _ in plan.terms) == list(range(n))
            assert sorted(dst for _, dst, _ in plan.terms) == list(range(n))
            assert plan.new_structure.num_blocks == n


def test_multiplicity_free_single_term():
    s = TensorStructure(LEGS)
    for key in s.block_order:
        assert len(SU2.bend_right(key, dual=False)) == 1
        assert len(SU2.bend_left(key, dual=False)) == 1


# --- coefficient spot values ------------------------------------------------------


@pytest.mark.parametrize("dj", [1, 2, 3, 4])
def test_unit_codomain_coefficient_is_sqrt_qdim(dj):
    """``a`` is the unit, ``B(1, b, b) == 1``, so the coefficient collapses."""
    b = SU2Sector(dj)
    space = GradedSpace.new(SU2, {b: 1})
    key = TensorStructure((Leg(space, OUT), Leg(space, IN))).block_order[0]
    ((_, coeff),) = SU2.bend_right(key, dual=False)
    assert abs(coeff - sqrt(SU2.qdim(b))) < 1e-14
    assert abs(SU2.b_symbol(SU2.unit, b, b) - 1.0) < 1e-14


def test_b_symbol_has_unit_modulus_on_every_exercised_triple():
    triples = [
        (a, b, c)
        for a in map(SU2Sector, range(5))
        for b in map(SU2Sector, range(5))
        for c in SU2.fusion(a, b)
    ]
    assert len(triples) > 20
    for a, b, c in triples:
        assert abs(abs(SU2.b_symbol(a, b, c)) - 1.0) < 1e-13


def test_coefficients_match_the_written_formula():
    t = su2()
    for axis, right in ((2, True), (3, False)):
        for src, _, coeff in bend_plan(t.structure, axis).terms:
            key = t.structure.block_order[src]
            tree = key.output_tree if right else key.input_tree
            b, c = tree.uncoupled[-1], tree.coupled
            a = tree.lines()[-2] if tree.rank >= 2 else SU2.unit
            want = sqrt(SU2.qdim(c) / SU2.qdim(a)) * SU2.b_symbol(a, b, c)
            assert abs(coeff - want) < 1e-13


# --- norm preservation, and the negative companion --------------------------------


@pytest.mark.parametrize(("outputs", "inputs"), SPLITS)
@pytest.mark.parametrize("legs", [LEGS, DUAL_LEGS])
def test_repartition_preserves_norm(legs, outputs, inputs):
    t = su2(legs, seed=31)
    assert abs(tenet.norm(t.repartition(outputs, inputs)) - tenet.norm(t)) < 1e-12


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_every_starting_axis_preserves_norm(axis):
    """Five starting axes across two layouts, three sectors of differing qdim."""
    t = su2(seed=31)
    outputs = tuple(a for a in range(4) if (t.legs[a].side is OUT) != (a == axis))
    inputs = tuple(a for a in range(4) if a not in outputs)
    assert abs(tenet.norm(t.repartition(outputs, inputs)) - tenet.norm(t)) < 1e-12
    assert {SU2.qdim(k.coupled) for k in t.structure.block_order} >= {1.0, 2.0, 3.0}


@pytest.mark.parametrize("axis", [2, 3])
def test_dropping_the_qdim_prefactor_breaks_the_norm(axis):
    """The teeth of the criterion above: ``B`` alone would not conserve the norm."""
    t = su2(seed=31)

    def no_prefactor(key, coeff):
        tree = key.output_tree if axis == 2 else key.input_tree
        a = tree.lines()[-2] if tree.rank >= 2 else SU2.unit
        return coeff * sqrt(SU2.qdim(a) / SU2.qdim(tree.coupled))

    assert abs(tenet.norm(bend(t, axis)) - tenet.norm(t)) < 1e-12
    # off by percents, not by 1e-12: the criterion above has teeth
    assert abs(tenet.norm(bent_with(t, axis, no_prefactor)) - tenet.norm(t)) > 1e-3 * tenet.norm(t)


@pytest.mark.parametrize("axis", [2, 3])
def test_dropping_the_b_symbol_leaves_the_norm_alone_but_moves_the_dense_array(axis):
    """Why the norm is not sufficient on its own: ``|B| == 1``."""
    t = su2(seed=31)

    def no_b(key, coeff):
        tree = key.output_tree if axis == 2 else key.input_tree
        b, c = tree.uncoupled[-1], tree.coupled
        a = tree.lines()[-2] if tree.rank >= 2 else SU2.unit
        return coeff / SU2.b_symbol(a, b, c)

    wrong = bent_with(t, axis, no_b)
    assert abs(tenet.norm(wrong) - tenet.norm(t)) < 1e-12
    perm = (*(i for i in range(t.ndim) if i != axis), axis)
    assert not np.allclose(np.asarray(wrong.to_dense()), dense_oracle(t, perm[:-1], perm[-1:]))


# --- the Frobenius-Schur branch ---------------------------------------------------


@pytest.mark.parametrize("dj", [1, 2, 3])
def test_fs_factor_is_the_only_difference_between_a_dual_and_a_direct_bend(dj):
    """Coefficient level: ``chi_j = (-1)**2j``, so half-integer spins flip the sign."""
    b = SU2Sector(dj)
    space = GradedSpace.new(SU2, {b: 1})
    key = TensorStructure((Leg(space, OUT), Leg(space, IN))).block_order[0]
    ((_, direct),) = SU2.bend_right(key, dual=False)
    ((_, already_dual),) = SU2.bend_right(key, dual=True)
    assert already_dual == pytest.approx(SU2.frobenius_schur(b) * direct, abs=1e-14)
    assert (already_dual < 0) == (dj % 2 == 1)  # a genuine sign for half-integer spin


def test_fs_branch_is_exercised_by_a_half_integer_dual_leg():
    """The bent leg starts ``dual=True`` and carries spin 1/2: the sign is real."""
    t = su2(DUAL_LEGS)
    axis = 3  # Leg(V, IN, dual=True): V carries spin 1/2 and spin 1
    assert t.legs[axis].dual
    flipped = 0
    for src, _, coeff in bend_plan(t.structure, axis).terms:
        key = t.structure.block_order[src]
        b = moved_sector(t, axis, key)
        ((_, direct),) = SU2.bend_left(key, dual=False)
        assert coeff == pytest.approx(SU2.frobenius_schur(b) * direct, abs=1e-13)
        flipped += b.two_j % 2 == 1
    assert flipped  # half-integer sectors really are present, so signs really flip


@pytest.mark.parametrize("axis", [2, 3])
def test_dropping_the_fs_factor_moves_the_dense_array_of_an_already_dual_leg(axis):
    """Dense observation: a missing ``chi`` is invisible to shapes, sectors and norm.

    ``axis`` 2 is a ``bend_right`` and 3 a ``bend_left``; both start ``dual=True``
    on a space carrying spin 1/2, so ``chi = -1`` on part of the expansion.
    """
    t = su2(DUAL_LEGS, seed=13)
    assert t.legs[axis].dual

    def no_fs(key, coeff):
        return coeff * SU2.frobenius_schur(moved_sector(t, axis, key))

    good, wrong = bend(t, axis), bent_with(t, axis, no_fs)
    assert abs(tenet.norm(wrong) - tenet.norm(t)) < 1e-12  # the norm cannot see it
    assert not np.allclose(np.asarray(wrong.to_dense()), np.asarray(good.to_dense()))
    perm = (*(i for i in range(t.ndim) if i != axis), axis)
    np.testing.assert_allclose(
        np.asarray(good.to_dense()), np.transpose(np.asarray(t.to_dense()), perm), atol=1e-11
    )


def test_integer_only_spins_would_not_catch_a_missing_fs_factor():
    """Stated as a test so the insufficiency #38 warns about is on the record."""
    integer = GradedSpace.new(SU2, {SINGLET: 2, ONE: 2})
    t = SymmetricTensor.random((Leg(integer, OUT), Leg(integer, IN, dual=True)), seed=3)
    with_fs = bend(t, 1)
    without_fs = bent_with(
        t, 1, lambda key, c: c * SU2.frobenius_schur(key.input_tree.uncoupled[-1])
    )
    np.testing.assert_allclose(with_fs.blocks, without_fs.blocks, atol=1e-14)


# --- the dense oracle --------------------------------------------------------------


@pytest.mark.parametrize(("outputs", "inputs"), MOVING)
@pytest.mark.parametrize("legs", [LEGS, DUAL_LEGS])
def test_dense_oracle(legs, outputs, inputs):
    t = su2(legs, seed=37)
    np.testing.assert_allclose(
        np.asarray(t.repartition(outputs, inputs).to_dense()),
        dense_oracle(t, outputs, inputs),
        atol=1e-11,
    )


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_bend_dense_oracle(axis):
    t = su2(seed=37)
    if axis not in (2, 3):  # bend acts on the last leg of its side
        outputs = tuple(a for a in range(4) if (t.legs[a].side is OUT) != (a == axis))
        inputs = tuple(a for a in range(4) if a not in outputs)
        np.testing.assert_allclose(
            np.asarray(t.repartition(outputs, inputs).to_dense()),
            dense_oracle(t, outputs, inputs),
            atol=1e-11,
        )
        return
    perm = (*(i for i in range(t.ndim) if i != axis), axis)
    np.testing.assert_allclose(
        np.asarray(bend(t, axis).to_dense()),
        np.transpose(np.asarray(t.to_dense()), perm),
        atol=1e-11,
    )


def test_dense_oracle_is_not_vacuous():
    """Every factor is needed: scale one coefficient and the oracle fails."""
    t = su2(seed=37)
    wrong = bent_with(t, 2, lambda key, c: 2 * c)
    perm = (0, 1, 3, 2)
    assert not np.allclose(np.asarray(wrong.to_dense()), np.transpose(t.to_dense(), perm))


# --- the full-side bend -------------------------------------------------------------


def test_full_side_bend_empties_a_side():
    t = su2()
    column = t.repartition((0, 1, 2, 3), ())
    assert column.structure.in_axes == ()
    for key in column.structure.block_order:
        assert key.input_tree.rank == 0
        assert key.coupled == SU2.unit
    column.structure.validate()
    np.testing.assert_allclose(
        np.asarray(column.to_dense()), dense_oracle(t, (0, 1, 2, 3), ()), atol=1e-11
    )

    row = t.repartition((), (0, 1, 2, 3))
    assert row.structure.out_axes == ()
    assert all(key.output_tree.rank == 0 for key in row.structure.block_order)
    np.testing.assert_allclose(
        np.asarray(row.to_dense()), dense_oracle(t, (), (0, 1, 2, 3)), atol=1e-11
    )
    assert abs(tenet.norm(column) - tenet.norm(row)) < 1e-12


# --- round trips --------------------------------------------------------------------


@pytest.mark.parametrize("axis", [2, 3])
@pytest.mark.parametrize("legs", [LEGS, DUAL_LEGS])
def test_bend_round_trip(legs, axis):
    t = su2(legs, seed=19)
    there = bend(t, axis)
    back = bend(there, there.ndim - 1)
    assert tenet.allclose(back.transpose(restore(t.ndim, axis)), t, atol=1e-12)


@pytest.mark.parametrize(
    ("legs", "axis"),
    [
        (LEGS, 2),
        (LEGS, 3),
        (DUAL_LEGS, 3),  # a leg that begins dual=True
        ((Leg(V, OUT), Leg(U, OUT), Leg(U, IN)), 1),
        ((Leg(U, IN), Leg(V, OUT)), 1),  # a rank-1 side, bent away entirely
        ((Leg(U, IN, dual=True), Leg(V, OUT)), 0),
    ],
)
def test_round_trip_from_six_starting_axes(legs, axis):
    t = SymmetricTensor.random(legs, seed=23)
    there = bend(t, axis)
    back = bend(there, there.ndim - 1)
    assert tenet.allclose(back.transpose(restore(t.ndim, axis)), t, atol=1e-12)


@pytest.mark.parametrize(("outputs", "inputs"), SPLITS)
@pytest.mark.parametrize("legs", [LEGS, DUAL_LEGS])
def test_repartition_round_trip(legs, outputs, inputs):
    t = su2(legs, seed=29)
    r = t.repartition(outputs, inputs)
    where = {a: i for i, a in enumerate((*outputs, *inputs))}
    back = r.repartition(
        tuple(where[a] for a in t.structure.out_axes),
        tuple(where[a] for a in t.structure.in_axes),
    )
    order = (*t.structure.out_axes, *t.structure.in_axes)
    assert back.legs == tuple(t.legs[i] for i in order)
    assert tenet.allclose(back, t.transpose(order), atol=1e-12)


def test_bend_left_is_not_merely_the_reciprocal_modulus():
    """The round trip pins the phase too: ``conj`` is what makes it exact."""
    t = su2()
    for src, _, right_coeff in bend_plan(t.structure, 2).terms:
        key = t.structure.block_order[src]
        ((back_key, left_coeff),) = SU2.bend_left(SU2.bend_right(key, dual=False)[0][0], dual=True)
        assert back_key == key
        assert abs(right_coeff * left_coeff - 1.0) < 1e-13


# --- unchanged #32 criteria ----------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1])
def test_bend_still_refuses_a_non_last_leg_of_its_side(axis):
    t = su2()  # OUT axes (0, 2), IN axes (1, 3)
    with pytest.raises(ValueError, match=f"axis {axis}"):
        bend(t, axis)


def test_repartition_never_triggers_the_bend_precondition():
    t = su2()
    for outputs, inputs in SPLITS:
        t.repartition(outputs, inputs)


def test_identity_repartition_moves_nothing():
    t = su2()
    r = t.repartition(t.structure.out_axes, t.structure.in_axes)
    assert r == t.transpose((0, 2, 1, 3))
    assert [leg.dual for leg in r.legs] == [t.legs[i].dual for i in (0, 2, 1, 3)]


def test_abelian_providers_still_take_the_bend_unique_path():
    """#32's exact-1.0 criterion, unchanged: no float arithmetic touches a block."""
    for provider, space in (
        (U1, GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})),
        (Trivial, GradedSpace.new(Trivial, {tenet.symmetry.TrivialSector(): 2})),
    ):
        legs = (Leg(space, OUT), Leg(space, IN), Leg(space, OUT))
        t = SymmetricTensor.random(legs, seed=17)
        plan = bend_plan(t.structure, 2)
        assert all(coeff == 1.0 and isinstance(coeff, float) for _, _, coeff in plan.terms)
        r = bend(t, 2)
        for src, dst, _ in plan.terms:
            np.testing.assert_array_equal(r.blocks[dst], np.transpose(t.blocks[src], (0, 1, 2)))
        assert provider.bend_right(t.structure.block_order[0], dual=True)[0][1] == 1.0


# --- map-level (#29) and composition (#30) consistency ---------------------------------


def test_map_level_shapes_follow_the_new_layout():
    t = su2()
    r = t.repartition((0, 1, 2), (3,))
    layout = tenet.map_layout(r.structure)
    mats = tenet.to_matrices(r)
    assert set(mats) == set(layout.sectors)
    for c, m in mats.items():
        assert tuple(m.shape) == layout.shape(c)
    assert tenet.allclose(tenet.from_matrices(r.structure, mats), r)


def test_map_level_empty_side_gives_column_and_row_vectors():
    t = su2()
    column = t.repartition((0, 1, 2, 3), ())
    assert tenet.map_layout(column.structure).sectors == (SU2.unit,)
    assert tenet.to_matrices(column)[SU2.unit].shape[1] == 1
    row = t.repartition((), (0, 1, 2, 3))
    assert tenet.to_matrices(row)[SU2.unit].shape[0] == 1


def test_composition_agrees_with_bending_on_the_shared_index():
    a = SymmetricTensor.random((Leg(V, OUT), Leg(U, OUT), Leg(V, IN)), seed=47)
    b = SymmetricTensor.random((Leg(V, OUT), Leg(U, IN)), seed=53)
    da, db = a.to_dense(), b.to_dense()
    np.testing.assert_allclose((a @ b).to_dense(), np.tensordot(da, db, ([2], [0])), atol=1e-10)

    a2 = a.repartition((0,), (1, 2))
    b2 = b.repartition((1, 0), ())
    assert a2.as_map().domain.matches(b2.as_map().codomain) is None
    assert all(leg.dual for leg in (a2.legs[1], b2.legs[0]))
    composed = a2 @ b2
    assert composed.legs == (a.legs[0],)
    np.testing.assert_allclose(
        np.asarray(composed.to_dense()), np.tensordot(da, db, ([1, 2], [1, 0])), atol=1e-10
    )


# --- plan and module hygiene ------------------------------------------------------------


def test_bend_plan_stays_cached_frozen_and_array_free_for_su2():
    s = su2().structure
    plan = bend_plan(s, 2)
    assert bend_plan(s, 2) is plan
    assert hash(plan) == hash(bend_plan(s, 2))
    for _, _, coeff in plan.terms:
        assert isinstance(coeff, complex | float)
        assert not hasattr(coeff, "shape")
    assert not dataclasses.fields(SU2Provider)[1:]  # bend_braided added no provider field


def test_bend_braided_is_provider_agnostic():
    src = pathlib.Path(bend_braided.__globals__["__file__"]).read_text()
    assert "SU2" not in src
    assert "if provider ==" not in src
    assert "to_dense()" not in src


# --- backends -----------------------------------------------------------------------------


@pytest.mark.parametrize(("outputs", "inputs"), MOVING)
def test_jax_repartition_matches_numpy(outputs, inputs):
    use_jax()
    t = su2(seed=37)
    r = t.to_backend("jax").repartition(outputs, inputs)
    assert r.backend == "jax"
    np.testing.assert_allclose(
        np.asarray(r.to_backend("numpy").to_dense()), dense_oracle(t, outputs, inputs), atol=1e-11
    )


@pytest.mark.parametrize("axis", [2, 3])
def test_jax_round_trip(axis):
    use_jax()
    t = su2(seed=19)
    there = bend(t.to_backend("jax"), axis)
    back = bend(there, there.ndim - 1)
    assert back.backend == "jax"
    assert tenet.allclose(back.transpose(restore(t.ndim, axis)).to_backend("numpy"), t, atol=1e-12)
