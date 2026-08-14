"""Tests for ``RepartitionPlan``/``repartition_plan`` — one per acceptance criterion of #61.

No new mathematics: #61 composes the transpose→bend→transpose chain at plan time
instead of executing it step by step, so the reference is the *sequential* chain
itself, spelled out in :func:`sequential` below exactly as ``repartition`` used to
run it. What is tested is that the composition is faithful (exactly so wherever
every coefficient is 1, to 1e-13 where SU(2)'s B-symbols are involved), that the
plan is static/array-free/cached like every other plan, that fusing never grows
the term count, and that a repartition which bends nothing does not build a plan.
"""

import dataclasses

import numpy as np
import pytest

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.permutation import permutation_plan
from tenet.ops.repartition import (
    RepartitionPlan,
    bend,
    bend_plan,
    repartition,
    repartition_plan,
)
from tenet.structure import TensorStructure
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

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V_SU2 = GradedSpace.new(SU2, {HALF: 2, ONE: 3})
W_SU2 = GradedSpace.new(SU2, {SINGLET: 2, HALF: 3})
V_U1 = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
W_U1 = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})
V_FZ2 = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 3})
W_FZ2 = GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 1})
V_TR = GradedSpace.new(Trivial, {TrivialSector(): 2})
W_TR = GradedSpace.new(Trivial, {TrivialSector(): 3})

UF = ProductProvider((U1, fZ2))


def uf(charge: int, parity: int) -> ProductSector:
    return ProductSector((U1Sector(charge), FZ2Sector(parity)))


V_UF = GradedSpace.new(UF, {uf(0, 0): 2, uf(1, 1): 3})
W_UF = GradedSpace.new(UF, {uf(0, 0): 3, uf(1, 1): 1})


def legs(V, W, dual=False):
    """Rank 4, sides interleaved, so every split below moves at least one leg."""
    return (Leg(V, OUT, dual=dual, name="a"), Leg(W, IN, name="b"), Leg(W, OUT), Leg(V, IN))


# splits that keep each side ascending: within-side order is left alone so the
# non-Abelian providers are asked the same questions as the Abelian ones
SPLITS = (
    ((0, 1, 2), (3,)),
    ((0, 2), (1, 3)),
    ((0,), (1, 2, 3)),
    ((1, 3), (0, 2)),
    ((0, 1, 2, 3), ()),
    ((), (0, 1, 2, 3)),
)
ALIGNED = ((0, 2), (1, 3))  # the sides each leg already has: zero crossings

ABELIAN = [
    pytest.param(V_TR, W_TR, id="trivial"),
    pytest.param(V_U1, W_U1, id="u1"),
    pytest.param(V_FZ2, W_FZ2, id="fz2"),
]


def sequential(t: SymmetricTensor, outputs, inputs) -> SymmetricTensor:
    """``repartition`` as it ran before #61: one tensor operation per step."""
    labels = list(range(t.ndim))
    want = {ax: OUT for ax in outputs} | {ax: IN for ax in inputs}
    for ax in [a for a in range(t.ndim) if t.legs[a].side is not want[a]]:
        p = labels.index(ax)
        t = t.transpose(tuple(i for i in range(t.ndim) if i != p) + (p,))
        labels.append(labels.pop(p))
        t = bend(t, t.ndim - 1)
    position = {a: p for p, a in enumerate(labels)}
    return t.transpose(tuple(position[ax] for ax in (*outputs, *inputs)))


def sequential_terms(structure: TensorStructure, outputs, inputs) -> int:
    """Total number of terms the sequential chain executes, summed over its steps."""
    ndim = len(structure.legs)
    labels = list(range(ndim))
    want = {ax: OUT for ax in outputs} | {ax: IN for ax in inputs}
    total = 0
    for ax in [a for a in range(ndim) if structure.legs[a].side is not want[a]]:
        p = labels.index(ax)
        plan = permutation_plan(structure, tuple(i for i in range(ndim) if i != p) + (p,))
        total += len(plan.terms)
        structure = plan.new_structure
        labels.append(labels.pop(p))
        bplan = bend_plan(structure, ndim - 1)
        total += len(bplan.terms)
        structure = bplan.new_structure
    position = {a: p for p, a in enumerate(labels)}
    plan = permutation_plan(structure, tuple(position[ax] for ax in (*outputs, *inputs)))
    return total + len(plan.terms)


# --- the plan object itself ---------------------------------------------------------


def test_plan_is_frozen_hashable_and_array_free():
    structure = TensorStructure(legs(V_SU2, W_SU2))
    plan = repartition_plan(structure, (0, 2, 3), (1,))

    assert type(plan).__dataclass_params__.frozen
    assert RepartitionPlan.__slots__  # slots=True: no __dict__ to smuggle arrays into
    assert not hasattr(plan, "__dict__")
    hash(plan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.perm = ()

    for field in dataclasses.fields(plan):
        value = getattr(plan, field.name)
        if field.name == "new_structure":
            assert isinstance(value, TensorStructure)
            continue
        assert isinstance(value, tuple) and not hasattr(value, "__array__"), field.name
    assert all(type(a) is int for a in plan.perm)
    assert sorted(plan.perm) == list(range(4))
    for src, dst, coeff in plan.terms:
        assert type(src) is int and type(dst) is int
        assert isinstance(coeff, (int, float, complex)) and not hasattr(coeff, "__array__")


def test_cache_returns_one_object_for_equal_but_independent_structures():
    """Identity, not equality: the cache hits on value, not on object."""
    one = repartition_plan(TensorStructure(legs(V_SU2, W_SU2)), (0, 2, 3), (1,))
    other = repartition_plan(
        TensorStructure(
            (
                Leg(GradedSpace.new(SU2, dict(V_SU2.sectors)), OUT, name="a"),
                Leg(W_SU2, IN, name="b"),
                Leg(GradedSpace.new(SU2, dict(W_SU2.sectors)), OUT),
                Leg(V_SU2, IN),
            )
        ),
        (0, 2, 3),
        (1,),
    )
    assert one is other


# --- the fused result against the sequential chain -----------------------------------


@pytest.mark.parametrize(("V", "W"), ABELIAN)
@pytest.mark.parametrize(("outputs", "inputs"), SPLITS)
def test_abelian_blocks_are_exactly_equal_to_the_sequential_result(V, W, outputs, inputs):
    """Every coefficient is exactly 1 (#32), so fusing changes no floating-point bit."""
    t = SymmetricTensor.random(legs(V, W), seed=5)
    got, want = repartition(t, outputs, inputs), sequential(t, outputs, inputs)
    assert got.structure == want.structure
    for a, b in zip(got.blocks, want.blocks, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("dual", [False, True], ids=["direct", "dual-half-integer"])
@pytest.mark.parametrize(("outputs", "inputs"), SPLITS)
def test_su2_matches_the_sequential_result_to_1e_13(dual, outputs, inputs):
    """``dual=True`` on a half-integer leg is the #38 Frobenius-Schur sign case.

    The tolerance is 1e-13 of the *tensor's* scale, not of each entry: fusing
    multiplies the chain's coefficients in Python and applies them once, so an
    entry that is itself a near-cancellation of two O(1) numbers has no relative
    accuracy left to compare, while the deviation stays at ~1 ulp of the norm
    (measured max |fused - sequential| here: 9e-16 against entries of order 1).
    """
    t = SymmetricTensor.random(legs(V_SU2, W_SU2, dual=dual), seed=7)
    assert t.legs[0].dual is dual and HALF in dict(V_SU2.sectors)
    got, want = repartition(t, outputs, inputs), sequential(t, outputs, inputs)
    assert got.structure == want.structure
    scale = max(np.abs(b).max() for b in want.blocks)
    for a, b in zip(got.blocks, want.blocks, strict=True):
        assert np.allclose(a, b, rtol=1e-13, atol=1e-13 * scale)


# --- fusing never grows the work -----------------------------------------------------


TERM_MATRIX = [
    pytest.param(V_SU2, W_SU2, False, SPLITS, id="su2"),
    pytest.param(V_SU2, W_SU2, True, SPLITS, id="su2-dual"),
    pytest.param(V_FZ2, W_FZ2, False, SPLITS, id="fz2"),
    # ProductProvider implements no BendingCoefficients, so only the aligned split
    pytest.param(V_UF, W_UF, False, (ALIGNED,), id="product"),
]


@pytest.mark.parametrize(("V", "W", "dual", "splits"), TERM_MATRIX)
def test_fused_term_count_never_exceeds_the_sequential_total(V, W, dual, splits, record_property):
    structure = TensorStructure(legs(V, W, dual=dual))
    for outputs, inputs in splits:
        fused = len(repartition_plan(structure, outputs, inputs).terms)
        total = sequential_terms(structure, outputs, inputs)
        assert fused <= total, (outputs, inputs, fused, total)
        record_property(f"{V.provider.name}{'-dual' if dual else ''}{outputs}", f"{fused}/{total}")


# --- the aligned path pays nothing ---------------------------------------------------


def test_no_crossing_leg_builds_no_plan():
    """The early return: a repartition that bends nothing is a plain transpose."""
    t = SymmetricTensor.random(legs(V_SU2, W_SU2), seed=3)
    before = repartition_plan.cache_info()
    got = repartition(t, *ALIGNED)
    assert repartition_plan.cache_info() == before
    want = t.transpose((*ALIGNED[0], *ALIGNED[1]))
    assert got.structure == want.structure
    for a, b in zip(got.blocks, want.blocks, strict=True):
        assert np.array_equal(a, b)
