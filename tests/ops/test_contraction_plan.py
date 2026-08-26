"""Tests for ``ContractionPlan``/``contraction_plan`` — one per acceptance criterion of #53.

No new mathematics is tested here, deliberately: #53 moved code, and #51/#52's
dense oracles are its regression suite. What *is* tested is the planning
contract — array-freeness, cache identity, prediction of the output structure
without executing, refusal before one sub-plan is built, and the reuse the
docs/design.md's "Plan caching" section promises, counted with ``cache_info()`` instead
of asserted in a docstring.
"""

import dataclasses
import pathlib
import sys

import numpy as np
import pytest
from helpers import NoBendProvider

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import map_layout
from tenet.ops.contraction import ContractionPlan, contraction_plan
from tenet.ops.permutation import permutation_plan
from tenet.ops.repartition import repartition_plan
from tenet.structure import TensorStructure
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

# --- spaces, one per provider ------------------------------------------------------

V_SU2 = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 1})
W_SU2 = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 2})
V_U1 = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
W_U1 = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 2})
V_FZ2 = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 1})
W_FZ2 = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 2})
V_TR = GradedSpace.new(Trivial, {TrivialSector(): 2})
W_TR = GradedSpace.new(Trivial, {TrivialSector(): 3})

UF = ProductProvider((U1, fZ2))


def uf(charge: int, parity: int) -> ProductSector:
    return ProductSector((U1Sector(charge), FZ2Sector(parity)))


V_UF = GradedSpace.new(UF, {uf(0, 0): 2, uf(1, 1): 1})
W_UF = GradedSpace.new(UF, {uf(0, 0): 1, uf(1, 1): 2})

# The bend-refusal vehicle. #312 forwarded ``BendingCoefficients`` through products, so
# ``UF`` bends like any other provider now; ``helpers.NoBendProvider`` keeps its sectors
# and its fZ2 signs and withholds exactly that capability.
NB = NoBendProvider(UF)
V_NB = GradedSpace.new(NB, {uf(0, 0): 2, uf(1, 1): 1})
W_NB = GradedSpace.new(NB, {uf(0, 0): 1, uf(1, 1): 2})

# ``(id, V, W, bends)`` — every provider, with a bending pattern only where the
# provider implements BendingCoefficients. Since #312 that is every row including the
# product; ``no-bend`` is the stub that still does not.
PROVIDERS = [
    pytest.param(V_SU2, W_SU2, True, id="su2"),
    pytest.param(V_U1, W_U1, True, id="u1"),
    pytest.param(V_FZ2, W_FZ2, True, id="fz2"),
    pytest.param(V_TR, W_TR, True, id="trivial"),
    pytest.param(V_UF, W_UF, True, id="product"),
    pytest.param(V_NB, W_NB, False, id="no-bend"),
]


def composition_shaped(V, W):
    """``a``'s whole domain against ``b``'s whole codomain: zero bends, any provider."""
    return ((Leg(V, OUT), Leg(W, IN)), (Leg(W, OUT), Leg(V, IN)), ((1,), (0,)))


def bending(V, W):
    """A free IN leg on ``a`` and a contracted OUT leg on ``a``: both must cross."""
    return ((Leg(V, OUT, dual=True), Leg(W, IN)), (Leg(V, OUT), Leg(W, IN)), ((0,), (0,)))


def pair(case, seeds=(0, 1)):
    a_legs, b_legs, axes = case
    return (
        SymmetricTensor.random(a_legs, seed=seeds[0]),
        SymmetricTensor.random(b_legs, seed=seeds[1]),
        axes,
    )


# --- the plan object itself ---------------------------------------------------------


def test_plan_is_frozen_hashable_and_array_free():
    a, b, axes = pair(composition_shaped(V_SU2, W_SU2))
    plan = contraction_plan(a.structure, b.structure, axes)

    assert type(plan).__dataclass_params__.frozen
    assert ContractionPlan.__slots__  # slots=True: no __dict__ to smuggle arrays into
    assert not hasattr(plan, "__dict__")
    hash(plan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.final_transpose = ()

    for field in dataclasses.fields(plan):
        value = getattr(plan, field.name)
        if field.name == "new_structure":
            assert isinstance(value, TensorStructure)
            continue
        assert isinstance(value, tuple) and all(type(x) is int for x in value), field.name
        assert not hasattr(value, "__array__"), field.name


def test_cache_returns_one_object_for_equal_but_independent_structures():
    """Identity, not equality: the cache hits on value, not on object."""
    legs_a = (Leg(V_SU2, OUT), Leg(W_SU2, IN))
    legs_b = (Leg(W_SU2, OUT), Leg(V_SU2, IN))
    one = contraction_plan(TensorStructure(legs_a), TensorStructure(legs_b), ((1,), (0,)))
    # rebuilt from scratch: new Leg objects, new spaces, new structures
    other = contraction_plan(
        TensorStructure((Leg(GradedSpace.new(SU2, dict(V_SU2.sectors)), OUT), Leg(W_SU2, IN))),
        TensorStructure((Leg(W_SU2, OUT), Leg(GradedSpace.new(SU2, dict(V_SU2.sectors)), IN))),
        ((1,), (0,)),
    )
    assert one is other


def test_no_planning_package_exists_and_the_decision_is_recorded():
    assert not (pathlib.Path(tenet.__file__).parent / "planning").exists()
    # The note lives in a `#` comment, not the docstring: it is an engineering decision,
    # not part of the rendered API contract. Pin it against the source text.
    src = pathlib.Path(sys.modules[ContractionPlan.__module__].__file__).read_text()
    assert "Simplification: no ``src/tenet/planning/`` package" in src


# --- new_structure is predicted, not observed ---------------------------------------


@pytest.mark.parametrize(("V", "W", "bends"), PROVIDERS)
def test_predicted_new_structure_equals_the_executed_result(V, W, bends):
    cases = [composition_shaped(V, W)] + ([bending(V, W)] if bends else [])
    for case in cases:
        a, b, axes = pair(case)
        plan = contraction_plan(a.structure, b.structure, axes)
        got = tenet.tensordot(a, b, axes)
        assert plan.new_structure.legs == got.legs
        assert plan.new_structure == got.structure


# --- refusals come from the plan, before any block or sub-plan moves ------------------


REFUSALS = [
    pytest.param(
        ((Leg(V_SU2, OUT), Leg(W_SU2, IN)), (Leg(W_SU2, OUT), Leg(V_SU2, IN))),
        ((5,), (0,)),
        ValueError,
        "out of range",
        id="axis-out-of-range",
    ),
    pytest.param(
        ((Leg(V_SU2, OUT), Leg(W_SU2, IN)), (Leg(W_SU2, OUT), Leg(V_SU2, IN))),
        -1,
        ValueError,
        "is negative",
        id="negative-integer-form",
    ),
    pytest.param(
        ((Leg(V_SU2, OUT), Leg(W_SU2, IN)), (Leg(W_SU2, OUT), Leg(V_SU2, IN))),
        ((0, 0), (0, 1)),
        ValueError,
        "appears twice",
        id="repeated-axis",
    ),
    pytest.param(
        ((Leg(V_SU2, OUT), Leg(W_SU2, IN)), (Leg(W_SU2, OUT), Leg(V_SU2, IN))),
        ((0, 1), (0,)),
        ValueError,
        "different lengths",
        id="unpaired-axes",
    ),
    pytest.param(
        ((Leg(V_SU2, OUT), Leg(W_SU2, IN)), (Leg(W_U1, OUT), Leg(V_U1, IN))),
        ((1,), (0,)),
        ValueError,
        "no common fusion category",
        id="provider-mismatch",
    ),
    pytest.param(
        ((Leg(V_SU2, OUT), Leg(W_SU2, IN)), (Leg(W_SU2, OUT), Leg(V_SU2, IN))),
        ((1,), (1,)),
        ValueError,
        "do not contract",
        id="not-contractible",
    ),
    pytest.param(
        ((Leg(V_SU2, OUT), Leg(W_SU2, IN)), (Leg(W_SU2, OUT), Leg(V_SU2, IN))),
        ((0, 1), (1, 0)),
        ValueError,
        "no free leg",
        id="no-free-leg",
    ),
    pytest.param(
        ((Leg(V_NB, OUT, dual=True), Leg(W_NB, IN)), (Leg(V_NB, OUT), Leg(W_NB, IN))),
        ((0,), (0,)),
        CapabilityError,
        "BendingCoefficients",
        id="bend-without-capability",
    ),
]


@pytest.mark.parametrize(("legs", "axes", "error", "fragment"), REFUSALS)
def test_refusal_is_raised_by_the_plan_with_the_same_message_as_tensordot(
    legs, axes, error, fragment
):
    a = SymmetricTensor.random(legs[0], seed=0)
    b = SymmetricTensor.random(legs[1], seed=1)

    with pytest.raises(error) as from_plan:
        contraction_plan(a.structure, b.structure, axes)
    with pytest.raises(error) as from_tensordot:
        tenet.tensordot(a, b, axes)

    assert fragment in str(from_plan.value)
    assert str(from_plan.value) == str(from_tensordot.value)


def test_refusal_happens_before_one_sub_plan_is_built():
    """``permutation_plan`` is never reached: not one sub-plan, let alone a block."""
    a = SymmetricTensor.random((Leg(V_NB, OUT, dual=True), Leg(W_NB, IN)), seed=0)
    b = SymmetricTensor.random((Leg(V_NB, OUT), Leg(W_NB, IN)), seed=1)
    contraction_plan.cache_clear()

    before = permutation_plan.cache_info()
    with pytest.raises(CapabilityError):
        contraction_plan(a.structure, b.structure, ((0,), (0,)))
    with pytest.raises(CapabilityError):
        tenet.tensordot(a, b, ((0,), (0,)))
    assert permutation_plan.cache_info().misses == before.misses
    assert permutation_plan.cache_info().hits == before.hits


# --- plan reuse, counted ------------------------------------------------------------


def test_fifty_calls_are_one_miss_and_forty_nine_hits_for_every_plan_cache():
    """docs/design.md "Plan caching" as a checked property, over the whole sub-plan chain."""
    case = bending(V_SU2, W_SU2)
    a_legs, b_legs, axes = case
    tensors = [
        (SymmetricTensor.random(a_legs, seed=s), SymmetricTensor.random(b_legs, seed=100 + s))
        for s in range(50)
    ]
    caches = {
        "contraction_plan": contraction_plan,
        "permutation_plan": permutation_plan,
        # bend_plan is a plan-time sub-plan of repartition_plan since #61, so it is
        # consulted while that plan is built, not once per contraction
        "repartition_plan": repartition_plan,
        "map_layout": map_layout,
    }

    # the first call populates every cache; calls 2..50 must add no miss at all
    tenet.tensordot(*tensors[0], axes)
    first = {name: fn.cache_info() for name, fn in caches.items()}
    tenet.tensordot(*tensors[1], axes)
    per_call = {name: caches[name].cache_info().hits - first[name].hits for name in caches}

    for a, b in tensors[2:]:
        tenet.tensordot(a, b, axes)

    for name, fn in caches.items():
        info = fn.cache_info()
        assert info.misses == first[name].misses, f"{name} missed again after the first call"
        assert info.hits == first[name].hits + 49 * per_call[name], name
        assert per_call[name] > 0, f"{name} was never consulted; this case does not exercise it"

    # for contraction_plan the counting is exact: one entry, one miss, 49 hits
    assert per_call["contraction_plan"] == 1


def test_integer_type_does_not_fragment_the_cache():
    a, b, axes = pair(composition_shaped(V_U1, W_U1))
    plain = contraction_plan(a.structure, b.structure, ((1,), (0,)))
    before = contraction_plan.cache_info()
    numpy_axes = ((np.int64(1),), (np.int64(0),))
    assert contraction_plan(a.structure, b.structure, numpy_axes) is plain
    assert contraction_plan.cache_info().misses == before.misses
    # and the plan's own fields stay plain ints, whatever came in
    assert all(type(x) is int for x in plain.a_inputs)


# --- plan reuse under jit -----------------------------------------------------------


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  registration is the import's side effect

    return jax


def test_jit_traces_the_plan_once_and_a_new_treedef_replans():
    jax = use_jax()
    a_legs, b_legs, axes = composition_shaped(V_SU2, W_SU2)

    @jax.jit
    def contract(x, y):
        return tenet.tensordot(x, y, axes)

    def to_jax(t):
        return SymmetricTensor(t.structure, tuple(jax.numpy.asarray(x) for x in t.blocks))

    before = contraction_plan.cache_info()
    for s in range(10):
        a = to_jax(SymmetricTensor.random(a_legs, seed=s))
        b = to_jax(SymmetricTensor.random(b_legs, seed=100 + s))
        contract(a, b)
    assert contraction_plan.cache_info().misses - before.misses <= 1

    # a changed degeneracy is a new treedef: replanning here is correct, not a leak
    other = GradedSpace.new(SU2, {SU2Sector(0): 3, SU2Sector(1): 1})
    mid = contraction_plan.cache_info().misses
    contract(
        to_jax(SymmetricTensor.random((Leg(other, OUT), Leg(W_SU2, IN)), seed=7)),
        to_jax(SymmetricTensor.random(b_legs, seed=8)),
    )
    assert contraction_plan.cache_info().misses == mid + 1


def test_jit_result_matches_the_numpy_backend():
    jax = use_jax()
    a_legs, b_legs, axes = composition_shaped(V_SU2, W_SU2)
    a = SymmetricTensor.random(a_legs, seed=2)
    b = SymmetricTensor.random(b_legs, seed=3)
    want = tenet.tensordot(a, b, axes)

    def to_jax(t):
        return SymmetricTensor(t.structure, tuple(jax.numpy.asarray(x) for x in t.blocks))

    got = jax.jit(lambda x, y: tenet.tensordot(x, y, axes))(to_jax(a), to_jax(b))
    assert got.structure == want.structure
    for x, y in zip(got.blocks, want.blocks, strict=True):
        np.testing.assert_allclose(np.asarray(x), y, atol=1e-12)


# --- provider identity separates plans ----------------------------------------------


def test_provider_identity_separates_plans_and_results():
    """U(1) charges 0/1 and fZ2 parities 0/1 at identical degeneracies.

    Same dimensions, same sector counts, different fusion rules — which is the
    property the "gauge belongs in cache keys" note was reaching for. Provider
    identity is already inside every cache key, via ``Leg`` → ``GradedSpace``.
    """

    def case(V):
        # rank 3 on ``a``: U(1) forbids the (1, 1, ...) blocks that fZ2 allows,
        # since 1 + 1 = 2 leaves the space while 1 + 1 = 0 stays inside it
        return ((Leg(V, OUT), Leg(V, OUT), Leg(V, IN)), (Leg(V, OUT), Leg(V, IN)), ((2,), (0,)))

    a1, b1, axes = pair(case(V_U1))
    a2, b2, _ = pair(case(V_FZ2))
    assert a1.to_dense().shape == a2.to_dense().shape  # indistinguishable by dimension
    assert a1.structure.num_blocks != a2.structure.num_blocks  # distinguishable by fusion

    p1 = contraction_plan(a1.structure, b1.structure, axes)
    p2 = contraction_plan(a2.structure, b2.structure, axes)
    assert p1 is not p2
    assert p1.new_structure != p2.new_structure
    assert p1.new_structure.provider != p2.new_structure.provider

    d1 = tenet.tensordot(a1, b1, axes).to_dense()
    d2 = tenet.tensordot(a2, b2, axes).to_dense()
    assert d1.shape == d2.shape
    assert not np.allclose(d1, d2)
