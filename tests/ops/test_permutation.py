"""Tests for axis permutation — issue #21.

The dense oracle (``np.transpose(T.to_dense(), p)``) is what pins the ``axes``
convention; everything else here is structure.
"""

import dataclasses
import itertools
import pathlib

import numpy as np
import pytest

import tenet
import tenet.ops.permutation
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.fusion_tree import FusionTree
from tenet.ops.permutation import PermutationPlan, permutation_plan
from tenet.symmetry import (
    SU2,
    U1,
    CapabilityError,
    PermutationCoefficients,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
)

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 2, ONE: 3})
W = GradedSpace.new(SU2, {SINGLET: 2, HALF: 3})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
Q2 = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})

# interleaved sides throughout: OUT at axes 0, 2 and IN at axes 1, 3
SU2_LEGS = (Leg(V, OUT, name="a"), Leg(W, IN, name="b"), Leg(W, OUT), Leg(V, IN))
U1_LEGS = (Leg(Q, OUT), Leg(Q2, IN), Leg(Q2, OUT), Leg(Q, IN))
TRIV_LEGS = tuple(
    Leg(GradedSpace.new(Trivial, {TrivialSector(): m}), side)
    for m, side in ((2, OUT), (3, IN), (4, OUT), (5, IN))
)

# permutations of (0, 1, 2, 3) that interleave the subsequences (0, 2) and (1, 3)
# without disturbing either — case A, available to every provider
CASE_A = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2), (1, 0, 2, 3), (1, 0, 3, 2), (1, 3, 0, 2))
ALL_PERMS = tuple(itertools.permutations(range(4)))


def su2() -> SymmetricTensor:
    return SymmetricTensor.random(SU2_LEGS, seed=7)


def inverse(p):
    out = [0] * len(p)
    for i, a in enumerate(p):
        out[a] = i
    return tuple(out)


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


@dataclasses.dataclass(frozen=True, slots=True)
class FusionOnly:
    """FusionProvider + QuantumDimension, nothing else — no permutation capability."""

    name: str = "FusionOnly"

    @property
    def unit(self):
        return TrivialSector()

    def dual(self, a):
        return a

    def fusion(self, a, b):
        return (TrivialSector(),)

    def n_symbol(self, a, b, c):
        return 1

    def qdim(self, a):
        return 1.0


# --- spelling, validation, legs ------------------------------------------------


def test_spellings_are_equivalent():
    t = su2()
    p = (1, 0, 3, 2)
    assert t.transpose(1, 0, 3, 2) == t.transpose(p)
    assert t.transpose(p) == tenet.transpose(t, p)
    assert t.transpose(list(p)) == t.transpose(p)


def test_empty_call_reverses():
    t = SymmetricTensor.random(U1_LEGS, seed=11)
    assert t.transpose() == t.transpose((3, 2, 1, 0))
    assert tenet.transpose(t) == t.transpose((3, 2, 1, 0))
    assert t.transpose(None) == t.transpose((3, 2, 1, 0))
    # reversal is a within-side permutation, so SU(2) refuses it rather than lying
    with pytest.raises(CapabilityError):
        su2().transpose()


@pytest.mark.parametrize(
    ("axes", "needle"),
    [((0, 1), "length 2"), ((0, 1, 2, 9), "axis 9"), ((0, 1, 1, 2), "axis 1")],
)
def test_axes_validation(axes, needle):
    with pytest.raises(ValueError, match=needle):
        su2().transpose(axes)


def test_legs_travel_and_side_never_changes():
    t = su2()
    p = (1, 0, 3, 2)
    r = t.transpose(p)
    assert r.legs == tuple(t.legs[i] for i in p)
    assert {leg.side for leg in r.legs} == {OUT, IN}
    assert [leg.side for leg in r.legs] == [t.legs[i].side for i in p]
    assert r.legs[0].name == "b" and r.legs[1].name == "a"
    # side moves as *positions*, never as leg metadata
    assert r.structure.out_axes == (1, 3) and t.structure.out_axes == (0, 2)
    assert r.codomain == t.codomain and r.domain == t.domain


# --- case A --------------------------------------------------------------------


def test_case_a_su2_block_order_identical_and_blocks_transposed():
    t = su2()
    p = (1, 0, 3, 2)
    r = t.transpose(p)
    assert r.structure.block_order == t.structure.block_order
    assert r.blocks is not t.blocks
    for old, new in zip(t.blocks, r.blocks, strict=True):
        np.testing.assert_array_equal(new, np.transpose(old, p))


@pytest.mark.parametrize("p", CASE_A)
def test_case_a_su2_dense_oracle(p):
    t = su2()
    np.testing.assert_allclose(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p), atol=1e-12)


def test_case_a_preserves_norm_and_commutes_with_conj():
    t = su2() + su2() * 1j
    for p in CASE_A:
        assert abs(tenet.norm(t.transpose(p)) - tenet.norm(t)) < 1e-12
        assert tenet.allclose(t.transpose(p).conj(), t.conj().transpose(p))


def test_case_a_needs_no_permutation_capability():
    provider = FusionOnly()
    assert not isinstance(provider, PermutationCoefficients)
    space = GradedSpace.new(provider, {TrivialSector(): 3})
    legs = (Leg(space, OUT), Leg(space, IN), Leg(space, OUT))
    t = SymmetricTensor.random(legs, seed=3)
    r = t.transpose((1, 0, 2))
    assert r.structure.block_order == t.structure.block_order
    np.testing.assert_array_equal(r.blocks[0], np.transpose(t.blocks[0], (1, 0, 2)))


# --- case B refusal ------------------------------------------------------------


@pytest.mark.parametrize(("p", "side"), [((2, 1, 0, 3), "OUT"), ((0, 3, 2, 1), "IN")])
def test_case_b_su2_refused(p, side):
    t = su2()
    with pytest.raises(CapabilityError) as excinfo:
        t.transpose(p)
    message = str(excinfo.value)
    assert "SU2" in message
    assert str(p) in message and side in message
    assert "Milestone 4" in message and "F-moves" in message


def test_case_b_su2_refusal_is_not_a_silent_result():
    t = su2()
    results = []
    for p in ALL_PERMS:
        try:
            results.append(t.transpose(p))
        except CapabilityError:
            pass
    assert len(results) == len(CASE_A)


# --- case B, Abelian -----------------------------------------------------------


@pytest.mark.parametrize("p", ALL_PERMS)
def test_case_b_u1_exhaustive_dense_oracle(p):
    t = SymmetricTensor.random(U1_LEGS, seed=11)
    np.testing.assert_allclose(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p), atol=1e-12)


@pytest.mark.parametrize("p", ALL_PERMS)
def test_case_b_trivial_exhaustive_dense_oracle(p):
    t = SymmetricTensor.random(TRIV_LEGS, seed=13)
    np.testing.assert_allclose(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p), atol=1e-12)


def test_case_b_remaps_rather_than_reorders():
    t = SymmetricTensor.random(U1_LEGS, seed=11)
    p = (2, 1, 0, 3)  # swaps the two OUT legs: a genuine within-side permutation
    r = t.transpose(p)
    assert r.structure.block_order != t.structure.block_order
    old = sorted(
        tuple(s[i] for i in p) for s in map(t.structure.block_shape, t.structure.block_order)
    )
    new = sorted(map(r.structure.block_shape, r.structure.block_order))
    assert old == new


# --- algebra -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("legs", "perms"), [(U1_LEGS, ALL_PERMS), (TRIV_LEGS, ALL_PERMS), (SU2_LEGS, CASE_A)]
)
def test_round_trip(legs, perms):
    t = SymmetricTensor.random(legs, seed=5)
    for p in perms:
        assert tenet.allclose(t.transpose(p).transpose(inverse(p)), t)


@pytest.mark.parametrize(("legs", "perms"), [(U1_LEGS, ALL_PERMS), (SU2_LEGS, CASE_A)])
def test_composition(legs, perms):
    t = SymmetricTensor.random(legs, seed=5)
    checked = 0
    for p, q in itertools.product(perms, repeat=2):
        composed = tuple(p[i] for i in q)
        try:
            # q is case A for the *original* leg pattern, which need not be case A
            # for t.transpose(p)'s pattern; those pairs are legitimately refused
            right = t.transpose(composed)
            left = t.transpose(p).transpose(q)
        except CapabilityError:
            continue
        assert tenet.allclose(left, right)
        checked += 1
    assert checked > len(perms)


@pytest.mark.parametrize("legs", [SU2_LEGS, U1_LEGS, TRIV_LEGS])
def test_identity_permutation(legs):
    t = SymmetricTensor.random(legs, seed=2)
    assert t.transpose(0, 1, 2, 3) == t


# --- the plan ------------------------------------------------------------------


def test_plan_is_cached_frozen_and_array_free():
    s = su2().structure
    plan = permutation_plan(s, (1, 0, 3, 2))
    assert permutation_plan(s, (1, 0, 3, 2)) is plan
    assert hash(plan) == hash(permutation_plan(s, (1, 0, 3, 2)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.axes = (0, 1, 2, 3)
    for field in dataclasses.fields(PermutationPlan):
        value = getattr(plan, field.name)
        assert not hasattr(value, "shape"), field.name
    assert plan.terms == tuple((i, i, 1.0) for i in range(s.num_blocks))
    assert plan.new_structure.legs == tuple(s.legs[i] for i in (1, 0, 3, 2))


def test_case_b_plan_terms_are_a_bijection():
    s = SymmetricTensor.random(U1_LEGS, seed=11).structure
    plan = permutation_plan(s, (2, 1, 0, 3))
    assert len(plan.terms) == s.num_blocks
    assert sorted(dst for _, dst, _ in plan.terms) == list(range(s.num_blocks))
    assert all(coeff == 1.0 for _, _, coeff in plan.terms)


# --- the capability ------------------------------------------------------------


def test_permutation_coefficients_is_runtime_checkable():
    assert isinstance(U1, PermutationCoefficients)
    assert isinstance(Trivial, PermutationCoefficients)
    assert not isinstance(SU2, PermutationCoefficients)


@pytest.mark.parametrize(
    ("provider", "uncoupled"),
    [
        (U1, (U1Sector(1), U1Sector(-1), U1Sector(2))),
        (Trivial, (TrivialSector(), TrivialSector(), TrivialSector())),
    ],
)
def test_permute_tree_single_unit_term(provider, uncoupled):
    coupled = tenet.coupled_sectors(provider, uncoupled)[0]
    tree = tenet.fusion_trees(provider, uncoupled, coupled)[0]
    perm = (2, 0, 1)
    terms = provider.permute_tree(tree, perm)
    assert len(terms) == 1
    permuted, coeff = terms[0]
    assert coeff == 1.0
    assert isinstance(permuted, FusionTree)
    permuted.validate(provider)
    assert permuted.uncoupled == tuple(tree.uncoupled[i] for i in perm)
    assert permuted.coupled == tree.coupled


# --- module hygiene ------------------------------------------------------------


def test_module_is_numpy_free_and_provider_agnostic():
    src = pathlib.Path(tenet.ops.permutation.__file__).read_text()
    assert "np." not in src
    assert "import numpy" not in src
    assert "to_dense()" not in src
    assert "U1Provider" not in src and "SU2Provider" not in src
    assert "if provider ==" not in src
    assert 'ar.do("transpose"' in src


# --- backends ------------------------------------------------------------------


@pytest.mark.parametrize("p", CASE_A)
def test_jax_case_a_su2(p):
    use_jax()
    t = su2()
    r = t.to_backend("jax").transpose(p)
    assert r.backend == "jax"
    np.testing.assert_allclose(
        r.to_backend("numpy").to_dense(), np.transpose(t.to_dense(), p), atol=1e-12
    )


@pytest.mark.parametrize("p", ALL_PERMS)
def test_jax_case_b_u1_exhaustive(p):
    use_jax()
    t = SymmetricTensor.random(U1_LEGS, seed=11)
    r = t.to_backend("jax").transpose(p)
    assert r.backend == "jax"
    np.testing.assert_allclose(
        r.to_backend("numpy").to_dense(), np.transpose(t.to_dense(), p), atol=1e-12
    )
