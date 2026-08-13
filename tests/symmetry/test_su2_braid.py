"""SU(2) ``permute_tree`` via the Artin braid decomposition — issue #36.

The dense oracle (``np.transpose(T.to_dense(), p)``) is the point of the whole
file: within-side SU(2) permutation is a braid, and only a dense round trip
tells a correct expansion from a numerically plausible one.
"""

import dataclasses
import itertools
import pathlib

import numpy as np
import pytest

import tenet
import tenet.ops.permutation
import tenet.symmetry.base
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.fusion_tree import FusionTree, coupled_sectors, fusion_trees
from tenet.ops.permutation import PermutationPlan, permutation_plan
from tenet.symmetry import SU2, PermutationCoefficients, SU2Sector
from tenet.symmetry.base import permute_braided_tree

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 2, ONE: 3})
W = GradedSpace.new(SU2, {SINGLET: 2, HALF: 3})

# same interleaved leg pattern as tests/ops/test_permutation.py: OUT at 0, 2
SU2_LEGS = (Leg(V, OUT, name="a"), Leg(W, IN, name="b"), Leg(W, OUT), Leg(V, IN))
ALL_PERMS = tuple(itertools.permutations(range(4)))
CASE_A = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2), (1, 0, 2, 3), (1, 0, 3, 2), (1, 3, 0, 2))

# rank 5: OUT at axes 0, 1, 2 and IN at axes 3, 4
A5 = GradedSpace.new(SU2, {SINGLET: 1, HALF: 1})
B5 = GradedSpace.new(SU2, {HALF: 1, ONE: 1})
LEGS5 = (Leg(A5, OUT), Leg(B5, OUT), Leg(A5, OUT), Leg(B5, IN), Leg(A5, IN))
PERMS5 = (
    (1, 2, 0, 3, 4),  # 3-cycle within OUT
    (2, 0, 1, 3, 4),  # the other 3-cycle
    (2, 1, 0, 3, 4),  # full reversal within OUT
    (0, 1, 2, 4, 3),  # reversal within IN
    (2, 1, 0, 4, 3),  # both sides reversed
    (1, 0, 2, 3, 4),
    (0, 2, 1, 3, 4),
    (1, 2, 0, 4, 3),
    (2, 0, 1, 4, 3),
    (0, 1, 3, 2, 4),  # within-side reorder plus a changed interleaving
    (4, 2, 1, 0, 3),
)

SMALL = (SINGLET, HALF, ONE)


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


def uncoupled_sets(rank: int, budget: int = 4):
    """All sector tuples of ``rank`` lines from ``SMALL`` with ``sum(dj) <= budget``."""
    return [u for u in itertools.product(SMALL, repeat=rank) if sum(s.two_j for s in u) <= budget]


# --- the capability ------------------------------------------------------------


def test_su2_now_provides_permutation_coefficients():
    assert isinstance(SU2, PermutationCoefficients)


def test_permuted_trees_are_valid_and_relabelled():
    for u in uncoupled_sets(4):
        for coupled in coupled_sectors(SU2, u):
            for tree in fusion_trees(SU2, u, coupled):
                for perm in itertools.permutations(range(4)):
                    terms = SU2.permute_tree(tree, perm)
                    assert terms
                    for permuted, coeff in terms:
                        assert isinstance(permuted, FusionTree)
                        permuted.validate(SU2)
                        assert permuted.uncoupled == tuple(tree.uncoupled[i] for i in perm)
                        assert permuted.coupled == tree.coupled
                        assert isinstance(coeff, (int, float, complex))
                        assert not hasattr(coeff, "shape")


# --- elementary swap -----------------------------------------------------------


@pytest.mark.parametrize(("coupled", "expected"), [(SINGLET, -1.0), (ONE, +1.0)])
def test_rank_two_spot_value(coupled, expected):
    tree = fusion_trees(SU2, (HALF, HALF), coupled)[0]
    ((permuted, coeff),) = SU2.permute_tree(tree, (1, 0))
    assert permuted == tree  # the swap of two equal labels is a pure phase here
    assert coeff == expected


def test_genuine_multi_term_expansion():
    tree = fusion_trees(SU2, (HALF, HALF, HALF), HALF)[0]
    terms = SU2.permute_tree(tree, (0, 2, 1))
    assert len(terms) > 1
    inners = {t.inner for t, _ in terms}
    assert len(inners) == len(terms)  # distinct inner lines, not repeated trees
    assert all(abs(c) > 1e-9 for _, c in terms)


def test_r_symbol_is_its_own_inverse():
    """Invariant 12 by demonstration: SU(2) is symmetric, so ``perm`` fixes the braid."""
    for a, b in itertools.product(range(5), repeat=2):
        for c in range(abs(a - b), a + b + 1, 2):
            assert SU2.r_symbol(SU2Sector(a), SU2Sector(b), SU2Sector(c)) ** 2 == 1


# --- tree-level algebra --------------------------------------------------------


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_tree_level_unitarity(rank):
    for u in uncoupled_sets(rank):
        for coupled in coupled_sectors(SU2, u):
            src = fusion_trees(SU2, u, coupled)
            for perm in itertools.permutations(range(rank)):
                dst = fusion_trees(SU2, tuple(u[i] for i in perm), coupled)
                assert len(src) == len(dst)
                index = {t: j for j, t in enumerate(dst)}
                m = np.zeros((len(src), len(dst)), dtype=complex)
                for i, tree in enumerate(src):
                    for t2, coeff in SU2.permute_tree(tree, perm):
                        m[i, index[t2]] += coeff
                np.testing.assert_allclose(m @ m.conj().T, np.eye(len(src)), atol=1e-12)


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_adjacent_swap_is_an_involution(rank):
    for u in uncoupled_sets(rank):
        for coupled in coupled_sectors(SU2, u):
            for tree in fusion_trees(SU2, u, coupled):
                for i in range(rank - 1):
                    swap = tuple(range(i)) + (i + 1, i) + tuple(range(i + 2, rank))
                    back: dict = {}
                    for t1, c1 in SU2.permute_tree(tree, swap):
                        for t2, c2 in SU2.permute_tree(t1, swap):
                            back[t2] = back.get(t2, 0) + c1 * c2
                    assert abs(back.pop(tree) - 1.0) < 1e-13
                    assert all(abs(c) < 1e-13 for c in back.values())


def test_composition_matches_the_composite_permutation():
    checked = 0
    for u in uncoupled_sets(4):
        for coupled in coupled_sectors(SU2, u):
            for tree in fusion_trees(SU2, u, coupled):
                for p, q in itertools.product(itertools.permutations(range(4)), repeat=2):
                    composed = tuple(p[i] for i in q)
                    left: dict = {}
                    for t1, c1 in SU2.permute_tree(tree, p):
                        for t2, c2 in SU2.permute_tree(t1, q):
                            left[t2] = left.get(t2, 0) + c1 * c2
                    right = dict(SU2.permute_tree(tree, composed))
                    for t2 in left.keys() | right.keys():
                        assert abs(left.get(t2, 0) - right.get(t2, 0)) < 1e-12
                    checked += 1
    assert checked > 0


def test_identity_permutation_is_exact_and_arithmetic_free():
    for u in uncoupled_sets(4):
        for coupled in coupled_sectors(SU2, u):
            for tree in fusion_trees(SU2, u, coupled):
                terms = SU2.permute_tree(tree, (0, 1, 2, 3))
                assert terms == ((tree, 1.0),)
                assert terms[0][1] == 1.0 and type(terms[0][1]) is float


def test_permute_braided_tree_is_cached():
    tree = fusion_trees(SU2, (HALF, HALF, HALF), HALF)[0]
    first = permute_braided_tree(SU2, tree, (2, 0, 1))
    assert permute_braided_tree(SU2, tree, (2, 0, 1)) is first
    assert SU2.permute_tree(tree, (2, 0, 1)) is first


# --- the dense oracle ----------------------------------------------------------


@pytest.mark.parametrize("p", ALL_PERMS)
def test_dense_oracle_rank_four_all_24_permutations(p):
    t = su2()
    np.testing.assert_allclose(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p), atol=1e-12)


@pytest.mark.parametrize("p", PERMS5)
def test_dense_oracle_rank_five(p):
    t = SymmetricTensor.random(LEGS5, seed=17)
    np.testing.assert_allclose(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p), atol=1e-11)


@pytest.mark.parametrize("p", ALL_PERMS)
def test_round_trip(p):
    t = su2()
    assert tenet.allclose(t.transpose(p).transpose(inverse(p)), t, atol=1e-12)


def test_norm_preserved_and_conj_commutes():
    t = su2() + su2() * 1j
    for p in ALL_PERMS:
        assert abs(tenet.norm(t.transpose(p)) - tenet.norm(t)) < 1e-12
        assert tenet.allclose(t.transpose(p).conj(), t.conj().transpose(p))


def test_plan_fills_every_target_block():
    """The ``plan fills N of M`` ValueError never fires over either sweep."""
    for t, perms in ((su2(), ALL_PERMS), (SymmetricTensor.random(LEGS5, seed=17), PERMS5)):
        for p in perms:
            plan = permutation_plan(t.structure, p)
            n = plan.new_structure.num_blocks
            assert {dst for _, dst, c in plan.terms if c != 0} == set(range(n))
            t.transpose(p)  # would raise if the plan under-filled


# --- case A is untouched -------------------------------------------------------


def test_case_a_still_free():
    t = su2()
    for p in CASE_A:
        plan = permutation_plan(t.structure, p)
        assert plan.terms == tuple((i, i, 1.0) for i in range(t.structure.num_blocks))
        assert all(coeff == 1.0 for _, _, coeff in plan.terms)
        r = t.transpose(p)
        assert r.structure.block_order == t.structure.block_order
        for old, new in zip(t.blocks, r.blocks, strict=True):
            np.testing.assert_array_equal(new, np.transpose(old, p))


def test_plan_stays_frozen_hashable_and_array_free():
    s = su2().structure
    plan = permutation_plan(s, (2, 1, 0, 3))
    assert permutation_plan(s, (2, 1, 0, 3)) is plan
    assert hash(plan) == hash(permutation_plan(s, (2, 1, 0, 3)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.axes = (0, 1, 2, 3)
    for field in dataclasses.fields(PermutationPlan):
        assert not hasattr(getattr(plan, field.name), "shape"), field.name
    assert all(isinstance(c, (int, float, complex)) for _, _, c in plan.terms)


# --- module hygiene ------------------------------------------------------------


def test_no_provider_branch_in_permutation_module():
    src = pathlib.Path(tenet.ops.permutation.__file__).read_text()
    assert "SU2" not in src.split('"""', 2)[2]  # only the module docstring may name it
    assert "if provider ==" not in src
    assert "isinstance(provider" not in src
    assert "Milestone 4" not in src


def test_braid_helpers_touch_no_arrays():
    src = pathlib.Path(tenet.symmetry.base.__file__).read_text()
    braid = src[src.index("def permute_braided_tree") : src.index("def bend_unique")]
    assert "np." not in braid
    assert "to_dense" not in braid
    assert "SU2" not in braid


# --- backends ------------------------------------------------------------------


@pytest.mark.parametrize("p", ALL_PERMS)
def test_jax_dense_oracle_all_24_permutations(p):
    use_jax()
    t = su2()
    r = t.to_backend("jax").transpose(p)
    assert r.backend == "jax"
    np.testing.assert_allclose(
        r.to_backend("numpy").to_dense(), np.transpose(t.to_dense(), p), atol=1e-12
    )
