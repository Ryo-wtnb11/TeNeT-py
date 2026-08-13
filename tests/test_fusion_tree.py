"""Tests for tenet.fusion_tree — one test per acceptance criterion of issue #7."""

import os
import random
import subprocess
import sys
from collections import Counter
from dataclasses import FrozenInstanceError, fields
from math import prod

import pytest

from tenet import FusionTree, coupled_sectors, fusion_trees
from tenet.symmetry import SU2, U1, SU2Sector, Trivial, TrivialSector, U1Sector

HALF = SU2Sector(1)
ONE = SU2Sector(2)
SINGLET = SU2Sector(0)


def su2_trees(uncoupled, coupled):
    return fusion_trees(SU2, uncoupled, coupled)


# --- dataclass shape ------------------------------------------------------------


def test_field_order_is_the_documented_sort_order():
    assert [f.name for f in fields(FusionTree)] == [
        "uncoupled",
        "inner",
        "multiplicities",
        "coupled",
    ]


def test_tree_is_frozen_slotted_and_hashable():
    t = FusionTree((HALF, HALF), (), (0,), SINGLET)
    assert not hasattr(t, "__dict__")  # slots
    assert hash(t) == hash(FusionTree((HALF, HALF), (), (0,), SINGLET))
    with pytest.raises(FrozenInstanceError):
        t.coupled = ONE


def test_trees_are_totally_ordered():
    trees = su2_trees((HALF, HALF, HALF), HALF)
    assert sorted(trees) == list(trees)
    assert trees[0] < trees[1]


def test_sort_order_is_stable_across_processes():
    """Same enumeration, different PYTHONHASHSEED, byte-identical output."""
    script = (
        "from tenet import fusion_trees, coupled_sectors\n"
        "from tenet.symmetry import SU2, SU2Sector\n"
        "u = (SU2Sector(1),) * 4\n"
        "print(fusion_trees(SU2, u, SU2Sector(0)))\n"
        "print([fusion_trees(SU2, u, c) for c in coupled_sectors(SU2, u)])\n"
    )
    outs = set()
    for seed in ("0", "1", "12345", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(
            subprocess.run(
                [sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True
            ).stdout
        )
    assert len(outs) == 1


# --- length invariants ----------------------------------------------------------


@pytest.mark.parametrize(
    ("uncoupled", "inner", "mults"),
    [
        ((HALF, HALF, HALF), (), (0, 0)),  # inner too short
        ((HALF, HALF, HALF), (SINGLET, ONE), (0, 0)),  # inner too long
        ((HALF, HALF, HALF), (SINGLET,), (0,)),  # multiplicities too short
        ((HALF, HALF), (), (0, 0)),  # multiplicities too long
        ((), (SINGLET,), ()),  # rank 0 must be empty
        ((HALF,), (), (0,)),  # rank 1 must be empty
    ],
)
def test_length_invariants_enforced_at_construction(uncoupled, inner, mults):
    with pytest.raises(ValueError):
        FusionTree(uncoupled, inner, mults, SINGLET)


def test_valid_lengths_accepted():
    assert FusionTree((HALF, HALF, HALF), (SINGLET,), (0, 0), HALF).rank == 3


# --- lines / vertices -----------------------------------------------------------


def test_lines_and_vertices_follow_left_association():
    t = FusionTree((HALF, HALF, HALF), (ONE,), (0, 0), HALF)
    assert t.lines() == (HALF, ONE, HALF)
    assert t.vertices() == ((HALF, HALF, ONE, 0), (ONE, HALF, HALF, 0))


def test_degenerate_lines_and_vertices():
    assert FusionTree((), (), (), SINGLET).lines() == ()
    assert FusionTree((), (), (), SINGLET).vertices() == ()
    assert FusionTree((HALF,), (), (), HALF).lines() == (HALF,)
    assert FusionTree((HALF,), (), (), HALF).vertices() == ()


# --- validate -------------------------------------------------------------------


def test_validate_accepts_a_good_tree():
    FusionTree((HALF, HALF, HALF), (ONE,), (0, 0), HALF).validate(SU2)


def test_validate_rejects_a_fusion_rule_violation_naming_the_vertex():
    bad = FusionTree((HALF, HALF, HALF), (SU2Sector(3),), (0, 0), HALF)
    with pytest.raises(ValueError, match="vertex 0"):
        bad.validate(SU2)


def test_validate_names_the_second_offending_vertex():
    bad = FusionTree((HALF, HALF, HALF), (SINGLET,), (0, 0), SU2Sector(4))
    with pytest.raises(ValueError, match="vertex 1"):
        bad.validate(SU2)


def test_validate_rejects_multiplicity_out_of_range():
    bad = FusionTree((HALF, HALF), (), (1,), SINGLET)
    with pytest.raises(ValueError, match=r"vertex 0.*range\(1\)"):
        bad.validate(SU2)


def test_validate_rejects_negative_multiplicity():
    with pytest.raises(ValueError, match="vertex 0"):
        FusionTree((HALF, HALF), (), (-1,), SINGLET).validate(SU2)


def test_validate_rejects_bad_degenerate_couplings():
    with pytest.raises(ValueError, match="rank-0"):
        FusionTree((), (), (), ONE).validate(SU2)
    with pytest.raises(ValueError, match="rank-1"):
        FusionTree((HALF,), (), (), ONE).validate(SU2)


# --- degenerate ranks -----------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "a", "b"),
    [(Trivial, TrivialSector(), None), (U1, U1Sector(3), U1Sector(4)), (SU2, HALF, ONE)],
)
def test_degenerate_ranks(provider, a, b):
    empty = fusion_trees(provider, (), provider.unit)
    assert len(empty) == 1
    assert empty[0] == FusionTree((), (), (), provider.unit)
    assert len(fusion_trees(provider, (a,), a)) == 1
    assert fusion_trees(provider, (a,), a)[0].coupled == a
    if b is not None:
        assert fusion_trees(provider, (a,), b) == ()


# --- U(1): abelian means external sectors determine the basis -------------------


def test_u1_has_exactly_one_tree_for_the_charge_sum():
    qs = (U1Sector(2), U1Sector(-5), U1Sector(1))
    total = U1Sector(sum(q.charge for q in qs))
    assert len(fusion_trees(U1, qs, total)) == 1
    for other in (U1Sector(-3), U1Sector(0), U1Sector(7)):
        if other != total:
            assert fusion_trees(U1, qs, other) == ()
    assert coupled_sectors(U1, qs) == (total,)


# --- SU(2): the headline test ---------------------------------------------------


def test_su2_three_halves_to_half_gives_two_trees_differing_only_by_inner_line():
    trees = su2_trees((HALF, HALF, HALF), HALF)
    assert len(trees) == 2
    assert [t.inner for t in trees] == [(SINGLET,), (ONE,)]
    # external sectors != complete fusion basis
    assert {(t.uncoupled, t.coupled) for t in trees} == {((HALF, HALF, HALF), HALF)}
    assert trees[0] != trees[1]


def test_su2_four_halves_multiplicity_counting():
    u = (HALF,) * 4
    counts = {c: len(su2_trees(u, c)) for c in coupled_sectors(SU2, u)}
    assert counts == {SINGLET: 2, ONE: 3, SU2Sector(4): 1}
    all_trees = [t for c in counts for t in su2_trees(u, c)]
    assert sum(SU2.irrep_dim(t.coupled) for t in all_trees) == 2**4


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 5])
def test_su2_dimension_identity_on_random_lists(rank):
    rng = random.Random(rank)
    for _ in range(10):
        u = tuple(SU2Sector(rng.randrange(5)) for _ in range(rank))
        trees = [t for c in coupled_sectors(SU2, u) for t in fusion_trees(SU2, u, c)]
        assert sum(SU2.irrep_dim(t.coupled) for t in trees) == prod(SU2.irrep_dim(x) for x in u)


# --- coupled_sectors ------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "uncoupled"),
    [
        (Trivial, (TrivialSector(),) * 3),
        (U1, (U1Sector(1), U1Sector(-2), U1Sector(4))),
        (SU2, (HALF, ONE, HALF, SU2Sector(3))),
        (SU2, ()),
        (SU2, (ONE,)),
    ],
)
def test_coupled_sectors_agrees_with_enumeration_and_is_sorted(provider, uncoupled):
    cs = coupled_sectors(provider, uncoupled)
    assert list(cs) == sorted(set(cs))
    union = {t.coupled for c in cs for t in fusion_trees(provider, uncoupled, c)}
    assert union == set(cs)
    assert all(fusion_trees(provider, uncoupled, c) for c in cs)


# --- enumeration hygiene --------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "uncoupled"),
    [
        (Trivial, (TrivialSector(),) * 4),
        (U1, (U1Sector(1), U1Sector(1), U1Sector(-3))),
        (SU2, (HALF, HALF, ONE, HALF)),
        (SU2, ()),
    ],
)
def test_every_tree_validates_is_unique_and_sorted(provider, uncoupled):
    for c in coupled_sectors(provider, uncoupled):
        trees = fusion_trees(provider, uncoupled, c)
        for t in trees:
            t.validate(provider)
            assert t.uncoupled == tuple(uncoupled)
            assert t.coupled == c
        assert len(set(trees)) == len(trees)
        assert list(trees) == sorted(trees)


def test_enumeration_accepts_any_sequence_and_is_cached_consistently():
    u = [HALF, HALF, HALF]
    assert fusion_trees(SU2, u, HALF) is fusion_trees(SU2, tuple(u), HALF)


def test_multiplicities_present_even_when_multiplicity_free():
    trees = su2_trees((HALF, HALF, HALF), HALF)
    assert all(t.multiplicities == (0, 0) for t in trees)
    assert Counter(len(t.multiplicities) for t in trees) == {2: 2}
