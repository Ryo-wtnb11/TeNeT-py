"""Tests for tenet.structure — one test per acceptance criterion of issue #8."""

import dataclasses
import enum
import os
import subprocess
import sys
from itertools import product
from math import prod

import numpy as np
import pytest

from tenet import (
    IN,
    OUT,
    FusionBlockKey,
    FusionTree,
    GradedSpace,
    Leg,
    TensorStructure,
    coupled_sectors,
    fusion_trees,
)
from tenet.symmetry import (
    SU2,
    U1,
    SU2Sector,
    SUNProvider,
    SUNSector,
    Trivial,
    TrivialSector,
    U1Sector,
)

HALF = SU2Sector(1)
ONE = SU2Sector(2)
SINGLET = SU2Sector(0)


def su2_space(degeneracies):
    """``su2_space({1: 2})`` → space with ``m_{1/2} = 2``; keys are ``two_j``."""
    return GradedSpace.new(SU2, {SU2Sector(j): m for j, m in degeneracies.items()})


def u1_space(degeneracies):
    return GradedSpace.new(U1, {U1Sector(q): m for q, m in degeneracies.items()})


def su2_half_structure(out_deg=2, in_deg=3):
    """Three spin-1/2 OUT legs, one spin-1/2 IN leg — the multi-block workhorse."""
    return TensorStructure(
        (
            *(Leg(su2_space({1: out_deg}), OUT) for _ in range(3)),
            Leg(su2_space({1: in_deg}), IN),
        )
    )


# --- frozen / hashable ----------------------------------------------------------


def test_key_and_structure_are_frozen_hashable_dict_keys():
    s = su2_half_structure()
    k = s.block_order[0]
    assert not hasattr(k, "__dict__")  # slots
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.output_tree = k.input_tree
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.legs = ()
    assert {k: 1, s: 2}[s] == 2
    assert hash(s) == hash(TensorStructure(s.legs))
    assert hash(k) == hash(FusionBlockKey(k.output_tree, k.input_tree))


def test_memoized_hash_equals_the_field_tuple_hash():
    """M9 (#59): the cached ``_hash`` must stay the dataclass hash, field for field.

    Adding a field without extending ``__post_init__`` would silently make two
    distinct values collide; this is the check that fails when that happens.
    """
    s = su2_half_structure()
    k = s.block_order[0]
    for obj in (s.legs[0].space, s.legs[0], k.output_tree, k, s):
        fields = tuple(getattr(obj, f.name) for f in dataclasses.fields(obj))
        assert "_hash" not in {f.name for f in dataclasses.fields(obj)}
        assert hash(obj) == hash(fields)


def test_key_field_order_is_the_documented_sort_order():
    assert [f.name for f in dataclasses.fields(FusionBlockKey)] == ["output_tree", "input_tree"]


def test_keys_are_totally_ordered():
    keys = su2_half_structure().block_order
    assert sorted(keys) == list(keys)
    assert keys[0] < keys[1]


def test_coupled_is_the_shared_sector():
    for k in su2_half_structure().block_order:
        assert k.coupled == k.output_tree.coupled == k.input_tree.coupled


def test_legs_are_coerced_to_a_tuple_and_emptiness_rejected():
    s = TensorStructure([Leg(su2_space({1: 2}), OUT)])
    assert isinstance(s.legs, tuple)
    assert hash(s)
    with pytest.raises(ValueError, match="at least one leg"):
        TensorStructure(())


# --- array-free (invariant 8) ---------------------------------------------------


def _walk(obj, seen=None):
    """Yield every object reachable from ``obj`` through dataclass fields / tuples."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        children = [getattr(obj, f.name) for f in dataclasses.fields(obj)]
    elif isinstance(obj, tuple):
        children = list(obj)
    else:
        children = []
    for child in children:
        yield from _walk(child, seen)


def test_structure_graph_is_array_free_and_immutable():
    s = su2_half_structure()
    roots = (s, *s.block_order, *(s.axis_sectors(k) for k in s.block_order))
    reachable = [o for root in roots for o in _walk(root)]
    assert len(reachable) > 20  # the walk actually walked
    for o in reachable:
        assert not isinstance(o, np.ndarray), f"array reachable: {o!r}"
        assert not isinstance(o, list | dict | set | bytearray), f"mutable container: {o!r}"
        assert isinstance(o, tuple | int | str | bool | float | enum.Enum | type(None)) or (
            dataclasses.is_dataclass(o) and o.__dataclass_params__.frozen
        ), f"neither frozen nor a scalar: {o!r}"


# --- determinism ----------------------------------------------------------------


def test_block_order_is_identical_across_python_hash_seeds():
    script = (
        "from tenet import IN, OUT, GradedSpace, Leg, TensorStructure\n"
        "from tenet.symmetry import SU2, SU2Sector\n"
        "sp = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 2, SU2Sector(2): 1})\n"
        "s = TensorStructure((Leg(sp, OUT), Leg(sp, OUT), Leg(sp, IN, dual=True)))\n"
        "print(s.block_order)\n"
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


def _block_order_by_joint_enumeration(s):
    """``block_order``'s definition, spelled out: walk every leg assignment at once.

    The reference for ``_block_cross``, which gets the same tuple out of the two sides
    enumerated *separately*. Quadratically slower and that is the point — the definition
    is what the fast rule has to reproduce, key for key and in the same order.
    """
    keys = set()
    for assignment in product(*(leg.sectors for leg in s.legs)):
        uncoupled = tuple(leg.fused_sector(a) for leg, a in zip(s.legs, assignment, strict=True))
        out_u = tuple(uncoupled[i] for i in s.out_axes)
        in_u = tuple(uncoupled[i] for i in s.in_axes)
        for c in coupled_sectors(s.provider, out_u):
            keys.update(
                FusionBlockKey(ot, it)
                for ot in fusion_trees(s.provider, out_u, c)
                for it in fusion_trees(s.provider, in_u, c)
            )
    return tuple(sorted(keys))


@pytest.mark.parametrize(
    "space",
    [
        su2_space({0: 2, 1: 1, 2: 3}),
        su2_space({0: 1, 1: 1, 2: 1, 3: 1}),
        u1_space({-1: 2, 0: 1, 2: 3}),
        GradedSpace.new(SUNProvider(3), {SUNSector((0, 0)): 2, SUNSector((1, 0)): 1}),
    ],
    ids=["su2", "su2-wide", "u1", "su3"],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_block_order_matches_the_joint_enumeration_of_every_leg(space, ndim):
    for n_out in range(ndim + 1):
        for duals in product((False, True), repeat=ndim):
            s = TensorStructure(
                tuple(Leg(space, OUT if i < n_out else IN, dual=d) for i, d in enumerate(duals))
            )
            assert s.block_order == _block_order_by_joint_enumeration(s)
            assert s.num_blocks == len(s.block_order)


def test_block_order_is_independent_of_input_ordering():
    pairs = [(SU2Sector(0), 1), (SU2Sector(1), 2), (SU2Sector(2), 3)]
    forward = GradedSpace.new(SU2, pairs)
    backward = GradedSpace.new(SU2, reversed(pairs))
    assert forward == backward
    a = TensorStructure((Leg(forward, OUT), Leg(forward, IN)))
    b = TensorStructure((Leg(backward, OUT), Leg(backward, IN)))
    assert a.block_order == b.block_order


# --- index_of -------------------------------------------------------------------


def test_index_of_inverts_block_order_and_rejects_foreign_keys():
    s = su2_half_structure()
    assert s.num_blocks == len(s.block_order) > 0
    for i, k in enumerate(s.block_order):
        assert s.index_of(k) == i
    foreign = FusionBlockKey(
        FusionTree((HALF, HALF, HALF), (ONE,), (0, 0), SU2Sector(3)),
        FusionTree((ONE,), (), (), ONE),
    )
    with pytest.raises(KeyError):
        s.index_of(foreign)


# --- validate -------------------------------------------------------------------


def test_validate_accepts_a_real_structure():
    s = su2_half_structure()
    s.validate()
    for k in s.block_order:
        s.validate(k)


def test_validate_rejects_mixed_providers():
    s = TensorStructure((Leg(su2_space({1: 2}), OUT), Leg(u1_space({0: 2}), IN)))
    with pytest.raises(ValueError, match="provider"):
        s.validate()


def test_validate_rejects_an_output_tree_of_the_wrong_rank():
    bad = FusionBlockKey(
        FusionTree((HALF, HALF), (), (0,), SINGLET),
        FusionTree((SINGLET,), (), (), SINGLET),
    )
    with pytest.raises(ValueError, match="output_tree has rank 2.*3 output legs"):
        su2_half_structure().validate(bad)


def test_validate_rejects_an_input_tree_of_the_wrong_rank():
    bad = FusionBlockKey(
        FusionTree((HALF, HALF, HALF), (SINGLET,), (0, 0), HALF),
        FusionTree((HALF, HALF), (), (0,), HALF),
    )
    with pytest.raises(ValueError, match="input_tree has rank 2.*1 input legs"):
        su2_half_structure().validate(bad)


def test_validate_rejects_disagreeing_coupled_sectors():
    bad = FusionBlockKey(
        FusionTree((HALF, HALF, HALF), (SINGLET,), (0, 0), HALF),
        FusionTree((ONE,), (), (), ONE),
    )
    with pytest.raises(ValueError, match="coupled sectors disagree"):
        su2_half_structure().validate(bad)


def test_validate_rejects_an_uncoupled_label_absent_from_the_leg_space():
    bad = FusionBlockKey(
        FusionTree((HALF, HALF, ONE), (SINGLET,), (0, 0), ONE),
        FusionTree((ONE,), (), (), ONE),
    )
    with pytest.raises(ValueError, match=r"axis 2: SU2Sector\(two_j=2\) is not a sector"):
        su2_half_structure().validate(bad)


def test_validate_rejects_an_invalid_tree():
    bad = FusionBlockKey(
        FusionTree((HALF, HALF, HALF), (SU2Sector(3),), (0, 0), HALF),
        FusionTree((HALF,), (), (), HALF),
    )
    with pytest.raises(ValueError, match="vertex 0"):
        su2_half_structure().validate(bad)


# --- invariant 7: public axis order ---------------------------------------------


def test_block_shape_is_in_public_axis_order_not_regrouped():
    degs = (2, 3, 5, 7)
    sides = (OUT, IN, OUT, IN)
    s = TensorStructure(
        tuple(Leg(u1_space({0: m}), side) for m, side in zip(degs, sides, strict=True))
    )
    (key,) = s.block_order
    assert s.out_axes == (0, 2)
    assert s.in_axes == (1, 3)
    assert s.block_shape(key) == degs
    regrouped = tuple(degs[i] for i in s.out_axes + s.in_axes)
    assert regrouped == (2, 5, 3, 7)
    assert s.block_shape(key) != regrouped  # not vacuous


# --- SU(2): a charge tuple cannot serve as a block key --------------------------


def test_su2_has_more_keys_than_external_sector_assignments():
    s = su2_half_structure()
    # Each leg's space holds only spin-1/2, so there is exactly one sector assignment.
    assignments = prod(len(leg.space) for leg in s.legs)
    assert assignments == 1
    # (1/2,1/2,1/2) -> 1/2 has 2 trees and -> 3/2 has 1; the IN side reaches only 1/2.
    assert s.num_blocks == 2
    assert s.num_blocks > assignments
    assert len({s.axis_sectors(k) for k in s.block_order}) == 1
    assert [k.output_tree.inner for k in s.block_order] == [(SINGLET,), (ONE,)]


def test_at_least_two_su2_keys_share_axis_sectors():
    s = su2_half_structure()
    shared = [k for k in s.block_order if s.axis_sectors(k) == (HALF, HALF, HALF, HALF)]
    assert len(shared) >= 2
    assert len(set(shared)) == len(shared)


# --- U(1) contrast: abelian degeneration ----------------------------------------


def test_u1_axis_sectors_is_injective_over_block_order():
    sp = u1_space({-1: 2, 0: 3, 1: 4})
    s = TensorStructure((Leg(sp, OUT), Leg(sp, OUT), Leg(sp, IN)))
    assert len({s.axis_sectors(k) for k in s.block_order}) == s.num_blocks
    # one block per charge assignment with q0 + q1 == q2
    assert s.num_blocks == sum(a.charge + b.charge == c.charge for a in sp for b in sp for c in sp)


def test_trivial_provider_has_exactly_one_block():
    sp = GradedSpace.new(Trivial, {TrivialSector(): 4})
    s = TensorStructure((Leg(sp, OUT), Leg(sp, IN)))
    assert s.num_blocks == 1
    assert s.block_shape(s.block_order[0]) == (4, 4)


# --- duality round-trip ---------------------------------------------------------


def test_u1_dual_legs_round_trip_and_conserve_charge():
    sp = u1_space({-1: 2, 0: 3, 1: 4})
    legs = (Leg(sp, OUT), Leg(sp, OUT, dual=True), Leg(sp, IN, dual=True))
    s = TensorStructure(legs)
    s.validate()
    assert s.num_blocks > 1
    for k in s.block_order:
        sectors = s.axis_sectors(k)
        assert all(a in leg.space for leg, a in zip(legs, sectors, strict=True))
        assert all(m > 0 for m in s.block_shape(k))
        signed = [(-1 if leg.dual else 1) * a.charge for leg, a in zip(legs, sectors, strict=True)]
        assert signed[0] + signed[1] == k.coupled.charge == signed[2]


# --- rank-0 side ----------------------------------------------------------------


def test_no_in_legs_gives_rank_0_input_trees_coupling_to_the_unit():
    sp = su2_space({0: 1, 1: 2})
    s = TensorStructure((Leg(sp, OUT), Leg(sp, OUT)))
    assert s.in_axes == ()
    assert s.num_blocks > 0
    for k in s.block_order:
        assert k.input_tree.rank == 0
        assert k.coupled == SU2.unit
        assert len(s.block_shape(k)) == s.ndim == 2
    # only (0,0) and (1/2,1/2) fuse to the singlet
    assert {s.axis_sectors(k) for k in s.block_order} == {(SINGLET, SINGLET), (HALF, HALF)}


def test_no_out_legs_is_symmetric():
    sp = su2_space({1: 2})
    s = TensorStructure((Leg(sp, IN), Leg(sp, IN)))
    assert s.out_axes == ()
    assert [k.output_tree.rank for k in s.block_order] == [0]
    assert s.block_order[0].coupled == SU2.unit


# --- block sparsity -------------------------------------------------------------


def test_su2_structure_is_block_sparse():
    s = su2_half_structure(out_deg=2, in_deg=3)
    stored = sum(prod(s.block_shape(k)) for k in s.block_order)
    assert stored == 2 * (2 * 2 * 2 * 3) == 48  # 2 keys, each of shape (2, 2, 2, 3)
    dense = prod(leg.space.dim for leg in s.legs)
    assert dense == (2 * 2) ** 3 * (3 * 2) == 384
    assert stored < dense / 4


# --- the whole-table shape accessor ---------------------------------------------


def test_block_shapes_is_block_shape_over_block_order():
    """The two spellings are the same information; only their cost differs (#307).

    Asserted on a structure whose legs carry *different* degeneracies, because the
    table is the one structure-keyed table that does not delegate to the
    degeneracy-free pattern -- a shape depends on degeneracies, an index does not --
    so a bug that read the pattern's table instead would be invisible on a
    degeneracy-1 structure.
    """
    s = su2_half_structure(out_deg=2, in_deg=3)
    assert s.block_shapes == tuple(s.block_shape(k) for k in s.block_order)
    assert len(s.block_shapes) == s.num_blocks
    assert len({sum(shape) for shape in s.block_shapes}) >= 1


def test_block_shapes_distinguishes_structures_that_share_a_pattern():
    """Two structures with one sector pattern and different degeneracies.

    ``block_order`` and ``index_of`` are shared across a degeneracy pattern; shapes are
    not. If ``block_shapes`` ever started delegating the way its neighbours do, every
    tensor built through it would get its sibling's block shapes -- silently, since the
    key set is identical.
    """
    a = su2_half_structure(out_deg=2, in_deg=3)
    b = su2_half_structure(out_deg=5, in_deg=1)
    assert a.block_order == b.block_order
    assert a.block_shapes != b.block_shapes
