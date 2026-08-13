"""Acceptance tests for :mod:`tenet.leg` — one test per issue-6 criterion."""

import dataclasses

import pytest

from tenet import IN, OUT, GradedSpace, Leg
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

V_SU2 = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 3})
V_U1 = GradedSpace.new(U1, {U1Sector(-3): 1, U1Sector(0): 2, U1Sector(3): 4})


def test_frozen_hashable_dict_key() -> None:
    leg = Leg(V_SU2, OUT)
    assert leg == Leg(V_SU2, OUT)
    assert {leg: "a"}[Leg(V_SU2, OUT)] == "a"
    with pytest.raises(dataclasses.FrozenInstanceError):
        leg.dual = True  # type: ignore[misc]


def test_four_side_dual_combinations_distinct() -> None:
    """Invariant-2 regression test: side and dual never collapse into one flag."""
    legs = [
        Leg(V_SU2, OUT, dual=False),
        Leg(V_SU2, OUT, dual=True),
        Leg(V_SU2, IN, dual=False),
        Leg(V_SU2, IN, dual=True),
    ]
    assert len(set(legs)) == 4
    assert len({hash(leg) for leg in legs}) == 4


def test_no_way_to_change_side() -> None:
    assert not hasattr(Leg, "with_side")
    leg = Leg(V_SU2, OUT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        leg.side = IN  # type: ignore[misc]


def test_dualized_is_an_involution_touching_only_dual() -> None:
    leg = Leg(V_U1, IN, dual=True, name="k")
    once = leg.dualized()
    assert once.dualized() == leg
    assert (once.space, once.side, once.name) == (leg.space, leg.side, leg.name)
    assert once.dual is not leg.dual


def test_name_participates_in_equality_and_renamed() -> None:
    leg = Leg(V_U1, OUT)
    assert leg.renamed("p") != leg
    assert leg.renamed("p").renamed(None) == leg


def test_su2_self_dual_label_does_not_collapse_the_flag() -> None:
    dual_leg = Leg(V_SU2, OUT, dual=True)
    assert dual_leg.fused_sector(SU2Sector(1)) == SU2Sector(1)
    assert dual_leg != Leg(V_SU2, OUT, dual=False)


def test_u1_fused_sector_negates_and_round_trips() -> None:
    assert Leg(V_U1, IN, dual=True).fused_sector(U1Sector(3)) == U1Sector(-3)
    for dual in (False, True):
        leg = Leg(V_U1, IN, dual=dual)
        for a in leg.sectors:
            assert leg.space_sector(leg.fused_sector(a)) == a


def test_provider_and_sectors_come_from_the_space() -> None:
    leg = Leg(V_SU2, IN, dual=True)
    assert leg.provider is SU2
    assert leg.sectors == tuple(V_SU2) == (SU2Sector(0), SU2Sector(1))


def test_degeneracy_takes_space_labels_even_when_dual() -> None:
    leg = Leg(V_U1, OUT, dual=True)
    for a in leg.sectors:
        assert leg.degeneracy(a) == V_U1.degeneracy(a)
    # A fused label is the wrong input: for U(1) it reads the mirrored sector.
    assert leg.degeneracy(leg.fused_sector(U1Sector(3))) == 1
    assert leg.degeneracy(U1Sector(3)) == 4
