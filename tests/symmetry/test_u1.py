"""Tests for tenet.symmetry.u1 — one test per acceptance criterion of issue #3."""

import random
from dataclasses import fields

import numpy as np
import pytest

from tenet.symmetry import (
    U1,
    ClebschGordanData,
    FusionRules,
    QuantumDimensionData,
    U1Provider,
    U1Sector,
    requires,
)

# static conformance check: fails type checking if U1Provider drifts from the protocols
_fusion: FusionRules = U1
_qdim: QuantumDimensionData = U1
_cgc: ClebschGordanData = U1

CHARGES = [-3, -2, -1, 0, 1, 2, 5]


# --- criterion 1: U1 satisfies FusionRules, QuantumDimensionData, ClebschGordanData ---


def test_u1_satisfies_capabilities():
    assert isinstance(U1, QuantumDimensionData)
    assert isinstance(U1, ClebschGordanData)
    assert requires(U1, QuantumDimensionData) is None
    assert requires(U1, ClebschGordanData) is None
    assert U1.name == "U1"
    assert hash(U1) == hash(U1Provider())
    for f in fields(U1):
        assert not isinstance(getattr(U1, f.name), np.ndarray)


# --- criterion 2: dual is an involution ---


def test_dual_is_an_involution():
    for n in CHARGES:
        a = U1Sector(n)
        assert U1.dual(a) == U1Sector(-n)
        assert U1.dual(U1.dual(a)) == a


# --- criterion 3: unit is a two-sided fusion identity ---


def test_unit_is_two_sided_identity():
    u = U1.unit
    assert u == U1Sector(0)
    for n in CHARGES:
        a = U1Sector(n)
        assert U1.fusion(u, a) == (a,)
        assert U1.fusion(a, u) == (a,)


# --- criterion 4: fusion-set associativity ---


def test_fusion_associativity():
    rng = random.Random(0)
    for _ in range(50):
        a, b, c = (U1Sector(rng.randint(-9, 9)) for _ in range(3))
        left = sorted(e for d in U1.fusion(a, b) for e in U1.fusion(d, c))
        right = sorted(e for d in U1.fusion(b, c) for e in U1.fusion(a, d))
        assert left == right


# --- criterion 5: n_symbol is the membership predicate for fusion ---


def test_n_symbol_matches_fusion_membership():
    for na in CHARGES:
        for nb in CHARGES:
            a, b = U1Sector(na), U1Sector(nb)
            for nc in range(-12, 13):
                c = U1Sector(nc)
                assert U1.n_symbol(a, b, c) == int(c in U1.fusion(a, b))


# --- criterion 6: multiplicity-free ---


def test_fusion_is_multiplicity_free():
    for na in CHARGES:
        for nb in CHARGES:
            a, b = U1Sector(na), U1Sector(nb)
            assert sum(U1.n_symbol(a, b, c) for c in U1.fusion(a, b)) == 1


# --- criterion 7: sectors hashable, ordered, readable repr ---


def test_sectors_are_hashable_ordered_and_repr_round_trips():
    assert len({U1Sector(1), U1Sector(1), U1Sector(2)}) == 2
    assert {U1Sector(3): "x"}[U1Sector(3)] == "x"
    assert sorted(U1Sector(n) for n in [2, -1, 0]) == [
        U1Sector(-1),
        U1Sector(0),
        U1Sector(2),
    ]
    assert repr(U1Sector(-2)) == "U1Sector(charge=-2)"
    assert eval(repr(U1Sector(-2))) == U1Sector(-2)  # noqa: S307


# --- dimensions and Clebsch-Gordan ---


def test_dimensions_are_trivial():
    for n in CHARGES:
        assert U1.qdim(U1Sector(n)) == 1.0
        assert U1.irrep_dim(U1Sector(n)) == 1


def test_cgc_shape_value_and_read_only():
    a, b = U1Sector(2), U1Sector(-3)
    (c,) = U1.fusion(a, b)
    cgc = U1.cgc(a, b, c)
    assert cgc.shape == (1, 1, 1, U1.n_symbol(a, b, c))
    assert cgc[0, 0, 0, 0] == 1.0
    assert not cgc.flags.writeable


def test_cgc_raises_on_forbidden_triple():
    with pytest.raises(ValueError):
        U1.cgc(U1Sector(1), U1Sector(1), U1Sector(0))
