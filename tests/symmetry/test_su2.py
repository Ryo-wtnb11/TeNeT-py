"""Tests for tenet.symmetry.su2 — one test per acceptance criterion of issue #4."""

import random
from collections import Counter
from dataclasses import fields
from fractions import Fraction
from math import sqrt

import numpy as np
import pytest

from tenet.symmetry import SU2, ClebschGordanData, FusionRules, QuantumDimensionData, SU2Sector
from tenet.symmetry._su2_coeff import value, w6j
from tenet.symmetry.su2 import _SU2_GAUGE, SU2Provider

# static conformance: fails type checking if SU2Provider drifts from the protocols
_fusion: FusionRules = SU2
_qdim: QuantumDimensionData = SU2
_cgc: ClebschGordanData = SU2

SECTORS = [SU2Sector(two_j) for two_j in range(7)]


def spin_matrices(two_j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(Jz, Jp, Jm)`` in the descending-m basis: index i carries m = j - i."""
    d = two_j + 1
    j = two_j / 2
    ms = np.array([j - i for i in range(d)])
    jz = np.diag(ms)
    jp = np.zeros((d, d))
    for i in range(1, d):  # J+ raises m: column i (m) -> row i-1 (m+1)
        m = ms[i]
        jp[i - 1, i] = sqrt(j * (j + 1) - m * (m + 1))
    return jz, jp, jp.T


# --- sector label ---------------------------------------------------------------


def test_sector_rejects_negative_two_j():
    with pytest.raises(ValueError):
        SU2Sector(-1)


def test_sector_rejects_non_int_two_j():
    with pytest.raises(TypeError):
        SU2Sector(0.5)


def test_sector_is_hashable_and_ordered():
    assert sorted({SU2Sector(2), SU2Sector(0), SU2Sector(2)}) == [SU2Sector(0), SU2Sector(2)]


# --- fusion ---------------------------------------------------------------------


def test_half_times_half_is_singlet_plus_triplet():
    assert SU2.fusion(SU2Sector(1), SU2Sector(1)) == (SU2Sector(0), SU2Sector(2))


def test_triangle_rule_grid():
    for a in SECTORS:
        for b in SECTORS:
            for c in SECTORS:
                expected = int(
                    abs(a.two_j - b.two_j) <= c.two_j <= a.two_j + b.two_j
                    and (a.two_j + b.two_j + c.two_j) % 2 == 0
                )
                assert SU2.n_symbol(a, b, c) == expected
                assert (c in SU2.fusion(a, b)) == bool(expected)


def test_dimension_counting():
    for a in SECTORS:
        for b in SECTORS:
            assert SU2.irrep_dim(a) * SU2.irrep_dim(b) == sum(
                SU2.irrep_dim(c) for c in SU2.fusion(a, b)
            )


def test_qdim_equals_irrep_dim():
    for a in SECTORS:
        assert SU2.qdim(a) == float(SU2.irrep_dim(a))


def test_self_dual_and_unit_is_two_sided_identity():
    for a in SECTORS:
        assert SU2.dual(a) == a
        assert SU2.fusion(a, SU2.unit) == (a,)
        assert SU2.fusion(SU2.unit, a) == (a,)


def test_fusion_associativity_random_triples():
    rng = random.Random(0)
    for _ in range(50):
        a, b, c = (SU2Sector(rng.randrange(7)) for _ in range(3))
        left = Counter(e for d in SU2.fusion(a, b) for e in SU2.fusion(d, c))
        right = Counter(e for d in SU2.fusion(b, c) for e in SU2.fusion(a, d))
        assert left == right


# --- Clebsch-Gordan -------------------------------------------------------------


def test_cgc_shape_and_read_only():
    a, b, c = SU2Sector(2), SU2Sector(1), SU2Sector(3)
    arr = SU2.cgc(a, b, c)
    assert arr.shape == (SU2.irrep_dim(a), SU2.irrep_dim(b), SU2.irrep_dim(c), 1)
    assert not arr.flags.writeable
    with pytest.raises(ValueError):
        arr[0, 0, 0, 0] = 1.0


def test_cgc_invalid_triple_raises():
    with pytest.raises(ValueError):
        SU2.cgc(SU2Sector(1), SU2Sector(1), SU2Sector(1))


def test_cg_orthonormality():
    for a in SECTORS:
        for b in SECTORS:
            da, db = SU2.irrep_dim(a), SU2.irrep_dim(b)
            m = np.concatenate(
                [SU2.cgc(a, b, c)[..., 0].reshape(da * db, -1) for c in SU2.fusion(a, b)],
                axis=1,
            )
            eye = np.eye(da * db)
            assert np.allclose(m.T @ m, eye, atol=1e-12)
            assert np.allclose(m @ m.T, eye, atol=1e-12)


def test_cg_intertwiner_against_explicit_spin_matrices():
    for a in SECTORS:
        for b in SECTORS:
            for c in SU2.fusion(a, b):
                cgc = SU2.cgc(a, b, c)[..., 0]
                for xa, xb, xc in zip(
                    spin_matrices(a.two_j),
                    spin_matrices(b.two_j),
                    spin_matrices(c.two_j),
                    strict=True,
                ):
                    lhs = np.einsum("iI,Ijk->ijk", xa, cgc) + np.einsum("jJ,iJk->ijk", xb, cgc)
                    rhs = np.einsum("ijK,Kk->ijk", cgc, xc)
                    assert np.allclose(lhs, rhs, atol=1e-12)


def test_condon_shortley_singlet_value():
    # <1/2,1/2; 1/2,-1/2 | 0,0> = +1/sqrt(2); the m1=-1/2 partner carries the minus sign.
    cgc = SU2.cgc(SU2Sector(1), SU2Sector(1), SU2Sector(0))[..., 0]
    assert cgc[0, 1, 0] == pytest.approx(1 / sqrt(2), abs=1e-12)
    assert cgc[1, 0, 0] == pytest.approx(-1 / sqrt(2), abs=1e-12)


def test_condon_shortley_full_one_times_half_table():
    a, b = SU2Sector(2), SU2Sector(1)  # j1 = 1, j2 = 1/2
    r13, r23 = sqrt(1 / 3), sqrt(2 / 3)
    # descending m: (m1, m2) = (1 - i1, 1/2 - i2)
    three_half = SU2.cgc(a, b, SU2Sector(3))[..., 0]
    half = SU2.cgc(a, b, SU2Sector(1))[..., 0]
    expected_32 = {  # (i1, i2, i3) -> value, i3 indexes m3 = 3/2, 1/2, -1/2, -3/2
        (0, 0, 0): 1.0,
        (0, 1, 1): r13,
        (1, 0, 1): r23,
        (1, 1, 2): r23,
        (2, 0, 2): r13,
        (2, 1, 3): 1.0,
    }
    expected_12 = {  # i3 indexes m3 = 1/2, -1/2
        (0, 1, 0): r23,
        (1, 0, 0): -r13,
        (1, 1, 1): r13,
        (2, 0, 1): -r23,
    }
    for idx, val in expected_32.items():
        assert three_half[idx] == pytest.approx(val, abs=1e-12)
    for idx, val in expected_12.items():
        assert half[idx] == pytest.approx(val, abs=1e-12)
    assert np.count_nonzero(three_half) == len(expected_32)
    assert np.count_nonzero(half) == len(expected_12)


def test_w6j_known_value_and_zero():
    # {1 1 1; 1 1 1} = 1/6; a triangle-violating argument list gives exactly zero.
    assert value(w6j(2, 2, 2, 2, 2, 2)) == pytest.approx(1 / 6, abs=1e-12)
    assert w6j(2, 2, 2, 2, 2, 1) == (0, Fraction(0))


# --- provider hygiene (invariant 8) ---------------------------------------------


def test_provider_is_hashable_and_array_free():
    assert hash(SU2) == hash(SU2Provider())
    for f in fields(SU2):
        assert not isinstance(getattr(SU2, f.name), np.ndarray)


def test_gauge_fingerprint_is_pinned():
    assert _SU2_GAUGE == (
        "3j=condon-shortley;cg=condon-shortley;f=tks-su2irrep;r=tks-su2irrep;fs=tks-su2irrep"
    )
