"""Tests for tenet.space — one test per acceptance criterion of issue #5."""

from dataclasses import FrozenInstanceError, dataclass, fields

import numpy as np
import pytest

from tenet import GradedSpace
from tenet.symmetry import (
    SU2,
    CapabilityError,
    Sector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
)

HALF = SU2Sector(1)
ONE = SU2Sector(2)


@dataclass(frozen=True, slots=True)
class _NoCGCProvider:
    """Abelian-ish provider without ``irrep_dim`` — reduced_dim must still work."""

    name: str = "NoCGC"

    @property
    def unit(self) -> U1Sector:
        return U1Sector(0)

    def dual(self, a: U1Sector) -> U1Sector:
        return U1Sector(-a.charge)

    def fusion(self, a: U1Sector, b: U1Sector) -> tuple[Sector, ...]:
        return (U1Sector(a.charge + b.charge),)

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1


def test_new_normalizes_insertion_order():
    v = GradedSpace.new(SU2, {HALF: 4, ONE: 3})
    w = GradedSpace.new(SU2, {ONE: 3, HALF: 4})
    assert v == w
    assert hash(v) == hash(w)
    assert v.sectors == ((HALF, 4), (ONE, 3))


def test_new_accepts_iterable_of_pairs():
    assert GradedSpace.new(SU2, [(ONE, 3), (HALF, 4)]) == GradedSpace.new(SU2, {HALF: 4, ONE: 3})


def test_rejects_duplicate_sectors():
    with pytest.raises(ValueError, match="duplicate"):
        GradedSpace.new(SU2, [(HALF, 4), (HALF, 2)])


@pytest.mark.parametrize("m", [0, -1])
def test_rejects_non_positive_degeneracy(m):
    with pytest.raises(ValueError, match="positive"):
        GradedSpace.new(SU2, {HALF: m})


def test_rejects_foreign_sector_type():
    with pytest.raises(TypeError):
        GradedSpace.new(SU2, {U1Sector(1): 2})
    with pytest.raises(TypeError):
        GradedSpace.new(SU2, {HALF: 4, TrivialSector(): 1})


def test_frozen_and_hashable_as_dict_key():
    v = GradedSpace.new(SU2, {HALF: 4})
    with pytest.raises(FrozenInstanceError):
        v.sectors = ()
    assert {v: "ok"}[GradedSpace.new(SU2, {HALF: 4})] == "ok"


def test_reduced_dim_vs_dim():
    v = GradedSpace.new(SU2, {HALF: 4, ONE: 3})
    assert v.reduced_dim == 7
    assert v.dim == 4 * 2 + 3 * 3 == 17


def test_sector_offset_consistent():
    v = GradedSpace.new(SU2, {HALF: 4, ONE: 3, SU2Sector(0): 2})
    offsets = [v.sector_offset(a) for a in v]
    assert offsets == sorted(offsets)
    for (a, m), nxt in zip(v.sectors, list(v)[1:], strict=False):
        assert v.sector_offset(a) + m * SU2.irrep_dim(a) == v.sector_offset(nxt)
    last, m_last = v.sectors[-1]
    assert v.sector_offset(last) + m_last * SU2.irrep_dim(last) == v.dim


def test_sector_offset_absent_raises():
    with pytest.raises(KeyError):
        GradedSpace.new(SU2, {HALF: 4}).sector_offset(ONE)


def test_absent_sector():
    v = GradedSpace.new(SU2, {HALF: 4})
    assert v.degeneracy(ONE) == 0
    assert ONE not in v
    assert HALF in v
    assert len(v) == 1


def test_trivial_provider_dim_equals_reduced_dim():
    v = GradedSpace.new(Trivial, {TrivialSector(): 5})
    assert v.dim == v.reduced_dim == 5
    assert v.sector_offset(TrivialSector()) == 0


def test_reduced_dim_without_clebsch_gordan():
    v = GradedSpace.new(_NoCGCProvider(), {U1Sector(1): 2, U1Sector(-1): 3})
    assert v.reduced_dim == 5
    with pytest.raises(CapabilityError):
        _ = v.dim
    with pytest.raises(CapabilityError):
        v.sector_offset(U1Sector(1))


def test_no_ndarray_reachable():
    v = GradedSpace.new(SU2, {HALF: 4, ONE: 3})
    for f in fields(v):
        value = getattr(v, f.name)
        assert not isinstance(value, np.ndarray)
        for item in value if isinstance(value, tuple) else ():
            assert not isinstance(item, np.ndarray)
            for sub in item if isinstance(item, tuple) else ():
                assert not isinstance(sub, np.ndarray)


# --- direct_sum (#142) ----------------------------------------------------------


def test_direct_sum_disjoint_sectors():
    from tenet.symmetry import U1

    v = GradedSpace.new(U1, {U1Sector(0): 2})
    w = GradedSpace.new(U1, {U1Sector(1): 3})
    assert v.direct_sum(w) == GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})


def test_direct_sum_overlapping_sectors():
    from tenet.symmetry import U1

    v = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    w = GradedSpace.new(U1, {U1Sector(1): 3, U1Sector(-1): 4})
    expected = GradedSpace.new(U1, {U1Sector(-1): 4, U1Sector(0): 2, U1Sector(1): 4})
    assert v.direct_sum(w) == expected
    assert w.direct_sum(v) == expected  # commutative at the label level


def test_direct_sum_subspace_operand():
    v = GradedSpace.new(SU2, {HALF: 4, ONE: 3})
    w = GradedSpace.new(SU2, {HALF: 1})
    assert v.direct_sum(w) == GradedSpace.new(SU2, {HALF: 5, ONE: 3})


def test_direct_sum_with_itself_doubles_degeneracies():
    v = GradedSpace.new(SU2, {HALF: 4, ONE: 3})
    assert v.direct_sum(v) == GradedSpace.new(SU2, {HALF: 8, ONE: 6})


def test_direct_sum_refuses_a_mismatched_provider():
    from tenet.symmetry import U1

    v = GradedSpace.new(U1, {U1Sector(0): 2})
    w = GradedSpace.new(Trivial, {TrivialSector(): 2})
    with pytest.raises(TypeError, match="never casts between symmetries"):
        v.direct_sum(w)
