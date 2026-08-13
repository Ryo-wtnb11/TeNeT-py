"""Tests for tenet.space — one test per acceptance criterion of issue #5.

``tenet.symmetry.u1`` / ``tenet.symmetry.su2`` are being implemented in parallel
(issues #3/#4), so the SU(2)/U(1) criteria use the local stand-in providers below.
Swap them for the real providers in a follow-up.
"""

from dataclasses import FrozenInstanceError, dataclass, fields

import numpy as np
import pytest

from tenet import GradedSpace
from tenet.symmetry import CapabilityError, Sector, Trivial, TrivialSector


@dataclass(frozen=True, slots=True, order=True)
class _SU2LikeSector(Sector):
    """Stand-in for SU2Sector: ``two_j`` is twice the spin."""

    two_j: int


@dataclass(frozen=True, slots=True)
class _SU2Like:
    """Stand-in for the SU(2) provider: only what GradedSpace needs."""

    name: str = "SU2-like"

    @property
    def unit(self) -> _SU2LikeSector:
        return _SU2LikeSector(0)

    def dual(self, a: Sector) -> Sector:
        return a

    def fusion(self, a: _SU2LikeSector, b: _SU2LikeSector) -> tuple[Sector, ...]:
        return tuple(
            _SU2LikeSector(t) for t in range(abs(a.two_j - b.two_j), a.two_j + b.two_j + 1, 2)
        )

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1 if c in self.fusion(a, b) else 0

    def irrep_dim(self, a: _SU2LikeSector) -> int:
        return a.two_j + 1

    def cgc(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True, slots=True, order=True)
class _U1LikeSector(Sector):
    """Stand-in for U1Sector."""

    charge: int


SU2 = _SU2Like()
HALF = _SU2LikeSector(1)
ONE = _SU2LikeSector(2)


@dataclass(frozen=True, slots=True)
class _NoCGCProvider:
    """Abelian-ish provider without ``irrep_dim`` — reduced_dim must still work."""

    name: str = "NoCGC"

    @property
    def unit(self) -> _U1LikeSector:
        return _U1LikeSector(0)

    def dual(self, a: _U1LikeSector) -> _U1LikeSector:
        return _U1LikeSector(-a.charge)

    def fusion(self, a: _U1LikeSector, b: _U1LikeSector) -> tuple[Sector, ...]:
        return (_U1LikeSector(a.charge + b.charge),)

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
        GradedSpace.new(SU2, {_U1LikeSector(1): 2})
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
    v = GradedSpace.new(SU2, {HALF: 4, ONE: 3, _SU2LikeSector(0): 2})
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
    v = GradedSpace.new(_NoCGCProvider(), {_U1LikeSector(1): 2, _U1LikeSector(-1): 3})
    assert v.reduced_dim == 5
    with pytest.raises(CapabilityError):
        _ = v.dim
    with pytest.raises(CapabilityError):
        v.sector_offset(_U1LikeSector(1))


def test_no_ndarray_reachable():
    v = GradedSpace.new(SU2, {HALF: 4, ONE: 3})
    for f in fields(v):
        value = getattr(v, f.name)
        assert not isinstance(value, np.ndarray)
        for item in value if isinstance(value, tuple) else ():
            assert not isinstance(item, np.ndarray)
            for sub in item if isinstance(item, tuple) else ():
                assert not isinstance(sub, np.ndarray)
