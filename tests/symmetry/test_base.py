"""Tests for tenet.symmetry.base — one test per acceptance criterion of issue #2."""

from dataclasses import FrozenInstanceError, dataclass, fields

import numpy as np
import pytest

from tenet.symmetry import (
    CapabilityError,
    ClebschGordanData,
    FusionRules,
    QuantumDimensionData,
    Sector,
    Trivial,
    TrivialProvider,
    TrivialSector,
    requires,
)

# static conformance check: fails type checking if TrivialProvider drifts from the protocols
_fusion: FusionRules = Trivial
_qdim: QuantumDimensionData = Trivial
_cgc: ClebschGordanData = Trivial


@dataclass(frozen=True, slots=True, order=True)
class _IntSector(Sector):
    """A second sector type, to check ordering is per-type."""

    charge: int


@dataclass(frozen=True, slots=True)
class _NoCGCProvider:
    """Provider with a quantum dimension but no Clebsch-Gordan tensors."""

    name: str = "NoCGC"

    @property
    def unit(self) -> TrivialSector:
        return TrivialSector()

    def dual(self, a: Sector) -> Sector:
        return a

    def fusion(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        return (TrivialSector(),)

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1.0


# --- criterion 1: sectors are frozen, hashable, usable as keys, sort deterministically ---


def test_sector_is_frozen():
    with pytest.raises(FrozenInstanceError):
        _IntSector(1).charge = 2


def test_sector_hash_is_value_based():
    assert hash(TrivialSector()) == hash(TrivialSector())
    assert TrivialSector() == TrivialSector()


def test_sector_usable_as_dict_key_and_set_member():
    d = {TrivialSector(): "u", _IntSector(1): "one"}
    assert d[TrivialSector()] == "u"
    assert d[_IntSector(1)] == "one"
    assert len({TrivialSector(), TrivialSector(), _IntSector(1)}) == 2


def test_sectors_sort_deterministically():
    unsorted = [_IntSector(2), _IntSector(-1), _IntSector(0)]
    assert sorted(unsorted) == [_IntSector(-1), _IntSector(0), _IntSector(2)]
    assert sorted(reversed(unsorted)) == sorted(unsorted)


def test_comparing_different_sector_types_raises():
    with pytest.raises(TypeError):
        _ = TrivialSector() < _IntSector(0)


# --- criterion 2: protocols, and Trivial satisfies all three ---


def test_capabilities_are_runtime_checkable_and_trivial_satisfies_them():
    assert isinstance(Trivial, QuantumDimensionData)
    assert isinstance(Trivial, ClebschGordanData)


def test_fusion_rules_is_a_protocol():
    assert FusionRules._is_protocol
    assert QuantumDimensionData._is_protocol
    assert ClebschGordanData._is_protocol
    assert set(FusionRules.__protocol_attrs__) == {
        "name",
        "unit",
        "fusion",
        "n_symbol",
    }


def test_provider_lacking_a_capability_is_not_an_instance():
    assert isinstance(_NoCGCProvider(), QuantumDimensionData)
    assert not isinstance(_NoCGCProvider(), ClebschGordanData)


# --- criterion 3: Trivial values ---


def test_trivial_provider_values():
    u = Trivial.unit
    assert u == TrivialSector()
    assert Trivial.dual(u) == u
    assert Trivial.fusion(u, u) == (u,)
    assert Trivial.n_symbol(u, u, u) == 1
    assert Trivial.qdim(u) == 1.0
    assert Trivial.irrep_dim(u) == 1
    cgc = Trivial.cgc(u, u, u)
    assert cgc.shape == (1, 1, 1, 1)
    assert cgc[0, 0, 0, 0] == 1.0


# --- criterion 4: providers hashable and array-free ---


def test_provider_is_hashable_and_array_free():
    assert hash(Trivial) == hash(TrivialProvider())
    for f in fields(Trivial):
        assert not isinstance(getattr(Trivial, f.name), np.ndarray)


# --- criterion 5: requires() ---


def test_requires_passes_for_supported_capability():
    assert requires(Trivial, ClebschGordanData) is None
    assert requires(Trivial, QuantumDimensionData) is None


def test_requires_raises_naming_provider_and_capability():
    with pytest.raises(CapabilityError) as excinfo:
        requires(_NoCGCProvider(), ClebschGordanData)
    message = str(excinfo.value)
    assert "_NoCGCProvider" in message
    assert "ClebschGordanData" in message


# --- criterion 6: no provider-identity branching in base.py ---


def test_base_module_has_no_provider_branching():
    from pathlib import Path

    import tenet.symmetry.base as base

    source = Path(base.__file__).read_text()
    assert "if provider ==" not in source
    assert "isinstance(provider, TrivialProvider)" not in source
