"""The M24a capability lattice (#158): aliases, isinstance identity, the split flip.

Stage (a) is additive with zero behavior change, and the three load-bearing
halves of that claim are asserted here: the old protocol names still resolve
and give the same ``isinstance`` answers they always did; ``supports`` is
``requires``' non-raising sibling; and ``frobenius_schur * twist`` is
bit-identical to the bundled ``flip_phase`` on every provider.
"""

import pytest

from tenet.symmetry import (
    SU2,
    U1,
    Z2,
    AssociatorData,
    BraidingData,
    CapabilityError,
    ClebschGordan,
    ClebschGordanData,
    DaggerData,
    DualityData,
    FlipPhase,
    FSIndicatorData,
    FZ2Sector,
    MultiplicityRecoupling,
    PivotalData,
    ProductProvider,
    ProductSector,
    QuantumDimension,
    QuantumDimensionData,
    RecouplingData,
    SU2Sector,
    Trivial,
    TrivialSector,
    TwistData,
    U1Sector,
    Z2Sector,
    fZ2,
    requires,
    supports,
)

PRODUCT = ProductProvider((U1, SU2))
PROVIDERS = [Trivial, U1, Z2, fZ2, SU2, PRODUCT]
SAMPLES = {
    Trivial: [TrivialSector()],
    U1: [U1Sector(-2), U1Sector(0), U1Sector(3)],
    Z2: [Z2Sector(0), Z2Sector(1)],
    fZ2: [FZ2Sector(0), FZ2Sector(1)],
    SU2: [SU2Sector(0), SU2Sector(1), SU2Sector(2), SU2Sector(3)],
    PRODUCT: [ProductSector((U1Sector(1), SU2Sector(1)))],
}


def test_renamed_protocols_are_the_same_objects():
    """Pure renames: the old name *is* the new class, so isinstance answers are
    identical by construction."""
    assert QuantumDimension is QuantumDimensionData
    assert ClebschGordan is ClebschGordanData


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p.name)
def test_composition_aliases_answer_as_before(provider):
    """The bundles decompose, but the deprecated compositions keep the historical
    isinstance answers: only SU(2) (and SU(N)/the SU(3) fixture) carried the
    full recoupling bundle."""
    assert isinstance(provider, RecouplingData) == (provider is SU2)
    assert not isinstance(provider, MultiplicityRecoupling)
    assert isinstance(provider, FlipPhase)
    assert isinstance(provider, QuantumDimensionData)
    assert isinstance(provider, ClebschGordanData)


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p.name)
def test_the_new_small_protocols(provider):
    assert isinstance(provider, FSIndicatorData)
    assert isinstance(provider, TwistData)
    # the markers are satisfied by everything, deliberately (stage a)
    assert isinstance(provider, PivotalData)
    assert isinstance(provider, DaggerData)
    # F/R/B belong to SU(2) alone among these six
    for capability in (AssociatorData, BraidingData, DualityData):
        assert isinstance(provider, capability) == (provider is SU2)


def test_supports_is_requires_non_raising_sibling():
    assert supports(SU2, BraidingData)
    assert not supports(U1, BraidingData)
    assert requires(SU2, BraidingData) is None
    with pytest.raises(CapabilityError, match="BraidingData"):
        requires(U1, BraidingData)


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p.name)
def test_flip_phase_is_bit_identical_to_chi_times_theta(provider):
    """The #142 split criterion, honored bit for bit: the product of the two new
    methods equals the bundled ``flip_phase`` exactly on every provider."""
    for a in SAMPLES[provider]:
        assert provider.frobenius_schur(a) * provider.twist(a) == provider.flip_phase(a)


def test_sun_isinstance_identity():
    racah = pytest.importorskip("racah")  # noqa: F841  # the optional SU(N) wheel
    from tenet.symmetry.sun import SUNProvider, SUNSector

    su3 = SUNProvider(3)
    assert isinstance(su3, RecouplingData)
    assert isinstance(su3, MultiplicityRecoupling)
    assert isinstance(su3, FSIndicatorData) and isinstance(su3, TwistData)
    eight = SUNSector((1, 1))
    assert su3.frobenius_schur(eight) * su3.twist(eight) == su3.flip_phase(eight)
