"""The M24 capability lattice (#158): per-provider isinstance answers and the split flip.

The deprecated composition aliases left in stage (b); what stays pinned here is
the honest per-capability profile of every provider, ``supports`` as
``requires``' non-raising sibling, and the numeric contract of the split flip
scalar: ``frobenius_schur`` and ``twist`` value-pinned per provider, so their
product is exactly the pre-split flip phase.
"""

import pytest

from tenet.symmetry import (
    SU2,
    U1,
    Z2,
    AssociatorData,
    BMatrixData,
    BraidingData,
    CapabilityError,
    ClebschGordanData,
    DaggerData,
    DualityData,
    FMatrixData,
    FSIndicatorData,
    FZ2Sector,
    PivotalData,
    ProductProvider,
    ProductSector,
    QuantumDimensionData,
    RMatrixData,
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


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p.name)
def test_the_small_protocols(provider):
    assert isinstance(provider, FSIndicatorData)
    assert isinstance(provider, TwistData)
    assert isinstance(provider, QuantumDimensionData)
    assert isinstance(provider, ClebschGordanData)
    # the markers are satisfied by everything, deliberately (stage a)
    assert isinstance(provider, PivotalData)
    assert isinstance(provider, DaggerData)
    # scalar F/R/B belong to SU(2) alone among these six
    for capability in (AssociatorData, BraidingData, DualityData):
        assert isinstance(provider, capability) == (provider is SU2)
    # none of the six is multiplicity-bearing
    for capability in (FMatrixData, RMatrixData, BMatrixData):
        assert not isinstance(provider, capability)


def test_supports_is_requires_non_raising_sibling():
    assert supports(SU2, BraidingData)
    assert not supports(U1, BraidingData)
    assert requires(SU2, BraidingData) is None
    with pytest.raises(CapabilityError, match="BraidingData"):
        requires(U1, BraidingData)


# chi and theta value-pinned per provider and sector: the flip scalar (their
# product) is the numeric contract the deleted bundled method used to carry.
CHI_THETA = {
    Trivial: {TrivialSector(): (1.0, 1.0)},
    U1: {U1Sector(-2): (1.0, 1.0), U1Sector(0): (1.0, 1.0), U1Sector(3): (1.0, 1.0)},
    Z2: {Z2Sector(0): (1.0, 1.0), Z2Sector(1): (1.0, 1.0)},
    fZ2: {FZ2Sector(0): (1.0, 1.0), FZ2Sector(1): (1.0, -1.0)},
    SU2: {
        SU2Sector(0): (1, 1),
        SU2Sector(1): (-1, 1),
        SU2Sector(2): (1, 1),
        SU2Sector(3): (-1, 1),
    },
    PRODUCT: {ProductSector((U1Sector(1), SU2Sector(1))): (-1, 1)},
}


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p.name)
def test_chi_and_theta_values_are_pinned(provider):
    """The #142 split, value-pinned: chi and theta per provider and sector, so
    the flip scalar ``chi * theta`` stays exactly the pre-split value."""
    assert set(CHI_THETA[provider]) == set(SAMPLES[provider])
    for a, (chi, theta) in CHI_THETA[provider].items():
        assert provider.frobenius_schur(a) == chi
        assert provider.twist(a) == theta


def test_sun_isinstance_identity():
    racah = pytest.importorskip("racah")  # noqa: F841  # the optional SU(N) wheel
    from tenet.symmetry.sun import SUNProvider, SUNSector

    su3 = SUNProvider(3)
    for capability in (AssociatorData, BraidingData, DualityData, FSIndicatorData):
        assert isinstance(su3, capability)
    for capability in (FMatrixData, RMatrixData, BMatrixData):
        assert isinstance(su3, capability)
    assert isinstance(su3, TwistData)
    eight = SUNSector((1, 1))  # the adjoint: real, so chi = +1; symmetric, so theta = 1
    assert su3.frobenius_schur(eight) == 1
    assert su3.twist(eight) == 1
