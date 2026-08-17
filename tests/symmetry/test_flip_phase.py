"""The ``FlipPhase`` capability across providers, and SU(3)'s flip contract — #142.

``flip_phase(a) = chi_a * theta_a``. Every provider the suite builds implements
it: 1 for the bosonic Abelian providers, ``(-1)^parity`` for fZ2 (chi is +1
there, so the whole factor is the twist), the Frobenius-Schur phase for
SU(2)/SU(N) (symmetric braiding, twist 1), and the product over components for
a Deligne product. SU(3) exercises the space relabel with genuinely non-self-dual
sectors, which no other provider in the suite reaches.
"""

import numpy as np
import pytest
from _su3_fixture import SU3, SU3Sector

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, flip, tensordot
from tenet.symmetry import (
    SU2,
    U1,
    Z2,
    FlipPhase,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    U1Sector,
    fZ2,
)
from tenet.symmetry.sun import SUNProvider


@pytest.mark.parametrize(
    "provider",
    [Trivial, U1, Z2, fZ2, SU2, SUNProvider(3), ProductProvider((U1, SU2)), SU3],
    ids=lambda p: p.name,
)
def test_every_provider_implements_flip_phase(provider):
    assert isinstance(provider, FlipPhase)


def test_flip_phase_values():
    assert U1.flip_phase(U1Sector(3)) == 1.0
    assert fZ2.flip_phase(FZ2Sector(0)) == 1.0
    assert fZ2.flip_phase(FZ2Sector(1)) == -1.0
    assert SU2.flip_phase(SU2Sector(1)) == -1  # half-integer: chi = -1, theta = 1
    assert SU2.flip_phase(SU2Sector(2)) == 1


def test_product_flip_phase_is_the_product_over_components():
    product = ProductProvider((U1, SU2))
    for q in (0, 2):
        for two_j in (0, 1, 2):
            a = ProductSector((U1Sector(q), SU2Sector(two_j)))
            assert product.flip_phase(a) == U1.flip_phase(a.components[0]) * SU2.flip_phase(
                a.components[1]
            )


def test_su3_contract_after_flipping_both_legs():
    """Contract (a) on a provider whose sectors are not self-dual: the flip
    relabels ``3`` to ``3bar`` and back, and the contraction result is unchanged."""
    space = GradedSpace.new(SU3, {SU3Sector((1, 0)): 1, SU3Sector((0, 1)): 1})
    a = SymmetricTensor.random((Leg(space, OUT), Leg(space, IN, name="x")), seed=13)
    b = SymmetricTensor.random((Leg(space, OUT, name="y"), Leg(space, IN)), seed=14)
    ref = np.asarray(tensordot(a, b, ((1,), (0,))).to_dense())
    got = np.asarray(tensordot(flip(a, 1), flip(b, 0), ((1,), (0,))).to_dense())
    assert np.max(np.abs(got - ref)) <= 1e-12


def test_su3_flip_relabels_three_to_threebar():
    three = GradedSpace.new(SU3, {SU3Sector((1, 0)): 2})
    t = SymmetricTensor.random((Leg(three, OUT), Leg(three, IN)), seed=15)
    flipped = flip(t, 0)
    assert tuple(flipped.legs[0].space) == (SU3Sector((0, 1)),)
    assert flipped.legs[0].dual is True
    assert flipped.structure.block_order == t.structure.block_order
