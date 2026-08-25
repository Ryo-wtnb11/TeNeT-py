"""Contractions and factorizations where an operand carries no block at all.

A tensor whose legs cannot couple to any total charge is legal and has zero blocks:
``SymmetricTensor.random`` on such legs returns one. Everything here pins what the
operations do with it, and the oracle needs no sign bookkeeping of its own -- the dense
expansion of a block-less tensor is all zeros, so the dense contraction is all zeros
whatever the graded transposition would have done to it.
"""

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops import linalg
from tenet.symmetry import (
    SU2,
    U1,
    FZ2Sector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

# Per provider: (a space, a space that cannot couple with it to anything). The pair
# is what makes a block-less tensor; Trivial has no such pair -- every Trivial
# structure couples -- so its entry is a zero-dimensional space instead.
CASES = {
    "trivial": (
        GradedSpace.new(Trivial, {TrivialSector(): 2}),
        GradedSpace.new(Trivial, {}),
    ),
    "u1": (
        GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1}),
        GradedSpace.new(U1, {U1Sector(3): 2}),
    ),
    "fz2": (
        GradedSpace.new(fZ2, {FZ2Sector(0): 2}),
        GradedSpace.new(fZ2, {FZ2Sector(1): 1}),
    ),
    "su2": (
        GradedSpace.new(SU2, {SU2Sector(0): 2}),
        GradedSpace.new(SU2, {SU2Sector(1): 1}),
    ),
}


@pytest.fixture(params=sorted(CASES))
def spaces(request):
    return CASES[request.param]


def _empty(v, w):
    """A block-less tensor on ``(v OUT, w OUT, v IN)`` -- the codomain cannot reach 0."""
    t = SymmetricTensor.random((Leg(v, OUT), Leg(w, OUT), Leg(v, IN)), seed=0)
    assert t.blocks == ()
    return t


def test_the_fixtures_really_are_block_less(spaces):
    v, w = spaces
    e = _empty(v, w)
    assert e.structure.num_blocks == 0
    assert not np.any(e.to_dense())


def test_tensordot_against_the_dense_zeros(spaces):
    v, w = spaces
    e = _empty(v, w)
    b = SymmetricTensor.random((Leg(v, OUT), Leg(v, IN)), seed=1)
    for got, want in (
        (tenet.tensordot(e, b, axes=((2,), (0,))), np.tensordot(e.to_dense(), b.to_dense(), 1)),
        (tenet.tensordot(b, e, axes=((1,), (0,))), np.tensordot(b.to_dense(), e.to_dense(), 1)),
    ):
        assert got.shape == want.shape
        assert np.allclose(got.to_dense(), want)


def test_both_operands_block_less_with_an_output_that_couples(spaces):
    """Two block-less operands whose imbalances cancel: the output *does* admit blocks.

    They are then zeros, and NumPy ``float64`` — neither operand has a backend or a
    dtype to hand over. (Trivial is the degenerate column: its second space is empty,
    so the output has no sector either and there is no block to type.)
    """
    v, w = spaces
    a = SymmetricTensor.random((Leg(w, OUT), Leg(v, IN)), seed=0)
    b = SymmetricTensor.random((Leg(v, OUT), Leg(w, IN)), seed=1)
    assert (a.blocks, b.blocks) == ((), ())
    c = tenet.tensordot(a, b, axes=((1,), (0,)))
    assert c.legs == (a.legs[0], b.legs[1])
    assert np.allclose(c.to_dense(), np.tensordot(a.to_dense(), b.to_dense(), 1))
    assert not np.any(c.to_dense())
    assert bool(c.blocks) == bool(w.sectors)
    for block in c.blocks:
        assert isinstance(block, np.ndarray)
        assert block.dtype == np.float64


def test_the_zeros_take_the_other_operand_s_dtype():
    """A couplable output with one block-carrying operand -- an SU(2) arrangement.

    Over an abelian provider it cannot happen: the output legs are exactly the operands'
    free legs, and a couplable ``b`` contributes zero net charge, so the output inherits
    the block-less operand's imbalance and is block-less too. Fusion multiplicity is what
    breaks that, and it is why this test is not parametrized over the providers.
    """
    zero = GradedSpace.new(SU2, {SU2Sector(0): 1})
    one = GradedSpace.new(SU2, {SU2Sector(2): 1})
    a = SymmetricTensor.random((Leg(zero, OUT), Leg(one, IN)), seed=0)
    b = SymmetricTensor.random(
        (Leg(one, OUT), Leg(one, OUT), Leg(one, IN)), seed=1, dtype=np.complex128
    )
    assert a.blocks == () and b.blocks != ()
    c = tenet.tensordot(a, b, axes=((1,), (0,)))
    assert c.blocks != ()
    assert all(block.dtype == np.complex128 for block in c.blocks)
    assert np.allclose(c.to_dense(), np.tensordot(a.to_dense(), b.to_dense(), 1))
    assert not np.any(c.to_dense())


def test_einsum_and_einsum_chain_agree_with_tensordot(spaces):
    v, w = spaces
    e = _empty(v, w)
    b = SymmetricTensor.random((Leg(v, OUT), Leg(v, IN)), seed=1)
    want = tenet.tensordot(e, b, axes=((2,), (0,)))
    assert tenet.allclose(tenet.einsum("abc,cd->abd", e, b), want)
    chained = tenet.einsum_chain([("abc,cd->abd", e, b, "")])
    assert tenet.allclose(chained, want)
    # and a second step, so the block-less operand travels as an unapplied plan
    b2 = SymmetricTensor.random((Leg(v, OUT), Leg(v, IN)), seed=2)
    assert tenet.allclose(
        tenet.einsum_chain([("abc,cd->abd", e, b, ""), ("abc,cd->abd", None, b2, "")]),
        tenet.einsum("abc,cd->abd", want, b2),
    )


def test_compose_with_a_block_less_operand(spaces):
    v, w = spaces
    a = SymmetricTensor.random((Leg(v, OUT), Leg(w, OUT), Leg(v, IN)), seed=0)
    b = SymmetricTensor.random((Leg(v, OUT), Leg(v, IN)), seed=1)
    assert a.blocks == ()
    c = a @ b
    assert c.legs == (a.legs[0], a.legs[1], b.legs[1])
    assert not np.any(c.to_dense())
    assert not np.any((b @ SymmetricTensor.random((Leg(v, OUT), Leg(w, IN)), seed=1)).to_dense())


def test_the_scalar_reductions(spaces):
    v, w = spaces
    e = _empty(v, w)
    assert tenet.norm(e) == 0.0
    assert tenet.inner(e, e) == 0.0
    traceable = SymmetricTensor.random((Leg(v, OUT), Leg(w, OUT), Leg(v, IN), Leg(v, IN)), seed=0)
    assert traceable.blocks == ()
    traced = tenet.trace(traceable, axes=(0, 2))
    assert not np.any(traced.to_dense())
    # full_trace wants a square map, and a square map couples through the identity, so
    # it only ever meets a block-less one when a leg's space is itself empty.
    square = SymmetricTensor.random((Leg(v, OUT), Leg(w, OUT), Leg(v, IN), Leg(w, IN)), seed=0)
    assert bool(square.blocks) == bool(w.sectors)
    assert float(tenet.full_trace(square)) == 0.0 or square.blocks


def test_svd_qr_and_lq_return_block_less_factors(spaces):
    v, w = spaces
    m = SymmetricTensor.random((Leg(v, OUT), Leg(w, IN)), seed=0)
    assert m.blocks == ()
    u, s, vh = linalg.svd(m)
    assert (u.blocks, s.blocks, vh.blocks) == ((), (), ())
    assert s.legs[0].space.sectors == ()
    assert tenet.allclose(u @ s @ vh, m)
    q, r = linalg.qr(m)
    assert (q.blocks, r.blocks) == ((), ())
    assert tenet.allclose(q @ r, m)
    left, right = linalg.lq(m)
    assert tenet.allclose(left @ right, m)


def test_polar_returns_the_zero_positive_factor(spaces):
    v, w = spaces
    m = SymmetricTensor.random((Leg(v, OUT), Leg(w, IN)), seed=0)
    isometry, positive = linalg.polar(m)
    assert isometry.blocks == ()
    assert not np.any(positive.to_dense())
    assert all(block.dtype == np.float64 for block in positive.blocks)
    assert tenet.allclose(isometry @ positive, m)


def test_the_double_layer_contraction_of_an_uncouplable_peps_site():
    """The state that started this: a charged physical leg on trivial bonds.

    The five-leg site of the evolution doctest, with ``phys`` carrying charge and every
    bond trivial: the codomain reaches only ``+-1`` and the domain only ``0``, so the
    site is block-less. Building the double layer over its physical leg used to raise
    ``IndexError`` out of the lowering rather than returning the implied zeros.
    """
    phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    v = GradedSpace.new(U1, {U1Sector(0): 1})
    legs = (Leg(v, IN), Leg(v, OUT), Leg(v, OUT), Leg(v, IN), Leg(phys, OUT))
    site = SymmetricTensor.random(legs, seed=0)
    assert site.blocks == ()

    double = tenet.einsum("abcdp,efghp->aebfcgdh", site, tenet.adjoint(site))
    assert double.shape == (1, 1, 1, 1, 1, 1, 1, 1)
    assert not np.any(double.to_dense())
    assert np.allclose(
        double.to_dense(),
        np.einsum("abcdp,efghp->aebfcgdh", site.to_dense(), tenet.adjoint(site).to_dense()),
    )
