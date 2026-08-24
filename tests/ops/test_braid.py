"""Tests for ``tenet.braid`` — issue #287 (M82).

``levels`` are the lines' *incoming* planar order; a pair crosses when that order
and the leg order disagree. Three things pin the primitive: monotone levels are
``transpose`` (bit for bit, plan object included), an inverted pair with the
identity permutation is YASTN's ``swap_gate`` on a dense fermionic oracle, and the
crossings compose like a symmetric braiding (involution, Yang-Baxter).
"""

import itertools

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.permutation import (
    _pattern_braid_plan,
    _pattern_plan,
    braid_plan,
    permutation_plan,
)
from tenet.structure import _pattern
from tenet.symmetry import SU2, U1, FZ2Sector, SU2Sector, U1Sector, fZ2

FV = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})
FZ2_LEGS = (Leg(FV, OUT), Leg(FV, OUT), Leg(FV, IN))
# the dense basis of ``FV``: two even columns then two odd ones
FZ2_PARITY = (0, 0, 1, 1)

QV = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2})
U1_LEGS = (Leg(QV, OUT), Leg(QV, OUT), Leg(QV, IN))

SV = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2})
SU2_LEGS = (Leg(SV, OUT), Leg(SV, OUT), Leg(SV, IN))

ALL_LEGS = pytest.mark.parametrize("legs", [FZ2_LEGS, U1_LEGS, SU2_LEGS], ids=["fZ2", "U1", "SU2"])


def tensor(legs, seed=0):
    return SymmetricTensor.random(legs, seed=seed)


def crossed(levels):
    pairs = itertools.combinations(range(len(levels)), 2)
    return [(i, j) for i, j in pairs if levels[i] > levels[j]]


# --- monotone levels are exactly transpose ---------------------------------


@ALL_LEGS
@pytest.mark.parametrize("axes", list(itertools.permutations(range(3))))
def test_monotone_levels_are_transpose_bit_for_bit(legs, axes):
    t = tensor(legs)
    want = tenet.transpose(t, axes)
    for levels in [(0, 1, 2), (0, 0, 0), (3, 7, 9), (1, 1, 4)]:
        got = tenet.braid(t, axes, levels)
        assert got.structure == want.structure
        for a, b in zip(got.blocks, want.blocks, strict=True):
            assert np.array_equal(a, b)


@ALL_LEGS
def test_monotone_levels_reuse_the_transpose_plan(legs):
    """Not merely equal — the same plan, so the fast path cannot drift."""
    t = tensor(legs)
    assert braid_plan(t.structure, (2, 0, 1), (0, 1, 2)) == permutation_plan(t.structure, (2, 0, 1))
    pattern = _pattern(t.structure)
    assert _pattern_braid_plan(pattern, (2, 0, 1), (0, 1, 2)) is _pattern_plan(pattern, (2, 0, 1))


# --- the swap gate ----------------------------------------------------------


@pytest.mark.parametrize("levels", list(itertools.permutations(range(3))))
def test_swap_gate_matches_yastn_on_a_dense_fermionic_oracle(levels):
    """``(-1)^(sum p_i p_j)`` over the crossed pairs — YASTN's ``swap_gate``."""
    t = tensor(FZ2_LEGS)
    dense = t.to_dense()
    sign = np.ones_like(dense)
    for idx in itertools.product(*(range(n) for n in dense.shape)):
        odd = sum(FZ2_PARITY[idx[i]] * FZ2_PARITY[idx[j]] for i, j in crossed(levels))
        sign[idx] = -1 if odd % 2 else 1
    assert np.allclose(tenet.braid(t, (0, 1, 2), levels).to_dense(), sign * dense)


def test_the_swap_gate_leaves_the_legs_alone():
    t = tensor(FZ2_LEGS)
    swapped = tenet.braid(t, (0, 1, 2), (1, 0, 2))
    assert swapped.legs == t.legs
    assert swapped != t  # the odd blocks did change sign


@pytest.mark.parametrize("levels", [(1, 0, 2), (0, 2, 1), (2, 1, 0)])
def test_the_swap_gate_squares_to_the_identity(levels):
    t = tensor(FZ2_LEGS)
    assert tenet.braid(tenet.braid(t, (0, 1, 2), levels), (0, 1, 2), levels) == t


# --- braid o braid^-1 = id --------------------------------------------------


@ALL_LEGS
@pytest.mark.parametrize("axes", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("levels", [(0, 1, 2), (1, 0, 2), (2, 1, 0)])
def test_braid_undone_by_its_inverse_is_the_identity(legs, axes, levels):
    """Undo the permutation, then re-cross the same pairs: bit for bit ``t`` again.

    The inverse exists because the crossings factor: ``braid(axes, levels)`` is the
    swap gate of ``levels`` followed by ``transpose(axes)``, and both are involutions
    up to their own inverse.
    """
    t = tensor(legs)
    back = tuple(sorted(range(3), key=lambda i: axes[i]))
    undone = tenet.braid(tenet.transpose(tenet.braid(t, axes, levels), back), (0, 1, 2), levels)
    assert undone.structure == t.structure
    for a, b in zip(undone.blocks, t.blocks, strict=True):
        assert np.array_equal(a, b)


@ALL_LEGS
@pytest.mark.parametrize("axes", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("levels", [(1, 0, 2), (2, 1, 0), (0, 2, 1)])
def test_the_crossings_read_only_the_incoming_and_outgoing_orders(legs, axes, levels):
    """A pair crosses iff the level order and the final leg order disagree about it,
    so the level crossings factor out of the permutation as a plain swap gate."""
    t = tensor(legs)
    want = tenet.transpose(tenet.braid(t, (0, 1, 2), levels), axes)
    got = tenet.braid(t, axes, levels)
    for a, b in zip(got.blocks, want.blocks, strict=True):
        assert np.array_equal(a, b)


# --- Yang-Baxter ------------------------------------------------------------


@ALL_LEGS
def test_yang_baxter_on_three_lines(legs):
    """``b1 b2 b1 == b2 b1 b2``: two spellings of the reversal, crossings and all."""
    t = tensor(legs)
    b1, b2 = (1, 0, 2), (0, 2, 1)

    def word(*steps):
        out = t
        for axes in steps:
            out = tenet.braid(out, axes, (0, 1, 2))
        return out

    left, right = word(b1, b2, b1), word(b2, b1, b2)
    assert left.structure == right.structure
    for a, b in zip(left.blocks, right.blocks, strict=True):
        assert np.allclose(a, b)


@ALL_LEGS
def test_the_crossing_sense_does_not_matter(legs):
    """Symmetric braiding: over and under are the same crossing, so a pair that the
    permutation inverts *and* the levels cross is not crossed at all."""
    t = tensor(legs)
    over = tenet.braid(t, (1, 0, 2), (0, 1, 2))  # one crossing, from the permutation
    under = tenet.braid(t, (1, 0, 2), (1, 0, 2))  # the levels un-cross it again
    assert over == tenet.transpose(t, (1, 0, 2))
    assert under == tenet.braid(over, (0, 1, 2), (1, 0, 2))


@ALL_LEGS
def test_yang_baxter_with_the_permutation_spelled_out(legs):
    """The same word written as one reversing braid, crossings and all."""
    t = tensor(legs)
    stepwise = t
    for axes, levels in (((1, 0, 2), (0, 1, 2)), ((0, 2, 1), (0, 1, 2)), ((1, 0, 2), (0, 1, 2))):
        stepwise = tenet.braid(stepwise, axes, levels)
    one_shot = tenet.braid(t, (2, 1, 0), (0, 1, 2))
    assert stepwise.structure == one_shot.structure
    for a, b in zip(stepwise.blocks, one_shot.blocks, strict=True):
        assert np.allclose(a, b)


# --- a bosonic provider has no crossing to pay ------------------------------


@pytest.mark.parametrize("legs", [U1_LEGS, SU2_LEGS], ids=["U1", "SU2"])
@pytest.mark.parametrize("levels", list(itertools.permutations(range(3))))
def test_a_bosonic_crossing_is_the_identity_morphism(legs, levels):
    """R is trivial on the grading of a bosonic provider, so braid == transpose."""
    t = tensor(legs)
    assert tenet.braid(t, (0, 1, 2), levels) == tenet.transpose(t, (0, 1, 2))


# --- validation -------------------------------------------------------------


def test_levels_of_the_wrong_length_are_refused():
    t = tensor(FZ2_LEGS)
    with pytest.raises(ValueError, match="length 2, expected 3"):
        tenet.braid(t, (0, 1, 2), (0, 1))


def test_non_integer_levels_are_refused():
    t = tensor(FZ2_LEGS)
    with pytest.raises(ValueError, match="must all be integers"):
        tenet.braid(t, (0, 1, 2), (0.0, 1.0, 2.0))


def test_a_bad_permutation_is_refused_as_for_transpose():
    t = tensor(FZ2_LEGS)
    with pytest.raises(ValueError, match="repeated"):
        tenet.braid(t, (0, 0, 1), (0, 1, 2))
