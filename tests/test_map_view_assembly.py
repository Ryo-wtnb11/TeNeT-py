"""The two ``to_matrices`` assemblies agree, and the mutable one really writes in place.

Issue #257 (M69). ``map_view.to_matrices`` picks its assembly on the blocks' backend:
pure concatenation where arrays are immutable, one strided copy per block into a
preallocated matrix where they are not. The concatenating path defines the values, so
the only thing that has to be checked is that the other one reproduces them exactly --
and that its copy really is one copy, i.e. that the destination reshape is a view and
not a discarded temporary, which would make every write silently vanish.
"""

import numpy as np
import pytest

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import _concatenated, _tables, from_matrices, map_layout, to_matrices
from tenet.symmetry import SU2, U1, SU2Sector, Trivial, TrivialSector, U1Sector

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 3, ONE: 2})
W = GradedSpace.new(SU2, {HALF: 2})
H = GradedSpace.new(SU2, {HALF: 3})
TRIV = GradedSpace.new(Trivial, {TrivialSector(): 3})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})

LEGS = [
    pytest.param((Leg(TRIV, OUT), Leg(TRIV, IN)), id="trivial"),
    pytest.param((Leg(Q, OUT), Leg(Q, IN), Leg(Q, OUT)), id="u1"),
    pytest.param((Leg(V, OUT), Leg(W, IN), Leg(V, OUT)), id="su2"),
    # three sectors per space, legs alternating OUT/IN: the grid-completeness case
    pytest.param((Leg(V, OUT), Leg(V, IN), Leg(V, OUT), Leg(V, IN)), id="su2-grid"),
    # (1/2, 1/2, 1/2) -> 1/2 has two trees: two row bands, same external sectors
    pytest.param((Leg(H, OUT), Leg(H, OUT), Leg(H, OUT), Leg(H, IN)), id="su2-two-trees"),
    pytest.param((Leg(W, OUT), Leg(W, OUT)), id="su2-empty-in"),
    pytest.param((Leg(W, IN), Leg(W, IN)), id="su2-empty-out"),
]


def _reference(t: SymmetricTensor) -> dict:
    """``to_matrices`` forced onto the concatenating path, whatever the backend is."""
    layout = map_layout(t.structure)
    bands, _ = _tables(t.structure)
    order = layout.axes_order
    return _concatenated(t, layout, bands, order, order != tuple(range(t.ndim)))


@pytest.mark.parametrize("legs", LEGS)
def test_the_two_assemblies_are_bit_identical(legs):
    t = SymmetricTensor.random(legs, seed=0)
    got, want = to_matrices(t), _reference(t)
    assert sorted(got) == sorted(want)
    for c in want:
        # bit-identical, not allclose: the mutable path places the same bytes
        assert np.array_equal(got[c], want[c]), c


@pytest.mark.parametrize("legs", LEGS)
def test_the_round_trip_is_exact_through_the_in_place_assembly(legs):
    t = SymmetricTensor.random(legs, seed=1)
    back = from_matrices(t.structure, to_matrices(t))
    for a, b in zip(t.blocks, back.blocks, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("legs", LEGS)
def test_from_matrices_stays_zero_copy(legs):
    """Every block on the way back is a view into its sector matrix, never a copy."""
    t = SymmetricTensor.random(legs, seed=2)
    mats = to_matrices(t)
    back = from_matrices(t.structure, mats)
    for block in back.blocks:
        assert any(np.shares_memory(block, m) for m in mats.values())


@pytest.mark.parametrize("legs", LEGS)
def test_the_destination_reshape_is_a_view_not_a_temporary(legs):
    """The claim the in-place path rests on: splitting a 2-D slice's axes never copies.

    If it copied, the assignment would land in a discarded temporary and every matrix
    would come back holding whatever ``empty`` allocated.
    """
    t = SymmetricTensor.random(legs, seed=3)
    layout = map_layout(t.structure)
    bands, shapes = _tables(t.structure)
    for (c, cells), (rbands, cbands) in zip(layout.grid, bands, strict=True):
        out = np.empty(layout.shape(c))
        for (_, ro, dr), indices in zip(rbands, cells, strict=True):
            for i, (_, co, dc) in zip(indices, cbands, strict=True):
                view = out[ro : ro + dr, co : co + dc].reshape(shapes[i])
                assert np.shares_memory(view, out), (c, i)


# The in-place path on PyTorch is not exercised here: tests/backends/test_torch.py
# holds a hygiene fence making itself the only test module allowed to import torch, and
# it is an existing test. Its own suite runs every map-view operation on torch tensors,
# so the path is covered there -- including under autograd.
