"""The two ``to_matrices`` assemblies agree, and the mutable one really writes in place.

Issue #257 (M69). ``map_view.to_matrices`` picks its assembly on the blocks' backend:
pure concatenation where arrays are immutable, one strided copy per block into a
preallocated matrix where they are not. The concatenating path defines the values, so
the only thing that has to be checked is that the other one reproduces them exactly --
and that its copy really is one copy, i.e. that the destination reshape is a view and
not a discarded temporary, which would make every write silently vanish.
"""

import autoray as ar
import numpy as np
import pytest
from helpers import count_backend_calls

import tenet.map_view as map_view_module
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import (
    _concatenated,
    _rects,
    _tables,
    from_matrices,
    map_layout,
    to_matrices,
)
from tenet.symmetry import SU2, U1, SU2Sector, Trivial, TrivialSector, U1Sector

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 3, ONE: 2})
W = GradedSpace.new(SU2, {HALF: 2})
H = GradedSpace.new(SU2, {HALF: 3})
TRIV = GradedSpace.new(Trivial, {TrivialSector(): 3})
RAGGED = GradedSpace.new(SU2, {SU2Sector(j): j + 1 for j in range(4)})
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
    # degeneracy 1, 2, 3, 4: equal degeneracies make every grid uniform and leave the
    # ragged half of the rectangle cut unexercised
    pytest.param(
        (Leg(RAGGED, OUT), Leg(RAGGED, OUT), Leg(RAGGED, IN), Leg(RAGGED, IN)),
        id="su2-ragged",
    ),
    # ... and with the sides interleaved, so ``axes_order`` is a real permutation
    pytest.param(
        (Leg(RAGGED, OUT), Leg(RAGGED, IN), Leg(RAGGED, OUT), Leg(RAGGED, IN)),
        id="su2-ragged-permuted",
    ),
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


def _count_ar_do(monkeypatch, fn):
    """Every ``ar.do`` name ``fn()`` reaches, counted. Sees the module's own calls."""
    counts: dict[str, int] = {}
    real = ar.do

    def spy(name, *args, **kwargs):
        counts[name] = counts.get(name, 0) + 1
        return real(name, *args, **kwargs)

    monkeypatch.setattr(map_view_module.ar, "do", spy)
    fn()
    return counts


@pytest.mark.parametrize("legs", LEGS)
def test_the_in_place_path_moves_each_block_exactly_once(monkeypatch, legs):
    """One copy per block, and not one concatenate or materialising reshape anywhere.

    The concatenating path's three passes are exactly ``reshape`` per block plus a
    ``concatenate`` per band; if either name reappears here the extra pass is back.
    """
    t = SymmetricTensor.random(legs, seed=5)
    counts = _count_ar_do(monkeypatch, lambda: to_matrices(t))
    assert "concatenate" not in counts
    assert "reshape" not in counts
    # one preallocation per coupled sector, and nothing else that allocates
    assert counts.get("empty", 0) == len(map_layout(t.structure).sectors)


@pytest.mark.parametrize("legs", LEGS)
def test_the_source_block_is_never_copied_before_the_write(legs):
    """``to_matrices`` reads the caller's blocks through a view, never a materialisation."""
    t = SymmetricTensor.random(legs, seed=6)
    layout = map_layout(t.structure)
    order = layout.axes_order
    for i, block in enumerate(t.blocks):
        view = block if order == tuple(range(t.ndim)) else np.transpose(block, order)
        assert np.shares_memory(view, t.blocks[i])


@pytest.mark.parametrize("legs", LEGS)
def test_the_concatenating_path_still_only_concatenates(monkeypatch, legs):
    """The immutable-backend path is unchanged: no ``empty``, nothing written in place."""
    t = SymmetricTensor.random(legs, seed=7)
    counts = _count_ar_do(monkeypatch, lambda: _reference(t))
    assert "empty" not in counts
    assert "zeros" not in counts


# --- the rectangle cut ``from_matrices`` reads back through (#324) --------------------


def _extent_pairs(structure) -> int:
    """Distinct ``(row extent, column extent)`` pairs of the layout, summed over sectors."""
    layout = map_layout(structure)
    return sum(
        len({extent for _, _, extent in layout.row_bands(c)})
        * len({extent for _, _, extent in layout.col_bands(c)})
        for c in layout.sectors
    )


@pytest.mark.parametrize("legs", LEGS)
def test_the_rectangles_are_the_distinct_extent_pairs_and_cover_the_grid(legs):
    """The cut is the layout's, and it is complete: every block in exactly one cell."""
    structure = SymmetricTensor.random(legs, seed=8).structure
    rects = _rects(structure)
    assert sum(len(r) for _, r in rects) == _extent_pairs(structure)
    seen = [i for _, rectangles in rects for *_, indices in rectangles for i in indices]
    assert sorted(seen) == list(range(structure.num_blocks))


@pytest.mark.parametrize("shape", ["uniform", "ragged"])
def test_the_dispatch_count_is_the_extent_pairs_and_not_the_block_count(monkeypatch, shape):
    """The scaling claim: ``from_matrices`` dispatches per rectangle, never per block.

    The analogue of
    ``tests/ops/test_batch.py::test_the_dispatch_count_is_the_grouping_s_and_not_the_term_count_s``.
    A rectangle costs at most four calls -- split the slice, swap the middle axes, and
    where its cells share a shape, split them and permute them into public axis order --
    so the bound is that times the number of distinct extent pairs. It is the *ragged*
    fixture that makes the claim non-trivial: at equal degeneracies every grid is one
    rectangle and any grouping at all would pass.
    """
    space = V if shape == "uniform" else RAGGED
    legs = (Leg(space, OUT), Leg(space, OUT), Leg(space, OUT), Leg(space, IN), Leg(space, IN))
    t = SymmetricTensor.random(legs, seed=9)
    mats = to_matrices(t)
    _rects(t.structure)  # the cut is structural; not part of the dispatch count

    calls: list[str] = []
    with count_backend_calls(monkeypatch, lambda name, args, kwargs: calls.append(name)):
        back = from_matrices(t.structure, mats)

    pairs = _extent_pairs(t.structure)
    assert len(calls) <= 4 * pairs
    assert pairs * 4 < t.structure.num_blocks  # and the block count is the loser
    for a, b in zip(t.blocks, back.blocks, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("legs", LEGS)
def test_jax_takes_the_cell_walk_and_gets_the_same_numbers(legs):
    """JAX is outside the gate, and its blocks are the NumPy ones to the bit.

    Mirrors ``tests/ops/test_batch.py::test_jax_takes_the_loop_and_gets_the_same_numbers``.
    Nothing in the rectangle cut needs a NumPy-only primitive -- this is the test that
    would catch it if it did -- but whether trading Python iterations for array calls
    pays is a property of the backend, so only NumPy has been measured.
    """
    pytest.importorskip("jax")
    t = SymmetricTensor.random(legs, seed=10)
    back = from_matrices(t.structure, to_matrices(t.to_backend("jax")))
    assert back.backend == "jax"
    for a, b in zip(t.blocks, back.blocks, strict=True):
        assert np.array_equal(a, np.asarray(b))
