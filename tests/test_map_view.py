"""Tests for the coupled-sector matrix lowering — issue #29."""

import dataclasses
import math
import pathlib

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import (
    _degeneracies,
    from_matrices,
    map_layout,
    to_matrices,
    tree_structure,
)
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry import SU2, U1, SU2Sector, Trivial, TrivialSector, U1Sector

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 3, ONE: 2})
W = GradedSpace.new(SU2, {HALF: 2})
H = GradedSpace.new(SU2, {HALF: 3})
TRIV = GradedSpace.new(Trivial, {TrivialSector(): 3})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})

# three sectors per space, legs alternating OUT/IN: the grid-completeness case
GRID_LEGS = (Leg(V, OUT), Leg(V, IN), Leg(V, OUT), Leg(V, IN))
# (1/2, 1/2, 1/2) -> 1/2 has two trees: two row bands with the same external sectors
TREE_LEGS = (Leg(H, OUT), Leg(H, OUT), Leg(H, OUT), Leg(H, IN))
SU2_LEGS = (Leg(V, OUT), Leg(W, IN), Leg(V, OUT))
NORM_LEGS = (Leg(V, OUT), Leg(W, IN), Leg(V, OUT), Leg(W, IN))
U1_LEGS = (Leg(Q, OUT), Leg(Q, IN), Leg(Q, OUT))
TRIV_LEGS = (Leg(TRIV, OUT), Leg(TRIV, IN))
ALL_OUT_LEGS = (Leg(W, OUT), Leg(W, OUT))
ALL_IN_LEGS = (Leg(W, IN), Leg(W, IN))

ROUND_TRIP_LEGS = [
    pytest.param(TRIV_LEGS, id="trivial"),
    pytest.param(U1_LEGS, id="u1"),
    pytest.param(SU2_LEGS, id="su2"),
    pytest.param(GRID_LEGS, id="su2-grid"),
    pytest.param(TREE_LEGS, id="su2-two-trees"),
    pytest.param(ALL_OUT_LEGS, id="su2-empty-in"),
    pytest.param(ALL_IN_LEGS, id="su2-empty-out"),
]


def use_jax():
    jax = pytest.importorskip("jax")
    # Simplification: global x64, same as tests/ops/test_basic.py — the 1e-12
    # tolerances below are meaningless in float32.
    jax.config.update("jax_enable_x64", True)
    return jax


def dense_matrix(t):
    """``t.to_dense()`` regrouped as (OUT physical axes) x (IN physical axes)."""
    order = (*t.structure.out_axes, *t.structure.in_axes)
    dense = t.to_dense().transpose(order)
    n_out = len(t.structure.out_axes)
    rows = int(np.prod(dense.shape[:n_out], dtype=int))
    return dense.reshape(rows, -1)


# --- the layout object ----------------------------------------------------------


def test_layout_is_cached_frozen_and_array_free():
    s = TensorStructure(GRID_LEGS)
    layout = map_layout(s)
    assert map_layout(TensorStructure(GRID_LEGS)) is layout
    assert map_layout(s) is layout
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.sectors = ()
    assert hash(layout) == hash(map_layout(s))
    for field in dataclasses.fields(layout):
        value = getattr(layout, field.name)
        assert not hasattr(value, "shape"), field.name
        assert isinstance(value, TensorStructure | tuple), field.name


def test_layout_sectors_are_sorted_and_exactly_the_coupled_ones():
    s = TensorStructure(GRID_LEGS)
    layout = map_layout(s)
    assert layout.sectors == tuple(sorted(layout.sectors))
    assert set(layout.sectors) == {k.coupled for k in s.block_order}


def test_layout_axes_order_is_out_then_in():
    s = TensorStructure(GRID_LEGS)
    assert map_layout(s).axes_order == (*s.out_axes, *s.in_axes)


@pytest.mark.parametrize("legs", ROUND_TRIP_LEGS)
def test_layout_shape_matches_an_independent_sum_of_block_shapes(legs):
    s = TensorStructure(legs)
    layout = map_layout(s)
    for c in layout.sectors:
        keys = [k for k in s.block_order if k.coupled == c]
        rows = {
            k.output_tree: int(np.prod([s.block_shape(k)[a] for a in s.out_axes], dtype=int))
            for k in keys
        }
        cols = {
            k.input_tree: int(np.prod([s.block_shape(k)[a] for a in s.in_axes], dtype=int))
            for k in keys
        }
        assert layout.shape(c) == (sum(rows.values()), sum(cols.values()))


@pytest.mark.parametrize("legs", ROUND_TRIP_LEGS)
def test_bands_tile_their_axis_contiguously(legs):
    layout = map_layout(TensorStructure(legs))
    for c in layout.sectors:
        for bands, total in (
            (layout.row_bands(c), layout.shape(c)[0]),
            (layout.col_bands(c), layout.shape(c)[1]),
        ):
            offset = 0
            for _, off, extent in bands:
                assert off == offset
                offset += extent
            assert offset == total


def test_band_order_is_block_order_laid_out_by_configuration():
    """Derived, never invented — this is what makes #30's matmul re-indexing-free.

    The trees are ``block_order``'s and nothing else; what the layout adds is the order.
    That order is now the *irrep configuration* order: the bands are laid out one
    configuration -- one choice of uncoupled irrep per leg on this side -- at a time,
    configurations sorted by ``(degeneracies, uncoupled labels)`` and the trees inside a
    configuration in the block order they already arrive in.

    This replaces the ``(extent, degeneracies, tree)`` rule. It is strictly finer: a
    configuration's bands share an extent *and* a shape by construction, whereas equal
    extents may carry different shapes — ``(1, 2, 3)`` and ``(3, 2, 1)`` are both six
    rows tall. Both keys are functions of *this side's* ordered legs, so what the old
    rule bought is unchanged: the two operands of a composition still agree band for
    band, and a configuration landing in one contiguous stripe is what lets the block
    cut read a whole rectangle of the grid with one slice.
    """
    s = TensorStructure(GRID_LEGS)
    layout = map_layout(s)
    for c in layout.sectors:
        keys = [k for k in s.block_order if k.coupled == c]
        for bands, trees, axes in (
            (layout.row_bands(c), dict.fromkeys(k.output_tree for k in keys), s.out_axes),
            (layout.col_bands(c), dict.fromkeys(k.input_tree for k in keys), s.in_axes),
        ):
            position = {tree: i for i, tree in enumerate(trees)}
            assert [tree for tree, _, _ in bands] != []
            assert {tree for tree, _, _ in bands} == set(position)

            # one configuration at a time, and each in one uninterrupted run
            configs = [tree.uncoupled for tree, _, _ in bands]
            assert len(set(configs)) == len({tuple(g) for g in configs})
            runs = [u for i, u in enumerate(configs) if i == 0 or configs[i - 1] != u]
            assert len(runs) == len(set(runs))

            # configurations ordered by (degeneracies, labels); trees by block order
            keyed = [(_degeneracies(s, axes, tree), tree.uncoupled) for tree, _, _ in bands]
            assert keyed == sorted(keyed)
            for (dims, u), (next_dims, next_u) in zip(keyed, keyed[1:], strict=False):
                assert (dims, u) <= (next_dims, next_u)
            for (tree, _, _), (nxt, _, _) in zip(bands, bands[1:], strict=False):
                if tree.uncoupled == nxt.uncoupled:
                    assert position[tree] < position[nxt]  # inside a configuration

            # a band's extent is its configuration's, so one configuration is one shape
            for tree, _, extent in bands:
                assert extent == math.prod(_degeneracies(s, axes, tree))


def test_the_tree_table_is_degeneracy_independent():
    """One entry serves every bond dimension: the key excludes the degeneracies.

    The same irreps and the same leg pattern at ``{SU2Sector(j): 2}`` and at
    ``{SU2Sector(j): j + 1}`` are one cache entry and one table — which is what makes a
    structure-keyed table survive an optimization that moves the degeneracies at every
    bond. TensorKit keys ``sectorstructure`` on sector content alone and frostspin keys
    its structural table on irrep labels; this is the same statement, asserted.
    """
    irreps = [SU2Sector(j) for j in range(4)]
    flat = GradedSpace.new(SU2, {a: 2 for a in irreps})
    ragged = GradedSpace.new(SU2, {a: a.two_j + 1 for a in irreps})
    pattern = (OUT, OUT, IN, IN)
    a, b = (
        TensorStructure(tuple(Leg(space, side) for side in pattern)) for space in (flat, ragged)
    )
    assert a != b  # the structures differ, and only in their degeneracies
    assert tree_structure(a) is tree_structure(b)

    table = tree_structure(a)
    assert table  # the fixture is not empty
    for _, out_configs, in_configs in table:
        for configs in (out_configs, in_configs):
            assert [u for u, _ in configs] == sorted(u for u, _ in configs)
            assert all(trees for _, trees in configs)
            assert all(t.uncoupled == u for u, trees in configs for t in trees)
    # 1,967 structural entries at SU(2) rank 6 either way; here, the same count
    assert a.num_blocks == b.num_blocks


# --- grid completeness ----------------------------------------------------------


def test_su2_grid_is_complete_and_partitions_block_order():
    s = TensorStructure(GRID_LEGS)
    assert len({leg.space for leg in s.legs}) == 1
    assert len(V.sectors) >= 3
    layout = map_layout(s)

    seen = []
    for c, cells in layout.grid:
        rbands, cbands = layout.row_bands(c), layout.col_bands(c)
        assert len(cells) == len(rbands)
        for row, (ot, _, _) in zip(cells, rbands, strict=True):
            assert len(row) == len(cbands)
            for i, (it, _, _) in zip(row, cbands, strict=True):
                assert 0 <= i < s.num_blocks
                assert s.block_order[i] == FusionBlockKey(ot, it)
                seen.append(i)
    assert sorted(seen) == list(range(s.num_blocks))  # no index twice, none missing


def test_su2_two_trees_with_equal_external_sectors_get_distinct_row_bands():
    s = TensorStructure(TREE_LEGS)
    layout = map_layout(s)
    bands = layout.row_bands(HALF)
    same = [(tree, off) for tree, off, _ in bands if tree.uncoupled == (HALF, HALF, HALF)]
    assert len(same) == 2
    (t0, o0), (t1, o1) = same
    assert t0 != t1 and t0.inner != t1.inner  # told apart only by the internal line
    assert o0 != o1


# --- to_matrices / from_matrices ------------------------------------------------


@pytest.mark.parametrize("legs", ROUND_TRIP_LEGS)
def test_to_matrices_keys_and_shapes(legs):
    t = SymmetricTensor.random(legs, seed=0)
    layout = map_layout(t.structure)
    mats = to_matrices(t)
    assert set(mats) == set(layout.sectors)
    for c, m in mats.items():
        assert tuple(m.shape) == layout.shape(c)


@pytest.mark.parametrize("legs", ROUND_TRIP_LEGS)
def test_round_trip_is_exact(legs):
    t = SymmetricTensor.random(legs, seed=1)
    back = from_matrices(t.structure, to_matrices(t))
    assert back.structure is t.structure
    assert back == t
    for a, b in zip(back.blocks, t.blocks, strict=True):
        assert np.array_equal(a, b)  # exact, not allclose


def test_to_matrices_does_not_mutate_the_tensor():
    t = SymmetricTensor.random(GRID_LEGS, seed=2)
    before = tuple(b.copy() for b in t.blocks)
    structure = t.structure
    _ = to_matrices(t)
    assert t.structure is structure
    for a, b in zip(t.blocks, before, strict=True):
        assert np.array_equal(a, b)


def test_blocks_land_where_the_layout_says():
    t = SymmetricTensor.random(GRID_LEGS, seed=3)
    layout = map_layout(t.structure)
    mats = to_matrices(t)
    order = layout.axes_order
    for c, cells in layout.grid:
        for row, (_, ro, dr) in zip(cells, layout.row_bands(c), strict=True):
            for i, (_, co, dc) in zip(row, layout.col_bands(c), strict=True):
                want = t.blocks[i].transpose(order).reshape(dr, dc)
                np.testing.assert_array_equal(mats[c][ro : ro + dr, co : co + dc], want)


def test_linearity():
    a = SymmetricTensor.random(NORM_LEGS, seed=4)
    b = SymmetricTensor.random(NORM_LEGS, seed=5)
    ma, mb = to_matrices(a), to_matrices(b)
    mc = to_matrices(2.5 * a + b * (-0.75))
    for c in mc:
        np.testing.assert_allclose(mc[c], 2.5 * ma[c] - 0.75 * mb[c], atol=1e-12)


# --- norm identity --------------------------------------------------------------


def test_su2_norm_identity_and_the_necessity_of_the_qdim_weight():
    t = SymmetricTensor.random(NORM_LEGS, seed=6)
    mats = to_matrices(t)
    assert len({SU2.qdim(c) for c in mats}) > 1
    weighted = sum(SU2.qdim(c) * float(np.sum(np.abs(m) ** 2)) for c, m in mats.items())
    assert tenet.norm(t) ** 2 == pytest.approx(weighted, abs=1e-12)

    unweighted = sum(float(np.sum(np.abs(m) ** 2)) for m in mats.values())
    assert abs(unweighted - weighted) > 0.1 * weighted  # not vacuously true


# --- dense oracle ---------------------------------------------------------------


def test_su2_singular_values_match_the_dense_matrix():
    t = SymmetricTensor.random(NORM_LEGS, seed=7)
    dense = dense_matrix(t)
    want = np.sort(np.linalg.svd(dense, compute_uv=False))[::-1]

    got = np.sort(
        np.concatenate(
            [
                np.repeat(np.linalg.svd(m, compute_uv=False), SU2.irrep_dim(c))
                for c, m in to_matrices(t).items()
            ]
        )
    )[::-1]
    n = min(len(got), len(want))
    np.testing.assert_allclose(got[:n], want[:n], atol=1e-10)
    np.testing.assert_allclose(np.concatenate([got[n:], want[n:]]), 0.0, atol=1e-10)
    assert np.linalg.matrix_rank(dense, tol=1e-10) == int(np.sum(got > 1e-10))


# --- errors ---------------------------------------------------------------------


def test_from_matrices_rejects_a_wrong_shape():
    t = SymmetricTensor.random(SU2_LEGS, seed=8)
    mats = to_matrices(t)
    c = next(iter(mats))
    mats[c] = np.zeros((mats[c].shape[0] + 1, mats[c].shape[1]))
    with pytest.raises(ValueError) as e:
        from_matrices(t.structure, mats)
    assert repr(c) in str(e.value)
    assert str(map_layout(t.structure).shape(c)) in str(e.value)
    assert str(tuple(mats[c].shape)) in str(e.value)


def test_from_matrices_rejects_a_missing_sector():
    t = SymmetricTensor.random(SU2_LEGS, seed=9)
    mats = to_matrices(t)
    c = next(iter(mats))
    del mats[c]
    with pytest.raises(ValueError) as e:
        from_matrices(t.structure, mats)
    assert repr(c) in str(e.value)
    assert str(map_layout(t.structure).shape(c)) in str(e.value)


def test_from_matrices_rejects_an_unknown_sector():
    t = SymmetricTensor.random(SU2_LEGS, seed=10)
    mats = to_matrices(t)
    alien = SU2Sector(101)
    mats[alien] = np.zeros((1, 1))
    with pytest.raises(ValueError) as e:
        from_matrices(t.structure, mats)
    assert repr(alien) in str(e.value)


def test_from_matrices_rejects_matrices_of_mixed_dtype():
    """The dtype refusal lives here, on the untrusted input, not on the blocks (#328)."""
    t = SymmetricTensor.random(GRID_LEGS, seed=11)
    mats = to_matrices(t)
    c = next(iter(mats))
    mats[c] = mats[c].astype(np.complex128)
    assert len(mats) > 1
    with pytest.raises(ValueError, match="from_matrices: matrices must share one dtype"):
        from_matrices(t.structure, mats)


def test_from_matrices_still_produces_a_tensor_the_checked_constructor_accepts():
    t = SymmetricTensor.random(SU2_LEGS, seed=12)
    back = from_matrices(t.structure, to_matrices(t))
    shapes = back.structure.block_shapes
    assert all(tuple(b.shape) == s for b, s in zip(back.blocks, shapes, strict=True))
    assert SymmetricTensor(back.structure, back.blocks) == back  # the trust boundary, re-run


# --- backend agnosticism --------------------------------------------------------


@pytest.mark.parametrize("legs", ROUND_TRIP_LEGS)
def test_jax_round_trip_is_exact(legs):
    use_jax()
    t = SymmetricTensor.random(legs, seed=11).to_backend("jax")
    back = from_matrices(t.structure, to_matrices(t))
    assert back.backend == "jax"
    assert back == t


def test_jax_norm_identity():
    use_jax()
    t = SymmetricTensor.random(NORM_LEGS, seed=12).to_backend("jax")
    mats = to_matrices(t)
    assert {ar.infer_backend(m) for m in mats.values()} == {"jax"}
    weighted = sum(SU2.qdim(c) * float(np.sum(np.abs(np.asarray(m)) ** 2)) for c, m in mats.items())
    assert tenet.norm(t) ** 2 == pytest.approx(weighted, abs=1e-12)


def test_jax_matches_numpy_matrices():
    use_jax()
    t = SymmetricTensor.random(GRID_LEGS, seed=13)
    numpy_mats = to_matrices(t)
    jax_mats = to_matrices(t.to_backend("jax"))
    assert set(jax_mats) == set(numpy_mats)
    for c, m in jax_mats.items():
        np.testing.assert_allclose(np.asarray(m), numpy_mats[c], atol=1e-12)


# --- module hygiene -------------------------------------------------------------


def test_map_view_has_no_numpy_no_to_dense_and_no_provider_branching():
    src = pathlib.Path(tenet.map_view.__file__).read_text()
    assert "import numpy" not in src
    assert "np." not in src
    assert "to_dense(" not in src
    assert "if provider ==" not in src
    assert "isinstance(provider" not in src
