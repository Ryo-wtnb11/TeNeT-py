"""Tests for the coupled-sector matrix lowering — issue #29."""

import dataclasses
import functools
import math
import pathlib
import re
import tempfile

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
from tenet.symmetry import (
    SU2,
    U1,
    Z2,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    Z2Sector,
    fZ2,
)
from tenet.symmetry.sun import SUNProvider, SUNSector

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

# One space per symmetry the library ships, for the storage-contract count below. Ragged
# degeneracies where the symmetry allows them: a uniform one hides the case where the
# blocks of a coupled sector differ in shape.
Z2_SPACE = GradedSpace.new(Z2, {Z2Sector(0): 2, Z2Sector(1): 3})
FZ2_SPACE = GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 2})
SU3 = SUNProvider(3)
SU3_SPACE = GradedSpace.new(SU3, {SUNSector((0, 0)): 2, SUNSector((1, 0)): 2})
PRODUCT = ProductProvider((fZ2, U1, SU2))
PRODUCT_SPACE = GradedSpace.new(
    PRODUCT,
    {
        ProductSector((FZ2Sector(0), U1Sector(0), SU2Sector(0))): 2,
        ProductSector((FZ2Sector(1), U1Sector(1), SU2Sector(1))): 2,
    },
)


def _square(space):
    """Two OUT legs and two IN legs on one space -- a map with something to contract."""
    return (Leg(space, OUT), Leg(space, OUT), Leg(space, IN), Leg(space, IN))


EVERY_SYMMETRY = [
    pytest.param(_square(TRIV), id="trivial"),
    pytest.param(_square(Z2_SPACE), id="z2"),
    pytest.param(_square(Q), id="u1"),
    pytest.param(_square(FZ2_SPACE), id="fz2"),
    pytest.param(_square(V), id="su2"),
    pytest.param(_square(SU3_SPACE), id="su3"),
    pytest.param(_square(PRODUCT_SPACE), id="fz2xu1xsu2"),
]

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


def _body(path):
    """``path``'s source with every docstring removed.

    Prose may name a symmetry -- an example is allowed to be about SU(2) -- and the claim
    below is about the code.
    """
    return re.sub(r'"""[\s\S]*?"""', "", path.read_text())


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(path, id=path.stem)
        for path in sorted(pathlib.Path(tenet.ops.__file__).parent.glob("*.py"))
        + [pathlib.Path(tenet.map_view.__file__), pathlib.Path(tenet.structure.__file__)]
        if path.stem != "__init__"
    ],
)
def test_no_operation_branches_on_which_symmetry_it_has(path):
    """One code path per operation, whatever the symmetry is.

    A provider reaches the operations through two doors and only two: the coefficients it
    supplies, which a *plan* has folded into indices and scalars before any array moves,
    and ``requires(provider, Capability)``, which **refuses** — it says an operation is
    undefined for that symmetry, and raises. Neither is a branch on identity, and a
    ``provider != provider`` comparison between two operands is a check that they share a
    symmetry, not a choice of route.

    So the operations are one implementation each. A module that named a symmetry in its
    body would either be special-casing what the plan already carries, or building a fast
    path whose slow twin is the one everybody else gets tested on. Both are how a library
    acquires a symmetry it silently supports better than the others.
    """
    body = _body(path)
    for pattern in ("if provider ==", "isinstance(provider", "provider.name ==", "if symmetry =="):
        assert pattern not in body, f"{path.name} branches on the provider: {pattern}"
    for name in ("SU2", "U1Provider", "FZ2", "Z2Provider", "SUNProvider", "TrivialProvider"):
        assert not re.search(rf"\b{name}\b", body), f"{path.name} names {name} in its body"


# --- the storage is what the hot path reads -------------------------------------


def _blocks_reads(fn):
    """How many times ``fn()`` makes a tensor cut its blocks out of its matrices."""
    original = SymmetricTensor.blocks
    reads = 0

    def probe(self):
        nonlocal reads
        reads += self._views is None  # a memoized cut is not a cut
        return original.fget(self)

    SymmetricTensor.blocks = property(probe)
    try:
        fn()
    finally:
        SymmetricTensor.blocks = original
    return reads


def _matrix_backed(t):
    """``(structure, matrices)`` -- ``from_matrices`` of these holds no cut."""
    return t.structure, to_matrices(t)


@pytest.mark.parametrize(
    "legs", [pytest.param(U1_LEGS, id="u1"), pytest.param(GRID_LEGS, id="su2-grid")]
)
def test_the_hot_operations_never_cut_a_block(legs):
    """The storage contract, as a count: *zero*, not a bound with a constant in it.

    ``data`` is the storage and ``blocks`` is the interop view of it, so an operation
    that reads ``blocks`` cuts every block of its operand out of the matrices before it
    starts -- free-ish on NumPy, a graph node with a backward pass on a traced backend,
    and never free. Each operation below is defined per coupled sector or per plan term,
    and a term reaches its source through one slice of the source's own matrix, so none
    of them has any reason to ask.
    """
    # matrix-backed operands, which is what every intermediate of a chain is: a tensor
    # built from blocks keeps the cut its constructor already made, and would hide the
    # question by answering ``blocks`` out of that memo
    a = from_matrices(*_matrix_backed(SymmetricTensor.random(legs, seed=21)))
    b = from_matrices(*_matrix_backed(SymmetricTensor.random(legs, seed=22)))
    assert a._views is None and b._views is None

    def hot():
        tenet.norm(a)
        tenet.inner(a, b)
        tenet.transpose(a, (a.ndim - 1, *range(a.ndim - 1)))
        tenet.repartition(a, tuple(range(a.ndim - 1)), (a.ndim - 1,))
        tenet.tensordot(a, b, axes=((1,), (0,)))
        tenet.linalg.svd(a)

    assert _blocks_reads(hot) == 0


@pytest.mark.parametrize("legs", EVERY_SYMMETRY)
def test_the_hot_operations_never_cut_a_block_on_any_symmetry(legs):
    """The same count, on every symmetry the library ships. Still *zero*.

    Matrix-native is a property of the lowering, not of the provider: ``map_layout``
    defines ``T ≃ ⊕_c B_c ⊗ id_c`` for whatever the fusion rules are, composition is a
    matmul per coupled sector on all of them, and the symmetry enters through the
    coefficients -- F/R symbols, Clebsch-Gordan data, Koszul signs -- which the plan has
    already folded away by the time anything moves. So a provider that read ``blocks``
    here would be an operation taking a provider-specific path it does not need, not a
    symmetry that resists the storage.

    Fermions and product symmetries are in the list because they are where such a path
    would be: ``fZ2`` carries a sign per crossing and ``ProductProvider`` distributes
    every coefficient over its factors.
    """
    a = from_matrices(*_matrix_backed(SymmetricTensor.random(legs, seed=31)))
    b = from_matrices(*_matrix_backed(SymmetricTensor.random(legs, seed=32)))

    def hot():
        tenet.norm(a)
        tenet.inner(a, b)
        tenet.transpose(a, (1, 0, 3, 2))
        tenet.repartition(a, (0,), (1, 2, 3))
        tenet.tensordot(a, b, axes=((2, 3), (0, 1)))
        tenet.adjoint(a)
        tenet.conj(a)
        a + b
        a * 2.0
        tenet.linalg.svd(a)
        tenet.linalg.qr(a)
        tenet.flip_dual(a, 0)
        tenet.flip_dual(a, 2, inv=True)
        tenet.flip_dual(a, (0, 1, 2, 3))

    assert _blocks_reads(hot) == 0


# --- the whole public surface, against the same contract ------------------------


def _public_surface():
    """Every public callable the library exposes, as ``qualified name -> object``.

    Three doors and only three: ``tenet.__all__``, ``tenet.ops.linalg.__all__`` and the
    public attributes of ``SymmetricTensor``. Classes are not operations and drop out;
    everything left has to be classified below, which is what makes the classification a
    partition of the surface rather than a list somebody remembered to extend.
    """
    surface = {f"tenet.{name}": getattr(tenet, name) for name in tenet.__all__}
    surface |= {
        f"linalg.{name}": getattr(tenet.ops.linalg, name) for name in tenet.ops.linalg.__all__
    }
    surface |= {
        f"SymmetricTensor.{name}": getattr(SymmetricTensor, name)
        for name in vars(SymmetricTensor)
        if not name.startswith("_")
    }
    return {k: v for k, v in surface.items() if callable(v) and not isinstance(v, type)}


EXCLUDED = {
    # --- boundaries: where untrusted arrays enter, or where the caller decides shapes ---
    "SymmetricTensor.to_dense": "the dense boundary itself; reading the tensor apart is the job",
    "SymmetricTensor.from_dense": "a dense array arrives from outside and is checked into blocks",
    "SymmetricTensor.from_blocks": "the caller names blocks; they are gathered once, here",
    "SymmetricTensor.with_blocks": "the caller replaces named blocks, so the rest is cut out",
    "SymmetricTensor.set_params": "quimb's leaf protocol: the caller hands back per-block arrays",
    "SymmetricTensor.items": "the interop walk -- reading a tensor apart is what it is for",
    "SymmetricTensor.block": "one named block, by key: the single-block spelling of ``items``",
    "SymmetricTensor.save": "the archive's format is the blocks",
    "tenet.save": "the free function behind ``SymmetricTensor.save``",
    "tenet.apply_blocks": "the caller supplies ``fn``, so the shapes it may read are not ours",
    "SymmetricTensor.apply_blocks": "the method behind ``tenet.apply_blocks``",
    "tenet.zip_blocks": "``apply_blocks``' two-operand sibling, a caller's ``fn`` the same way",
    "tenet.to_symmetry": "defined in the dense basis: it is ``to_dense`` then ``from_dense``",
    "SymmetricTensor.to_symmetry": "the method behind ``tenet.to_symmetry``",
    # --- not operations on a tensor: there is no operand to cut ---
    "tenet.map_layout": "a query on a structure; no tensor is involved",
    "tenet.fusion_trees": "a fusion-rules query on sector labels; no tensor is involved",
    "tenet.coupled_sectors": "a fusion-rules query on sector labels; no tensor is involved",
    "tenet.enable_jax": "a process-wide pytree registration, not an operation on a tensor",
}

OPEN = {
    "tenet.fuse": (
        "merges bands into one, so the target's rows are a reordering of the source's "
        "that no plan states; the derivation is not in this branch"
    ),
    "SymmetricTensor.fuse": "the method behind ``tenet.fuse``",
    "tenet.unfuse": "``fuse`` run backwards, and open for the same reason",
    "SymmetricTensor.unfuse": "the method behind ``tenet.unfuse``",
    "tenet.direct_sum": (
        "puts one operand in the leading degeneracy slots and the other in the trailing "
        "ones; ``embed``'s index map covers the leading half only"
    ),
    "SymmetricTensor.direct_sum": "the method behind ``tenet.direct_sum``",
}


@functools.cache
def _material(legs):
    """The arrays every case below is built from, as ``(structure, matrices)`` pairs.

    Kept as matrices rather than as tensors because a *tensor remembers a cut*: an
    operand that some construction step had already read as blocks would answer the
    measurement out of that memo, and the count would come back zero for the wrong
    reason. Rehydrating through ``from_matrices`` is cheap and gives a tensor that has
    never been cut, which is also what every intermediate of a real chain is.
    """
    space = legs[0].space
    sectors = space.sectors
    smaller = GradedSpace.new(
        space.provider,
        tuple((a, max(1, m - 1)) for a, m in sectors[1:]) if len(sectors) > 1 else sectors,
    )
    small_legs = tuple(dataclasses.replace(leg, space=smaller) for leg in legs)
    square = (Leg(space, OUT), Leg(space, IN))
    archive = pathlib.Path(tempfile.mkdtemp()) / "operand.npz"
    tenet.save(SymmetricTensor.random(legs, seed=41), archive)

    def pair(t):
        return t.structure, to_matrices(t)

    return {
        "legs": legs,
        "small_legs": small_legs,
        "archive": archive,
        "a": pair(SymmetricTensor.random(legs, seed=41)),
        "b": pair(SymmetricTensor.random(legs, seed=42)),
        "sq": pair(SymmetricTensor.random(square, seed=43)),
        "small": pair(SymmetricTensor.random(small_legs, seed=44)),
        "big": pair(tenet.embed(SymmetricTensor.random(small_legs, seed=45), legs)),
        "positive": pair(
            tenet.apply_blocks(tenet.linalg.eigh(SymmetricTensor.random(square, seed=46))[0], abs)
        ),
        "values": pair(
            tenet.linalg.svd(SymmetricTensor.random(legs, seed=47), ((0, 1), (2, 3)))[1]
        ),
        "fused": pair(tenet.fuse(SymmetricTensor.random(legs, seed=48), (0, 1))),
    }


def _operations(legs):
    """One thunk per public callable that is not excluded, on freshly uncut operands."""
    m = _material(legs)
    a, b, sq = (from_matrices(*m[k]) for k in ("a", "b", "sq"))
    small, big = from_matrices(*m["small"]), from_matrices(*m["big"])
    positive, values = from_matrices(*m["positive"]), from_matrices(*m["values"])
    fused = from_matrices(*m["fused"])
    legs, small_legs, archive = m["legs"], m["small_legs"], m["archive"]
    out2, in2 = legs[:2], legs[2:]
    lin = tenet.linalg
    return {
        # contraction, and the scalars that leave the tensor world
        "tenet.tensordot": lambda: tenet.tensordot(a, b, ((2, 3), (0, 1))),
        "tenet.einsum": lambda: tenet.einsum("abcd,cdef->abef", a, b),
        "tenet.einsum_chain": lambda: tenet.einsum_chain([("abcd,cdef->abef", a, b, "")]),
        "tenet.compose": lambda: tenet.compose(a, b),
        "tenet.trace": lambda: tenet.trace(a, (0, 2)),
        "tenet.full_trace": lambda: tenet.full_trace(tenet.tensordot(a, b, ((2, 3), (0, 1)))),
        "tenet.inner": lambda: tenet.inner(a, b),
        "tenet.norm": lambda: tenet.norm(a),
        "SymmetricTensor.norm": lambda: a.norm(),
        # the diagram moves
        "tenet.transpose": lambda: tenet.transpose(a, (1, 0, 3, 2)),
        "SymmetricTensor.transpose": lambda: a.transpose(1, 0, 3, 2),
        "tenet.repartition": lambda: tenet.repartition(a, (0,), (1, 2, 3)),
        "SymmetricTensor.repartition": lambda: a.repartition((0,), (1, 2, 3)),
        "tenet.bend": lambda: tenet.bend(a, 1),
        "tenet.braid": lambda: tenet.braid(a, (1, 0, 3, 2), (0, 1, 2, 3)),
        "tenet.twist": lambda: tenet.twist(a, (0, 3)),
        "tenet.flip_dual": lambda: tenet.flip_dual(a, (0, 1, 2, 3)),
        "tenet.adjoint": lambda: tenet.adjoint(a),
        "SymmetricTensor.adjoint": lambda: a.adjoint(),
        "tenet.conj": lambda: tenet.conj(a),
        "SymmetricTensor.conj": lambda: a.conj(),
        "tenet.as_map": lambda: tenet.as_map(a),
        "SymmetricTensor.as_map": lambda: a.as_map(),
        # arithmetic
        "tenet.add": lambda: tenet.add(a, b),
        "tenet.subtract": lambda: tenet.subtract(a, b),
        "tenet.multiply": lambda: tenet.multiply(a, 2.0),
        "tenet.divide": lambda: tenet.divide(a, 2.0),
        "tenet.negative": lambda: tenet.negative(a),
        "tenet.allclose": lambda: tenet.allclose(a, b),
        # blockwise maps whose ``fn`` is the library's own, and the diagonal
        "tenet.block_sqrt": lambda: tenet.block_sqrt(positive),
        "SymmetricTensor.block_sqrt": lambda: positive.block_sqrt(),
        "tenet.block_power": lambda: tenet.block_power(positive, 0.5),
        "SymmetricTensor.block_power": lambda: positive.block_power(0.5),
        "tenet.map_diagonal": lambda: tenet.map_diagonal(a),
        # graded-space plumbing
        "tenet.embed": lambda: tenet.embed(small, legs),
        "SymmetricTensor.embed": lambda: small.embed(legs),
        "tenet.restrict": lambda: tenet.restrict(big, small_legs),
        "SymmetricTensor.restrict": lambda: big.restrict(small_legs),
        # constructors and interop that hold no operand to cut
        "tenet.identity": lambda: tenet.identity((legs[0],)),
        "tenet.isometry": lambda: tenet.isometry(out2, in2),
        "tenet.random_isometry": lambda: tenet.random_isometry(out2, in2, seed=1),
        "tenet.from_matrices": lambda: from_matrices(*m["a"]),
        "tenet.to_matrices": lambda: to_matrices(a),
        "tenet.load": lambda: tenet.load(archive),
        "SymmetricTensor.load": lambda: SymmetricTensor.load(archive),
        "SymmetricTensor.random": lambda: SymmetricTensor.random(legs, seed=1),
        "SymmetricTensor.zeros": lambda: SymmetricTensor.zeros(legs),
        "SymmetricTensor.from_data": lambda: SymmetricTensor.from_data(a.structure, a.data),
        "SymmetricTensor.astype": lambda: a.astype("complex128"),
        "SymmetricTensor.copy": lambda: a.copy(),
        "SymmetricTensor.to_backend": lambda: a.to_backend("numpy"),
        "SymmetricTensor.get_params": lambda: a.get_params(),
        # linalg
        "linalg.svd": lambda: lin.svd(a, ((0, 1), (2, 3))),
        "linalg.qr": lambda: lin.qr(a, ((0, 1), (2, 3))),
        "linalg.lq": lambda: lin.lq(a, ((0, 1), (2, 3))),
        "linalg.polar": lambda: lin.polar(a, ((0, 1), (2, 3))),
        "linalg.eigh": lambda: lin.eigh(sq),
        "linalg.eig": lambda: lin.eig(sq),
        "linalg.eigvals": lambda: lin.eigvals(sq),
        "linalg.expm": lambda: lin.expm(sq, alpha=0.5),
        "linalg.left_null": lambda: lin.left_null(a, ((0, 1, 2), (3,))),
        "linalg.right_null": lambda: lin.right_null(a, ((0,), (1, 2, 3))),
        "linalg.select_bond": lambda: lin.select_bond(values, cutoff=1e-12),
        "linalg.svd_truncated": lambda: lin.svd_truncated(a, ((0, 1), (2, 3)), cutoff=1e-12),
        "linalg.eigh_truncated": lambda: lin.eigh_truncated(sq, cutoff=1e-12),
        # the open violations, measured so that the list above cannot quietly rot
        "tenet.fuse": lambda: tenet.fuse(a, (0, 1)),
        "SymmetricTensor.fuse": lambda: a.fuse((0, 1)),
        "tenet.unfuse": lambda: tenet.unfuse(fused, 0, out2),
        "SymmetricTensor.unfuse": lambda: fused.unfuse(0, out2),
        "tenet.direct_sum": lambda: tenet.direct_sum(a, b, 0),
        "SymmetricTensor.direct_sum": lambda: a.direct_sum(b, 0),
    }


def test_the_classification_covers_the_public_surface_exactly():
    """No public callable is unclassified, and none is classified twice.

    This is what stops the storage contract from being re-discovered one profile at a
    time. A new operation lands in one of three places or this fails: measured below at
    zero cuts, named in ``EXCLUDED`` with the reason it is allowed to read blocks, or
    named in ``OPEN`` with the reason it still does. There is no fourth option and no
    default, so "nobody thought to add it to the list" is not reachable.
    """
    surface = set(_public_surface())
    measured = set(_operations(EVERY_SYMMETRY[2].values[0]))
    assert measured.isdisjoint(EXCLUDED)
    assert set(OPEN) <= measured
    missing = surface - measured - set(EXCLUDED)
    assert not missing, f"unclassified public callables: {sorted(missing)}"
    stale = (measured | set(EXCLUDED)) - surface
    assert not stale, f"classified but no longer public: {sorted(stale)}"


@pytest.mark.parametrize("legs", EVERY_SYMMETRY)
def test_the_public_surface_never_cuts_a_block(legs):
    """The storage contract over the *whole* surface, on every symmetry. Still zero.

    The eleven hot operations above were a sample, and a sample is how the contract kept
    being broken somewhere else: each violation surfaced on its own out of a profile,
    which is the wrong instrument for a rule that is supposed to hold everywhere. This
    walks the enumeration instead.

    Every operand is rehydrated from matrices immediately before it is measured, because
    the cut is memoized: an operand that something else had already read as blocks would
    report zero without the operation having been matrix-native at all. That is how the
    count under-reported before, and it is why the fixtures are stored as matrices rather
    than as tensors.
    """
    for name in _operations(legs):
        if name in OPEN:
            continue
        _operations(legs)[name]()  # warm every plan and layout cache; a cold build is not it
        reads = _blocks_reads(_operations(legs)[name])
        assert reads == 0, f"{name} cut its operand into blocks {reads} time(s)"


@pytest.mark.parametrize("legs", EVERY_SYMMETRY)
def test_the_open_violations_are_still_open(legs):
    """``OPEN`` is held to being true, so an exemption cannot outlive what earned it.

    An entry that has quietly become matrix-native fails here, which is the prompt to
    move it into the measured set. An exemption nobody is made to re-earn is how a
    known-bad list becomes a permanent one.
    """
    for name in OPEN:
        _operations(legs)[name]()
        assert _blocks_reads(_operations(legs)[name]) > 0, f"{name} no longer cuts: move it up"
