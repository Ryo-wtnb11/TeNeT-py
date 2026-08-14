"""Tests for ``tenet.embed`` — issue #83.

The discriminating criterion here is the dense oracle. ``embed`` places source
data in the **leading degeneracy slots**, and the dense layout's within-slab
index is ``alpha * d_a + m``, so for SU(2) the embedded data is *strided* in the
dense array rather than a leading corner of it — and every later sector's slab
offset moves too. Every dense assertion below is therefore written through
per-axis ``(alpha, m)`` index maps, never through a dense prefix, which is the
one detail a U(1)-only test would pass while being wrong.

The projection back down (the adjoint of ``embed``, deliberately not shipped —
see the ``ponytail:`` note in :mod:`tenet.ops.embed`) is performed by hand in
:func:`project`, which doubles as the reference implementation for whoever opens
that follow-up.
"""

import re
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import (
    SU2,
    U1,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

# --- the cases -------------------------------------------------------------------
# Three legs (OUT, OUT, IN) per provider, in three flavours: the source, a target
# that only grows existing degeneracies, and a target that additionally carries a
# sector the source never had. Trivial has exactly one sector, so it has no
# new-sector flavour and says so with ``None``.

TS = TrivialSector()
UF = ProductProvider((U1, fZ2))


def uf(charge: int, parity: int) -> ProductSector:
    return ProductSector((U1Sector(charge), FZ2Sector(parity)))


def spaces(provider, *mappings):
    return tuple(GradedSpace.new(provider, m) for m in mappings)


CASES = {
    "trivial": (
        spaces(Trivial, {TS: 2}, {TS: 3}, {TS: 2}),
        spaces(Trivial, {TS: 3}, {TS: 4}, {TS: 3}),
        None,
    ),
    "u1": (
        spaces(
            U1,
            {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1},
            {U1Sector(0): 2, U1Sector(1): 3},
            {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1},
        ),
        spaces(
            U1,
            {U1Sector(-1): 3, U1Sector(0): 4, U1Sector(1): 2},
            {U1Sector(0): 3, U1Sector(1): 4},
            {U1Sector(-1): 3, U1Sector(0): 4, U1Sector(1): 2},
        ),
        spaces(
            U1,
            {U1Sector(-1): 3, U1Sector(0): 4, U1Sector(1): 2, U1Sector(2): 2},
            {U1Sector(-1): 1, U1Sector(0): 3, U1Sector(1): 4},
            {U1Sector(-1): 3, U1Sector(0): 4, U1Sector(1): 2, U1Sector(2): 2},
        ),
    ),
    # the issue's own SU(2) structure: {j=0: 2, j=1/2: 1} grown to {2, 1} -> {3, 2, 1}
    "su2": (
        spaces(
            SU2,
            {SU2Sector(0): 2, SU2Sector(1): 1},
            {SU2Sector(0): 2, SU2Sector(1): 1},
            {SU2Sector(0): 2, SU2Sector(1): 1},
        ),
        spaces(
            SU2,
            {SU2Sector(0): 3, SU2Sector(1): 2},
            {SU2Sector(0): 3, SU2Sector(1): 2},
            {SU2Sector(0): 3, SU2Sector(1): 2},
        ),
        spaces(
            SU2,
            {SU2Sector(0): 3, SU2Sector(1): 2, SU2Sector(2): 1},
            {SU2Sector(0): 3, SU2Sector(1): 2, SU2Sector(2): 1},
            {SU2Sector(0): 3, SU2Sector(1): 2, SU2Sector(2): 1},
        ),
    ),
    # axis 1 starts purely even, so fZ2 gets a genuine new-sector target too
    "fz2": (
        spaces(
            fZ2,
            {FZ2Sector(0): 2, FZ2Sector(1): 3},
            {FZ2Sector(0): 3},
            {FZ2Sector(0): 2, FZ2Sector(1): 3},
        ),
        spaces(
            fZ2,
            {FZ2Sector(0): 3, FZ2Sector(1): 4},
            {FZ2Sector(0): 4},
            {FZ2Sector(0): 3, FZ2Sector(1): 4},
        ),
        spaces(
            fZ2,
            {FZ2Sector(0): 3, FZ2Sector(1): 4},
            {FZ2Sector(0): 4, FZ2Sector(1): 2},
            {FZ2Sector(0): 3, FZ2Sector(1): 4},
        ),
    ),
    "product": (
        spaces(
            UF, {uf(0, 0): 2, uf(1, 1): 3}, {uf(0, 0): 2, uf(1, 1): 1}, {uf(0, 0): 2, uf(1, 1): 3}
        ),
        spaces(
            UF, {uf(0, 0): 3, uf(1, 1): 4}, {uf(0, 0): 3, uf(1, 1): 2}, {uf(0, 0): 3, uf(1, 1): 4}
        ),
        spaces(
            UF,
            {uf(0, 0): 3, uf(1, 1): 4, uf(2, 0): 2},
            {uf(0, 0): 3, uf(1, 1): 2},
            {uf(0, 0): 3, uf(1, 1): 4, uf(2, 0): 2},
        ),
    ),
}

SIDES = (OUT, OUT, IN)
PAIRS = [
    (name, flavour)
    for name, case in CASES.items()
    for flavour in ("grow", "new")
    if case[{"grow": 1, "new": 2}[flavour]] is not None
]


def legs(spaces_, names=("a", "b", "c")):
    return tuple(Leg(s, side, name=n) for s, side, n in zip(spaces_, SIDES, names, strict=True))


def fixture(name, flavour, *, seed=0, dtype=np.float64):
    """``(source tensor, target legs)`` for one provider and one target flavour."""
    src, grow, new = CASES[name]
    source = legs(src)
    return SymmetricTensor.random(source, seed=seed, dtype=dtype), legs(
        grow if flavour == "grow" else new
    )


def project(embedded, source):
    """The restriction, by hand: the leading degeneracy slots of each source key.

    Five lines, and the reference implementation for a future ``restrict`` —
    which is not shipped because its refusal semantics (what happens to non-zero
    data in a dropped slot) have no caller to settle them.
    """
    return tuple(
        embedded.block(key)[tuple(slice(0, m) for m in source.structure.block_shape(key))]
        for key in source.structure.block_order
    )


def index_maps(source_legs, target_legs):
    """Per-axis dense index maps ``[offset(a) + alpha * d_a + m]``, source order.

    Written through the ``(alpha, m)`` decomposition on purpose: with ``d_a > 1``
    the embedded data is strided in the dense index, not a prefix of it.
    """
    maps = []
    for sleg, tleg in zip(source_legs, target_legs, strict=True):
        d = sleg.provider.irrep_dim
        maps.append(
            np.array(
                [
                    tleg.space.sector_offset(a) + alpha * d(a) + m
                    for a in sleg.space
                    for alpha in range(sleg.space.degeneracy(a))
                    for m in range(d(a))
                ]
            )
        )
    return maps


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  registration is the import's side effect

    return jax


# --- exports ----------------------------------------------------------------------


def test_exported_from_tenet_and_ops_and_as_a_method():
    assert "embed" in tenet.__all__ and "embed" in tenet.ops.__all__
    assert tenet.embed is tenet.ops.embed
    t, big = fixture("su2", "new")
    assert t.embed(big) == tenet.embed(t, big)


# --- the fact the loop rests on ---------------------------------------------------


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_source_keys_are_a_subset_of_target_keys(name, flavour):
    t, big = fixture(name, flavour)
    assert set(t.structure.block_order) <= set(tenet.embed(t, big).structure.block_order)


# --- round trip, zeros, norm ------------------------------------------------------


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_hand_projection_round_trip_is_exact(name, flavour):
    t, big = fixture(name, flavour)
    for got, want in zip(project(tenet.embed(t, big), t), t.blocks, strict=True):
        assert np.abs(np.asarray(got) - np.asarray(want)).max() == 0.0


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_new_and_padded_slots_are_exactly_zero(name, flavour):
    t, big = fixture(name, flavour)
    e = tenet.embed(t, big)
    source_keys = set(t.structure.block_order)
    for key, block in e.items():
        block = np.asarray(block)
        if key not in source_keys:
            assert np.abs(block).max() == 0.0
            continue
        # zero the source's own corner; whatever padding was added must be zero
        rest = block.copy()
        rest[tuple(slice(0, m) for m in t.structure.block_shape(key))] = 0.0
        assert np.abs(rest).max() == 0.0


def masses(x):
    """Every block entry, magnitude-sorted, zeros dropped. Order-independent."""
    v = np.abs(np.concatenate([np.asarray(b).ravel() for b in x.blocks]))
    return np.sort(v[v != 0])


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_norm_is_preserved(name, flavour):
    t, big = fixture(name, flavour)
    e = tenet.embed(t, big)
    # The bit-exact half, and the one that is about `embed`: the multiset of
    # non-zero entries is *identical*. An implementation that normalized, scaled
    # or randomized the new slots (quimb's `expand_bond_dimension` has a
    # `rand_strength` knob for exactly that) fails here immediately.
    assert np.array_equal(masses(e), masses(t))
    # The norm itself is equal to the last few ulps rather than bit-identical,
    # and the reason is NumPy, not embed: `ar.do("sum", ...)` sums pairwise, so
    # padding an array with exact zeros re-groups the additions
    # (`(a**2).sum() != (np.pad(a, 1)**2).sum()` for a plain random array).
    # Non-trivial for SU(2) all the same: the qdim weight makes the sum a
    # genuinely weighted one, and a lost or duplicated weight is far larger
    # than this.
    assert float(tenet.norm(e)) == pytest.approx(float(tenet.norm(t)), rel=1e-15, abs=0.0)


# --- the dense oracle, through (alpha, m) -----------------------------------------


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_dense_image_is_the_source_scattered_through_the_index_maps(name, flavour):
    t, big = fixture(name, flavour)
    dense = tenet.embed(t, big).to_dense()
    maps = index_maps(t.legs, big)
    assert np.abs(dense[np.ix_(*maps)] - t.to_dense()).max() == 0.0
    # nothing else is non-zero
    assert abs(np.abs(dense).sum() - np.abs(t.to_dense()).sum()) < 1e-14


# --- composition ------------------------------------------------------------------


def compose_fixture(name):
    """``(t, u, grown legs for each, space -> grown space)`` sharing one leg."""
    src, grow, _ = CASES[name]
    mapping = dict(zip(src, grow, strict=True))
    t = SymmetricTensor.random(legs(src), seed=3)
    u = SymmetricTensor.random((Leg(src[2], OUT, name="c"), Leg(src[1], IN, name="d")), seed=4)
    return t, u, mapping


def grow_legs(t, mapping):
    return tuple(replace(leg, space=mapping[leg.space]) for leg in t.legs)


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_compose_of_embedded_equals_embedded_compose(name):
    t, u, mapping = compose_fixture(name)
    composed = t @ u
    assert tenet.allclose(
        tenet.embed(t, grow_legs(t, mapping)) @ tenet.embed(u, grow_legs(u, mapping)),
        tenet.embed(composed, grow_legs(composed, mapping)),
    )


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_tensordot_of_embedded_equals_embedded_tensordot(name):
    t, u, mapping = compose_fixture(name)
    contracted = tenet.tensordot(t, u, axes=([2], [0]))
    got = tenet.tensordot(
        tenet.embed(t, grow_legs(t, mapping)),
        tenet.embed(u, grow_legs(u, mapping)),
        axes=([2], [0]),
    )
    assert tenet.allclose(got, tenet.embed(contracted, grow_legs(contracted, mapping)))
    # and the dense image of the contraction is the original's, scattered
    dense = got.to_dense()
    maps = index_maps(contracted.legs, got.legs)
    assert np.abs(dense[np.ix_(*maps)] - contracted.to_dense()).max() < 1e-12


# --- structural refusals ----------------------------------------------------------


def su2_legs(spaces_=None):
    return legs(spaces_ or CASES["su2"][0])


def test_refuses_a_smaller_degeneracy_naming_axis_sector_and_both_numbers():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    shrunk = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 1})
    target = (Leg(shrunk, OUT), *t.legs[1:])
    with pytest.raises(ValueError, match=r"axis 0 sector .* degeneracy 2 .* but 1 in the target"):
        tenet.embed(t, target)


def test_refuses_a_sector_absent_from_the_target():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    dropped = GradedSpace.new(SU2, {SU2Sector(1): 4})
    target = (Leg(dropped, OUT), *t.legs[1:])
    with pytest.raises(ValueError, match="absent from the target space entirely"):
        tenet.embed(t, target)


def test_refuses_a_side_mismatch():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    target = (*t.legs[:2], replace(t.legs[2], side=OUT))
    with pytest.raises(ValueError, match=re.escape("axis 2 has side 'in'")) as exc:
        tenet.embed(t, target)
    assert "repartition" in str(exc.value)


def test_refuses_a_dual_mismatch():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    target = (replace(t.legs[0], dual=True), *t.legs[1:])
    with pytest.raises(ValueError, match="axis 0 has dual=False.*target leg has dual=True"):
        tenet.embed(t, target)


def test_refuses_a_provider_mismatch():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    q = GradedSpace.new(U1, {U1Sector(0): 4})
    with pytest.raises(ValueError, match="axis 0 has provider SU2.*target leg has U1"):
        tenet.embed(t, (Leg(q, OUT), Leg(q, OUT), Leg(q, IN)))


def test_refuses_a_leg_count_mismatch():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    with pytest.raises(ValueError, match="target has 2 legs, but the tensor has 3"):
        tenet.embed(t, t.legs[:2])


def test_refusal_is_identical_inside_jit():
    """The check is metadata against metadata, so it fires at trace time too."""
    jax = use_jax()
    t = SymmetricTensor.random(su2_legs(), seed=0).to_backend("jax")
    dropped = (Leg(GradedSpace.new(SU2, {SU2Sector(1): 4}), OUT), *t.legs[1:])

    @partial(jax.jit, static_argnums=1)
    def f(x, target):
        return tenet.norm(tenet.embed(x, target))

    with pytest.raises(ValueError, match="absent from the target space entirely"):
        f(t, dropped)


# --- name, identity ---------------------------------------------------------------


def test_differing_names_are_accepted_and_the_target_wins():
    """``name`` is user bookkeeping, exactly the stance ``ProductSpace.matches`` takes."""
    t, big = fixture("su2", "new")
    renamed = tuple(leg.renamed(f"bond-{i}") for i, leg in enumerate(big))
    e = tenet.embed(t, renamed)
    assert tuple(leg.name for leg in e.legs) == ("bond-0", "bond-1", "bond-2")
    for got, want in zip(project(e, t), t.blocks, strict=True):
        assert np.abs(got - want).max() == 0.0


@pytest.mark.parametrize("name", list(CASES))
def test_identity_embedding_returns_the_same_block_objects(name):
    t = SymmetricTensor.random(legs(CASES[name][0]), seed=7)
    e = tenet.embed(t, t.legs)
    assert e == t
    assert all(a is b for a, b in zip(e.blocks, t.blocks, strict=True))


# --- traceability -----------------------------------------------------------------
# `embed` changes the TensorStructure but is *traceable*, and that is not in tension
# with #64's StructureChangingError: what #64 refuses is a structure decided from
# block *values*. Here the target comes from `legs`, static metadata the caller
# chose outside the trace — the exact argument #77 makes for `svd(..., bond=B)`.


def test_jit_traces_once_per_structure_and_retraces_when_legs_change():
    jax = use_jax()
    t, big = fixture("su2", "new")
    t = t.to_backend("jax")
    other, _ = fixture("su2", "new", seed=11)
    grow = legs(CASES["su2"][1])
    traces = []

    @partial(jax.jit, static_argnums=1)
    def f(x, target):
        traces.append(target)
        return tenet.norm(tenet.embed(x, target))

    a = f(t, big)
    b = f(other.to_backend("jax"), big)
    assert len(traces) == 1  # different block values, same structure: no retrace
    assert not np.isclose(float(a), float(b))
    f(t, grow)
    assert len(traces) == 2  # different legs: retrace


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_gradient_matches_central_finite_differences(name):
    jax = use_jax()
    t, big = fixture(name, "new", seed=2)
    w = SymmetricTensor.random(big, seed=5)

    def scalar(x):
        # depends on *where* the data lands, not only on its norm
        return tenet.norm(tenet.embed(x, big) + w)

    g = jax.grad(scalar)(t.to_backend("jax"))
    assert [b.shape for b in g.blocks] == [b.shape for b in t.blocks]

    h = 1e-6
    for i, block in enumerate(t.blocks):
        for idx in np.ndindex(block.shape):

            def shifted(delta, i=i, idx=idx):
                blocks = list(np.array(b, copy=True) for b in t.blocks)
                blocks[i][idx] += delta
                return float(scalar(t.set_params(blocks)))

            fd = (shifted(h) - shifted(-h)) / (2 * h)
            assert abs(fd - float(np.asarray(g.blocks[i])[idx])) < 1e-6


# --- dtypes and backends ----------------------------------------------------------


def test_complex_blocks_keep_their_dtype_and_pad_with_exact_zeros():
    t, big = fixture("su2", "new")
    t = SymmetricTensor(t.structure, tuple((b * (1 + 2j)).astype(np.complex128) for b in t.blocks))
    e = tenet.embed(t, big)
    assert e.dtype == np.complex128
    for got, want in zip(project(e, t), t.blocks, strict=True):
        assert np.abs(got - want).max() == 0.0
    source_keys = set(t.structure.block_order)
    for key, block in e.items():
        rest = np.asarray(block).copy()
        if key in source_keys:
            rest[tuple(slice(0, m) for m in t.structure.block_shape(key))] = 0
        assert np.array_equal(rest, np.zeros_like(rest))


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_jax_backed_blocks_stay_jax_and_satisfy_the_same_criteria(name, flavour):
    use_jax()
    import autoray as ar

    t, big = fixture(name, flavour)
    e = tenet.embed(t.to_backend("jax"), big)
    assert {ar.infer_backend(b) for b in e.blocks} == {"jax"}
    assert float(tenet.norm(e)) == pytest.approx(float(tenet.norm(t)), rel=1e-15, abs=0.0)
    for got, want in zip(project(e, t), t.blocks, strict=True):
        assert np.abs(np.asarray(got) - want).max() == 0.0


# --- what is deliberately absent --------------------------------------------------


def test_no_plan_object_no_cache_and_the_decision_is_recorded():
    import pathlib
    import sys

    module = sys.modules["tenet.ops.embed"]

    assert not any(n.lower().endswith("plan") for n in vars(module))
    source = pathlib.Path(module.__file__).read_text()
    assert "functools" not in source and "@cache" not in source
    assert "ponytail: no plan object and no cache" in source
