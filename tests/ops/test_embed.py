"""Tests for ``tenet.embed`` (#83), ``tenet.restrict`` (#90) and ``direct_sum`` (#91).

The discriminating criterion here is the dense oracle. ``embed`` places source
data in the **leading degeneracy slots**, and the dense layout's within-slab
index is ``alpha * d_a + m``, so for SU(2) the embedded data is *strided* in the
dense array rather than a leading corner of it — and every later sector's slab
offset moves too. Every dense assertion below is therefore written through
per-axis ``(alpha, m)`` index maps, never through a dense prefix, which is the
one detail a U(1)-only test would pass while being wrong.

The projection back down is :func:`tenet.restrict`, and it is *never* checked
against itself here: :func:`project` performs it by hand and stays the
independent oracle every ``restrict`` assertion is written against.
"""

import math
import re
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
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

    Five lines, and the reference implementation :func:`tenet.restrict` (#90) was
    promoted from. It stays here as the *independent* oracle: every ``restrict``
    assertion below is written against this, never against ``restrict`` itself.
    ``restrict`` adds to it exactly what those five lines cannot decide — the
    refusal of non-zero data in a dropped slot.
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
                return float(scalar(SymmetricTensor(t.structure, blocks)))

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
    assert "Simplification: no plan object and no cache" in source
    # #90 discharged the "no restrict" note; #91 added its two.
    assert "Simplification: no `restrict`" not in source
    assert "Simplification: `_check_containment` is shared" in source
    assert "Simplification: `axes` as a set" in source
    assert "Simplification: pairwise only" in source


# ==================================================================================
# restrict (#90) — embed's adjoint
# ==================================================================================


def qdim_inner(x, y):
    """``⟨x, y⟩`` in the qdim-weighted inner product ``tenet.norm`` is the norm of."""
    return sum(
        x.provider.qdim(key.coupled) * float(np.sum(np.asarray(a) * np.asarray(b)))
        for (key, a), b in zip(x.items(), y.blocks, strict=True)
    )


def discarded_mass(source, target_structure):
    """``‖what restrict throws away‖``, by hand — the oracle for Pythagoras.

    Whole keys the target does not carry contribute all of their mass; kept keys
    contribute whatever sits outside the leading slots. Same ``qdim`` weight as
    :func:`tenet.norm`, which is what makes the identity exact rather than close.
    """
    kept = set(target_structure.block_order)
    total = 0.0
    for key, block in source.items():
        rest = np.asarray(block).copy()
        if key in kept:
            rest[tuple(slice(0, m) for m in target_structure.block_shape(key))] = 0
        total += source.provider.qdim(key.coupled) * float(np.sum(np.abs(rest) ** 2))
    return math.sqrt(total)


def worst_key(source, target_structure):
    """The key carrying the largest discarded mass — what the refusal must name."""
    kept = set(target_structure.block_order)
    scored = []
    for key, block in source.items():
        rest = np.asarray(block).copy()
        if key in kept:
            rest[tuple(slice(0, m) for m in target_structure.block_shape(key))] = 0
        scored.append((source.provider.qdim(key.coupled) * float(np.sum(np.abs(rest) ** 2)), key))
    return max(scored, key=lambda p: p[0])[1]


# --- exports ----------------------------------------------------------------------


def test_restrict_exported_from_tenet_and_ops_and_as_a_method():
    assert "restrict" in tenet.__all__ and "restrict" in tenet.ops.__all__
    assert tenet.restrict is tenet.ops.restrict
    t, big = fixture("su2", "new")
    e = tenet.embed(t, big)
    assert e.restrict(t.legs) == tenet.restrict(e, t.legs)


# --- the fact the loop rests on ---------------------------------------------------


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_restrict_target_keys_are_a_subset_of_source_keys(name, flavour):
    t, big = fixture(name, flavour)
    u = SymmetricTensor.random(big, seed=19)
    r = tenet.restrict(u, t.legs, atol=math.inf)
    assert set(r.structure.block_order) <= set(u.structure.block_order)


# --- round trips ------------------------------------------------------------------


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_restrict_undoes_embed_exactly_and_the_default_check_passes(name, flavour):
    """``restrict(embed(t, L), t.legs) == t`` at 0.0, residual exactly 0.0.

    Exact because ``embed`` writes zeros into precisely the slots ``restrict``
    drops — so the default (checking) form is the one called here.
    """
    t, big = fixture(name, flavour)
    e = tenet.embed(t, big)
    r = tenet.restrict(e, t.legs)
    assert r.structure == t.structure
    assert discarded_mass(e, t.structure) == 0.0
    for got, want, hand in zip(r.blocks, t.blocks, project(e, t), strict=True):
        assert np.abs(np.asarray(got) - np.asarray(want)).max() == 0.0
        assert np.abs(np.asarray(got) - np.asarray(hand)).max() == 0.0


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_embed_after_restrict_is_an_idempotent_orthogonal_projection(name):
    """``P = embed ∘ restrict``: ``P² = P``, and orthogonality *is* adjointness."""
    t, big = fixture(name, "new")
    u = SymmetricTensor.random(big, seed=13)

    def P(x):
        return tenet.embed(tenet.restrict(x, t.legs, atol=math.inf), big)

    once = P(u)
    for got, want in zip(P(once).blocks, once.blocks, strict=True):
        assert np.abs(got - want).max() == 0.0

    a = SymmetricTensor.random(t.legs, seed=14)
    b = SymmetricTensor.random(big, seed=15)
    assert qdim_inner(tenet.embed(a, big), b) == pytest.approx(
        qdim_inner(a, tenet.restrict(b, t.legs, atol=math.inf)), rel=0.0, abs=1e-14
    )


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_pythagoras_is_exact(name, flavour):
    t, big = fixture(name, flavour)
    u = SymmetricTensor.random(big, seed=17)
    r = tenet.restrict(u, t.legs, atol=math.inf)
    residual = discarded_mass(u, t.structure)
    assert float(tenet.norm(u)) ** 2 == pytest.approx(
        float(tenet.norm(r)) ** 2 + residual**2, rel=1e-14
    )


# --- the refusal ------------------------------------------------------------------


def test_refuses_non_zero_data_in_a_dropped_slot_naming_residual_atol_and_worst_key():
    t, big = fixture("su2", "grow")  # same sectors, larger degeneracies: slots only
    u = SymmetricTensor.random(big, seed=18)
    with pytest.raises(ValueError, match="restrict: discarded data") as exc:
        tenet.restrict(u, t.legs)
    message = str(exc.value)
    assert f"residual {discarded_mass(u, t.structure):.6g}" in message
    assert "atol" in message and "tenet.PROJECT" in message and "math.inf" in message
    assert str(worst_key(u, t.structure)) in message


def test_refuses_a_whole_dropped_sector():
    t, big = fixture("su2", "new")  # `big` carries a sector `t` never had
    u = SymmetricTensor.random(big, seed=20)
    dropped = {k for k in u.structure.block_order if k not in set(t.structure.block_order)}
    assert dropped  # the branch being exercised is live
    with pytest.raises(ValueError, match="restrict: discarded data"):
        tenet.restrict(u, t.legs)


def test_a_discarded_mass_below_atol_is_accepted_silently():
    t, big = fixture("su2", "new")
    noisy = tenet.embed(t, big) + SymmetricTensor.random(big, seed=21) * 1e-13
    r = tenet.restrict(noisy, t.legs)  # no raise
    assert discarded_mass(noisy, t.structure) > 0.0
    for got, hand in zip(r.blocks, project(noisy, t), strict=True):
        assert np.abs(np.asarray(got) - np.asarray(hand)).max() == 0.0


def test_the_default_atol_is_relative_and_the_same_fraction_is_refused_at_any_scale():
    t, big = fixture("su2", "new")
    e = tenet.embed(t, big)
    bad = e + SymmetricTensor.random(big, seed=22) * 1e-3
    good = e + SymmetricTensor.random(big, seed=22) * 1e-13
    for scale in (1.0, 1e6):
        with pytest.raises(ValueError, match="restrict: discarded data"):
            tenet.restrict(bad * scale, t.legs)
        tenet.restrict(good * scale, t.legs)  # accepted at both scales


# --- structural refusals ----------------------------------------------------------
# The mirror of `embed`'s, through the same `_check_containment` with its arguments
# swapped: the messages say `restrict:` and "the tensor" is now the *larger* side.


def test_restrict_refuses_a_larger_target_degeneracy_naming_axis_sector_and_both():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    grown = GradedSpace.new(SU2, {SU2Sector(0): 3, SU2Sector(1): 1})
    target = (Leg(grown, OUT), *t.legs[1:])
    with pytest.raises(
        ValueError, match=r"restrict: axis 0 sector .* degeneracy 2 in the tensor but 3 in the"
    ):
        tenet.restrict(t, target)


def test_restrict_refuses_a_target_sector_absent_from_the_tensor():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    extra = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(2): 1})
    target = (Leg(extra, OUT), *t.legs[1:])
    with pytest.raises(ValueError, match="absent from the tensor entirely") as exc:
        tenet.restrict(t, target)
    assert str(exc.value).startswith("restrict:")


def test_restrict_refuses_a_side_mismatch():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    target = (*t.legs[:2], replace(t.legs[2], side=OUT))
    with pytest.raises(ValueError, match=re.escape("restrict: axis 2 has side 'in'")) as exc:
        tenet.restrict(t, target)
    assert "repartition, not restrict" in str(exc.value)


def test_restrict_refuses_a_dual_mismatch():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    target = (replace(t.legs[0], dual=True), *t.legs[1:])
    with pytest.raises(ValueError, match="restrict: axis 0 has dual=False.*dual=True"):
        tenet.restrict(t, target)


def test_restrict_refuses_a_provider_mismatch():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    q = GradedSpace.new(U1, {U1Sector(0): 1})
    with pytest.raises(ValueError, match="restrict: axis 0 has provider SU2.*target leg has U1"):
        tenet.restrict(t, (Leg(q, OUT), Leg(q, OUT), Leg(q, IN)))


def test_restrict_refuses_a_leg_count_mismatch():
    t = SymmetricTensor.random(su2_legs(), seed=0)
    with pytest.raises(ValueError, match="restrict: target has 2 legs, but the tensor has 3"):
        tenet.restrict(t, t.legs[:2])


def test_restrict_refusal_is_identical_inside_jit():
    jax = use_jax()
    t = SymmetricTensor.random(su2_legs(), seed=0).to_backend("jax")
    extra = (Leg(GradedSpace.new(SU2, {SU2Sector(2): 1}), OUT), *t.legs[1:])

    @partial(jax.jit, static_argnums=1)
    def f(x, target):
        return tenet.norm(tenet.restrict(x, target, atol=math.inf))

    with pytest.raises(ValueError, match="absent from the tensor entirely"):
        f(t, extra)


# --- the dense oracle, through (alpha, m) -----------------------------------------


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_restrict_dense_image_is_the_source_gathered_through_the_index_maps(name, flavour):
    t, big = fixture(name, flavour)
    u = SymmetricTensor.random(big, seed=23)
    r = tenet.restrict(u, t.legs, atol=math.inf)
    maps = index_maps(t.legs, big)
    assert np.abs(u.to_dense()[np.ix_(*maps)] - r.to_dense()).max() == 0.0


# --- traceability -----------------------------------------------------------------
# Same boundary `embed` sits on, and the same #77 argument: the target comes from
# `legs`, static metadata chosen outside the trace, so the *projection* traces. The
# tolerance comparison is a concrete-value question and JAX says so in its own voice.


def test_restrict_traces_once_per_structure_only_with_atol_inf():
    jax = use_jax()
    t, big = fixture("su2", "new")
    u = SymmetricTensor.random(big, seed=24).to_backend("jax")
    other = SymmetricTensor.random(big, seed=25).to_backend("jax")
    traces = []

    @partial(jax.jit, static_argnums=1)
    def f(x, target):
        traces.append(target)
        return tenet.norm(tenet.restrict(x, target, atol=math.inf))

    a, b = f(u, t.legs), f(other, t.legs)
    assert len(traces) == 1  # different block values, same structure: no retrace
    assert not np.isclose(float(a), float(b))
    f(u, legs(CASES["su2"][1]))
    assert len(traces) == 2  # different legs: retrace

    @partial(jax.jit, static_argnums=1)
    def checked(x, target):
        return tenet.norm(tenet.restrict(x, target))

    with pytest.raises(jax.errors.ConcretizationTypeError):
        checked(u, t.legs)


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_restrict_gradient_matches_central_finite_differences(name):
    """``restrict``'s cotangent is a zero-pad — which is to say, it is ``embed``."""
    jax = use_jax()
    t, big = fixture(name, "new", seed=2)
    u = SymmetricTensor.random(big, seed=26)
    w = SymmetricTensor.random(t.legs, seed=27)

    def scalar(x):
        return tenet.norm(tenet.restrict(x, t.legs, atol=math.inf) + w)

    g = jax.grad(scalar)(u.to_backend("jax"))
    assert [b.shape for b in g.blocks] == [b.shape for b in u.blocks]
    # exact zeros in the dropped slots: the adjoint of a slice is a zero-pad
    for key, block in g.items():
        rest = np.asarray(block).copy()
        if key in set(t.structure.block_order):
            rest[tuple(slice(0, m) for m in t.structure.block_shape(key))] = 0
        assert np.abs(rest).max() == 0.0

    h = 1e-6
    for i, block in enumerate(u.blocks):
        for idx in np.ndindex(block.shape):

            def shifted(delta, i=i, idx=idx):
                blocks = [np.array(b, copy=True) for b in u.blocks]
                blocks[i][idx] += delta
                return float(scalar(SymmetricTensor(u.structure, blocks)))

            fd = (shifted(h) - shifted(-h)) / (2 * h)
            assert abs(fd - float(np.asarray(g.blocks[i])[idx])) < 1e-6


# --- dtypes and backends ----------------------------------------------------------


def test_restrict_keeps_complex_dtype_and_measures_the_residual_from_abs_squared():
    t, big = fixture("su2", "new")
    u = SymmetricTensor.random(big, seed=28)
    u = SymmetricTensor(u.structure, tuple((b * (3 + 4j)).astype(np.complex128) for b in u.blocks))
    r = tenet.restrict(u, t.legs, atol=math.inf)
    assert r.dtype == np.complex128
    for got, hand in zip(r.blocks, project(u, t), strict=True):
        assert np.abs(got - hand).max() == 0.0
    with pytest.raises(ValueError, match=f"residual {discarded_mass(u, t.structure):.6g}"):
        tenet.restrict(u, t.legs)


@pytest.mark.parametrize(("name", "flavour"), PAIRS)
def test_restrict_on_jax_blocks_stays_jax_and_satisfies_the_same_criteria(name, flavour):
    use_jax()
    import autoray as ar

    t, big = fixture(name, flavour)
    u = SymmetricTensor.random(big, seed=29)
    r = tenet.restrict(u.to_backend("jax"), t.legs, atol=math.inf)
    assert {ar.infer_backend(b) for b in r.blocks} == {"jax"}
    for got, hand in zip(r.blocks, project(u, t), strict=True):
        assert np.abs(np.asarray(got) - np.asarray(hand)).max() == 0.0


# ==================================================================================
# direct_sum (#91) — the graded ⊕ of two tensors
# ==================================================================================

# The second operand's axis-0 space: overlapping the first's only partly, so both
# the shared-sector and the sector-in-one-operand-only branches are live.
ALT = {
    "trivial": {TS: 3},
    "u1": {U1Sector(0): 1, U1Sector(1): 2, U1Sector(2): 2},
    "su2": {SU2Sector(1): 2, SU2Sector(2): 1},
    "fz2": {FZ2Sector(0): 1, FZ2Sector(1): 2},
    "product": {uf(1, 1): 2, uf(2, 0): 3},
}


def sum_fixture(name, seeds=(31, 32)):
    """``(t, u)`` agreeing on every leg but axis 0 — the one-axis bond merge."""
    src = CASES[name][0]
    alt = (GradedSpace.new(src[0].provider, ALT[name]), *src[1:])
    return (
        SymmetricTensor.random(legs(src), seed=seeds[0]),
        SymmetricTensor.random(legs(alt), seed=seeds[1]),
    )


# Two summed axes (0 and 2, one OUT and one IN), U(1), chosen so that a result key
# takes its axis-0 sector only from `t`'s space and its axis-2 sector only from
# `u`'s: the block neither operand contributes to.
def two_axis_fixture(seeds=(33, 34)):
    b = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1, U1Sector(2): 1})
    t = SymmetricTensor.random(
        (
            Leg(GradedSpace.new(U1, {U1Sector(0): 2}), OUT, name="a"),
            Leg(b, OUT, name="b"),
            Leg(GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2}), IN, name="c"),
        ),
        seed=seeds[0],
    )
    u = SymmetricTensor.random(
        (
            Leg(GradedSpace.new(U1, {U1Sector(1): 1}), OUT, name="a"),
            Leg(b, OUT, name="b"),
            Leg(GradedSpace.new(U1, {U1Sector(2): 1, U1Sector(3): 2}), IN, name="c"),
        ),
        seed=seeds[1],
    )
    return t, u


# Four maps for the block-diagonal functoriality law: A: x <- y, A2: y <- z, and
# the primed pair on disjoint-ish spaces.
MAP_SPACES = {
    "u1": (
        {U1Sector(0): 2, U1Sector(1): 1},
        {U1Sector(0): 1, U1Sector(1): 2},
        {U1Sector(0): 2, U1Sector(1): 1},
        {U1Sector(1): 1, U1Sector(2): 2},
        {U1Sector(1): 2, U1Sector(2): 1},
        {U1Sector(1): 1, U1Sector(2): 1},
    ),
    "su2": (
        {SU2Sector(0): 2, SU2Sector(1): 1},
        {SU2Sector(0): 1, SU2Sector(1): 2},
        {SU2Sector(0): 2, SU2Sector(1): 1},
        {SU2Sector(1): 1, SU2Sector(2): 2},
        {SU2Sector(1): 2, SU2Sector(2): 1},
        {SU2Sector(1): 1, SU2Sector(2): 1},
    ),
}


def map_fixture(name):
    provider = {"u1": U1, "su2": SU2}[name]
    x, y, z, xp, yp, zp = (GradedSpace.new(provider, m) for m in MAP_SPACES[name])

    def morphism(out, in_, seed):
        return SymmetricTensor.random((Leg(out, OUT, name="l"), Leg(in_, IN, name="r")), seed=seed)

    return (
        morphism(x, y, 41),
        morphism(y, z, 42),
        morphism(xp, yp, 43),
        morphism(yp, zp, 44),
    )


def peak(x):
    """``max |x|``, and ``0.0`` for an empty slot (an operand lacking the sector)."""
    x = np.asarray(x)
    return np.abs(x).max() if x.size else 0.0


def widths(t, u, structure, key, axis):
    """``(m_t, m_u)`` on ``axis`` for ``key`` — ``0`` where an operand lacks the sector."""
    a = structure.axis_sectors(key)[axis]
    return t.legs[axis].space.degeneracy(a), u.legs[axis].space.degeneracy(a)


def trailing_maps(u_legs, target_legs, t_legs, axes):
    """:func:`index_maps` for ``u``, with ``alpha`` shifted past ``t``'s slots.

    Reuses ``index_maps`` unchanged and adds ``m_t(a) * d_a`` inside each sector's
    slab — the trailing placement written through ``(alpha, m)``, not as a suffix.
    """
    maps = index_maps(u_legs, target_legs)
    for ax in axes:
        leg = u_legs[ax]
        d = leg.provider.irrep_dim
        maps[ax] = maps[ax] + np.concatenate(
            [
                np.full(leg.space.degeneracy(a) * d(a), t_legs[ax].space.degeneracy(a) * d(a))
                for a in leg.space
            ]
        )
    return maps


# --- exports and spaces -----------------------------------------------------------


def test_direct_sum_exported_from_tenet_and_ops_and_as_a_method():
    assert "direct_sum" in tenet.__all__ and "direct_sum" in tenet.ops.__all__
    assert tenet.direct_sum is tenet.ops.direct_sum
    t, u = sum_fixture("su2")
    assert t.direct_sum(u, 0) == tenet.direct_sum(t, u, 0)


@pytest.mark.parametrize("name", list(CASES))
def test_summed_spaces_are_sector_wise_degeneracy_sums(name):
    t, u = sum_fixture(name)
    d = tenet.direct_sum(t, u, 0)
    sectors = {*t.legs[0].space, *u.legs[0].space}
    assert set(d.legs[0].space) == sectors
    for a in sectors:
        assert d.legs[0].space.degeneracy(a) == t.legs[0].space.degeneracy(a) + u.legs[
            0
        ].space.degeneracy(a)
    # every non-summed leg is `t`'s, name included
    assert d.legs[1:] == t.legs[1:]


def test_su2_bond_merge_is_the_issues_own_arithmetic():
    t, u = sum_fixture("su2")
    d = tenet.direct_sum(t, u, 0)
    assert d.legs[0].space == GradedSpace.new(
        SU2, {SU2Sector(0): 2, SU2Sector(1): 3, SU2Sector(2): 1}
    )


@pytest.mark.parametrize("name", list(CASES))
def test_the_key_set_contains_both_operands_keys(name):
    t, u = sum_fixture(name)
    d = tenet.direct_sum(t, u, 0)
    assert set(t.structure.block_order) | set(u.structure.block_order) <= set(
        d.structure.block_order
    )


# --- placement --------------------------------------------------------------------


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_one_summed_axis_puts_t_in_the_leading_slots_and_u_in_the_trailing_ones(name):
    t, u = sum_fixture(name)
    d = tenet.direct_sum(t, u, 0)
    tkeys, ukeys = set(t.structure.block_order), set(u.structure.block_order)
    for key, block in d.items():
        block = np.asarray(block)
        mt, mu = widths(t, u, d.structure, key, 0)
        assert block.shape[0] == mt + mu
        lead, trail = block[:mt], block[mt:]
        if key in tkeys:
            assert peak(lead - np.asarray(t.block(key))) == 0.0
        else:
            assert peak(lead) == 0.0
        if key in ukeys:
            assert peak(trail - np.asarray(u.block(key))) == 0.0
        else:
            assert peak(trail) == 0.0


def test_two_summed_axes_put_the_operands_on_the_diagonal_corners_and_zero_the_rest():
    t, u = two_axis_fixture()
    axes = (0, 2)
    d = tenet.direct_sum(t, u, axes)
    tkeys, ukeys = set(t.structure.block_order), set(u.structure.block_order)
    for key, block in d.items():
        block = np.asarray(block)
        (mt0, mu0), (mt2, mu2) = (widths(t, u, d.structure, key, ax) for ax in axes)
        lead = (slice(0, mt0), slice(None), slice(0, mt2))
        trail = (slice(mt0, mt0 + mu0), slice(None), slice(mt2, mt2 + mu2))
        mixed = (
            (slice(0, mt0), slice(None), slice(mt2, mt2 + mu2)),
            (slice(mt0, mt0 + mu0), slice(None), slice(0, mt2)),
        )
        if key in tkeys:
            assert peak(block[lead] - np.asarray(t.block(key))) == 0.0
        else:
            assert peak(block[lead]) == 0.0
        if key in ukeys:
            assert peak(block[trail] - np.asarray(u.block(key))) == 0.0
        else:
            assert peak(block[trail]) == 0.0
        for corner in mixed:
            assert peak(block[corner]) == 0.0


def test_a_key_neither_operand_has_is_an_exactly_zero_block():
    t, u = two_axis_fixture()
    d = tenet.direct_sum(t, u, (0, 2))
    orphans = [
        key
        for key in d.structure.block_order
        if key not in set(t.structure.block_order) | set(u.structure.block_order)
    ]
    assert orphans  # the branch is reachable, not assumed so
    for key in orphans:
        assert np.abs(np.asarray(d.block(key))).max() == 0.0


# --- norm, embed, functoriality ---------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_norm_adds_in_quadrature_for_one_summed_axis(name):
    t, u = sum_fixture(name)
    d = tenet.direct_sum(t, u, 0)
    assert float(tenet.norm(d)) ** 2 == pytest.approx(
        float(tenet.norm(t)) ** 2 + float(tenet.norm(u)) ** 2, rel=1e-14
    )


def test_norm_adds_in_quadrature_for_two_summed_axes():
    t, u = two_axis_fixture()
    d = tenet.direct_sum(t, u, (0, 2))
    assert float(tenet.norm(d)) ** 2 == pytest.approx(
        float(tenet.norm(t)) ** 2 + float(tenet.norm(u)) ** 2, rel=1e-14
    )


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_embed_is_the_zero_second_operand_case(name):
    t, u = sum_fixture(name)
    d = tenet.direct_sum(t, u, 0)
    zero = SymmetricTensor.zeros(u.legs)
    assert tenet.allclose(tenet.direct_sum(t, zero, 0), tenet.embed(t, d.legs))


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_block_diagonal_functoriality_through_compose_and_tensordot(name):
    """``(A ⊕ B) ∘ (A' ⊕ B') == (A ∘ A') ⊕ (B ∘ B')``.

    The executable proof that the mixed corners are zero: a non-zero corner would
    couple ``A`` to ``B'`` and break this identity immediately.
    """
    a, a2, b, b2 = map_fixture(name)
    axes = (0, 1)
    left = tenet.direct_sum(a, b, axes) @ tenet.direct_sum(a2, b2, axes)
    assert tenet.allclose(left, tenet.direct_sum(a @ a2, b @ b2, axes))
    dotted = tenet.tensordot(
        tenet.direct_sum(a, b, axes), tenet.direct_sum(a2, b2, axes), axes=([1], [0])
    )
    assert tenet.allclose(
        dotted,
        tenet.direct_sum(
            tenet.tensordot(a, a2, axes=([1], [0])), tenet.tensordot(b, b2, axes=([1], [0])), axes
        ),
    )


# --- the dense oracle, through (alpha, m) -----------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_direct_sum_dense_image_holds_both_operands_and_nothing_else(name):
    t, u = sum_fixture(name)
    d = tenet.direct_sum(t, u, 0)
    dense = d.to_dense()
    assert np.abs(dense[np.ix_(*index_maps(t.legs, d.legs))] - t.to_dense()).max() == 0.0
    trailing = trailing_maps(u.legs, d.legs, t.legs, (0,))
    assert np.abs(dense[np.ix_(*trailing)] - u.to_dense()).max() == 0.0
    assert np.abs(dense).sum() == pytest.approx(
        np.abs(t.to_dense()).sum() + np.abs(u.to_dense()).sum(), rel=1e-14
    )


# --- dtype, names -----------------------------------------------------------------


def test_dtype_promotion_reaches_every_block_including_the_untouched_ones():
    import autoray as ar

    t, u = two_axis_fixture()
    u = SymmetricTensor(u.structure, tuple((b * (1 + 2j)).astype(np.complex128) for b in u.blocks))
    d = tenet.direct_sum(t, u, (0, 2))
    assert {ar.get_dtype_name(b) for b in d.blocks} == {"complex128"}


def test_differing_names_are_accepted_and_t_wins():
    """``name`` is user bookkeeping, exactly the stance ``ProductSpace.matches`` takes."""
    t, u = sum_fixture("su2")
    other = tuple(leg.renamed(f"other-{i}") for i, leg in enumerate(u.legs))
    renamed = SymmetricTensor.from_blocks(
        other, dict(zip(TensorStructure(other).block_order, u.blocks, strict=True))
    )
    d = tenet.direct_sum(t, renamed, 0)
    assert tuple(leg.name for leg in d.legs) == ("a", "b", "c")


# --- structural refusals ----------------------------------------------------------


def test_direct_sum_refuses_a_leg_count_mismatch():
    t, u = sum_fixture("su2")
    two = SymmetricTensor.random(u.legs[:2], seed=35)
    with pytest.raises(ValueError, match="the operands have 3 and 2 legs"):
        tenet.direct_sum(t, two, 0)


def test_direct_sum_refuses_a_provider_mismatch():
    t, _ = sum_fixture("su2")
    q = GradedSpace.new(U1, {U1Sector(0): 2})
    other = SymmetricTensor.random((Leg(q, OUT), Leg(q, OUT), Leg(q, IN)), seed=36)
    with pytest.raises(ValueError, match="axis 0 has provider SU2 in t but U1 in u"):
        tenet.direct_sum(t, other, 0)


def test_direct_sum_refuses_a_side_mismatch():
    t, u = sum_fixture("su2")
    flipped = SymmetricTensor.random((u.legs[0], u.legs[1], replace(u.legs[2], side=OUT)), seed=37)
    with pytest.raises(ValueError, match=re.escape("axis 2 has side 'in' in t but 'out' in u")):
        tenet.direct_sum(t, flipped, 0)


def test_direct_sum_refuses_a_dual_mismatch_citing_that_v_plus_v_star_is_not_graded():
    t, u = sum_fixture("su2")
    dualized = SymmetricTensor.random((replace(u.legs[0], dual=True), *u.legs[1:]), seed=38)
    with pytest.raises(ValueError, match="axis 0 has dual=False in t but dual=True in u") as exc:
        tenet.direct_sum(t, dualized, 0)
    assert "V ⊕ V* is not a graded space" in str(exc.value)


def test_direct_sum_refuses_a_non_summed_axis_whose_spaces_differ():
    t, u = sum_fixture("su2")
    with pytest.raises(ValueError, match="axis 0 is not summed but the operands' spaces differ"):
        tenet.direct_sum(t, u, 1)


@pytest.mark.parametrize(
    ("axes", "message"),
    [
        ((), "axes is empty"),
        ((0, 0), "duplicate axis"),
        ((0, -3), "duplicate axis"),
        (3, "axis 3 is out of range"),
        (-4, "axis -4 is out of range"),
    ],
)
def test_direct_sum_refuses_bad_axes(axes, message):
    t, u = sum_fixture("su2")
    with pytest.raises(ValueError, match=message):
        tenet.direct_sum(t, u, axes)


def test_direct_sum_accepts_a_negative_axis_index():
    t, u = two_axis_fixture()
    assert tenet.direct_sum(t, u, (0, -1)) == tenet.direct_sum(t, u, (0, 2))


# --- traceability and gradients ---------------------------------------------------
# Same #77 argument as `embed`: the result structure comes from the operands' legs
# and `axes`, all static metadata, so this is nothing like #64's
# StructureChangingError — nothing here is decided from a block value.


def test_direct_sum_traces_once_per_structure_pair():
    jax = use_jax()
    t, u = sum_fixture("su2")
    other, _ = sum_fixture("su2", seeds=(39, 32))
    traces = []

    @partial(jax.jit, static_argnums=2)
    def f(x, y, axes):
        traces.append(axes)
        return tenet.norm(tenet.direct_sum(x, y, axes))

    tj, uj = t.to_backend("jax"), u.to_backend("jax")
    a = f(tj, uj, 0)
    b = f(other.to_backend("jax"), uj, 0)
    assert len(traces) == 1  # different block values, same structures: no retrace
    assert not np.isclose(float(a), float(b))
    f(tj, uj, (0,))
    assert len(traces) == 2  # a different `axes` object is a different static arg


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_direct_sum_gradients_match_central_finite_differences_in_each_operand(name):
    jax = use_jax()
    t, u = sum_fixture(name)
    d = tenet.direct_sum(t, u, 0)
    w = SymmetricTensor.random(d.legs, seed=45)

    def scalar(x, y):
        return tenet.norm(tenet.direct_sum(x, y, 0) + w)

    # both operands travel together: mixed backends are a refusal (below), so the
    # traced side and the frozen one are converted as a pair
    for slot in (0, 1):
        operand, frozen = (t, u) if slot == 0 else (u, t)

        def pair(x, f, slot=slot):
            """``scalar`` with ``x`` in the slot being differentiated."""
            return scalar(x, f) if slot == 0 else scalar(f, x)

        jax_frozen = frozen.to_backend("jax")
        g = jax.grad(lambda x, f=jax_frozen: pair(x, f))(operand.to_backend("jax"))
        assert [b.shape for b in g.blocks] == [b.shape for b in operand.blocks]
        h = 1e-6
        for i, block in enumerate(operand.blocks):
            for idx in np.ndindex(block.shape):

                def shifted(delta, i=i, idx=idx, operand=operand, frozen=frozen, pair=pair):
                    blocks = [np.array(b, copy=True) for b in operand.blocks]
                    blocks[i][idx] += delta
                    return float(pair(SymmetricTensor(operand.structure, blocks), frozen))

                fd = (shifted(h) - shifted(-h)) / (2 * h)
                assert abs(fd - float(np.asarray(g.blocks[i])[idx])) < 1e-6


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_direct_sum_is_linear_in_its_first_operand(name):
    """``(a + b) ⊕ u == (a ⊕ u) + (b ⊕ 0)``.

    #91's criterion spells the right-hand side ``(a ⊕ u) + (b ⊕ u)``, which
    double-counts ``u``: the sum of two direct sums puts ``u`` in the trailing
    slots twice. Linearity is *separate* in each operand, so the second term
    carries a zero second operand — the same statement, without the typo.
    """
    t, u = sum_fixture(name)
    other, _ = sum_fixture(name, seeds=(46, 32))
    zero = SymmetricTensor.zeros(u.legs)
    assert tenet.allclose(
        tenet.direct_sum(t + other, u, 0),
        tenet.direct_sum(t, u, 0) + tenet.direct_sum(other, zero, 0),
        rtol=0.0,
        atol=1e-14,
    )


def test_direct_sum_refuses_mixed_backend_operands():
    """``add`` promotes because every block meets both operands; here they do not.

    A key only ``t`` contributes to never sees ``u``'s backend, so silence would
    return a tensor holding NumPy *and* JAX blocks — the backend twin of the
    mixed-dtype bug the promotion above exists to prevent.
    """
    use_jax()
    t, u = sum_fixture("su2")
    with pytest.raises(ValueError, match="operands are on different backends"):
        tenet.direct_sum(t, u.to_backend("jax"), 0)


@pytest.mark.parametrize("name", list(CASES))
def test_direct_sum_on_jax_blocks_stays_jax(name):
    use_jax()
    import autoray as ar

    t, u = sum_fixture(name)
    d = tenet.direct_sum(t.to_backend("jax"), u.to_backend("jax"), 0)
    assert {ar.infer_backend(b) for b in d.blocks} == {"jax"}
    assert float(tenet.norm(d)) ** 2 == pytest.approx(
        float(tenet.norm(t)) ** 2 + float(tenet.norm(u)) ** 2, rel=1e-14
    )
