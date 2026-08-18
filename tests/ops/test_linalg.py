"""Tests for the fixed-structure blockwise decompositions — issue #57.

Every criterion here is stated on a *gauge-invariant* quantity: reconstruction,
``U†U``, singular values, norms. The SVD is unique only up to per-singular-value
phases and LAPACK's QR does not fix the sign of ``R``'s diagonal, so an
elementwise comparison of factors against anything would be testing the backend's
mood, not this library.
"""

import dataclasses
import pathlib
import sys

import autoray as ar
import numpy as np
import pytest
from helpers import sector_parity

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import as_map, map_layout, to_matrices
from tenet.symmetry import (
    SU2,
    U1,
    CapabilityError,
    FZ2Sector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

REPARTITION_MODULE = sys.modules["tenet.ops.repartition"]

# --- the cases ------------------------------------------------------------------
# Sides interleave everywhere (OUT at 0, 2 and IN at 1, 3), and the split
# ((0, 1), (2, 3)) therefore bends axis 1 and axis 2: repartition does real work,
# including the transposes that carry the fZ2 signs.

SPLIT = ((0, 1), (2, 3))

T_TRIV = tuple(GradedSpace.new(Trivial, {TrivialSector(): m}) for m in (2, 3, 4, 3))
Q1 = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
Q2 = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})
V = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 3})
W = GradedSpace.new(SU2, {SU2Sector(1): 2, SU2Sector(2): 1})
X = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(2): 2})
F1 = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 3})
F2 = GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 1})

LEGS = {
    "trivial": tuple(
        Leg(s, side, name=f"l{i}")
        for i, (s, side) in enumerate(zip(T_TRIV, (OUT, IN, OUT, IN), strict=True))
    ),
    "u1": (Leg(Q1, OUT, name="a"), Leg(Q2, IN, name="b"), Leg(Q2, OUT), Leg(Q1, IN)),
    # asymmetric on purpose: some coupled sectors are row-limited, others column-limited
    "su2": (Leg(V, OUT, name="a"), Leg(W, IN, name="b"), Leg(X, OUT), Leg(V, IN)),
    "fz2": (Leg(F1, OUT, name="a"), Leg(F2, IN, name="b"), Leg(F2, OUT), Leg(F1, IN)),
}
PROVIDERS = ["trivial", "u1", "su2", "fz2"]


def tensor(name: str, seed: int = 0) -> SymmetricTensor:
    return SymmetricTensor.random(LEGS[name], seed=seed)


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  registration is the import's side effect

    return jax


def to_jax(t: SymmetricTensor) -> SymmetricTensor:
    import jax.numpy as jnp

    return SymmetricTensor(t.structure, tuple(jnp.asarray(b) for b in t.blocks))


def singular_values(s: SymmetricTensor) -> dict:
    """``{c: sigma_c}`` read back out of the diagonal operator, the documented way."""
    return {c: np.asarray(ar.do("diagonal", m)) for c, m in to_matrices(s).items()}


def dense_matrix(t: SymmetricTensor) -> np.ndarray:
    """``t.to_dense()`` regrouped as (OUT physical axes) x (IN physical axes)."""
    order = (*t.structure.out_axes, *t.structure.in_axes)
    dense = t.to_dense().transpose(order)
    rows = int(np.prod(dense.shape[: len(t.structure.out_axes)], dtype=int))
    return dense.reshape(rows, -1)


# --- leg layout and the bond space ----------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_factor_legs_are_left_bond_and_bond_right(name):
    t = tensor(name)
    left, right = SPLIT
    m = tenet.repartition(t, left, right)
    u, s, vh = tenet.linalg.svd(t, axes=SPLIT)

    bond = u.legs[-1].space
    assert u.legs == (*m.codomain, Leg(bond, IN))
    assert s.legs == (Leg(bond, OUT), Leg(bond, IN))
    assert vh.legs == (Leg(bond, OUT), *m.domain)
    # every non-bond leg keeps space, dual and name from the repartitioned tensor
    for got, want in zip(u.legs[:-1], m.codomain, strict=True):
        assert (got.space, got.dual, got.name) == (want.space, want.dual, want.name)
    for got, want in zip(vh.legs[1:], m.domain, strict=True):
        assert (got.space, got.dual, got.name) == (want.space, want.dual, want.name)


@pytest.mark.parametrize("name", PROVIDERS)
def test_qr_factor_legs(name):
    t = tensor(name)
    m = tenet.repartition(t, *SPLIT)
    q, r = tenet.linalg.qr(t, axes=SPLIT)
    bond = q.legs[-1].space
    assert q.legs == (*m.codomain, Leg(bond, IN))
    assert r.legs == (Leg(bond, OUT), *m.domain)


def test_bond_space_is_min_of_the_map_layout_shape_both_ways_round():
    """Row-limited *and* column-limited sectors in one tensor, or the min is vacuous."""
    t = tensor("su2")
    m = tenet.repartition(t, *SPLIT)
    layout = map_layout(m.structure)
    shapes = [layout.shape(c) for c in layout.sectors]
    assert any(rows < cols for rows, cols in shapes)
    assert any(rows > cols for rows, cols in shapes)

    bond = tenet.linalg.svd(t, axes=SPLIT)[0].legs[-1].space
    assert tuple(bond) == layout.sectors
    assert bond.sectors == tuple((c, min(*layout.shape(c))) for c in layout.sectors)
    assert tenet.linalg.qr(t, axes=SPLIT)[0].legs[-1].space == bond


def test_zero_rank_sectors_keep_their_full_bond_degeneracy():
    """The rank of ``B_c`` is data; its shape is metadata (invariant 10)."""
    t = tensor("su2")
    zeroed = SymmetricTensor(t.structure, tuple(np.zeros_like(b) for b in t.blocks))
    layout = map_layout(tenet.repartition(t, *SPLIT).structure)
    u, s, _ = tenet.linalg.svd(zeroed, axes=SPLIT)
    assert u.legs[-1].space.sectors == tuple((c, min(*layout.shape(c))) for c in layout.sectors)
    for sigma in singular_values(s).values():
        np.testing.assert_array_equal(sigma, 0.0)


# --- reconstruction --------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_svd_reconstructs_the_repartitioned_tensor(name):
    t = tensor(name, seed=1)
    u, s, vh = tenet.linalg.svd(t, axes=SPLIT)
    assert tenet.allclose(u @ s @ vh, tenet.repartition(t, *SPLIT), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", PROVIDERS)
def test_qr_reconstructs_the_repartitioned_tensor(name):
    t = tensor(name, seed=2)
    q, r = tenet.linalg.qr(t, axes=SPLIT)
    assert tenet.allclose(q @ r, tenet.repartition(t, *SPLIT), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", PROVIDERS)
def test_round_trip_back_to_the_original_sides_and_axis_order(name):
    """The sides interleave, so the returning transpose is not the identity."""
    t = tensor(name, seed=3)
    u, s, vh = tenet.linalg.svd(t, axes=SPLIT)
    back = tenet.repartition(u @ s @ vh, (0, 2), (1, 3)).transpose((0, 2, 1, 3))
    assert back.legs == t.legs
    assert tenet.allclose(back, t, rtol=0, atol=1e-12)


def test_fz2_split_crosses_odd_legs_and_the_signs_survive_the_round_trip():
    t = tensor("fz2", seed=4)
    # the two bent axes both carry odd sectors, so the transposes inside
    # repartition pick up genuine Koszul signs
    assert all(any(sector_parity(a) for a in leg.space) for leg in t.legs[1:3])
    m = tenet.repartition(t, *SPLIT)

    u, s, vh = tenet.linalg.svd(t, axes=SPLIT)
    assert tenet.allclose(u @ s @ vh, m, rtol=0, atol=1e-12)
    bond = Leg(u.legs[-1].space, OUT)
    assert tenet.allclose(tenet.adjoint(u) @ u, tenet.identity((bond,)), rtol=0, atol=1e-12)
    back = tenet.repartition(u @ s @ vh, (0, 2), (1, 3)).transpose((0, 2, 1, 3))
    assert tenet.allclose(back, t, rtol=0, atol=1e-12)


# --- isometry --------------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_per_sector_isometry(name):
    t = tensor(name, seed=5)
    u, _, vh = tenet.linalg.svd(t, axes=SPLIT)
    q, _ = tenet.linalg.qr(t, axes=SPLIT)
    for factor in (u, q):
        bond = tenet.identity((Leg(factor.legs[-1].space, OUT),))
        assert tenet.allclose(tenet.adjoint(factor) @ factor, bond, rtol=0, atol=1e-12)
    bond = tenet.identity((Leg(vh.legs[0].space, OUT),))
    assert tenet.allclose(vh @ tenet.adjoint(vh), bond, rtol=0, atol=1e-12)


# --- the singular values themselves ----------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_s_is_diagonal_real_non_negative_and_descending(name):
    t = tensor(name, seed=6)
    _, s, _ = tenet.linalg.svd(t, axes=SPLIT)
    assert not np.issubdtype(np.asarray(s.blocks[0]).dtype, np.complexfloating)
    for c, mat in to_matrices(s).items():
        mat = np.asarray(mat)
        sigma = np.diagonal(mat)
        np.testing.assert_allclose(mat - np.diag(sigma), 0.0, atol=1e-14, err_msg=repr(c))
        assert np.all(sigma >= 0.0)
        assert np.all(np.diff(sigma) <= 1e-14)


def test_s_stays_real_for_a_complex_tensor():
    t = tensor("su2", seed=7)
    t = SymmetricTensor(t.structure, tuple(b + 0.5j * b[::-1] for b in t.blocks))
    u, s, vh = tenet.linalg.svd(t, axes=SPLIT)
    assert np.issubdtype(np.asarray(u.blocks[0]).dtype, np.complexfloating)
    assert not np.issubdtype(np.asarray(s.blocks[0]).dtype, np.complexfloating)
    assert tenet.allclose(u @ s @ vh, tenet.repartition(t, *SPLIT), rtol=0, atol=1e-12)


def test_su2_norm_identity_carries_the_qdim_weight():
    t = tensor("su2", seed=8)
    _, s, _ = tenet.linalg.svd(t, axes=SPLIT)
    sigmas = singular_values(s)
    assert len({SU2.qdim(c) for c in sigmas}) > 1  # the weight is not vacuous
    weighted = sum(SU2.qdim(c) * float(np.sum(sigma**2)) for c, sigma in sigmas.items())
    assert float(tenet.norm(t)) ** 2 == pytest.approx(weighted, abs=1e-12)

    unweighted = sum(float(np.sum(sigma**2)) for sigma in sigmas.values())
    assert abs(unweighted - weighted) > 0.1 * weighted


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_dense_oracle_singular_values(name):
    """The #29 oracle, now fed from ``svd``'s own ``S``.

    ``axes=None`` matricizes along the tensor's current sides, which is exactly
    what :func:`dense_matrix` densifies — and it needs no bend, so no dual leg
    enters the dense expansion.
    """
    t = tensor(name, seed=9)
    provider = t.provider
    _, s, _ = tenet.linalg.svd(t)
    got = np.sort(
        np.concatenate(
            [np.repeat(sigma, provider.irrep_dim(c)) for c, sigma in singular_values(s).items()]
        )
    )[::-1]
    want = np.sort(np.linalg.svd(dense_matrix(t), compute_uv=False))[::-1]
    n = min(len(got), len(want))
    np.testing.assert_allclose(got[:n], want[:n], atol=1e-10)
    np.testing.assert_allclose(np.concatenate([got[n:], want[n:]]), 0.0, atol=1e-10)


# --- the map-style spelling -------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_as_map_svd_equals_the_axes_spelling_and_bends_nothing(name, monkeypatch):
    t = tensor(name, seed=10)
    axes = (t.structure.out_axes, t.structure.in_axes)

    def no_bends(*args, **kwargs):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("as_map().svd() must perform zero bends")

    monkeypatch.setattr(REPARTITION_MODULE, "bend", no_bends)
    view = t.as_map().svd()
    want = tenet.linalg.svd(t, axes=axes)
    for got, expected in zip(view, want, strict=True):
        assert got.structure == expected.structure
        assert tenet.allclose(got, expected, rtol=0, atol=1e-14)
    assert tenet.allclose(view[0] @ view[1] @ view[2], tenet.repartition(t, *axes), atol=1e-12)


def test_as_map_qr_matches_the_axes_spelling():
    t = tensor("su2", seed=11)
    axes = (t.structure.out_axes, t.structure.in_axes)
    for got, expected in zip(t.as_map().qr(), tenet.linalg.qr(t, axes=axes), strict=True):
        assert tenet.allclose(got, expected, rtol=0, atol=1e-14)


def test_the_milestone_7_refusal_is_gone():
    """Replaces ``tests/ops/test_map.py::test_svd_names_milestone_7_and_to_matrices``."""
    u, s, vh = tensor("su2", seed=12).as_map().svd()
    assert (u.ndim, s.ndim, vh.ndim) == (3, 2, 3)


# --- refusals and validation -------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FusionOnly:
    """FusionRules + QuantumDimensionData, nothing else — no bending capability."""

    name: str = "FusionOnly"

    @property
    def unit(self):
        return TrivialSector()

    def dual(self, a):
        return a

    def fusion(self, a, b):
        return (TrivialSector(),)

    def n_symbol(self, a, b, c):
        return 1

    def qdim(self, a):
        return 1.0


def capability_less() -> SymmetricTensor:
    space = GradedSpace.new(FusionOnly(), {TrivialSector(): 2})
    legs = (Leg(space, OUT), Leg(space, IN), Leg(space, OUT), Leg(space, IN))
    return SymmetricTensor.random(legs, seed=13)


@pytest.mark.parametrize("fn", [tenet.linalg.svd, tenet.linalg.qr])
def test_a_crossing_split_is_refused_without_bending_coefficients(fn):
    with pytest.raises(CapabilityError) as excinfo:
        fn(capability_less(), axes=SPLIT)
    message = str(excinfo.value)
    assert "FusionOnly" in message
    assert "BendingCoefficients" in message
    assert "axis 1" in message  # the first crossing axis, in the caller's numbering


@pytest.mark.parametrize("fn", [tenet.linalg.svd, tenet.linalg.qr])
def test_a_non_crossing_split_needs_no_capability(fn):
    t = capability_less()
    factors = fn(t, axes=(t.structure.out_axes, t.structure.in_axes))
    assert tenet.allclose(
        factors[0] @ factors[1] if len(factors) == 2 else factors[0] @ factors[1] @ factors[2],
        tenet.repartition(t, t.structure.out_axes, t.structure.in_axes),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("axes", "match"),
    [
        (((0, 1), (2,)), "axis 3"),
        (((0, 1), (1, 2, 3)), "twice"),
        (((-1, 1), (2, 3)), "out of range"),
        (((0, 1), (2, 4)), "out of range"),
        ((("x", 1), (2, 3)), "must all be integers"),
    ],
    ids=["missing", "repeat", "negative", "too-large", "not-an-int"],
)
def test_axes_validation_is_the_house_style(axes, match):
    with pytest.raises(ValueError, match=match):
        tenet.linalg.svd(tensor("u1"), axes=axes)


def test_ar_do_svd_still_raises():
    """Invariant 11: the dispatch list stays closed until M8 settles a contract."""
    with pytest.raises(Exception):  # noqa: B017  the exception type is autoray's business
        ar.do("svd", tensor("su2"))
    assert not hasattr(tenet, "svd")


# --- jit ---------------------------------------------------------------------------


def test_jit_traces_once_and_matches_the_numpy_backend():
    jax = use_jax()
    traces = []

    @jax.jit
    def factorize(x):
        traces.append(1)
        return tenet.linalg.svd(x, axes=SPLIT)

    t = tensor("su2", seed=14)
    u, s, vh = factorize(to_jax(t))
    factorize(to_jax(tensor("su2", seed=15)))
    assert len(traces) == 1  # same structure, same treedef: no retrace

    want_u, want_s, want_vh = tenet.linalg.svd(t, axes=SPLIT)
    assert u.structure == want_u.structure and vh.structure == want_vh.structure
    assert tenet.allclose(s, want_s, rtol=0, atol=1e-10)  # gauge-invariant
    assert tenet.allclose(u @ s @ vh, tenet.repartition(t, *SPLIT), rtol=0, atol=1e-10)


def test_jit_qr_traces_once_and_matches_the_numpy_backend():
    jax = use_jax()
    traces = []

    @jax.jit
    def factorize(x):
        traces.append(1)
        return tenet.linalg.qr(x, axes=SPLIT)

    t = tensor("su2", seed=16)
    q, r = factorize(to_jax(t))
    factorize(to_jax(tensor("su2", seed=17)))
    assert len(traces) == 1

    want_q, _ = tenet.linalg.qr(t, axes=SPLIT)
    assert q.structure == want_q.structure
    assert tenet.allclose(q @ r, tenet.repartition(t, *SPLIT), rtol=0, atol=1e-10)
    bond = tenet.identity((Leg(q.legs[-1].space, OUT),))
    assert tenet.allclose(tenet.adjoint(q) @ q, bond, rtol=0, atol=1e-10)


def test_jit_of_a_single_factor_traces():
    """The issue's literal spelling: only the first factor comes out."""
    jax = use_jax()
    u = jax.jit(lambda x: tenet.linalg.svd(x, axes=SPLIT)[0])(to_jax(tensor("su2", seed=18)))
    assert isinstance(u, SymmetricTensor)
    assert u.backend == "jax"


# --- svd(bond=...): the projection onto a pre-decided bond space (#77) ---------------


def full_bond_space(name, axes=SPLIT, seed=0):
    return tenet.linalg.svd(tensor(name, seed=seed), axes=axes)[0].legs[-1].space


def halved(space):
    """``space`` with every degeneracy halved (at least 1) — a genuine subspace."""
    return GradedSpace.new(space.provider, {c: max(1, m // 2) for c, m in space.sectors})


@pytest.mark.parametrize("name", PROVIDERS)
def test_bond_keeps_the_largest_singular_values_per_sector(name):
    """The kept set is a per-sector *prefix*, checked against an independent slice."""
    t = tensor(name, seed=20)
    b = halved(full_bond_space(name, seed=20))
    _, s_full, _ = tenet.linalg.svd(t, axes=SPLIT)
    u, s, vh = tenet.linalg.svd(t, axes=SPLIT, bond=b)

    assert u.legs[-1].space == b
    assert s.legs == (Leg(b, OUT), Leg(b, IN))
    assert vh.legs[0] == Leg(b, OUT)
    full, kept = singular_values(s_full), singular_values(s)
    assert set(kept) == set(b)
    for c, k in b.sectors:
        np.testing.assert_allclose(kept[c], np.sort(full[c])[::-1][:k], rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", PROVIDERS)
def test_bond_drops_a_sector_entirely(name):
    """A sector absent from ``B`` is gone from the factors, not zeroed."""
    space = full_bond_space(name, seed=21)
    dropped = space.sectors[0][0]
    b = GradedSpace.new(space.provider, [(c, m) for c, m in space.sectors if c != dropped])
    if len(b) == 0:
        pytest.skip("single-sector layout")
    u, s, vh = tenet.linalg.svd(tensor(name, seed=21), axes=SPLIT, bond=b)
    assert dropped not in u.legs[-1].space
    assert set(singular_values(s)) == set(b)
    assert vh.legs[0].space == b


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_full_bond_reproduces_bond_none_exactly(name):
    """Not a code path: the projection at full rank is the identity slice."""
    t = tensor(name, seed=22)
    want = tenet.linalg.svd(t, axes=SPLIT)
    got = tenet.linalg.svd(t, axes=SPLIT, bond=want[0].legs[-1].space)
    for a, b in zip(got, want, strict=True):
        assert a.structure == b.structure
        for x, y in zip(a.blocks, b.blocks, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("name", PROVIDERS)
def test_as_map_svd_with_a_bond_agrees_and_bends_nothing(name, monkeypatch):
    t = tensor(name, seed=23)
    axes = (t.structure.out_axes, t.structure.in_axes)
    b = halved(tenet.linalg.svd(t)[0].legs[-1].space)

    def no_bends(*args, **kwargs):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("as_map().svd(bond=...) must perform zero bends")

    monkeypatch.setattr(REPARTITION_MODULE, "bend", no_bends)
    for got, want in zip(
        t.as_map().svd(bond=b), tenet.linalg.svd(t, axes=axes, bond=b), strict=True
    ):
        assert got.structure == want.structure
        assert tenet.allclose(got, want, rtol=0, atol=1e-14)


@pytest.mark.parametrize("name", PROVIDERS)
def test_bond_is_the_best_approximation_and_pythagoras_still_measures_the_error(name):
    t = tensor(name, seed=24)
    space = full_bond_space(name, seed=24)
    b = halved(space)
    u, s, vh = tenet.linalg.svd(t, axes=SPLIT, bond=b)
    m = tenet.repartition(t, *SPLIT)
    assert not tenet.allclose(u @ s @ vh, m, rtol=0, atol=1e-6)  # exactness is conditional now

    full = singular_values(tenet.linalg.svd(t, axes=SPLIT)[1])
    qdim = t.provider.qdim
    dropped = sum(
        qdim(c) * float(np.sum(np.sort(sigma)[::-1][b.degeneracy(c) :] ** 2))
        for c, sigma in full.items()
    )
    error = float(tenet.norm(m)) ** 2 - float(tenet.norm(u @ s @ vh)) ** 2
    assert error == pytest.approx(dropped, abs=1e-12)


def bad_bonds(name="u1"):
    space = full_bond_space(name)
    absent = GradedSpace.new(U1, {U1Sector(7): 1})
    c, m = space.sectors[0]
    too_big = GradedSpace.new(space.provider, {c: m + 3})
    return absent, too_big, c, m


def test_a_bond_with_an_absent_sector_is_a_value_error():
    absent, _, _, _ = bad_bonds()
    with pytest.raises(ValueError) as excinfo:
        tenet.linalg.svd(tensor("u1"), axes=SPLIT, bond=absent)
    message = str(excinfo.value)
    assert repr(U1Sector(7)) in message
    assert "asks for 1 singular values" in message  # both degeneracies are named
    assert "degeneracy there is 0" in message
    assert "does not appear" in message


def test_a_bond_asking_for_too_many_singular_values_is_a_value_error():
    _, too_big, c, m = bad_bonds()
    with pytest.raises(ValueError) as excinfo:
        tenet.linalg.svd(tensor("u1"), axes=SPLIT, bond=too_big)
    message = str(excinfo.value)
    assert repr(c) in message
    assert f"{m + 3} singular values" in message
    assert f"degeneracy there is {m}" in message
    assert "min(rows_c, cols_c)" in message


def test_the_refusal_reads_no_block_value():
    """NaN everywhere and the refusal is unchanged: it is metadata against metadata."""
    t = tensor("u1")
    nan = SymmetricTensor(t.structure, tuple(np.full_like(b, np.nan) for b in t.blocks))
    _, too_big, _, _ = bad_bonds()
    with pytest.raises(ValueError, match="subspace of the untruncated bond"):
        tenet.linalg.svd(nan, axes=SPLIT, bond=too_big)


@pytest.mark.parametrize("which", [0, 1], ids=["absent-sector", "too-large-degeneracy"])
def test_the_refusals_are_identical_inside_and_outside_jit(which):
    jax = use_jax()
    b = bad_bonds()[which]
    t = tensor("u1", seed=25)

    with pytest.raises(ValueError) as eager:
        tenet.linalg.svd(t, axes=SPLIT, bond=b)
    with pytest.raises(ValueError) as traced:
        jax.jit(lambda x: tenet.linalg.svd(x, axes=SPLIT, bond=b)[1])(to_jax(t))
    assert str(traced.value) == str(eager.value)


def test_jit_and_grad_go_straight_through_a_fixed_bond():
    """The traceable half of the boundary; ``svd_truncated``'s refusal is the other."""
    jax = use_jax()
    t = to_jax(tensor("su2", seed=26))
    b = halved(full_bond_space("su2", seed=26))

    s = jax.jit(lambda x: tenet.linalg.svd(x, axes=SPLIT, bond=b)[1])(t)
    assert s.legs[0].space == b
    assert s.backend == "jax"

    g = jax.grad(lambda x: tenet.norm(tenet.linalg.svd(x, axes=SPLIT, bond=b)[1]))(t)
    assert all(np.isfinite(np.asarray(blk)).all() for blk in g.blocks)


def test_the_bond_is_part_of_the_static_signature():
    """Changing ``B`` retraces; changing block values does not."""
    jax = use_jax()
    traces = []
    space = full_bond_space("su2", seed=27)

    def raw(x, b):
        traces.append(b)
        return tenet.linalg.svd(x, axes=SPLIT, bond=b)[1]

    factorize = jax.jit(raw, static_argnums=1)
    factorize(to_jax(tensor("su2", seed=27)), space)
    factorize(to_jax(tensor("su2", seed=28)), space)  # different values, same signature
    assert len(traces) == 1
    factorize(to_jax(tensor("su2", seed=27)), halved(space))
    assert len(traces) == 2


# --- module hygiene ----------------------------------------------------------------


def test_linalg_has_no_numpy_no_to_dense_and_no_provider_branching():
    src = pathlib.Path(tenet.ops.linalg.__file__).read_text()
    assert "import numpy" not in src
    assert "np." not in src
    assert "to_dense(" not in src
    assert "if provider ==" not in src
    assert "isinstance(provider" not in src


def test_the_bond_leg_is_the_identity_mirror_convention():
    """Non-dual on both sides, differing only in ``side`` — see ``identity`` (#30)."""
    u, s, vh = tenet.linalg.svd(tensor("su2", seed=19), axes=SPLIT)
    assert u.legs[-1] == Leg(s.legs[0].space, IN)
    assert s.legs[0].dual is False and s.legs[1].dual is False
    assert vh.legs[0] == s.legs[0]
    # composable in the plain (space, dual, order) sense: side is not compared
    assert as_map(u).domain.matches(as_map(s).codomain) is None
    assert as_map(s).domain.matches(as_map(vh).codomain) is None
