"""Tests for ``eigh``, ``polar`` and ``lq`` — issue #63.

Extends ``tests/ops/test_linalg.py`` rather than forking it: the cases, the
spaces, the ``FusionOnly`` provider and the jit helpers are imported from there,
and every criterion is again stated on a *gauge-invariant* quantity —
reconstruction, isometry, eigenvalues, norms. Eigenvector phases and the sign of
``L``'s diagonal are the backend's mood, not this library's contract.

The square cases use the leg pattern ``(A OUT, B IN, A IN, B OUT)``: the split
``((0, 1), (2, 3))`` bends axes 1 and 2, and the two bends flip ``dual`` on both
sides in the same way, so the repartitioned map comes out square — a genuinely
crossing split that ``eigh`` accepts.
"""

import dataclasses
import sys

import autoray as ar
import numpy as np
import pytest
from helpers import sector_parity
from test_linalg import (
    F1,
    F2,
    LEGS,
    Q1,
    Q2,
    SPLIT,
    T_TRIV,
    V,
    W,
    capability_less,
    dense_matrix,
    tensor,
    to_jax,
    use_jax,
)

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import from_matrices, map_layout, to_matrices
from tenet.structure import TensorStructure
from tenet.symmetry import SU2, U1, CapabilityError, U1Sector

REPARTITION_MODULE = sys.modules["tenet.ops.repartition"]

PROVIDERS = ["trivial", "u1", "su2", "fz2"]

# (A OUT, B IN, A IN, B OUT): square after ``SPLIT``, which crosses axes 1 and 2.
SQUARE_LEGS = {
    "trivial": (
        Leg(T_TRIV[0], OUT, name="a"),
        Leg(T_TRIV[1], IN, name="b"),
        Leg(T_TRIV[0], IN, name="c"),
        Leg(T_TRIV[1], OUT, name="d"),
    ),
    "u1": (Leg(Q1, OUT, name="a"), Leg(Q2, IN, name="b"), Leg(Q1, IN), Leg(Q2, OUT)),
    "su2": (Leg(V, OUT, name="a"), Leg(W, IN, name="b"), Leg(V, IN), Leg(W, OUT)),
    "fz2": (Leg(F1, OUT, name="a"), Leg(F2, IN, name="b"), Leg(F1, IN), Leg(F2, OUT)),
}

# (A OUT, B OUT, A IN, B IN): square with no bend at all, hence no dual leg — the
# only shape ``to_dense`` can be asked about.
FLAT_LEGS = {
    name: (legs[0], dataclasses.replace(legs[3], name="b"), legs[2], dataclasses.replace(legs[1]))
    for name, legs in SQUARE_LEGS.items()
}


def hermitian(legs, seed: int = 0) -> SymmetricTensor:
    """A self-adjoint map: ``(B_c + B_c†)/2`` per coupled sector, put straight back.

    Built through ``to_matrices``/``from_matrices`` because ``adjoint`` returns a
    tensor with the *sides* swapped in place, which is a different structure and
    therefore not addable.
    """
    m = tenet.repartition(SymmetricTensor.random(legs, seed=seed), *SPLIT)
    return from_matrices(m.structure, {c: (b + b.conj().T) / 2 for c, b in to_matrices(m).items()})


def interleaved(h: SymmetricTensor) -> SymmetricTensor:
    """``h`` pushed back to ``SQUARE_LEGS``' own sides and axis order.

    ``eigh(interleaved(h), axes=SPLIT)`` then factorizes ``h`` again, through a
    split that really bends.
    """
    return tenet.repartition(h, (0, 3), (1, 2)).transpose((0, 2, 3, 1))


def with_domain_names(rec: SymmetricTensor, m: SymmetricTensor) -> SymmetricTensor:
    """``V @ W @ V†``'s IN legs renamed to ``m``'s domain names.

    ``V``'s OUT legs carry the *codomain*'s names, so the adjoint's IN legs do too;
    ``Leg.name`` is part of ``TensorStructure`` equality, so the comparison needs
    the normalization. Names never enter a fusion tree, so the blocks are reused
    as they are.
    """
    k = len(m.codomain)
    legs = (
        *rec.legs[:k],
        *(
            dataclasses.replace(leg, name=want.name)
            for leg, want in zip(rec.legs[k:], m.domain, strict=True)
        ),
    )
    return SymmetricTensor(TensorStructure(legs), rec.blocks)


def eigenvalues(w: SymmetricTensor) -> dict:
    """``{c: w_c}`` read off the diagonal operator, the way ``singular_values`` does."""
    return {c: np.asarray(ar.do("diagonal", mat)) for c, mat in to_matrices(w).items()}


# --- eigh: legs and the bond space -------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_eigh_factor_legs_are_bond_bond_and_left_bond(name):
    h = hermitian(SQUARE_LEGS[name])
    m = tenet.repartition(interleaved(h), *SPLIT)
    w, v = tenet.linalg.eigh(interleaved(h), axes=SPLIT)

    bond = v.legs[-1].space
    assert w.legs == (Leg(bond, OUT), Leg(bond, IN))
    assert v.legs == (*m.codomain, Leg(bond, IN))
    for got, want in zip(v.legs[:-1], m.codomain, strict=True):
        assert (got.space, got.dual, got.name) == (want.space, want.dual, want.name)
    # #57's identity-mirror convention: non-dual on both sides, differing only in side
    assert w.legs[0].dual is False and w.legs[1].dual is False
    assert v.legs[-1] == Leg(w.legs[0].space, IN)


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_eigh_bond_space_is_the_fused_domain_and_min_is_a_no_op(name):
    h = hermitian(SQUARE_LEGS[name])
    layout = map_layout(h.structure)
    w, _ = tenet.linalg.eigh(h)
    bond = w.legs[0].space
    for c in layout.sectors:
        rows, cols = layout.shape(c)
        assert rows == cols  # the square map makes ``min`` vacuous — asserted, not assumed
        assert bond.degeneracy(c) == rows
    assert tuple(bond) == layout.sectors


# --- eigh: reconstruction and isometry ----------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_eigh_reconstructs_after_normalizing_the_domain_names(name):
    h = hermitian(SQUARE_LEGS[name], seed=1)
    m = tenet.repartition(interleaved(h), *SPLIT)
    w, v = tenet.linalg.eigh(interleaved(h), axes=SPLIT)
    rec = with_domain_names(v @ w @ tenet.adjoint(v), m)
    assert rec.structure == m.structure
    assert tenet.allclose(rec, m, rtol=0, atol=1e-12)


def test_the_unnormalized_eigh_reconstruction_refuses_on_names_alone():
    """Pinned, not incidental: ``eigh`` never invents a naming convention."""
    h = hermitian(SQUARE_LEGS["su2"], seed=1)
    w, v = tenet.linalg.eigh(h)
    rec = v @ w @ tenet.adjoint(v)
    assert rec.structure != h.structure
    assert not tenet.allclose(rec, h, rtol=0, atol=1e-12)
    # spaces and dualities do agree — only ``name`` differs, and only on the domain
    assert [(leg.space, leg.dual, leg.side) for leg in rec.legs] == [
        (leg.space, leg.dual, leg.side) for leg in h.legs
    ]
    assert [leg.name for leg in rec.legs[2:]] == [leg.name for leg in h.legs[:2]]
    assert tenet.allclose(with_domain_names(rec, h), h, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", PROVIDERS)
def test_eigh_eigenvectors_are_an_isometry(name):
    h = hermitian(SQUARE_LEGS[name], seed=2)
    _, v = tenet.linalg.eigh(interleaved(h), axes=SPLIT)
    bond = tenet.identity((Leg(v.legs[-1].space, OUT),))
    assert tenet.allclose(tenet.adjoint(v) @ v, bond, rtol=0, atol=1e-12)


# --- eigh: the eigenvalues themselves ------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_w_is_diagonal_real_and_ascending_within_each_sector(name):
    """Ascending is LAPACK's order, kept — the deliberate contrast with ``svd``'s ``S``."""
    h = hermitian(SQUARE_LEGS[name], seed=3)
    w, _ = tenet.linalg.eigh(h)
    assert not np.issubdtype(np.asarray(w.blocks[0]).dtype, np.complexfloating)
    for c, mat in to_matrices(w).items():
        mat = np.asarray(mat)
        values = np.diagonal(mat)
        np.testing.assert_allclose(mat - np.diag(values), 0.0, atol=1e-14, err_msg=repr(c))
        assert np.all(np.diff(values) >= -1e-14), repr(c)

    _, s, _ = tenet.linalg.svd(h)
    for sigma in (np.diagonal(np.asarray(mat)) for mat in to_matrices(s).values()):
        assert np.all(np.diff(sigma) <= 1e-14)  # svd is descending; eigh is not re-sorted


def test_w_stays_real_for_a_complex_hermitian_tensor():
    h = hermitian(SQUARE_LEGS["su2"], seed=4)
    h = SymmetricTensor(h.structure, tuple(np.asarray(b, dtype=complex) for b in h.blocks))
    h = from_matrices(h.structure, {c: (b + b.conj().T) / 2 for c, b in to_matrices(h).items()})
    h = from_matrices(h.structure, {c: b + 0.3j * (b - b.T) for c, b in to_matrices(h).items()})
    w, v = tenet.linalg.eigh(h)
    assert np.issubdtype(np.asarray(v.blocks[0]).dtype, np.complexfloating)
    assert not np.issubdtype(np.asarray(w.blocks[0]).dtype, np.complexfloating)
    assert tenet.allclose(with_domain_names(v @ w @ tenet.adjoint(v), h), h, rtol=0, atol=1e-12)


def test_su2_eigh_norm_identity_carries_the_qdim_weight():
    h = hermitian(SQUARE_LEGS["su2"], seed=5)
    values = eigenvalues(tenet.linalg.eigh(h)[0])
    assert len({SU2.qdim(c) for c in values}) > 1  # the weight is not vacuous
    weighted = sum(SU2.qdim(c) * float(np.sum(v**2)) for c, v in values.items())
    assert float(tenet.norm(h)) ** 2 == pytest.approx(weighted, abs=1e-12)

    unweighted = sum(float(np.sum(v**2)) for v in values.values())
    assert abs(unweighted - weighted) > 0.1 * weighted


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_dense_oracle_eigenvalues(name):
    """``eigh``'s own ``W``, against ``eigvalsh`` of the densified matrix.

    ``FLAT_LEGS`` needs no bend, so no dual leg enters the dense expansion.
    """
    h = hermitian(FLAT_LEGS[name], seed=6)
    provider = h.provider
    got = np.sort(
        np.concatenate(
            [
                np.repeat(v, provider.irrep_dim(c))
                for c, v in eigenvalues(tenet.linalg.eigh(h)[0]).items()
            ]
        )
    )
    want = np.sort(np.linalg.eigvalsh(dense_matrix(h)))
    assert len(got) == len(want)
    np.testing.assert_allclose(got, want, atol=1e-10)


# --- eigh: the squareness refusal, and the hermiticity check we do not do -------------


def test_a_non_square_split_is_refused_naming_both_axes_and_both_legs():
    t = tensor("u1")  # ((0, 1), (2, 3)) leaves domain and codomain genuinely different
    with pytest.raises(ValueError) as excinfo:
        tenet.linalg.eigh(t, axes=SPLIT)
    message = str(excinfo.value)
    assert "eigh" in message
    assert "position 0" in message
    assert "public axis 0" in message and "public axis 2" in message
    assert "(space, dual)" in message and "in the same order" in message
    assert "dimensions alone are never enough" in message


def test_equal_dimensions_with_reversed_charges_are_still_refused():
    """The whole point of comparing spaces rather than sizes."""
    up = GradedSpace.new(U1, {U1Sector(1): 2, U1Sector(-1): 3})
    down = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(1): 3})
    assert up.dim == down.dim
    t = SymmetricTensor.random((Leg(up, OUT, name="a"), Leg(down, IN, name="b")), seed=7)
    with pytest.raises(ValueError, match="eigh: the map is not square at position 0"):
        tenet.linalg.eigh(t)


def test_a_non_hermitian_input_is_not_refused():
    """Documented behavior, not an oversight: no tolerance lives in this module.

    A tolerance comparison is a data-dependent branch and cannot run in a traced
    region (invariant 9). The user-side check is two lines with what exists.
    """
    t = SymmetricTensor.random(FLAT_LEGS["u1"], seed=8)
    mats = to_matrices(tenet.repartition(t, *SPLIT))
    asymmetry = max(float(np.linalg.norm(b - b.conj().T)) for b in mats.values())
    assert asymmetry > 1e-3  # genuinely not Hermitian

    w, v = tenet.linalg.eigh(t, axes=SPLIT)  # accepted, and it returns something
    assert np.all(np.isfinite(np.asarray(w.blocks[0])))
    # ...which is the backend's one-triangle read, not a factorization of ``t``
    m = tenet.repartition(t, *SPLIT)
    assert not tenet.allclose(with_domain_names(v @ w @ tenet.adjoint(v), m), m, atol=1e-8)
    # and it *is* the factorization of the Hermitian part the backend actually read
    lower = from_matrices(m.structure, {c: np.tril(b) + np.tril(b, -1).T for c, b in mats.items()})
    w2, v2 = tenet.linalg.eigh(lower)
    np.testing.assert_allclose(
        np.concatenate([eigenvalues(w)[c] for c in sorted(eigenvalues(w))]),
        np.concatenate([eigenvalues(w2)[c] for c in sorted(eigenvalues(w2))]),
        atol=1e-10,
    )
    assert v2 is not None


# --- polar ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_polar_factor_legs(side):
    t = tensor("su2", seed=9)
    m = tenet.repartition(t, *SPLIT)
    w, p = tenet.linalg.polar(t, axes=SPLIT, side=side)

    assert w.structure == m.structure  # no bond leg survives: ``polar`` never reshapes
    mirrored = m.domain if side == "left" else m.codomain
    assert p.legs == (
        *(dataclasses.replace(leg, side=OUT) for leg in mirrored),
        *(dataclasses.replace(leg, side=IN) for leg in mirrored),
    )


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_polar_reconstructs_the_repartitioned_tensor(name, side):
    t = tensor(name, seed=10)
    w, p = tenet.linalg.polar(t, axes=SPLIT, side=side)
    product = w @ p if side == "left" else p @ w
    assert tenet.allclose(product, tenet.repartition(t, *SPLIT), rtol=0, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
def test_polar_p_is_self_adjoint_and_positive_semi_definite(side):
    t = tensor("su2", seed=11)
    _, p = tenet.linalg.polar(t, axes=SPLIT, side=side)
    for c, mat in to_matrices(p).items():
        np.testing.assert_allclose(mat, np.asarray(mat).conj().T, atol=1e-12, err_msg=repr(c))
    for c, values in eigenvalues(tenet.linalg.eigh(p)[0]).items():
        assert np.all(values >= -1e-12), repr(c)
    # the adjoint reconstructs ``P`` once its axes are put back in public order
    back = tenet.adjoint(p).transpose((2, 3, 0, 1))
    assert tenet.allclose(back, p, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", PROVIDERS)
def test_polar_w_is_an_isometry_on_a_full_rank_tensor(name):
    """Square per sector, so ``W`` is unitary rather than merely one-sided.

    On a *rectangular* map only the narrow side's identity can hold — ``SPLIT`` on
    ``LEGS`` is row-limited in some sectors and column-limited in others by design
    (#57), which is exactly the partial-isometry statement below.
    """
    t = SymmetricTensor.random(SQUARE_LEGS[name], seed=12)
    m = tenet.repartition(t, *SPLIT)
    for mat in to_matrices(m).values():
        rows, cols = mat.shape
        assert rows == cols == np.linalg.matrix_rank(mat)  # full rank, or the test is vacuous

    w, _ = tenet.linalg.polar(t, axes=SPLIT)
    assert tenet.allclose(tenet.adjoint(w) @ w, tenet.identity(m.domain), rtol=0, atol=1e-12)
    assert tenet.allclose(w @ tenet.adjoint(w), tenet.identity(m.codomain), rtol=0, atol=1e-12)


def test_polar_w_is_only_a_partial_isometry_when_a_sector_is_rank_deficient():
    """``W†W`` is an orthogonal projector: idempotent, trace == ``W``'s own rank.

    The rank of ``B_c`` is *data*, and it shows up in ``P``, not in ``W``: the
    compact SVD hands back orthonormal ``u`` and ``vh`` whatever the singular
    values are, so ``W = u vh`` keeps rank ``min(rows, cols)`` and the deficiency
    surfaces as a zero eigenvalue of ``P`` — ``P`` singular, ``W`` a projector onto
    a proper subspace of the domain in every column-limited sector.
    """
    t = tensor("su2", seed=13)
    m = tenet.repartition(t, *SPLIT)
    mats = dict(to_matrices(m))
    victim = next(c for c, b in mats.items() if b.shape[0] <= b.shape[1])
    mats[victim] = np.asarray(mats[victim]).copy()
    mats[victim][0, :] = 0.0
    assert np.linalg.matrix_rank(mats[victim], tol=1e-10) < min(mats[victim].shape)
    deficient = from_matrices(m.structure, mats)

    w, p = tenet.linalg.polar(deficient)
    assert tenet.allclose(w @ p, deficient, rtol=0, atol=1e-12)
    assert not tenet.allclose(tenet.adjoint(w) @ w, tenet.identity(m.domain), rtol=0, atol=1e-12)
    for c, proj in to_matrices(tenet.adjoint(w) @ w).items():
        proj = np.asarray(proj)
        np.testing.assert_allclose(proj @ proj, proj, atol=1e-12, err_msg=repr(c))
        rank = min(np.asarray(mats[c]).shape)
        assert float(np.trace(proj)) == pytest.approx(rank, abs=1e-10)
        assert np.linalg.matrix_rank(np.asarray(to_matrices(w)[c]), tol=1e-10) == rank

    # the deficiency lands in ``P``: rank(P_c) == rank(B_c), sector by sector
    for c, mat in to_matrices(p).items():
        assert np.linalg.matrix_rank(np.asarray(mat), tol=1e-10) == np.linalg.matrix_rank(
            np.asarray(mats[c]), tol=1e-10
        )
    assert min(np.abs(eigenvalues(tenet.linalg.eigh(p)[0])[victim])) == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("side", ["left", "right"])
def test_polar_agrees_with_the_public_ops_oracle(side):
    """``W = U @ Vh``, ``P = Vh† @ S @ Vh`` — the same mathematics, three round trips."""
    t = tensor("su2", seed=14)
    u, s, vh = tenet.linalg.svd(t, axes=SPLIT)
    want_w = u @ vh
    want_p = tenet.adjoint(vh) @ s @ vh if side == "left" else u @ s @ tenet.adjoint(u)

    w, p = tenet.linalg.polar(t, axes=SPLIT, side=side)
    assert tenet.allclose(w, want_w, rtol=0, atol=1e-12)
    assert tenet.allclose(p, want_p, rtol=0, atol=1e-12)


# --- lq ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_lq_factor_legs_share_qrs_bond_space(name):
    t = tensor(name, seed=15)
    m = tenet.repartition(t, *SPLIT)
    ell, q = tenet.linalg.lq(t, axes=SPLIT)
    bond = ell.legs[-1].space
    assert ell.legs == (*m.codomain, Leg(bond, IN))
    assert q.legs == (Leg(bond, OUT), *m.domain)
    assert bond == tenet.linalg.qr(t, axes=SPLIT)[0].legs[-1].space


@pytest.mark.parametrize("name", PROVIDERS)
def test_lq_reconstructs_and_q_is_a_co_isometry(name):
    t = tensor(name, seed=16)
    ell, q = tenet.linalg.lq(t, axes=SPLIT)
    assert tenet.allclose(ell @ q, tenet.repartition(t, *SPLIT), rtol=0, atol=1e-12)
    bond = tenet.identity((Leg(q.legs[0].space, OUT),))
    assert tenet.allclose(q @ tenet.adjoint(q), bond, rtol=0, atol=1e-12)


# --- fZ2: the crossing split and the round trip ----------------------------------------


def test_fz2_split_crosses_odd_legs_for_all_three_and_the_signs_survive():
    t = tensor("fz2", seed=17)
    assert all(any(sector_parity(a) for a in leg.space) for leg in t.legs[1:3])
    m = tenet.repartition(t, *SPLIT)

    for side in ("left", "right"):
        w, p = tenet.linalg.polar(t, axes=SPLIT, side=side)
        product = w @ p if side == "left" else p @ w
        assert tenet.allclose(product, m, rtol=0, atol=1e-12)
        back = tenet.repartition(product, (0, 2), (1, 3)).transpose((0, 2, 1, 3))
        assert back.legs == t.legs
        assert tenet.allclose(back, t, rtol=0, atol=1e-12)

    ell, q = tenet.linalg.lq(t, axes=SPLIT)
    assert tenet.allclose(ell @ q, m, rtol=0, atol=1e-12)
    assert tenet.allclose(
        q @ tenet.adjoint(q), tenet.identity((Leg(q.legs[0].space, OUT),)), rtol=0, atol=1e-12
    )
    back = tenet.repartition(ell @ q, (0, 2), (1, 3)).transpose((0, 2, 1, 3))
    assert tenet.allclose(back, t, rtol=0, atol=1e-12)

    h = hermitian(SQUARE_LEGS["fz2"], seed=18)
    ti = interleaved(h)
    assert all(any(sector_parity(a) for a in leg.space) for leg in ti.legs[1:3])
    w, v = tenet.linalg.eigh(ti, axes=SPLIT)
    rec = with_domain_names(v @ w @ tenet.adjoint(v), h)
    assert tenet.allclose(rec, h, rtol=0, atol=1e-12)
    assert tenet.allclose(
        tenet.adjoint(v) @ v,
        tenet.identity((Leg(v.legs[-1].space, OUT),)),
        rtol=0,
        atol=1e-12,
    )
    back = tenet.repartition(rec, (0, 3), (1, 2)).transpose((0, 2, 3, 1))
    assert back.legs == ti.legs
    assert tenet.allclose(back, ti, rtol=0, atol=1e-12)


# --- the map-style spelling ---------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_as_map_spellings_match_the_axes_spelling_and_bend_nothing(name, monkeypatch):
    h = hermitian(FLAT_LEGS[name], seed=19)
    axes = (h.structure.out_axes, h.structure.in_axes)

    def no_bends(*args, **kwargs):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("as_map() decompositions must perform zero bends")

    monkeypatch.setattr(REPARTITION_MODULE, "bend", no_bends)
    pairs = [
        (h.as_map().eigh(), tenet.linalg.eigh(h, axes=axes)),
        (h.as_map().polar(), tenet.linalg.polar(h, axes=axes)),
        (h.as_map().polar(side="right"), tenet.linalg.polar(h, axes=axes, side="right")),
        (h.as_map().lq(), tenet.linalg.lq(h, axes=axes)),
    ]
    for view, want in pairs:
        for got, expected in zip(view, want, strict=True):
            assert got.structure == expected.structure
            assert tenet.allclose(got, expected, rtol=0, atol=1e-14)


# --- refusals and validation ---------------------------------------------------------


@pytest.mark.parametrize("fn", [tenet.linalg.eigh, tenet.linalg.polar, tenet.linalg.lq])
def test_a_crossing_split_is_refused_without_bending_coefficients(fn):
    with pytest.raises(CapabilityError) as excinfo:
        fn(capability_less(), axes=SPLIT)
    message = str(excinfo.value)
    assert "FusionOnly" in message
    assert "BendingCoefficients" in message
    assert "axis 1" in message


@pytest.mark.parametrize("fn", [tenet.linalg.eigh, tenet.linalg.polar, tenet.linalg.lq])
def test_a_non_crossing_split_needs_no_capability(fn):
    t = capability_less()
    axes = (t.structure.out_axes, t.structure.in_axes)
    want = tenet.repartition(t, *axes)
    if fn is tenet.linalg.eigh:
        want = from_matrices(
            want.structure, {c: (b + b.conj().T) / 2 for c, b in to_matrices(want).items()}
        )
        w, v = fn(want)
        product = with_domain_names(v @ w @ tenet.adjoint(v), want)
    else:
        a, b = fn(t, axes=axes)
        product = a @ b
    assert tenet.allclose(product, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fn", [tenet.linalg.eigh, tenet.linalg.polar, tenet.linalg.lq])
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
def test_axes_validation_is_the_house_style(fn, axes, match):
    with pytest.raises(ValueError, match=match):
        fn(tensor("u1"), axes=axes)


def test_polar_refuses_an_unknown_side():
    with pytest.raises(ValueError, match="side must be"):
        tenet.linalg.polar(tensor("u1"), axes=SPLIT, side="middle")


@pytest.mark.parametrize("name", ["eigh", "polar", "lq"])
def test_ar_do_still_raises_and_no_bare_attribute_is_created(name):
    """Invariant 11: the dispatch list stays closed until M8 settles a contract."""
    with pytest.raises(Exception):  # noqa: B017  the exception type is autoray's business
        ar.do(name, tensor("su2"))
    assert not hasattr(tenet, name)


# --- jit -------------------------------------------------------------------------------


def test_jit_eigh_traces_once_and_matches_the_numpy_backend():
    jax = use_jax()
    traces = []

    @jax.jit
    def factorize(x):
        traces.append(1)
        return tenet.linalg.eigh(x, axes=SPLIT)

    h = hermitian(FLAT_LEGS["su2"], seed=20)
    w, v = factorize(to_jax(h))
    factorize(to_jax(hermitian(FLAT_LEGS["su2"], seed=21)))
    assert len(traces) == 1  # same structure, same treedef: no retrace

    want_w, want_v = tenet.linalg.eigh(h, axes=SPLIT)
    assert w.structure == want_w.structure and v.structure == want_v.structure
    assert tenet.allclose(w, want_w, rtol=0, atol=1e-10)  # gauge-invariant
    assert tenet.allclose(with_domain_names(v @ w @ tenet.adjoint(v), h), h, rtol=0, atol=1e-10)


@pytest.mark.parametrize("side", ["left", "right"])
def test_jit_polar_traces_once_and_matches_the_numpy_backend(side):
    jax = use_jax()
    traces = []

    @jax.jit
    def factorize(x):
        traces.append(1)
        return tenet.linalg.polar(x, axes=SPLIT, side=side)

    t = tensor("su2", seed=22)
    w, p = factorize(to_jax(t))
    factorize(to_jax(tensor("su2", seed=23)))
    assert len(traces) == 1

    want_w, want_p = tenet.linalg.polar(t, axes=SPLIT, side=side)
    assert w.structure == want_w.structure and p.structure == want_p.structure
    assert tenet.allclose(w, want_w, rtol=0, atol=1e-10)
    assert tenet.allclose(p, want_p, rtol=0, atol=1e-10)
    product = w @ p if side == "left" else p @ w
    assert tenet.allclose(product, tenet.repartition(t, *SPLIT), rtol=0, atol=1e-10)


def test_jit_lq_traces_once_and_matches_the_numpy_backend():
    jax = use_jax()
    traces = []

    @jax.jit
    def factorize(x):
        traces.append(1)
        return tenet.linalg.lq(x, axes=SPLIT)

    t = tensor("su2", seed=24)
    ell, q = factorize(to_jax(t))
    factorize(to_jax(tensor("su2", seed=25)))
    assert len(traces) == 1

    want_l, _ = tenet.linalg.lq(t, axes=SPLIT)
    assert ell.structure == want_l.structure
    assert tenet.allclose(ell @ q, tenet.repartition(t, *SPLIT), rtol=0, atol=1e-10)
    bond = tenet.identity((Leg(q.legs[0].space, OUT),))
    assert tenet.allclose(q @ tenet.adjoint(q), bond, rtol=0, atol=1e-10)


def test_the_legs_used_here_really_are_the_test_linalg_ones():
    """The imports are load-bearing: these tests extend #57's cases, never fork them."""
    assert LEGS["su2"][0].space is V and LEGS["su2"][1].space is W
    assert SQUARE_LEGS["fz2"][0].space is F1 and SQUARE_LEGS["fz2"][1].space is F2
    assert U1 is Q1.provider
