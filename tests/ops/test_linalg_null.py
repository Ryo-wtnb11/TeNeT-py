"""Tests for ``left_null`` and ``right_null`` — issue #88.

Extends ``tests/ops/test_linalg.py`` rather than forking it: the spaces, the
``SPLIT`` partition and the jit helpers are imported from there.

Every criterion is stated on a **gauge-invariant** quantity — ``N† N``, ``N† T``,
the projector ``N N†``, the span identity — because ``N`` is only defined up to a
unitary on the bond: its basis is whatever LAPACK's Householder sequence
produces, which is the backend's mood and not this library's contract. The
gradient criterion uses a projector objective for the same reason; an elementwise
objective on ``N`` would be meaningless.

The fixture is a *tall* three-leg map ``(a OUT, b OUT, c IN)``: the ``test_linalg``
four-leg cases have ``rows_c == cols_c`` in every coupled sector for U(1) and fZ2,
i.e. an empty complement on both sides, which is the refusal and not the
computation. The same tensor read the other way round — ``axes=MIRROR`` — is the
wide map ``right_null`` wants, so one fixture serves both directions and the two
are literally the same factorization (asserted).
"""

import sys

import autoray as ar
import numpy as np
import pytest
from test_linalg import (
    F1,
    F2,
    Q1,
    Q2,
    SPLIT,
    T_TRIV,
    V,
    W,
    X,
    tensor,
    to_jax,
    use_jax,
)

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import map_layout, to_matrices

REPARTITION_MODULE = sys.modules["tenet.ops.repartition"]

PROVIDERS = ["trivial", "u1", "su2", "fz2"]

# (a OUT, b OUT, c IN): every coupled sector is genuinely tall, for all four
# providers, so `left_null` has a non-empty complement everywhere.
NULL_LEGS = {
    "trivial": (
        Leg(T_TRIV[0], OUT, name="a"),
        Leg(T_TRIV[1], OUT, name="b"),
        Leg(T_TRIV[1], IN, name="c"),
    ),
    "u1": (Leg(Q1, OUT, name="a"), Leg(Q2, OUT, name="b"), Leg(Q2, IN, name="c")),
    "su2": (Leg(V, OUT, name="a"), Leg(W, OUT, name="b"), Leg(X, IN, name="c")),
    "fz2": (Leg(F1, OUT, name="a"), Leg(F2, OUT, name="b"), Leg(F2, IN, name="c")),
}

TALL = ((0, 1), (2,))  # the current partition: what axes=None and as_map() use
MIRROR = ((2,), (0, 1))  # the same tensor as a wide map: right_null's side


def tall(name: str, seed: int = 0, dtype=np.float64) -> SymmetricTensor:
    return SymmetricTensor.random(NULL_LEGS[name], seed=seed, dtype=dtype)


def expected_bond(m: SymmetricTensor, *, left: bool) -> GradedSpace:
    """``{c: rows - cols}`` (or the mirror) computed independently from ``map_layout``."""
    layout = map_layout(m.structure)
    keep = {}
    for c in layout.sectors:
        rows, cols = layout.shape(c)
        extra = rows - cols if left else cols - rows
        if extra > 0:
            keep[c] = extra
    return GradedSpace.new(m.provider, keep)


def matrices(t: SymmetricTensor) -> dict:
    return {c: np.asarray(b) for c, b in to_matrices(t).items()}


# --- legs, conventions and the bond space ------------------------------------------


def test_exported_and_reachable_as_a_map_method():
    assert "left_null" in tenet.ops.linalg.__all__
    assert "right_null" in tenet.ops.linalg.__all__
    t = tall("su2")
    assert tenet.allclose(t.as_map().left_null(), tenet.linalg.left_null(t), rtol=0, atol=0)


@pytest.mark.parametrize("name", PROVIDERS)
def test_as_map_agrees_block_for_block_and_bends_nothing(name, monkeypatch):
    """``t.as_map().left_null()`` is ``left_null(t)`` at the current partition."""
    t = tall(name, seed=1)
    want = tenet.linalg.left_null(t, TALL)

    def no_bends(*args, **kwargs):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("as_map().left_null() must perform zero bends")

    monkeypatch.setattr(REPARTITION_MODULE, "bend", no_bends)
    got = t.as_map().left_null()
    assert got.structure == want.structure
    for a, b in zip(got.blocks, want.blocks, strict=True):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() == 0.0


def test_as_map_right_null_agrees_and_bends_nothing(monkeypatch):
    """The wide direction, on a tensor whose *current* partition is wide."""
    t = tenet.repartition(tall("su2", seed=1), *MIRROR)
    want = tenet.linalg.right_null(t)

    def no_bends(*args, **kwargs):  # pragma: no cover
        raise AssertionError("as_map().right_null() must perform zero bends")

    monkeypatch.setattr(REPARTITION_MODULE, "bend", no_bends)
    got = t.as_map().right_null()
    assert got.structure == want.structure
    for a, b in zip(got.blocks, want.blocks, strict=True):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() == 0.0


@pytest.mark.parametrize("name", PROVIDERS)
def test_left_null_legs_are_qrs_q_legs(name):
    t = tall(name)
    m = tenet.repartition(t, *TALL)
    n = tenet.linalg.left_null(t, TALL)
    bond = n.legs[-1].space
    assert n.legs == (*m.codomain, Leg(bond, IN))
    # the identity mirror convention: non-dual on both sides, differing only in side
    assert n.legs[-1].dual is False
    assert tenet.identity((Leg(bond, OUT),)).legs == (Leg(bond, OUT), Leg(bond, IN))


@pytest.mark.parametrize("name", PROVIDERS)
def test_right_null_legs_are_lqs_q_legs(name):
    t = tall(name)
    m = tenet.repartition(t, *MIRROR)
    n = tenet.linalg.right_null(t, MIRROR)
    bond = n.legs[0].space
    assert n.legs == (Leg(bond, OUT), *m.domain)
    assert n.legs[0].dual is False


@pytest.mark.parametrize("name", PROVIDERS)
def test_bond_space_is_exactly_rows_minus_cols(name):
    t = tall(name)
    left = tenet.linalg.left_null(t, TALL)
    right = tenet.linalg.right_null(t, MIRROR)
    assert left.legs[-1].space == expected_bond(tenet.repartition(t, *TALL), left=True)
    assert right.legs[0].space == expected_bond(tenet.repartition(t, *MIRROR), left=False)


def test_sectors_without_a_complement_are_absent_from_the_bond():
    """SU(2) with a mixed layout: two tall sectors, two wide ones, and the bond
    carries exactly the tall ones — decided from shapes, so it stays traceable."""
    t = tensor("su2", seed=4)
    m = tenet.repartition(t, *SPLIT)
    layout = map_layout(m.structure)
    tall_sectors = {c for c in layout.sectors if layout.shape(c)[0] > layout.shape(c)[1]}
    wide_sectors = {c for c in layout.sectors if layout.shape(c)[0] < layout.shape(c)[1]}
    assert tall_sectors and wide_sectors  # or the criterion is vacuous

    bond = tenet.linalg.left_null(t, SPLIT).legs[-1].space
    assert set(bond) == tall_sectors
    assert bond == expected_bond(m, left=True)
    mirror = tenet.linalg.right_null(t, SPLIT).legs[0].space
    assert set(mirror) == wide_sectors


# --- the two defining identities ---------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_left_null_is_an_isometry_annihilating_the_map(name):
    t = tall(name, seed=2)
    m = tenet.repartition(t, *TALL)
    n = tenet.linalg.left_null(t, TALL)
    bond = tenet.identity((Leg(n.legs[-1].space, OUT),))

    gram = tenet.adjoint(n) @ n
    assert gram.legs == bond.legs  # structural, exactly
    assert tenet.allclose(gram, bond, rtol=0, atol=1e-13)
    assert float(tenet.norm(tenet.adjoint(n) @ m)) <= 1e-12 * float(tenet.norm(t))


@pytest.mark.parametrize("name", PROVIDERS)
def test_right_null_is_a_co_isometry_annihilating_the_map(name):
    t = tall(name, seed=2)
    m = tenet.repartition(t, *MIRROR)
    n = tenet.linalg.right_null(t, MIRROR)
    bond = tenet.identity((Leg(n.legs[0].space, OUT),))

    gram = n @ tenet.adjoint(n)
    assert gram.legs == bond.legs
    assert tenet.allclose(gram, bond, rtol=0, atol=1e-13)
    assert float(tenet.norm(m @ tenet.adjoint(n))) <= 1e-12 * float(tenet.norm(t))


@pytest.mark.parametrize("name", PROVIDERS)
def test_span_is_complete_q_q_dagger_plus_n_n_dagger_is_the_identity(name):
    """The complement is *the whole* complement of the structural image.

    Stated per **coupled sector of the map**, which is the only place the
    statement has content: a sector the codomain carries but the map does not
    couple to has no rows in ``B_c`` at all, so neither ``Q`` nor ``N`` — nor,
    therefore, ``identity(codomain)``, which does carry it — can say anything
    about it.
    """
    t = tall(name, seed=3)
    m = tenet.repartition(t, *TALL)
    q, _ = tenet.linalg.qr(t, TALL)
    n = tenet.linalg.left_null(t, TALL)
    span = matrices(q @ tenet.adjoint(q) + n @ tenet.adjoint(n))
    layout = map_layout(m.structure)
    for c in layout.sectors:
        rows = layout.shape(c)[0]
        assert np.abs(span[c] - np.eye(rows)).max() <= 1e-12


def test_right_null_is_left_null_of_the_adjoint_and_not_a_second_implementation():
    """``N_right = (q[:, rows:])†`` with the *same* ``q`` ``left_null`` slices, so the
    two matrices agree to the last bit rather than merely up to a bond gauge."""
    t = tall("su2", seed=5)
    m = tenet.repartition(t, *MIRROR)
    right = matrices(tenet.linalg.right_null(t, MIRROR))
    left = matrices(tenet.linalg.left_null(tenet.adjoint(m)))
    assert set(right) == set(left)
    for c, mat in right.items():
        assert np.abs(mat - left[c].conj().T).max() == 0.0


# --- the shape-null stance, as an executable claim ----------------------------------


def zero_one_sector(t: SymmetricTensor) -> SymmetricTensor:
    """The blocks of the *first* coupled sector zeroed; every other block kept."""
    dead = t.structure.block_order[0].coupled
    return SymmetricTensor(
        t.structure,
        tuple(
            np.zeros_like(b) if key.coupled == dead else b
            for key, b in zip(t.structure.block_order, t.blocks, strict=True)
        ),
    )


def test_a_rank_deficient_sector_changes_no_structure_and_is_still_annihilated():
    """Structure is metadata, rank is data (``_lower``'s rule, complemented).

    The numerical null space of the zeroed sector is strictly *larger* than what
    comes back here — the structural complement is a subspace of it, never the
    whole of it. Reaching the rest needs a rank tolerance, i.e. the separate,
    non-traceable sibling named in ``left_null``'s docstring; it is deliberately
    not this function.
    """
    t = tall("su2", seed=6)
    degenerate = zero_one_sector(t)
    m = tenet.repartition(degenerate, *TALL)

    healthy = tenet.linalg.left_null(t, TALL).legs[-1].space
    got = tenet.linalg.left_null(degenerate, TALL)
    assert got.legs[-1].space == healthy
    assert float(tenet.norm(tenet.adjoint(got) @ m)) <= 1e-14


# --- refusal -------------------------------------------------------------------------


def test_an_empty_complement_is_a_value_error_naming_every_sector_shape():
    t = tall("u1", seed=7)
    with pytest.raises(ValueError) as excinfo:
        tenet.linalg.left_null(t, MIRROR)  # wide: no sector has rows > cols
    message = str(excinfo.value)
    assert "left_null" in message and "rows_c > cols_c" in message
    assert "tenet.linalg.right_null" in message  # the mirrored case is named
    layout = map_layout(tenet.repartition(t, *MIRROR).structure)
    for c in layout.sectors:
        rows, cols = layout.shape(c)
        assert f"{c!r}: (rows={rows}, cols={cols})" in message


def test_the_mirror_refusal_points_back_at_left_null():
    with pytest.raises(ValueError) as excinfo:
        tenet.linalg.right_null(tall("u1", seed=7), TALL)
    message = str(excinfo.value)
    assert "right_null" in message and "cols_c > rows_c" in message
    assert "tenet.linalg.left_null" in message


def test_the_refusal_reads_no_block_value_and_is_identical_inside_jit():
    jax = use_jax()
    t = tall("u1", seed=8)
    nan = SymmetricTensor(t.structure, tuple(np.full_like(b, np.nan) for b in t.blocks))

    with pytest.raises(ValueError) as eager:
        tenet.linalg.left_null(nan, MIRROR)
    with pytest.raises(ValueError) as traced:
        jax.jit(lambda x: tenet.norm(tenet.linalg.left_null(x, MIRROR)))(to_jax(t))
    assert str(traced.value) == str(eager.value)


# --- traceability and gradients ------------------------------------------------------


def test_jit_traces_once_per_structure_and_the_constraint_holds_under_it():
    """The traceable side of #64's line, for the reason #77 and #83 state: the bond
    space comes from ``map_layout``, i.e. metadata, never from a block value — so
    changing the values does not retrace, and no ``StructureChangingError`` applies."""
    jax = use_jax()
    traces = []

    @jax.jit
    def residual(x):
        traces.append(1)
        return tenet.norm(tenet.adjoint(tenet.linalg.left_null(x)) @ x)

    got = float(residual(to_jax(tall("su2", seed=9))))
    residual(to_jax(tall("su2", seed=10)))  # different values, same structure
    assert len(traces) == 1
    assert got <= 1e-12


def test_jit_returns_a_jax_backed_isometry():
    jax = use_jax()
    n = jax.jit(lambda x: tenet.linalg.left_null(x))(to_jax(tall("su2", seed=11)))
    assert isinstance(n, SymmetricTensor) and n.backend == "jax"


def weight(t: SymmetricTensor, seed: int = 13) -> SymmetricTensor:
    """A fixed random endomorphism of the codomain, for the projector objective."""
    codomain = tenet.repartition(t, *TALL).codomain
    return SymmetricTensor.random(tenet.identity(codomain).legs, seed=seed)


def projector_objective(x, w):
    """A gauge-invariant scalar: ``N`` enters only through ``N N†``.

    An elementwise objective on ``N`` itself would be meaningless — the
    complement's basis is gauge, so its entries are the backend's Householder
    sequence and nothing a gradient should be compared against.
    """
    n = tenet.linalg.left_null(x)
    return tenet.norm((n @ tenet.adjoint(n)) @ w)


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_gradient_of_a_projector_objective_matches_central_differences(name):
    jax = use_jax()
    t = tall(name, seed=12)
    w = weight(t)

    g = jax.grad(lambda x: projector_objective(x, w))(t.to_backend("jax"))
    assert [b.shape for b in g.blocks] == [b.shape for b in t.blocks]

    h = 1e-6
    for i, block in enumerate(t.blocks):
        for idx in np.ndindex(block.shape):

            def shifted(delta, i=i, idx=idx):
                blocks = [np.array(b, copy=True) for b in t.blocks]
                blocks[i][idx] += delta
                return float(projector_objective(SymmetricTensor(t.structure, blocks), w))

            fd = (shifted(h) - shifted(-h)) / (2 * h)
            assert abs(fd - float(np.asarray(g.blocks[i])[idx])) < 1e-6


def test_the_gradient_of_the_identically_zero_constraint_is_zero():
    """``‖N† T‖²`` is zero for every ``T``, so its gradient must be zero too — the
    statement that the constraint is *differentiably* respected."""
    jax = use_jax()
    t = to_jax(tall("su2", seed=14))
    g = jax.grad(lambda x: tenet.norm(tenet.adjoint(tenet.linalg.left_null(x)) @ x) ** 2)(t)
    assert max(float(np.abs(np.asarray(b)).max()) for b in g.blocks) <= 1e-12


def test_complete_qr_gradients_work_at_the_declared_jax_floor():
    """JAX 0.10.0's CHANGELOG: "We now support differentiation of
    ``jax.lax.linalg.qr`` for wide matrices **and when ``full_matrices`` is
    ``True``**." #80 bumped ``jax>=0.10`` for the first half; ``left_null``'s
    ``mode="complete"`` on a tall block consumes the second. No further bump."""
    jax = use_jax()
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((5, 3)))
    weight = jnp.asarray(rng.standard_normal((5, 5)))

    def scalar(x):
        q = ar.do("linalg.qr", x, mode="complete")[0]
        n = q[:, 3:]
        return jnp.sum(weight * (n @ n.T))  # the projector again: gauge-invariant

    g = np.asarray(jax.grad(scalar)(a))
    assert np.isfinite(g).all()
    h = 1e-6
    for idx in np.ndindex(a.shape):
        step = np.zeros(a.shape)
        step[idx] = h
        fd = (float(scalar(a + step)) - float(scalar(a - step))) / (2 * h)
        assert abs(fd - g[idx]) < 1e-6


# --- complex blocks ------------------------------------------------------------------


def complex_tensor(name="u1", seed=15):
    t = tall(name, seed=seed)
    return SymmetricTensor(
        t.structure, tuple((b * (1 + 2j)).astype(np.complex128) for b in t.blocks)
    )


def test_complex_blocks_satisfy_both_identities():
    t = complex_tensor()
    m = tenet.repartition(t, *TALL)
    n = tenet.linalg.left_null(t, TALL)
    assert n.dtype == np.complex128
    bond = tenet.identity((Leg(n.legs[-1].space, OUT),), dtype=np.complex128)
    assert tenet.allclose(tenet.adjoint(n) @ n, bond, rtol=0, atol=1e-12)
    assert float(tenet.norm(tenet.adjoint(n) @ m)) <= 1e-12 * float(tenet.norm(t))

    mm = tenet.repartition(t, *MIRROR)
    r = tenet.linalg.right_null(t, MIRROR)
    rbond = tenet.identity((Leg(r.legs[0].space, OUT),), dtype=np.complex128)
    assert tenet.allclose(r @ tenet.adjoint(r), rbond, rtol=0, atol=1e-12)
    assert float(tenet.norm(mm @ tenet.adjoint(r))) <= 1e-12 * float(tenet.norm(t))


def test_gradient_of_a_real_objective_through_a_complex_null_space():
    """The parameters stay real and the tensor is ``x + i·c``, so ``jax.grad``'s
    real-input convention applies and finite differences are comparable."""
    jax = use_jax()
    t = tall("u1", seed=16)
    imag = tall("u1", seed=17)
    w = weight(t)

    def scalar(x):
        n = tenet.linalg.left_null(x + imag.to_backend(x.backend) * 1j)
        return tenet.norm((n @ tenet.adjoint(n)) @ w)

    g = jax.grad(scalar)(t.to_backend("jax"))
    h = 1e-6
    for i, block in enumerate(t.blocks):
        for idx in np.ndindex(block.shape):
            blocks = [np.array(b, copy=True) for b in t.blocks]
            blocks[i][idx] += h
            up = float(scalar(SymmetricTensor(t.structure, blocks)))
            blocks[i][idx] -= 2 * h
            down = float(scalar(SymmetricTensor(t.structure, blocks)))
            assert abs((up - down) / (2 * h) - float(np.asarray(g.blocks[i])[idx].real)) < 1e-6


# --- tenet.ad is untouched ------------------------------------------------------------


def test_ad_names_unchanged_and_install_changes_nothing():
    """No ``1/(sigma_i - sigma_j)`` anywhere in a complete QR, so no broadening
    applies and ``tenet.ad._NAMES`` does not grow."""
    use_jax()
    import tenet.ad

    assert tenet.ad._NAMES == ("linalg.svd", "linalg.eigh")
    t = to_jax(tall("su2", seed=18))
    before = tenet.linalg.left_null(t)
    tenet.ad.install()
    try:
        after = tenet.linalg.left_null(t)
    finally:
        tenet.ad.uninstall()
    for a, b in zip(before.blocks, after.blocks, strict=True):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() == 0.0
