"""The truncated Hermitian route — issue #205 (M40): ``eigh_truncated`` and ``eigh(bond=)``.

The discriminating criterion here is **the sign**. On a positive-definite input the
Hermitian route and the SVD route are the same operator, and
:func:`test_positive_definite_reproduces_svd_truncated` says so factor for factor on U(1),
fermionic parity and SU(2). On an *indefinite* one they are not, and
:func:`test_the_svd_route_cannot_reconstruct_an_indefinite_operator` is the test that fails
on the SVD route and passes on this one — the whole reason #205 exists.

The other axis is the ordering. ``svd_truncated`` slices a prefix because singular values
come back descending; eigenvalues come back **ascending and signed**, so the kept set is a
gather. Every assertion about which eigenvalues survived is checked against dense
``numpy.linalg.eigh`` with the same kept set, never against the implementation's own idea
of the order.
"""

import numpy as np
import pytest
from test_linalg import PROVIDERS, SPLIT, tensor, to_jax, use_jax

import tenet
from tenet import GradedSpace, StructureChangingError, SymmetricTensor
from tenet.map_view import to_matrices
from tenet.symmetry import CapabilityError

AXES = ((0, 1), (2, 3))


def hermitian(name: str, seed: int = 5) -> SymmetricTensor:
    """A random **indefinite** self-adjoint endomorphism, ``(x + x†) / 2``.

    Built from a random tensor on the leg pattern of ``m @ adjoint(m)`` rather than from
    ``m @ adjoint(m)`` itself: that product is rank-deficient in any sector where the
    codomain outgrows the domain, and a wall of exact zeros makes the *selection*
    ambiguous — which is a fine thing to document but a terrible thing to test a keep
    order against. A symmetrized random endomorphism is full rank, non-degenerate and
    indefinite already, with no shift to tune.
    """
    m = tenet.repartition(tensor(name), *SPLIT)
    pattern = (m @ tenet.adjoint(m)).legs  # the legs of a square endomorphism
    x = SymmetricTensor.random(pattern, seed=seed)
    return (x + tenet.transpose(tenet.adjoint(x), (2, 3, 0, 1))) / 2


def positive(name: str, seed: int = 5) -> SymmetricTensor:
    """``h @ h``: positive definite, so ``|w| == sigma`` and the two routes must agree."""
    h = tenet.repartition(hermitian(name, seed), *AXES)
    return h @ h


def full_bond(h) -> int:
    """The untruncated *dense* bond dimension ``Sum_c qdim(c) m_c`` of ``eigh(h, AXES)``."""
    space = tenet.linalg.eigh(h, AXES)[0].structure.legs[0].space
    return int(sum(h.provider.qdim(c) * m for c, m in space.sectors))


def dense_eigh(h):
    """``{c: (w ascending, V)}`` from NumPy, not from tenet."""
    return {
        c: np.linalg.eigh(np.asarray(b))
        for c, b in to_matrices(tenet.repartition(h, *AXES)).items()
    }


def diagonal(w) -> dict:
    return {c: np.asarray(np.diagonal(np.asarray(m))) for c, m in to_matrices(w).items()}


# --- the pairing with svd_truncated ------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_positive_definite_reproduces_svd_truncated(name):
    """``m @ adjoint(m)`` is positive, so ``|w| == sigma`` and the two routes agree."""
    h = positive(name)
    w, v = tenet.linalg.eigh_truncated(h, AXES, max_bond=4)
    u, s, _ = tenet.linalg.svd_truncated(h, AXES, max_bond=4)

    assert w.structure.legs[0].space == s.structure.legs[0].space
    assert v.structure == u.structure
    for c, values in diagonal(w).items():
        assert np.all(values > 0)  # positive: no sign to lose
        assert sorted(values, reverse=True) == pytest.approx(sorted(diagonal(s)[c], reverse=True))
    # and the same subspace, gauge and all: the projectors coincide
    assert bool(tenet.allclose(v @ tenet.adjoint(v), u @ tenet.adjoint(u), atol=1e-10))


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_bond_reproduces_the_one_call_form(name):
    """``eigh(t, axes, bond=B)`` is ``eigh_truncated``'s factorization projected."""
    h = hermitian(name)
    w0, v0 = tenet.linalg.eigh_truncated(h, AXES, max_bond=4)
    w1, v1 = tenet.linalg.eigh(h, AXES, bond=w0.structure.legs[0].space)

    assert w1.structure == w0.structure and v1.structure == v0.structure
    for a, b in ((w0, w1), (v0, v1)):
        for x, y in zip(a.blocks, b.blocks, strict=True):
            assert np.allclose(np.asarray(x), np.asarray(y), atol=1e-12)


# --- the sign, which is the point ---------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_signs_are_kept_and_match_dense_numpy(name):
    h = hermitian(name)
    w, _ = tenet.linalg.eigh_truncated(h, AXES, max_bond=full_bond(h) // 2)
    reference = dense_eigh(h)
    negatives = 0
    for c, values in diagonal(w).items():
        available = reference[c][0]
        # every kept value is one of the sector's eigenvalues, sign included
        for value in values:
            assert np.min(np.abs(available - value)) < 1e-10
        negatives += int(np.sum(values < 0))
        # descending in |w|, which is the documented order
        assert list(np.abs(values)) == pytest.approx(sorted(np.abs(values), reverse=True))
    assert negatives > 0, "the fixture is not indefinite; the test would prove nothing"


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_svd_route_cannot_reconstruct_an_indefinite_operator(name):
    """The failing-on-the-other-route test. Same kept dimension, same operator.

    ``V @ W @ adjoint(V)`` reconstructs the truncated operator with its signs; the SVD's
    ``U @ S @ adjoint(U)`` — a single isometry used on both index groups — replaces every
    negative eigenvalue with ``+|w|`` and is wrong by twice their sum.
    """
    h = hermitian(name)
    m = tenet.repartition(h, *AXES)
    full = full_bond(h)

    # keep *everything*, so truncation contributes nothing and only the signs can differ
    w, v = tenet.linalg.eigh_truncated(h, AXES, max_bond=full)
    u, s, _ = tenet.linalg.svd_truncated(h, AXES, max_bond=full)
    assert u.structure.legs[-1].space == v.structure.legs[-1].space

    assert float(tenet.norm(v @ w @ tenet.adjoint(v) - m)) < 1e-10  # exact, signs and all
    svd_error = float(tenet.norm(u @ s @ tenet.adjoint(u) - m))
    assert svd_error > 1e-3  # the SVD route is not even close

    # and the error is exactly twice the weight of the eigenvalues whose sign it dropped
    negatives = [
        (c, float(value)) for c, values in diagonal(w).items() for value in values if value < 0
    ]
    assert negatives, "the fixture is not indefinite; the test would prove nothing"
    assert svd_error**2 == pytest.approx(
        sum(h.provider.qdim(c) * (2 * value) ** 2 for c, value in negatives), rel=1e-8
    )

    # truncated to half the bond, where negatives still survive the cut, the Hermitian
    # route is strictly better -- the defect is not an artefact of keeping everything
    w, v = tenet.linalg.eigh_truncated(h, AXES, max_bond=full // 2)
    u, s, _ = tenet.linalg.svd_truncated(h, AXES, max_bond=full // 2)
    assert any(value < 0 for values in diagonal(w).values() for value in values)
    assert float(tenet.norm(u @ s @ tenet.adjoint(u) - m)) > float(
        tenet.norm(v @ w @ tenet.adjoint(v) - m)
    )


def test_reconstruction_matches_dense_numpy_with_the_same_kept_set():
    """The oracle is `numpy.linalg.eigh` truncated by hand to the same indices."""
    name = "su2"
    h = hermitian(name)
    w, v = tenet.linalg.eigh_truncated(h, AXES, max_bond=4)
    got = to_matrices(tenet.repartition(v @ w @ tenet.adjoint(v), *AXES))

    reference = dense_eigh(h)
    for c, values in diagonal(w).items():
        wc, vc = reference[c]
        idx = [int(np.argmin(np.abs(wc - value))) for value in values]
        want = (vc[:, idx] * wc[idx]) @ vc[:, idx].conj().T
        assert np.allclose(np.asarray(got[c]), want, atol=1e-10)


# --- the truncation rule is svd_truncated's, shared not copied ---------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_max_bond_bounds_the_dense_dimension(name):
    h = hermitian(name)
    w, _ = tenet.linalg.eigh_truncated(h, AXES, max_bond=5)
    bond = w.structure.legs[0].space
    dense = sum(h.provider.qdim(c) * m for c, m in bond.sectors)
    assert dense <= 5
    assert 5 - dense <= max(h.provider.qdim(c) for c in bond) - 1  # the documented undershoot


def test_the_selection_object_is_the_shared_one():
    """#209's `BondSelection` is what decides here — one keep rule, not two."""
    h = hermitian("su2")
    w, _ = tenet.linalg.eigh_truncated(h, AXES, max_bond=6)
    # the same rule fed the same magnitudes gives the same bond
    magnitudes = {
        c: np.abs(np.linalg.eigvalsh(np.asarray(b)))
        for c, b in to_matrices(tenet.repartition(h, *AXES)).items()
    }
    from tenet.ops.linalg import _decide, _spectrum  # noqa: PLC0415

    selection = _decide(
        _spectrum(magnitudes, "eigh_truncated"), h.provider, 6, None, "rsum2", False, "x"
    )
    assert selection.bond == w.structure.legs[0].space


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_renorm_preserves_the_norm(name):
    h = hermitian(name)
    w, v = tenet.linalg.eigh_truncated(h, AXES, max_bond=4, renorm=True)
    assert float(tenet.norm(w)) == pytest.approx(
        float(tenet.norm(tenet.repartition(h, *AXES))), rel=1e-10
    )


# --- refusals ----------------------------------------------------------------------


def test_a_non_square_map_is_refused_naming_eigh_truncated():
    with pytest.raises(ValueError, match="eigh_truncated: the map is not square"):
        tenet.linalg.eigh_truncated(tensor("u1"), SPLIT, max_bond=2)


def test_argument_refusals_name_eigh_truncated():
    h = hermitian("u1")
    with pytest.raises(ValueError, match="eigh_truncated needs at least one"):
        tenet.linalg.eigh_truncated(h, AXES)
    with pytest.raises(ValueError, match="max_bond must be a positive"):
        tenet.linalg.eigh_truncated(h, AXES, max_bond=0)
    with pytest.raises(ValueError, match="unknown cutoff_mode"):
        tenet.linalg.eigh_truncated(h, AXES, cutoff=1e-9, cutoff_mode="nope")
    with pytest.raises(TypeError, match="renorm must be a bool"):
        tenet.linalg.eigh_truncated(h, AXES, max_bond=2, renorm=1)
    with pytest.raises(ValueError, match="eigh_truncated: cutoff="):
        tenet.linalg.eigh_truncated(h, AXES, cutoff=1e6, cutoff_mode="abs")


def test_bond_refuses_a_non_subspace_structurally():
    """``_keep_counts``' existing message, before a single block is diagonalized."""
    h = hermitian("u1")
    _, untruncated, _ = __import__("tenet.ops.linalg", fromlist=["_lower"])._lower(h, AXES)
    too_big = GradedSpace.new(h.provider, {c: m + 1 for c, m in untruncated.sectors})
    with pytest.raises(ValueError, match="bond asks for"):
        tenet.linalg.eigh(h, AXES, bond=too_big)


def test_a_capability_less_provider_is_refused():
    from test_linalg import capability_less  # noqa: PLC0415

    t = capability_less()
    with pytest.raises(CapabilityError):
        tenet.linalg.eigh_truncated(t, axes=SPLIT, max_bond=2)


# --- the trace boundary, both sides ------------------------------------------------


def test_jit_refuses_eigh_truncated_and_says_why():
    jax = use_jax()
    h = to_jax(hermitian("su2"))
    with pytest.raises(StructureChangingError) as excinfo:
        jax.jit(lambda x: tenet.linalg.eigh_truncated(x, AXES, max_bond=4)[0])(h)
    message = str(excinfo.value)
    assert "eigh_truncated" in message
    assert "depend on the block values" in message
    assert "outside the traced region" in message


def test_grad_refuses_eigh_truncated_too():
    jax = use_jax()
    h = to_jax(hermitian("su2"))

    def scalar(x):
        w, v = tenet.linalg.eigh_truncated(x, AXES, max_bond=4)
        return tenet.norm(v @ w @ tenet.adjoint(v))

    with pytest.raises(StructureChangingError, match="eigh_truncated"):
        jax.grad(scalar)(h)


def test_eigh_with_a_bond_is_jittable_right_next_to_the_refusal():
    jax = use_jax()
    h = hermitian("su2")
    bond = tenet.linalg.eigh_truncated(h, AXES, max_bond=4)[0].structure.legs[0].space
    w = jax.jit(lambda x: tenet.linalg.eigh(x, AXES, bond=bond)[0])(to_jax(h))
    assert isinstance(w, SymmetricTensor)
    assert w.backend == "jax"
    assert w.structure.legs[0].space == bond


def test_eigh_with_a_bond_is_differentiable():
    """The gather is a permutation, not a shape decision, so ``grad`` goes through it."""
    jax = use_jax()
    h = hermitian("su2")
    bond = tenet.linalg.eigh_truncated(h, AXES, max_bond=4)[0].structure.legs[0].space
    from tenet import ad  # noqa: PLC0415  # importing it registers nothing; install() does

    ad.install()
    try:

        def scalar(x):
            w, v = tenet.linalg.eigh(x, AXES, bond=bond)
            return tenet.norm(v @ w @ tenet.adjoint(v))

        g = jax.grad(scalar)(to_jax(h))
        blocks = [np.asarray(b) for b in g.blocks]
        assert all(np.all(np.isfinite(b)) for b in blocks)
        assert max(float(np.max(np.abs(b))) for b in blocks) > 0
        # against central differences on one entry of one block
        primal = to_jax(h)
        i = (0,) * primal.blocks[0].ndim
        eps = 1e-6

        def bump(delta):

            b0 = primal.blocks[0].at[i].add(delta)
            return float(scalar(SymmetricTensor(primal.structure, (b0, *primal.blocks[1:]))))

        fd = (bump(eps) - bump(-eps)) / (2 * eps)
        assert float(blocks[0][i]) == pytest.approx(fd, abs=1e-5)
    finally:
        ad.uninstall()
