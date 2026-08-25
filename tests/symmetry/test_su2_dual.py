"""Tests for SU(2)'s ``DualBasis`` — one test per acceptance criterion of issue #37.

The point of the file: ``SU2.dual(a) == a`` and yet duality is *not* a no-op. The
whole content of it lives in ``z_matrix``, which is derived from the CG singlet and
checked here against three independent oracles — the closed form ``Z[i, dj-i] =
(-1)**i``, the generator relation ``Z J Z^-1 == -J.T`` (which never touches ``cgc``),
and the dense cup/cap round trip through ``to_dense``.
"""

import dataclasses
from math import sqrt

import numpy as np
import pytest
from test_su2 import spin_matrices  # same directory, pytest's prepend import mode

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.symmetry import SU2, CapabilityError, DualBasis, SU2Sector


# The doubled-spin entry points this file was written against; the pure-Python module is gone
# since #180 and the provider is the only coefficient surface left.
def cg_tensor(dj1: int, dj2: int, dj3: int) -> np.ndarray:
    return SU2.cgc(SU2Sector(dj1), SU2Sector(dj2), SU2Sector(dj3))[..., 0]


def keyed(legs, blocks):
    """Every block, keyed in ``block_order``."""
    return dict(zip(TensorStructure(tuple(legs)).block_order, blocks, strict=True))


def frobenius_schur(dj: int) -> int:
    return SU2.frobenius_schur(SU2Sector(dj))


def z_matrix(dj: int) -> np.ndarray:
    return SU2.z_matrix(SU2Sector(dj))


DJS = range(9)


def closed_form(dj: int) -> np.ndarray:
    """``Z[i, dj - i] = (-1)**i`` written out independently of the implementation."""
    z = np.zeros((dj + 1, dj + 1))
    for i in range(dj + 1):
        z[i, dj - i] = (-1) ** i
    return z


# --- the capability -------------------------------------------------------------


def test_su2_is_a_dual_basis_provider():
    assert isinstance(SU2, DualBasis)
    _: DualBasis = SU2  # static conformance


@pytest.mark.parametrize("dj", DJS)
def test_provider_shape_and_read_only(dj):
    z = SU2.z_matrix(SU2Sector(dj))
    assert z.shape == (dj + 1, dj + 1)
    assert not z.flags.writeable


def test_provider_rejects_a_foreign_sector():
    from tenet.symmetry import U1Sector

    with pytest.raises(ValueError):
        SU2.z_matrix(U1Sector(1))


def test_no_provider_field():
    """``z_matrix`` used to be a ``functools.cache``d module function and the identity
    was asserted here; since #180 it comes from racah, which owns its own caches, so
    what is left to pin is that the provider still carries no array field."""
    np.testing.assert_array_equal(z_matrix(3), SU2.z_matrix(SU2Sector(3)))
    assert not z_matrix(3).flags.writeable
    assert not dataclasses.fields(SU2)[1:]  # only ``name``
    assert hash(SU2) == hash(SU2)


# --- the explicit form ----------------------------------------------------------


@pytest.mark.parametrize("dj", DJS)
def test_explicit_antidiagonal_form_entrywise(dj):
    np.testing.assert_allclose(z_matrix(dj), closed_form(dj), rtol=0.0, atol=1e-14)


def test_singlet_is_the_abelian_one_by_one():
    np.testing.assert_array_equal(z_matrix(0), np.ones((1, 1)))


@pytest.mark.parametrize("dj", DJS)
def test_unitarity(dj):
    z = z_matrix(dj)
    eye = np.eye(dj + 1)
    np.testing.assert_allclose(z @ z.conj().T, eye, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(z.conj().T @ z, eye, rtol=0.0, atol=1e-14)


# --- Frobenius-Schur ------------------------------------------------------------


@pytest.mark.parametrize("dj", DJS)
def test_frobenius_schur_is_a_property_of_z(dj):
    assert frobenius_schur(dj) == (-1) ** dj
    np.testing.assert_allclose(z_matrix(dj).T, frobenius_schur(dj) * z_matrix(dj), atol=1e-14)


@pytest.mark.parametrize("dj", [1, 3, 5, 7])
def test_half_integer_z_is_antisymmetric(dj):
    """The case that separates the two candidate conventions: Z.T == -Z."""
    np.testing.assert_allclose(z_matrix(dj).T, -z_matrix(dj), rtol=0.0, atol=1e-14)


# --- the independent oracle (no CG coefficients involved) -----------------------


@pytest.mark.parametrize("dj", range(1, 7))
def test_generator_intertwining_relation(dj):
    z = z_matrix(dj)
    zinv = np.linalg.inv(z)
    for gen in spin_matrices(dj):
        np.testing.assert_allclose(z @ gen @ zinv, -gen.T, rtol=0.0, atol=1e-12)


# --- the derivation -------------------------------------------------------------


@pytest.mark.parametrize("dj", DJS)
def test_cg_relation_with_the_sign_asserted(dj):
    derived = sqrt(dj + 1) * cg_tensor(dj, dj, 0)[:, :, 0]
    np.testing.assert_allclose(z_matrix(dj), derived, rtol=0.0, atol=1e-14)
    # the sign is asserted, not absorbed: no overall flip is needed anywhere
    assert derived[0, dj] == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("dj", range(1, 9))
def test_self_dual_label_is_not_a_no_op(dj):
    a = SU2Sector(dj)
    assert SU2.dual(a) == a
    assert not np.allclose(SU2.z_matrix(a), np.eye(dj + 1))


# --- to_dense -------------------------------------------------------------------

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 2, ONE: 3})  # dj = 1 and dj = 2, so d_a > 1
W = GradedSpace.new(SU2, {SINGLET: 2, HALF: 3})


def dualize_axis(t: SymmetricTensor, axis: int) -> SymmetricTensor:
    """The same blocks, with one leg's ``dual`` flag flipped (labels are unchanged)."""
    legs = tuple(
        dataclasses.replace(leg, dual=not leg.dual) if i == axis else leg
        for i, leg in enumerate(t.legs)
    )
    return SymmetricTensor.from_blocks(legs, keyed(legs, t.blocks))


def test_to_dense_on_a_dual_su2_leg_succeeds():
    t = dualize_axis(SymmetricTensor.random((Leg(V, OUT), Leg(W, IN)), seed=7), 1)
    assert t.legs[1].dual
    dense = t.to_dense()
    assert dense.shape == t.shape
    assert np.isfinite(dense).all()


@pytest.mark.parametrize("axis", [0, 1])
def test_dual_leg_dense_content_is_the_z_insertion(axis):
    """Slab reversal plus alternating sign, per sector, for a space with d_a > 1."""
    legs = (Leg(V, OUT), Leg(V, IN))
    plain = SymmetricTensor.random(legs, seed=11)
    got = dualize_axis(plain, axis).to_dense()

    want = np.array(plain.to_dense())
    for a in (HALF, ONE):
        d = SU2.irrep_dim(a)
        m = V.degeneracy(a)
        start = V.sector_offset(a)
        sl = [slice(None), slice(None)]
        sl[axis] = slice(start, start + m * d)
        slab = np.moveaxis(want[tuple(sl)], axis, 0).reshape(m, d, -1)
        # within-slab index is alpha * d_a + i, so Z acts on the trailing irrep index
        slab = np.einsum("ki,aix->akx", z_matrix(a.two_j), slab)
        want[tuple(sl)] = np.moveaxis(slab.reshape(m * d, -1), 0, axis)

    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
    assert not np.allclose(got, plain.to_dense())  # the insertion is not a no-op


def cup(dj: int, *, dual: bool) -> np.ndarray:
    """``1 -> V_j (x) V_j`` (or ``V_j (x) V_j^*``) with the single block set to 1."""
    space = GradedSpace.new(SU2, {SU2Sector(dj): 1})
    legs = (Leg(space, OUT), Leg(space, OUT, dual=dual))
    structure = TensorStructure(legs)
    (key,) = structure.block_order  # `1 -> V_j (x) V_j` has exactly one fusion channel
    return SymmetricTensor.from_blocks(legs, {key: np.ones(structure.block_shape(key))}).to_dense()


@pytest.mark.parametrize("dj", [1, 2, 3])
def test_cup_is_z_up_to_normalization(dj):
    """Two direct legs: the cup *is* ``Z_j``, normalized by ``1/sqrt(d_j)``."""
    np.testing.assert_allclose(cup(dj, dual=False) * sqrt(dj + 1), z_matrix(dj), atol=1e-12)


@pytest.mark.parametrize("dj", [1, 2, 3])
def test_cup_with_a_dual_leg_is_the_evaluation_map(dj):
    """``V_j (x) V_j^* -> 1``: the Z insertion turns the cup into the identity."""
    np.testing.assert_allclose(
        cup(dj, dual=True) * sqrt(dj + 1), np.eye(dj + 1), rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("dj", [1, 2, 3])
def test_cap_against_cup_gives_the_identity_with_the_fs_sign(dj):
    c = cup(dj, dual=False)
    eye = np.eye(dj + 1)
    # composed as Z Z^T (partner's first index onto the cup's second): unitarity
    np.testing.assert_allclose((c @ c.T) * (dj + 1), eye, rtol=0.0, atol=1e-12)
    # composed the other way round (Z Z) the Frobenius-Schur sign appears
    np.testing.assert_allclose((c @ c) * (dj + 1), frobenius_schur(dj) * eye, atol=1e-12)


# --- containment ----------------------------------------------------------------


def _code_without_docstrings(path) -> str:
    """The module's source with every string literal removed, so a grep sees calls only."""
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(tree)


def test_z_matrix_is_only_used_by_to_dense_in_library_code():
    import pathlib

    import tenet

    global _SU2_COEFF_CODE

    src = pathlib.Path(tenet.__file__).parent
    _SU2_COEFF_CODE = _code_without_docstrings(src / "symmetry" / "_su2_coeff.py")
    users = {
        p.name
        for p in src.rglob("*.py")
        if "z_matrix" in p.read_text() and p.name not in {"base.py", "__init__.py"}
    }
    # #82 moved `_tree_cgt` (the one caller of `z_matrix`) out of tensor.py and
    # into ops/dense.py, where the dense boundary now lives, unedited.
    #
    # `_su2_coeff.py` is a *docstring* mention, not a call: the scan is textual and
    # cannot tell them apart. It is there because that module has to say which tier
    # `z_matrix` stays on and why -- gauge data, unlike the F/R it does serve -- and
    # deleting the sentence to satisfy a grep would delete the reason the split exists.
    assert users == {
        "dense.py",
        "su2.py",
        "u1.py",
        "fz2.py",
        "z2.py",
        "sun.py",
        "_sun_coeff.py",
        "_su2_coeff.py",
    }
    assert "z_matrix" not in _SU2_COEFF_CODE, (
        "_su2_coeff.py now *calls* z_matrix; it is admitted to the set above as a "
        "docstring mention only, and a call means the tier split has moved"
    )


def test_to_dense_still_refuses_a_provider_without_dual_basis():
    @dataclasses.dataclass(frozen=True, slots=True)
    class NoDual:
        name: str = "NoDual"

    with pytest.raises(CapabilityError, match="DualBasis"):
        from tenet.ops.dense import _refuse_dual

        _refuse_dual(NoDual(), 1)
