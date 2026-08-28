"""Tests for tenet.tensor — one test per acceptance criterion of issue #9."""

import dataclasses
from itertools import product
from math import sqrt

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.symmetry import SU2, U1, CapabilityError, SU2Sector, Trivial, TrivialSector, U1Sector

HALF, ONE, SINGLET = SU2Sector(1), SU2Sector(2), SU2Sector(0)
V = GradedSpace.new(SU2, {HALF: 4, ONE: 3})
W = GradedSpace.new(SU2, {HALF: 2})
S = GradedSpace.new(SU2, {HALF: 1})  # a single spin-1/2, no degeneracy

TRIV = GradedSpace.new(Trivial, {TrivialSector(): 3})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})


def keyed(legs, blocks):
    """Every block, keyed in ``block_order``."""
    return dict(zip(TensorStructure(tuple(legs)).block_order, blocks, strict=True))


def spin_matrices(two_j: int) -> tuple[np.ndarray, np.ndarray]:
    """``(Jz, J+)`` in the descending-m basis used by ``cg_tensor``: index i has m = j - i."""
    d = two_j + 1
    j = two_j / 2
    ms = np.array([j - i for i in range(d)])
    jp = np.zeros((d, d))
    for i in range(1, d):  # J+ raises m: column i (m) -> row i-1 (m+1)
        m = ms[i]
        jp[i - 1, i] = sqrt(j * (j + 1) - m * (m + 1))
    return np.diag(ms), jp


def leg_generator(leg: Leg, which: int) -> np.ndarray:
    """The single-leg spin operator on the dense space, honouring ``alpha * d_a + m``."""
    blocks = []
    for a, m in leg.space.sectors:
        blocks.append(np.kron(np.eye(m), spin_matrices(a.two_j)[which]))
    out = np.zeros((leg.space.dim, leg.space.dim))
    off = 0
    for b in blocks:
        out[off : off + len(b), off : off + len(b)] = b
        off += len(b)
    return out


def apply_generator(dense: np.ndarray, legs, which: int) -> np.ndarray:
    """``Σ_i (J)_i · dense``."""
    total = np.zeros_like(dense)
    for i, leg in enumerate(legs):
        total += np.moveaxis(np.tensordot(leg_generator(leg, which), dense, axes=([1], [i])), 0, i)
    return total


# --- structure, immutability, views ---------------------------------------------


def test_frozen_hashable_structure_tuple_blocks():
    t = SymmetricTensor.zeros((Leg(TRIV, OUT), Leg(TRIV, IN)))
    assert isinstance(t.blocks, tuple)
    hash(t.structure)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.blocks = ()


def test_wrong_block_count_raises():
    legs = (Leg(V, OUT), Leg(W, IN), Leg(V, OUT))
    structure = TensorStructure(tuple(legs))
    assert len(structure.block_order) > 1
    with pytest.raises(ValueError):
        SymmetricTensor(structure, (np.zeros((4, 2, 4)),))


def test_wrong_block_shape_names_index_and_expected_shape():
    legs = (Leg(V, OUT), Leg(W, IN))
    blocks = list(SymmetricTensor.zeros(legs).blocks)
    blocks[0] = np.zeros((1, 1))
    with pytest.raises(ValueError, match=r"block 0 has shape \(1, 1\), expected"):
        SymmetricTensor.from_blocks(legs, keyed(legs, blocks))


def test_mixed_dtypes_raise():
    legs = (Leg(V, OUT), Leg(W, IN), Leg(V, OUT))
    blocks = [b.astype(np.float32) for b in SymmetricTensor.zeros(legs).blocks]
    blocks[0] = blocks[0].astype(np.float64)
    with pytest.raises(ValueError, match="share one dtype"):
        SymmetricTensor.from_blocks(legs, keyed(legs, blocks))


# --- the unvalidated internal path (#328) ---------------------------------------


def test_the_unvalidated_constructor_is_not_public():
    import tenet.tensor

    assert "_unchecked" not in tenet.tensor.__all__
    assert not hasattr(tenet, "_unchecked")
    assert not any(name.endswith("unchecked") for name in dir(tenet))


@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: tenet.tensordot(a, b, ((1,), (0,))),
        lambda a, b: tenet.transpose(a, (1, 0)),
        lambda a, b: tenet.add(a, a),
        lambda a, b: tenet.adjoint(a),
        lambda a, b: tenet.fuse(
            SymmetricTensor.random((Leg(V, OUT), Leg(V, OUT), Leg(V, IN)), seed=2), (0, 1)
        ),
        lambda a, b: tenet.bend(SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=3), 1),
    ],
)
def test_the_fast_path_and_the_checked_path_agree_on_a_real_result(op):
    """Why skipping validation is safe: the blocks really do have the checked shapes."""
    legs = (Leg(V, OUT), Leg(V, IN))
    a = SymmetricTensor.random(legs, seed=0)
    b = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)
    got = op(a, b)
    shapes = got.structure.block_shapes
    assert len(got.blocks) == len(shapes)
    assert all(tuple(x.shape) == s for x, s in zip(got.blocks, shapes, strict=True))
    # the trust boundary, re-run over what the internal path produced: it accepts
    again = SymmetricTensor(got.structure, got.blocks)
    assert again == got


def test_block_view_is_identity_and_items_round_trip():
    legs = (Leg(V, OUT), Leg(W, IN))
    t = SymmetricTensor.random(legs, seed=0)
    for key in t.structure.block_order:
        assert t.blocks[t.structure.index_of(key)] is t.block(key)
    again = SymmetricTensor.from_blocks(legs, keyed(legs, [b for _, b in t.items()]))
    assert again.structure == t.structure
    assert all(np.array_equal(x, y) for x, y in zip(again.blocks, t.blocks, strict=True))


def test_domain_codomain_and_block_shapes_are_public_order():
    c1, d1 = GradedSpace.new(U1, {U1Sector(0): 2}), GradedSpace.new(U1, {U1Sector(0): 3})
    c2, d2 = GradedSpace.new(U1, {U1Sector(0): 4}), GradedSpace.new(U1, {U1Sector(0): 5})
    legs = (Leg(c1, OUT), Leg(d1, IN), Leg(c2, OUT), Leg(d2, IN))
    t = SymmetricTensor.zeros(legs)
    assert t.ndim == 4
    assert t.codomain == (legs[0], legs[2])
    assert t.domain == (legs[1], legs[3])
    assert all(b.shape == (2, 3, 4, 5) for b in t.blocks)


def test_readme_example():
    t = SymmetricTensor.zeros((Leg(V, OUT), Leg(W, IN), Leg(V, OUT)))
    assert t.ndim == 3
    assert len(t.codomain) == 2
    assert len(t.domain) == 1


def test_no_implicit_densification():
    assert not hasattr(SymmetricTensor, "__array__")
    t = SymmetricTensor.zeros((Leg(TRIV, OUT), Leg(TRIV, IN)))
    assert np.asarray(t).dtype == object


def test_random_is_reproducible_and_shaped():
    legs = (Leg(V, OUT), Leg(W, IN))
    a, b = (SymmetricTensor.random(legs, seed=7) for _ in range(2))
    assert all(np.array_equal(x, y) for x, y in zip(a.blocks, b.blocks, strict=True))
    assert not np.array_equal(a.blocks[0], SymmetricTensor.random(legs, seed=8).blocks[0])
    for key, block in a.items():
        assert block.shape == a.structure.block_shape(key)


# --- to_dense guards -------------------------------------------------------------


def test_to_dense_accepts_a_dual_su2_leg():
    # #32 narrowed the guard to a DualBasis capability check; #37 supplies the
    # SU(2) Z-isomorphism, so the guard no longer fires. Content: test_su2_dual.py.
    legs = (Leg(V, OUT), Leg(W, IN, dual=True))
    t = SymmetricTensor.random(legs, seed=5)
    dense = t.to_dense()
    assert dense.shape == t.shape
    assert dense.any()


def test_to_dense_requires_clebsch_gordan():
    @dataclasses.dataclass(frozen=True, slots=True)
    class Bare:
        """A provider with fusion data but no CG capability."""

        name: str = "Bare"

        @property
        def unit(self):
            return TrivialSector()

        def dual(self, a):
            return a

        def fusion(self, a, b):
            return (TrivialSector(),)

        def n_symbol(self, a, b, c):
            return 1

    space = GradedSpace.new(Bare(), {TrivialSector(): 2})
    t = SymmetricTensor.zeros((Leg(space, OUT), Leg(space, IN)))
    with pytest.raises(CapabilityError, match="ClebschGordanData"):
        t.to_dense()


# --- to_dense values -------------------------------------------------------------


def test_to_dense_shape():
    t = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN), Leg(V, OUT)), seed=1)
    assert t.to_dense().shape == tuple(leg.space.dim for leg in t.legs)


def test_trivial_to_dense_is_the_block():
    legs = (Leg(TRIV, OUT), Leg(TRIV, IN), Leg(TRIV, OUT))
    t = SymmetricTensor.random(legs, seed=2)
    assert len(t.blocks) == 1
    np.testing.assert_allclose(t.to_dense(), t.blocks[0])


def test_u1_to_dense_matches_hand_built_scatter():
    legs = (Leg(Q, OUT), Leg(Q, IN), Leg(Q, OUT))
    t = SymmetricTensor.random(legs, seed=3)
    expected = np.zeros(tuple(leg.space.dim for leg in legs))
    for key, block in t.items():
        sectors = t.structure.axis_sectors(key)
        slabs = tuple(
            slice(leg.space.sector_offset(a), leg.space.sector_offset(a) + leg.degeneracy(a))
            for leg, a in zip(legs, sectors, strict=True)
        )
        expected[slabs] = block  # d_a == 1, so a slab is just the degeneracy range
    dense = t.to_dense()
    np.testing.assert_allclose(dense, expected, atol=1e-14)

    # every charge-violating entry (out charges != in charges) is exactly zero
    charges = [[a.charge for a, m in leg.space.sectors for _ in range(m)] for leg in legs]
    for idx in product(*(range(leg.space.dim) for leg in legs)):
        q = [charges[i][j] for i, j in enumerate(idx)]
        if q[0] + q[2] != q[1]:
            assert dense[idx] == 0.0


def test_su2_norm_identity():
    legs = (Leg(V, OUT), Leg(W, IN), Leg(V, OUT))
    t = SymmetricTensor.random(legs, seed=4)
    expected = sum(SU2.qdim(key.coupled) * np.sum(block**2) for key, block in t.items())
    assert np.sum(t.to_dense() ** 2) == pytest.approx(expected, abs=1e-10)


def test_su2_equivariance_of_singlet_tensor():
    legs = (Leg(V, OUT), Leg(W, OUT), Leg(V, OUT))
    t = SymmetricTensor.random(legs, seed=5)
    # all legs OUT => the input tree is rank 0, so every block couples to the singlet
    assert all(key.coupled == SINGLET for key in t.structure.block_order)
    dense = t.to_dense()
    for which in (0, 1):  # Jz, J+
        assert np.abs(apply_generator(dense, legs, which)).max() < 1e-10


def test_su2_basis_completeness():
    legs = (Leg(S, OUT), Leg(S, OUT), Leg(S, OUT), Leg(S, IN))
    structure = TensorStructure(legs)
    keys = [k for k in structure.block_order if structure.axis_sectors(k) == (HALF,) * 4]
    assert len(keys) == 2
    assert keys[0].output_tree.inner != keys[1].output_tree.inner

    denses = []
    for key in keys:
        blocks = [np.zeros(structure.block_shape(k)) for k in structure.block_order]
        blocks[structure.index_of(key)] = np.ones(structure.block_shape(key))
        denses.append(SymmetricTensor(structure, blocks).to_dense().ravel())
    a, b = denses
    overlap = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    assert abs(abs(overlap) - 1.0) > 1e-3


def test_to_dense_is_linear_in_the_blocks():
    legs = (Leg(V, OUT), Leg(W, IN), Leg(V, OUT))
    t1 = SymmetricTensor.random(legs, seed=10)
    t2 = SymmetricTensor.random(legs, seed=11)
    a, b = 2.5, -0.75
    combos = [a * x + b * y for x, y in zip(t1.blocks, t2.blocks, strict=True)]
    mixed = SymmetricTensor.from_blocks(legs, keyed(legs, combos))
    np.testing.assert_allclose(mixed.to_dense(), a * t1.to_dense() + b * t2.to_dense(), atol=1e-12)
