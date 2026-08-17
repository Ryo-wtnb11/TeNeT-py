"""Tests for ``tenet.flip`` — issue #142.

The two TensorKit contracts anchor everything: (a) flipping both legs of a
contractible pair leaves the contraction unchanged, and (b) ``flip`` is not an
involution — a double flip pays ``chi_a * theta_a`` once per tree, which is
``-1`` on an SU(2) half-integer line and, through the twist, on an odd fZ2 line.
The dense oracle builds the Z contraction by hand from ``provider.z_matrix`` per
sector, the same stance as the cup/cap oracle in ``tests/symmetry/test_su2_dual.py``.
"""

import dataclasses

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, flip, tensordot
from tenet.ops.contraction import contractible
from tenet.symmetry import (
    SU2,
    U1,
    Z2,
    CapabilityError,
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

HALF, ONE = SU2Sector(1), SU2Sector(2)
EVEN, ODD = FZ2Sector(0), FZ2Sector(1)

U1_FREE = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
U1_BOND = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})
SU2_FREE = GradedSpace.new(SU2, {SU2Sector(0): 1, HALF: 2, ONE: 1})
SU2_BOND = GradedSpace.new(SU2, {HALF: 2})
FZ2_SPACE = GradedSpace.new(fZ2, {EVEN: 2, ODD: 2})
Z2_SPACE = GradedSpace.new(Z2, {Z2Sector(0): 2, Z2Sector(1): 2})
TRIV_SPACE = GradedSpace.new(Trivial, {TrivialSector(): 3})

PRODUCT = ProductProvider((U1, SU2))
PROD_SPACE = GradedSpace.new(
    PRODUCT, {ProductSector((U1Sector(0), HALF)): 2, ProductSector((U1Sector(0), SU2Sector(0))): 1}
)


def pair(free, bond, *, seeds=(1, 2)):
    """``a`` and ``b`` with ``a``'s IN leg contractible against ``b``'s OUT leg."""
    a = SymmetricTensor.random((Leg(free, OUT), Leg(bond, IN, name="x")), seed=seeds[0])
    b = SymmetricTensor.random((Leg(bond, OUT, name="y"), Leg(free, IN)), seed=seeds[1])
    return a, b


def dense(t):
    return np.asarray(t.to_dense())


# --- contract (a): flipping both legs of a contracted pair changes nothing ------


@pytest.mark.parametrize(
    "free, bond",
    [(U1_FREE, U1_BOND), (FZ2_SPACE, FZ2_SPACE), (SU2_FREE, SU2_BOND)],
    ids=["u1", "fz2", "su2"],
)
def test_contract_after_flipping_both_legs(free, bond):
    a, b = pair(free, bond)
    ref = dense(tensordot(a, b, ((1,), (0,))))
    got = dense(tensordot(flip(a, 1), flip(b, 0), ((1,), (0,))))
    assert np.max(np.abs(got - ref)) <= 1e-12
    # and again from the dual-start convention, exercising the other branch
    a2, b2 = flip(a, 1), flip(b, 0)
    ref2 = dense(tensordot(a2, b2, ((1,), (0,))))
    got2 = dense(tensordot(flip(a2, 1), flip(b2, 0), ((1,), (0,))))
    assert np.max(np.abs(got2 - ref2)) <= 1e-12


def test_contract_after_flip_with_a_bend_in_the_lowering():
    """The contracted pair sits OUT-vs-IN backwards, so tensordot must bend it."""
    a = SymmetricTensor.random((Leg(SU2_BOND, OUT, name="x"), Leg(SU2_FREE, IN)), seed=11)
    b = SymmetricTensor.random((Leg(SU2_FREE, OUT), Leg(SU2_BOND, IN, name="y")), seed=12)
    ref = dense(tensordot(a, b, ((0,), (1,))))
    got = dense(tensordot(flip(a, 0), flip(b, 1), ((0,), (1,))))
    assert np.max(np.abs(got - ref)) <= 1e-12


# --- flipping one leg alone breaks contractibility, two different ways ----------


def test_one_flipped_leg_breaks_contractibility_by_space_inequality():
    a, b = pair(U1_FREE, U1_BOND)
    flipped = flip(a, 1)
    assert flipped.legs[1].space != a.legs[1].space  # the U(1) relabel is visible
    assert not contractible(flipped.legs[1], b.legs[0])
    with pytest.raises(ValueError, match="do not contract"):
        tensordot(flipped, b, ((1,), (0,)))


@pytest.mark.parametrize(
    "free, bond",
    [(SU2_FREE, SU2_BOND), (Z2_SPACE, Z2_SPACE), (FZ2_SPACE, FZ2_SPACE)],
    ids=["su2", "z2", "fz2"],
)
def test_one_flipped_leg_breaks_contractibility_by_outward_dual(free, bond):
    a, b = pair(free, bond)
    flipped = flip(a, 1)
    assert flipped.legs[1].space == a.legs[1].space  # self-dual sectors: same space
    assert not contractible(flipped.legs[1], b.legs[0])
    with pytest.raises(ValueError, match="do not contract"):
        tensordot(flipped, b, ((1,), (0,)))


# --- contract (b): non-involutivity, and the exact inverse ----------------------


@pytest.mark.parametrize(
    "space, sign",
    [
        (GradedSpace.new(SU2, {HALF: 2}), -1),
        (GradedSpace.new(SU2, {ONE: 2}), 1),
        (GradedSpace.new(fZ2, {ODD: 2}), -1),
        (GradedSpace.new(U1, {U1Sector(1): 2}), 1),
        (Z2_SPACE, 1),
    ],
    ids=["su2-half", "su2-integer", "fz2-odd", "u1", "z2"],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_double_flip_sign(space, sign, axis):
    t = SymmetricTensor.random((Leg(space, OUT), Leg(space, IN)), seed=5)
    twice = flip(flip(t, axis), axis)
    assert tenet.allclose(twice, t if sign == 1 else tenet.negative(t))


def test_fz2_double_flip_sign_comes_from_the_twist():
    """chi is +1 for fZ2 (fz2.py pins the TensorKitSectors source), so this sign
    is the twist and nothing else — a frobenius_schur-only stub would return +t."""
    assert fZ2.flip_phase(ODD) == -1.0
    assert fZ2.flip_phase(EVEN) == 1.0
    space = GradedSpace.new(fZ2, {ODD: 3})
    t = SymmetricTensor.random((Leg(space, OUT), Leg(space, IN)), seed=6)
    assert tenet.allclose(flip(flip(t, 0), 0), tenet.negative(t))


@pytest.mark.parametrize(
    "space",
    [TRIV_SPACE, U1_BOND, Z2_SPACE, FZ2_SPACE, SU2_FREE, PROD_SPACE],
    ids=["trivial", "u1", "z2", "fz2", "su2", "u1xsu2"],
)
@pytest.mark.parametrize("dual", [False, True])
def test_inv_round_trips_bit_identical(space, dual):
    t = SymmetricTensor.random((Leg(space, OUT, dual=dual), Leg(space, IN)), seed=7)
    back = flip(flip(t, 0), 0, inv=True)
    assert back.structure == t.structure
    for x, y in zip(back.blocks, t.blocks, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_multi_axis_is_the_sequential_fold_and_order_free():
    t = SymmetricTensor.random((Leg(SU2_FREE, OUT), Leg(SU2_BOND, IN)), seed=8)
    both = flip(t, (0, 1))
    assert both.structure == flip(flip(t, 0), 1).structure
    for x, y in zip(both.blocks, flip(flip(t, 0), 1).blocks, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    for x, y in zip(both.blocks, flip(t, (1, 0)).blocks, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# --- the dense oracle -----------------------------------------------------------


def z_by_hand(provider, old_space, new_space):
    """The dense ``V -> V*`` map per sector, from ``provider.z_matrix`` directly.

    ``to_dense`` lays each sector out as a contiguous ``m_a * d_a`` slab with the
    within-slab index ``alpha * d_a + i``; ``flip`` sends sector ``a``'s slab to
    ``dual(a)``'s slab of the relabelled space through ``Z_{dual(a)}`` (the matrix
    ``dense.py`` keys on the *space* sector), so the hand-built map is
    ``M[o_a + alpha d_a + q, o'_abar + alpha d_abar + p] = Z_abar[p, q]``.
    """
    M = np.zeros((old_space.dim, new_space.dim))
    for a, m in old_space.sectors:
        abar = provider.dual(a)
        z = np.asarray(provider.z_matrix(abar))
        d_a, d_ab = provider.irrep_dim(a), provider.irrep_dim(abar)
        o, n = old_space.sector_offset(a), new_space.sector_offset(abar)
        for alpha in range(m):
            rows = slice(o + alpha * d_a, o + (alpha + 1) * d_a)
            cols = slice(n + alpha * d_ab, n + (alpha + 1) * d_ab)
            M[rows, cols] = z.T
    return M


@pytest.mark.parametrize(
    "provider, space",
    [
        (U1, U1_FREE),
        (fZ2, FZ2_SPACE),
        (SU2, GradedSpace.new(SU2, {HALF: 2})),
        (SU2, GradedSpace.new(SU2, {ONE: 2})),
    ],
    ids=["u1", "fz2", "su2-two_j-1", "su2-two_j-2"],
)
def test_dense_oracle(provider, space):
    other = {
        U1: U1_BOND,
        fZ2: FZ2_SPACE,
        SU2: GradedSpace.new(SU2, {SU2Sector(0): 1, HALF: 2, ONE: 2}),
    }[provider]
    t = SymmetricTensor.random((Leg(space, OUT), Leg(other, IN)), seed=9)
    flipped = flip(t, 0)
    M = z_by_hand(provider, t.legs[0].space, flipped.legs[0].space)
    hand = np.moveaxis(np.tensordot(dense(t), M, axes=([0], [0])), -1, 0)
    assert np.max(np.abs(dense(flipped) - hand)) <= 1e-12


def test_flip_needs_flip_phase_not_dual_basis():
    """A ProductProvider has no z_matrix, so ``to_dense`` refuses its dual leg —
    ``flip`` needs only ``FlipPhase`` and works anyway."""
    t = SymmetricTensor.random((Leg(PROD_SPACE, OUT, dual=True), Leg(PROD_SPACE, IN)), seed=10)
    with pytest.raises(CapabilityError, match="DualBasis"):
        t.to_dense()
    flipped = flip(t, 0)
    assert flipped.legs[0].dual is False


# --- the capability gate --------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FusionOnly:
    """FusionProvider and nothing else — no flip phase."""

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


def test_flip_refuses_a_provider_without_flip_phase():
    space = GradedSpace.new(FusionOnly(), {TrivialSector(): 2})
    t = SymmetricTensor.zeros((Leg(space, OUT), Leg(space, IN)))
    with pytest.raises(CapabilityError) as err:
        flip(t, 0)
    message = str(err.value)
    assert "FlipPhase" in message
    assert "chi_a * theta_a" in message
    assert "wrong sign" in message


# --- structural invariants ------------------------------------------------------


def test_flip_preserves_blocks_order_shapes_names_sides():
    legs = (
        Leg(SU2_FREE, OUT, name="a"),
        Leg(SU2_BOND, IN, name="b"),
        Leg(SU2_BOND, OUT, name="c"),
        Leg(SU2_FREE, IN, name="d"),
    )
    t = SymmetricTensor.random(legs, seed=13)
    f = flip(t, (1, 2))
    assert f.structure.block_order == t.structure.block_order
    assert f.structure.num_blocks == t.structure.num_blocks
    for key in t.structure.block_order:
        assert f.structure.block_shape(key) == t.structure.block_shape(key)
    for i, (old, new) in enumerate(zip(t.legs, f.legs, strict=True)):
        assert new.name == old.name
        assert new.side is old.side
        assert new.dual == (old.dual if i not in (1, 2) else not old.dual)
        if i not in (1, 2):
            assert new == old


def test_flip_no_axes_is_identity():
    t = SymmetricTensor.random((Leg(U1_BOND, OUT), Leg(U1_BOND, IN)), seed=14)
    assert flip(t, ()) is t


def test_flip_axis_validation():
    t = SymmetricTensor.random((Leg(U1_BOND, OUT, name="p"), Leg(U1_BOND, IN, name="p")), seed=15)
    with pytest.raises(ValueError, match="out of range"):
        flip(t, 2)
    with pytest.raises(ValueError, match="repeated"):
        flip(t, (0, 0))
    with pytest.raises(ValueError, match="no leg is named"):
        flip(t, "q")
    with pytest.raises(ValueError, match="ambiguous"):
        flip(t, "p")


def test_flip_by_leg_name():
    t = SymmetricTensor.random((Leg(U1_BOND, OUT, name="p"), Leg(U1_BOND, IN, name="q")), seed=16)
    by_name, by_axis = flip(t, "q"), flip(t, 1)
    assert by_name.structure == by_axis.structure
    for x, y in zip(by_name.blocks, by_axis.blocks, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_flip_keeps_a_real_dtype_real():
    space = GradedSpace.new(SU2, {HALF: 2})
    t = SymmetricTensor.random((Leg(space, OUT, dual=True), Leg(space, IN)), seed=17)
    flipped = flip(t, 0)  # pays chi = -1
    for block in flipped.blocks:
        assert np.asarray(block).dtype == np.float64


# --- backends -------------------------------------------------------------------


def test_flip_on_jax_matches_numpy_and_jits():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  # registration is the import's side effect

    space = GradedSpace.new(SU2, {HALF: 2})
    t = SymmetricTensor.random((Leg(space, OUT, dual=True), Leg(space, IN)), seed=18)
    on_jax = flip(t.to_backend("jax"), 0)
    assert on_jax.backend == "jax"
    assert tenet.allclose(on_jax.to_backend("numpy"), flip(t, 0))
    jitted = jax.jit(lambda x: flip(x, 0))(t.to_backend("jax"))
    assert tenet.allclose(jitted.to_backend("numpy"), flip(t, 0))
