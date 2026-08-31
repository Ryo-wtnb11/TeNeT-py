"""Tests for ``tenet.flip_dual`` — issue #142.

The two TensorKit contracts anchor everything: (a) flipping both legs of a
contractible pair leaves the contraction unchanged, and (b) ``flip_dual`` is not an
involution — a double flip pays ``chi_a * theta_a`` once per tree, which is
``-1`` on an SU(2) half-integer line and, through the twist, on an odd fZ2 line.
The dense oracle builds the Z contraction by hand from ``provider.z_matrix`` per
sector, the same stance as the cup/cap oracle in ``tests/symmetry/test_su2_dual.py``.
"""

import dataclasses
import itertools

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, flip_dual, tensordot
from tenet.map_view import map_layout
from tenet.ops.contraction import contractible
from tenet.structure import TensorStructure
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

FPRODUCT = ProductProvider((fZ2, U1, SU2))
FPROD_SPACE = GradedSpace.new(
    FPRODUCT,
    {
        ProductSector((EVEN, U1Sector(0), SU2Sector(0))): 2,
        ProductSector((ODD, U1Sector(1), HALF)): 2,
    },
)

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
    got = dense(tensordot(flip_dual(a, 1), flip_dual(b, 0), ((1,), (0,))))
    assert np.max(np.abs(got - ref)) <= 1e-12
    # and again from the dual-start convention, exercising the other branch
    a2, b2 = flip_dual(a, 1), flip_dual(b, 0)
    ref2 = dense(tensordot(a2, b2, ((1,), (0,))))
    got2 = dense(tensordot(flip_dual(a2, 1), flip_dual(b2, 0), ((1,), (0,))))
    assert np.max(np.abs(got2 - ref2)) <= 1e-12


def test_contract_after_flip_with_a_bend_in_the_lowering():
    """The contracted pair sits OUT-vs-IN backwards, so tensordot must bend it."""
    a = SymmetricTensor.random((Leg(SU2_BOND, OUT, name="x"), Leg(SU2_FREE, IN)), seed=11)
    b = SymmetricTensor.random((Leg(SU2_FREE, OUT), Leg(SU2_BOND, IN, name="y")), seed=12)
    ref = dense(tensordot(a, b, ((0,), (1,))))
    got = dense(tensordot(flip_dual(a, 0), flip_dual(b, 1), ((0,), (1,))))
    assert np.max(np.abs(got - ref)) <= 1e-12


# --- flipping one leg alone breaks contractibility, two different ways ----------


def test_one_flipped_leg_breaks_contractibility_by_space_inequality():
    a, b = pair(U1_FREE, U1_BOND)
    flipped = flip_dual(a, 1)
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
    flipped = flip_dual(a, 1)
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
    twice = flip_dual(flip_dual(t, axis), axis)
    assert tenet.allclose(twice, t if sign == 1 else tenet.negative(t))


def test_fz2_double_flip_sign_comes_from_the_twist():
    """chi is +1 for fZ2 (fz2.py pins the TensorKitSectors source), so this sign
    is the twist and nothing else — a frobenius_schur-only stub would return +t."""
    assert fZ2.frobenius_schur(ODD) * fZ2.twist(ODD) == -1.0
    assert fZ2.frobenius_schur(EVEN) * fZ2.twist(EVEN) == 1.0
    space = GradedSpace.new(fZ2, {ODD: 3})
    t = SymmetricTensor.random((Leg(space, OUT), Leg(space, IN)), seed=6)
    assert tenet.allclose(flip_dual(flip_dual(t, 0), 0), tenet.negative(t))


@pytest.mark.parametrize(
    "space",
    [TRIV_SPACE, U1_BOND, Z2_SPACE, FZ2_SPACE, SU2_FREE, PROD_SPACE],
    ids=["trivial", "u1", "z2", "fz2", "su2", "u1xsu2"],
)
@pytest.mark.parametrize("dual", [False, True])
def test_inv_round_trips_bit_identical(space, dual):
    t = SymmetricTensor.random((Leg(space, OUT, dual=dual), Leg(space, IN)), seed=7)
    back = flip_dual(flip_dual(t, 0), 0, inv=True)
    assert back.structure == t.structure
    for x, y in zip(back.blocks, t.blocks, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_multi_axis_is_the_sequential_fold_and_order_free():
    t = SymmetricTensor.random((Leg(SU2_FREE, OUT), Leg(SU2_BOND, IN)), seed=8)
    both = flip_dual(t, (0, 1))
    assert both.structure == flip_dual(flip_dual(t, 0), 1).structure
    for x, y in zip(both.blocks, flip_dual(flip_dual(t, 0), 1).blocks, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    for x, y in zip(both.blocks, flip_dual(t, (1, 0)).blocks, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# --- the dense oracle -----------------------------------------------------------


def z_by_hand(provider, old_space, new_space):
    """The dense ``V -> V*`` map per sector, from ``provider.z_matrix`` directly.

    ``to_dense`` lays each sector out as a contiguous ``m_a * d_a`` slab with the
    within-slab index ``alpha * d_a + i``; ``flip_dual`` sends sector ``a``'s slab to
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
    flipped = flip_dual(t, 0)
    M = z_by_hand(provider, t.legs[0].space, flipped.legs[0].space)
    hand = np.moveaxis(np.tensordot(dense(t), M, axes=([0], [0])), -1, 0)
    assert np.max(np.abs(dense(flipped) - hand)) <= 1e-12


def test_flip_dual_needs_only_the_phase_capabilities_not_dual_basis():
    """``flip_dual`` gates on ``FSIndicatorData`` + ``TwistData``, and on nothing else.

    The vehicle changed in #312 and the claim did not. This used to ride on
    ``ProductProvider``, which had no ``z_matrix``, so ``to_dense`` refused its dual leg
    while ``flip_dual`` went through — a neat demonstration that the two gate on
    different capabilities. A product now forwards ``DualBasis``, so it no longer
    demonstrates anything; the stub below lacks ``z_matrix`` on purpose and carries the
    two phase capabilities, which is the statement stripped of its accident.
    """

    @dataclasses.dataclass(frozen=True, slots=True)
    class PhasesOnly:
        """The flip scalar and nothing to expand a dual leg with."""

        name: str = "PhasesOnly"

        @property
        def unit(self):
            return TrivialSector()

        def dual(self, a):
            return a

        def fusion(self, a, b):
            return (TrivialSector(),)

        def n_symbol(self, a, b, c):
            return 1

        def frobenius_schur(self, a) -> int:
            return 1

        def twist(self, a) -> int:
            return 1

        # ClebschGordanData too, so `to_dense` gets far enough to refuse on the
        # capability this case is about rather than on a missing expansion basis.
        def irrep_dim(self, a) -> int:
            return 1

        def cgc(self, a, b, c):
            return np.ones((1, 1, 1, 1))

    space = GradedSpace.new(PhasesOnly(), {TrivialSector(): 2})
    t = SymmetricTensor.random((Leg(space, OUT, dual=True), Leg(space, IN)), seed=10)
    with pytest.raises(CapabilityError, match="DualBasis"):
        t.to_dense()
    flipped = flip_dual(t, 0)
    assert flipped.legs[0].dual is False


def test_a_product_dual_leg_now_expands():
    """The other half of the change: ``to_dense`` on a product's dual leg used to raise.

    A product forwards ``DualBasis`` since #312, so the refusal this file's neighbour
    above used to ride on is gone — recorded here so the widening is a test rather than
    an absence.
    """
    t = SymmetricTensor.random((Leg(PROD_SPACE, OUT, dual=True), Leg(PROD_SPACE, IN)), seed=10)
    assert t.to_dense().shape == t.shape


# --- the capability gate --------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FusionOnly:
    """Fusion rules and the dual map, nothing else — no flip scalar."""

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


def test_flip_dual_refuses_a_provider_without_the_flip_capabilities():
    space = GradedSpace.new(FusionOnly(), {TrivialSector(): 2})
    t = SymmetricTensor.zeros((Leg(space, OUT), Leg(space, IN)))
    with pytest.raises(CapabilityError) as err:
        flip_dual(t, 0)
    message = str(err.value)
    assert "FSIndicatorData" in message
    assert "chi_a * theta_a" in message
    assert "wrong sign" in message


# --- structural invariants ------------------------------------------------------


def test_flip_dual_preserves_blocks_order_shapes_names_sides():
    legs = (
        Leg(SU2_FREE, OUT, name="a"),
        Leg(SU2_BOND, IN, name="b"),
        Leg(SU2_BOND, OUT, name="c"),
        Leg(SU2_FREE, IN, name="d"),
    )
    t = SymmetricTensor.random(legs, seed=13)
    f = flip_dual(t, (1, 2))
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


def test_flip_dual_no_axes_is_identity():
    t = SymmetricTensor.random((Leg(U1_BOND, OUT), Leg(U1_BOND, IN)), seed=14)
    assert flip_dual(t, ()) is t


def test_flip_dual_axis_validation():
    t = SymmetricTensor.random((Leg(U1_BOND, OUT, name="p"), Leg(U1_BOND, IN, name="p")), seed=15)
    with pytest.raises(ValueError, match="out of range"):
        flip_dual(t, 2)
    with pytest.raises(ValueError, match="repeated"):
        flip_dual(t, (0, 0))
    with pytest.raises(ValueError, match="no leg is named"):
        flip_dual(t, "q")
    with pytest.raises(ValueError, match="ambiguous"):
        flip_dual(t, "p")


def test_flip_dual_by_leg_name():
    t = SymmetricTensor.random((Leg(U1_BOND, OUT, name="p"), Leg(U1_BOND, IN, name="q")), seed=16)
    by_name, by_axis = flip_dual(t, "q"), flip_dual(t, 1)
    assert by_name.structure == by_axis.structure
    for x, y in zip(by_name.blocks, by_axis.blocks, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_flip_dual_keeps_a_real_dtype_real():
    space = GradedSpace.new(SU2, {HALF: 2})
    t = SymmetricTensor.random((Leg(space, OUT, dual=True), Leg(space, IN)), seed=17)
    flipped = flip_dual(t, 0)  # pays chi = -1
    for block in flipped.blocks:
        assert np.asarray(block).dtype == np.float64


# --- backends -------------------------------------------------------------------


def test_flip_dual_on_jax_matches_numpy_and_jits():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  # registration is the import's side effect

    space = GradedSpace.new(SU2, {HALF: 2})
    t = SymmetricTensor.random((Leg(space, OUT, dual=True), Leg(space, IN)), seed=18)
    on_jax = flip_dual(t.to_backend("jax"), 0)
    assert on_jax.backend == "jax"
    assert tenet.allclose(on_jax.to_backend("numpy"), flip_dual(t, 0))
    jitted = jax.jit(lambda x: flip_dual(x, 0))(t.to_backend("jax"))
    assert tenet.allclose(jitted.to_backend("numpy"), flip_dual(t, 0))


# --- the matrix route -----------------------------------------------------------


def block_route(t, picked, inv):
    """``flip_dual`` written the way it was before it read ``data``: one scalar per block.

    The reference the matrix route is held bit-identical to. It is spelled out here
    rather than imported because it is exactly what the library no longer does.
    """
    structure = t.structure
    provider = structure.provider
    legs = list(structure.legs)
    for ax in picked:
        leg = legs[ax]
        relabelled = GradedSpace.new(
            provider, tuple((provider.dual(a), m) for a, m in leg.space.sectors)
        )
        legs[ax] = dataclasses.replace(leg, space=relabelled, dual=not leg.dual)
    new_structure = TensorStructure(tuple(legs))

    tree_pos = {ax: k for k, ax in enumerate(structure.out_axes)}
    tree_pos |= {ax: k for k, ax in enumerate(structure.in_axes)}

    blocks = []
    for key, block in zip(structure.block_order, t.blocks, strict=True):
        factor = 1.0
        for ax in picked:
            leg = structure.legs[ax]
            out = leg.side is OUT
            tree = key.output_tree if out else key.input_tree
            a = tree.uncoupled[tree_pos[ax]]
            base = complex(provider.frobenius_schur(a) * provider.twist(a))
            if not out:
                base = base.conjugate()
            if not inv:
                factor *= base if leg.dual else 1.0
            else:
                factor *= 1.0 if leg.dual else base.conjugate()
        if factor != 1:
            block = block * (factor.real if factor.imag == 0 else factor)
        blocks.append(block)
    return SymmetricTensor(new_structure, tuple(blocks))


@pytest.mark.parametrize("space", [FZ2_SPACE, FPROD_SPACE], ids=["fz2", "fz2xu1xsu2"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128], ids=["real", "complex"])
def test_the_matrix_route_is_bit_identical_to_the_block_route(space, dtype):
    """Every flip of a four-leg tensor, against the implementation this replaced.

    On the two gradings where the coefficient is not 1 and where a wrong one would hide:
    ``fZ2`` pays the twist on an odd line, and the product pays it distributed over three
    factors. ``np.array_equal`` and not ``allclose``: the flip is a multiplication by
    ``+-1``, so anything but equality is a different operation.
    """
    legs = (
        Leg(space, OUT),
        Leg(space, OUT, dual=True),
        Leg(space, IN),
        Leg(space, IN, dual=True),
    )
    t = SymmetricTensor.random(legs, seed=41).astype(dtype)
    for r in range(1, 5):
        for picked in itertools.combinations(range(4), r):
            for inv in (False, True):
                want = block_route(t, picked, inv)
                got = flip_dual(t, picked, inv=inv)
                assert got.structure == want.structure, (picked, inv)
                for i, (g, w) in enumerate(zip(got.blocks, want.blocks, strict=True)):
                    assert np.asarray(g).dtype == np.asarray(w).dtype, (picked, inv, i)
                    assert np.array_equal(np.asarray(g), np.asarray(w)), (picked, inv, i)


def test_a_u1_leg_whose_duals_sort_in_reverse_keeps_every_block_on_its_key():
    """The case the matrix route had to derive rather than assume.

    ``dual(q) = -q`` reverses the sector order of ``{0, 1, 2}``, and ``GradedSpace.new``
    sorts, so the flipped leg's *space* lists its sectors in the opposite order and with
    the degeneracies permuted with them. Nothing downstream of ``Leg.fused_sector`` sees
    that: the block keys, their shapes and the coupled-sector layout all come back
    identical, which is why the flip is a scaling of the stored matrices and not a
    gather. Ragged degeneracies, so a block landing on the wrong key would be a wrong
    shape and not merely a wrong number.
    """
    space = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1, U1Sector(2): 3})
    t = SymmetricTensor.random((Leg(space, OUT), Leg(space, OUT), Leg(space, IN)), seed=42)
    f = flip_dual(t, 0)

    assert [q for q, _ in space.sectors] == [U1Sector(0), U1Sector(1), U1Sector(2)]
    assert [q for q, _ in f.legs[0].space.sectors] == [
        U1Sector(-2),
        U1Sector(-1),
        U1Sector(0),
    ]
    assert [m for _, m in f.legs[0].space.sectors] == [3, 1, 2]

    assert f.structure.block_order == t.structure.block_order
    assert map_layout(f.structure).rows == map_layout(t.structure).rows
    assert map_layout(f.structure).cols == map_layout(t.structure).cols
    for key in t.structure.block_order:
        assert f.structure.block_shape(key) == t.structure.block_shape(key)
        # chi * theta is 1 on U(1), so a block that landed on its own key is untouched
        assert np.array_equal(np.asarray(f.block(key)), np.asarray(t.block(key)))
    assert flip_dual(f, 0, inv=True).structure == t.structure


def test_a_flip_that_costs_nothing_keeps_the_storage_it_was_handed():
    """A bosonic grading's flip is a relabel, and a relabel moves no array.

    ``chi * theta`` is 1 everywhere on U(1), so there is nothing to multiply by and the
    result holds the very arrays its operand did -- neither gathered nor cut, whichever
    form the operand was already in.
    """
    space = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 3})
    t = SymmetricTensor.random((Leg(space, OUT), Leg(space, IN)), seed=43)
    f = flip_dual(t, 0)
    assert all(a is b for a, b in zip(f.data, t.data, strict=True))
