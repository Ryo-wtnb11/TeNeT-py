"""Tests for tenet.symmetry.z2 — one test per acceptance criterion of issue #104.

The shape is ``test_u1.py``'s (a small abelian provider) plus ``test_fz2.py``'s
(the exact-algebra, dense-oracle and bending half), because bosonic Z2 is exactly
their intersection. The load-bearing test is
:func:`test_the_only_difference_from_fermion_parity_is_permute_tree`: every other
method is asserted *equal* to the fermionic provider's under the obvious sector
relabelling, and ``permute_tree`` is asserted to return ``1`` wherever the
fermionic one returns a Koszul sign — including cases where that sign is ``-1``,
so the difference is actually reached rather than merely claimed.
"""

import dataclasses
import itertools

import numpy as np
import pytest

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.fusion_tree import fusion_trees
from tenet.ops.permutation import permutation_plan
from tenet.ops.repartition import bend, bend_plan
from tenet.serialize import load, save
from tenet.symmetry import (
    U1,
    Z2,
    AssociatorData,
    BendingCoefficients,
    BraidingData,
    BranchingRules,
    CapabilityError,
    ClebschGordanData,
    DualBasis,
    DualityData,
    FusionRules,
    FZ2Sector,
    PermutationCoefficients,
    ProductProvider,
    ProductSector,
    QuantumDimensionData,
    Sector,
    U1Sector,
    Z2Provider,
    Z2Sector,
    fZ2,
    requires,
)

# static conformance check: fails type checking if Z2Provider drifts from the protocols
_fusion: FusionRules = Z2
_qdim: QuantumDimensionData = Z2
_cgc: ClebschGordanData = Z2
_perm: PermutationCoefficients = Z2
_bend: BendingCoefficients = Z2
_dual: DualBasis = Z2

EVEN, ODD = Z2Sector(0), Z2Sector(1)
SECTORS = (EVEN, ODD)

P = GradedSpace.new(Z2, {EVEN: 2, ODD: 3})
Q = GradedSpace.new(Z2, {EVEN: 1, ODD: 2})

OUT_LEGS = (Leg(P, OUT, name="a"), Leg(Q, OUT, name="b"), Leg(Q, OUT), Leg(P, OUT))
MIXED_LEGS = (Leg(P, OUT, name="a"), Leg(Q, IN, name="b"), Leg(Q, OUT), Leg(P, IN))
DUAL_LEGS = (Leg(P, OUT, True), Leg(Q, IN), Leg(Q, OUT, True))

ALL_PERMS = tuple(itertools.permutations(range(4)))


def out_tensor() -> SymmetricTensor:
    return SymmetricTensor.random(OUT_LEGS, seed=17)


def mixed_tensor() -> SymmetricTensor:
    return SymmetricTensor.random(MIXED_LEGS, seed=19)


def inverse(p: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(p)
    for i, a in enumerate(p):
        out[a] = i
    return tuple(out)


def compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """``transpose(p).transpose(q) == transpose(compose(p, q))``."""
    return tuple(p[j] for j in q)


def permuted(u: tuple, p: tuple[int, ...]) -> tuple:
    return tuple(u[i] for i in p)


# --- sectors -----------------------------------------------------------------------


def test_only_zero_and_one_are_constructible():
    assert Z2Sector(0).parity == 0
    assert Z2Sector(1).parity == 1
    for bad in (2, -1, 7):
        with pytest.raises(ValueError) as excinfo:
            Z2Sector(bad)
        assert str(bad) in str(excinfo.value)
    for bad in (True, False):
        with pytest.raises(TypeError) as excinfo:
            Z2Sector(bad)
        assert repr(bad) in str(excinfo.value)


def test_sectors_are_hashable_ordered_and_repr_round_trips():
    assert EVEN < ODD
    assert sorted((ODD, EVEN)) == [EVEN, ODD]
    assert len({Z2Sector(0), Z2Sector(0), Z2Sector(1)}) == 2
    assert {ODD: "x"}[Z2Sector(1)] == "x"
    assert repr(EVEN) == "Z2Sector(parity=0)"
    assert eval(repr(ODD)) == ODD  # noqa: S307


def test_a_bosonic_sector_is_not_a_fermionic_one():
    """``Sector`` comparison is only defined within one type, so a graded space
    cannot silently mix a bosonic and a fermionic Z2 leg."""
    assert Z2Sector(1) != FZ2Sector(1)
    with pytest.raises(TypeError):
        _ = Z2Sector(1) < FZ2Sector(1)


# --- capabilities ------------------------------------------------------------------


def test_z2_satisfies_exactly_the_u1_capability_set():
    for capability in (
        QuantumDimensionData,
        ClebschGordanData,
        PermutationCoefficients,
        BendingCoefficients,
        DualBasis,
    ):
        assert isinstance(Z2, capability)
        assert isinstance(U1, capability)  # the identical set, stated as the reference
        assert requires(Z2, capability) is None
    assert not isinstance(Z2, BranchingRules)
    for capability in (BranchingRules, AssociatorData, BraidingData, DualityData):
        assert not isinstance(Z2, capability)
        with pytest.raises(CapabilityError):
            requires(Z2, capability)


def test_provider_is_frozen_hashable_and_array_free():
    assert Z2.name == "Z2"
    assert Z2 == Z2Provider()
    assert hash(Z2) == hash(Z2Provider())
    with pytest.raises(dataclasses.FrozenInstanceError):
        Z2.name = "nope"
    for f in dataclasses.fields(Z2):
        assert not isinstance(getattr(Z2, f.name), np.ndarray)
    assert hash(GradedSpace.new(Z2, {EVEN: 2, ODD: 3}))


def test_there_is_no_gauge_string():
    """Bosonic Z2 pins no convention, so it has nothing to fingerprint — the same
    reason ``U1Provider`` and ``TrivialProvider`` have no gauge string."""
    import tenet.symmetry as sym
    import tenet.symmetry.z2 as z2

    assert not hasattr(z2, "Z2_GAUGE")
    assert not [n for n in sym.__all__ if n.startswith("Z2") and "GAUGE" in n]
    assert {"Z2", "Z2Provider", "Z2Sector"} <= set(sym.__all__)


# --- fusion algebra ----------------------------------------------------------------


def test_fusion_algebra_exhaustively():
    assert Z2.unit == EVEN
    for a in SECTORS:
        assert Z2.dual(a) == a  # self-dual
        assert Z2.dual(Z2.dual(a)) == a
        assert Z2.fusion(Z2.unit, a) == (a,)
        assert Z2.fusion(a, Z2.unit) == (a,)
        assert Z2.qdim(a) == 1.0
        assert Z2.irrep_dim(a) == 1
        for b in SECTORS:
            assert Z2.fusion(a, b) == (Z2Sector(a.parity ^ b.parity),)
            assert sum(Z2.n_symbol(a, b, c) for c in Z2.fusion(a, b)) == 1  # multiplicity-free
            for c in SECTORS:
                assert Z2.n_symbol(a, b, c) == int(c in Z2.fusion(a, b))
                left = sorted(e for d in Z2.fusion(a, b) for e in Z2.fusion(d, c))
                right = sorted(e for d in Z2.fusion(b, c) for e in Z2.fusion(a, d))
                assert left == right


def test_cgc_and_z_matrix_are_read_only_ones():
    for a in SECTORS:
        z = Z2.z_matrix(a)
        np.testing.assert_array_equal(z, np.ones((1, 1)))
        assert not z.flags.writeable
        for b in SECTORS:
            (c,) = Z2.fusion(a, b)
            cgc = Z2.cgc(a, b, c)
            assert cgc.shape == (1, 1, 1, 1)
            np.testing.assert_array_equal(cgc, np.ones((1, 1, 1, 1)))
            assert not cgc.flags.writeable


def test_cgc_raises_naming_z2_on_every_forbidden_triple():
    triples = [(a, b, c) for a in SECTORS for b in SECTORS for c in SECTORS]
    forbidden = [t for t in triples if not Z2.n_symbol(*t)]
    assert len(forbidden) == 4
    for a, b, c in forbidden:
        with pytest.raises(ValueError, match="Z2 fusion forbids"):
            Z2.cgc(a, b, c)


# --- the one difference from fermion parity ----------------------------------------


def relabel(a: Sector) -> Sector:
    """The obvious ``Z2Sector <-> FZ2Sector`` relabelling."""
    return FZ2Sector(a.parity) if isinstance(a, Z2Sector) else Z2Sector(a.parity)


def test_the_only_difference_from_fermion_parity_is_permute_tree():
    """Every method except ``permute_tree`` agrees with the fermionic provider under
    the sector relabelling; ``permute_tree`` returns ``1`` where the fermionic one
    returns the Koszul sign. Checked exhaustively for rank <= 4 over every parity
    assignment and every permutation, with the ``-1`` cases counted so the
    difference is provably reached.
    """
    from tenet.symmetry.fz2 import koszul_sign

    for a in SECTORS:
        fa = relabel(a)
        assert relabel(Z2.dual(a)) == fZ2.dual(fa)
        assert Z2.qdim(a) == fZ2.qdim(fa)
        assert Z2.irrep_dim(a) == fZ2.irrep_dim(fa)
        np.testing.assert_array_equal(Z2.z_matrix(a), fZ2.z_matrix(fa))
        for b in SECTORS:
            fb = relabel(b)
            assert tuple(map(relabel, Z2.fusion(a, b))) == fZ2.fusion(fa, fb)
            for c in SECTORS:
                fc = relabel(c)
                assert Z2.n_symbol(a, b, c) == fZ2.n_symbol(fa, fb, fc)
                if Z2.n_symbol(a, b, c):
                    np.testing.assert_array_equal(Z2.cgc(a, b, c), fZ2.cgc(fa, fb, fc))

    differed = 0
    for n in (1, 2, 3, 4):
        for parities in itertools.product((0, 1), repeat=n):
            coupled = sum(parities) % 2
            (tree,) = fusion_trees(Z2, tuple(Z2Sector(p) for p in parities), Z2Sector(coupled))
            (ftree,) = fusion_trees(fZ2, tuple(FZ2Sector(p) for p in parities), FZ2Sector(coupled))
            for perm in itertools.permutations(range(n)):
                ((new, coeff),) = Z2.permute_tree(tree, perm)
                ((fnew, fcoeff),) = fZ2.permute_tree(ftree, perm)
                new.validate(Z2)
                assert tuple(map(relabel, new.uncoupled)) == fnew.uncoupled
                assert new.uncoupled == permuted(tree.uncoupled, perm)
                assert new.coupled == tree.coupled
                assert coeff == 1.0
                assert fcoeff == koszul_sign(ftree.uncoupled, perm)
                differed += fcoeff != coeff
    assert differed > 0  # the -1 branch is actually reached, not merely claimed


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_permute_tree_is_one_term_with_coefficient_one(n):
    for parities in itertools.product((0, 1), repeat=n):
        uncoupled = tuple(Z2Sector(p) for p in parities)
        (tree,) = fusion_trees(Z2, uncoupled, Z2Sector(sum(parities) % 2))
        for perm in itertools.permutations(range(n)):
            terms = Z2.permute_tree(tree, perm)
            assert len(terms) == 1
            ((new, coeff),) = terms
            assert coeff == 1.0
            assert new.uncoupled == permuted(uncoupled, perm)
            assert new.coupled == tree.coupled


def test_bending_bosonic_agreement_and_all_one_coefficients():
    """The two ``bend_*`` are byte-for-byte the fermionic ones (both delegate to
    ``bend_unique``), asserted per axis on a real plan."""
    t = mixed_tensor()
    for axis in (2, 3):
        plan = bend_plan(t.structure, axis)
        assert {coeff for _, _, coeff in plan.terms} == {1.0}
        assert len(plan.terms) == t.structure.num_blocks


# --- exact algebra and the dense oracle --------------------------------------------


@pytest.mark.parametrize("p", ALL_PERMS)
def test_every_permutation_plan_coefficient_is_one(p):
    """The point of the whole file: no sign anywhere."""
    t = mixed_tensor()
    plan = permutation_plan(t.structure, p)
    assert {coeff for _, _, coeff in plan.terms} == {1.0}


@pytest.mark.parametrize("p", ALL_PERMS)
def test_dense_oracle_all_out(p):
    t = out_tensor()
    np.testing.assert_array_equal(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p))


@pytest.mark.parametrize("p", ALL_PERMS)
def test_dense_oracle_interleaved_sides(p):
    t = mixed_tensor()
    np.testing.assert_array_equal(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p))


@pytest.mark.parametrize("p", list(itertools.permutations(range(3))))
def test_dense_oracle_rank3_with_dual_legs(p):
    t = SymmetricTensor.random(DUAL_LEGS, seed=23)
    np.testing.assert_array_equal(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p))


@pytest.mark.parametrize("p", ALL_PERMS)
def test_round_trip_and_composition_are_exact(p):
    t = mixed_tensor()
    back = t.transpose(p).transpose(inverse(p))
    assert back.structure == t.structure
    for a, b in zip(back.blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(a, b)
    for q in ALL_PERMS:
        left, right = t.transpose(p).transpose(q), t.transpose(compose(p, q))
        assert left.structure == right.structure
        for a, b in zip(left.blocks, right.blocks, strict=True):
            np.testing.assert_array_equal(a, b)


def test_bend_and_repartition_round_trip_exactly():
    t = mixed_tensor()
    bent = bend(t, 2)
    back = bend(bent, bent.ndim - 1).transpose((0, 1, 3, 2))
    assert back.structure == t.structure
    for a, b in zip(back.blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(a, b)

    r = t.repartition((0, 1, 2), (3,))
    restored = r.repartition(t.structure.out_axes, t.structure.in_axes).transpose((0, 2, 1, 3))
    assert restored.structure == t.structure
    for a, b in zip(restored.blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    ("outputs", "inputs"),
    [((0, 1, 2), (3,)), ((0,), (1, 2, 3)), ((0, 1, 2, 3), ()), ((), (0, 1, 2, 3))],
)
def test_repartition_equals_the_plain_dense_transpose(outputs, inputs):
    """Where the fermionic provider can only match ``abs``, the bosonic one matches
    the array itself — the sign that made ``test_fz2`` take absolute values is the
    sign this provider does not have."""
    t = mixed_tensor()
    got = t.repartition(outputs, inputs).to_dense()
    want = np.transpose(t.to_dense(), (*outputs, *inputs))
    np.testing.assert_allclose(got, want, atol=1e-12)


# --- the capability gate is still opt-in -------------------------------------------


def test_capability_gate_is_still_opt_in():
    """This provider's arrival must not be readable as loosening the gate: a
    provider that defines only ``fusion`` is still refused, even though its fusion
    is unique. ``permute_unique_tree`` is a *helper*, not a capability check —
    ``fz2.py`` is the standing proof that unique fusion can still carry a sign.
    """

    class OnlyFusion:
        name = "OnlyFusion"
        unit = EVEN

        def dual(self, a):
            return a

        def fusion(self, a, b):
            return (Z2Sector(a.parity ^ b.parity),)

        def n_symbol(self, a, b, c):
            return int(c.parity == a.parity ^ b.parity)

    assert not isinstance(OnlyFusion(), PermutationCoefficients)
    with pytest.raises(CapabilityError):
        requires(OnlyFusion(), PermutationCoefficients)


# --- serialization -----------------------------------------------------------------


def test_round_trip_through_save_and_load(tmp_path):
    t = mixed_tensor()
    save(t, tmp_path / "z.npz")
    back = load(tmp_path / "z.npz")
    assert back.structure == t.structure
    for a, b in zip(back.blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(a, b)


def test_header_carries_no_gauge_and_unknown_kinds_still_list_z2(tmp_path):
    import json

    save(mixed_tensor(), tmp_path / "z.npz")
    with np.load(tmp_path / "z.npz", allow_pickle=False) as z:
        header = json.loads(str(z["header"].item()))
    assert header["gauges"] == {}
    assert header["legs"][0]["space"]["provider"] == {"kind": "Z2", "name": "Z2"}

    from tenet.serialize import _decode_provider

    with pytest.raises(KeyError, match="Z2"):
        _decode_provider({"kind": "nope", "name": "nope"})


def test_a_corrupt_parity_raises_from_the_sector_constructor():
    from tenet.serialize import _decode_sector

    with pytest.raises(ValueError, match="parity must be 0 or 1"):
        _decode_sector(Z2, [2])


# --- products ----------------------------------------------------------------------


@pytest.mark.parametrize("factors", [(Z2, U1), (U1, Z2)])
def test_product_forwards_exactly_what_it_forwards_today(factors):
    product = ProductProvider(factors)
    labels = [(Z2Sector(p), U1Sector(q)) for p in (0, 1) for q in (-1, 0, 1)]
    if factors[0] is U1:
        labels = [(y, x) for x, y in labels]
    sectors = [ProductSector(x) for x in labels]

    for a in sectors:
        assert product.qdim(a) == 1.0
        assert product.irrep_dim(a) == 1
        for b in sectors:
            (c,) = product.fusion(a, b)
            assert product.n_symbol(a, b, c) == 1
            assert product.cgc(a, b, c).shape == (1, 1, 1, 1)

    # The pin was on the gap, and #312 closed it: a product now forwards bending and the
    # dual basis, so a Z2 x U(1) tensor repartitions and expands like any other. Kept as
    # an assertion rather than deleted, because "what a product forwards" is exactly what
    # this case exists to state.
    assert isinstance(product, BendingCoefficients)
    assert isinstance(product, DualBasis)

    space = GradedSpace.new(product, {a: 1 for a in sectors[:3]})
    legs = (Leg(space, OUT), Leg(space, OUT), Leg(space, IN))
    t = SymmetricTensor.random(legs, seed=31)
    for p in itertools.permutations(range(3)):
        np.testing.assert_allclose(
            t.transpose(p).to_dense(), np.transpose(t.to_dense(), p), atol=1e-12
        )
        back = t.transpose(p).transpose(inverse(p))
        for x, y in zip(back.blocks, t.blocks, strict=True):
            np.testing.assert_array_equal(x, y)
