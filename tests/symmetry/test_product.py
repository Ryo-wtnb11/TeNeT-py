"""Tests for tenet.symmetry.product — one test per acceptance criterion of issue #40.

Three oracles, all written without reference to the implementation:

* ``independent_cgc_entry`` — a plain nested loop over component indices, so the
  ``cgc`` check is not a restatement of the einsum/reshape.
* ``supersign`` — the dense-side Koszul sign of #39, re-derived here from each
  axis's parity vector, restricted to pairs on the same side (TeNeT-py's stated
  convention). It reads the fZ2 component of a product sector, which is the only
  thing that makes it a product test rather than a copy of ``test_fz2``.
* ``u1_shadow`` — a fermionic ``U(1) x fZ2`` tensor re-expressed on a plain
  ``U(1)`` space (parity is a function of charge there), which pins the product's
  dense layout to the mixed-radix layout the flattening convention promises.
"""

import dataclasses
import itertools
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pytest
from helpers import dense_repartition

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.fusion_tree import FusionTree, coupled_sectors, fusion_trees
from tenet.ops.permutation import permutation_plan
from tenet.structure import TensorStructure
from tenet.symmetry import (
    SU2,
    U1,
    BendingCoefficients,
    CapabilityError,
    ClebschGordanData,
    DualBasis,
    FusionRules,
    FZ2Sector,
    PermutationCoefficients,
    ProductProvider,
    ProductSector,
    QuantumDimensionData,
    SU2Sector,
    U1Sector,
    fZ2,
)
from tenet.symmetry.base import Sector, permute_unique_tree
from tenet.symmetry.coherence import supports
from tenet.symmetry.product import assemble, mu_decode, mu_encode, project

# static conformance: fails type checking if ProductProvider drifts from the protocols
_fusion: FusionRules = ProductProvider((U1, U1))
_qdim: QuantumDimensionData = ProductProvider((U1, U1))
_cgc: ClebschGordanData = ProductProvider((U1, U1))

UU = ProductProvider((U1, U1))
UF = ProductProvider((U1, fZ2))
SU = ProductProvider((SU2, U1))

ALL_PERMS = tuple(itertools.permutations(range(4)))
# legs are (OUT, IN, OUT, IN): case A keeps the relative order inside {0, 2} and {1, 3}
CASE_A = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2), (1, 0, 2, 3), (1, 0, 3, 2), (1, 3, 0, 2))


def ps(*components: Sector) -> ProductSector:
    return ProductSector(components)


def q(c: int) -> U1Sector:
    return U1Sector(c)


def uu(c0: int, c1: int) -> ProductSector:
    return ps(q(c0), q(c1))


def uf(c: int, p: int) -> ProductSector:
    return ps(q(c), FZ2Sector(p))


def su(two_j: int, c: int) -> ProductSector:
    return ps(SU2Sector(two_j), q(c))


# --- spaces and tensors -------------------------------------------------------------

UU_SPACE = GradedSpace.new(UU, {uu(0, 0): 2, uu(1, 0): 1, uu(0, 1): 1, uu(1, 1): 1})
# a genuine "fermionic U(1)": parity is charge mod 2, docs/design.md's motivating example
UF_SPACE = GradedSpace.new(UF, {uf(0, 0): 2, uf(1, 1): 1, uf(-1, 1): 1})
SU_SPACE = GradedSpace.new(SU, {su(0, 0): 1, su(1, 1): 1})

UU_LEGS = (Leg(UU_SPACE, OUT), Leg(UU_SPACE, IN), Leg(UU_SPACE, OUT), Leg(UU_SPACE, IN))
UF_LEGS = (Leg(UF_SPACE, OUT), Leg(UF_SPACE, IN), Leg(UF_SPACE, OUT), Leg(UF_SPACE, IN))
SU_LEGS = (Leg(SU_SPACE, OUT), Leg(SU_SPACE, IN), Leg(SU_SPACE, OUT), Leg(SU_SPACE, IN))


def uu_tensor() -> SymmetricTensor:
    return SymmetricTensor.random(UU_LEGS, seed=11)


def uf_tensor() -> SymmetricTensor:
    return SymmetricTensor.random(UF_LEGS, seed=13)


def su_tensor() -> SymmetricTensor:
    return SymmetricTensor.random(SU_LEGS, seed=17)


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def inverse(p: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(p)
    for i, a in enumerate(p):
        out[a] = i
    return tuple(out)


def compose(p: tuple[int, ...], r: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(p[j] for j in r)


def all_trees(provider, uncoupled) -> tuple[FusionTree, ...]:
    return tuple(
        t
        for c in coupled_sectors(provider, uncoupled)
        for t in fusion_trees(provider, uncoupled, c)
    )


def supersign(legs, p, parity_of) -> np.ndarray:
    """Dense Koszul sign array shaped like ``np.transpose(dense, p)``, counting only
    inversions between two axes of the same side (#39's convention)."""
    pars = [
        np.concatenate([np.full(m, parity_of(a)) for a, m in legs[ax].space.sectors]) for ax in p
    ]
    sides = [legs[ax].side for ax in p]
    n = len(p)
    sign = np.ones(tuple(len(v) for v in pars))
    for j in range(n):
        for k in range(j + 1, n):
            if p[j] <= p[k] or sides[j] is not sides[k]:
                continue
            sj, sk = [1] * n, [1] * n
            sj[j], sk[k] = len(pars[j]), len(pars[k])
            sign = sign * (-1.0) ** (pars[j].reshape(sj) * pars[k].reshape(sk))
    return sign


def u1_shadow(t: SymmetricTensor) -> SymmetricTensor:
    """The same blocks on a plain U(1) space carrying only the charge component.

    Legitimate exactly because :data:`UF_SPACE` is a *fermionic* U(1): parity is a
    function of charge, so dropping the fZ2 component loses no label and the two
    structures are block-for-block isomorphic. The dense arrays must then be equal
    entry for entry, which is what pins the product's slab layout.
    """
    legs = tuple(
        Leg(GradedSpace.new(U1, {a.components[0]: m for a, m in leg.space.sectors}), leg.side)
        for leg in t.legs
    )
    shadow = TensorStructure(legs)
    index = {
        tuple(a.charge for a in shadow.axis_sectors(k)): i for i, k in enumerate(shadow.block_order)
    }
    blocks: list = [None] * shadow.num_blocks
    for key, block in t.items():
        charges = tuple(a.components[0].charge for a in t.structure.axis_sectors(key))
        blocks[index[charges]] = block
    assert all(b is not None for b in blocks)
    return SymmetricTensor(shadow, tuple(blocks))


# --- test-local stub factors --------------------------------------------------------


@dataclass(frozen=True, slots=True, order=True)
class MSector(Sector):
    label: int


@dataclass(frozen=True, slots=True)
class MultStub:
    """Z2 fusion with ``n_symbol == 2`` on the single triple ``(1, 1) -> 0``.

    The only thing in the repository that reaches ``n > 1``; it exists so the
    mixed-radix ``mu`` machinery is covered rather than merely written.
    """

    name: str = "Mult"

    @property
    def unit(self) -> MSector:
        return MSector(0)

    def dual(self, a: MSector) -> MSector:
        return a

    def fusion(self, a: MSector, b: MSector) -> tuple[MSector, ...]:
        return (MSector(a.label ^ b.label),)

    def n_symbol(self, a: MSector, b: MSector, c: MSector) -> int:
        if c.label != a.label ^ b.label:
            return 0
        return 2 if (a.label, b.label) == (1, 1) else 1


@dataclass(frozen=True, slots=True, order=True)
class ASector(Sector):
    label: int


@dataclass(frozen=True, slots=True)
class SplitStub:
    """Everything fuses to everything, and ``permute_tree`` returns *two* terms.

    A rank-3 tree therefore has two siblings differing only in their inner line,
    which is what makes the distributed product in ``ProductProvider.permute_tree``
    live code rather than a loop that always runs once.
    """

    coeffs: tuple[float, ...] = (0.6, 0.8)
    name: str = "Split"

    @property
    def unit(self) -> ASector:
        return ASector(0)

    def dual(self, a: ASector) -> ASector:
        return a

    def fusion(self, a: ASector, b: ASector) -> tuple[ASector, ...]:
        return (ASector(0), ASector(1))

    def n_symbol(self, a: ASector, b: ASector, c: ASector) -> int:
        return 1

    def permute_tree(self, tree, perm):
        uncoupled = tuple(tree.uncoupled[i] for i in perm)
        trees = fusion_trees(self, uncoupled, tree.coupled)
        return tuple(zip(trees, self.coeffs, strict=True))


@dataclass(frozen=True, slots=True)
class BareStub:
    """Unique fusion, one term, no ``qdim``/``cgc`` at all — the missing-capability side."""

    name: str = "Bare"

    @property
    def unit(self) -> ASector:
        return ASector(0)

    def dual(self, a: ASector) -> ASector:
        return a

    def fusion(self, a: ASector, b: ASector) -> tuple[ASector, ...]:
        return (ASector(a.label ^ b.label),)

    def n_symbol(self, a: ASector, b: ASector, c: ASector) -> int:
        return int(c.label == a.label ^ b.label)

    def permute_tree(self, tree, perm):
        return permute_unique_tree(self, tree, perm)


@dataclass(frozen=True, slots=True)
class StillStub:
    """``qdim``/``cgc`` but *no* ``permute_tree`` — the missing-permutation side.

    Every shipped provider now has ``permute_tree`` (#36 gave SU(2) one), so the
    "isinstance optimistic, call authoritative" contract needs a factor that still
    refuses; this is it.
    """

    name: str = "Still"

    @property
    def unit(self) -> ASector:
        return ASector(0)

    def dual(self, a: ASector) -> ASector:
        return a

    def fusion(self, a: ASector, b: ASector) -> tuple[ASector, ...]:
        return (ASector(a.label ^ b.label),)

    def n_symbol(self, a: ASector, b: ASector, c: ASector) -> int:
        return int(c.label == a.label ^ b.label)

    def qdim(self, a: ASector) -> float:
        return 1.0

    def irrep_dim(self, a: ASector) -> int:
        return 1

    def cgc(self, a: ASector, b: ASector, c: ASector) -> np.ndarray:
        if not self.n_symbol(a, b, c):
            raise ValueError(f"{self.name}: forbidden triple")
        return np.ones((1, 1, 1, 1))


# --- provider value semantics -------------------------------------------------------


def test_provider_is_frozen_hashable_and_array_free():
    assert ProductProvider((U1, U1)) == ProductProvider((U1, U1))
    assert hash(ProductProvider((U1, U1))) == hash(ProductProvider((U1, U1)))
    assert UU != UF
    with pytest.raises(dataclasses.FrozenInstanceError):
        UU.factors = (U1,)
    for f in dataclasses.fields(UU):
        assert not isinstance(getattr(UU, f.name), np.ndarray)
    assert hash(UU_SPACE)
    assert hash(TensorStructure(UU_LEGS))


def test_one_factor_is_refused():
    with pytest.raises(ValueError) as excinfo:
        ProductProvider((U1,))
    assert "1" in str(excinfo.value)
    with pytest.raises(ValueError):
        ProductProvider(())


def test_factors_are_normalized_to_a_tuple():
    assert ProductProvider([U1, fZ2]) == UF


def test_name_is_a_property_not_a_field():
    assert UF.name == "U1 x fZ2"
    assert SU.name == "SU2 x U1"
    assert "name" not in {f.name for f in dataclasses.fields(UU)}


# --- sectors ------------------------------------------------------------------------


def test_product_sector_is_frozen_slotted_and_lexicographic():
    a = uu(0, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.components = ()
    assert not hasattr(a, "__dict__")
    unsorted = [uu(1, 0), uu(0, 1), uu(0, 0), uu(1, 1)]
    assert sorted(unsorted) == [uu(0, 0), uu(0, 1), uu(1, 0), uu(1, 1)]
    assert len({uu(0, 1), uu(0, 1)}) == 1


def test_product_sector_orders_trees_and_block_keys():
    order = TensorStructure(SU_LEGS).block_order
    assert order == tuple(sorted(order))
    assert len(order) > 1
    trees = all_trees(SU, (su(1, 1), su(1, 1), su(1, 1)))
    assert trees == tuple(sorted(set(trees)))


# --- componentwise algebra ----------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "sectors"),
    [
        (UU, (uu(0, 0), uu(1, 0), uu(0, 1), uu(-1, 2))),
        (UF, (uf(0, 0), uf(1, 1), uf(-1, 1), uf(2, 0))),
        (SU, (su(0, 0), su(1, 1), su(2, -1))),
    ],
)
def test_componentwise_algebra(provider, sectors):
    factors = provider.factors
    assert provider.unit == ProductSector(tuple(f.unit for f in factors))
    for a in sectors:
        assert provider.dual(a) == ProductSector(
            tuple(f.dual(x) for f, x in zip(factors, a.components, strict=True))
        )
        assert provider.qdim(a) == math.prod(
            f.qdim(x) for f, x in zip(factors, a.components, strict=True)
        )
        assert provider.irrep_dim(a) == math.prod(
            f.irrep_dim(x) for f, x in zip(factors, a.components, strict=True)
        )
        for b in sectors:
            expected = {
                ProductSector(cs)
                for cs in itertools.product(
                    *(
                        f.fusion(x, y)
                        for f, x, y in zip(factors, a.components, b.components, strict=True)
                    )
                )
            }
            assert set(provider.fusion(a, b)) == expected
            for c in sectors:
                assert provider.n_symbol(a, b, c) == math.prod(
                    f.n_symbol(x, y, z)
                    for f, x, y, z in zip(
                        factors, a.components, b.components, c.components, strict=True
                    )
                )


def test_n_symbol_zero_cases_are_reached():
    assert UU.n_symbol(uu(1, 0), uu(0, 1), uu(1, 1)) == 1
    assert UU.n_symbol(uu(1, 0), uu(0, 1), uu(0, 1)) == 0  # first component forbids
    assert UU.n_symbol(uu(1, 0), uu(0, 1), uu(1, 0)) == 0  # second component forbids
    assert SU.n_symbol(su(1, 1), su(1, 1), su(0, 3)) == 0


def test_fusion_is_canonically_sorted_for_su2_times_u1():
    a, b = su(3, 1), su(2, -2)
    got = SU.fusion(a, b)
    assert got == tuple(sorted(got))
    assert len(got) == len(SU2.fusion(a.components[0], b.components[0]))


# --- arity / type validation --------------------------------------------------------


@pytest.mark.parametrize("call", ["dual", "fusion", "n_symbol"])
def test_wrong_component_count_names_the_position(call):
    bad = ps(q(1))
    args = {"dual": (bad,), "fusion": (uu(0, 0), bad), "n_symbol": (uu(0, 0), uu(0, 0), bad)}[call]
    with pytest.raises(TypeError) as excinfo:
        getattr(UU, call)(*args)
    message = str(excinfo.value)
    assert f"argument {len(args) - 1}" in message
    assert "1 components" in message


@pytest.mark.parametrize("call", ["dual", "fusion", "n_symbol"])
def test_wrong_component_type_names_the_position(call):
    bad = uf(1, 1)  # a U(1) x fZ2 sector handed to a U(1) x U(1) provider
    args = {"dual": (bad,), "fusion": (bad, uu(0, 0)), "n_symbol": (uu(0, 0), bad, uu(0, 0))}[call]
    with pytest.raises(TypeError) as excinfo:
        getattr(UU, call)(*args)
    message = str(excinfo.value)
    assert "component 1" in message
    assert "FZ2Sector" in message and "U1Sector" in message


def test_non_product_sector_is_refused():
    with pytest.raises(TypeError) as excinfo:
        UU.dual(q(1))
    assert "U1Sector" in str(excinfo.value)


def test_graded_space_gap_fails_loudly_at_the_first_fusion_query():
    """``GradedSpace.new`` compares against ``type(provider.unit)``, which is
    ``ProductSector`` for *every* product — so this space is accepted. The
    documented mitigation is that the provider itself refuses at the first query."""
    space = GradedSpace.new(UU, {uf(0, 0): 1, uf(1, 1): 1})
    assert len(space) == 2  # the gap: silently accepted here
    with pytest.raises(TypeError):
        UU.fusion(uf(1, 1), uf(1, 1))
    legs = (Leg(space, OUT), Leg(space, OUT), Leg(space, IN))
    with pytest.raises(TypeError):
        _ = TensorStructure(legs).block_order


# --- mu_encode / mu_decode ----------------------------------------------------------


@pytest.mark.parametrize("ns", [(1, 1), (2, 1), (1, 3), (2, 3), (2, 2, 2)])
def test_mu_is_a_bijection_onto_range_prod(ns):
    total = math.prod(ns)
    seen = {}
    for mus in itertools.product(*(range(n) for n in ns)):
        mu = mu_encode(ns, mus)
        assert 0 <= mu < total
        assert mu not in seen
        seen[mu] = mus
        assert mu_decode(ns, mu) == mus
    assert sorted(seen) == list(range(total))


def test_mu_is_c_order_mixed_radix_with_the_first_factor_most_significant():
    ns = (2, 3)
    assert mu_encode(ns, (1, 0)) == 3
    assert mu_encode(ns, (0, 1)) == 1
    # the same flattening np.reshape performs, which is the whole point
    flat = np.arange(6).reshape(ns)
    for mus in itertools.product(range(2), range(3)):
        assert mu_encode(ns, mus) == flat[mus]


# --- project / assemble -------------------------------------------------------------


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_project_assemble_round_trip_over_every_su2_times_u1_tree(rank):
    labels = (su(0, 0), su(1, 1), su(2, -1))
    checked = 0
    for uncoupled in itertools.product(labels, repeat=rank):
        for tree in all_trees(SU, uncoupled):
            parts = tuple(project(SU, tree, i) for i in range(len(SU.factors)))
            for i, part in enumerate(parts):
                part.validate(SU.factors[i])
                assert part.rank == tree.rank
            assert assemble(SU, parts) == tree
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_assemble_project_round_trip_over_factor_tree_tuples(rank):
    su2_labels = (SU2Sector(0), SU2Sector(1))
    u1_labels = (q(0), q(1))
    checked = 0
    for su2_u in itertools.product(su2_labels, repeat=rank):
        for u1_u in itertools.product(u1_labels, repeat=rank):
            for t0 in all_trees(SU2, su2_u):
                for t1 in all_trees(U1, u1_u):
                    joined = assemble(SU, (t0, t1))
                    joined.validate(SU)
                    assert tuple(project(SU, joined, i) for i in range(2)) == (t0, t1)
                    checked += 1
    assert checked > 0


def test_project_rejects_an_out_of_range_factor():
    (tree,) = fusion_trees(UU, (uu(1, 0), uu(0, 1)), uu(1, 1))
    with pytest.raises(IndexError):
        project(UU, tree, 2)


def test_assemble_rejects_wrong_counts_and_ranks():
    (t0,) = fusion_trees(U1, (q(1), q(1)), q(2))
    (t1,) = fusion_trees(U1, (q(1),), q(1))
    with pytest.raises(ValueError):
        assemble(UU, (t0,))
    with pytest.raises(ValueError):
        assemble(UU, (t0, t1))


def test_rank_zero_and_rank_one_trees_survive_the_bijection():
    for tree in (
        FusionTree((), (), (), UU.unit),
        FusionTree((uu(1, -1),), (), (), uu(1, -1)),
    ):
        tree.validate(UU)
        parts = tuple(project(UU, tree, i) for i in range(2))
        assert assemble(UU, parts) == tree


# --- multiplicity, via the stub -----------------------------------------------------


def test_multiplicity_product_and_mixed_radix_projection():
    mm = ProductProvider((MultStub(), MultStub()))
    one, zero = MSector(1), MSector(0)
    a = ps(one, one)
    c = ps(zero, zero)
    assert mm.n_symbol(a, a, c) == 4
    tree = FusionTree((a, a), (), (3,), c)
    tree.validate(mm)
    parts = tuple(project(mm, tree, i) for i in range(2))
    assert tuple(p.multiplicities for p in parts) == ((1,), (1,))
    for i, part in enumerate(parts):
        part.validate(mm.factors[i])
    assert assemble(mm, parts) == tree
    # every mu in range(4) decodes to a distinct pair and reassembles
    seen = set()
    for mu in range(4):
        t = FusionTree((a, a), (), (mu,), c)
        pair = tuple(project(mm, t, i).multiplicities[0] for i in range(2))
        seen.add(pair)
        assert assemble(mm, tuple(project(mm, t, i) for i in range(2))) == t
    assert len(seen) == 4


def test_multiplicity_times_a_multiplicity_free_factor():
    mu1 = ProductProvider((MultStub(), U1))
    a = ps(MSector(1), q(1))
    c = ps(MSector(0), q(2))
    assert mu1.n_symbol(a, a, c) == 2 * U1.n_symbol(q(1), q(1), q(2))
    assert len(fusion_trees(mu1, (a, a), c)) == 2


# --- cgc ----------------------------------------------------------------------------


def independent_cgc_entry(provider, a, b, c, ia, ib, ic, imu):
    """Component indices by hand, no einsum and no reshape anywhere."""
    dims = [
        (f.irrep_dim(x), f.irrep_dim(y), f.irrep_dim(z), f.n_symbol(x, y, z))
        for f, x, y, z in zip(
            provider.factors, a.components, b.components, c.components, strict=True
        )
    ]
    # each flat index split back into one index per factor, most significant first
    idx = [
        mu_decode(tuple(d[axis] for d in dims), flat) for axis, flat in enumerate((ia, ib, ic, imu))
    ]
    value = 1.0
    for k, f in enumerate(provider.factors):
        g = f.cgc(a.components[k], b.components[k], c.components[k])
        value = value * g[idx[0][k], idx[1][k], idx[2][k], idx[3][k]]
    return value


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [
        (su(1, 1), su(1, 1), su(2, 2)),
        (su(1, 1), su(1, -1), su(0, 0)),
        (su(2, 0), su(1, 1), su(3, 1)),
        (uf(1, 1), uf(1, 1), uf(2, 0)),
    ],
)
def test_cgc_shape_and_entrywise_against_an_independent_loop(a, b, c):
    provider = SU if isinstance(a.components[0], SU2Sector) else UF
    got = provider.cgc(a, b, c)
    assert got.shape == (
        provider.irrep_dim(a),
        provider.irrep_dim(b),
        provider.irrep_dim(c),
        provider.n_symbol(a, b, c),
    )
    for index in itertools.product(*(range(n) for n in got.shape)):
        assert got[index] == pytest.approx(independent_cgc_entry(provider, a, b, c, *index))


def test_cgc_raises_on_a_forbidden_triple():
    with pytest.raises(ValueError):
        SU.cgc(su(1, 1), su(1, 1), su(0, 5))  # U(1) component forbids
    with pytest.raises(ValueError):
        SU.cgc(su(1, 1), su(1, 1), su(4, 2))  # SU(2) component forbids


def test_cgc_is_the_kronecker_product_in_c_order():
    a, b, c = su(1, 1), su(1, 1), su(2, 2)
    g0 = SU2.cgc(a.components[0], b.components[0], c.components[0])
    g1 = U1.cgc(a.components[1], b.components[1], c.components[1])
    expected = np.einsum("abcm,ABCM->aAbBcCmM", g0, g1).reshape(
        tuple(p * q_ for p, q_ in zip(g0.shape, g1.shape, strict=True))
    )
    np.testing.assert_allclose(SU.cgc(a, b, c), expected, atol=1e-15)


# --- capability forwarding ----------------------------------------------------------


def test_missing_capability_names_the_factor():
    bare = ProductProvider((BareStub(), U1))
    a = ps(ASector(1), q(1))
    for call, args in (("qdim", (a,)), ("irrep_dim", (a,)), ("cgc", (a, a, ps(ASector(0), q(2))))):
        with pytest.raises(CapabilityError) as excinfo:
            getattr(bare, call)(*args)
        message = str(excinfo.value)
        assert "factor 0 (Bare)" in message


def test_capability_present_when_every_factor_has_it():
    assert UF.qdim(uf(1, 1)) == 1.0
    assert UF.irrep_dim(uf(1, 1)) == 1
    assert SU.qdim(su(1, 0)) == 2.0
    assert SU.irrep_dim(su(1, 0)) == 2


# --- SU(2) x U(1): forwarding, now that #36 landed -----------------------------------


def test_su2_times_u1_permute_tree_is_forwarded():
    """#36 gave SU(2) a ``permute_tree``, so the product has one too — by the same
    "iff every factor has it" rule that used to refuse this call."""
    assert isinstance(ProductProvider((SU2, U1)), PermutationCoefficients) is True
    (tree,) = fusion_trees(SU, (su(1, 1), su(1, 1)), su(2, 2))
    terms = SU.permute_tree(tree, (1, 0))
    assert terms
    for t, coeff in terms:
        t.validate(SU)
        assert t.uncoupled == (su(1, 1), su(1, 1))
        assert coeff != 0


def test_su2_times_u1_within_side_transpose_succeeds():
    t = su_tensor()
    r = t.transpose((2, 1, 0, 3))
    assert r.structure.legs == tuple(t.legs[i] for i in (2, 1, 0, 3))


# a within-side swap (2, 1, 0, 3) plus a full reversal and two mixed cases: enough to
# reach genuinely multi-term SU(2) expansions without paying for all 24 permutations
SU_PERMS = ((2, 1, 0, 3), (0, 3, 2, 1), (3, 2, 1, 0), (2, 3, 0, 1), (1, 2, 3, 0))

# spin-1 is what makes it interesting: with only {0, 1/2} every SU(2) expansion has a
# single term, so the distributed product would never multiply anything out
SU_RICH_SPACE = GradedSpace.new(SU, {su(0, 0): 1, su(1, 1): 1, su(2, 0): 1})
SU_RICH_LEGS = (
    Leg(SU_RICH_SPACE, OUT),
    Leg(SU_RICH_SPACE, IN),
    Leg(SU_RICH_SPACE, OUT),
    Leg(SU_RICH_SPACE, IN),
)


def test_the_su2_factor_really_expands_into_several_terms():
    """Guards the oracle below: without this the dense check could pass on a diet of
    one-term expansions and prove nothing about the distributed product."""
    unc = (su(1, 1), su(1, 1), su(2, 0), su(2, 0))
    counts = {len(SU.permute_tree(t, (0, 2, 3, 1))) for t in all_trees(SU, unc)}
    assert max(counts) > 1


@pytest.mark.parametrize("p", SU_PERMS)
def test_dense_oracle_su2_times_u1(p):
    """The distributed product against a dense oracle, with a factor that really does
    return several terms — what the ``SplitStub`` could only simulate."""
    t = SymmetricTensor.random(SU_RICH_LEGS, seed=19)
    np.testing.assert_allclose(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p), atol=1e-11)


def test_isinstance_is_optimistic_and_the_call_is_authoritative():
    """``runtime_checkable`` only checks for the *presence* of ``permute_tree``, and
    ``ProductProvider`` defines it unconditionally so the error can name the factor.
    The call, not ``isinstance``, is the contract."""
    still = ProductProvider((StillStub(), U1))
    assert isinstance(still, PermutationCoefficients) is True
    a = ps(ASector(1), q(1))
    (tree,) = fusion_trees(still, (a, a), ps(ASector(0), q(2)))
    with pytest.raises(CapabilityError) as excinfo:
        still.permute_tree(tree, (1, 0))
    message = str(excinfo.value)
    assert "factor 0 (Still)" in message
    assert "PermutationCoefficients" in message


@pytest.mark.parametrize("p", CASE_A)
def test_su2_times_u1_case_a_transpose_succeeds(p):
    """#21's provider-free path, reached through a product provider: only the
    OUT/IN interleaving changes, so no braid and no capability is needed."""
    t = su_tensor()
    plan = permutation_plan(t.structure, p)
    assert {coeff for _, _, coeff in plan.terms} == {1.0}
    r = t.transpose(p)
    assert r.structure.block_order == t.structure.block_order
    for a, b in zip(r.blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(a, np.transpose(b, p))


# --- distributed product, via the stub ----------------------------------------------


def split_tree(provider):
    uncoupled = (ps(ASector(1), q(1)),) * 3
    trees = all_trees(provider, uncoupled)
    return trees[0]


def test_distributed_product_two_terms():
    prod = ProductProvider((SplitStub(), U1))
    tree = split_tree(prod)
    terms = prod.permute_tree(tree, (1, 2, 0))
    assert len(terms) == 2
    assert sorted(c for _, c in terms) == [0.6, 0.8]
    trees = [t for t, _ in terms]
    assert len(set(trees)) == 2
    for t in trees:
        t.validate(prod)
        assert t.uncoupled == tuple(tree.uncoupled[i] for i in (1, 2, 0))


def test_distributed_product_four_terms():
    prod = ProductProvider((SplitStub(), SplitStub()))
    uncoupled = (ps(ASector(1), ASector(1)),) * 3
    tree = all_trees(prod, uncoupled)[0]
    terms = prod.permute_tree(tree, (2, 0, 1))
    assert len(terms) == 4
    assert sorted(round(c, 10) for _, c in terms) == [0.36, 0.48, 0.48, 0.64]
    assert len({t for t, _ in terms}) == 4
    for t, _ in terms:
        t.validate(prod)


def test_zero_coefficient_terms_are_dropped():
    prod = ProductProvider((SplitStub(coeffs=(0.6, 0.0)), U1))
    terms = prod.permute_tree(split_tree(prod), (1, 2, 0))
    assert len(terms) == 1
    assert terms[0][1] == 0.6


# --- dense oracles ------------------------------------------------------------------


@pytest.mark.parametrize("p", ALL_PERMS)
def test_dense_oracle_u1_times_u1(p):
    t = uu_tensor()
    np.testing.assert_allclose(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p), atol=1e-12)


@pytest.mark.parametrize("p", ALL_PERMS)
def test_dense_oracle_u1_box_fz2(p):
    """The headline criterion: one exhaustive check exercising the non-trivial
    ``dual`` of U(1), the Koszul sign of fZ2, and the project/assemble bijection."""
    t = uf_tensor()
    expected = supersign(UF_LEGS, p, lambda a: a.components[1].parity) * np.transpose(
        t.to_dense(), p
    )
    np.testing.assert_allclose(t.transpose(p).to_dense(), expected, atol=1e-12)


def test_the_fz2_sign_is_demonstrably_forwarded():
    t = uf_tensor()
    p = (2, 1, 0, 3)  # swaps the two OUT lines
    plan = permutation_plan(t.structure, p)
    assert any(coeff == -1.0 for _, _, coeff in plan.terms)
    r = t.transpose(p)
    negated = [
        (src, dst)
        for src, dst, coeff in plan.terms
        if coeff == -1.0
        and np.any(t.blocks[src])
        and np.array_equal(r.blocks[dst], -np.transpose(t.blocks[src], p))
    ]
    assert negated


def test_u1_times_u1_carries_no_sign_at_all():
    t = uu_tensor()
    for p in ALL_PERMS:
        assert {coeff for _, _, coeff in permutation_plan(t.structure, p).terms} == {1.0}


def test_dense_layout_is_the_mixed_radix_layout():
    """A fermionic ``U(1) x fZ2`` tensor and its plain-``U(1)`` shadow expand to the
    same dense array: the product's slabs sit where the flattening convention says."""
    t = uf_tensor()
    np.testing.assert_allclose(t.to_dense(), u1_shadow(t).to_dense(), atol=1e-15)
    assert t.shape == u1_shadow(t).shape


# --- exact algebra ------------------------------------------------------------------


@pytest.mark.parametrize("factory", [uu_tensor, uf_tensor])
@pytest.mark.parametrize("p", ALL_PERMS)
def test_round_trip_and_norm_and_conj(factory, p):
    t = factory()
    back = t.transpose(p).transpose(inverse(p))
    assert back.structure == t.structure
    for a, b in zip(back.blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(a, b)
    # Simplification: approx, not ==: the qdim-weighted sum runs over the blocks in a
    # different order after a permutation, so the last ulp is not reproducible.
    assert tenet.norm(t.transpose(p)) == pytest.approx(tenet.norm(t), abs=1e-12)
    assert t.transpose(p).conj() == t.conj().transpose(p)


@pytest.mark.parametrize("factory", [uu_tensor, uf_tensor])
def test_composition_is_exact(factory):
    t = factory()
    for p in ALL_PERMS:
        for r in ALL_PERMS:
            left = t.transpose(p).transpose(r)
            right = t.transpose(compose(p, r))
            assert left.structure == right.structure
            for a, b in zip(left.blocks, right.blocks, strict=True):
                np.testing.assert_array_equal(a, b)


# --- nesting ------------------------------------------------------------------------

NESTED = ProductProvider((ProductProvider((U1, U1)), fZ2))


def nested(c0: int, c1: int, parity: int) -> ProductSector:
    return ps(uu(c0, c1), FZ2Sector(parity))


NESTED_SPACE = GradedSpace.new(
    NESTED, {nested(0, 0, 0): 2, nested(1, 0, 1): 1, nested(-1, 0, 1): 1}
)
NESTED_LEGS = (
    Leg(NESTED_SPACE, OUT),
    Leg(NESTED_SPACE, IN),
    Leg(NESTED_SPACE, OUT),
    Leg(NESTED_SPACE, IN),
)


def test_nesting_is_not_flattening():
    assert NESTED != ProductProvider((U1, U1, fZ2))
    assert NESTED.unit != ProductProvider((U1, U1, fZ2)).unit


@pytest.mark.parametrize("p", [(0, 1, 2, 3), (2, 1, 0, 3), (3, 2, 1, 0), (1, 3, 0, 2)])
def test_nesting_works_end_to_end(p):
    structure = TensorStructure(NESTED_LEGS)
    assert structure.num_blocks > 0
    structure.validate()
    t = SymmetricTensor.random(NESTED_LEGS, seed=23)
    expected = supersign(NESTED_LEGS, p, lambda a: a.components[1].parity) * np.transpose(
        t.to_dense(), p
    )
    np.testing.assert_allclose(t.transpose(p).to_dense(), expected, atol=1e-12)


def test_nested_projection_recurses():
    uncoupled = (nested(1, 0, 1), nested(-1, 0, 1))
    (tree,) = fusion_trees(NESTED, uncoupled, nested(0, 0, 0))
    outer = tuple(project(NESTED, tree, i) for i in range(2))
    outer[0].validate(UU)
    outer[1].validate(fZ2)
    inner = tuple(project(UU, outer[0], i) for i in range(2))
    assert assemble(UU, inner) == outer[0]
    assert assemble(NESTED, outer) == tree


# --- backend agnosticism ------------------------------------------------------------


@pytest.mark.parametrize("p", [(2, 1, 0, 3), (3, 1, 2, 0), (0, 3, 2, 1)])
def test_jax_matches_numpy(p):
    use_jax()
    t = uf_tensor()
    j = t.to_backend("jax")
    assert j.transpose(p).backend == "jax"
    for a, b in zip(j.transpose(p).blocks, t.transpose(p).blocks, strict=True):
        np.testing.assert_allclose(np.asarray(a), b, atol=1e-12)
    back = j.transpose(p).transpose(inverse(p))
    for a, b in zip(back.blocks, j.blocks, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# --- bending and duality forwarding (#312) ------------------------------------------
#
# A product had no ``BendingCoefficients`` and no ``DualBasis``, so ``repartition`` --
# and therefore every ``tenet.network`` container, whose write barrier repartitions --
# refused every product provider. The coefficients were never the obstacle: a bend moves
# one line, that line carries one component per factor, and each moves under its own
# factor's coefficient. What had held it back was the *oracle*: ``tests/helpers.py``'s
# dense-side parity vector covered multiplets rather than dense indices and padded a
# length mismatch with zeros, which deleted every Koszul sign on a fermionic factor
# beside a non-Abelian one. Fixed in the same change; see ``helpers._parities``.

#: fZ2 x U(1) x SU(2) -- the spinful-fermion grading, and the reason this matters. It is
#: the first product here with a fermionic *and* a non-Abelian factor, which is exactly
#: the combination the old oracle could not weigh.
FUS = ProductProvider((fZ2, U1, SU2))


def fus(parity: int, charge: int, two_j: int) -> ProductSector:
    return ProductSector((FZ2Sector(parity), U1Sector(charge), SU2Sector(two_j)))


#: A Hubbard-shaped site: empty and doubly-occupied are even spin singlets, the
#: singly-occupied states an odd ``j = 1/2`` doublet.
FUS_SPACE = GradedSpace.new(FUS, {fus(0, 0, 0): 1, fus(1, 1, 1): 1, fus(0, 2, 0): 1})


@pytest.mark.parametrize(
    ("provider", "space"),
    [(UU, UU_SPACE), (UF, UF_SPACE), (SU, SU_SPACE), (FUS, FUS_SPACE)],
    ids=lambda x: getattr(x, "name", ""),
)
def test_bending_and_duality_are_forwarded(provider, space):
    """The capability answers, and the answer is the right shape.

    ``supports`` is a *structural* check -- it asks whether the object has the methods --
    so on a product it is True for every forwarded capability regardless of the factors,
    exactly as it already was for ``cgc`` and ``qdim``. The refusal for a factor that
    lacks one is call-time; see
    :func:`test_a_factor_without_bending_refuses_the_whole_product`. So what is asserted
    here is that the call *works*, not that the flag is set.
    """
    assert supports(provider, BendingCoefficients)
    assert supports(provider, DualBasis)
    key = TensorStructure((Leg(space, OUT), Leg(space, OUT), Leg(space, IN))).block_order[0]
    for terms in (provider.bend_right(key, dual=False), provider.bend_right(key, dual=True)):
        assert terms
        assert all(isinstance(coeff, (int, float, complex)) for _, coeff in terms)
    sector = next(iter(space))
    z = provider.z_matrix(sector)
    assert z.shape == (provider.irrep_dim(sector), provider.irrep_dim(provider.dual(sector)))


@pytest.mark.parametrize(
    ("provider", "space"),
    [(UU, UU_SPACE), (UF, UF_SPACE), (SU, SU_SPACE), (FUS, FUS_SPACE)],
    ids=lambda x: getattr(x, "name", ""),
)
@pytest.mark.parametrize("partition", [((0,), (1, 2, 3)), ((0, 1, 2), (3,)), ((0, 2), (1, 3))])
def test_repartition_matches_explicit_dense_expansion(provider, space, partition):
    """The acceptance criterion: every bend against the dense array it must reproduce.

    ``dense_repartition`` builds the answer from ``to_dense()`` alone -- transposes into
    fermionic line order, pays the Koszul sign of the permutation relating the two line
    sequences, and transposes back -- so it shares no code with the plan under test.
    """
    legs = (Leg(space, OUT), Leg(space, OUT), Leg(space, IN), Leg(space, IN))
    t = SymmetricTensor.random(legs, seed=5)
    outs, ins = partition
    want, _ = dense_repartition(t.to_dense(), legs, outs, ins)
    np.testing.assert_allclose(tenet.repartition(t, outs, ins).to_dense(), want, atol=1e-12)


def test_repartition_round_trips_back_to_the_original():
    """There and back is the identity, coefficients included -- a bend that dropped a
    phase would still land on the right *structure* and fail only here."""
    legs = (Leg(FUS_SPACE, OUT), Leg(FUS_SPACE, OUT), Leg(FUS_SPACE, IN), Leg(FUS_SPACE, IN))
    t = SymmetricTensor.random(legs, seed=7)
    there = tenet.repartition(t, (0,), (1, 2, 3))
    back = tenet.repartition(there, (0, 1), (2, 3))
    assert back.legs == t.legs
    assert tenet.allclose(back, t)


def test_z_matrix_is_the_kronecker_product_of_the_factors():
    """Stated as the Kronecker product; asserted against one, in the flattening
    ``cgc`` uses -- the two agree by construction and this is that construction
    cashed."""
    for sector in FUS_SPACE:
        want = np.array([[1.0]])
        for factor, component in zip(FUS.factors, sector.components, strict=True):
            want = np.kron(want, factor.z_matrix(component))
        np.testing.assert_allclose(FUS.z_matrix(sector), want, atol=1e-15)


def test_a_factor_without_bending_refuses_the_whole_product():
    """Forwarded iff *every* factor has it, and the message names the factor."""

    @dataclass(frozen=True, slots=True)
    class NoBend:
        name: str = "NoBend"

        @property
        def unit(self) -> MSector:
            return MSector(0)

        def dual(self, a: MSector) -> MSector:
            return a

        def fusion(self, a: MSector, b: MSector) -> tuple[MSector, ...]:
            return (MSector((a.label + b.label) % 2),)

        def n_symbol(self, a: MSector, b: MSector, c: MSector) -> int:
            return int(c in self.fusion(a, b))

    product = ProductProvider((U1, NoBend()))
    # `supports` stays True -- it is structural, and the product *has* the methods. That
    # is the pre-existing contract for every forwarded capability (`cgc`, `qdim`), not
    # something bending introduced, and it is why the refusal below is the real check.
    assert supports(product, BendingCoefficients)
    space = GradedSpace.new(product, {ProductSector((U1Sector(0), MSector(0))): 1})
    key = TensorStructure((Leg(space, OUT), Leg(space, IN))).block_order[0]
    with pytest.raises(CapabilityError, match="factor 1 .NoBend."):
        product.bend_right(key, dual=False)


# --- exports and hygiene ------------------------------------------------------------

SRC = pathlib.Path(sys.modules["tenet"].__file__).parent
PRODUCT_SRC = (SRC / "symmetry" / "product.py").read_text()


def test_exports():
    import tenet.symmetry as sym

    assert sym.ProductProvider is ProductProvider
    assert sym.ProductSector is ProductSector
    assert {"ProductProvider", "ProductSector"} <= set(sym.__all__)


def test_module_is_backend_free_and_numpy_only_in_the_two_dense_methods():
    """NumPy appears in ``cgc`` and ``z_matrix`` and nowhere else.

    Those two are the module's only *dense* outputs -- a Clebsch-Gordan tensor and a
    duality matrix -- and each is built by an iterated outer product of the factors'.
    Everything else here is sector combinatorics and coefficient arithmetic, which must
    stay array-free; a ``np.`` appearing outside them is the module growing a numerical
    layer it has no business having.
    """
    assert "ar.do" not in PRODUCT_SRC
    assert "autoray" not in PRODUCT_SRC
    assert "to_dense" not in PRODUCT_SRC
    bodies = [
        PRODUCT_SRC.split(f"    def {name}(")[1].split("\n    def ")[0]
        for name in ("cgc", "z_matrix")
    ]
    assert PRODUCT_SRC.count("np.") == sum(b.count("np.") for b in bodies)
    assert all(b.count("np.") >= 2 for b in bodies)


def test_no_provider_identity_branching():
    for needle in ("provider == ", "isinstance(provider,", "U1Provider", "SU2Provider"):
        assert needle not in PRODUCT_SRC
    assert PRODUCT_SRC.count("requires(") >= 1


def test_a_spinful_fermion_grading_reaches_the_network_layer():
    """The point of #312, end to end: ``fZ2 x U(1) x SU(2)`` through ``tenet.network``.

    Every ``MPS`` write repartitions -- the container normalizes each site onto its
    stated partition -- so before bending was forwarded a product provider could build
    an ``MPS`` and then fail on the first ``norm()`` or ``canonize_()``. That is the
    grading a spinful fermion wants (parity, charge, total spin), so the refusal fell
    exactly where it hurt.

    The state is at half filling in the total singlet: the right boundary carries charge
    ``N``, which is what makes the chain non-trivial. Left as ``UNIT`` on both ends it
    would be forced onto the vacuum, since every physical charge here is non-negative --
    a legal but empty check.
    """
    from tenet.network import MPS

    n = 4
    unit = GradedSpace.new(FUS, {FUS.unit: 1})
    right = GradedSpace.new(FUS, {fus(0, n, 0): 1})
    bond = GradedSpace.new(
        FUS, {fus(0, 0, 0): 1, fus(1, 1, 1): 1, fus(0, 2, 0): 1, fus(1, 3, 1): 1, fus(0, 4, 0): 1}
    )
    sites = [
        SymmetricTensor.random(
            (
                Leg(unit if i == 0 else bond, OUT),
                Leg(FUS_SPACE, OUT),
                Leg(right if i == n - 1 else bond, IN),
            ),
            seed=i,
        )
        for i in range(n)
    ]
    psi = MPS(sites)
    psi.canonize_()
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)

    # Genuinely entangled, so the canonization is not passing on a product state.
    entropy = psi.entanglement_entropy()
    assert set(entropy) == set(range(n - 1))
    assert all(s > 1e-6 for s in entropy.values()), entropy

    # And the SU(2) factor is doing its work: a mid-chain bond holds fewer multiplets
    # than dense states, which is the whole reason to grade by total spin.
    mid = psi[n // 2].legs[0].space
    assert mid.reduced_dim < mid.dim
