"""M12a: the multiplicity plumbing in ``tenet.symmetry.base``, driven by SU(3).

Every provider tenet ships is multiplicity-free, so ``N^c_ab > 1`` has no
in-repo coverage at all. SU(3)'s ``8 x 8 -> 8`` has ``N = 2``, and the vendored
``tests/fixtures/su3_*.txt`` carry real pentagon/hexagon-satisfying coefficients
for it (see :mod:`tests.symmetry._su3_fixture`).

``to_dense`` is the oracle throughout: ``_tree_cgt`` needed no change for M12a, so
every agreement below is the *new* array-valued braid/bend expansion checked
against untouched code.
"""

import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pytest
from _su3_fixture import EIGHT, ONE, SIX, SU3, THREE, THREEBAR, SU3Sector

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.fusion_tree import coupled_sectors, fusion_trees
from tenet.symmetry import (
    U1,
    CapabilityError,
    MultiplicityRecoupling,
    RecouplingData,
    U1Sector,
    bend_braided,
)
from tenet.symmetry.product import ProductProvider, ProductSector

FIXTURES = Path(__file__).parent.parent / "fixtures"

E = GradedSpace.new(SU3, {EIGHT: 1})
V = GradedSpace.new(SU3, {THREE: 1})
W = GradedSpace.new(SU3, {THREEBAR: 1})

# Three adjoints on the OUT side: the rank-3 output tree runs over ``8 x 8 -> e``
# and ``e x 8 -> 8``, so its block set contains the ``N = 2`` vertex four times.
LEGS = (Leg(E, OUT, name="a"), Leg(E, OUT, name="b"), Leg(E, OUT), Leg(E, IN))
# A mixed-sector layout with a non-self-dual leg, interleaved sides.
MIXED = (Leg(E, OUT), Leg(V, IN), Leg(E, OUT), Leg(W, IN))

SPLITS = (
    ((0, 1, 2), (3,)),
    ((0, 1), (2, 3)),
    ((0,), (1, 2, 3)),
    ((0, 1, 2, 3), ()),
    ((), (0, 1, 2, 3)),
)


def su3(legs=LEGS, seed=11) -> SymmetricTensor:
    return SymmetricTensor.random(legs, seed=seed)


# --- provenance --------------------------------------------------------------


@pytest.mark.parametrize("name", ["su3_f.txt", "su3_r.txt", "su3_cg.txt"])
def test_fixture_header_names_racah_as_oracle(name: str) -> None:
    """A silent regeneration from a different oracle must show up in the diff."""
    header = [line for line in (FIXTURES / name).read_text().splitlines() if line.startswith("#")]
    assert len(header) == 9, "the nine racah provenance lines must be preserved verbatim"
    assert header[0].endswith("over {1,3,3bar,6,6bar,8,10,10bar,15,15bar,27}")
    assert header[1] == (
        "# oracle: racah (Rust crate, cgc-gen feature; SUNRepresentations.jl v0.4.0 gauge)"
    )
    assert header[2].startswith("# generator: examples/dump_su3_fixtures.rs")
    assert header[3] == "# racah version: 0.1.1"
    assert header[4] == "# racah commit: 6e05f16cc79c88379a8ebfb3062a63a0920af3db"
    assert header[8].startswith("# columns: ")


# --- the capability ----------------------------------------------------------


def test_provider_supplies_both_recoupling_capabilities() -> None:
    assert isinstance(SU3, MultiplicityRecoupling)
    assert isinstance(SU3, RecouplingData)


def test_eight_times_eight_has_multiplicity_two() -> None:
    assert SU3.n_symbol(EIGHT, EIGHT, EIGHT) == 2
    assert SU3.fusion(THREE, THREEBAR) == (ONE, EIGHT)
    assert SU3.fusion(THREE, THREE) == (THREEBAR, SIX)


def test_symbol_shapes_match_the_protocol() -> None:
    assert SU3.r_matrix(EIGHT, EIGHT, EIGHT).shape == (2, 2)
    assert SU3.b_matrix(EIGHT, EIGHT, EIGHT).shape == (2, 2)
    assert SU3.f_matrix(EIGHT, EIGHT, EIGHT, EIGHT, EIGHT, EIGHT).shape == (2, 2, 2, 2)
    # a one-dimensional vertex still carries its axis
    assert SU3.f_matrix(THREE, THREEBAR, THREE, THREE, ONE, ONE).shape == (1, 1, 1, 1)


def test_r_matrix_is_an_involution() -> None:
    """SU(3) is a symmetric category, so ``R^{ba} R^{ab} == 1`` as matrices."""
    r = SU3.r_matrix(EIGHT, EIGHT, EIGHT)
    assert np.allclose(r @ r, np.eye(2), atol=1e-12)


# --- tree level --------------------------------------------------------------


def test_braid_expands_the_multiplicity_labels() -> None:
    """The claim M12a deletes: an F-move mixes ``(e, mu)`` with ``(f, mu')``."""
    tree = next(
        t
        for t in fusion_trees(SU3, (EIGHT, EIGHT, EIGHT), EIGHT)
        if t.inner == (EIGHT,) and t.multiplicities == (0, 0)
    )
    terms = SU3.permute_tree(tree, (0, 2, 1))
    labels = {t.multiplicities for t, _ in terms}
    assert len(labels) > 1, "a scalar-path braid would return one multiplicity label"
    for permuted, coeff in terms:
        permuted.validate(SU3)
        assert permuted.coupled == tree.coupled
        assert not hasattr(coeff, "shape")


@pytest.mark.parametrize("rank", [2, 3])
def test_tree_level_unitarity(rank: int) -> None:
    """The permutation matrix between tree bases is unitary, multiplicity included."""
    for u in itertools.product((THREE, THREEBAR, EIGHT), repeat=rank):
        for coupled in coupled_sectors(SU3, u):
            src = fusion_trees(SU3, u, coupled)
            for perm in itertools.permutations(range(rank)):
                dst = fusion_trees(SU3, tuple(u[i] for i in perm), coupled)
                assert len(src) == len(dst)
                index = {t: j for j, t in enumerate(dst)}
                m = np.zeros((len(src), len(dst)), dtype=complex)
                for i, tree in enumerate(src):
                    for t2, coeff in SU3.permute_tree(tree, perm):
                        m[i, index[t2]] += coeff
                assert np.allclose(m @ m.conj().T, np.eye(len(src)), atol=1e-10)


# --- tensor level ------------------------------------------------------------


def test_block_set_contains_a_multiplicity_two_vertex() -> None:
    t = su3()
    seen = {
        (a, b, c)
        for key in t.structure.block_order
        for tree in (key.output_tree, key.input_tree)
        for a, b, c, _ in tree.vertices()
    }
    assert any(SU3.n_symbol(*v) == 2 for v in seen)
    assert any(k.output_tree.multiplicities[-1] == 1 for k in t.structure.block_order)


@pytest.mark.parametrize("legs", [LEGS, MIXED])
@pytest.mark.parametrize("perm", [(0, 2, 1, 3), (1, 0, 2, 3), (2, 1, 0, 3), (0, 1, 3, 2)])
def test_transpose_matches_the_dense_oracle(legs, perm) -> None:
    t = su3(legs)
    got = np.asarray(t.transpose(perm).to_dense())
    assert np.allclose(got, np.transpose(np.asarray(t.to_dense()), perm), atol=1e-10)


@pytest.mark.parametrize("perm", [(0, 2, 1, 3), (2, 1, 0, 3), (1, 2, 0, 3)])
def test_transpose_round_trips(perm) -> None:
    t = su3()
    inverse = tuple(sorted(range(4), key=perm.__getitem__))
    back = t.transpose(perm).transpose(inverse)
    assert back.structure == t.structure
    for a, b in zip(back.blocks, t.blocks, strict=True):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-10)


@pytest.mark.parametrize("split", SPLITS)
def test_repartition_matches_the_dense_oracle(split) -> None:
    outputs, inputs = split
    t = su3()
    got = np.asarray(t.repartition(outputs, inputs).to_dense())
    want = np.transpose(np.asarray(t.to_dense()), (*outputs, *inputs))
    assert np.allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("split", SPLITS)
def test_repartition_round_trips(split) -> None:
    outputs, inputs = split
    t = su3()
    back = t.repartition(outputs, inputs).repartition((0, 1, 2), (3,))
    assert back.structure == t.structure
    for a, b in zip(back.blocks, t.blocks, strict=True):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-10)


def test_bend_is_multi_term_at_a_multiplicity_vertex() -> None:
    """``bend_braided``'s hardcoded ``mu = 0`` is a real expansion now."""
    from tenet.structure import FusionBlockKey

    src = next(
        t
        for t in fusion_trees(SU3, (EIGHT, EIGHT, EIGHT), EIGHT)
        if t.inner == (EIGHT,) and t.multiplicities == (0, 0)
    )
    dst = fusion_trees(SU3, (EIGHT,), EIGHT)[0]
    terms = SU3.bend_right(FusionBlockKey(src, dst), dual=False)
    assert len(terms) == 2
    assert {k.input_tree.multiplicities[-1] for k, _ in terms} == {0, 1}


def test_bend_still_refuses_a_scalar_only_provider() -> None:
    """Without :class:`MultiplicityRecoupling` the old ``CapabilityError`` stands."""
    from tenet.structure import FusionBlockKey

    src = next(t for t in fusion_trees(SU3, (EIGHT, EIGHT, EIGHT), EIGHT) if t.inner == (EIGHT,))
    dst = fusion_trees(SU3, (EIGHT,), EIGHT)[0]
    scalar = _ScalarOnly()
    assert not isinstance(scalar, MultiplicityRecoupling)
    with pytest.raises(CapabilityError, match="matrix-valued B-symbols"):
        bend_braided(scalar, FusionBlockKey(src, dst), right=True, dual=False)


@dataclasses.dataclass(frozen=True, slots=True)
class _ScalarOnly:
    """SU(3) with the array-valued capability withheld; everything else forwarded."""

    name = "SU3-scalar-only"

    def __getattr__(self, item: str):
        if item in ("f_matrix", "r_matrix", "b_matrix"):
            raise AttributeError(item)
        return getattr(SU3, item)


# --- SU(3) x U(1) ------------------------------------------------------------


def test_product_with_u1_reaches_multiplicity_two() -> None:
    """``product.py``'s mixed-radix ``mu`` is no longer stub-only coverage."""
    prod = ProductProvider((SU3, U1))
    a = ProductSector((EIGHT, U1Sector(1)))
    c = ProductSector((EIGHT, U1Sector(2)))
    assert prod.n_symbol(a, a, c) == 2
    assert c in prod.fusion(a, a)

    space = GradedSpace.new(prod, {a: 1})
    top = GradedSpace.new(prod, {ProductSector((EIGHT, U1Sector(3))): 1})
    t = SymmetricTensor.random(
        (Leg(space, OUT), Leg(space, OUT), Leg(space, OUT), Leg(top, IN)), seed=3
    )
    assert any(
        prod.n_symbol(*v) == 2
        for key in t.structure.block_order
        for a_, b_, c_, _ in key.output_tree.vertices()
        for v in [(a_, b_, c_)]
    )
    got = np.asarray(t.transpose((0, 2, 1, 3)).to_dense())
    want = np.transpose(np.asarray(t.to_dense()), (0, 2, 1, 3))
    assert np.allclose(got, want, atol=1e-10)


def test_su3_sector_ordering_is_by_dynkin() -> None:
    assert sorted([EIGHT, ONE, THREE]) == [ONE, THREE, EIGHT]
    assert SU3Sector((0, 0)) == ONE
