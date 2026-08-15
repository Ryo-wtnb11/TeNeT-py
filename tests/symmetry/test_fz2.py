"""Tests for tenet.symmetry.fz2 — one test per acceptance criterion of issue #39.

Two oracles, both written independently of the provider:

* ``bubble_sign`` — a bubble sort that multiplies ``-1`` per adjacent transposition
  of two odd lines. It never mentions inversions, so it is a genuine check on
  :func:`koszul_sign`'s formula and not a restatement of it.
* ``supersign`` (``tests/helpers.py``, shared with the contraction suite since
  #51) — the dense-side Koszul sign, built from each axis's parity vector
  in the space's canonical sector order (``FZ2Sector(0)``'s slab first, then
  ``FZ2Sector(1)``'s, since ``irrep_dim == 1``). For an output entry indexed by
  ``(n_0, ..., n_{N-1})`` with ``q_j = par_{p[j]}[n_j]`` the sign is
  ``(-1) ** sum(q_j * q_k for j < k if p[j] > p[k])``, optionally restricted to
  pairs on the same side — that restriction *is* TeNeT-py's convention (public
  axis order across sides is our own ndarray bookkeeping, invariant 3) and the
  interleaved test is what pins it.
"""

import dataclasses
import itertools
import pathlib
import sys

import numpy as np
import pytest
from helpers import supersign

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.fusion_tree import FusionTree, fusion_trees
from tenet.ops.permutation import permutation_plan
from tenet.ops.repartition import bend, bend_plan
from tenet.structure import FusionBlockKey
from tenet.symmetry import (
    FZ2_GAUGE,
    SU2,
    BendingCoefficients,
    ClebschGordan,
    DualBasis,
    FusionProvider,
    FZ2Provider,
    FZ2Sector,
    PermutationCoefficients,
    QuantumDimension,
    fZ2,
)
from tenet.symmetry.fz2 import koszul_sign

# static conformance check: fails type checking if FZ2Provider drifts from the protocols
_fusion: FusionProvider = fZ2
_qdim: QuantumDimension = fZ2
_cgc: ClebschGordan = fZ2

EVEN, ODD = FZ2Sector(0), FZ2Sector(1)
SECTORS = (EVEN, ODD)

P = GradedSpace.new(fZ2, {EVEN: 2, ODD: 3})
Q = GradedSpace.new(fZ2, {EVEN: 1, ODD: 2})

OUT_LEGS = (Leg(P, OUT, name="a"), Leg(Q, OUT, name="b"), Leg(Q, OUT), Leg(P, OUT))
MIXED_LEGS = (Leg(P, OUT, name="a"), Leg(Q, IN, name="b"), Leg(Q, OUT), Leg(P, IN))

ALL_PERMS = tuple(itertools.permutations(range(4)))
# permutations of (0, 1, 2, 3) that interleave the subsequences (0, 2) and (1, 3)
# without disturbing either — case A, no braid at all
CASE_A = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2), (1, 0, 2, 3), (1, 0, 3, 2), (1, 3, 0, 2))


def out_tensor() -> SymmetricTensor:
    return SymmetricTensor.random(OUT_LEGS, seed=17)


def mixed_tensor() -> SymmetricTensor:
    return SymmetricTensor.random(MIXED_LEGS, seed=19)


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


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


def bubble_sign(parities: tuple[int, ...], perm: tuple[int, ...]) -> float:
    """Independent reference: bubble ``perm`` back to sorted order, ``-1`` per
    adjacent swap of two odd lines. No inversion counting anywhere."""
    seq = [(i, parities[i]) for i in perm]
    sign = 1.0
    swapped = True
    while swapped:
        swapped = False
        for k in range(len(seq) - 1):
            if seq[k][0] > seq[k + 1][0]:
                if seq[k][1] and seq[k + 1][1]:
                    sign = -sign
                seq[k], seq[k + 1] = seq[k + 1], seq[k]
                swapped = True
    return sign


# --- sectors -----------------------------------------------------------------------


def test_only_zero_and_one_are_constructible():
    assert FZ2Sector(0).parity == 0
    assert FZ2Sector(1).parity == 1
    for bad in (2, -1, 7):
        with pytest.raises(ValueError) as excinfo:
            FZ2Sector(bad)
        assert str(bad) in str(excinfo.value)
    for bad in (True, False):
        with pytest.raises(TypeError) as excinfo:
            FZ2Sector(bad)
        assert repr(bad) in str(excinfo.value)


def test_sectors_order_even_before_odd_and_hash():
    assert EVEN < ODD
    assert sorted((ODD, EVEN)) == [EVEN, ODD]
    assert len({FZ2Sector(0), FZ2Sector(0), FZ2Sector(1)}) == 2
    assert repr(EVEN) == "FZ2Sector(parity=0)"


# --- provider basics ---------------------------------------------------------------


def test_provider_basics():
    assert fZ2.name == "fZ2"
    assert fZ2.unit == EVEN
    for a in SECTORS:
        assert fZ2.dual(a) == a
        assert fZ2.qdim(a) == 1.0
        assert fZ2.irrep_dim(a) == 1
        for b in SECTORS:
            assert fZ2.fusion(a, b) == (FZ2Sector(a.parity ^ b.parity),)
            for c in SECTORS:
                assert fZ2.n_symbol(a, b, c) == int(c.parity == a.parity ^ b.parity)


def test_n_symbol_allows_one_triple_and_forbids_three():
    triples = [(a, b, c) for a in SECTORS for b in SECTORS for c in SECTORS]
    assert sum(fZ2.n_symbol(*t) for t in triples) == 4  # one allowed c per (a, b)
    assert sum(1 for t in triples if not fZ2.n_symbol(*t)) == 4


def test_cgc_shape_value_and_read_only():
    for a in SECTORS:
        for b in SECTORS:
            (c,) = fZ2.fusion(a, b)
            cgc = fZ2.cgc(a, b, c)
            assert cgc.shape == (1, 1, 1, 1)
            np.testing.assert_array_equal(cgc, np.ones((1, 1, 1, 1)))
            assert not cgc.flags.writeable


def test_cgc_raises_on_forbidden_triple():
    with pytest.raises(ValueError):
        fZ2.cgc(ODD, ODD, ODD)


def test_provider_is_frozen_hashable_and_array_free():
    assert fZ2 == FZ2Provider()
    assert hash(fZ2) == hash(FZ2Provider())
    with pytest.raises(dataclasses.FrozenInstanceError):
        fZ2.name = "nope"
    for f in dataclasses.fields(fZ2):
        assert not isinstance(getattr(fZ2, f.name), np.ndarray)
    assert hash(GradedSpace.new(fZ2, {EVEN: 2, ODD: 3}))


def test_capabilities():
    assert isinstance(fZ2, PermutationCoefficients)
    assert isinstance(fZ2, QuantumDimension)
    assert isinstance(fZ2, ClebschGordan)
    assert isinstance(fZ2, BendingCoefficients)
    assert isinstance(fZ2, DualBasis)


def test_gauge_is_a_module_string_not_a_field():
    assert isinstance(FZ2_GAUGE, str)
    assert "koszul" in FZ2_GAUGE
    assert "twist" in FZ2_GAUGE
    assert "FZ2_GAUGE" not in {f.name for f in dataclasses.fields(fZ2)}


# --- koszul_sign -------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_koszul_sign_exhaustive_against_bubble_sort(n):
    """The single most important test in the issue: every permutation, every
    parity assignment, against an independently written bubble sort."""
    for parities in itertools.product((0, 1), repeat=n):
        uncoupled = tuple(FZ2Sector(p) for p in parities)
        for perm in itertools.permutations(range(n)):
            assert koszul_sign(uncoupled, perm) == bubble_sign(parities, perm)


def test_koszul_sign_anchors():
    assert koszul_sign((ODD, ODD), (1, 0)) == -1.0
    assert koszul_sign((ODD, EVEN), (1, 0)) == 1.0
    assert koszul_sign((EVEN, ODD), (1, 0)) == 1.0
    assert koszul_sign((EVEN, EVEN), (1, 0)) == 1.0
    for n in range(1, 6):
        for parities in itertools.product((0, 1), repeat=n):
            u = tuple(FZ2Sector(p) for p in parities)
            assert koszul_sign(u, tuple(range(n))) == 1.0
        # symmray's closed-form branch for a full reversal of n odd lines
        assert koszul_sign((ODD,) * n, tuple(reversed(range(n)))) == float((-1) ** (n // 2))


def test_koszul_sign_is_a_cocycle():
    n = 4
    perms = tuple(itertools.permutations(range(n)))
    for parities in itertools.product((0, 1), repeat=n):
        u = tuple(FZ2Sector(x) for x in parities)
        for p in perms:
            for q in perms:
                assert koszul_sign(u, compose(p, q)) == koszul_sign(u, p) * koszul_sign(
                    permuted(u, p), q
                )


def test_koszul_sign_is_direction_invariant():
    n = 4
    for parities in itertools.product((0, 1), repeat=n):
        u = tuple(FZ2Sector(x) for x in parities)
        for p in itertools.permutations(range(n)):
            assert koszul_sign(u, p) == koszul_sign(permuted(u, p), inverse(p))


# --- permute_tree ------------------------------------------------------------------


def all_rank4_trees():
    for parities in itertools.product((0, 1), repeat=4):
        uncoupled = tuple(FZ2Sector(x) for x in parities)
        (coupled,) = {FZ2Sector(sum(parities) % 2)}
        (tree,) = fusion_trees(fZ2, uncoupled, coupled)
        yield tree


def test_permute_tree_is_one_term_with_the_koszul_coefficient():
    for tree in all_rank4_trees():
        for perm in ALL_PERMS:
            terms = fZ2.permute_tree(tree, perm)
            assert len(terms) == 1
            ((new_tree, coeff),) = terms
            new_tree.validate(fZ2)
            assert new_tree.uncoupled == permuted(tree.uncoupled, perm)
            assert new_tree.coupled == tree.coupled
            assert coeff == koszul_sign(tree.uncoupled, perm)


def test_coefficient_ignores_the_spine():
    """The coefficient reads the uncoupled parities and nothing else — not the
    coupled sector, not the inner lines. False for SU(2), whose exchange sign
    ``(-1)^(j_a+j_b-j_c)`` reads the fused channel; true here because ``F ≡ 1``.
    Asserted so a future reader knows why the code may ignore the spine.

    Two readings, both checked: :func:`koszul_sign` applied to trees that differ
    only in ``inner``/``coupled`` (constructed directly, since fZ2's fusion makes
    only one of them a valid label), and the plan coefficients of a real tensor,
    which must be constant across every block sharing an uncoupled assignment.
    """
    uncoupled = (ODD, ODD, EVEN, ODD)
    perm = (1, 0, 3, 2)
    trees = [
        FusionTree(uncoupled, inner, (0, 0, 0), coupled)
        for inner, coupled in (((EVEN, EVEN), ODD), ((ODD, ODD), EVEN))
    ]
    assert len({koszul_sign(t.uncoupled, perm) for t in trees}) == 1
    assert fZ2.permute_tree(trees[0], perm)[0][1] == koszul_sign(uncoupled, perm)

    t = mixed_tensor()
    for p in ALL_PERMS:
        plan = permutation_plan(t.structure, p)
        by_uncoupled: dict[tuple, set] = {}
        for src, _, coeff in plan.terms:
            key = t.structure.block_order[src]
            label = (key.output_tree.uncoupled, key.input_tree.uncoupled)
            by_uncoupled.setdefault(label, set()).add(coeff)
        assert all(len(v) == 1 for v in by_uncoupled.values())


# --- dense oracles -----------------------------------------------------------------


@pytest.mark.parametrize("p", ALL_PERMS)
def test_dense_oracle_all_out(p):
    t = out_tensor()
    dense = t.to_dense()
    expected = supersign(OUT_LEGS, p, per_side=False) * np.transpose(dense, p)
    np.testing.assert_allclose(t.transpose(p).to_dense(), expected, atol=1e-12)


@pytest.mark.parametrize("p", ALL_PERMS)
def test_dense_oracle_interleaved_sides(p):
    t = mixed_tensor()
    dense = t.to_dense()
    expected = supersign(MIXED_LEGS, p, per_side=True) * np.transpose(dense, p)
    np.testing.assert_allclose(t.transpose(p).to_dense(), expected, atol=1e-12)


@pytest.mark.parametrize("p", CASE_A)
def test_case_a_carries_no_sign_at_all(p):
    t = mixed_tensor()
    plan = permutation_plan(t.structure, p)
    assert {coeff for _, _, coeff in plan.terms} == {1.0}
    assert plan.new_structure.block_order == t.structure.block_order
    np.testing.assert_allclose(t.transpose(p).to_dense(), np.transpose(t.to_dense(), p), atol=1e-12)


def test_minus_one_is_actually_reached():
    t = mixed_tensor()
    p = (2, 1, 0, 3)  # swaps the two OUT lines
    plan = permutation_plan(t.structure, p)
    assert any(coeff == -1.0 for _, _, coeff in plan.terms)
    r = t.transpose(p)
    negated = [
        (src, dst)
        for src, dst, coeff in plan.terms
        if coeff == -1.0
        and np.array_equal(r.blocks[dst], -np.transpose(t.blocks[src], p))
        and np.any(t.blocks[src])
    ]
    assert negated


# --- exact algebra -----------------------------------------------------------------


@pytest.mark.parametrize("p", ALL_PERMS)
def test_round_trip_is_exact(p):
    t = mixed_tensor()
    back = t.transpose(p).transpose(inverse(p))
    assert back.structure == t.structure
    for a, b in zip(back.blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("p", ALL_PERMS)
def test_composition_is_exact(p):
    t = mixed_tensor()
    for q in ALL_PERMS:
        left = t.transpose(p).transpose(q)
        right = t.transpose(compose(p, q))
        assert left.structure == right.structure
        for a, b in zip(left.blocks, right.blocks, strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("p", ALL_PERMS)
def test_norm_conj_and_dtype(p):
    t = mixed_tensor()
    assert tenet.norm(t.transpose(p)) == tenet.norm(t)
    assert t.transpose(p).conj() == t.conj().transpose(p)
    assert t.dtype == np.float64
    assert t.transpose(p).dtype == np.float64


# --- #32 capabilities: bending -----------------------------------------------------


def test_z_matrix_is_read_only_ones():
    for a in SECTORS:
        z = fZ2.z_matrix(a)
        np.testing.assert_array_equal(z, np.ones((1, 1)))
        assert not z.flags.writeable


@pytest.mark.parametrize("axis", [2, 3])
def test_bend_plan_coefficients_are_all_one(axis):
    t = mixed_tensor()
    plan = bend_plan(t.structure, axis)
    assert {coeff for _, _, coeff in plan.terms} == {1.0}
    assert len(plan.terms) == t.structure.num_blocks


def test_bend_round_trip_is_exact():
    t = mixed_tensor()
    bent = bend(t, 2)  # last OUT leg -> IN
    back = bend(bent, bent.ndim - 1)  # and straight back
    restored = back.transpose((0, 1, 3, 2))
    assert restored.structure == t.structure
    for a, b in zip(restored.blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(a, b)


def test_bend_dense_oracle_on_a_sign_free_bend():
    """A single bend with no within-side reordering: every coefficient is 1 and the
    dense array is untouched, exactly as for U(1).

    Splits whose decomposition (transpose, bend, transpose) *does* reorder lines
    within a side are excluded on purpose — there the graded sign is real and a
    plain dense array cannot carry it, which is TensorKitSectors' own warning that
    "plain arrays with FermionParity labels cannot preserve all fermionic signs".
    :func:`test_repartition_differs_from_dense_by_a_sign_only` covers those.
    """
    legs = (Leg(P, OUT), Leg(Q, OUT), Leg(Q, OUT))
    t = SymmetricTensor.random(legs, seed=23)
    # the bent leg carries the odd sector
    assert ODD in legs[-1].space
    np.testing.assert_allclose(
        t.repartition((0, 1), (2,)).to_dense(), np.transpose(t.to_dense(), (0, 1, 2)), atol=1e-12
    )
    pair = (Leg(P, OUT), Leg(Q, OUT))
    t2 = SymmetricTensor.random(pair, seed=29)
    np.testing.assert_allclose(t2.repartition((0,), (1,)).to_dense(), t2.to_dense(), atol=1e-12)


@pytest.mark.parametrize(
    ("outputs", "inputs"),
    [((0, 1, 2), (3,)), ((0,), (1, 2, 3)), ((0, 1, 2, 3), ()), ((), (0, 1, 2, 3))],
)
def test_repartition_differs_from_dense_by_a_sign_only(outputs, inputs):
    t = mixed_tensor()
    got = t.repartition(outputs, inputs).to_dense()
    want = np.transpose(t.to_dense(), (*outputs, *inputs))
    np.testing.assert_allclose(np.abs(got), np.abs(want), atol=1e-12)


def test_repartition_round_trip_is_exact():
    t = mixed_tensor()
    r = t.repartition((0, 1, 2), (3,))
    back = r.repartition(t.structure.out_axes, t.structure.in_axes)
    assert back.transpose((0, 2, 1, 3)).structure == t.structure
    for a, b in zip(back.transpose((0, 2, 1, 3)).blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(a, b)


# --- backend agnosticism -----------------------------------------------------------


@pytest.mark.parametrize("p", [(2, 1, 0, 3), (3, 1, 2, 0), (0, 3, 2, 1)])
def test_jax_matches_numpy(p):
    use_jax()
    t = mixed_tensor()
    j = t.to_backend("jax")
    assert j.transpose(p).backend == "jax"
    for a, b in zip(j.transpose(p).blocks, t.transpose(p).blocks, strict=True):
        np.testing.assert_allclose(np.asarray(a), b, atol=1e-12)
    back = j.transpose(p).transpose(inverse(p))
    for a, b in zip(back.blocks, j.blocks, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# --- hygiene -----------------------------------------------------------------------

SRC = pathlib.Path(sys.modules["tenet"].__file__).parent
FZ2_SRC = (SRC / "symmetry" / "fz2.py").read_text()


def test_fz2_module_is_array_free_apart_from_two_constants():
    assert "ar.do" not in FZ2_SRC
    assert "autoray" not in FZ2_SRC
    assert "to_dense" not in FZ2_SRC
    assert FZ2_SRC.count("np.ones") == 2


def test_no_provider_identity_branching_outside_fz2():
    # serialize.py (#94) names every provider type, and must: a file format needs a
    # kind <-> type registry to rebuild a provider from JSON. That is a dict lookup
    # on ``type(provider)``, not a behavioural branch on provider identity — the
    # ``isinstance``/``==`` assertions below still apply to it.
    naming_allowed = ("__init__.py", "serialize.py")
    for path in SRC.rglob("*.py"):
        if path.name == "fz2.py":
            continue
        text = path.read_text()
        assert "FZ2Provider" not in text or path.name in naming_allowed
        assert "isinstance(provider, FZ2" not in text
        assert "provider == " not in text
        assert "fZ2" not in text or path.name in naming_allowed


def test_capability_gate_is_still_opt_in():
    """fZ2's arrival must not have loosened the gate: a provider opts in by
    defining ``permute_tree`` — SU(2) does since #36 (the negative case lives in
    tests/ops/test_permutation.py)."""
    assert isinstance(SU2, PermutationCoefficients)


def test_fusion_block_key_is_reachable():
    t = mixed_tensor()
    assert isinstance(t.structure.block_order[0], FusionBlockKey)
