"""Batched plan application — issue #316.

``apply_plan`` used to walk a plan one term at a time; one SU(2) PEPS ``tensordot``
is 447,752 such iterations on 64-element blocks, so contraction cost tracked the
square of the block count while the ``qr`` beside it stayed BLAS-bound. The terms
now group into a few hundred array operations, and this module holds the two claims
that makes: the batched result is the looped result **to the bit**, and the number of
array operations is a function of the grouping rather than of the term count.

``_looped`` is the oracle rather than a dense expansion on purpose. The mathematics of
each plan is already tested where the plan is built (``test_repartition.py``,
``test_permutation.py``, ``test_contraction.py``); what is new here is only the
execution, so the criterion is that the two executions of the *same* plan agree
exactly — ``np.array_equal``, never ``allclose``, since anything less would let a
reassociated sum pass.
"""

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.batch import MIN_BATCH_ROWS, batch_plan
from tenet.ops.permutation import permutation_plan
from tenet.ops.repartition import _looped, apply_plan, repartition_plan
from tenet.symmetry import (
    SU2,
    U1,
    Z2,
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
from tenet.symmetry.sun import SUNProvider, SUNSector

SU3 = SUNProvider(3)
UF = ProductProvider((U1, fZ2))


def _product(charge: int, parity: int) -> ProductSector:
    return ProductSector((U1Sector(charge), FZ2Sector(parity)))


# Two degeneracy patterns per provider: uniform, where every block has the same shape
# and the whole plan is one bucket, and ragged, where it is many.
SPACES = {
    "trivial": (
        GradedSpace.new(Trivial, {TrivialSector(): 2}),
        GradedSpace.new(Trivial, {TrivialSector(): 3}),
    ),
    "u1": (
        GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 2, U1Sector(1): 2}),
        GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(0): 3, U1Sector(1): 2}),
    ),
    "z2": (
        GradedSpace.new(Z2, {Z2Sector(0): 2, Z2Sector(1): 2}),
        GradedSpace.new(Z2, {Z2Sector(0): 3, Z2Sector(1): 1}),
    ),
    "fz2": (
        GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2}),
        GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 1}),
    ),
    "su2": (
        GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2, SU2Sector(2): 2}),
        GradedSpace.new(SU2, {SU2Sector(0): 3, SU2Sector(1): 1, SU2Sector(2): 2}),
    ),
    "su3": (
        GradedSpace.new(SU3, {SUNSector((0, 0)): 2, SUNSector((1, 0)): 2, SUNSector((0, 1)): 2}),
        GradedSpace.new(SU3, {SUNSector((0, 0)): 1, SUNSector((1, 0)): 3, SUNSector((0, 1)): 2}),
    ),
    "product": (
        GradedSpace.new(UF, {_product(0, 0): 2, _product(1, 1): 2, _product(-1, 1): 2}),
        GradedSpace.new(UF, {_product(0, 0): 1, _product(1, 1): 3, _product(-1, 1): 2}),
    ),
}

PROVIDERS = list(SPACES)
SHAPES = ["uniform", "ragged"]
RANKS = [2, 5, 8]


def tensor(provider: str, shape: str, rank: int, dtype=None) -> SymmetricTensor:
    """A rank-``rank`` tensor with sides interleaved, so every split needs a real bend."""
    space = SPACES[provider][SHAPES.index(shape)]
    legs = tuple(Leg(space, OUT if i % 2 == 0 else IN) for i in range(rank))
    kwargs = {} if dtype is None else {"dtype": dtype}
    return SymmetricTensor.random(legs, seed=3 + rank, **kwargs)


def bend_plan_of(t: SymmetricTensor):
    """A repartition that bends every leg to the other side — the many-coefficient plan."""
    axes = tuple(range(t.ndim))
    plan = repartition_plan(t.structure, axes[1::2], axes[::2])
    return plan.new_structure, plan.perm, plan.terms


def transpose_plan_of(t: SymmetricTensor):
    """A plain transpose — every coefficient exactly 1 on a bosonic Abelian provider."""
    axes = (*range(1, t.ndim), 0)
    plan = permutation_plan(t.structure, axes)
    return plan.new_structure, plan.axes, plan.terms


def looped(t, structure, perm, terms):
    """What ``apply_plan`` used to build, term by term."""
    blocks = _looped(t, perm, terms, {})
    return tuple(blocks[i] for i in range(structure.num_blocks))


def assert_bit_identical(t, structure, perm, terms):
    got = apply_plan(t, structure, perm, terms, "test")
    want = looped(t, structure, perm, terms)
    assert got.structure == structure
    for a, b in zip(got.blocks, want, strict=True):
        assert ar.to_numpy(a).dtype == ar.to_numpy(b).dtype
        assert np.array_equal(ar.to_numpy(a), ar.to_numpy(b))
    return got


# --- bit-level agreement -----------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rank", RANKS)
def test_batched_equals_looped_on_a_bending_plan(provider, shape, rank):
    """Every provider, both degeneracy patterns, three ranks — the many-coefficient plan."""
    t = tensor(provider, shape, rank)
    assert_bit_identical(t, *bend_plan_of(t))


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rank", RANKS)
def test_batched_equals_looped_on_a_unit_coefficient_plan(provider, shape, rank):
    """The other extreme: a plain transpose, whose coefficients are 1 wherever a
    provider has no braiding to pay for."""
    t = tensor(provider, shape, rank)
    structure, perm, terms = transpose_plan_of(t)
    if provider in {"trivial", "u1", "z2"}:
        assert all(c == 1 for _, _, c in terms)
    assert_bit_identical(t, structure, perm, terms)


@pytest.mark.parametrize("provider", ["su2", "su3", "product"])
def test_the_bending_plans_really_carry_non_unit_coefficients(provider):
    """The control: without this the many-coefficient rows above are the unit ones."""
    t = tensor(provider, "uniform", 5)
    _, _, terms = bend_plan_of(t)
    assert sum(1 for _, _, c in terms if c != 1) > len(terms) // 4


def test_the_rank_8_su2_case_is_the_one_the_issue_measures():
    """Batching only matters where the buckets are large; assert this fixture has them."""
    t = tensor("su2", "uniform", 8)
    structure, perm, terms = bend_plan_of(t)
    groups, loose = batch_plan(structure, perm, terms)
    assert len(terms) > 5000
    assert len(groups) == 1  # uniform degeneracies: one block shape, one group
    # only the width-1 bucket is left, and it is a small fraction of the terms
    assert len(loose) < len(terms) // 50


@pytest.mark.parametrize("shape", SHAPES)
def test_batched_equals_looped_on_a_complex_tensor(shape):
    """A complex block times a real coefficient, and a complex coefficient too."""
    t = tensor("su2", shape, 5, dtype=np.complex128)
    got = assert_bit_identical(t, *bend_plan_of(t))
    assert got.blocks[0].dtype == np.complex128
    structure, perm, terms = bend_plan_of(t)
    turned = tuple((s, d, c * 1j) for s, d, c in terms)
    assert_bit_identical(t, structure, perm, turned)


# --- dtype ------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.float64, np.complex128])
def test_dtype_survives_the_coefficients(dtype):
    """A ``complex64`` tensor comes back ``complex64``: the coefficients are cast, not
    promoted."""
    t = tensor("su2", "uniform", 5, dtype=dtype)
    assert t.blocks[0].dtype == dtype
    got = apply_plan(t, *bend_plan_of(t), "test")
    assert all(b.dtype == dtype for b in got.blocks)


def test_a_complex_coefficient_still_complexifies_a_real_tensor():
    """The cast follows the coefficient's kind, so a real block times ``i`` is complex —
    and at the block's own precision, not promoted to double."""
    t = tensor("su2", "uniform", 5, dtype=np.float32)
    structure, perm, terms = bend_plan_of(t)
    turned = tuple((s, d, c * 1j) for s, d, c in terms)
    got = assert_bit_identical(t, structure, perm, turned)
    assert all(b.dtype == np.complex64 for b in got.blocks)


# --- the operation count ----------------------------------------------------------


@pytest.fixture
def ops(monkeypatch):
    """Count what ``apply_plan`` dispatches through ``autoray``."""
    counted: list[str] = []
    real = ar.do

    def do(name, *args, **kwargs):
        counted.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(ar, "do", do)
    return counted


def predicted(structure, perm, terms):
    """The dispatch count the grouping alone predicts.

    One ``stack`` and one ``transpose`` per shape group; per multiplicity bucket one
    ``array`` for the coefficients, one ``multiply``, one ``reshape`` and ``width - 1``
    ``add``s; one ``transpose`` per distinct source the loop still handles. The term
    count appears nowhere in it.
    """
    groups, loose = batch_plan(structure, perm, terms)
    total = 2 * len(groups) + len({s for s, _, _ in loose})
    for _, buckets in groups:
        total += sum(width + 2 for _, _, width, _ in buckets)
    return total


@pytest.mark.parametrize("rank", [5, 8])
def test_the_dispatch_count_is_the_grouping_s_and_not_the_term_count_s(ops, rank):
    """The scaling claim, structurally: the count is exactly the grouping's formula."""
    t = tensor("su2", "uniform", rank)
    structure, perm, terms = bend_plan_of(t)
    batch_plan(structure, perm, terms)  # plan side; not part of the dispatch count
    ops.clear()
    apply_plan(t, structure, perm, terms, "test")
    assert len(ops) == predicted(structure, perm, terms)


def test_the_dispatch_count_barely_moves_when_the_term_count_multiplies(ops):
    """Rank 5 to rank 8 multiplies the terms; the buckets, and so the ops, do not follow."""
    counts = {}
    for rank in (5, 8):
        t = tensor("su2", "uniform", rank)
        structure, perm, terms = bend_plan_of(t)
        batch_plan(structure, perm, terms)
        ops.clear()
        apply_plan(t, structure, perm, terms, "test")
        counts[rank] = (len(ops), len(terms))
    (small_ops, small_terms), (big_ops, big_terms) = counts[5], counts[8]
    assert big_terms > 700 * small_terms  # 487 terms to 363,232
    assert big_ops < 15 * small_ops  # 28 dispatches to 354
    assert big_ops < big_terms // 500


def test_a_one_term_destination_stays_on_the_loop():
    """The guard the Abelian case needs: with nothing to fuse, batching is pure loss."""
    t = tensor("u1", "ragged", 2)
    structure, perm, terms = transpose_plan_of(t)
    groups, loose = batch_plan(structure, perm, terms)
    assert not groups and len(loose) == len(terms)
    assert MIN_BATCH_ROWS >= 2


# --- backends ---------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["u1", "su2"])
def test_jax_batched_equals_jax_looped(provider):
    """JAX runs the same formulation — no NumPy-only segment-sum primitive anywhere.

    The PyTorch row of this is in ``tests/backends/test_torch.py``, which is the one
    module allowed to import torch.
    """
    pytest.importorskip("jax")
    t = tensor(provider, "ragged", 5).to_backend("jax")
    got = assert_bit_identical(t, *bend_plan_of(t))
    assert got.backend == "jax"


def test_the_public_operation_is_unchanged():
    """The end of the chain: ``repartition`` itself, against the dense oracle."""
    t = tensor("u1", "ragged", 4)
    r = tenet.repartition(t, (0, 2), (1, 3))
    np.testing.assert_allclose(r.to_dense(), np.transpose(t.to_dense(), (0, 2, 1, 3)), atol=1e-12)
