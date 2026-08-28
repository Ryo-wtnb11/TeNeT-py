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
from helpers import count_backend_calls

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import is_identity_plan as _is_identity
from tenet.map_view import map_layout
from tenet.ops.batch import MIN_BATCH_ROWS, batch_plan
from tenet.ops.permutation import permutation_plan
from tenet.ops.repartition import (
    _batches,
    _looped,
    apply_plan,
    repartition_plan,
)
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
    """A rank-``rank`` tensor with sides interleaved, so every split needs a real bend.

    At rank 8 the space is thinned to two sectors. That is a CI-budget trim and it is
    measured rather than guessed: three sectors give 23,872 blocks and 363,232 terms,
    two give 323 and 1,205, and the plan that the grid rows spend their time *building*
    goes from 5.9 s to 52 ms. Degeneracy is not the knob -- block count, term count,
    multiplicities and group count are functions of the sectors alone, so thinning the
    degeneracies instead leaves the cost where it was.

    Two sectors still carry what rank 8 is here for: more than one internal bond, and
    buckets wide enough to batch. The full three-sector case is not lost -- it is
    :func:`test_the_rank_8_su2_case_is_the_one_the_issue_measures`, which pins the
    fixture the issue measured, once rather than across the grid.

    The suite's CI job runs at 26 minutes against a 30 minute cap on ``main`` (#320);
    this module went over it.
    """
    space = SPACES[provider][SHAPES.index(shape)]
    if rank == 8:
        keep = [a for a, _ in space.sectors][:2]
        space = GradedSpace.new(space.provider, {a: space.degeneracy(a) for a in keep})
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
    """Batching only matters where the buckets are large; assert this fixture has them.

    Built here rather than through :func:`tensor`, which thins rank 8 to two sectors for
    the grid's sake. This is the one place the issue's own three-sector fixture runs, so
    it is the one place the term count it reports is asserted.
    """
    space = SPACES["su2"][0]
    legs = tuple(Leg(space, OUT if i % 2 == 0 else IN) for i in range(8))
    t = SymmetricTensor.random(legs, seed=11)
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
    """Count the array operations ``apply_plan`` issues.

    Both routes to a backend are counted: ``ar.do``, and the function
    [lib_fn][tenet.backend.lib_fn] resolved once and called per block. The second is
    why the cache is cleared around the patch -- a function resolved before the patch
    would call the backend without passing through the counter.
    """
    counted: list[str] = []
    with count_backend_calls(monkeypatch, lambda name, args, kwargs: counted.append(name)):
        yield counted


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


def gathered(structure):
    """The dispatches that gathering the result's blocks into its storage costs.

    A tensor holds one dense matrix per coupled sector, so building one from blocks is
    one ``empty`` per sector plus, where the public axis order is not
    ``(*out_axes, *in_axes)``, one transposed **view** per block. Neither is the term
    count, which is the claim; on the SU(2) fixtures here the terms outnumber the blocks
    several times over.
    """
    layout = map_layout(structure)
    permuted = layout.axes_order != tuple(range(structure.ndim))
    return len(layout.sectors) + (structure.num_blocks if permuted else 0)


@pytest.mark.parametrize("rank", [5, 8])
def test_the_dispatch_count_is_the_grouping_s_and_not_the_term_count_s(ops, rank):
    """The scaling claim, structurally: the count is exactly the grouping's formula."""
    t = tensor("su2", "uniform", rank)
    structure, perm, terms = bend_plan_of(t)
    batch_plan(structure, perm, terms)  # plan side; not part of the dispatch count
    t.blocks  # noqa: B018  # cutting the source's blocks out is the tensor's, memoized
    ops.clear()
    apply_plan(t, structure, perm, terms, "test")
    assert len(ops) == predicted(structure, perm, terms) + gathered(structure)
    assert len(ops) < len(terms)


def test_the_dispatch_count_barely_moves_when_the_term_count_multiplies(ops):
    """Rank 5 to rank 8 multiplies the terms; the buckets, and so the ops, do not follow.

    Both ranks are built here on the full three-sector space rather than through
    :func:`tensor`, which thins rank 8 for the grid's sake: the whole claim is a *ratio*
    between the two term counts, so both sides have to come from one space.
    """
    space = SPACES["su2"][0]
    counts = {}
    for rank in (5, 8):
        legs = tuple(Leg(space, OUT if i % 2 == 0 else IN) for i in range(rank))
        t = SymmetricTensor.random(legs, seed=3 + rank)
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
def test_jax_takes_the_loop_and_gets_the_same_numbers(provider):
    """JAX is outside the gate, and its result is the looped one to the bit.

    [_batches][tenet.ops.repartition._batches] carries the measurement: batching costs
    JAX 1.1x to 2.8x where it saves NumPy 1.4x to 3.3x, because an array library's
    per-call overhead is large enough that the loop was never the cost. The formulation
    is not the reason -- nothing in [batch_plan][tenet.ops.batch.batch_plan] needs a
    NumPy-only primitive, and this test is what would catch it if it did.

    The PyTorch row of this is in ``tests/backends/test_torch.py``, which is the one
    module allowed to import torch.
    """
    pytest.importorskip("jax")
    t = tensor(provider, "ragged", 5).to_backend("jax")
    structure, perm, terms = bend_plan_of(t)
    # su2's bending plan has buckets to batch; u1's is one term per destination and has
    # none, which is the Abelian case the loop would have kept anyway
    assert bool(batch_plan(structure, perm, terms)[0]) is (provider == "su2")
    assert not _batches(t)  # and the backend gate declines it either way
    got = assert_bit_identical(t, structure, perm, terms)
    assert got.backend == "jax"


def test_the_public_operation_is_unchanged():
    """The end of the chain: ``repartition`` itself, against the dense oracle."""
    t = tensor("u1", "ragged", 4)
    r = tenet.repartition(t, (0, 2), (1, 3))
    np.testing.assert_allclose(r.to_dense(), np.transpose(t.to_dense(), (0, 2, 1, 3)), atol=1e-12)


# --- the plan that asks for nothing ------------------------------------------------


def test_a_contraction_that_bends_nothing_restores_with_the_identity(ops):
    """The restore of an unbent contraction is one term per block, all of them in place.

    It is not a rare shape: any ``tensordot`` whose contracted axes already sit on the
    right sides composes it. Walking it rebuilds the tensor it read -- 613,468 terms on
    an SU(2) rank-8 intermediate -- so ``apply_plan`` hands the source straight back,
    and the dispatch count for the whole restore is zero.
    """
    t = tensor("su2", "uniform", 5)
    structure, perm, terms = bend_plan_of(t)
    assert not _is_identity(structure, perm, terms)  # a real bend is not the identity

    bent = apply_plan(t, structure, perm, terms, "test")
    back = repartition_plan(bent.structure, t.structure.out_axes, t.structure.in_axes)
    assert _is_identity(*_plan_args(back)) is (back.new_structure == bent.structure)

    in_place = tuple((i, i, 1) for i in range(bent.structure.num_blocks))
    identity = (bent.structure, tuple(range(bent.ndim)), in_place)
    assert _is_identity(*identity)
    ops.clear()
    assert apply_plan(bent, *identity, "test") is bent
    assert ops == []


def _plan_args(plan):
    return plan.new_structure, plan.perm, plan.terms


def test_the_identity_is_not_claimed_for_a_plan_that_drops_or_repeats_a_block():
    """``len(terms) == num_blocks`` is not enough: the destinations have to be all of them."""
    t = tensor("u1", "ragged", 3)
    n = t.structure.num_blocks
    assert n > 2
    repeated = (*((i, i, 1) for i in range(n - 2)), (n - 2, n - 2, 1), (n - 2, n - 2, 1))
    assert len(repeated) == n
    assert not _is_identity(t.structure, tuple(range(t.ndim)), repeated)
    scaled_term = (*((i, i, 1) for i in range(n - 1)), (n - 1, n - 1, 2))
    assert not _is_identity(t.structure, tuple(range(t.ndim)), scaled_term)
