"""The batched recoupling writes what the term walk writes, out of one buffer per bucket.

[lower_plan][tenet.map_view.lower_plan] runs a plan's terms in buckets: a bucket's sources
are gathered, scaled by the bucket's coefficients and summed, and only the sum reaches the
destination matrix. The gather is a fresh buffer -- an index array never returns a view --
so the scaling and the summation are written back into it rather than into two more arrays
of the bucket's size. A bucket is a multiple of the tensor on a plan big enough to have
one, so those spare arrays are traffic and not bookkeeping.

Writing in place is the kind of change that stays invisible until the arithmetic is read
back byte for byte, which is what this file does. The reference is the plan applied term
by term -- ``lower_plan`` declining, the route an immutable backend still takes -- and it
runs over a graded provider, a product of three, and an Abelian one, because a term that
carries no coefficient exercises neither the scaling nor the accumulation.
"""

import importlib

import numpy as np
import pytest
from helpers import count_backend_calls

import tenet.map_view as map_view_module
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, to_matrices
from tenet.map_view import lower_plan
from tenet.ops.batch import batch_plan
from tenet.ops.repartition import repartition, repartition_plan
from tenet.symmetry import (
    SU2,
    U1,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    U1Sector,
    fZ2,
)

repartition_module = importlib.import_module("tenet.ops.repartition")
"""The module, not the function ``tenet.ops`` re-exports under the same name."""

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
SU2_SPACE = GradedSpace.new(SU2, {ZERO: 2, HALF: 2, ONE: 2})
SU2_RAGGED = GradedSpace.new(SU2, {ZERO: 3, HALF: 2, ONE: 1})
U1_SPACE = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
FZ2_SPACE = GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 2})
PRODUCT = ProductProvider((fZ2, U1, SU2))
PRODUCT_SPACE = GradedSpace.new(
    PRODUCT,
    {
        ProductSector((FZ2Sector(0), U1Sector(0), SU2Sector(0))): 2,
        ProductSector((FZ2Sector(1), U1Sector(1), SU2Sector(1))): 2,
    },
)

# ``(space, outputs, inputs)``: a repartition that moves a leg across sides *and*
# reorders within a side, which is the pair that makes a graded provider pay a
# coefficient. Rank 5 on SU(2) is the case whose plan really buckets -- an Abelian
# grading is one term per destination and has nothing to group -- and it is here beside
# the Abelian ones because both routes have to agree.
CASES = [
    pytest.param(SU2_SPACE, (0, 1, 2, 3), (4,), id="su2-rank5"),
    pytest.param(SU2_RAGGED, (0, 1, 2, 3), (4,), id="su2-rank5-ragged"),
    pytest.param(PRODUCT_SPACE, (0, 1, 2, 3), (4,), id="fz2xu1xsu2-rank5"),
    pytest.param(FZ2_SPACE, (3, 1), (0, 2), id="fz2-rank4"),
    pytest.param(U1_SPACE, (3, 1), (0, 2), id="u1-rank4"),
]


def _tensor(space, rank, seed):
    legs = tuple(Leg(space, OUT if i % 2 == 0 else IN) for i in range(rank))
    return SymmetricTensor.random(legs, seed=seed)


def _plan(space, outputs, inputs, seed=11):
    t = _tensor(space, len(outputs) + len(inputs), seed)
    plan = repartition_plan(t.structure, outputs, inputs)
    return t, plan.new_structure, plan.perm, plan.terms


def _walked(monkeypatch, t, outputs, inputs):
    """The reference: the same plan applied one term at a time, then gathered."""
    monkeypatch.setattr(repartition_module, "lower_plan", lambda *args: None)
    return to_matrices(repartition(t, outputs, inputs))


@pytest.mark.parametrize(
    "space,outputs,inputs", [p.values for p in CASES], ids=[p.id for p in CASES]
)
def test_the_batched_recoupling_is_byte_identical_to_the_term_walk(
    monkeypatch, space, outputs, inputs
):
    """Block for block, byte for byte, against the route that defines the values."""
    t, structure, perm, terms = _plan(space, outputs, inputs)
    got = lower_plan(t, structure, perm, terms)
    assert got is not None
    want = _walked(monkeypatch, t, outputs, inputs)
    assert sorted(got) == sorted(want)
    for c in want:
        assert got[c].tobytes() == want[c].tobytes(), c


@pytest.mark.parametrize("blinded", ["multiply", "add"])
@pytest.mark.parametrize(
    "space,outputs,inputs", [p.values for p in CASES], ids=[p.id for p in CASES]
)
def test_a_backend_call_that_ignores_out_breaks_the_comparison(
    monkeypatch, blinded, space, outputs, inputs
):
    """The mutation check: the byte-identity test is not comparing a route with itself.

    Writing into ``out=`` is the whole of what the in-place gather buys, so a call that
    drops the keyword is the mutation that leaves every shape and every sector right and
    the arithmetic wrong -- a scaling that never reaches its buffer, an accumulation that
    never reaches its first summand. It is skipped, rather than silently passed, on the
    plans that have nothing for that call to do: an Abelian grading pays no coefficient,
    and a plan with one term per destination accumulates nothing.
    """
    t, structure, perm, terms = _plan(space, outputs, inputs)
    counts: dict[int, int] = {}
    for _, dst, _ in terms:
        counts[dst] = counts.get(dst, 0) + 1
    if blinded == "add" and not any(n > 1 for n in counts.values()):
        pytest.skip("one term per destination: nothing accumulates")
    if blinded == "multiply" and all(coeff == 1 for _, _, coeff in terms):
        pytest.skip("every coefficient is one: nothing scales")

    want = lower_plan(t, structure, perm, terms)
    real = map_view_module.lib_fn

    def blind(backend, name):
        fn = real(backend, name)
        if name != blinded:
            return fn

        def dropped(*args, **kwargs):
            return fn(*args, **{k: v for k, v in kwargs.items() if k != "out"})

        return dropped

    monkeypatch.setattr(map_view_module, "lib_fn", blind)
    broken = lower_plan(t, structure, perm, terms)
    assert broken is not None and want is not None
    assert any(broken[c].tobytes() != want[c].tobytes() for c in want)


def test_a_bucket_costs_one_gather_and_no_temporary():
    """The claim, as a count: every scaling and every summation lands in ``out=``.

    Counted as a formula in the plan's own structure -- buckets, their widths and the
    tail the grouping declined -- rather than as a number, so the assertion still says
    what it means when the fixture changes. The out-of-place counts are the ones that
    matter: each would be one more array the size of a bucket.
    """
    t, structure, perm, terms = _plan(SU2_SPACE, (0, 1, 2, 3), (4,), seed=8)
    groups, loose = batch_plan(structure, perm, terms)
    assert groups, "the fixture has to bucket for the count to say anything"

    calls: dict[tuple[str, bool], int] = {}
    with count_backend_calls(
        pytest.MonkeyPatch(),
        lambda name, a, k: calls.__setitem__(
            (name, "out" in k), calls.get((name, "out" in k), 0) + 1
        ),
    ):
        mats = lower_plan(t, structure, perm, terms)
    assert mats is not None

    buckets = [b for _, bs in groups for b in bs]
    written = {dst for _, _, _, dsts in buckets for dst in dsts}
    loose_scaled = loose_summed = 0
    for _, dst, coeff in loose:
        if dst in written:
            loose_summed += 1
        else:
            written.add(dst)
            loose_scaled += coeff != 1

    assert calls.get(("multiply", False), 0) == 0
    assert calls.get(("add", False), 0) == 0
    assert calls.get(("multiply", True), 0) == len(buckets) + loose_scaled
    assert calls.get(("add", True), 0) == sum(w - 1 for _, _, w, _ in buckets) + loose_summed
    # one stack per group, never one per term
    assert sum(n for (name, _), n in calls.items() if name == "stack") == len(groups)


def test_the_fixtures_really_carry_coefficients_and_multiplicities():
    """Without this the byte-identity above would prove nothing about the arithmetic."""
    for space, outputs, inputs in [p.values for p in CASES]:
        _, _, _, terms = _plan(space, outputs, inputs)
        counts: dict[int, int] = {}
        for _, dst, _ in terms:
            counts[dst] = counts.get(dst, 0) + 1
        graded = any(coeff != 1 for _, _, coeff in terms)
        assert graded == (space.provider is not U1), space.provider.name
        if space.provider in {SU2, PRODUCT}:
            assert max(counts.values()) > 1, space.provider.name


def test_the_gather_a_bucket_scales_in_place_is_its_own_buffer():
    """The premise of writing back into it: an index array never returns a view.

    Pinned because the in-place scaling is only safe while it holds -- were the gather a
    view of the stacked sources, scaling it would corrupt the sources the next bucket
    reads.
    """
    stacked = np.arange(24.0).reshape(6, 2, 2)
    gathered = stacked[np.array([0, 0, 3])]
    assert not np.shares_memory(gathered, stacked)
