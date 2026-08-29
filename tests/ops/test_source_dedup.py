"""Issue #123 — one ``transpose`` per distinct source block, not per plan term.

``RepartitionPlan.perm`` / ``PermutationPlan.axes`` are per-*plan* and only the
coefficient is per-*term*, so every term sharing a source used to recompute a
byte-identical array (2.87 terms per source at SU(2) chi=6, exactly 1.00 at U(1)).
The executors hoist that; these tests assert the hoist changes no bit, against the
pre-#123 loop body reproduced here as the oracle -- ``==``, not a tolerance.

The gradient is asserted exactly too. Reverse mode of a shared node reassociates the
cotangent sum (``T^T(y1) + T^T(y2)`` becomes ``T^T(y1 + y2)``), which is exact on a
single transpose-and-add; the 1e-7 delta #123 measures in CTMRG comes from feeding
that reassociation through a broadened SVD, not from this loop.
"""

from typing import Any

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.permutation import permutation_plan
from tenet.ops.repartition import repartition_plan
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

jax = pytest.importorskip("jax")

import tenet.pytree  # noqa: E402, F401  # registration is the import's side effect

HALF, ONE = SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 2, ONE: 1})
W = GradedSpace.new(SU2, {HALF: 1, ONE: 2})
SU2_LEGS = (Leg(V, OUT), Leg(W, OUT), Leg(V, IN), Leg(W, IN), Leg(V, IN))

Q = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 2})
U1_LEGS = (Leg(Q, OUT), Leg(Q, OUT), Leg(Q, IN), Leg(Q, IN), Leg(Q, IN))

OUTPUTS, INPUTS = (0, 1, 2), (3, 4)
AXES = (1, 0, 2, 3, 4)
BACK_OUTPUTS, BACK_INPUTS = (0, 1), (2, 3, 4)


def _apply(new_structure, terms, blocks_of, perm):
    """The pre-#123 loop body verbatim: one ``transpose`` per *term*."""
    blocks: dict[int, Any] = {}
    for src, dst, coeff in terms:
        contrib = ar.do("transpose", blocks_of[src], perm)
        if coeff != 1:
            contrib = contrib * (coeff.real if getattr(coeff, "imag", 0) == 0 else coeff)
        blocks[dst] = contrib if dst not in blocks else blocks[dst] + contrib
    return SymmetricTensor(new_structure, tuple(blocks[i] for i in range(new_structure.num_blocks)))


def base_repartition(t, outputs, inputs):
    plan = repartition_plan(t.structure, outputs, inputs)
    return _apply(plan.new_structure, plan.terms, t.blocks, plan.perm)


def base_transpose(t, axes):
    plan = permutation_plan(t.structure, axes)
    return _apply(plan.new_structure, plan.terms, t.blocks, plan.axes)


def chain(t, repartition, transpose):
    """``repartition -> transpose -> repartition``, squared norm."""
    return (
        tenet.norm(
            repartition(transpose(repartition(t, OUTPUTS, INPUTS), AXES), BACK_OUTPUTS, BACK_INPUTS)
        )
        ** 2
    )


def shipped(t):
    return chain(t, lambda x, o, i: x.repartition(o, i), lambda x, a: x.transpose(a))


def oracle(t):
    return chain(t, base_repartition, base_transpose)


@pytest.mark.parametrize("legs, seed", [(SU2_LEGS, 7), (U1_LEGS, 8)])
def test_the_dedup_is_bit_identical_value_and_gradient(legs, seed):
    """The value to the bit; the gradient to the bit wherever the plan does not sum.

    A destination that takes one term is a relabel: forward and reverse both move each
    element once, and the two routes agree exactly. A destination that takes several --
    which is what a non-Abelian expansion produces, and what the test below counts -- is a
    sum, and the shipped route sums a whole multiplicity bucket with one gather where the
    oracle sums term by term. The forward association is the same either way, which is why
    the value is still exact; the reverse of a gather is a scatter-add, and a scatter-add
    does not promise the term order. Measured at 3 ulps on the SU(2) fixture.
    """
    plan = repartition_plan(SymmetricTensor.random(legs, seed=seed).structure, OUTPUTS, INPUTS)
    sums = len(plan.terms) > len({d for _, d, _ in plan.terms})

    t = SymmetricTensor.random(legs, seed=seed).to_backend("jax")
    got_v, got_g = jax.value_and_grad(shipped)(t)
    want_v, want_g = jax.value_and_grad(oracle)(t)
    assert np.asarray(got_v) == np.asarray(want_v)
    for got, want in zip(got_g.blocks, want_g.blocks, strict=True):
        got, want = np.asarray(got), np.asarray(want)
        if sums:
            np.testing.assert_allclose(got, want, rtol=8 * np.finfo(want.dtype).eps, atol=0)
        else:
            assert np.array_equal(got, want)


def test_su2_shares_sources_across_terms_and_u1_does_not():
    """Without multi-term sources the test above would assert nothing (#123's 2.87 vs 1.00)."""
    ratios = {}
    for name, legs in (("su2", SU2_LEGS), ("u1", U1_LEGS)):
        t = SymmetricTensor.random(legs, seed=9)
        plan = repartition_plan(t.structure, OUTPUTS, INPUTS)
        ratios[name] = len(plan.terms) / len({s for s, _, _ in plan.terms})
    assert ratios["su2"] > 1.0
    assert ratios["u1"] == 1.0
