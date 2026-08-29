"""``einsum_chain``: the contraction chain as one plan per step — issue #260 (M71).

The claim under test is the principle, not a speedup: **nothing is materialized between
two lowerings**. Step ``k``'s restore and step ``k+1``'s operand lowering (and the bend
the composition rule demands at that pair) compose into one plan from step ``k``'s sector
matrices to step ``k+1``'s, so the chain writes one tensor — the last one.

The fixture is ``network/env.py``'s ``_heff2_full`` shape on every provider: the two-site
matvec, four pair-contractions, three of them with an explicit bend. It is reproduced
here from random tensors rather than from a DMRG state so that SU(2) and ``SUNProvider``
(3) — where 93–95 % of a bend's terms carry a coefficient — run the same chain the
Abelian providers do.

Bit-identity holds on every Abelian provider. It does not on SU(2)/SU(3), and cannot: a
composed plan applies ``coeff_k * coeff_{k+1}`` where the separate calls apply one after
the other, and sums duplicate ``(source, target)`` pairs before the blocks are touched
rather than after. Both are the same sum reassociated, which is a last-ulp difference and
is asserted as such.
"""

import numpy as np
import pytest
from helpers import count_backend_calls

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import map_layout
from tenet.network.env import _composed
from tenet.ops import contraction as ct
from tenet.symmetry import SU2, U1, Z2, FZ2Sector, SU2Sector, U1Sector, Z2Sector, fZ2
from tenet.symmetry.sun import SUNProvider, SUNSector

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
SU3 = SUNProvider(3)
S1, S3, S3B, S8 = SUNSector((0, 0)), SUNSector((1, 0)), SUNSector((0, 1)), SUNSector((1, 1))

#: ``(bond, physical, MPO bond)`` per provider.
SPACES = {
    "u1": (
        GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 2}),
        GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1}),
        GradedSpace.new(U1, {U1Sector(-2): 1, U1Sector(0): 2, U1Sector(2): 1}),
    ),
    "z2": (
        GradedSpace.new(Z2, {Z2Sector(0): 3, Z2Sector(1): 3}),
        GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1}),
        GradedSpace.new(Z2, {Z2Sector(0): 2, Z2Sector(1): 2}),
    ),
    "fz2": (
        GradedSpace.new(fZ2, {FZ2Sector(0): 3, FZ2Sector(1): 3}),
        GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2}),
        GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2}),
    ),
    "su2": (
        GradedSpace.new(SU2, {ZERO: 2, HALF: 3, ONE: 2}),
        GradedSpace.new(SU2, {HALF: 1}),
        GradedSpace.new(SU2, {ZERO: 1, ONE: 1}),
    ),
    "su3": (
        GradedSpace.new(SU3, {S1: 2, S3: 2, S3B: 2}),
        GradedSpace.new(SU3, {S3: 1}),
        GradedSpace.new(SU3, {S1: 1, S8: 1}),
    ),
}
PROVIDERS = list(SPACES)
GRADED = ["fz2", "su2", "su3"]
ABELIAN = ["u1", "z2", "fz2"]


def operands(name, seed=0):
    """``(aa, fr, fl, w1, w2)`` with ``_heff2_full``'s exact leg pattern."""
    b, p, m = SPACES[name]
    aa = SymmetricTensor.random((Leg(b, OUT), Leg(p, OUT), Leg(p, OUT), Leg(b, IN)), seed=seed)
    fr = SymmetricTensor.random((Leg(b, OUT), Leg(m, IN, True), Leg(b, IN)), seed=seed + 1)
    fl = SymmetricTensor.random((Leg(b, IN), Leg(m, OUT, True), Leg(b, OUT)), seed=seed + 2)
    w = [
        SymmetricTensor.random(
            (Leg(m, IN, True), Leg(p, OUT), Leg(p, IN), Leg(m, OUT, True)), seed=seed + k
        )
        for k in (3, 4)
    ]
    return aa, fr, fl, w[0], w[1]


def fused(aa, fr, fl, w1, w2):
    """``_heff2_full``'s chain, spelled once."""
    return tenet.einsum_chain(
        [
            ("apqr,rys->apqys", aa, fr, ""),
            ("apqys,mQqy->apQms", None, w2, "q"),
            ("apQms,xPpm->aPQxs", None, w1, "p"),
            ("aPQxs,axB->BPQs", None, fl, "a"),
        ]
    )


def unfused(aa, fr, fl, w1, w2):
    """The same four contractions as four separate calls, each writing its result."""
    t = tenet.einsum("apqr,rys->apqys", aa, fr)
    t = _composed("apqys,mQqy->apQms", t, w2, bend="q")
    t = _composed("apQms,xPpm->aPQxs", t, w1, bend="p")
    return _composed("aPQxs,axB->BPQs", t, fl, bend="a")


def _is_coefficient(block, other) -> bool:
    """Whether ``block * other`` is a plan's coefficient pass and not ordinary arithmetic.

    A plan scales a whole bucket at once, so its right operand is the column of
    coefficients ``cast_coefficients`` builds: one entry per row of the bucket and a
    trailing 1 for every axis of a block. Nothing else in a contraction has that shape.
    """
    shape = tuple(getattr(other, "shape", ()))
    return len(shape) == len(block.shape) and set(shape[1:]) <= {1}


class Spy:
    """Every element a coefficient multiply touches, and every tensor written."""

    def __init__(self):
        self.rides_a_write = 0  # scaled by one array multiply over a whole bucket
        self.own_pass = 0  # scaled into a temporary of its own
        self.terms = 0
        self.empty = 0
        self.applied = 0  # ``apply_plan`` calls: one tensor written out of a plan


@pytest.fixture
def spy(monkeypatch):
    """Count through ``ar.do`` *and* through the operator form ``ar.do`` cannot see."""

    import tenet.map_view as mv
    import tenet.ops.permutation as pm
    from tenet.ops import repartition as rp_fn  # noqa: F401  # the name is the function

    rp = __import__("sys").modules["tenet.ops.repartition"]
    counts = Spy()
    real_scaled, real_apply = mv.scaled, rp.apply_plan

    def scaled(block, coeff):
        counts.own_pass += block.size
        counts.terms += 1
        return real_scaled(block, coeff)

    def record(name, args, kwargs):
        if name == "multiply" and len(args) == 2 and _is_coefficient(*args):
            counts.rides_a_write += args[0].size
            counts.terms += 1
        elif name in {"empty", "zeros"}:
            counts.empty += 1

    def apply_plan(*args, **kwargs):
        counts.applied += 1
        return real_apply(*args, **kwargs)

    for module in (mv, pm, rp):
        monkeypatch.setattr(module, "scaled", scaled)
    monkeypatch.setattr(rp, "apply_plan", apply_plan)
    monkeypatch.setattr(ct, "apply_plan", apply_plan)
    with count_backend_calls(monkeypatch, record):
        yield counts


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_fused_chain_is_the_four_separate_calls(name):
    """Same tensor, to the bit on every Abelian provider and to the ulp on the others."""
    ops = operands(name)
    got, want = fused(*ops), unfused(*ops)
    assert got.structure == want.structure
    if name in ABELIAN:
        assert all(x.tobytes() == y.tobytes() for x, y in zip(got.blocks, want.blocks, strict=True))
    else:
        # composing the plans multiplies the coefficients together and sums duplicate
        # (source, target) pairs before any block moves; the separate calls apply them
        # one after the other. The same sum, reassociated.
        assert float(tenet.norm(tenet.subtract(got, want))) < 1e-14 * float(tenet.norm(got))
        pairs = zip(got.blocks, want.blocks, strict=True)
        assert not all(x.tobytes() == y.tobytes() for x, y in pairs)


@pytest.mark.parametrize("name", GRADED)
def test_the_graded_fixtures_really_carry_coefficients(name, spy):
    """The control on the control: without ``coeff != 1`` the identity above is vacuous."""
    fused(*operands(name))
    assert spy.terms > 0
    assert spy.rides_a_write > 0


@pytest.mark.parametrize("name", ["u1", "z2"])
def test_the_ungraded_fixtures_carry_none(name, spy):
    """U(1) and Z2 pay no coefficient anywhere in the chain, before or after."""
    fused(*operands(name))
    assert (spy.terms, spy.rides_a_write, spy.own_pass) == (0, 0, 0)


@pytest.mark.parametrize("name", PROVIDERS)
def test_no_tensor_is_written_between_two_lowerings(name, spy):
    """The principle, counted: one ``apply_plan`` for the whole chain, not one per step."""
    fused(*operands(name))
    assert spy.applied == 1


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_unfused_chain_writes_a_tensor_at_every_step(name, spy):
    """The baseline the count above is against: ten plan applications for four steps."""
    unfused(*operands(name))
    assert spy.applied == 10


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_only_allocations_are_each_step_s_sector_matrices(name, spy):
    """One ``empty`` per coupled sector per step, and nothing else preallocated."""
    ops = operands(name)
    steps = []
    real = ct._contracted

    def watch(a, b, axes, after):
        out = real(a, b, axes, after)
        steps.append(out)
        return out

    ct._contracted = watch
    try:
        fused(*ops)
    finally:
        ct._contracted = real
    # two operand lowerings per step, one sector matrix each, plus ``compose_lowered``'s
    # zero fill for a sector only one side carries
    assert spy.empty > 0
    assert spy.empty <= sum(3 * len(map_layout(s.source.structure).sectors) for s in steps)


@pytest.mark.parametrize("name", PROVIDERS)
def test_each_step_reads_the_previous_step_s_matrices(name):
    """``shares_memory``: a step's sources are views into the step before it, not copies."""
    products, reads = [], []
    real_compose, real_lower = ct.compose_lowered, ct.lower_plan

    def watch_compose(*args, **kwargs):
        out = real_compose(*args, **kwargs)
        products.append(out)
        return out

    def watch_lower(t, *args, **kwargs):
        reads.append(t)
        return real_lower(t, *args, **kwargs)

    ct.compose_lowered, ct.lower_plan = watch_compose, watch_lower
    try:
        fused(*operands(name))
    finally:
        ct.compose_lowered, ct.lower_plan = real_compose, real_lower

    # eight lowerings for four steps; the odd ones read the running result
    assert len(products) == 4
    assert len(reads) == 8
    for step, product in enumerate(products[:-1]):
        source = reads[2 * (step + 1)]
        assert source is product
        assert all(
            np.shares_memory(block, other)
            for block, other in zip(source.blocks, product.blocks, strict=True)
        )


def test_the_chain_refuses_a_first_step_that_names_none():
    a, b, *_ = operands("u1")
    with pytest.raises(ValueError, match="stands for the previous step"):
        tenet.einsum_chain([("apqr,rys->apqys", None, b, "")])


def test_the_chain_refuses_no_steps():
    with pytest.raises(ValueError, match="no steps were given"):
        tenet.einsum_chain([])


def test_the_chain_refuses_a_step_with_two_holes():
    a, b, *_ = operands("u1")
    with pytest.raises(ValueError, match="names None on both sides"):
        tenet.einsum_chain([("apqr,rys->apqys", a, b, ""), ("apqys,rys->apqr", None, None, "")])


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_hole_may_stand_on_either_side(name):
    """Operand order is categorical data, so the chain writes out which side it is on."""
    b, p, _ = SPACES[name]
    x = SymmetricTensor.random((Leg(b, OUT), Leg(p, OUT), Leg(b, IN)), seed=7)
    y = SymmetricTensor.random((Leg(b, OUT), Leg(b, IN)), seed=8)
    z = SymmetricTensor.random((Leg(b, OUT), Leg(b, IN)), seed=9)
    first = tenet.einsum("apr,rs->aps", x, y)
    left = tenet.einsum_chain([("apr,rs->aps", x, y, ""), ("aps,st->apt", None, z, "")])
    right = tenet.einsum_chain([("apr,rs->aps", x, y, ""), ("st,tpr->spr", z, None, "")])
    assert bool(tenet.allclose(left, tenet.einsum("aps,st->apt", first, z)))
    assert bool(tenet.allclose(right, tenet.einsum("st,tpr->spr", z, first)))
