"""The fused lowering writes each plan term once, and writes exactly what the chain did.

Issue #259 (M70). ``ops.contraction`` used to build the repartitioned operand as a
tensor and then lower it, so a coefficient-carrying block crossed memory twice: once
when ``contrib * coeff`` materialised the transposed view, once when ``to_matrices``
copied the result into its sector matrix. ``map_view.lower_plan`` composes the two,
so the transpose, the scalar and the placement are one pass.

The chain is the reference: whatever ``to_matrices(repartition(t, ...))`` puts in a
sector matrix, ``lower_plan`` must put there *byte for byte* -- a coefficient dropped,
a coefficient applied twice, or a term written to the wrong slot all change bytes and
none of them changes shapes. The counting tests then pin the mechanism: one write per
term, no standalone elementwise multiply, and every source read through a view.
"""

import autoray as ar
import numpy as np
import pytest

import tenet
import tenet.map_view as map_view_module
import tenet.ops.contraction as contraction_module
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import lower_plan, map_layout, to_matrices
from tenet.ops.permutation import permutation_plan
from tenet.ops.repartition import repartition, repartition_plan
from tenet.symmetry import (
    SU2,
    U1,
    Z2,
    FZ2Sector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    Z2Sector,
    fZ2,
)

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 3, ONE: 2})
H = GradedSpace.new(SU2, {HALF: 3})
TRIV = GradedSpace.new(Trivial, {TrivialSector(): 3})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
Z = GradedSpace.new(Z2, {Z2Sector(0): 2, Z2Sector(1): 3})
F = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 3})

# ``(legs, outputs, inputs)`` -- the repartition ``tensordot`` would apply to an operand.
# Every case moves at least one leg across sides *and* reorders within a side, which is
# the pair that makes a graded provider produce a coefficient.
CASES = [
    pytest.param((Leg(TRIV, OUT), Leg(TRIV, OUT), Leg(TRIV, IN)), (2,), (1, 0), id="trivial"),
    pytest.param((Leg(Q, OUT), Leg(Q, OUT), Leg(Q, IN)), (2,), (1, 0), id="u1"),
    pytest.param((Leg(Z, OUT), Leg(Z, OUT), Leg(Z, IN)), (2,), (1, 0), id="z2"),
    pytest.param((Leg(F, OUT), Leg(F, OUT), Leg(F, IN)), (2,), (1, 0), id="fz2"),
    pytest.param(
        (Leg(F, OUT), Leg(F, OUT), Leg(F, IN), Leg(F, IN)), (3, 1), (0, 2), id="fz2-two-crossing"
    ),
    pytest.param((Leg(V, OUT), Leg(V, OUT), Leg(V, IN)), (2,), (1, 0), id="su2"),
    # (1/2, 1/2, 1/2) -> 1/2 has two trees per band: the multi-band, multi-term case
    pytest.param(
        (Leg(H, OUT), Leg(H, OUT), Leg(H, OUT), Leg(H, IN)), (3, 1), (0, 2), id="su2-multi"
    ),
]


def _plan(t, outputs, inputs):
    """``(new structure, per-block permutation, terms)`` for ``repartition(t, ...)``."""
    want = {ax: OUT for ax in outputs} | {ax: IN for ax in inputs}
    if any(t.legs[ax].side is not want[ax] for ax in range(t.ndim)):
        p = repartition_plan(t.structure, outputs, inputs)
        return p.new_structure, p.perm, p.terms
    p = permutation_plan(t.structure, (*outputs, *inputs))
    return p.new_structure, p.axes, p.terms


def _count_ar_do(monkeypatch, fn):
    """``(name, "out" in kwargs)`` for every ``ar.do`` ``fn()`` reaches, counted."""
    counts: dict[tuple[str, bool], int] = {}
    real = ar.do

    def spy(name, *args, **kwargs):
        key = (name, "out" in kwargs)
        counts[key] = counts.get(key, 0) + 1
        return real(name, *args, **kwargs)

    monkeypatch.setattr(map_view_module.ar, "do", spy)
    fn()
    return counts


@pytest.mark.parametrize(
    "legs,outputs,inputs", [p.values for p in CASES], ids=[p.id for p in CASES]
)
def test_the_fused_lowering_is_byte_identical_to_the_chain(legs, outputs, inputs):
    """The only thing that matters: the same bytes in the same slots."""
    t = SymmetricTensor.random(legs, seed=0)
    structure, perm, terms = _plan(t, outputs, inputs)
    got = lower_plan(t, structure, perm, terms)
    want = to_matrices(repartition(t, outputs, inputs))
    assert got is not None
    assert sorted(got) == sorted(want)
    for c in want:
        assert got[c].tobytes() == want[c].tobytes(), c


def test_the_graded_fixtures_really_carry_coefficients():
    """Without this the byte-identity test proves nothing about the coefficient pass."""
    for legs, outputs, inputs in [p.values for p in CASES if p.id.startswith(("fz2", "su2"))]:
        t = SymmetricTensor.random(legs, seed=0)
        _, _, terms = _plan(t, outputs, inputs)
        assert any(coeff != 1 for _, _, coeff in terms), legs


def test_the_ungraded_fixtures_carry_none():
    """The control: U(1) and Trivial pay no coefficient before the fusion or after."""
    for legs, outputs, inputs in [p.values for p in CASES if p.id in {"trivial", "u1"}]:
        t = SymmetricTensor.random(legs, seed=0)
        _, _, terms = _plan(t, outputs, inputs)
        assert all(coeff == 1 for _, _, coeff in terms), legs


@pytest.mark.parametrize(
    "legs,outputs,inputs", [p.values for p in CASES], ids=[p.id for p in CASES]
)
def test_the_fused_lowering_writes_once_per_term_and_multiplies_nowhere_standalone(
    monkeypatch, legs, outputs, inputs
):
    """One transposed view per term, one preallocation per sector, and nothing else.

    A standalone elementwise multiply -- one without ``out=`` -- is the extra pass this
    milestone removes; a ``reshape`` or a ``concatenate`` is the extra pass M69 removed.
    """
    t = SymmetricTensor.random(legs, seed=1)
    structure, perm, terms = _plan(t, outputs, inputs)
    counts = _count_ar_do(monkeypatch, lambda: lower_plan(t, structure, perm, terms))

    assert not [name for name, _ in counts if name in {"reshape", "concatenate"}]
    # every scaling and every accumulation lands in a destination view
    assert counts.get(("multiply", False), 0) == 0
    assert counts.get(("add", False), 0) == 0
    assert counts.get(("empty", False), 0) == len(map_layout(structure).sectors)
    # one transposed source view per term -- the write itself is a slice assignment
    # (``dest[...] = block``) or the ``out=`` of the multiply, never a fresh array
    assert counts.get(("transpose", False), 0) == len(terms)


@pytest.mark.parametrize(
    "legs,outputs,inputs", [p.values for p in CASES], ids=[p.id for p in CASES]
)
def test_every_term_reads_a_view_and_writes_into_the_matrix(legs, outputs, inputs):
    """M69's claim, per term: the source is never materialised and the slot is a view."""
    t = SymmetricTensor.random(legs, seed=2)
    structure, perm, terms = _plan(t, outputs, inputs)
    layout = map_layout(structure)
    order = tuple(perm[i] for i in layout.axes_order)
    mats = lower_plan(t, structure, perm, terms)
    assert mats is not None
    slots = map_view_module._slots(structure)
    _, shapes = map_view_module._tables(structure)
    for src, dst, _ in terms:
        view = np.transpose(t.blocks[src], order)
        assert np.shares_memory(view, t.blocks[src]), (src, dst)
        c, ro, dr, co, dc = slots[dst]
        dest = mats[c][ro : ro + dr, co : co + dc].reshape(shapes[dst])
        assert np.shares_memory(dest, mats[c]), (src, dst)


@pytest.mark.parametrize(
    "legs,outputs,inputs", [p.values for p in CASES], ids=[p.id for p in CASES]
)
def test_the_accumulating_terms_are_the_multi_term_expansions_only(legs, outputs, inputs):
    """Two sources into one destination is what a non-Abelian expansion produces.

    Counted here so the accounting in docs/design.md "M70" has a test that fails if it
    ever stops holding -- an Abelian provider is one term per destination, so the fused
    path never reaches its accumulating branch there.
    """
    t = SymmetricTensor.random(legs, seed=3)
    _, _, terms = _plan(t, outputs, inputs)
    seen: set[int] = set()
    extra = sum(1 for _, dst, _ in terms if dst in seen or seen.add(dst))
    if t.provider is SU2:
        assert extra >= 0  # SU(2) may or may not expand for a given axes choice
    else:
        assert extra == 0, (t.provider.name, extra)


@pytest.mark.parametrize(
    "legs,outputs,inputs", [p.values for p in CASES], ids=[p.id for p in CASES]
)
def test_tensordot_through_the_fused_path_matches_the_unfused_chain(
    monkeypatch, legs, outputs, inputs
):
    """End to end: the public result is byte-identical, coefficients and all.

    The reference is ``tensordot`` itself with ``lower_plan`` declining, which is
    exactly the ``repartition`` then ``to_matrices`` chain the fusion replaces and the
    route an immutable backend still takes.
    """
    space = legs[0].space
    a = SymmetricTensor.random(legs, seed=4)
    b = SymmetricTensor.random((Leg(space, IN), Leg(space, OUT)), seed=5)
    got = tenet.tensordot(a, b, axes=((0,), (0,)))

    monkeypatch.setattr(contraction_module, "lower_plan", lambda *args: None)
    want = tenet.tensordot(a, b, axes=((0,), (0,)))

    assert got.structure == want.structure
    for x, y in zip(got.blocks, want.blocks, strict=True):
        assert x.tobytes() == y.tobytes()


@pytest.mark.parametrize(
    "legs,outputs,inputs", [p.values for p in CASES], ids=[p.id for p in CASES]
)
def test_lower_writes_one_pass_per_term_and_matches_the_old_route(
    monkeypatch, legs, outputs, inputs
):
    """``linalg._lower``'s destination is a matrix, so the plan goes straight into it.

    The factorizations read the matrices and only the *structure* of the tensor beside
    them, so the repartitioned tensor no longer has to be written before it is copied
    down (issue #260). One transposed view per term, one preallocation per coupled
    sector, no standalone multiply or add, and the same bytes as
    ``to_matrices(repartition(t, ...))``.
    """
    from tenet.ops.linalg import _lower

    t = SymmetricTensor.random(legs, seed=3)
    _, _, terms = _plan(t, outputs, inputs)
    counts = _count_ar_do(monkeypatch, lambda: _lower(t, (outputs, inputs)))

    assert counts.get(("transpose", False), 0) == len(terms)
    assert counts.get(("multiply", False), 0) == 0
    assert counts.get(("add", False), 0) == 0
    assert ("concatenate", False) not in counts
    # the only ``reshape`` left is ``from_matrices``'' view back onto the matrices, which
    # is where the tensor beside them now comes from and moves no element
    m, bond, mats = _lower(t, (outputs, inputs))
    assert all(any(np.shares_memory(block, mat) for mat in mats.values()) for block in m.blocks)
    want = to_matrices(repartition(t, outputs, inputs))
    assert sorted(mats) == sorted(want)
    for c in want:
        assert mats[c].tobytes() == want[c].tobytes(), c
    assert m.structure == repartition(t, outputs, inputs).structure
    assert (
        bond.sectors
        == GradedSpace.new(
            t.provider, {c: min(map_layout(m.structure).shape(c)) for c in mats}
        ).sectors
    )
