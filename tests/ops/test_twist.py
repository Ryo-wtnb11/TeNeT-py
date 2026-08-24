"""``tenet.twist`` and the closure that pays it (M82 phase 2, #287).

The twist is a scalar per block, so the interesting statement is not the scalar: it is
that a **closed** fermionic loop stops depending on where it was cut once the closure
pays it. The 4-cycle below is the smallest network that has one -- four rank-3 tensors,
no double layer, no PEPS -- and it is the reproducer the milestone was opened on.
"""

import itertools

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network.common import composed, supplies_in
from tenet.symmetry import U1, FZ2Sector, U1Sector, fZ2

FZ2_V = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
U1_V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})

#: The 4-cycle: wire -> its two tensors, and each tensor's public leg labels.
LOOP = {"u": ("A", "B"), "w": ("B", "D"), "x": ("D", "C"), "v": ("C", "A")}
LEGS = {"A": "auv", "B": "ubw", "C": "vcx", "D": "wxd"}
WIRINGS = list(itertools.product("AB", "BD", "DC", "CA"))


def build(space, sides, seed=1):
    """The four tensors; ``sides[wire]`` names the tensor holding that wire's OUT end."""
    return {
        name: SymmetricTensor.random(
            tuple(Leg(space, OUT if (c not in LOOP or sides[c] == name) else IN) for c in labels),
            seed=seed + k,
        )
        for k, (name, labels) in enumerate(LEGS.items())
    }


def connected(legs, order):
    seen = set(legs[order[0]])
    for name in order[1:]:
        if not seen & set(legs[name]):
            return False
        seen |= set(legs[name])
    return True


def contract(tensors, legs, order):
    """Contract in ``order``; also report whether any step had to bend a wire."""
    acc, labels, bent = None, "", False
    for name in order:
        here = legs[name]
        if acc is None:
            acc, labels = tensors[name], here
            continue
        shared = [c for c in labels if c in here]
        out = "".join(c for c in labels if c not in shared)
        out += "".join(c for c in here if c not in shared)
        turns = sum(not supplies_in(acc.legs[labels.index(c)]) for c in shared)
        bent = bent or min(turns, len(shared) - turns) > 0
        acc, labels = composed(f"{labels},{here}->{out}", acc, tensors[name]), out
    return acc, labels, bent


def dense_in(t, labels, target):
    """One public leg order for every spelling, through ``tenet.transpose``.

    A graded tensor's dense array carries the Koszul signs of *its own* leg order, so
    ``np.transpose`` on the array is not a leg permutation and would compare nothing.
    """
    return np.asarray(tenet.transpose(t, [labels.index(c) for c in target]).to_dense())


def cut_legs(cut):
    """``LEGS`` with ``cut``'s two ends left open, as ``cut`` and its upper case."""
    legs = dict(LEGS)
    holder = LOOP[cut][0]
    legs[holder] = LEGS[holder].replace(cut, cut.upper())
    return legs


def closed_with_trace(tensors, cut, order):
    acc, labels, _ = contract(tensors, cut_legs(cut), order)
    i, j = labels.index(cut.upper()), labels.index(cut)
    rest = "".join(c for k, c in enumerate(labels) if k not in (i, j))
    return dense_in(tenet.trace(acc, (i, j)), rest, "abcd")


# --- the primitive -------------------------------------------------------------------


def test_the_twist_is_an_involution_on_fz2_and_the_identity_on_u1():
    for space, graded in ((FZ2_V, True), (U1_V, False)):
        t = SymmetricTensor.random((Leg(space, OUT), Leg(space, IN)), seed=0)
        assert tenet.allclose(tenet.twist(tenet.twist(t, 0), 0), t)
        # theta = 1 on a bosonic provider, so the tensor comes back untouched -- the
        # very same object, which is what keeps those paths bit-identical.
        assert (tenet.twist(t, 0) is t) is not graded


def test_the_twist_moves_no_metadata():
    t = SymmetricTensor.random((Leg(FZ2_V, OUT), Leg(FZ2_V, IN), Leg(FZ2_V, OUT)), seed=3)
    assert tenet.twist(t, (0, 2)).structure == t.structure
    assert tenet.twist(t, ()) is t


# --- the closure ---------------------------------------------------------------------


@pytest.mark.parametrize("sides", WIRINGS)
def test_the_traced_loop_does_not_depend_on_which_wire_was_cut(sides):
    """``trace`` is the supertrace, and that is what makes the loop's value unique.

    The open tree is order-free already, so everything the loop's value depends on lives
    in the closure: cut any of the four wires, contract the rest in any connected order,
    close with ``trace``, and the answer is one tensor. Before the twist was paid the
    four cuts spread by 2.0.
    """
    tensors = build(FZ2_V, dict(zip(LOOP, sides, strict=True)))
    values = [
        closed_with_trace(tensors, cut, order)
        for cut in sorted(LOOP)
        for order in itertools.permutations("ABCD")
        if connected(cut_legs(cut), order)
    ]
    assert len(values) == 32
    scale = np.abs(values[0]).max()
    assert scale > 1e-8, "a test whose oracle is all zeros proves nothing"
    for v in values[1:]:
        np.testing.assert_allclose(v, values[0], atol=1e-12 * scale)


@pytest.mark.parametrize("sides", WIRINGS)
def test_a_loop_closed_by_a_composition_is_the_closure_trace_takes(sides):
    """The placement M82 phase 2 left open, closed in phase 3 by the step itself.

    A loop also closes inside one ``composed`` step that contracts two wires at once, and
    the step knows: two disjoint blobs joined over ``k`` wires close ``k - 1`` cycles, so
    a step that contracts more than one wire closes something and pays ``theta`` on the
    wires it bends. Every connected order of the 4-cycle then lands on ``trace``'s
    closure, which is what says the two closures are one object.
    """
    tensors = build(FZ2_V, dict(zip(LOOP, sides, strict=True)))
    ref = closed_with_trace(tensors, "u", ("A", "B", "D", "C"))
    scale = np.abs(ref).max()
    assert scale > 1e-8, "a test whose oracle is all zeros proves nothing"
    for order in itertools.permutations("ABCD"):
        if connected(LEGS, order):
            got = dense_in(*contract(tensors, LEGS, order)[:2], "abcd")
            np.testing.assert_allclose(got, ref, atol=1e-12 * scale)
