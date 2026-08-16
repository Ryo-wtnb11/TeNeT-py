"""Tests for tenet.einsum — one test per acceptance criterion of #52.

``einsum`` owns no mathematics, so most of what is checked here is *layering*:
the equation becomes an ``axes`` pair and a permutation, and everything else is
``tensordot``'s (#51) and ``transpose``'s (#21). Two consequences shape the file.

* The dense oracle is ``np.einsum`` on the expansions, except for the graded
  providers, where it is ``np.einsum`` times :func:`helpers.supersign` — the same
  scale #39 and #51 weigh their fermionic results on, never this module's own
  idea of what the sign should be.
* Every leg layout below is bend-free (``a``'s contracted legs IN and free legs
  OUT, ``b``'s the mirror), so the product provider — which has no
  ``BendingCoefficients`` — runs the identical equations as everyone else.
"""

import numpy as np
import pytest
from helpers import supersign

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import (
    SU2,
    U1,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

# --- spaces -----------------------------------------------------------------------

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 2, ONE: 1})
W = GradedSpace.new(SU2, {ZERO: 1, HALF: 2})
U = GradedSpace.new(SU2, {HALF: 1, ONE: 2})

Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
P = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2})

T3 = GradedSpace.new(Trivial, {TrivialSector(): 3})
T2 = GradedSpace.new(Trivial, {TrivialSector(): 2})

EVEN, ODD = FZ2Sector(0), FZ2Sector(1)
FP = GradedSpace.new(fZ2, {EVEN: 2, ODD: 3})
FQ = GradedSpace.new(fZ2, {EVEN: 1, ODD: 2})
FR = GradedSpace.new(fZ2, {EVEN: 2, ODD: 1})
FS = GradedSpace.new(fZ2, {EVEN: 1, ODD: 1})

UF = ProductProvider((U1, fZ2))


def uf(charge: int, parity: int) -> ProductSector:
    return ProductSector((U1Sector(charge), FZ2Sector(parity)))


S1 = GradedSpace.new(UF, {uf(0, 0): 2, uf(1, 1): 1, uf(-1, 1): 1})
S2 = GradedSpace.new(UF, {uf(0, 0): 1, uf(1, 1): 2})


def spaces(x, y, z, w):
    """``(a legs, b legs, outer a legs, outer b legs)`` in the bend-free layout."""
    return (
        (Leg(x, OUT), Leg(y, OUT), Leg(z, IN)),
        (Leg(z, OUT), Leg(w, IN), Leg(x, IN)),
        (Leg(x, OUT), Leg(y, OUT)),
        (Leg(z, IN), Leg(w, IN)),
    )


PROVIDERS = {
    "trivial": (spaces(T3, T2, T2, T3), False),
    "u1": (spaces(Q, P, P, Q), False),
    "su2": (spaces(V, W, U, W), False),
    "fz2": (spaces(FP, FQ, FR, FS), True),
    "product": (spaces(S1, S2, S2, S1), True),
}

# plain, a within-side braid of A's free labels, an interleave, and an outer product
EQUATIONS = ["abc,cde->abde", "abc,cde->bade", "abc,cde->adbe", "ab,cd->abcd"]


def pair(a_legs, b_legs, seeds=(0, 1)):
    return (
        SymmetricTensor.random(a_legs, seed=seeds[0]),
        SymmetricTensor.random(b_legs, seed=seeds[1]),
    )


def operands(provider_id, equation, seeds=(0, 1)):
    a_legs, b_legs, oa_legs, ob_legs = PROVIDERS[provider_id][0]
    terms, _ = split(equation)
    legs = (a_legs, b_legs) if len(terms[0]) == 3 else (oa_legs, ob_legs)
    return pair(*legs, seeds)


def split(equation):
    lhs, _, rhs = equation.partition("->")
    return lhs.split(","), rhs


def dense_oracle(a, b, equation, graded):
    """``np.einsum`` of the expansions, times the Koszul signs of the two reorders.

    Written as three explicit steps — graded-transpose each operand into
    ``(free..., contracted...)`` / ``(contracted..., free...)`` order, contract,
    graded-transpose into the requested output order — because that is what the
    equation *says*; for a non-graded provider every sign is ``+1`` and this is
    literally ``np.einsum`` (asserted below).
    """
    (ta, tb), out = split(equation)
    shared = [x for x in ta if x in tb]
    fa = [x for x in ta if x not in shared]
    fb = [x for x in tb if x not in shared]

    pa = tuple(ta.index(x) for x in fa + shared)
    pb = tuple(tb.index(x) for x in shared + fb)
    da, db = np.transpose(a.to_dense(), pa), np.transpose(b.to_dense(), pb)
    if graded:
        da = supersign(a.legs, pa, per_side=True) * da
        db = supersign(b.legs, pb, per_side=True) * db

    c = np.tensordot(da, db, axes=len(shared))
    free_legs = tuple(a.legs[ta.index(x)] for x in fa) + tuple(b.legs[tb.index(x)] for x in fb)
    perm = tuple((fa + fb).index(x) for x in out)
    got = np.transpose(c, perm)
    return supersign(free_legs, perm, per_side=True) * got if graded else got


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  registration is the import's side effect

    return jax


def to_numpy(t: SymmetricTensor) -> SymmetricTensor:
    return SymmetricTensor(t.structure, tuple(np.asarray(x) for x in t.blocks))


# --- docs/design.md's own worked example ----------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y", "z"),
    [pytest.param(Q, P, Q, id="u1"), pytest.param(V, W, U, id="su2")],
)
def test_readme_array_style_contraction_example(x, y, z):
    """docs/design.md "Array-style contraction", literally: ``A.legs == (a, b, c)``,
    ``B.legs == (c_dual, d, e)``, ``tenet.einsum("abc,cde->abde", A, B)``."""
    c_leg = Leg(z, OUT)
    a = SymmetricTensor.random((Leg(x, OUT), Leg(y, OUT), c_leg), seed=0)
    b = SymmetricTensor.random((Leg(z, OUT, dual=True), Leg(y, IN), Leg(x, IN)), seed=1)
    assert b.legs[0].space == c_leg.space and b.legs[0].dual is not c_leg.dual

    got = tenet.einsum("abc,cde->abde", a, b)
    assert got.legs == (a.legs[0], a.legs[1], b.legs[1], b.legs[2])
    np.testing.assert_allclose(
        got.to_dense(), np.einsum("abc,cde->abde", a.to_dense(), b.to_dense()), atol=1e-12
    )


# --- dense oracles, every provider × four equations ----------------------------------


@pytest.mark.parametrize("provider_id", list(PROVIDERS))
@pytest.mark.parametrize("equation", EQUATIONS)
def test_dense_oracle(provider_id, equation):
    graded = PROVIDERS[provider_id][1]
    a, b = operands(provider_id, equation, seeds=(3, 4))
    want = dense_oracle(a, b, equation, graded)
    np.testing.assert_allclose(tenet.einsum(equation, a, b).to_dense(), want, atol=1e-10)


@pytest.mark.parametrize("provider_id", [p for p in PROVIDERS if not PROVIDERS[p][1]])
@pytest.mark.parametrize("equation", EQUATIONS)
def test_the_oracle_is_plain_np_einsum_when_nothing_is_graded(provider_id, equation):
    a, b = operands(provider_id, equation, seeds=(3, 4))
    np.testing.assert_allclose(
        dense_oracle(a, b, equation, graded=False),
        np.einsum(equation, a.to_dense(), b.to_dense()),
        atol=1e-12,
    )


def test_fz2_supersign_is_not_identically_one_and_plain_einsum_disagrees():
    """The graded criterion with teeth: a within-side reorder of two odd-carrying
    fZ2 legs, where ``np.einsum`` alone is the wrong answer."""
    equation = "abc,cde->bade"
    a, b = operands("fz2", equation, seeds=(3, 4))
    sign = supersign(a.legs, (1, 0, 2), per_side=True)
    assert np.any(sign != 1.0), "a test whose sign is identically +1 proves nothing"
    got = tenet.einsum(equation, a, b).to_dense()
    plain = np.einsum(equation, a.to_dense(), b.to_dense())
    assert not np.allclose(got, plain, atol=1e-8)
    np.testing.assert_allclose(got, dense_oracle(a, b, equation, graded=True), atol=1e-10)


# --- equivalence with the layer below ------------------------------------------------


HAND_WRITTEN = [
    # (equation, axes, perm) — axes and perm read off the equation by a human
    ("abc,cde->abde", ((2,), (0,)), (0, 1, 2, 3)),
    ("abc,cde->bade", ((2,), (0,)), (1, 0, 2, 3)),
    ("abc,cde->adbe", ((2,), (0,)), (0, 2, 1, 3)),
    ("abc,cde->edba", ((2,), (0,)), (3, 2, 1, 0)),
    ("ab,cd->abcd", ((), ()), (0, 1, 2, 3)),
]


@pytest.mark.parametrize(("equation", "axes", "perm"), HAND_WRITTEN)
@pytest.mark.parametrize("provider_id", ["su2", "fz2"])
def test_einsum_is_tensordot_then_transpose_written_out_by_hand(provider_id, equation, axes, perm):
    a, b = operands(provider_id, equation, seeds=(5, 6))
    want = tenet.transpose(tenet.tensordot(a, b, axes), perm)
    assert tenet.allclose(tenet.einsum(equation, a, b), want, atol=1e-12)


def test_shared_labels_are_ordered_by_first_appearance_in_the_first_operand(monkeypatch):
    """Asserted on the ``axes`` actually handed down, not inferred from the result."""
    recorded = []
    real = tenet.ops.contraction.tensordot

    def recording_tensordot(a, b, axes):
        recorded.append(axes)
        return real(a, b, axes)

    monkeypatch.setattr(tenet.ops.contraction, "tensordot", recording_tensordot)

    a = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN), Leg(U, IN), Leg(W, IN)), seed=0)
    b = SymmetricTensor.random((Leg(U, OUT), Leg(W, OUT), Leg(W, OUT), Leg(V, IN)), seed=1)
    # b lists the shared labels c, b, d in a different order than a does
    for _ in range(3):
        tenet.einsum("abcd,cbde->ae", a, b)
    assert recorded == [((1, 2, 3), (1, 0, 2))] * 3


def test_the_final_transpose_is_categorical_not_a_relabelling():
    """Reordering the output labels by hand and calling ``tensordot`` with pre-permuted
    axes would "save a transpose" and silently return the wrong basis element. For fZ2
    the two differ by a Koszul sign, which is exactly the damage that shortcut does."""
    a, b = operands("fz2", "abc,cde->bade", seeds=(5, 6))
    plain = tenet.einsum("abc,cde->abde", a, b)
    braided = tenet.einsum("abc,cde->bade", a, b)
    assert braided.legs == (plain.legs[1], plain.legs[0], plain.legs[2], plain.legs[3])
    assert not np.allclose(braided.to_dense(), np.transpose(plain.to_dense(), (1, 0, 2, 3)))
    # ...and it is the transpose of the plain result, done categorically
    assert tenet.allclose(braided, tenet.transpose(plain, (1, 0, 2, 3)), atol=1e-12)


# --- one operand ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "legs",
    [
        pytest.param((Leg(V, OUT), Leg(W, OUT), Leg(U, IN)), id="su2"),
        pytest.param((Leg(FP, OUT), Leg(FQ, OUT), Leg(FR, IN)), id="fz2"),
    ],
)
def test_single_operand_is_exactly_a_transpose(legs):
    t = SymmetricTensor.random(legs, seed=2)
    got = tenet.einsum("abc->bca", t)
    want = t.transpose(1, 2, 0)
    assert got.structure == want.structure
    assert got == want
    assert tenet.einsum("abc->abc", t) == t


# --- the implicit output rule ---------------------------------------------------------


def test_implicit_output_is_numpys_alphabetical_rule():
    a, b = operands("su2", "abc,cde->abde", seeds=(0, 1))
    assert tenet.einsum("abc,cde", a, b) == tenet.einsum("abc,cde->abde", a, b)
    t = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT)), seed=2)
    assert tenet.einsum("ba", t) == tenet.einsum("ba->ab", t)
    assert tenet.einsum("ba", t) == t.transpose(1, 0)
    np.testing.assert_allclose(
        tenet.einsum("abc,cde", a, b).to_dense(),
        np.einsum("abc,cde", a.to_dense(), b.to_dense()),
        atol=1e-10,
    )


def test_whitespace_is_ignored_like_numpy():
    a, b = operands("su2", "abc,cde->abde", seeds=(0, 1))
    assert tenet.einsum(" abc , cde -> abde ", a, b) == tenet.einsum("abc,cde->abde", a, b)


# --- refusals -------------------------------------------------------------------------


def su2_operands():
    a, b = operands("su2", "abc,cde->abde", seeds=(0, 1))
    return a, b


REFUSALS = [
    # three or more operands are no longer refused (#67); see test_einsum_multi.py
    pytest.param("abc->abc", 2, ["1 comma-separated term", "2 operand"], id="operand-count"),
    pytest.param("ab,cde->abde", 2, ["term 'ab'", "operand 0 is 3-dimensional"], id="term-length"),
    pytest.param("aac,cde->cde", 2, ["tenet.trace", "repeated inside"], id="trace"),
    pytest.param("aac,cde->acde", 2, ["diagonal is not defined"], id="diagonal"),
    pytest.param("aac,cae->e", 2, ["occurs 3 times"], id="label-thrice"),
    pytest.param("abc,cde->abdez", 2, ["appears in no input term"], id="unknown-output"),
    pytest.param("abc,cde->abdd", 2, ["repeated in the output"], id="repeated-output"),
    pytest.param("abc,cde->abd", 2, ["missing from the output", "not equivariant"], id="summed"),
    pytest.param("a...,cde->abde", 2, ["ellipsis", "do not broadcast"], id="ellipsis"),
    pytest.param("ab1,cde->ab1de", 2, ["is not an ASCII letter"], id="non-alpha"),
    pytest.param("abc,cde->abde", 0, ["no operands"], id="zero-operands"),
]


@pytest.mark.parametrize(("equation", "n", "fragments"), REFUSALS)
def test_loud_refusals(equation, n, fragments):
    a, b = su2_operands()
    args = (a, b, b)[:n]
    with pytest.raises(ValueError) as excinfo:
        tenet.einsum(equation, *args)
    message = str(excinfo.value)
    assert message.startswith("einsum: ")
    for fragment in fragments:
        assert fragment in message, message


def test_the_label_thrice_refusal_outranks_the_repeated_in_term_one():
    """A label on three ends is not a trace and not a diagonal — it is nonsense that
    happens to *look* like one, so the global count is checked first."""
    a, b = su2_operands()
    with pytest.raises(ValueError, match="occurs 3 times"):
        tenet.einsum("abc,cbb->a", a, b)


def test_every_refusal_message_is_distinct():
    a, b = su2_operands()
    messages = set()
    for param in REFUSALS:
        equation, n, _ = param.values
        with pytest.raises(ValueError) as excinfo:
            tenet.einsum(equation, *(a, b, b)[:n])
        messages.add(str(excinfo.value))
    assert len(messages) == len(REFUSALS)


def test_nothing_is_refused_by_the_parser_that_tensordot_should_refuse():
    """Layering, the direction that matters: legs that do not contract fail *below*
    the parser, and the message is ``tensordot``'s, naming public axes."""
    a = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT), Leg(U, IN)), seed=0)
    b = SymmetricTensor.random((Leg(W, OUT), Leg(W, IN), Leg(V, IN)), seed=1)  # axis 0 is not U*
    with pytest.raises(ValueError) as excinfo:
        tenet.einsum("abc,cde->abde", a, b)
    message = str(excinfo.value)
    assert message.startswith("tensordot: ")
    assert "axis 2 of a" in message and "axis 0 of b" in message
    assert "spaces differ" in message


# --- exports and dispatch ---------------------------------------------------------------


def test_einsum_is_exported():
    assert "einsum" in tenet.__all__ and "einsum" in tenet.ops.__all__
    assert tenet.einsum is tenet.ops.einsum


def test_ar_do_einsum_is_the_direct_call():
    import autoray as ar

    a, b = su2_operands()
    assert ar.do("einsum", "abc,cde->abde", a, b) == tenet.einsum("abc,cde->abde", a, b)


# --- jit and grad --------------------------------------------------------------------------


def test_jit_matches_the_numpy_backend():
    jax = use_jax()
    a, b = operands("su2", "abc,cde->bade", seeds=(0, 1))
    want = tenet.einsum("abc,cde->bade", a, b)
    f = jax.jit(lambda x, y: tenet.einsum("abc,cde->bade", x, y))
    got = f(a.to_backend("jax"), b.to_backend("jax"))
    assert got.backend == "jax"
    assert tenet.allclose(to_numpy(got), want, atol=1e-12)


def test_grad_matches_central_differences():
    jax = use_jax()
    a, b = operands("su2", "abc,cde->bade", seeds=(0, 1))
    jb = b.to_backend("jax")

    def loss(t):
        return tenet.norm(tenet.einsum("abc,cde->bade", t, jb))

    grad = jax.grad(loss)(a.to_backend("jax"))
    h = 1e-6
    checked = 0
    for i, block in enumerate(a.blocks):
        if not block.size:
            continue
        for k in (0, block.size - 1):
            shifted = []
            for sign in (+1, -1):
                blocks = [np.array(x, copy=True) for x in a.blocks]
                blocks[i].reshape(-1)[k] += sign * h
                shifted.append(float(loss(SymmetricTensor(a.structure, tuple(blocks)))))
            fd = (shifted[0] - shifted[1]) / (2 * h)
            assert fd == pytest.approx(float(np.asarray(grad.blocks[i]).reshape(-1)[k]), abs=1e-6)
            checked += 1
    assert checked
