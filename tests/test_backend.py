"""Backend boundaries: an operation on two tensors resolves one backend for both.

``src/tenet/backend.py`` exists so that ``ar.do``'s dispatch is paid once per call
rather than once per sector. Hoisting it moved a question the loop used to answer
implicitly: ``ar.do(name, x, y)`` reads the backend off ``x`` alone, and a hoisted
``lib_fn(backend, name)`` has to be told which backend. Resolving it off one operand
is wrong whenever the two are not on the same one, and they need not be --

* a CTMRG corner is a NumPy *constant* times a traced JAX tensor, which is the case
  that first showed up, in ``compose``;
* the torch half is stricter: ``torch.matmul``/``torch.add`` refuse an ``ndarray``
  outright, so dispatching to torch is not enough on its own and the NumPy operand
  has to be converted. That is why ``promote`` exists rather than a bare
  ``ar.infer_backend_multi``.

Every case below fails on the pre-``promote`` code. The mixed pairs are asserted
against the same-backend answer with ``np.array_equal`` and not ``allclose``:
promotion must not be a numerical event.

``direct_sum`` is deliberately absent -- it *refuses* two backends, with a message
naming ``to_backend``, because a key only one operand contributes to would never
meet the other's backend and the result would carry blocks of both. That refusal is
pinned in ``tests/ops/test_embed.py``; it is the one place where promoting would be
the bug.
"""

import jax
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.backend import promote
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 2})
W = GradedSpace.new(SU2, {HALF: 2, ONE: 1})
U = GradedSpace.new(SU2, {ZERO: 1, HALF: 2})

A_LEGS = (Leg(V, OUT), Leg(W, IN))
B_LEGS = (Leg(W, OUT), Leg(U, IN))


def use_jax():
    jax.config.update("jax_enable_x64", True)  # as in tests/ops/test_map.py
    return jax


# ``fn`` takes two tensors; ``same`` says whether they need the same structure.
BINARY = {
    "compose": (False, lambda a, b: a @ b),
    "tensordot": (False, lambda a, b: tenet.tensordot(a, b, axes=([1], [0]))),
    "einsum": (False, lambda a, b: tenet.einsum("ij,jk->ik", a, b)),
    "add": (True, tenet.add),
    "subtract": (True, tenet.subtract),
    "inner": (True, tenet.inner),
    "zip_blocks": (True, lambda a, b: tenet.zip_blocks(a, b, lambda x, y: x * y)),
    "allclose": (True, lambda a, b: tenet.allclose(a, b)),
}


def operands(same_structure):
    return (
        SymmetricTensor.random(A_LEGS, seed=0),
        SymmetricTensor.random(A_LEGS if same_structure else B_LEGS, seed=1),
    )


def values(result):
    """Every number a result carries, as NumPy, whatever it is."""
    if isinstance(result, SymmetricTensor):
        return [np.asarray(m) for m in result.data]
    return [np.asarray(result)]


# --- promote itself -------------------------------------------------------------


def test_promote_returns_one_backend_and_leaves_a_matched_pair_untouched():
    """The same-backend path is every call that is not a mixed one; it must allocate."""
    a = tuple(np.ones((2, 2)) for _ in range(3))
    b = tuple(np.zeros((2, 2)) for _ in range(3))
    backend, xs, ys = promote(a, b)
    assert backend == "numpy"
    assert xs is a and ys is b


def test_promote_moves_only_the_odd_side():
    use_jax()
    a = tuple(np.ones((2, 2)) for _ in range(2))
    b = tuple(jax.numpy.zeros((2, 2)) for _ in range(2))
    backend, xs, ys = promote(a, b)
    assert backend == "jax"
    assert ys is b  # already there
    assert [np.asarray(x).tolist() for x in xs] == [np.ones((2, 2)).tolist()] * 2


def test_promote_on_a_blockless_operand_has_nothing_to_dispatch():
    """A tensor whose legs cannot couple carries no array, so there is nothing to move."""
    other = (np.ones(2),)
    assert promote((), other) == ("numpy", (), other)
    assert promote(other, ()) == ("numpy", other, ())


# --- the operations -------------------------------------------------------------


@pytest.mark.parametrize("op", sorted(BINARY))
@pytest.mark.parametrize("other", ["jax", "torch"])
@pytest.mark.parametrize("order", ["numpy first", "numpy second"])
def test_a_numpy_operand_meeting_a_foreign_one_dispatches_to_the_foreign_backend(op, other, order):
    """``lib_fn`` resolved off one side sends the other side's arrays to the wrong library.

    On JAX that is a ``TracerArrayConversionError`` under ``jit`` (and a silent
    NumPy result outside one); on torch it is a ``TypeError`` from ``torch.matmul``
    and friends, which refuse an ``ndarray`` even when it is the *second* argument.
    """
    pytest.importorskip(other)
    use_jax()
    same, fn = BINARY[op]
    a, b = operands(same)
    mixed = (a, b.to_backend(other)) if order == "numpy first" else (a.to_backend(other), b)

    # the oracle is ``to_backend`` on *both* operands, not the NumPy answer: promoting
    # must land exactly where converting by hand would, and the foreign library's own
    # reductions need not agree with NumPy's to the last bit (they do not, for ``inner``)
    got = fn(*mixed)
    want = fn(a.to_backend(other), b.to_backend(other))
    if isinstance(got, SymmetricTensor):
        assert got.backend == other
    for x, y in zip(values(got), values(want), strict=True):
        assert np.array_equal(x, y)


def test_a_numpy_constant_composed_with_a_traced_tensor_stays_traced():
    """The CTMRG corner, spelled as a trace: a constant times a traced tensor.

    Outside ``jit`` the NumPy operand is silently absorbed by JAX's ufunc protocol,
    so only a trace makes the wrong dispatch observable.
    """
    use_jax()
    a, b = operands(same_structure=False)
    jb = b.to_backend("jax")

    def f(blocks):
        return (a @ SymmetricTensor(jb.structure, tuple(blocks))).blocks

    got = jax.jit(f)(jb.blocks)
    for x, y in zip(got, (a @ b).blocks, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# --- the zeros a composition invents --------------------------------------------


def _disjoint_pair():
    """Two composable U(1) maps that share no coupled sector, whose product has one.

    ``a``'s sectors are ``P ∩ R``, ``b``'s are ``R ∩ S`` and the composition's are
    ``P ∩ S``; with ``P, R, S = {0,1}, {1,2}, {0,2}`` those are ``{1}``, ``{2}`` and
    ``{0}``, so no product runs and every sector of the result is a zero block.
    """

    def sp(*charges):
        return GradedSpace.new(U1, {U1Sector(c): 2 for c in charges})

    P, R, S = sp(0, 1), sp(1, 2), sp(0, 2)
    return (
        SymmetricTensor.random((Leg(P, OUT), Leg(R, IN)), seed=0),
        SymmetricTensor.random((Leg(R, OUT), Leg(S, IN)), seed=1),
    )


@pytest.mark.parametrize("other", ["jax", "torch"])
@pytest.mark.parametrize("order", ["numpy first", "numpy second"])
def test_the_zeros_of_a_productless_composition_take_the_pair_s_backend(other, order):
    """No product decides the backend, so the fallback must not be argument order.

    ``block_ref`` answers with the *first* operand that carries a block, which made
    ``numpy_tensor @ jax_tensor`` come back NumPy while ``jax_tensor @ numpy_tensor``
    came back JAX -- the same pair, two answers.
    """
    pytest.importorskip(other)
    use_jax()
    a, b = _disjoint_pair()
    x, y = (a, b.to_backend(other)) if order == "numpy first" else (a.to_backend(other), b)

    got = x @ y
    assert got.backend == other
    for m in got.data:
        assert not np.asarray(m).any()
