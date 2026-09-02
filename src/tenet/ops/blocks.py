"""Elementwise maps over the reduced blocks.

**Coefficient space, not dense space.** This is the one module in ``ops`` that is
*non-linear* in the blocks, and non-linearity is exactly what does not commute
with dense expansion: ``T = Σ_τ A^(τ) ⊗ C^(τ)`` is a sum of tensor products, so
``apply_blocks(t, f).to_dense()`` and ``f(t.to_dense())`` are two different
operations whenever the Clebsch-Gordan factor is not all-ones. On a rank-3 tensor
with positive blocks the two agree exactly for U(1) and differ by ``1.673`` on a
dense scale of ``3.82`` for SU(2).

That difference is why nothing here is registered in ``array/dispatch.py``:
``ar.do("sqrt", t)`` must keep raising, because autoray's ``"sqrt"`` means the
dense elementwise one and this is not it.

It is also why the functions are ``block_sqrt`` / ``block_power`` rather than
YASTN's and symmray's bare ``sqrt``. Those two are Abelian (and
fermionic-Abelian), where a blockwise map and a dense elementwise map are the
same numbers in a different order, so one name serves both; the ``1.673`` above
is the non-Abelian case where it stops serving. The precedent is ``reshape``:
when a numpy name's meaning does not exist for a symmetric tensor, this package
refuses the numpy name and spells its own operation differently (``fuse`` /
``unfuse`` there, ``block_*`` here) rather than rebinding it. TensorKit's
``sqrt`` is a third meaning again -- the matrix square root of a positive
tensor map -- so no reference lends the bare name to this operation.

As in ``tenet.ops.basic``, there is no NumPy call anywhere in this module.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.backend import promote
from tenet.ops.basic import _check_same_structure

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["apply_blocks", "block_power", "block_sqrt", "zip_blocks"]

Array = Any


def apply_blocks(t: "SymmetricTensor", fn: Callable[[Array], Array]) -> "SymmetricTensor":
    """``fn`` applied to each reduced block. **Coefficient space, not dense space.**

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose reduced blocks ``fn`` maps over.
    fn : callable
        An **elementwise and shape-preserving** function of one backend array.
        It is not checked (a shape change is caught by
        ``SymmetricTensor.__post_init__``, a value-dependent one is the caller's
        problem), and it is not required that ``fn(0) == 0``.

    Returns
    -------
    SymmetricTensor
        ``fn`` of every block, on ``t``'s unchanged structure.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> tenet.allclose(tenet.apply_blocks(a, lambda blk: 2 * blk), a + a)
    True

    Notes
    -----
    Structure is untouched, so this is linear-algebra-free, backend-generic,
    traceable and differentiable: it is exactly ``t.set_params(map(fn,
    t.get_params()))``, which is quimb's ``Tensor.apply_to_arrays`` over the same
    parameter protocol.

    ``fn`` not being required to satisfy ``fn(0) == 0`` is a structural fact: the
    blocks hold only allowed fusion channels, so *any* ``fn`` returns a valid
    symmetric tensor and every symmetry-forbidden dense entry stays exactly zero.

    What it does **not** do is commute with dense expansion. ``T = Σ_τ A^(τ) ⊗
    C^(τ)``, so for a non-Abelian provider ``apply_blocks(t, f).to_dense() !=
    f(t.to_dense())`` — off by ``1.673`` on a dense scale of ``3.82`` for a rank-3
    SU(2) tensor with ``f = sqrt``. For every shipped Abelian provider
    (all-ones CG, ``d_a == 1``) they agree exactly. If you want dense-elementwise
    semantics, densify explicitly.
    """
    from tenet.tensor import SymmetricTensor

    return SymmetricTensor(t.structure, tuple(fn(b) for b in t.blocks))


def _over_data(t: "SymmetricTensor", fn: Callable[[Array], Array]) -> "SymmetricTensor":
    """[apply_blocks][tenet.apply_blocks] for an ``fn`` this module supplies itself.

    Parameters
    ----------
    t : SymmetricTensor
        The operand; its structure is the result's.
    fn : callable
        A **shape-agnostic** elementwise function -- one of this module's own, never a
        caller's.

    Returns
    -------
    SymmetricTensor
        ``fn`` of every coupled-sector matrix, on ``t``'s unchanged structure.

    Notes
    -----
    A coupled sector's matrix is exactly its blocks laid side by side: the
    ``(output tree, input tree)`` grid is complete, so every cell is written once and
    none is left zero (``tenet.map_view``'s module docstring). An elementwise map is
    therefore the *same numbers* whether it runs over the matrices or over the blocks
    cut out of them, and running it over the matrices leaves the result holding storage
    rather than views (invariant 8).

    ``apply_blocks`` and ``zip_blocks`` keep the block route because there the ``fn`` is
    the caller's. Their contract says elementwise, but nothing checks it, and an ``fn``
    that reads ``b.shape`` -- a per-block normalization, a broadcast against an axis --
    would see the matrix instead of the block and silently compute something else. Here
    ``fn`` is ``sqrt`` or ``power``, whose value at an entry depends on that entry only,
    so there is nothing for the shape to change.
    """
    from tenet.tensor import SymmetricTensor

    return SymmetricTensor.from_data(t.structure, tuple(fn(m) for m in t.data))


def block_sqrt(t: "SymmetricTensor") -> "SymmetricTensor":
    """Blockwise ``sqrt``. The ``svd`` splitter: ``u @ sqrt(s)``, ``sqrt(s) @ vh``.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose blocks are square-rooted, entry by entry.

    Returns
    -------
    SymmetricTensor
        The elementwise square root of every block, on ``t``'s structure.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> q = tenet.apply_blocks(a, abs)  # non-negative blocks
    >>> tenet.allclose(tenet.block_sqrt(tenet.block_power(q, 2)), q)
    True

    Notes
    -----
    Blockwise, i.e. on the coefficients — see [apply_blocks][tenet.apply_blocks]
    for the blockwise/dense caveat, which for SU(2) is 44% of the array's own
    scale and completely silent.

    For the ``S`` returned by [tenet.linalg.svd][tenet.ops.linalg.svd] this *is* the matrix square
    root, because ``svd`` builds ``S``'s blocks with ``ar.do("diag", ...)`` and a
    diagonal matrix's elementwise and matrix square roots coincide. That is a fact
    about ``S``, not about ``sqrt``: for a non-diagonal ``t``, ``sqrt(t) @ sqrt(t)
    != t``.

    Negative or complex entries are the backend's business: ``sqrt(-1.0)`` is
    ``nan`` under NumPy, JAX and torch alike, and nothing here clips, guards or
    regularizes.

    Runs over the coupled-sector matrices, not over the blocks cut out of them: a
    coupled sector's matrix is exactly its blocks laid side by side, so an elementwise
    map is the same numbers either way, and this one's ``fn`` is the library's own and
    reads no shape. ``apply_blocks`` keeps the block route because there it is not.
    """
    return _over_data(t, lambda m: ar.do("sqrt", m))


def block_power(t: "SymmetricTensor", p: Any) -> "SymmetricTensor":
    """Blockwise ``t ** p``. ``p`` is a scalar exponent, never a tensor.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose blocks are raised to ``p``, entry by entry.
    p : scalar
        The exponent — a Python number or 0-d backend array, never a tensor.

    Returns
    -------
    SymmetricTensor
        Every block raised to ``p``, on ``t``'s structure.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> tenet.allclose(tenet.block_power(a, 2), tenet.apply_blocks(a, lambda b: b * b))
    True

    Notes
    -----
    Same coefficient-space semantics and the same backend-owned branch cuts and
    ``nan``s as [block_sqrt][tenet.block_sqrt]; ``p = -0.5`` is the inverse-√S of
    canonical-form and gauge-fixing loops. It runs over ``data`` for the same reason.
    """
    return _over_data(t, lambda m: ar.do("power", m, p))


def zip_blocks(
    a: "SymmetricTensor", b: "SymmetricTensor", fn: Callable[[Array, Array], Array]
) -> "SymmetricTensor":
    """``fn`` over the aligned block pairs of two tensors sharing one structure.

    The two-argument sibling of [apply_blocks][tenet.apply_blocks], and
    **coefficient space, not dense space** in exactly the same sense.

    Parameters
    ----------
    a : SymmetricTensor
        The left operand; its structure is the result's.
    b : SymmetricTensor
        The right operand. Its structure must equal ``a``'s exactly — same
        provider, same legs (``space``, ``side``, ``dual``) in the same order —
        so that ``block_order`` pairs the blocks index for index.
    fn : callable
        An **elementwise and shape-preserving** function of two backend arrays of
        equal shape. It is not checked (a shape change is caught by
        ``SymmetricTensor.__post_init__``), and it is not required that
        ``fn(0, 0) == 0``.

    Returns
    -------
    SymmetricTensor
        ``fn`` of every aligned block pair, on the operands' shared structure.

    Raises
    ------
    ValueError
        If the structures differ — different providers, different ``ndim``, or a
        differing leg; the message names the first differing axis and both legs,
        in [add][tenet.add]'s style. Nothing is widened and nothing is aligned by
        sector label (invariant 11).

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> legs = (Leg(V, OUT), Leg(V, IN))
    >>> q = SymmetricTensor.random(legs, seed=0)
    >>> d = SymmetricTensor.random(legs, seed=1)
    >>> p = tenet.zip_blocks(q, d, lambda x, y: x / (2.5 - y))  # a Jacobi step
    >>> p.structure == q.structure
    True

    Notes
    -----
    **Why this does not reopen [multiply][tenet.multiply]'s refusal.** ``multiply``
    refuses a second ``SymmetricTensor`` because ``a * b`` *asks for a categorical
    operation and there is none*: a tensor is ``Σ_τ A^(τ) ⊗ C^(τ)``, so an
    entrywise dense product has no expression in the reduced blocks, and the
    plausible-looking blockwise answer would be a silently different tensor —
    ``multiply`` is defined by dense semantics and must keep them. This function
    makes the opposite declaration in its name and signature: it is a map over
    *coefficients*, the caller supplies the map, and no claim is made that it
    commutes with [to_dense][tenet.SymmetricTensor.to_dense]. It cannot be reached by
    an operator (``*``, ``/``) and cannot be reached by accident, and requiring
    one shared structure is what makes "the aligned block pair" mean something:
    ``block_order`` is a pure function of the structure, so equal structures give
    equal key tuples in equal order. The same argument already licenses the unary
    ``apply_blocks``; the arity is not what was ever in question.

    The consumer this exists for is the Jacobi preconditioner of a Davidson step,
    ``q / (lambda - diag)`` over the reduced storage a solver iterates on, with
    ``diag`` from [map_diagonal][tenet.map_diagonal]. That quotient is a
    coefficient-space statement about the solver's own vector, not a statement
    about the dense tensor, which is precisely the distinction ``multiply``'s
    refusal draws.

    Structure is untouched, so this is linear-algebra-free, backend-generic,
    traceable and differentiable, exactly as ``apply_blocks`` is.
    """
    from tenet.tensor import SymmetricTensor

    _check_same_structure(a, b, "zip_blocks")
    # ``fn`` is handed two arrays of one backend even when the operands arrive on two:
    # a NumPy block and a torch one have no ``+`` between them, and which backend the
    # pair lands on is not ``fn``'s question to answer (``ops.basic.add`` promotes for
    # the same reason).
    _, xs, ys = promote(a.blocks, b.blocks)
    return SymmetricTensor(a.structure, tuple(fn(x, y) for x, y in zip(xs, ys, strict=True)))
