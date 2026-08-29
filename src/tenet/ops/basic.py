"""Linear arithmetic, conjugation and the Frobenius norm.

Everything here is linear in the blocks, so no categorical machinery is needed:
blocks are combined index-aligned, because ``block_order`` is a pure function of
the (frozen) structure and equal structures therefore give equal key tuples in
equal order.

Two rules the rest of the package leans on:

* mismatched structures fail loudly — addition never widens a graded space
  (invariant 11; a ``pad_to``/``union`` helper can arrive later as an explicit
  call);
* [norm][tenet.norm] carries the quantum-dimension weight, which is what makes it
  agree with ``‖to_dense()‖_F``.

There is deliberately no NumPy call anywhere in this module: arithmetic must
already work on JAX (or torch) blocks, so every array touch goes through
``autoray``. ``tests/backends/test_torch.py`` is what enforces the torch half of
that — it runs every function here on torch blocks and compares against the
NumPy result.
"""

import numbers
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.map_view import map_layout
from tenet.symmetry.base import QuantumDimensionData, requires

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "add",
    "allclose",
    "conj",
    "divide",
    "multiply",
    "negative",
    "norm",
    "subtract",
]


def _check_same_structure(a: "SymmetricTensor", b: "SymmetricTensor", op: str) -> None:
    """Raise ``ValueError`` naming the first differing axis and both legs.

    ``a.structure == b.structure`` is the whole check (leg equality *is*
    structure equality); the walk below only exists so the message says which
    axis differs, because "structures differ" is useless in a stack trace.
    """
    if a.structure == b.structure:
        return
    if a.provider != b.provider:
        raise ValueError(
            f"{op}: tensors come from different providers, {a.provider.name} and {b.provider.name}"
        )
    if a.ndim != b.ndim:
        raise ValueError(f"{op}: tensors have different ndim, {a.ndim} and {b.ndim}")
    for i, (x, y) in enumerate(zip(a.legs, b.legs, strict=True)):
        if x != y:
            raise ValueError(f"{op}: legs differ at public axis {i}: {x!r} vs {y!r}")
    raise ValueError(f"{op}: structures differ: {a.structure!r} vs {b.structure!r}")


def _scalar(s: Any, t: "SymmetricTensor", op: str) -> Any:
    """Return ``s`` if it is a scalar for ``t``, else raise ``TypeError``."""
    from tenet.tensor import SymmetricTensor

    if isinstance(s, SymmetricTensor):
        raise TypeError(
            f"{op}: elementwise products of two SymmetricTensors are not a defined "
            "categorical operation; use tenet.tensordot(a, b, axes=...) for a contraction, "
            "or a @ b for morphism composition. `*` is scalar multiplication only"
        )
    if isinstance(s, numbers.Number) or getattr(s, "ndim", None) == 0:
        return s
    raise TypeError(f"{op}: expected a scalar, got {type(s).__name__}")


def add(a: "SymmetricTensor", b: "SymmetricTensor") -> "SymmetricTensor":
    """``a + b``. Requires *identical* structures; near-misses are errors.

    Parameters
    ----------
    a : SymmetricTensor
        The left operand.
    b : SymmetricTensor
        The right operand. Its structure must equal ``a``'s exactly — same
        provider, same legs (``space``, ``side``, ``dual``) in the same order.

    Returns
    -------
    SymmetricTensor
        The blockwise sum, on the operands' shared structure.

    Raises
    ------
    ValueError
        If the structures differ — different providers, different ``ndim``, or a
        differing leg; the message names the first differing axis and both legs.
        Addition never widens a graded space (invariant 11).

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> b = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)
    >>> bool(tenet.allclose(tenet.add(a, b), a + b))
    True
    """
    from tenet.tensor import SymmetricTensor

    _check_same_structure(a, b, "add")
    # over the coupled-sector matrices, not the blocks: the two tensors share a structure,
    # so they share a layout, and every cell of a matrix belongs to exactly one block.
    # One backend call per sector rather than per block, and neither operand is cut.
    return SymmetricTensor.from_data(
        a.structure, tuple(ar.do("add", x, y) for x, y in zip(a.data, b.data, strict=True))
    )


def subtract(a: "SymmetricTensor", b: "SymmetricTensor") -> "SymmetricTensor":
    """``a - b``. Same structure rule as [add][tenet.add].

    Parameters
    ----------
    a : SymmetricTensor
        The left operand.
    b : SymmetricTensor
        The right operand; its structure must equal ``a``'s exactly.

    Returns
    -------
    SymmetricTensor
        The blockwise difference, on the operands' shared structure.

    Raises
    ------
    ValueError
        If the structures differ, exactly as [add][tenet.add] refuses.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> round(float(tenet.norm(tenet.subtract(a, a))), 6)
    0.0
    """
    from tenet.tensor import SymmetricTensor

    _check_same_structure(a, b, "subtract")
    return SymmetricTensor.from_data(
        a.structure, tuple(ar.do("subtract", x, y) for x, y in zip(a.data, b.data, strict=True))
    )


def multiply(t: "SymmetricTensor", s: Any) -> "SymmetricTensor":
    """``s * t`` for a scalar ``s`` — the only defined multiplication here.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor to scale.
    s : scalar
        A Python number or a 0-d backend array. Never a ``SymmetricTensor``:
        elementwise products of two tensors are not a defined categorical
        operation.

    Returns
    -------
    SymmetricTensor
        ``t`` with every block multiplied by ``s``, on ``t``'s structure.

    Raises
    ------
    TypeError
        If ``s`` is a ``SymmetricTensor`` (use [tensordot][tenet.tensordot] for a
        contraction, or ``a @ b`` for morphism composition), or if ``s`` is
        neither a number nor a 0-d array.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> bool(tenet.allclose(tenet.multiply(a, 2.0), a + a))
    True
    """
    from tenet.tensor import SymmetricTensor

    s = _scalar(s, t, "multiply")
    # ``_scalar`` has already refused anything that could broadcast a block to a new
    # shape, so ``t``'s validated shapes carry over unchanged (#328)
    return SymmetricTensor.from_data(t.structure, tuple(m * s for m in t.data))


def divide(t: "SymmetricTensor", s: Any) -> "SymmetricTensor":
    """``t / s`` for a scalar ``s``.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor to scale.
    s : scalar
        A Python number or a 0-d backend array; the same scalar rule as
        [multiply][tenet.multiply].

    Returns
    -------
    SymmetricTensor
        ``t`` with every block divided by ``s``, on ``t``'s structure.

    Raises
    ------
    TypeError
        If ``s`` is a ``SymmetricTensor``, or neither a number nor a 0-d array.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> bool(tenet.allclose(tenet.divide(a, 2.0), 0.5 * a))
    True
    """
    from tenet.tensor import SymmetricTensor

    s = _scalar(s, t, "divide")
    return SymmetricTensor.from_data(t.structure, tuple(m / s for m in t.data))


def negative(t: "SymmetricTensor") -> "SymmetricTensor":
    """``-t``.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor to negate.

    Returns
    -------
    SymmetricTensor
        ``t`` with every block negated, on ``t``'s structure.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> round(float(tenet.norm(tenet.negative(a) + a)), 6)
    0.0
    """
    from tenet.tensor import SymmetricTensor

    return SymmetricTensor.from_data(t.structure, tuple(-m for m in t.data))


def conj(t: "SymmetricTensor") -> "SymmetricTensor":
    """Complex-conjugate the reduced blocks; ``legs`` are left completely alone.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor to conjugate.

    Returns
    -------
    SymmetricTensor
        ``t`` with every block conjugated, on ``t``'s unchanged structure.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)  # real blocks
    >>> bool(tenet.allclose(tenet.conj(a), a))
    True

    Notes
    -----
    No ``dual`` flip, no side change: conjugation, duality and the categorical
    adjoint are three different things (invariant 2). In particular ``t.conj()``
    is **not** what you contract ``t`` against to obtain ``‖t‖²`` — that pairing
    needs the adjoint, [tenet.adjoint][].

    Blockwise conjugation equals dense-basis conjugation exactly because every
    provider here has real Clebsch-Gordan coefficients (all-ones for Trivial and
    U(1), Condon-Shortley for SU(2)).
    """
    # Simplification: a provider with complex CG would need a capability gate here — one
    # ``requires(provider, DaggerData)`` line once DaggerData grows content (M24a made it
    # the named marker for exactly this gap; it stays contentless until a counterexample
    # provider exists).
    from tenet.tensor import SymmetricTensor

    return SymmetricTensor.from_data(t.structure, tuple(ar.do("conj", m) for m in t.data))


def norm(t: "SymmetricTensor") -> Any:
    """``sqrt(Σ_τ qdim(c_τ) · ‖A_τ‖²)`` — the fusion-tree Frobenius norm.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose norm is taken.

    Returns
    -------
    scalar
        The backend's own scalar (a float64 scalar on NumPy, a traceable 0-d
        array on JAX), never a Python ``float``. Callers needing one say
        ``float(tenet.norm(t))``.

    Raises
    ------
    CapabilityError
        If ``t``'s provider does not implement
        [QuantumDimensionData][tenet.symmetry.QuantumDimensionData].

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> round(float(tenet.norm(a)), 6)
    0.182373

    Notes
    -----
    The quantum-dimension weight is the point: each key contributes ``‖A_τ‖²``
    once per coupled basis state and there are ``qdim(c)`` of them, so this
    equals the dense Frobenius norm of ``t.to_dense()`` while never densifying
    (that identity is an acceptance test). Dropping the
    weight is wrong for any non-Abelian provider.

    Uses ``qdim`` (capability ``QuantumDimensionData``), not ``irrep_dim``, so it is
    defined even for providers with no dense expansion at all.

    Returns the backend's own scalar, so the whole function is traceable and
    differentiable.

    **Summed per coupled sector, not per fusion tree.** The weight depends on the
    coupled sector alone and a sector's matrix is exactly its blocks laid side by side --
    the grid is complete, so every cell is written once and none is left zero -- which
    makes ``Σ_τ qdim(c_τ)·‖A_τ‖²`` and ``Σ_c qdim(c)·‖B_c‖²`` the same sum over the same
    numbers, the identity the map view's own conventions are stated in. Reading blocks
    to spell it the first way would cut every block of the tensor out of the matrices
    the reduction is about to run over anyway.
    """
    provider = t.provider
    requires(provider, QuantumDimensionData)
    if not t.data:
        return 0.0
    total = sum(
        # requires() above; raise-based check does not narrow
        provider.qdim(c)  # ty: ignore[unresolved-attribute]
        * ar.do("sum", ar.do("abs", mat) ** 2)
        for c, mat in zip(map_layout(t.structure).sectors, t.data, strict=True)
    )
    # No float(): concretizing here makes `norm` unusable under jit/grad/vmap.
    # NumPy blocks give a float64 scalar (float-compatible); JAX blocks give a
    # traceable 0-d array. Callers needing a Python float say float(tenet.norm(T)).
    return ar.do("sqrt", total)


def allclose(
    a: "SymmetricTensor", b: "SymmetricTensor", *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Tolerant comparison. Different structures give ``False``, never an error.

    Parameters
    ----------
    a : SymmetricTensor
        The left operand.
    b : SymmetricTensor
        The right operand; a structure mismatch is ``False``, not an error.
    rtol : float, optional
        Relative tolerance, forwarded to the backend's ``allclose``.
        Default ``1e-5``.
    atol : float, optional
        Absolute tolerance, forwarded likewise. Default ``1e-8``.

    Returns
    -------
    bool
        ``True`` iff the structures are equal and every pair of blocks is
        close under ``(rtol, atol)``.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> b = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)
    >>> tenet.allclose(a, a), tenet.allclose(a, b)
    (True, False)
    """
    if a.structure != b.structure:
        return False
    return all(
        bool(ar.do("allclose", x, y, rtol=rtol, atol=atol))
        for x, y in zip(a.blocks, b.blocks, strict=True)
    )
