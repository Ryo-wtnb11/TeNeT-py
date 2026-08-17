"""Linear arithmetic, conjugation and the Frobenius norm — Milestone 2.

Everything here is linear in the blocks, so no categorical machinery is needed:
blocks are combined index-aligned, because ``block_order`` is a pure function of
the (frozen) structure and equal structures therefore give equal key tuples in
equal order.

Two rules the rest of Milestone 2 leans on:

* mismatched structures fail loudly — addition never widens a graded space
  (invariant 11; a ``pad_to``/``union`` helper can arrive later as an explicit
  call);
* :func:`norm` carries the quantum-dimension weight, which is what makes it
  agree with ``‖to_dense()‖_F``.

There is deliberately no NumPy call anywhere in this module: arithmetic must
already work on JAX (or torch) blocks, so every array touch goes through
``autoray``. ``tests/backends/test_torch.py`` is what enforces the torch half of
that (#95) — it runs every function here on torch blocks and compares against
the NumPy result.
"""

import numbers
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.symmetry.base import QuantumDimension, requires

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
    """``a + b``. Requires *identical* structures; near-misses are errors."""
    from tenet.tensor import SymmetricTensor

    _check_same_structure(a, b, "add")
    return SymmetricTensor(
        a.structure, tuple(ar.do("add", x, y) for x, y in zip(a.blocks, b.blocks, strict=True))
    )


def subtract(a: "SymmetricTensor", b: "SymmetricTensor") -> "SymmetricTensor":
    """``a - b``. Same structure rule as :func:`add`."""
    from tenet.tensor import SymmetricTensor

    _check_same_structure(a, b, "subtract")
    return SymmetricTensor(
        a.structure, tuple(ar.do("subtract", x, y) for x, y in zip(a.blocks, b.blocks, strict=True))
    )


def multiply(t: "SymmetricTensor", s: Any) -> "SymmetricTensor":
    """``s * t`` for a scalar ``s`` — the only defined multiplication here."""
    from tenet.tensor import SymmetricTensor

    s = _scalar(s, t, "multiply")
    return SymmetricTensor(t.structure, tuple(b * s for b in t.blocks))


def divide(t: "SymmetricTensor", s: Any) -> "SymmetricTensor":
    """``t / s`` for a scalar ``s``."""
    from tenet.tensor import SymmetricTensor

    s = _scalar(s, t, "divide")
    return SymmetricTensor(t.structure, tuple(b / s for b in t.blocks))


def negative(t: "SymmetricTensor") -> "SymmetricTensor":
    """``-t``."""
    from tenet.tensor import SymmetricTensor

    return SymmetricTensor(t.structure, tuple(-b for b in t.blocks))


def conj(t: "SymmetricTensor") -> "SymmetricTensor":
    """Complex-conjugate the reduced blocks; ``legs`` are left completely alone.

    No ``dual`` flip, no side change: conjugation, duality and the categorical
    adjoint are three different things (invariant 2). In particular ``t.conj()``
    is **not** what you contract ``t`` against to obtain ``‖t‖²`` — that pairing
    needs the adjoint (Milestone 3).

    Blockwise conjugation equals dense-basis conjugation exactly because every
    provider here has real Clebsch-Gordan coefficients (all-ones for Trivial and
    U(1), Condon-Shortley for SU(2)).
    """
    # Simplification: a provider with complex CG would need a capability gate here — one
    # ``requires(...)`` line once such a provider exists. No marker protocol with
    # three implementations and no counterexample.
    from tenet.tensor import SymmetricTensor

    return SymmetricTensor(t.structure, tuple(ar.do("conj", b) for b in t.blocks))


def norm(t: "SymmetricTensor") -> Any:
    """``sqrt(Σ_τ qdim(c_τ) · ‖A_τ‖²)`` — the fusion-tree Frobenius norm.

    The quantum-dimension weight is the point: each key contributes ``‖A_τ‖²``
    once per coupled basis state and there are ``qdim(c)`` of them, so this
    equals the dense Frobenius norm of ``t.to_dense()`` while never densifying
    (that identity is an acceptance test). Dropping the
    weight is wrong for any non-Abelian provider.

    Uses ``qdim`` (capability ``QuantumDimension``), not ``irrep_dim``, so it is
    defined even for providers with no dense expansion at all.

    Returns the backend's own scalar, so the whole function is traceable and
    differentiable (Milestone 6).
    """
    provider = t.provider
    requires(provider, QuantumDimension)
    if not t.blocks:
        return 0.0
    total = sum(
        # requires() above; raise-based check does not narrow
        provider.qdim(key.coupled)  # ty: ignore[unresolved-attribute]
        * ar.do("sum", ar.do("abs", block) ** 2)
        for key, block in t.items()
    )
    # No float(): concretizing here makes `norm` unusable under jit/grad/vmap.
    # NumPy blocks give a float64 scalar (float-compatible); JAX blocks give a
    # traceable 0-d array. Callers needing a Python float say float(tenet.norm(T)).
    return ar.do("sqrt", total)


def allclose(
    a: "SymmetricTensor", b: "SymmetricTensor", *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Tolerant comparison. Different structures give ``False``, never an error."""
    if a.structure != b.structure:
        return False
    return all(
        bool(ar.do("allclose", x, y, rtol=rtol, atol=atol))
        for x, y in zip(a.blocks, b.blocks, strict=True)
    )
