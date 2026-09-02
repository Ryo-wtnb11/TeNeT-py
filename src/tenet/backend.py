"""Resolving ``autoray``'s dispatch once instead of once per block.

``ar.do(name, x, ...)`` infers the backend from the argument, looks the function up
and calls it, every time. That is the right thing at the edge of an operation and the
wrong thing inside a loop over blocks: a symmetric tensor has as many blocks as it has
fusion trees, an SU(2) rank-6 tensor with four sectors has 9,145 of them, and the
arrays being moved are dozens of elements each. Measured on NumPy, ``ar.do("reshape",
...)`` is 471 ns against 294 ns for the resolved function -- and the reshape itself is
306 ns, so the dispatch is most of what is left.

So hoist the lookup out of the loop and call the backend's own function inside it,
which is what ``symmray`` does with ``ar.get_lib_fn`` in its own block loops. The
result is identical: this resolves the same function ``ar.do`` would have reached.
"""

from collections.abc import Sequence
from functools import cache
from typing import Any

import autoray as ar

__all__ = ["lib_fn", "promote"]


@cache
def lib_fn(backend: str, name: str) -> Any:
    """``backend``'s implementation of ``name``, resolved once and cached.

    Parameters
    ----------
    backend : str
        An autoray backend name, as [SymmetricTensor.backend][tenet.SymmetricTensor.backend]
        reports it.
    name : str
        The operation, spelled as ``ar.do`` spells it.

    Returns
    -------
    callable
        The function ``ar.do(name, ...)`` would dispatch to on that backend.
    """
    return ar.get_lib_fn(backend, name)


def promote(a: Sequence[Any], b: Sequence[Any]) -> tuple[str, Sequence[Any], Sequence[Any]]:
    """One backend for two operands' arrays, with the odd side moved onto it.

    Every binary operation on two tensors has to answer this before it resolves a
    single backend function: a NumPy constant meeting a traced JAX tensor -- one
    CTMRG corner is exactly that -- must dispatch to JAX, and a NumPy operand
    meeting a torch one must dispatch to torch *and* be converted, because
    ``torch.add`` refuses an ``ndarray`` outright where ``jax.numpy.add`` accepts
    one. ``infer_backend_multi`` alone therefore fixes only half of it.

    Parameters
    ----------
    a, b : sequence of array
        The two operands' arrays -- ``data`` or ``blocks``, whichever the caller
        operates on. A tensor's arrays all share a backend, so the first of each
        decides for the rest and this costs two lookups per *call*, not per sector.

    Returns
    -------
    backend : str
        The backend both returned sequences are on, for
        [lib_fn][tenet.backend.lib_fn].
    a, b : sequence of array
        The inputs, with whichever side was not already on ``backend`` converted
        with ``ar.do("array", m, like=backend)`` -- the same spelling
        [SymmetricTensor.to_backend][tenet.SymmetricTensor.to_backend] uses. The
        matching side is returned untouched, so the same-backend path (every call
        that is not a mixed one) allocates nothing.

    Notes
    -----
    An empty operand carries no backend and no arrays to convert, so the pair is
    returned as-is under NumPy's name; a caller in that state has nothing to
    dispatch anyway (its loop over the pair runs zero times).
    """
    if not a or not b:
        return "numpy", a, b
    ba, bb = ar.infer_backend(a[0]), ar.infer_backend(b[0])
    if ba == bb:
        return ba, a, b
    backend = ar.infer_backend_multi(a[0], b[0])
    cast = lib_fn(backend, "array")
    return (
        backend,
        a if ba == backend else tuple(cast(m) for m in a),
        b if bb == backend else tuple(cast(m) for m in b),
    )
