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

from functools import cache
from typing import Any

import autoray as ar

__all__ = ["lib_fn"]


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
