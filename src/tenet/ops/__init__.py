"""Operations on :class:`~tenet.tensor.SymmetricTensor`.

The dependency edge is one-way: ``ops`` imports ``tensor``, never the reverse
(the dunders on ``SymmetricTensor`` use function-local imports).
"""

from tenet.ops.basic import (
    add,
    allclose,
    conj,
    divide,
    multiply,
    negative,
    norm,
    subtract,
)
from tenet.ops.fusion import fuse, unfuse
from tenet.ops.map import adjoint, compose, identity
from tenet.ops.permutation import transpose

__all__ = [
    "add",
    "adjoint",
    "allclose",
    "compose",
    "conj",
    "divide",
    "fuse",
    "identity",
    "multiply",
    "negative",
    "norm",
    "subtract",
    "transpose",
    "unfuse",
]
