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
from tenet.ops.permutation import transpose

__all__ = [
    "add",
    "allclose",
    "conj",
    "divide",
    "fuse",
    "multiply",
    "negative",
    "norm",
    "subtract",
    "transpose",
    "unfuse",
]
