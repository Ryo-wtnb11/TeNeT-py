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
from tenet.ops.cast import cast
from tenet.ops.contraction import einsum, tensordot, trace
from tenet.ops.embed import direct_sum, embed, restrict
from tenet.ops.fusion import fuse, unfuse
from tenet.ops.map import adjoint, compose, identity
from tenet.ops.permutation import transpose
from tenet.ops.repartition import bend, repartition

__all__ = [
    "add",
    "adjoint",
    "allclose",
    "bend",
    "cast",
    "compose",
    "conj",
    "direct_sum",
    "divide",
    "einsum",
    "embed",
    "fuse",
    "identity",
    "multiply",
    "negative",
    "norm",
    "repartition",
    "restrict",
    "subtract",
    "tensordot",
    "trace",
    "transpose",
    "unfuse",
]
