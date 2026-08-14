"""TeNeT-py: non-Abelian symmetric tensors with ndarray-style APIs."""

from tenet.fusion_tree import FusionTree, coupled_sectors, fusion_trees
from tenet.leg import IN, OUT, Leg, Side
from tenet.map_view import (
    MapLayout,
    TensorMapView,
    as_map,
    from_matrices,
    map_layout,
    to_matrices,
)
from tenet.ops import (
    add,
    adjoint,
    allclose,
    bend,
    compose,
    conj,
    divide,
    fuse,
    identity,
    multiply,
    negative,
    norm,
    repartition,
    subtract,
    tensordot,
    trace,
    transpose,
    unfuse,
)
from tenet.space import GradedSpace, ProductSpace
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.tensor import SymmetricTensor

__version__ = "0.1.0"

# Last: dispatch.py imports from tenet.ops, and registers this package with
# autoray as a side effect of `import tenet`.
from tenet import array  # noqa: E402, F401

__all__ = [
    "IN",
    "OUT",
    "FusionBlockKey",
    "FusionTree",
    "GradedSpace",
    "Leg",
    "MapLayout",
    "ProductSpace",
    "Side",
    "SymmetricTensor",
    "TensorMapView",
    "TensorStructure",
    "add",
    "adjoint",
    "allclose",
    "as_map",
    "bend",
    "compose",
    "conj",
    "coupled_sectors",
    "divide",
    "from_matrices",
    "fuse",
    "fusion_trees",
    "identity",
    "map_layout",
    "multiply",
    "negative",
    "norm",
    "repartition",
    "subtract",
    "tensordot",
    "to_matrices",
    "trace",
    "transpose",
    "unfuse",
]
