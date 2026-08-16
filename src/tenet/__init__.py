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
    apply_blocks,
    bend,
    cast,
    compose,
    conj,
    direct_sum,
    divide,
    einsum,
    embed,
    full_trace,
    fuse,
    identity,
    inner,
    isometry,
    linalg,
    multiply,
    negative,
    norm,
    power,
    random_isometry,
    repartition,
    restrict,
    sqrt,
    subtract,
    tensordot,
    trace,
    transpose,
    unfuse,
)
from tenet.serialize import load, save
from tenet.space import GradedSpace, ProductSpace
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry.base import StructureChangingError
from tenet.tensor import SymmetricTensor

__version__ = "0.1.0"

# Last: dispatch.py imports from tenet.ops, and registers this package with autoray as a
# side effect of `import tenet`. `network` (the M11a driver layer) is imported alongside
# it, for the same reason -- it imports tenet.ops -- so that `tenet.network` resolves
# after `import tenet`. It is deliberately **not** flattened into this namespace: `dmrg_`
# is not a tensor operation and `tenet.dmrg_` would read like one.
from tenet import (
    array,  # noqa: E402, F401
    network,  # noqa: E402, F401
)

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
    "StructureChangingError",
    "SymmetricTensor",
    "TensorMapView",
    "TensorStructure",
    "add",
    "adjoint",
    "allclose",
    "apply_blocks",
    "as_map",
    "bend",
    "cast",
    "compose",
    "conj",
    "coupled_sectors",
    "direct_sum",
    "divide",
    "einsum",
    "embed",
    "from_matrices",
    "full_trace",
    "fuse",
    "fusion_trees",
    "identity",
    "inner",
    "isometry",
    "linalg",
    "load",
    "map_layout",
    "multiply",
    "negative",
    "network",
    "norm",
    "power",
    "random_isometry",
    "repartition",
    "restrict",
    "save",
    "sqrt",
    "subtract",
    "tensordot",
    "to_matrices",
    "trace",
    "transpose",
    "unfuse",
]
