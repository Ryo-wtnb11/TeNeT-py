"""TeNeT-py: non-Abelian symmetric tensors with ndarray-style APIs.

[SymmetricTensor][tenet.SymmetricTensor] is the tensor type. Its categorical structure is
[Leg][tenet.Leg] (``IN``/``OUT``, [Side][tenet.Side]),
[GradedSpace][tenet.GradedSpace] / [ProductSpace][tenet.ProductSpace],
[FusionTree][tenet.FusionTree] and [TensorStructure][tenet.TensorStructure], and
[as_map][tenet.as_map] views it as a map from domain to codomain.

The operations group by what a caller comes for: arithmetic and reductions
([add][tenet.add], [multiply][tenet.multiply], [norm][tenet.norm], [trace][tenet.trace],
[inner][tenet.inner], [allclose][tenet.allclose]); contraction ([einsum][tenet.einsum],
[tensordot][tenet.tensordot], [compose][tenet.compose]); leg and axis moves
([transpose][tenet.transpose], [repartition][tenet.SymmetricTensor.repartition], [bend][tenet.bend],
[fuse][tenet.fuse] / [unfuse][tenet.unfuse], [conj][tenet.conj],
[adjoint][tenet.adjoint]); construction ([identity][tenet.identity],
[isometry][tenet.isometry], [from_matrices][tenet.from_matrices],
[direct_sum][tenet.direct_sum], [embed][tenet.SymmetricTensor.embed],
[to_symmetry][tenet.to_symmetry]); and block access ([to_matrices][tenet.to_matrices],
[apply_blocks][tenet.apply_blocks]).
[save][tenet.save] / [load][tenet.load] persist one tensor, and anything that would change
block structure raises [StructureChangingError][tenet.StructureChangingError].

Submodules with their own pages: ``tenet.linalg`` (decompositions), ``tenet.network``
(DMRG and CTMRG), ``tenet.symmetry`` (providers and sector labels), and the opt-in JAX
seams [tenet.pytree][] and [tenet.ad][].
"""

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
    block_power,
    block_sqrt,
    compose,
    conj,
    direct_sum,
    divide,
    einsum,
    embed,
    flip_dual,
    full_trace,
    fuse,
    identity,
    inner,
    isometry,
    linalg,
    multiply,
    negative,
    norm,
    random_isometry,
    repartition,
    restrict,
    subtract,
    tensordot,
    to_symmetry,
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
    "block_power",
    "block_sqrt",
    "compose",
    "conj",
    "coupled_sectors",
    "direct_sum",
    "divide",
    "einsum",
    "embed",
    "flip_dual",
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
    "random_isometry",
    "repartition",
    "restrict",
    "save",
    "subtract",
    "tensordot",
    "to_matrices",
    "to_symmetry",
    "trace",
    "transpose",
    "unfuse",
]
