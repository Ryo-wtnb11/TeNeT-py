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
([transpose][tenet.transpose], [repartition][tenet.repartition], [bend][tenet.bend],
[fuse][tenet.fuse] / [unfuse][tenet.unfuse], [conj][tenet.conj],
[adjoint][tenet.adjoint]); construction ([identity][tenet.identity],
[isometry][tenet.isometry], [from_matrices][tenet.from_matrices],
[direct_sum][tenet.direct_sum], [embed][tenet.embed],
[to_symmetry][tenet.to_symmetry]); and block access ([to_matrices][tenet.to_matrices],
[apply_blocks][tenet.apply_blocks]).
[save][tenet.save] / [load][tenet.load] persist one tensor, and anything that would change
block structure raises [StructureChangingError][tenet.StructureChangingError].
Every function that validates against an ``atol`` takes [PROJECT][tenet.PROJECT] there to
mean "project, don't check" instead.

Submodules with their own pages: ``tenet.linalg`` (decompositions), ``tenet.network``
(DMRG and CTMRG), ``tenet.symmetry`` (providers and sector labels), and the opt-in JAX
seams [tenet.pytree][] and [tenet.ad][], which [enable_jax][tenet.enable_jax] turns on.
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
    PROJECT,
    add,
    adjoint,
    allclose,
    apply_blocks,
    bend,
    block_power,
    block_sqrt,
    braid,
    compose,
    conj,
    direct_sum,
    divide,
    einsum,
    einsum_chain,
    flip_dual,
    full_trace,
    fuse,
    identity,
    inner,
    isometry,
    linalg,
    map_diagonal,
    multiply,
    negative,
    norm,
    random_isometry,
    restrict,
    subtract,
    tensordot,
    to_symmetry,
    trace,
    transpose,
    twist,
    unfuse,
    zip_blocks,
)

# `embed` and `repartition` are imported from their defining modules, not from
# `tenet.ops`, because each shares its name with the submodule that defines it. In
# `tenet.ops`'s namespace the submodule wins, so a documentation tool following
# `from tenet.ops import embed` lands on the module and the function drops off the
# `tenet` API page. Naming the leaf module resolves to the function (#173).
from tenet.ops.embed import embed
from tenet.ops.repartition import repartition
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


def enable_jax(*, ad: bool = False) -> None:
    """Turn on the JAX-facing features, in one call instead of three statements.

    Two separate effects, and only the first is on by default:

    1. **The pytree registration** ([tenet.pytree][]) — ``SymmetricTensor`` becomes a JAX
       pytree whose leaves are its blocks and whose treedef is its
       [TensorStructure][tenet.TensorStructure], so ``jit``, ``grad`` and ``vmap`` reach
       through it. This is local to this package: it registers *our* type with JAX and
       changes nothing about anyone else's.

    2. **The broadened SVD/eigh VJPs** ([tenet.ad][], with ``ad=True``) — the
       Lorentzian-broadened rules that stay finite at the degenerate spectra a non-Abelian
       symmetry produces. This one is **process-global and reaches other libraries**: the
       seam is ``autoray.register_function("jax", "linalg.svd", ...)``, autoray's own
       extension point, so afterwards *any* ``ar.do("linalg.svd", jax_array)`` in the
       process — quimb's included — gets the broadened VJP, and the broadened gradient is
       correct only for an objective that is gauge-invariant on each degenerate subspace.
       Mutating another library's dispatch table is the user's act, so it is opted into by
       name rather than defaulted on; ``tenet.ad``'s module docstring is the full argument.

    Parameters
    ----------
    ad : bool, optional
        Whether to also install ``tenet.ad``'s broadened VJPs, effect 2 above.
        Defaults to ``False``, which is the common case (the pytree alone). Pass
        ``True`` when differentiating through ``svd``/``eigh`` at a degenerate
        spectrum. To tune the broadening, call
        [tenet.ad.install][tenet.ad.install]``(epsilon=...)`` directly instead.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If JAX is not installed, naming the optional extra to install.

    Examples
    --------
    >>> import tenet
    >>> tenet.enable_jax()          # the pytree registration; calling it twice is a no-op
    >>> tenet.enable_jax()

    Notes
    -----
    Idempotent, in both halves: re-importing ``tenet.pytree`` is a ``sys.modules`` hit and
    ``tenet.ad.install()`` documents itself as idempotent, so repeat calls are harmless.

    The older spellings are unchanged and keep working — ``import tenet.pytree`` and
    ``tenet.ad.install()`` are what this function runs, and there is one implementation of
    each. JAX stays an optional dependency: nothing here is imported by core.
    """
    # The submodules carry the guarded JAX import, and this file must not: three source
    # greps hold "core never imports jax" up by the file walk rather than by review
    # (tests/array/test_dispatch.py, tests/test_tensor_properties.py,
    # tests/backends/test_pytree.py). So the failure is caught and re-raised here, which
    # is also what turns a traceback out of a submodule into one sentence.
    try:
        import tenet.pytree  # noqa: F401  # registration is the import's side effect

        if ad:
            import tenet.ad

            tenet.ad.install()
    except ImportError as exc:
        raise ImportError(
            "tenet.enable_jax() requires JAX, which is an optional dependency. "
            "Install it with `pip install 'tenet-py[jax]'` or `uv sync --extra jax`. "
            "The core library never imports JAX; get_params/set_params work without it."
        ) from exc


__all__ = [
    "IN",
    "OUT",
    "PROJECT",
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
    "braid",
    "compose",
    "conj",
    "coupled_sectors",
    "direct_sum",
    "divide",
    "einsum",
    "einsum_chain",
    "embed",
    "enable_jax",
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
    "map_diagonal",
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
    "twist",
    "unfuse",
    "zip_blocks",
]
