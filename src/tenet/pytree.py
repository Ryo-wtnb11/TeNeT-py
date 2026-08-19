"""Opt-in JAX PyTree registration. ``import tenet.pytree``; core never imports this.

The contract: the leaves of a [SymmetricTensor][tenet.SymmetricTensor] are its ``blocks``, in
``structure.block_order``; the aux data (the treedef) is its ``structure``, which is
frozen, hashable and array-free (invariant 8) and therefore a sound JIT cache key.

``_unflatten`` deliberately does **not** validate, because JAX calls it with sentinels,
tracers and whatever a ``jax.tree.map`` returned
(https://docs.jax.dev/en/latest/custom_pytrees.html). The public constructor remains the
trust boundary: ``SymmetricTensor(T.structure, T.blocks)`` re-runs the full check.

Batching recipe::

    batched = jax.tree.map(lambda *bs: jnp.stack(bs), T1, T2, T3)
    jax.vmap(lambda T: tenet.norm(T) ** 2)(batched)

``batched`` is a transport container, not a tensor: its block shapes do not match its
structure and nothing in ``tenet.*`` should be called on it. Inside the vmapped function
the leaves are ``BatchTracer``s whose ``.shape`` is the *unbatched* shape, so ops and
their validation behave normally.

Complex-dtype gradients are unverified: ``norm`` goes through ``abs``, whose JAX
derivative needs care for complex input. Tests here are real ``float64``.
"""

try:
    import jax
except ImportError as exc:  # loud, actionable, names the extra
    raise ImportError(
        "tenet.pytree requires JAX, which is an optional dependency. "
        "Install it with `pip install 'tenet-py[jax]'` or `uv sync --extra jax`. "
        "The core library never imports JAX; get_params/set_params work without it."
    ) from exc

from tenet.structure import TensorStructure
from tenet.tensor import Array, SymmetricTensor

__all__: list[str] = []


def _flatten(t: SymmetricTensor) -> tuple[tuple[Array, ...], TensorStructure]:
    """Leaves are the blocks in ``structure.block_order``; aux data is the structure."""
    return t.blocks, t.structure


def _unflatten(structure: TensorStructure, blocks: object) -> SymmetricTensor:
    """Rebuild WITHOUT running ``__init__`` / ``__post_init__``.

    This is the pytree contract, not a shortcut: JAX legitimately calls this with
    ``object()`` sentinels (vmap / structure inference), with ``SymmetricTensor``
    instances (``jax.jacobian``'s tree-of-trees), with batched tracers, and with
    whatever a user's ``jax.tree.map`` returned. None of those satisfy the block
    shape check, and none of them are errors. Validation belongs on the public
    constructor, which is untouched.
    """
    t = object.__new__(SymmetricTensor)
    object.__setattr__(t, "structure", structure)
    # ``blocks`` is whatever iterable the tree library hands back (see docstring);
    # it is deliberately typed ``object`` and left unvalidated here, so the
    # checker cannot see it is iterable.
    object.__setattr__(t, "blocks", tuple(blocks))  # ty: ignore[invalid-argument-type]
    return t


# Registration is the import's side effect; a second import is a no-op because the
# module object is cached in sys.modules, so no explicit guard flag is needed.
# register_pytree_node, not register_pytree_with_keys: the keyed variant buys nicer
# error paths (`.blocks[3]` over a positional index) for a third callback and a key
# type, on a leaf list that is already positional-by-contract. Cheap follow-up if
# debugging ever demands it.
jax.tree_util.register_pytree_node(SymmetricTensor, _flatten, _unflatten)
