# JAX and backends

A `SymmetricTensor`'s blocks are backend arrays, reached through
[`autoray`](https://github.com/jcmgray/autoray). NumPy, JAX and PyTorch all work, and the
same code runs on each.

## Which backend a tensor is on

```python
>>> import tenet
>>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
>>> from tenet.symmetry import U1, U1Sector
>>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
>>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
>>> t.backend, t.dtype
('numpy', dtype('float64'))

```

`backend` is inferred from the first block, `device` is the first block's own `.device`
(`None` when it has none), and all blocks share one dtype.

[`to_backend`][tenet.SymmetricTensor.to_backend] moves the blocks and takes an optional
`dtype=` applied **after** the move, so the target backend's own dtype choice does not
get the last word. [`astype`][tenet.SymmetricTensor.astype] casts in place on the current
backend:

```python
>>> t.astype("complex128").dtype
dtype('complex128')
>>> t.to_backend("numpy", dtype="complex128").dtype
dtype('complex128')

```

Under JAX's default configuration there is no per-array escape from `jax_enable_x64`, so
a request for `float64` truncates to float32 exactly as it does in `jnp.array`. Set
`jax.config.update("jax_enable_x64", True)` before building anything if you want float64.

PyTorch blocks are supported eager. `torch.compile` and `torch.func` are not part of the
contract, and `tenet.ad` is JAX-only.

## Turning JAX on

```python
import tenet

tenet.enable_jax()
```

One call, and it is idempotent. It performs the **pytree registration**:
`SymmetricTensor` becomes a JAX pytree whose leaves are its blocks, in
`structure.block_order`, and whose treedef is its
[`TensorStructure`][tenet.TensorStructure] — frozen, hashable and array-free, so it is a
sound `jit` cache key. `jit`, `grad` and `vmap` then reach through the tensor to the
blocks while the structure stays static metadata.

This is local to this package: it registers *our* type with JAX and changes nothing about
anyone else's. Without JAX installed it raises an `ImportError` naming the extra.

```python
import jax

g = jax.jit(jax.grad(lambda t: tenet.norm(t) ** 2))(t)
assert g.legs == t.legs
```

`get_params` / `set_params` are the same split spelled without JAX:
[`get_params`][tenet.SymmetricTensor.get_params] hands back the blocks in
`block_order`, and [`set_params`][tenet.SymmetricTensor.set_params] returns a new tensor
with the same structure and new data. They work on a core install.

## Batching with `vmap`

`jax.vmap` batches tensors that share one `TensorStructure`, because the structure is the
treedef:

```python
batched = jax.tree.map(lambda *bs: jnp.stack(bs), t1, t2, t3)
jax.vmap(lambda t: tenet.norm(t) ** 2)(batched)
```

`batched` is a transport container, not a tensor: its block shapes do not match its
structure, so call nothing from `tenet.*` on it. Inside the vmapped function the leaves
are tracers whose `.shape` is the unbatched shape, so operations and their validation
behave normally.

What this batches is a set of ansätze sharing one sector pattern — equivalently, one
total charge. Tensors with different sector patterns have different treedefs and cannot
be batched together.

## What traces and what refuses

An operation whose **output structure depends on the block values** cannot run inside a
traced region, and says so by raising
[`StructureChangingError`][tenet.symmetry.StructureChangingError] under `jax.jit`,
`jax.grad` or `jax.vmap`. The truncating factorizations are the ones you will meet:
`svd_truncated`, `eigh_truncated`, `select_bond`.

The pairing is: decide the structure once, outside; project onto it inside.

```python
selection = tenet.linalg.select_bond(t0, max_bond=D)   # outside jit/grad

@jax.jit
def step(t):
    u, s, vh = tenet.linalg.svd(t, bond=selection.bond)
    ...
```

`svd(..., bond=)` and `eigh(..., bond=)` are the traceable halves, because a
`GradedSpace` is frozen, array-free metadata. Everything else in the tensor layer —
`einsum`, `tensordot`, `compose`, `transpose`, `repartition`, `fuse`, the exact
factorizations — is shape-static and traces.

`tenet.network`'s DMRG runs outside `jit`/`grad` and makes no differentiability claim:
its sweep re-decides a bond space at every bond. Its two traceable pieces say so on
themselves — [`Env.heff2`][tenet.network.Env.heff2]'s prepared matvec through an injected
`compile=`, and [`EnvCTMc4v.update_`][tenet.network.EnvCTMc4v.update_] with `bond=B` in
CTMRG.

## Gradients at a degenerate spectrum

JAX's own SVD and eigh VJPs carry $1/(\sigma_i - \sigma_j)$ factors that are `NaN` at exact
degeneracy, and under a non-Abelian symmetry degeneracy inside a coupled sector is
generic. `tenet.ad` replaces them with the Lorentzian-broadened form,
$1/x \to x/(x^2 + \epsilon)$:

```python
tenet.enable_jax(ad=True)
```

Three things to know before you pass `ad=True`:

- **It is process-global for the JAX backend.** The seam is
  `autoray.register_function("jax", "linalg.svd", ...)`, autoray's own extension point,
  so afterwards any `ar.do("linalg.svd", jax_array)` in the process — another library's
  included — gets the broadened VJP. Mutating another library's dispatch table is your
  act, so you name it: `ad=True`, every time you want it.
- **The broadened gradient is correct exactly when the objective is gauge-invariant on
  each degenerate subspace.** Within a degenerate multiplet the singular vectors are
  defined only up to a unitary, so $dU/dA$ does not exist; what exists is the derivative
  of gauge-invariant combinations — `U S Vh`, the singular values, a projector onto the
  multiplet. Broadening returns the correct value for those. A gauge-dependent objective
  gets an `eps`-dependent answer.
- **`eps` is in units of $\sigma^2$.** Call
  [`tenet.ad.install`][tenet.ad.install]`(epsilon=...)` directly to tune it;
  [`tenet.ad.uninstall`][tenet.ad.uninstall] restores autoray's stock bindings.

`qr`, `lq` and `polar` need nothing of their own.

## Where next

- [Truncation](truncation.md) — the decide-outside / project-inside pairing in full.
- [`tenet.pytree`](../api/pytree.md) and [`tenet.ad`](../api/ad.md) — the reference.
- [VMC on an MPS](../tutorials/vmc.md) — `grad` straight through a symmetric ansatz.
