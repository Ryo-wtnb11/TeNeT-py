# VMC — a symmetric MPS through `jax.grad`

A variational energy minimized by gradient descent on the blocks of a symmetric MPS.
Source:
[`examples/toy_codes/vmc_mps.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/vmc_mps.py);
its output is committed on the [Toy VMC on an MPS](../examples/toy-vmc-mps.md) page.

```sh
uv run --extra jax python examples/toy_codes/vmc_mps.py
```

Nothing new is imported: no `optax`, no `quimb`, no extra module under `src/tenet`.

## The pieces

```python
import jax
import tenet

tenet.enable_jax()      # SymmetricTensor becomes a JAX pytree
```

The parameters of the ansatz are the tensors' blocks, and the treedef is each tensor's
`TensorStructure` — frozen, hashable and array-free. So an MPS is a pytree the moment
`enable_jax()` has run, and an SGD step is one line of `jax.tree.map`.

The objective is the Rayleigh quotient `<ψ|h|ψ> / <ψ|ψ>`, built from a left-to-right chain
of **pairwise** `tenet.einsum` calls. Three or more operands would bring in a contraction
path, which is a separate concern from the gradient.

`jax.grad` runs straight through that objective. Both a U(1) and an SU(2) ansatz descend
through the identical code path, and both energies fall monotonically across all 20 SGD
steps.

## Trivial boundary legs, not a rank-0 tensor

`SymmetricTensor` has no rank 0, and does not need one. The standard MPS convention gives
the left and right boundary legs the unit sector with degeneracy 1, so a fully closed
network is a rank-2 tensor with two trivial legs and a single `(1, 1)` block. That block
*is* the scalar, and reading it is where the tensor world is explicitly left — the same
move [`tenet.norm`][tenet.norm] makes.

## Both halves of the truncation pairing

The file uses both:

- [`svd`][tenet.ops.linalg.svd] — exact and shape-static — *inside* the differentiated and
  jitted path;
- [`svd_truncated`][tenet.ops.linalg.svd_truncated] — structure-changing — *outside* it,
  deciding a bond space that `svd(t, bond=...)` then reuses inside.

That is the pairing in [Truncation](../guide/truncation.md), on the smallest problem that
needs it.

## What `vmap` batches

`jax.vmap` batches samples that share one `TensorStructure`, because the structure is the
treedef. A per-sample computational-basis projector for a Monte-Carlo amplitude has a
sample-dependent sector pattern, hence a sample-dependent structure, hence a different
treedef; those cannot be batched together. What `vmap` does batch is a set of ansätze
sharing one sector pattern — equivalently, one total charge — which is the physically
meaningful batching for a symmetric ansatz. Sampling itself is outside this example.

`h` is a random symmetric two-site operator. Equivariance is automatic from the legs, and
the pipeline under test is identical; build the physical operator when you want a physics
result rather than a plumbing result.

## Where next

- [JAX and backends](../guide/jax-and-backends.md) — pytrees, `vmap`, and the broadened
  VJPs.
- [Truncation](../guide/truncation.md) — the decide-outside / project-inside pairing.
- [`tenet.pytree`](../api/pytree.md) and [`tenet.linalg`](../api/linalg.md).
