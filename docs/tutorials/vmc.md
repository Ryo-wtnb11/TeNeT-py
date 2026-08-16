# VMC — symmetric MPS → JAX pytree → grad → SGD step

Source: [`examples/vmc_mps.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/vmc_mps.py).
**Oracle:** the variational energy decreases across the SGD step, checked by
`tests/integration/test_vmc_mps.py`, which executes the file.

```sh
uv run --extra jax python examples/vmc_mps.py
```

## What it demonstrates

Entirely with library code that already exists — no new `src/tenet` module, no `optax`,
no `quimb`:

- a **symmetric open-boundary MPS whose parameters are `SymmetricTensor` blocks**, i.e. a
  JAX pytree the moment `tenet.pytree` is imported;
- an objective — the Rayleigh quotient `<ψ|h|ψ> / <ψ|ψ>` — built from a left-to-right
  chain of *pairwise* `tenet.einsum` calls (three or more operands need a contraction
  path, which is a separate concern);
- `jax.grad` straight through that objective, and a one-line SGD step written with
  `jax.tree.map`;
- `tenet.linalg.svd` (exact, shape-static) *inside* the differentiated and jitted path,
  and `tenet.linalg.svd_truncated` (structure-changing) *outside* it — plus the pairing of
  the two: `compress` deciding a bond space out here, `project` running `svd(t, bond=...)`
  in there.

## Trivial boundary legs, not a rank-0 tensor

`SymmetricTensor` has no rank 0, and it does not need one: the standard MPS convention
gives the left and right boundary legs the unit sector with degeneracy 1, so the fully
closed network is a rank-2 tensor with two trivial legs and a single `(1, 1)` block. That
block *is* the scalar, and `scalar()` is where the tensor world is explicitly left — the
same move `tenet.norm` makes.

## Honest limitation on batching

`jax.vmap` batches samples that share one `TensorStructure`, because the structure is the
treedef. A per-sample computational-basis projector for a genuine Monte-Carlo amplitude
has a sample-dependent sector pattern, hence a sample-dependent structure, hence a
different treedef — those cannot be `vmap`ed together. What `vmap` *does* batch is a set
of ansätze sharing one sector pattern (equivalently, one total charge), which is the
physically meaningful batching for a symmetric ansatz anyway. Sampling itself is out of
scope in this example.

`h` is a *random* symmetric two-site operator, not a Heisenberg term: equivariance is
automatic from the legs and the pipeline under test is identical. Build the physical
operator when a physics result — not a plumbing result — is wanted.

## Reference

- [`tenet.pytree`](../api/pytree.md) — the registration whose side effect is the pytree
- [`tenet.linalg`](../api/linalg.md) — `svd`, `svd_truncated`
