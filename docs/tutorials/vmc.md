# VMC — a symmetric MPS through `jax.grad`

A variational energy minimized by gradient descent on the blocks of a symmetric MPS.
Source:
[`examples/toy_codes/vmc_mps.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/vmc_mps.py);
its output is committed on the [Toy VMC on an MPS](../examples/toy-vmc-mps.md) page.

```sh
uv run --extra jax python examples/toy_codes/vmc_mps.py
```

Nothing new is imported: no `optax`, no `quimb`, no extra module under `src/tenet`. The
blocks below are illustrative rather than doctests — the page needs the `jax` extra, which
the core install does not carry. Every printed number quoted here is from that committed
output.

## Turning the ansatz into a pytree

```python
import jax
import tenet

tenet.enable_jax()      # SymmetricTensor becomes a JAX pytree
```

The parameters of the ansatz are the tensors' blocks, and the treedef is each tensor's
`TensorStructure` — frozen, hashable and array-free. So an MPS is a pytree the moment
`enable_jax()` has run, with no wrapper type and no parameter-flattening step of its own.

```python
SymmetricTensor.random(
    (Leg(spaces[i], OUT), Leg(phys, OUT), Leg(spaces[i + 1], IN)), seed=seed + i
).to_backend("jax")
```

One site tensor: left bond `OUT`, physical leg `OUT`, right bond `IN`, so charge flows
left to right along the chain. `to_backend("jax")` is called here rather than later,
because the blocks have to be JAX arrays before `grad` ever sees them as leaves. The first
site's left space and the last site's right space are the trivial one — unit sector,
degeneracy 1.

## Trivial boundary legs, not a rank-0 tensor

`SymmetricTensor` has no rank 0, and does not need one. The standard MPS convention gives
the left and right boundary legs the unit sector with degeneracy 1, so a fully closed
network is a rank-2 tensor with two trivial legs and a single `(1, 1)` block.

```python
num = tenet.einsum(f"aBC{rest}s,aBC{rest}z->sz", bra, hpsi)
den = tenet.einsum(f"a{phys}{rest}s,a{phys}{rest}z->sz", bra, psi)
return tenet.full_trace(num) / tenet.full_trace(den)
```

Both equations close every physical leg and the left boundary `a`, leaving the two
right-boundary legs `s` and `z` open — a rank-2 square map on the trivial space. That
block *is* the scalar, and [`full_trace`][tenet.full_trace] reading it is where the tensor
world is explicitly left, the same move [`tenet.norm`][tenet.norm] makes. Dividing the two
is what makes the objective the Rayleigh quotient
$\langle\psi\vert h \vert\psi\rangle/\langle\psi\vert\psi\rangle$, and therefore what makes
the gradient point at an eigenvector rather than merely at a shorter vector.

The whole objective is a left-to-right chain of **pairwise** `tenet.einsum` calls. Three
or more operands would bring in a contraction path, which is a separate concern from the
gradient; one static chain is also one `jit` trace.

## One SGD step

```python
e, grads = jax.value_and_grad(energy)(mps, h)
return [jax.tree.map(lambda p, g: p - lr * g, t, g) for t, g in zip(mps, grads)]
```

`grad` differentiates with respect to the first argument only, so `h` stays a constant of
the problem. `jax.tree.map` walks each tensor's blocks as leaves, so the update touches
block values and carries the grading through untouched — the step cannot leave the
symmetric manifold, which is why there is no projection back onto it anywhere.

Both a U(1) and an SU(2) ansatz descend through this identical code path, and the
committed output shows both energies falling monotonically across all 20 steps: U(1) from
`-0.682692` to `-1.050687`, SU(2) from `-1.033559` to `-1.090355`.

## Both halves of the truncation pairing

```python
def compress(t, max_bond=2):
    return tenet.linalg.svd_truncated(t, max_bond=max_bond)   # outside jit/grad

def project(t, bond):
    return tenet.linalg.svd(t, bond=bond)                     # inside
```

[`svd_truncated`][tenet.ops.linalg.svd_truncated] picks which sectors survive from the
singular *values*, so its output `TensorStructure` depends on the data: under a trace the
values are tracers and it raises. [`svd`][tenet.ops.linalg.svd] with `bond=` takes that
decision as frozen metadata and does only the numerics, so it traces and differentiates
like any other exact factorization.

```python
bond = compress(t0, max_bond=D)[0].legs[-1].space   # decided once, outside
jax.jit(jax.grad(lambda t: tenet.norm(project(t, bond)[1])))(t)
```

Decide the bond space once, outside; project onto it every iteration, inside. That is the
shape a differentiable CTMRG or variational iPEPS has, and this file is the smallest
problem that needs it. Re-deciding the kept subspace inside the loop would mean
differentiating through a discrete choice, which has no derivative.
[Truncation](../guide/truncation.md) works the pairing through in full.

The exact `svd` also appears for a second reason, gauge rather than truncation:
`canonicalize` splits site 0 and pushes `s @ vh` into site 1. The chain's product is
unchanged, so the energy is unchanged — a gauge transformation, not an approximation — and
site 0 comes back an isometry.

## What `vmap` batches

`jax.vmap` batches samples that share one `TensorStructure`, because the structure is the
treedef. A per-sample computational-basis projector for a Monte-Carlo amplitude has a
sample-dependent sector pattern, hence a sample-dependent structure, hence a different
treedef; those cannot be batched together. What `vmap` does batch is a set of ansätze
sharing one sector pattern — equivalently, one total charge — which is the physically
meaningful batching for a symmetric ansatz. Sampling itself is outside this example.

`h` is a random symmetric two-site operator rather than a Heisenberg term: its legs make
it equivariant whatever numbers are drawn, and the pipeline under test is identical. Build
the physical operator when you want a physics result rather than a plumbing result.

## Where next

- [JAX and backends](../guide/jax-and-backends.md) — pytrees, `vmap`, and the broadened
  VJPs.
- [Truncation](../guide/truncation.md) — the decide-outside / project-inside pairing.
- [`tenet.pytree`](../api/pytree.md) and [`tenet.linalg`](../api/linalg.md).
