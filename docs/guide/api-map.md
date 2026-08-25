# Which operation do I want?

The top-level `tenet` namespace is flat and wide on purpose: it is the tensor layer, and
a tensor layer has a lot of verbs. What is *not* flat is what those verbs mean. This page
sorts them by the question a caller arrives with, and marks how much categorical
machinery each one exposes:

- **common** — the operations an ordinary calculation uses. Learn these.
- **map** — operations that read a tensor as a map $D \to C$ rather than as an array of
  axes. Needed as soon as you factorize or compose.
- **categorical** — operations whose meaning is the symmetry's, not the array's:
  duality, braiding, twists, restriction. Correct use needs the
  [symmetry page](symmetries-and-providers.md).
- **low-level** — building blocks for writing your own algorithm. Nothing in a normal
  calculation calls these directly.

Every name below is checked against the implementation by `tests/test_api_map.py`, which
fails if a symbol on this page stops resolving.

## Building things

| Goal | Canonical API | Level |
| --- | --- | --- |
| Declare a symmetry sector space | `tenet.GradedSpace.new` | common |
| Attach a space to one axis | `tenet.Leg` (`IN` / `OUT`, `dual=`) | common |
| Random tensor, reproducibly | `tenet.SymmetricTensor.random` | common |
| Zero tensor of a given dtype | `tenet.SymmetricTensor.zeros` | common |
| Tensor from the blocks you know | `tenet.SymmetricTensor.from_blocks` | common |
| Tensor from a dense array | `tenet.SymmetricTensor.from_dense` | common |
| Replace some blocks, keep the rest | `tenet.SymmetricTensor.with_blocks` | common |
| The identity map on a leg tuple | `tenet.identity` | common |
| An isometry between two leg tuples | `tenet.isometry` / `tenet.random_isometry` | map |
| A standard physical site + operators | `tenet.models.spin_half`, `tenet.models.spinless_fermion`, `tenet.models.spinful_fermion`, `tenet.models.hard_core_boson` | common |
| A local operator with a charge leg | `tenet.network.local_op` | common |
| Which blocks does this tensor have? | `tenet.TensorStructure.block_order` (keys are `tenet.FusionBlockKey`) | map |

`GradedSpace` answers two size questions and they are not the same number: `reduced_dim`
is $\sum_a m_a$, what the stored blocks are made of, and `dim` is $\sum_a m_a d_a$, what
`to_dense` produces. See [Tensors, legs and spaces](tensors-legs-spaces.md).

## Contracting

| Goal | Canonical API | Level |
| --- | --- | --- |
| Contract explicit axis pairs | `tenet.tensordot` | common |
| Contract by index labels | `tenet.einsum` | common |
| Contract a chain with explicit bends | `tenet.einsum_chain` | low-level |
| Compose two maps, $A \circ B$ | `A @ B`, i.e. `tenet.compose` | map |
| Close a matched leg pair | `tenet.trace` | common |
| Close *every* leg to a scalar | `tenet.full_trace` | common |
| $\langle a, b\rangle$ | `tenet.inner` | common |
| $\lVert a\rVert$ | `tenet.norm` | common |

Every two-operand contraction is a composition: **operand 1 supplies the `IN` end of
every shared wire, operand 2 the `OUT` end**. For a fermionic provider the two ends of a
wire differ by a Koszul sign, so this is a rule and not a convention.
[Contraction](contraction.md) has it in full.

## Moving legs around

| Goal | Canonical API | Level |
| --- | --- | --- |
| Reorder axes | `tenet.transpose` | common |
| Move a leg between domain and codomain | `tenet.repartition` (method: `tenet.SymmetricTensor.repartition`) | map |
| Bend one line | `tenet.bend` | categorical |
| Group legs into one | `tenet.fuse` / `tenet.unfuse` | map |
| Exchange two legs with the braid coefficient | `tenet.braid` | categorical |
| Apply the topological twist | `tenet.twist` | categorical |
| Reverse a leg's `dual` flag | `tenet.flip_dual` | categorical |
| Complex conjugate, legs unchanged | `tenet.conj` | common |
| Hermitian adjoint, $D \leftrightarrow C$ | `tenet.adjoint` | map |

`transpose` reorders axes and never changes which side a leg is on; `repartition` is the
one that does, and on a fermionic or anyonic provider it costs
[`BendingCoefficients`][tenet.symmetry.BendingCoefficients].

## Decomposing and truncating

| Goal | Canonical API | Level |
| --- | --- | --- |
| Exact SVD of a map | `tenet.linalg.svd` | map |
| SVD *and* choose the bond in one call | `tenet.linalg.svd_truncated` | common |
| Choose a bond and keep the record | `tenet.linalg.select_bond` → `tenet.linalg.BondSelection` | common |
| Project onto an already-chosen bond | `tenet.linalg.svd(..., bond=...)` | map |
| Self-adjoint version of the two above | `tenet.linalg.eigh` / `tenet.linalg.eigh_truncated` | map |
| Orthogonal factorization | `tenet.linalg.qr` / `tenet.linalg.lq` | map |
| Polar decomposition | `tenet.linalg.polar` | map |
| Non-Hermitian spectrum | `tenet.linalg.eig` / `tenet.linalg.eigvals` | map |
| Matrix exponential | `tenet.linalg.expm` | map |
| Kernel / cokernel isometry | `tenet.linalg.left_null` / `tenet.linalg.right_null` | low-level |

`tenet.linalg` is a re-exported attribute of `tenet`, not an importable module path:
`import tenet` then `tenet.linalg.svd`, or `from tenet.ops.linalg import svd`.

`svd_truncated` is structure-changing — it reads singular *values* to decide which
sectors survive — so it raises
[`StructureChangingError`][tenet.symmetry.StructureChangingError] under `jax.jit`,
`jax.grad` or `jax.vmap`. The pair `select_bond` (outside) + `svd(..., bond=)` (inside)
is the traceable spelling. [Truncation](truncation.md) is the whole story, including what
`max_bond` bounds under a non-Abelian symmetry.

## Reading numbers out, and putting them in

| Goal | Canonical API | Level |
| --- | --- | --- |
| Dense array in the carrier basis | `tenet.SymmetricTensor.to_dense` | common |
| One matrix per coupled sector | `tenet.to_matrices` / `tenet.from_matrices` | map |
| Elementwise map over every block | `tenet.apply_blocks` | low-level |
| Elementwise map over two aligned tensors | `tenet.zip_blocks` | low-level |
| Diagonal of a square map | `tenet.map_diagonal` | low-level |
| $f(A)$ blockwise, $f = \sqrt{\cdot}$ or a power | `tenet.block_sqrt` / `tenet.block_power` | low-level |
| Flat parameter list, backend-agnostic | `tenet.SymmetricTensor.get_params` / `tenet.SymmetricTensor.set_params` | common |
| Persist one tensor | `tenet.save` / `tenet.load` | common |

`apply_blocks` and `zip_blocks` work in **coefficient space**: the function sees reduced
blocks, not dense entries, so a nonlinear function of a tensor is a nonlinear function of
its coefficients. That is usually not what a physical formula means — reach for them
knowing this.

## Changing the symmetry

| Goal | Canonical API | Level |
| --- | --- | --- |
| Restrict a tensor to a subgroup | `tenet.to_symmetry` | categorical |
| Enlarge a space's sectors, keep the values | `tenet.embed` / `tenet.restrict` | categorical |
| Stack two tensors on a direct-sum leg | `tenet.direct_sum` | map |
| Does this provider support X? | `tenet.symmetry.supports` / `tenet.symmetry.requires` | categorical |
| The shipped providers | `tenet.symmetry.U1`, `tenet.symmetry.SU2`, `tenet.symmetry.Z2`, `tenet.symmetry.fZ2`, `tenet.symmetry.Trivial`, `tenet.symmetry.ProductProvider`, `tenet.symmetry.sun.SUNProvider` | common |

A [`CapabilityError`][tenet.symmetry.CapabilityError] is a *categorical refusal*: the
operation has no meaning for the symmetry as declared. It is never a missing feature.

## Networks and algorithms — `tenet.network`

These are deliberately **not** flattened into `tenet`: `dmrg_` is not a tensor operation,
and a top-level spelling of it would read like one.

| Goal | Canonical API | Level |
| --- | --- | --- |
| A matrix product state | `tenet.network.MPS.product`, `tenet.network.MPS.random` | common |
| A Hamiltonian from a term list | `tenet.network.MPO.from_terms` | common |
| …from operator patterns and index arrays | `tenet.network.MPO.from_arrays` | common |
| …from a hand-written `W`, sparse or dense | `tenet.network.MPO.from_entries` / `tenet.network.MPO.from_w` | common |
| Apply an operator exactly | `tenet.network.MPO.apply` | common |
| Truncate a state | `tenet.network.MPS.compress_` | common |
| Ground state search | `tenet.network.dmrg_` (schedule: `tenet.network.Sweep`, result: `tenet.network.DMRG_out`) | common |
| Is it converged, or only plateaued? | `tenet.network.MPO.variance` | common |
| One-site / two-site expectation values | `tenet.network.expectation_1site`, `tenet.network.expectation_2site` | common |
| Every site in one pass | `tenet.network.expectation_profile` | common |
| Overlaps and matrix elements | `tenet.network.overlap`, `tenet.network.measure_mpo` | common |
| Correlators, entanglement entropy | `tenet.network.correlation_function`, `tenet.network.entropy` | common |
| Imaginary/real time evolution | `tenet.network.evolution_step_` (gates: `tenet.network.Gate`, `tenet.network.gates_nn`) | common |
| A 2D state on a lattice | `tenet.network.Peps` + `tenet.network.SquareLattice` / `tenet.network.CheckerboardLattice` / `tenet.network.RectangularUnitcell` | common |
| CTM environment, directional | `tenet.network.EnvCTM` | common |
| CTM environment, C4v (one corner, one edge) | `tenet.network.EnvCTMc4v` | common |
| Converge an environment | `tenet.network.EnvCTM.iterate_` | common |
| One environment sweep, traceable at a fixed bond | `tenet.network.EnvCTM.update_` | common |
| Neighbourhood environment for a full update | `tenet.network.EnvNTU` | common |
| MPS/MPO environment cache | `tenet.network.Env`, `tenet.network.Env.heff2` | low-level |
| Corner and edge primitives | `tenet.network.cor_tl`, `tenet.network.edge_t`, `tenet.network.append_vec_tl`, … | low-level |
| Projector construction | `tenet.network.proj_corners`, `tenet.network.corner2x2` | low-level |
| Krylov ground eigenpair | `tenet.network.lanczos` | low-level |
| One canonicalization / truncation sweep | `tenet.network.sweep_`, `tenet.network.truncate_` | low-level |
| Reverse every leg direction | `tenet.network.flip` | low-level |
| Singular values off a diagonal tensor | `tenet.network.spectrum` | low-level |

## The mutation convention

A trailing underscore means **the call mutates its first argument in place** and any
returned value is a *record of what happened*, not a new object:

```python
out = dmrg_(psi, h, chi=64)       # psi is mutated; out.psi is psi
assert out.psi is psi
env.iterate_(max_bond=24)         # env is mutated; the return is a CTMRG_out record
```

`dmrg_`, `sweep_`, `truncate_`, `compress_`, `iterate_`, `update_`, `evolution_step_`.
Everything without the underscore returns a new object and leaves its inputs alone —
including every operation in the tensor layer, where `SymmetricTensor`, `Leg`,
`GradedSpace` and `TensorStructure` are all immutable.

## JAX

| Goal | Canonical API | Level |
| --- | --- | --- |
| Make tensors JAX pytrees | `tenet.enable_jax` | common |
| …plus the broadened SVD/eigh VJPs | `tenet.enable_jax(ad=True)` | common |
| Tune the broadening, or undo it | `tenet.ad.install`, `tenet.ad.uninstall` | low-level |
| Move blocks to another backend | `tenet.SymmetricTensor.to_backend` | common |

`enable_jax()` registers *our* type with JAX and touches nothing else. `ad=True` is
**process-global for the JAX backend** — it rebinds `linalg.svd` in autoray's dispatch
table, so every library in the process gets the broadened VJP. That is why it is opted
into by name.

Nothing in `tenet.network` is differentiable as a whole. `dmrg_` and `iterate_` re-decide
bond structure from measured values every sweep and cannot run under a trace; the
traceable pieces are the ones that take an already-decided bond —
[`EnvCTMc4v.update_`][tenet.network.EnvCTMc4v.update_] with `bond=`, and
[`Env.heff2`][tenet.network.Env.heff2] with an injected `compile=`.
[JAX and backends](jax-and-backends.md) has the rule.

## Where next

- [Tensors, legs and spaces](tensors-legs-spaces.md) — the four objects everything is
  built from.
- [Contraction](contraction.md) — what makes two legs contractible.
- [The `tenet` API page](../api/tenet.md) — every signature, every refusal.
