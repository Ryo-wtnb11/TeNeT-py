````markdown
# tenet-py

**Symmetric tensor networks in pure Python with backend-native multidimensional arrays.**

`tenet-py` is a pure-Python library for symmetry-aware tensor networks with explicit tensor-map semantics, non-Abelian fusion structure, and reduced tensors stored directly as ordinary multidimensional arrays.

The design combines three main ideas:

- **TeNeT-style categorical tensor semantics**  
  tensors are linear maps with explicit domain and codomain, with non-Abelian structure represented using fusion trees;

- **symmray-style array storage**  
  reduced blocks are ordinary backend-native multidimensional arrays rather than entries in a custom packed storage format;

- **autoray-based backend dispatch**  
  NumPy, JAX, PyTorch, and other compatible array backends can be used without making one numerical framework part of the core tensor semantics.

JAX is intended to be a first-class backend, including support for automatic differentiation and compilation, but `tenet-py` itself is not designed around `jax.Array`.

> **Status:** early design and implementation. APIs are not stable.

---

# Motivation

A symmetric tensor is not simply a block-sparse ndarray.

The central object of `tenet-py` is a tensor map

\[
T\in\operatorname{Hom}(D,C),
\]

with

\[
C=C_1\otimes\cdots\otimes C_m,
\qquad
D=D_1\otimes\cdots\otimes D_n.
\]

The distinction between domain and codomain is part of the mathematical structure.

In a rigid tensor category,

\[
\operatorname{Hom}(D,C)
\simeq
C\otimes D^*,
\]

so domain legs correspond categorically to dual objects.

Consequently:

- moving a leg between domain and codomain is not generally `moveaxis`;
- permuting legs is not generally an ndarray `transpose`;
- changing fusion-tree basis can require nontrivial \(F\)-moves;
- braiding can require \(R\)-symbols;
- non-Abelian fusion may produce several intermediate channels;
- fusion multiplicities can introduce additional internal indices.

At the same time, once the categorical structure has been fixed, the remaining reduced degrees of freedom are ordinary dense multidimensional tensors.

`tenet-py` therefore keeps these layers separate.

```text
                         TensorMap
                            │
                    T ∈ Hom(D, C)
                            │
             ┌──────────────┴──────────────┐
             │                             │
       categorical structure          numerical data
             │                             │
       GradedSpace                  backend ndarray
       ProductSpace                 multidimensional
       FusionTree                  reduced block
       BlockKey                           │
       duality                            │
       F / R data                        │
             │                             │
             └──────── operations ─────────┘
                            │
                        autoray
                            │
             ┌──────────────┼──────────────┐
             │              │              │
           NumPy           JAX          PyTorch
```

The core rule is:

> **Categorical indices are structural metadata. Reduced numerical indices are ndarray axes.**

---

# Design goals

`tenet-py` aims to provide:

- explicit `domain -> codomain` tensor-map semantics;
- Abelian and non-Abelian symmetries;
- fusion-tree-indexed reduced tensors;
- multidimensional reduced blocks;
- backend-independent numerical operations through `autoray`;
- JAX automatic differentiation and compilation where appropriate;
- a clean separation between categorical structure and numerical execution.

It does **not** initially aim to provide:

- a Python port of TeNeT's native runtime;
- a custom ndarray implementation;
- manually packed tensor storage;
- custom GPU memory management;
- low-level CUDA kernel scheduling;
- a tensor object pretending to be an ordinary ndarray in every context.

---

# Core object: `TensorMap`

A tensor is explicitly represented as

\[
T:
D_1\otimes\cdots\otimes D_n
\longrightarrow
C_1\otimes\cdots\otimes C_m.
\]

The public API therefore keeps domain and codomain separate:

```python
T = TensorMap(
    codomain=(C1, C2, C3),
    domain=(D1, D2),
    blocks=blocks,
)
```

Conceptually:

```text
        C1   C2   C3
         ↑    ↑    ↑
         │    │    │
         └────T────┐
                   │
                 D1 D2
```

Internally this may be interpreted as the oriented boundary

\[
(C_1,C_2,C_3,D_1^*,D_2^*),
\]

but the public API retains the tensor-map interpretation.

```text
public:

codomain = (C1, C2, C3)
domain   = (D1, D2)

              ↓

categorical boundary:

(C1, C2, C3, dual(D1), dual(D2))
```

This distinction becomes essential for general duality, fermionic systems, braided tensor categories, and anyonic tensor networks.

---

# Graded spaces

A graded vector space is represented as

\[
V
=
\bigoplus_a
\mathbb C^{m_a}\otimes V_a,
\]

where

- \(a\) is a symmetry sector;
- \(V_a\) is the corresponding irreducible representation;
- \(m_a\) is its degeneracy.

For example, for SU(2),

\[
V
=
2V_{1/2}\oplus3V_1
\]

has

\[
m_{1/2}=2,
\qquad
m_1=3.
\]

Conceptually:

```python
V = GradedSpace(
    provider=SU2,
    sectors={
        SU2Sector(two_j=1): 2,
        SU2Sector(two_j=2): 3,
    },
)
```

The degeneracy dimension \(m_a\) must not be confused with the physical irrep dimension.

For SU(2),

\[
\dim V_j=2j+1,
\]

but a reduced tensor axis corresponding to sector \(j\) has dimension \(m_j\).

---

# Product spaces

Tensor-map domain and codomain are ordered products of graded spaces.

```python
C = ProductSpace(C1, C2, C3)
D = ProductSpace(D1, D2)
```

Ordering is meaningful.

In general,

\[
C_1\otimes C_2
\]

must not be silently identified with

\[
C_2\otimes C_1.
\]

Even in symmetric categories they may only be canonically isomorphic through a categorical permutation, and in braided categories the corresponding map may carry nontrivial data.

---

# Reduced blocks are ordinary ndarrays

The most important storage decision in `tenet-py` is:

\[
\boxed{
\text{one logical reduced block}
=
\text{one multidimensional backend array}
}
\]

The library does not initially introduce a custom flattened tensor payload.

For

\[
T:
D_1\otimes D_2
\rightarrow
C_1\otimes C_2\otimes C_3,
\]

a block with fixed external sectors

\[
(c_1,c_2,c_3;d_1,d_2)
\]

has shape

\[
(
m_{c_1},
m_{c_2},
m_{c_3},
m_{d_1},
m_{d_2}
).
\]

For example:

```python
block.shape == (2, 4, 3, 8, 5)
```

represents

\[
A_{\alpha_1\alpha_2\alpha_3\beta_1\beta_2}.
\]

For multiplicity-free fusion, the basic invariant is

```python
block.ndim == len(T.codomain) + len(T.domain)
```

The block may be any backend-native multidimensional array supported by the numerical layer:

```python
numpy.ndarray
jax.Array
torch.Tensor
...
```

The array backend is not part of the categorical meaning of the tensor.

---

# Why not packed storage?

A low-level native tensor runtime may represent all blocks using

```text
one contiguous buffer
+
offsets
+
shapes
+
strides
```

for precise control over allocation and kernel execution.

Rust TeNeT follows this general style.

`tenet-py` deliberately does not begin there.

Python numerical ecosystems already provide mature multidimensional array abstractions. Reimplementing their storage model would duplicate functionality and tightly couple the symmetry layer to low-level execution details.

Instead:

```text
TensorMap
│
├── BlockKey A → ndarray(...)
├── BlockKey B → ndarray(...)
├── BlockKey C → ndarray(...)
└── ...
```

The backend owns:

- memory allocation;
- multidimensional shape;
- strides/layout;
- device placement;
- dense kernels.

`tenet-py` owns:

- sectors;
- spaces;
- duality;
- fusion trees;
- allowed blocks;
- categorical transformations.

---

# Logical storage versus optimized execution

The canonical representation should remain simple even if optimized execution is introduced later.

For example:

```text
logical representation

A → array(shape=(4, 8, 16))
B → array(shape=(4, 8, 16))
C → array(shape=(4, 8, 16))
```

An optimized backend path may later combine them into

```text
physical execution representation

array(shape=(3, 4, 8, 16))
             ↑
          block batch
```

for batched contractions.

This optimization must remain hidden behind the logical block interface.

Thus:

\[
\boxed{
\text{logical block representation}
\neq
\text{physical execution representation}
}
\]

---

# Fusion trees

For Abelian symmetries, external sector labels are often sufficient to identify a reduced block.

For U(1),

\[
q_1\otimes q_2\to q_1+q_2
\]

is unique.

For non-Abelian symmetries this is no longer true.

Consider

\[
j_1\otimes j_2\otimes j_3.
\]

Using a left-associated fusion tree,

```text
j1     j2      j3
 \     /
  \   /
   j12
      \
       \
        J
```

different values of \(j_{12}\) correspond to different fusion channels.

The external sectors

\[
(j_1,j_2,j_3)
\]

and final sector \(J\) therefore do not uniquely determine the reduced tensor.

`tenet-py` treats the fusion tree as explicit structural metadata.

Conceptually:

```python
FusionTree(
    uncoupled=(j1, j2, j3),
    intermediate=(j12,),
    coupled=J,
)
```

The initial implementation uses one canonical tree convention, preferably left-associated.

General fusion-tree topologies can be introduced later.

---

# Block keys

A `TensorMap` has fusion structure on both codomain and domain.

A reduced block is therefore indexed by a pair

```python
BlockKey(
    codomain_tree=...,
    domain_tree=...,
)
```

with matching coupled sector.

```text
codomain tree                    domain tree

C1    C2    C3                    D1    D2
 \    /                            \    /
  c12                               d12
     \                              /
      \                            /
       └──── coupled sector ──────┘
```

For an invariant tensor map,

```python
key.codomain_tree.coupled == key.domain_tree.coupled
```

The canonical block store is conceptually

```python
Mapping[BlockKey, Array]
```

where each value is an ordinary multidimensional backend array.

---

# Three kinds of indices

`tenet-py` keeps three kinds of indices conceptually separate.

## Structural categorical indices

Examples:

- U(1) charge;
- SU(2) spin;
- intermediate fusion channels;
- fusion-tree topology;
- total coupled sector.

These are Python objects and metadata.

They are not ndarray dimensions.

---

## Reduced numerical indices

These represent degeneracy spaces such as

\[
\alpha=1,\ldots,m_a.
\]

These are ndarray dimensions.

A reduced block

\[
A_{\alpha\beta\gamma}
\]

is therefore naturally represented as

```python
array.shape == (m_a, m_b, m_c)
```

---

## Physical irrep indices

For example SU(2) has magnetic quantum numbers

\[
m=-j,-j+1,\ldots,j.
\]

These indices belong to symmetry tensors such as Clebsch–Gordan coefficients and are not stored explicitly in the reduced representation.

Schematically:

```text
full physical index
       │
       ├── sector label         → structural metadata
       ├── degeneracy index     → ndarray axis
       └── irrep basis index    → absorbed into symmetry tensors
```

---

# Fusion multiplicities

For a general fusion category,

\[
a\otimes b
=
\bigoplus_c N_{ab}^{c}\,c.
\]

When

\[
N_{ab}^{c}>1,
\]

a fusion vertex carries an additional multiplicity label.

The exact storage convention for multiplicity indices is intentionally left open during the first implementation.

Two natural possibilities are:

1. treat multiplicity labels as part of the fusion-tree structural key;
2. represent multiplicity basis dimensions as explicit reduced ndarray axes.

The choice should be made based on the transformation and contraction semantics rather than storage convenience.

The architecture must support either without redesigning `TensorMap`.

---

# Backend model

`tenet-py` uses `autoray` for backend dispatch.

The symmetry layer should not contain code such as

```python
if isinstance(x, np.ndarray):
    ...
elif isinstance(x, jax.Array):
    ...
elif isinstance(x, torch.Tensor):
    ...
```

Instead numerical operations are dispatched through a small array layer built on `autoray`.

Conceptually:

```python
import autoray as ar

ar.do("reshape", x, shape)
ar.do("transpose", x, axes)
ar.do("conj", x)
ar.do("tensordot", x, y, axes=...)
```

The goal is not to wrap every array operation.

Only the numerical primitives required by symmetric tensor algorithms should be exposed.

```text
TensorMap algorithms
        │
        ↓
categorical planning
        │
        ↓
small numerical interface
        │
        ↓
      autoray
        │
 ┌──────┼───────┐
 │      │       │
NumPy   JAX   PyTorch
```

---

# Why autoray?

Using `autoray` keeps two important concerns separate.

The tensor structure asks:

> Which blocks exist, and what categorical transformation relates them?

The backend asks:

> How should a dense multidimensional operation on those blocks be executed?

For example, a categorical operation may determine that

```text
block B
=
F00 * transpose(block A0)
+
F01 * transpose(block A1)
```

The structural layer determines:

- which blocks participate;
- which \(F\)-coefficients are required;
- which axes correspond.

The numerical layer only executes:

```python
transpose
multiply
add
```

using the backend of the block arrays.

---

# Backend consistency

A `TensorMap` should normally use one numerical backend consistently.

For example, all blocks should be NumPy arrays or all should be JAX arrays.

The library should validate backend compatibility rather than silently moving data between frameworks.

Explicit conversion may eventually be provided:

```python
T_jax = T.to_backend("jax")
T_numpy = T.to_backend("numpy")
```

Such operations should be explicit because they may imply device transfer or host synchronization.

---

# JAX as a first-class backend

Although the core representation is backend-independent, JAX is an important target.

For a JAX-backed `TensorMap`:

```python
blocks[key] = jax.Array(...)
```

the library should support JAX transformations where mathematically appropriate.

The intended model is:

```text
TensorMap
│
├── static structure
│   ├── domain
│   ├── codomain
│   ├── sectors
│   ├── fusion trees
│   └── BlockKeys
│
└── dynamic numerical leaves
    ├── jax.Array
    ├── jax.Array
    └── jax.Array
```

`TensorMap` can therefore be registered as a custom PyTree.

Conceptually:

```python
children = tuple(
    blocks[key]
    for key in canonical_block_order
)

aux_data = TensorMapStructure(
    codomain=codomain,
    domain=domain,
    block_keys=canonical_block_order,
)
```

The structural objects must therefore be immutable and hashable where JAX requires static metadata.

---

# JAX specialization

Different symmetry structures naturally produce different numerical programs.

Two tensors may differ in:

- sector content;
- fusion trees;
- reduced block shapes;
- domain/codomain;
- fusion multiplicities.

These may therefore correspond to different JAX compilations.

This is intentional.

```text
static categorical structure
           │
           ↓
     JAX specialization
           │
           ↓
      compiled program
           ↑
           │
 dynamic reduced arrays
```

The guiding interpretation is

\[
\boxed{
\text{categorical structure}
\approx
\text{static shape/type information}
}
\]

for a compiled JAX computation.

---

# NumPy as the reference backend

NumPy should remain the simplest reference backend.

Core algorithms should be testable using plain NumPy without requiring JAX.

This provides:

- easier debugging;
- deterministic reference implementations;
- simpler correctness tests;
- separation of mathematical correctness from compiler behavior.

JAX-specific tests can then verify:

- PyTree flattening;
- `jax.jit`;
- `jax.grad`;
- `jax.vmap`;
- backend consistency.

---

# Symmetry providers

Symmetries are represented using provider objects rather than hard-coded global branching.

Do not design algorithms around

```python
if symmetry == "u1":
    ...
elif symmetry == "su2":
    ...
```

Instead define capabilities such as:

```python
class FusionProvider(Protocol):
    @property
    def unit(self):
        ...

    def dual(self, sector):
        ...

    def fusion(self, a, b):
        ...
```

Future capabilities may include:

```text
fusion multiplicity
F-symbols
R-symbols
evaluation
coevaluation
pivotal data
Frobenius-Schur data
```

Algorithms should depend only on the capabilities they require.

```text
                      algorithms
                          │
                          ↓
                    provider API
                          │
            ┌─────────────┼─────────────┐
            │             │             │
           U1            SU2       Fibonacci
```

The provider layer is independent of the ndarray backend.

---

# Initial symmetry providers

## Trivial symmetry

Used to ensure that the entire abstraction reduces cleanly to ordinary dense tensor algebra.

There is exactly one sector.

---

## U(1)

Use integer charges:

\[
\mathbf 1=0,
\qquad
q^*=-q,
\qquad
q_1\otimes q_2=q_1+q_2.
\]

This provides the simplest nontrivial graded-space test.

---

## SU(2)

Use twice-spin integers:

```python
SU2Sector(two_j=1)  # j = 1/2
SU2Sector(two_j=2)  # j = 1
```

so half-integer representations are stored exactly.

Fusion is

\[
j_1\otimes j_2
=
\bigoplus_{j=|j_1-j_2|}^{j_1+j_2}j.
\]

SU(2) is included from the beginning to ensure that the architecture genuinely supports non-Abelian fusion.

The first milestone does not require CG coefficients or \(F\)-symbols.

---

# Duality

Duality belongs to the categorical structure.

For

\[
V=
\bigoplus_a
\mathbb C^{m_a}\otimes V_a,
\]

the dual is

\[
V^*
=
\bigoplus_a
(\mathbb C^{m_a})^*\otimes V_{a^*}.
\]

A domain factor appears categorically as a dual object in the oriented tensor boundary.

```text
TensorMap

       C1   C2
        ↑    ↑
        │    │
        └─ T ─┐
              │
              D

oriented boundary:

       C1  C2  D*
```

Moving a leg between domain and codomain must therefore not be implemented as merely changing an axis position.

For a general rigid category, it may require nontrivial rigidity transformations.

---

# Array transpose versus categorical permutation

These operations must remain conceptually distinct:

```text
backend array transpose
categorical permutation
braiding
F-move
adjoint
domain/codomain transpose
```

A reduced ndarray can of course be transposed numerically:

```python
ar.do("transpose", block, axes)
```

but the decision that a particular categorical operation corresponds to that transpose, possibly combined with \(F\)- or \(R\)-coefficients, belongs to `tenet-py`.

Thus `TensorMap` should not simply masquerade as an ndarray.

---

# Operation architecture

The long-term operation pipeline is:

```text
             TensorMap operation
                     │
                     ↓
           categorical analysis
                     │
         ┌───────────┼───────────┐
         │           │           │
     sector match   F moves    braiding
         │           │           │
         └───────────┼───────────┘
                     │
                     ↓
             numerical program
                     │
         transpose / reshape
          tensordot / matmul
        multiply / add / stack
                     │
                     ↓
                   autoray
                     │
       ┌─────────────┼─────────────┐
       │             │             │
     NumPy          JAX         PyTorch
```

The categorical layer should determine the numerical operation before backend execution.

---

# Contraction

A future contraction

\[
C=A\cdot B
\]

should conceptually proceed as

```text
TensorMap A
TensorMap B
     │
     ↓
validate contracted spaces
     │
     ↓
match compatible sectors
     │
     ↓
determine fusion-tree transforms
     │
     ↓
construct output BlockKeys
     │
     ↓
backend-native dense contractions
     │
     ↓
TensorMap C
```

The first implementation should prioritize correctness over batching or contraction scheduling.

---

# Dense expansion

A future explicit method

```python
T.to_dense()
```

may reconstruct the full physical tensor.

For SU(2), schematically,

\[
T_{
(\alpha_1m_1)
(\alpha_2m_2)
(\alpha_3m_3)
}
=
\sum_{\tau}
A^{(\tau)}_{\alpha_1\alpha_2\alpha_3}
C^{(\tau)}_{m_1m_2m_3}.
\]

The magnetic indices \(m_i\) exist only in the expanded representation.

Dense expansion is therefore a transformation, not the canonical tensor storage.

---

# Proposed package structure

```text
tenet-py/
│
├── src/
│   └── tenet/
│       ├── __init__.py
│       │
│       ├── symmetry/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── trivial.py
│       │   ├── u1.py
│       │   └── su2.py
│       │
│       ├── array/
│       │   ├── __init__.py
│       │   └── dispatch.py
│       │
│       ├── space.py
│       ├── fusion_tree.py
│       └── tensor_map.py
│
├── tests/
│   ├── test_space.py
│   ├── test_fusion_tree.py
│   ├── test_tensor_map.py
│   ├── test_backends.py
│   └── test_jax.py
│
└── README.md
```

The `array` module should remain intentionally small.

It exists only to centralize the subset of `autoray` functionality required internally.

Do not build a second array framework on top of `autoray`.

---

# Initial API sketch

```python
import numpy as np

from tenet import GradedSpace, TensorMap
from tenet.symmetry import SU2, SU2Sector


half = SU2Sector(two_j=1)
one = SU2Sector(two_j=2)


V = GradedSpace(
    provider=SU2,
    sectors={
        half: 4,
        one: 3,
    },
)

W = GradedSpace(
    provider=SU2,
    sectors={
        half: 2,
    },
)
```

A tensor

\[
T:W\rightarrow V\otimes V
\]

may then be constructed from ordinary arrays:

```python
blocks = {
    key0: np.zeros((4, 3, 2)),
    key1: np.zeros((3, 4, 2)),
}

T = TensorMap(
    codomain=(V, V),
    domain=(W,),
    blocks=blocks,
)
```

The same structural tensor can instead use JAX blocks:

```python
import jax.numpy as jnp

blocks = {
    key0: jnp.zeros((4, 3, 2)),
    key1: jnp.zeros((3, 4, 2)),
}
```

The `TensorMap` abstraction does not change.

---

# Backend conversion

A future explicit backend-conversion API may look like:

```python
T_jax = T.to_backend("jax")
T_numpy = T.to_backend("numpy")
```

This should map each reduced block while preserving all categorical metadata.

Conceptually:

```text
TensorMap[NumPy]
      │
      │ map blocks
      ↓
TensorMap[JAX]

same:
    domain
    codomain
    fusion trees
    BlockKeys

different:
    ndarray backend
```

---

# Relationship to TeNeT

[TeNeT](https://github.com/Ryo-wtnb11/TeNeT) provides several important conceptual foundations:

- explicit domain/codomain tensor maps;
- fusion-tree-based non-Abelian representations;
- duality and categorical orientation;
- provider-based symmetry semantics;
- separation of mathematical structure and execution.

However, `tenet-py` intentionally does not copy the Rust runtime architecture.

In particular, the initial Python implementation does not reproduce:

- contiguous packed tensor payloads;
- explicit block offsets and strides;
- custom storage traits;
- native backend execution plans;
- manual scratch allocation;
- CUDA-specific runtime infrastructure.

Those choices are appropriate for a native HPC library but are not necessary for the initial Python array model.

---

# Relationship to symmray

[symmray](https://github.com/jcmgray/symmray) motivates the numerical storage philosophy:

```text
symmetry metadata
      +
block key → raw backend ndarray
```

rather than introducing a custom dense-storage implementation.

`tenet-py` follows the same broad idea but extends it toward an explicitly categorical non-Abelian tensor-map representation.

Schematically:

```text
                  tenet-py
                     │
       ┌─────────────┴─────────────┐
       │                           │
 categorical semantics        array philosophy
       │                           │
     TeNeT                      symmray
       │                           │
 TensorMap                    backend ndarray
 domain/codomain                autoray
 fusion trees
 duality
 F/R structure
       │                           │
       └─────────────┬─────────────┘
                     │
                  tenet-py
```

---

# Relationship to JAX

JAX is not the tensor storage abstraction of `tenet-py`.

Instead:

\[
\boxed{
\text{JAX is one particularly powerful ndarray backend}
}
\]

This distinction matters because the same tensor structure should remain usable for:

- simple NumPy correctness tests;
- differentiable JAX calculations;
- potentially PyTorch-based workflows.

JAX-specific integration is built on top of the generic tensor representation rather than defining it.

---

# Roadmap

## Milestone 1 — Structural foundation

Implement:

- immutable sectors;
- provider protocol;
- trivial symmetry;
- U(1);
- SU(2) fusion;
- `GradedSpace`;
- `ProductSpace`;
- canonical `FusionTree`;
- `BlockKey`;
- `TensorMap`;
- multidimensional backend-native blocks;
- minimal `autoray` dispatch;
- NumPy tests;
- JAX PyTree tests.

No general contraction yet.

---

## Milestone 2 — Basic array operations

Implement:

- blockwise addition;
- scalar multiplication;
- conjugation;
- norm;
- backend conversion;
- basic structural validation.

Test with at least:

- NumPy;
- JAX.

---

## Milestone 3 — Categorical transformations

Introduce:

- CG data;
- \(F\)-symbols;
- \(R\)-symbols where applicable;
- recoupling;
- categorical permutation;
- adjoint;
- domain/codomain movement.

---

## Milestone 4 — Contraction

Implement general symmetric contraction using

```text
categorical planning
        ↓
block matching
        ↓
fusion-tree transformations
        ↓
autoray dense kernels
        ↓
output TensorMap
```

---

## Milestone 5 — Factorizations

Implement symmetry-aware:

- QR;
- SVD;
- eigendecomposition;
- truncation;
- reconstructed bond spaces.

Structural changes such as truncation should be represented explicitly at the Python level.

---

## Milestone 6 — Performance

Only after profiling:

- shape-based block bucketing;
- stacked block execution;
- batched GEMM;
- operation-plan caching;
- fusion-transform caching;
- JAX compilation specialization;
- custom kernels where justified.

The public `TensorMap` representation should remain independent of these optimizations.

---

# Design principles

## TensorMap first, ndarray second

The object is mathematically

\[
T\in\operatorname{Hom}(D,C),
\]

not merely an ndarray carrying symmetry annotations.

---

## Reduced blocks should remain ordinary arrays

Do not implement custom dense storage unless profiling provides a concrete reason.

---

## Use autoray for numerical dispatch

The symmetry library should not become coupled to NumPy, JAX, or PyTorch implementation details.

---

## JAX is important but not foundational

JAX-specific capabilities should integrate cleanly without making `jax.Array` part of the categorical model.

---

## Structural indices are not ndarray axes

Sectors and fusion channels are metadata.

Degeneracy indices are array dimensions.

---

## Non-Abelian fusion is explicit

Fusion trees are first-class immutable objects.

---

## Optimization must remain below the mathematical API

Packing, batching, kernel selection, and execution planning must not determine the public tensor semantics.

---

## Correctness before runtime sophistication

The first version should establish a mathematically coherent representation before implementing native-style scheduling or memory optimization.

---

# Project philosophy

The complete design can be summarized as:

```text
                         tenet-py
                            │
                    T ∈ Hom(D, C)
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
       structure                           values
          │                                   │
    GradedSpace                         backend ndarray
    ProductSpace                       multidimensional
    FusionTree                         reduced tensor
    BlockKey                                 │
    duality                                  │
    fusion                                   │
    F / R                                    │
          │                                   │
          └──────────── operations ───────────┘
                            │
                         autoray
                            │
              ┌─────────────┼─────────────┐
              │             │             │
            NumPy          JAX         PyTorch
                            │
                 ┌──────────┼──────────┐
                 │          │          │
                jit        grad       vmap
```

The core idea is deliberately simple:

\[
\boxed{
\text{categorical tensor structure in Python}
+
\text{reduced tensors as ordinary ndarrays}
+
\text{backend dispatch through autoray}
}
\]

Everything else should be built on top of that boundary.
````
