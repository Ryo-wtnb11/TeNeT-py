# TeNeT-py

**Symmetric tensor networks in pure Python with backend-native multidimensional arrays.**

`TeNeT-py` is a pure-Python library for symmetry-aware tensor networks with explicit tensor-map semantics and support for non-Abelian fusion structures.

The design combines three ideas:

- **TeNeT-style categorical semantics**  
  A tensor is a linear map with explicit domain and codomain. Duality, fusion trees, recoupling, and braiding are part of the tensor semantics.

- **symmray-style array storage**  
  Reduced blocks are ordinary multidimensional arrays supplied by an existing numerical backend. `TeNeT-py` does not implement its own ndarray or packed storage format.

- **autoray-based backend dispatch**  
  Dense numerical operations are dispatched through `autoray`, allowing the same symmetric tensor structure to use NumPy, JAX, PyTorch, and other compatible array backends.

JAX is intended to be a first-class backend for automatic differentiation, JIT compilation, and vectorization, but `jax.Array` is not part of the core tensor model.

> **Status:** early design and implementation. The API is not stable.

---

## Motivation

A symmetric tensor is more than a block-sparse ndarray.

The central object in `TeNeT-py` is a tensor map

```math
T \in \mathrm{Hom}(D,C),
```

where

```math
C = C_1 \otimes \cdots \otimes C_m,
\qquad
D = D_1 \otimes \cdots \otimes D_n.
```

The distinction between domain and codomain is part of the mathematical structure.

Using rigidity,

```math
\mathrm{Hom}(D,C)
\simeq
C \otimes D^*.
```

A domain leg therefore corresponds to a dual object when the tensor map is viewed as an oriented tensor.

This distinction matters because:

- moving a leg between domain and codomain is not generally an ndarray `moveaxis`;
- permuting tensor legs is not generally an ndarray `transpose`;
- changing fusion-tree basis may require an $F$-move;
- exchanging braided objects may require an $R$-move;
- non-Abelian fusion can have multiple allowed intermediate sectors;
- fusion multiplicities can introduce additional fusion channels.

At the same time, once all categorical data are fixed, the remaining variational degrees of freedom are ordinary dense multidimensional tensors.

`TeNeT-py` therefore separates **categorical structure** from **numerical arrays**.

```text
                           TensorMap
                              │
                       T ∈ Hom(D, C)
                              │
             ┌────────────────┴────────────────┐
             │                                 │
      categorical structure              numerical data
             │                                 │
        GradedSpace                     backend ndarray
        ProductSpace                    reduced blocks
        FusionTree                            │
        BlockKey                              │
        duality                               │
        fusion                                │
        F / R data                            │
             │                                 │
             └────────── operations ───────────┘
                              │
                           autoray
                              │
                 ┌────────────┼────────────┐
                 │            │            │
               NumPy         JAX        PyTorch
```

The basic rule is:

> **Categorical indices are structural metadata. Reduced numerical indices are ndarray axes.**

---

## Design goals

`TeNeT-py` aims to provide:

- explicit domain/codomain `TensorMap` semantics;
- Abelian and non-Abelian symmetries;
- fusion-tree-indexed reduced tensors;
- multidimensional backend-native reduced blocks;
- symmetry-provider-based algorithms;
- backend dispatch through `autoray`;
- JAX differentiation and compilation where appropriate;
- a clean separation between mathematical structure and numerical execution.

The initial implementation does **not** aim to provide:

- a Python port of the Rust TeNeT runtime;
- a custom ndarray implementation;
- manually packed tensor storage;
- explicit buffer offsets and strides;
- custom CUDA memory management;
- low-level kernel scheduling;
- transparent substitution of `TensorMap` for arbitrary NumPy/JAX arrays.

---

## Architecture

The intended architecture is:

```text
                  categorical layer
                         │
                         │
             ┌───────────┴───────────┐
             │                       │
        FusionProvider           GradedSpace
             │                       │
       fusion / dual             ProductSpace
       F / R symbols                 │
             │                       │
             └───────────┬───────────┘
                         │
                     FusionTree
                         │
                      BlockKey
                         │
                         ▼
                      TensorMap
                         │
                    reduced blocks
                         │
                         ▼
                 backend-native arrays
                         │
                         ▼
                      autoray
                         │
          ┌──────────────┼──────────────┐
          │              │              │
        NumPy           JAX          PyTorch
```

The symmetry layer never owns dense numerical storage.

The array backend never determines categorical meaning.

---

# Tensor spaces

## `GradedSpace`

A symmetry-graded space has the form

```math
V
=
\bigoplus_a
\mathbb{C}^{m_a}
\otimes
V_a,
```

where:

- $a$ is a symmetry sector;
- $V_a$ is the corresponding irreducible representation;
- $m_a$ is the degeneracy of that sector.

For example, an SU(2)-graded space may be

```math
V
=
2 V_{1/2}
\oplus
3 V_1.
```

Its reduced degeneracies are

```math
m_{1/2}=2,
\qquad
m_1=3.
```

A possible Python representation is:

```python
V = GradedSpace(
    provider=SU2,
    sectors={
        SU2Sector(two_j=1): 2,
        SU2Sector(two_j=2): 3,
    },
)
```

The degeneracy $m_j$ is distinct from the physical irrep dimension.

For SU(2),

```math
\dim(V_j)=2j+1,
```

but the corresponding reduced tensor axis has size $m_j$.

---

## `ProductSpace`

The domain and codomain of a tensor map are ordered tensor products.

```python
C = ProductSpace(C1, C2, C3)
D = ProductSpace(D1, D2)
```

This represents

```math
C=C_1\otimes C_2\otimes C_3,
```

and

```math
D=D_1\otimes D_2.
```

The ordering is part of the structure.

In general,

```math
C_1\otimes C_2
```

must not be silently identified with

```math
C_2\otimes C_1.
```

The two spaces may be related by a categorical isomorphism, but applying that isomorphism is an operation.

For braided categories that operation can be nontrivial.

---

# `TensorMap`

The main tensor object represents

```math
T:
D_1\otimes\cdots\otimes D_n
\longrightarrow
C_1\otimes\cdots\otimes C_m.
```

The public representation keeps domain and codomain explicit:

```python
T = TensorMap(
    codomain=(C1, C2, C3),
    domain=(D1, D2),
    blocks=blocks,
)
```

Conceptually:

```text
             C1    C2    C3
              ↑     ↑     ↑
              │     │     │
              └─────T─────┘
                    ↑
                 D1   D2
```

The same tensor can internally be viewed through the oriented boundary

```math
(C_1,C_2,C_3,D_1^*,D_2^*).
```

```text
TensorMap representation

codomain = (C1, C2, C3)
domain   = (D1, D2)

                 │
                 ▼

oriented categorical representation

(C1, C2, C3, D1*, D2*)
```

The public API should retain domain/codomain semantics instead of reducing everything to a single list of oriented ndarray axes.

---

# Reduced tensor representation

## Backend-native multidimensional blocks

The main storage decision is:

> **One logical reduced block is one ordinary multidimensional backend array.**

`TeNeT-py` does not initially flatten all blocks into one custom buffer.

Consider

```math
T:
D_1\otimes D_2
\longrightarrow
C_1\otimes C_2\otimes C_3.
```

For fixed external sectors

```math
(c_1,c_2,c_3;d_1,d_2),
```

the corresponding reduced block has shape

```math
(
m_{c_1},
m_{c_2},
m_{c_3},
m_{d_1},
m_{d_2}
).
```

For example:

```python
block.shape == (2, 4, 3, 8, 5)
```

represents

```math
A_{\alpha_1\alpha_2\alpha_3\beta_1\beta_2}.
```

The array axes correspond to reduced degeneracy spaces.

For the initial representation,

```python
block.ndim == len(T.codomain) + len(T.domain)
```

for every reduced block.

The actual block object may be:

```text
numpy.ndarray
jax.Array
torch.Tensor
...
```

as long as the backend is supported through the numerical dispatch layer.

---

## Why not packed storage?

The Rust TeNeT implementation can benefit from a representation of the form

```text
one contiguous buffer
+
block offsets
+
block shapes
+
block strides
```

because a native runtime may want direct control over allocation, memory placement, BLAS calls, GPU kernels, and scheduling.

`TeNeT-py` deliberately does not begin with that abstraction.

Instead:

```text
TensorMap
│
├── BlockKey A → ndarray(shape=...)
├── BlockKey B → ndarray(shape=...)
├── BlockKey C → ndarray(shape=...)
└── ...
```

The numerical backend owns:

- memory allocation;
- physical data layout;
- strides;
- device placement;
- dense kernels.

`TeNeT-py` owns:

- sectors;
- spaces;
- fusion structure;
- block identities;
- duality;
- categorical transformations.

This avoids implementing another ndarray layer in Python.

---

## Logical storage versus execution layout

The logical representation does not prevent later optimization.

For example:

```text
logical blocks

A → ndarray(shape=(4, 8, 16))
B → ndarray(shape=(4, 8, 16))
C → ndarray(shape=(4, 8, 16))
```

may later be lowered temporarily to

```text
execution bucket

ndarray(shape=(3, 4, 8, 16))
               ↑
            block axis
```

for a batched operation.

```text
               logical TensorMap
                      │
                      ▼
              execution lowering
                      │
          ┌───────────┼───────────┐
          │           │           │
        stack       reshape    transpose
          │           │           │
          └───────────┼───────────┘
                      │
                 batched kernel
                      │
                      ▼
               logical TensorMap
```

Thus:

```text
logical representation
        !=
execution representation
```

Storage optimization should not leak into the public mathematical API.

---

# Fusion trees

For Abelian symmetry, a tuple of external charges can often determine the fusion channel uniquely.

For example, in U(1),

```math
q_1 \otimes q_2
\longrightarrow
q_1+q_2.
```

Non-Abelian fusion is different.

For SU(2),

```math
j_1\otimes j_2
```

may contain several sectors.

For three factors, a left-associated fusion tree is

```text
j1       j2       j3
 \       /
  \     /
    j12
      \
       \
        J
```

The intermediate sector $j_{12}$ is part of the basis.

Therefore the external sectors

```math
(j_1,j_2,j_3)
```

and final sector $J$ are not sufficient to identify a reduced component.

`TeNeT-py` represents this structure explicitly.

```python
tree = FusionTree(
    uncoupled=(j1, j2, j3),
    intermediate=(j12,),
    coupled=J,
)
```

The initial implementation should use one canonical tree convention, such as left association.

Other tree bases are reached through explicit recoupling transformations rather than by introducing arbitrary tree topology immediately.

---

## Fusion multiplicity

For a general fusion rule,

```math
a\otimes b
=
\bigoplus_c N_{ab}^{c}\,c.
```

If

```math
N_{ab}^{c}>1,
```

multiple independent fusion vertices connect the same sectors.

`FusionTree` should therefore be capable of carrying fusion-multiplicity labels at its internal vertices.

These labels are **categorical basis data**, not external degeneracy axes.

The initial design therefore treats fusion multiplicity as part of the fusion-tree structure.

Schematically:

```python
FusionTree(
    uncoupled=(a, b, c),
    intermediate=(x,),
    multiplicities=(mu1, mu2),
    coupled=y,
)
```

The exact public representation can evolve, but fusion multiplicity should not be confused with the degeneracy dimensions stored in `GradedSpace`.

---

# Block keys

A tensor map contains a fusion tree for the codomain and a fusion tree for the domain.

A reduced block is therefore indexed by

```python
BlockKey(
    codomain_tree=...,
    domain_tree=...,
)
```

The two trees must have compatible coupled sectors.

For an invariant tensor map:

```python
key.codomain_tree.coupled == key.domain_tree.coupled
```

Conceptually:

```text
       codomain tree                      domain tree

C1         C2        C3                 D1         D2
 \         /                             \         /
  \       /                               \       /
     c12                                     d12
       \                                     /
        \                                   /
         └──────── coupled sector ─────────┘
```

The canonical storage model is:

```python
Mapping[BlockKey, Array]
```

where `Array` is a backend-native multidimensional tensor.

---

# Three kinds of indices

A central design rule is to keep three kinds of indices separate.

## 1. Categorical indices

Examples:

- U(1) charge $q$;
- SU(2) spin $j$;
- intermediate fusion sectors;
- fusion multiplicity labels;
- fusion-tree basis.

These belong to structural Python objects.

They are not ndarray axes.

---

## 2. Reduced degeneracy indices

For a sector $a$ with degeneracy $m_a$,

```math
\alpha=1,\ldots,m_a
```

is a numerical index.

These indices become ndarray axes.

For example:

```math
A_{\alpha\beta\gamma}
```

is stored as:

```python
A.shape == (m_a, m_b, m_c)
```

---

## 3. Irrep basis indices

An SU(2) irrep $j$ has basis indices

```math
m=-j,-j+1,\ldots,j.
```

These indices are encoded in symmetry tensors such as Clebsch-Gordan coefficients.

They are not explicitly stored in the reduced block.

```text
full physical leg
       │
       ├── sector label
       │      └── categorical metadata
       │
       ├── degeneracy index
       │      └── ndarray axis
       │
       └── irrep basis index
              └── encoded by symmetry tensors
```

---

# Symmetry providers

Symmetry-specific mathematics should be provided through capabilities rather than central branching.

Avoid:

```python
if symmetry == "u1":
    ...
elif symmetry == "su2":
    ...
```

Instead, tensor algorithms request operations from a provider.

A minimal initial interface may look conceptually like:

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

Later capabilities may include:

```text
fusion multiplicity
F-symbols
R-symbols
evaluation maps
coevaluation maps
pivotal data
Frobenius-Schur data
```

```text
                        tensor algorithms
                               │
                               ▼
                        provider interface
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
           U1                 SU2             Fibonacci
```

The provider layer is independent of the ndarray backend.

---

## Initial providers

### Trivial symmetry

There is one sector.

This provider is useful for verifying that `TensorMap` reduces naturally to ordinary dense tensor algebra.

---

### U(1)

The tensor unit is charge $0$.

Duality is

```math
q^*=-q.
```

Fusion is

```math
q_1\otimes q_2
=
q_1+q_2.
```

---

### SU(2)

Represent spin using twice-spin integers:

```python
SU2Sector(two_j=0)  # j = 0
SU2Sector(two_j=1)  # j = 1/2
SU2Sector(two_j=2)  # j = 1
```

This avoids floating-point sector labels.

The fusion outcomes satisfy

```math
|j_1-j_2|
\leq
j
\leq
j_1+j_2,
```

with unit spacing in $j$.

Equivalently, in twice-spin representation the allowed values differ by two.

SU(2) should be supported from the first structural milestone so that the design is genuinely non-Abelian from the beginning.

---

# Duality

Duality is part of the categorical structure rather than array storage.

A graded space

```math
V
=
\bigoplus_a
\mathbb{C}^{m_a}\otimes V_a
```

has dual sectors $a^*$ with the corresponding degeneracies.

A tensor map

```math
T\in\mathrm{Hom}(D,C)
```

can be viewed as an oriented tensor in

```math
C\otimes D^*.
```

```text
TensorMap

       C1       C2
        ↑        ↑
        │        │
        └─── T ──┘
             ↑
             D

oriented representation

       C1   C2   D*
```

Therefore moving a leg from codomain to domain must not be defined as only an ndarray axis move.

For a general rigid category, it may involve nontrivial categorical transformations.

---

# Array operations versus categorical operations

The following concepts must remain distinct:

```text
ndarray transpose
categorical permutation
F-move
braiding
adjoint
moving a leg between domain and codomain
```

For a particular reduced block, a categorical operation may eventually lower to operations such as:

```python
ar.do("transpose", block, axes)
```

but the symmetry layer decides whether additional coefficients or changes of fusion-tree basis are required.

For example, a recoupling transformation may have the schematic form

```math
B_{\tau'}
=
\sum_{\tau}
F_{\tau'\tau} A_{\tau}.
```

Here $\tau$ and $\tau'$ label fusion-tree channels.

The dense backend only performs the linear combination.

The categorical layer determines its meaning.

---

# Array backend

## `autoray`

`TeNeT-py` uses `autoray` as the numerical dispatch layer.

The core implementation should avoid explicit backend branching such as:

```python
if isinstance(x, np.ndarray):
    ...
elif isinstance(x, jax.Array):
    ...
elif isinstance(x, torch.Tensor):
    ...
```

Instead:

```python
import autoray as ar

ar.do("reshape", x, shape)
ar.do("transpose", x, axes)
ar.do("conj", x)
ar.do("matmul", x, y)
ar.do("tensordot", x, y, axes=axes)
```

The purpose of the internal array layer is not to wrap all of NumPy.

It should expose only the small set of dense primitives required by tensor operations.

```text
              categorical operation
                        │
                        ▼
               numerical lowering
                        │
            ┌───────────┼───────────┐
            │           │           │
       transpose     matmul      tensordot
            │           │           │
            └───────────┼───────────┘
                        │
                     autoray
                        │
           ┌────────────┼────────────┐
           │            │            │
         NumPy         JAX        PyTorch
```

---

## Backend consistency

One `TensorMap` should normally use one array backend.

For example:

```python
all NumPy blocks
```

or

```python
all JAX blocks
```

rather than a mixture.

Backend changes should be explicit:

```python
T_jax = T.to_backend("jax")
T_numpy = T.to_backend("numpy")
```

This matters because changing backend may involve device transfer or synchronization.

---

# NumPy

NumPy should serve as the reference backend.

Core mathematical operations should be testable without JAX.

This gives:

- simple reference implementations;
- easier debugging;
- deterministic structural tests;
- separation of categorical correctness from compiler behavior.

---

# JAX

JAX is a first-class backend, but it does not define the core tensor representation.

For a JAX-backed tensor:

```python
blocks[key] = jax.Array(...)
```

the categorical structure remains ordinary immutable Python metadata.

The intended PyTree model is:

```text
TensorMap
│
├── static structure
│   ├── domain
│   ├── codomain
│   ├── symmetry provider
│   ├── fusion trees
│   └── block keys
│
└── dynamic leaves
    ├── jax.Array
    ├── jax.Array
    └── ...
```

Conceptually:

```python
children = tuple(
    blocks[key]
    for key in canonical_block_order
)

metadata = TensorMapStructure(
    codomain=codomain,
    domain=domain,
    block_keys=canonical_block_order,
)
```

This allows the reduced numerical data to participate in:

```text
jax.jit
jax.grad
jax.vmap
```

while the sector and fusion structure remains static.

Different block structures may naturally lead to different JAX specializations.

That is expected.

---

# Structural planning and numerical execution

Long-term tensor operations should separate categorical planning from backend execution.

```text
                 TensorMap operation
                        │
                        ▼
                categorical analysis
                        │
            ┌───────────┼───────────┐
            │           │           │
        sector match   F-move     braiding
            │           │           │
            └───────────┼───────────┘
                        │
                        ▼
                 numerical program
                        │
           ┌────────────┼────────────┐
           │            │            │
       transpose      reshape     matmul
           │            │            │
           └────────────┼────────────┘
                        │
                     autoray
                        │
                        ▼
                backend execution
```

This is particularly important for JAX: Python-level fusion-tree logic should not be mixed unnecessarily with compiled numerical kernels.

---

# Contraction

A general contraction

```math
C = A \cdot B
```

should eventually follow:

```text
TensorMap A + TensorMap B
              │
              ▼
    validate contracted spaces
              │
              ▼
      determine compatible
       fusion-tree channels
              │
              ▼
       perform required
       basis transformations
              │
              ▼
     construct output blocks
              │
              ▼
    backend-native contractions
              │
              ▼
          TensorMap C
```

The initial implementation should prioritize correctness and clear semantics over contraction scheduling.

---

# Dense expansion

A future explicit operation

```python
T.to_dense()
```

can reconstruct the full physical tensor.

For SU(2), schematically:

```math
T_{(\alpha_1,m_1)(\alpha_2,m_2)(\alpha_3,m_3)}
=
\sum_{\tau}
A^{(\tau)}_{\alpha_1\alpha_2\alpha_3}
C^{(\tau)}_{m_1m_2m_3}.
```

Here:

- $\alpha_i$ are reduced degeneracy indices;
- $m_i$ are SU(2) irrep basis indices;
- $\tau$ denotes fusion-tree data;
- $A^{(\tau)}$ is the stored reduced tensor;
- $C^{(\tau)}$ contains the corresponding symmetry coefficients.

Physical irrep indices therefore appear only after explicit dense expansion.

Dense expansion is not the canonical storage format.

---

# Initial package structure

```text
TeNeT-py/
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
├── pyproject.toml
└── README.md
```

The `array` layer should remain intentionally small.

Do not build another array framework on top of `autoray`.

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

Suppose

```math
T:
W
\longrightarrow
V\otimes V.
```

After selecting valid fusion-tree pairs, its reduced blocks are ordinary arrays:

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

The same categorical tensor may use JAX arrays:

```python
import jax.numpy as jnp

blocks = {
    key0: jnp.zeros((4, 3, 2)),
    key1: jnp.zeros((3, 4, 2)),
}
```

Only the numerical backend changes.

The `TensorMap` semantics do not.

---

# Relationship to TeNeT

`TeNeT-py` takes several categorical design ideas from the Rust [TeNeT](https://github.com/Ryo-wtnb11/TeNeT) project:

- explicit domain and codomain;
- fusion-tree-based reduced tensors;
- explicit duality;
- provider-based symmetry data;
- separation between tensor semantics and execution.

It intentionally does **not** reproduce the Rust storage/runtime architecture.

In particular, the Python implementation does not initially require:

```text
flat contiguous payloads
block offsets
custom storage traits
manual scratch buffers
native execution plans
CUDA runtime infrastructure
```

These are execution-level choices rather than part of the mathematical `TensorMap` abstraction.

---

# Relationship to symmray

The numerical storage philosophy is inspired by [symmray](https://github.com/jcmgray/symmray):

```text
symmetry structure
       +
block key → backend-native ndarray
```

`TeNeT-py` applies this principle to an explicit fusion-tree-based tensor-map representation.

```text
                          TeNeT-py
                             │
             ┌───────────────┴───────────────┐
             │                               │
     categorical semantics              array model
             │                               │
           TeNeT                          symmray
             │                               │
       domain/codomain               backend ndarrays
       fusion trees                      autoray
       duality
       F / R data
             │                               │
             └───────────────┬───────────────┘
                             │
                          TeNeT-py
```

---

# Roadmap

## Milestone 1 — Structural foundation

Implement:

- immutable sector types;
- `FusionProvider`;
- trivial symmetry;
- U(1);
- SU(2);
- `GradedSpace`;
- `ProductSpace`;
- canonical `FusionTree`;
- fusion multiplicity labels;
- `BlockKey`;
- `TensorMap`;
- backend-native multidimensional reduced blocks;
- minimal `autoray` dispatch;
- NumPy tests;
- JAX PyTree tests.

No general contraction yet.

---

## Milestone 2 — Basic tensor operations

Implement:

- addition;
- scalar multiplication;
- conjugation;
- norm;
- backend conversion;
- structural equality and compatibility checks.

---

## Milestone 3 — Categorical transformations

Add:

- symmetry coefficients;
- $F$-moves;
- $R$-moves where applicable;
- recoupling;
- categorical permutations;
- adjoint;
- moving legs between domain and codomain.

---

## Milestone 4 — Contraction

Implement general symmetric contraction:

```text
categorical analysis
        │
        ▼
fusion-tree matching
        │
        ▼
basis transformations
        │
        ▼
autoray dense contractions
        │
        ▼
output TensorMap
```

---

## Milestone 5 — Factorizations

Implement symmetry-aware:

- QR;
- SVD;
- eigendecomposition;
- truncation;
- reconstruction of graded bond spaces.

Changes of sector content should remain explicit structural operations.

---

## Milestone 6 — Performance

Optimize only after profiling.

Possible optimizations include:

- grouping blocks by shape;
- stacked block execution;
- batched matrix multiplication;
- categorical-plan caching;
- recoupling-transform caching;
- JAX compilation specialization;
- custom kernels where measurements justify them.

These optimizations should not alter the public mathematical representation.

---

# Design principles

### `TensorMap` first, ndarray second

The primary object is

```math
T\in\mathrm{Hom}(D,C),
```

not an ndarray decorated with symmetry labels.

### Reduced blocks remain ordinary arrays

Do not introduce custom dense storage without a measured reason.

### `autoray` handles numerical dispatch

Categorical code should not be tied to NumPy, JAX, or PyTorch.

### JAX is a backend, not the tensor model

JAX integration should build on the same backend-independent `TensorMap` structure.

### Fusion trees are first-class objects

Non-Abelian basis information must remain explicit.

### Categorical indices are not array axes

Sector labels, intermediate sectors, and fusion multiplicities belong to the structural layer.

Reduced degeneracy indices belong to ndarray axes.

### Categorical operations are not ndarray operations

A raw axis transpose and a categorical permutation are different concepts even when they happen to produce the same numerical operation in a simple symmetry.

### Optimization stays below the mathematical API

Packing, bucketing, batching, caching, and kernel scheduling are execution details.

### Correctness before runtime sophistication

The first goal is a coherent and testable mathematical representation.

---

# Project philosophy

```text
                             TeNeT-py
                                │
                         T ∈ Hom(D,C)
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
          structure                              values
             │                                     │
       FusionProvider                       backend ndarray
       GradedSpace                         multidimensional
       ProductSpace                        reduced tensor
       FusionTree                                 │
       BlockKey                                   │
       duality                                    │
       fusion                                     │
       F / R                                      │
             │                                     │
             └────────────── operations ───────────┘
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

The core idea is:

```text
categorical tensor structure in Python
+
reduced tensors as ordinary multidimensional arrays
+
backend dispatch through autoray
```

Everything else should be built on top of that boundary.