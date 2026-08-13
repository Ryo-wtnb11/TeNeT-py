# tenet-py

**Symmetric tensor networks in pure Python with backend-native multidimensional arrays.**

`tenet-py` is a pure-Python library for symmetry-aware tensor networks with explicit tensor-map semantics, non-Abelian fusion structure, and reduced tensors stored directly as ordinary multidimensional arrays.

The design combines three main ideas:

- **TeNeT-style categorical tensor semantics**  
  Tensors are linear maps with explicit domain and codomain, and non-Abelian structure is represented using fusion trees.

- **symmray-style array storage**  
  Reduced blocks are ordinary backend-native multidimensional arrays rather than slices of a custom packed storage format.

- **autoray-based backend dispatch**  
  NumPy, JAX, PyTorch, and other compatible array backends can be used without making one numerical framework part of the core tensor semantics.

JAX is intended to be a first-class backend, including support for automatic differentiation and compilation, but `tenet-py` itself is not designed around `jax.Array`.

> **Status:** early design and implementation. The API is not stable.

---

## Motivation

A symmetric tensor is not simply a block-sparse ndarray.

The central object in `tenet-py` is a tensor map

$$
T \in \operatorname{Hom}(D,C),
$$

where

$$
C = C_1 \otimes \cdots \otimes C_m,
\qquad
D = D_1 \otimes \cdots \otimes D_n.
$$

The distinction between domain and codomain is part of the mathematical structure.

In a rigid tensor category,

$$
\operatorname{Hom}(D,C)
\simeq
C \otimes D^*,
$$

so a domain leg corresponds categorically to a dual object.

Consequently:

- moving a leg between domain and codomain is not generally an ndarray `moveaxis`;
- permuting legs is not generally an ndarray `transpose`;
- changing fusion-tree basis may require nontrivial \(F\)-moves;
- braiding may require \(R\)-symbols;
- non-Abelian fusion may produce several intermediate channels;
- fusion multiplicities may introduce additional internal degrees of freedom.

At the same time, once the categorical structure has been fixed, the remaining reduced degrees of freedom are ordinary dense multidimensional tensors.

`tenet-py` therefore separates categorical structure from numerical array storage.

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
        fusion                             │
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

The basic rule is:

> **Categorical indices are structural metadata. Reduced numerical indices are ndarray axes.**

---

## Design goals

`tenet-py` aims to provide:

- explicit `domain -> codomain` tensor-map semantics;
- Abelian and non-Abelian symmetries;
- fusion-tree-indexed reduced tensors;
- multidimensional reduced blocks;
- backend-independent numerical operations through `autoray`;
- JAX automatic differentiation and compilation where appropriate;
- a clean separation between categorical structure and numerical execution.

The initial implementation does **not** aim to provide:

- a Python port of the TeNeT native runtime;
- a custom ndarray implementation;
- manually packed tensor storage;
- custom GPU memory management;
- low-level CUDA kernel scheduling;
- transparent substitution of `TensorMap` for every NumPy or JAX ndarray operation.

---

## Core object: `TensorMap`

A tensor is explicitly represented as a map

$$
T:
D_1 \otimes \cdots \otimes D_n
\longrightarrow
C_1 \otimes \cdots \otimes C_m.
$$

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

Internally this may be interpreted through the oriented boundary

$$
(C_1, C_2, C_3, D_1^*, D_2^*),
$$

but the public representation remains a tensor map with explicit domain and codomain.

```text
public representation

codomain = (C1, C2, C3)
domain   = (D1, D2)

              ↓

internal oriented boundary

(C1, C2, C3, dual(D1), dual(D2))
```

This distinction becomes important for duality, fermions, braided tensor categories, and anyonic tensor networks.

---

## Graded spaces

A symmetry-graded vector space is represented as

$$
V
=
\bigoplus_a
\mathbb{C}^{m_a}
\otimes
V_a,
$$

where:

- \(a\) is a symmetry sector;
- \(V_a\) is the corresponding irreducible representation space;
- \(m_a\) is its degeneracy.

For example, an SU(2)-graded space

$$
V
=
2 V_{1/2}
\oplus
3 V_1
$$

has degeneracies

$$
m_{1/2} = 2,
\qquad
m_1 = 3.
$$

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

The degeneracy \(m_a\) must not be confused with the physical dimension of the irrep.

For SU(2),

$$
\dim V_j = 2j + 1,
$$

but a reduced tensor axis associated with sector \(j\) has size \(m_j\), not \(2j+1\).

---

## Product spaces

The domain and codomain are ordered products of graded spaces.

```python
C = ProductSpace(C1, C2, C3)
D = ProductSpace(D1, D2)
```

Ordering is meaningful.

In general,

$$
C_1 \otimes C_2
$$

must not be silently identified with

$$
C_2 \otimes C_1.
$$

Even when a canonical isomorphism exists, applying it is a tensor operation rather than a change of notation.

In braided categories, changing the ordering may additionally involve nontrivial braiding data.

---

## Reduced blocks are ordinary ndarrays

The central storage decision in `tenet-py` is:

> **One logical reduced block is one multidimensional backend-native array.**

The library does not initially introduce a custom flattened payload.

For

$$
T:
D_1 \otimes D_2
\longrightarrow
C_1 \otimes C_2 \otimes C_3,
$$

a block with fixed external sectors

$$
(c_1,c_2,c_3;d_1,d_2)
$$

has shape

$$
\left(
m_{c_1},
m_{c_2},
m_{c_3},
m_{d_1},
m_{d_2}
\right).
$$

For example,

```python
block.shape == (2, 4, 3, 8, 5)
```

represents a reduced tensor

$$
A_{\alpha_1 \alpha_2 \alpha_3 \beta_1 \beta_2}.
$$

For multiplicity-free fusion, the basic invariant is

```python
block.ndim == len(T.codomain) + len(T.domain)
```

A block may be any supported backend-native multidimensional array:

```text
numpy.ndarray
jax.Array
torch.Tensor
...
```

The array backend is not part of the categorical meaning of the tensor.

---

## Why not packed storage?

A low-level native tensor runtime may store all reduced blocks using

```text
one contiguous buffer
+
offsets
+
shapes
+
strides
```

to obtain precise control over allocation, memory placement, and kernel scheduling.

The Rust implementation of TeNeT follows this general strategy.

`tenet-py` deliberately starts from a different point.

Python numerical ecosystems already provide mature multidimensional array abstractions. Reimplementing an ndarray-like storage layer would duplicate functionality already provided by NumPy, JAX, PyTorch, and related libraries.

Instead:

```text
TensorMap
│
├── BlockKey A → ndarray(...)
├── BlockKey B → ndarray(...)
├── BlockKey C → ndarray(...)
└── ...
```

The numerical backend owns:

- memory allocation;
- multidimensional shape;
- physical strides and layout;
- device placement;
- dense numerical kernels.

`tenet-py` owns:

- sectors;
- graded spaces;
- duality;
- fusion trees;
- allowed block structure;
- categorical transformations.

---

## Logical storage and optimized execution

The logical representation should remain simple even if optimized execution is introduced later.

For example:

```text
logical representation

A → array(shape=(4, 8, 16))
B → array(shape=(4, 8, 16))
C → array(shape=(4, 8, 16))
```

A later optimized execution path may temporarily represent these blocks as

```text
array(shape=(3, 4, 8, 16))
             ↑
          block batch
```

to perform a batched operation.

This does not change the public tensor representation.

```text
logical TensorMap
      │
      ↓
execution lowering
      │
      ├── stack compatible blocks
      ├── reshape
      ├── transpose
      └── batch dense kernels
      │
      ↓
logical TensorMap
```

Thus the project distinguishes:

```text
logical tensor representation
          ≠
optimized execution representation
```

---

## Fusion trees

For Abelian symmetries, external sector labels are often sufficient to identify a reduced block.

For U(1),

$$
q_1 \otimes q_2
\longrightarrow
q_1 + q_2
$$

has a unique result.

For non-Abelian symmetries this is no longer true.

Consider

$$
j_1 \otimes j_2 \otimes j_3.
$$

With a left-associated fusion tree,

```text
j1     j2      j3
 \     /
  \   /
   j12
      \
       \
        J
```

the intermediate sector \(j_{12}\) may take several allowed values.

Therefore the external sectors

$$
(j_1,j_2,j_3)
$$

and the final sector \(J\) do not uniquely identify a reduced tensor component.

`tenet-py` treats the fusion tree as explicit immutable structural data.

Conceptually:

```python
tree = FusionTree(
    uncoupled=(j1, j2, j3),
    intermediate=(j12,),
    coupled=J,
)
```

The first implementation should use one canonical tree convention, preferably left-associated.

General tree topologies can be added later.

---

## Block keys

A `TensorMap` has fusion structure on both codomain and domain.

A reduced block is therefore indexed by a pair of fusion trees:

```python
key = BlockKey(
    codomain_tree=...,
    domain_tree=...,
)
```

For an invariant tensor map, the coupled sectors must match:

```python
key.codomain_tree.coupled == key.domain_tree.coupled
```

Conceptually:

```text
codomain tree                     domain tree

C1    C2    C3                     D1    D2
 \    /                             \    /
  c12                                d12
     \                               /
      \                             /
       └───── coupled sector ──────┘
```

The canonical block store is therefore

```python
Mapping[BlockKey, Array]
```

where every value is an ordinary multidimensional backend array.

---

## Three kinds of indices

`tenet-py` distinguishes three conceptually different types of indices.

### 1. Structural categorical indices

Examples include:

- U(1) charge;
- SU(2) spin;
- intermediate fusion channels;
- final coupled sector;
- fusion-tree topology.

These are structural Python objects.

They are not ndarray axes.

### 2. Reduced numerical indices

These describe degeneracy spaces such as

$$
\alpha = 1,\ldots,m_a.
$$

These are ndarray axes.

For example,

$$
A_{\alpha\beta\gamma}
$$

is naturally stored as

```python
array.shape == (m_a, m_b, m_c)
```

### 3. Physical irrep indices

For SU(2), these include magnetic quantum numbers

$$
m = -j, -j+1, \ldots, j.
$$

Such indices belong to symmetry tensors such as Clebsch-Gordan coefficients and are not explicitly stored in the reduced tensor.

Schematically:

```text
full physical index
       │
       ├── sector label         → structural metadata
       ├── degeneracy index     → ndarray axis
       └── irrep basis index    → encoded by symmetry tensors
```

---

## Fusion multiplicities

For a general fusion category,

$$
a \otimes b
=
\bigoplus_c
N_{ab}^{c}\, c.
$$

When

$$
N_{ab}^{c} > 1,
$$

a fusion vertex contains an additional multiplicity label.

The first implementation should not prematurely fix how all fusion-multiplicity degrees of freedom are stored.

Two natural possibilities are:

1. include multiplicity labels in the structural fusion-tree key;
2. expose multiplicity spaces as explicit dense axes of the reduced block.

This decision should be made from the requirements of \(F\)-moves, contraction, and basis transformations rather than from storage convenience.

The surrounding architecture should permit either choice without redesigning `TensorMap`.

---

## Backend model

`tenet-py` uses `autoray` for numerical backend dispatch.

The symmetry layer should avoid backend-specific branching such as

```python
if isinstance(x, np.ndarray):
    ...
elif isinstance(x, jax.Array):
    ...
elif isinstance(x, torch.Tensor):
    ...
```

Instead, dense numerical operations should be expressed through a small internal layer based on `autoray`.

For example:

```python
import autoray as ar

ar.do("reshape", x, shape)
ar.do("transpose", x, axes)
ar.do("conj", x)
ar.do("tensordot", x, y, axes=...)
```

The goal is not to create another array library.

Only the numerical primitives required by symmetric tensor algorithms should be wrapped.

```text
TensorMap algorithms
        │
        ↓
categorical analysis
        │
        ↓
small numerical interface
        │
        ↓
      autoray
        │
 ┌──────┼─────────┐
 │      │         │
NumPy   JAX    PyTorch
```

---

## Why `autoray`?

`autoray` lets `tenet-py` separate two questions.

The categorical layer asks:

> Which blocks participate, and which categorical transformation relates them?

The numerical backend asks:

> How should the resulting dense multidimensional operations be executed?

For example, categorical analysis may determine that

$$
B
=
F_{00}\,\operatorname{transpose}(A_0)
+
F_{01}\,\operatorname{transpose}(A_1).
$$

The categorical layer determines:

- which source blocks contribute;
- which \(F\)-coefficients appear;
- how reduced axes correspond;
- which output block is produced.

The numerical layer only performs operations such as:

```text
transpose
multiply
add
```

on backend-native arrays.

---

## Backend consistency

A `TensorMap` should normally use one numerical backend consistently.

For example, all reduced blocks should be NumPy arrays or all should be JAX arrays.

The library should validate backend compatibility rather than silently transfer data between frameworks.

Explicit conversion may later be provided:

```python
T_jax = T.to_backend("jax")
T_numpy = T.to_backend("numpy")
```

Such conversion should remain explicit because it may involve host-device transfer or synchronization.

---

## NumPy as the reference backend

NumPy should be the simplest reference backend.

Core algorithms should remain testable using plain NumPy without requiring JAX.

This provides:

- simple debugging;
- straightforward correctness tests;
- a reference implementation independent of compilation;
- separation of mathematical correctness from JAX-specific behavior.

---

## JAX as a first-class backend

JAX is not the core storage abstraction of `tenet-py`.

It is one particularly important array backend.

For a JAX-backed tensor,

```python
blocks[key] = jax.Array(...)
```

the library should support JAX transformations where the tensor operation is compatible with them.

The intended representation is:

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
└── dynamic leaves
    ├── jax.Array
    ├── jax.Array
    └── jax.Array
```

A `TensorMap` can therefore be registered as a custom PyTree.

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

The structural objects used in PyTree metadata should therefore be immutable and hashable where required.

---

## JAX specialization

Different categorical structures naturally correspond to different numerical programs.

Two tensors may differ in:

- sector content;
- fusion trees;
- reduced block shapes;
- domain and codomain;
- fusion multiplicities.

Such tensors may naturally lead to different JAX compilations.

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

The categorical structure therefore plays a role similar to static shape and type information in a compiled numerical program.

---

## Symmetry providers

Symmetry data should be exposed through providers rather than hard-coded global branching.

Avoid designs such as

```python
if symmetry == "u1":
    ...
elif symmetry == "su2":
    ...
```

Instead, algorithms should request mathematical capabilities.

A minimal interface may begin as:

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
fusion multiplicities
F-symbols
R-symbols
evaluation
coevaluation
pivotal data
Frobenius-Schur data
```

Algorithms should depend only on the capabilities they actually require.

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

The symmetry provider layer is independent of the numerical array backend.

---

## Initial symmetry providers

### Trivial symmetry

The trivial provider is used to verify that the `TensorMap` abstraction reduces cleanly to ordinary dense tensor algebra.

There is one sector.

### U(1)

Use integer charges with

$$
1_{\mathrm{unit}} = 0,
$$

$$
q^* = -q,
$$

and

$$
q_1 \otimes q_2
=
q_1 + q_2.
$$

The symbol `1_unit` above is only descriptive text; the actual API should use a normal Python representation for the tensor unit, such as the integer charge `0`.

### SU(2)

Use twice-spin integers:

```python
SU2Sector(two_j=1)  # j = 1/2
SU2Sector(two_j=2)  # j = 1
```

so half-integer spins are represented exactly.

Fusion is

$$
j_1 \otimes j_2
=
\bigoplus_{j = |j_1-j_2|}^{j_1+j_2}
j,
$$

with the usual unit spacing in \(j\).

SU(2) is included from the beginning so that the architecture is forced to support genuinely non-Abelian fusion.

The first milestone does not require Clebsch-Gordan coefficients or \(F\)-symbols.

---

## Duality

Duality belongs to the categorical structure.

For

$$
V
=
\bigoplus_a
\mathbb{C}^{m_a}
\otimes
V_a,
$$

the dual space is

$$
V^*
=
\bigoplus_a
\left(\mathbb{C}^{m_a}\right)^*
\otimes
V_{a^*}.
$$

A domain factor therefore appears as a dual object in the oriented tensor boundary.

```text
TensorMap

       C1   C2
        ↑    ↑
        │    │
        └─ T ─┐
              │
              D

oriented boundary

       C1  C2  D*
```

Moving a leg between domain and codomain must therefore not be implemented as merely changing an ndarray axis position.

For a general rigid category, such an operation may require nontrivial rigidity transformations.

---

## Array transpose versus categorical permutation

The following operations are conceptually distinct:

```text
backend ndarray transpose
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

but deciding whether a categorical operation reduces to that transpose, or additionally requires \(F\)- or \(R\)-coefficients, belongs to `tenet-py`.

Therefore `TensorMap` should not pretend to be an ordinary ndarray.

---

## Operation architecture

The intended long-term operation pipeline is:

```text
             TensorMap operation
                     │
                     ↓
           categorical analysis
                     │
         ┌───────────┼───────────┐
         │           │           │
     sector match   F-moves    braiding
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

The categorical layer determines the numerical program. The array backend executes it.

---

## Contraction

A future contraction

$$
C = A \cdot B
$$

should conceptually proceed as:

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
determine required fusion-tree transforms
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

The first implementation should prioritize correctness over batching and scheduling.

---

## Dense expansion

A future explicit operation

```python
T.to_dense()
```

may reconstruct the full physical tensor.

For SU(2), schematically,

$$
T_{
(\alpha_1,m_1)
(\alpha_2,m_2)
(\alpha_3,m_3)
}
=
\sum_{\tau}
A^{(\tau)}_{\alpha_1\alpha_2\alpha_3}
C^{(\tau)}_{m_1m_2m_3},
$$

where \(A^{(\tau)}\) is a reduced tensor and \(C^{(\tau)}\) denotes the corresponding symmetry tensor.

The magnetic quantum numbers \(m_i\) therefore appear only in the expanded representation.

Dense expansion is a transformation, not the canonical storage format.

---

## Proposed package structure

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

It should centralize only the subset of `autoray` functionality required internally.

It should not become a second array framework layered on top of `autoray`.

---

## Initial API sketch

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

$$
T:
W
\longrightarrow
V \otimes V
$$

may then be constructed from ordinary multidimensional arrays:

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

The same structural tensor may instead use JAX blocks:

```python
import jax.numpy as jnp

blocks = {
    key0: jnp.zeros((4, 3, 2)),
    key1: jnp.zeros((3, 4, 2)),
}
```

The categorical `TensorMap` representation does not change.

---

## Backend conversion

A future explicit conversion API may look like:

```python
T_jax = T.to_backend("jax")
T_numpy = T.to_backend("numpy")
```

Such conversion should map the reduced blocks while preserving all categorical metadata.

```text
TensorMap[NumPy]
      │
      │ convert blocks
      ↓
TensorMap[JAX]

unchanged:
    domain
    codomain
    fusion trees
    BlockKeys

changed:
    ndarray backend
```

---

## Relationship to TeNeT

[TeNeT](https://github.com/Ryo-wtnb11/TeNeT) provides several important conceptual foundations:

- explicit domain/codomain tensor-map semantics;
- fusion-tree-based non-Abelian representations;
- duality and categorical orientation;
- provider-based symmetry semantics;
- separation of mathematical structure from numerical execution.

`tenet-py` intentionally does not copy the Rust runtime architecture.

In particular, the initial Python implementation does not reproduce:

- contiguous packed tensor payloads;
- explicit block offsets and strides;
- custom storage traits;
- native execution plans;
- manual scratch allocation;
- CUDA-specific runtime infrastructure.

These are useful choices for a native HPC implementation but are not required for the initial Python array model.

---

## Relationship to symmray

[symmray](https://github.com/jcmgray/symmray) motivates the numerical storage philosophy:

```text
symmetry metadata
      +
block key → raw backend ndarray
```

rather than a custom dense storage implementation.

`tenet-py` follows this broad idea while extending it toward an explicitly categorical non-Abelian tensor-map model.

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
 F / R structure
       │                           │
       └─────────────┬─────────────┘
                     │
                  tenet-py
```

---

## Relationship to JAX

JAX is not the tensor storage abstraction of `tenet-py`.

Instead, JAX is one particularly powerful backend for the reduced arrays.

This distinction allows the same categorical tensor representation to be used for:

- NumPy reference calculations;
- JAX differentiation and compilation;
- potentially PyTorch-based workflows.

JAX-specific functionality should integrate with the generic representation rather than define it.

---

## Roadmap

### Milestone 1 — Structural foundation

Implement:

- immutable sector types;
- a minimal fusion-provider protocol;
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

### Milestone 2 — Basic tensor operations

Implement:

- blockwise addition;
- scalar multiplication;
- conjugation;
- norm;
- backend conversion;
- structural validation.

Test with at least NumPy and JAX.

### Milestone 3 — Categorical transformations

Introduce:

- Clebsch-Gordan data;
- \(F\)-symbols;
- \(R\)-symbols where applicable;
- recoupling;
- categorical permutation;
- adjoint;
- domain/codomain movement.

### Milestone 4 — Contraction

Implement general symmetric contraction using

```text
categorical analysis
        ↓
block matching
        ↓
fusion-tree transformations
        ↓
autoray dense kernels
        ↓
output TensorMap
```

### Milestone 5 — Factorizations

Implement symmetry-aware:

- QR;
- SVD;
- eigendecomposition;
- truncation;
- reconstructed bond spaces.

Structural changes such as truncation should be represented explicitly at the Python level.

### Milestone 6 — Performance

Only after profiling, consider:

- shape-based block bucketing;
- stacked block execution;
- batched matrix multiplication;
- operation-plan caching;
- fusion-transform caching;
- JAX compilation specialization;
- custom kernels where justified.

The public `TensorMap` representation should remain independent of these optimizations.

---

## Design principles

### TensorMap first, ndarray second

The mathematical object is

$$
T \in \operatorname{Hom}(D,C),
$$

not merely an ndarray with symmetry annotations.

### Reduced blocks remain ordinary arrays

Do not implement custom dense storage unless profiling provides a concrete reason.

### Use `autoray` for numerical dispatch

The symmetry layer should not depend directly on NumPy, JAX, or PyTorch implementation details.

### JAX is important but not foundational

JAX-specific capabilities should integrate cleanly without making `jax.Array` part of the categorical model.

### Structural indices are not ndarray axes

Sector labels and fusion channels are metadata.

Degeneracy indices are array dimensions.

### Non-Abelian fusion is explicit

Fusion trees are first-class immutable objects.

### Optimization stays below the mathematical API

Packing, batching, kernel selection, and execution planning must not determine public tensor semantics.

### Correctness before runtime sophistication

The first implementation should establish a mathematically coherent representation before introducing native-style scheduling or memory optimization.

---

## Project philosophy

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

The core idea is:

```text
categorical tensor structure in Python
+
reduced tensors as ordinary backend-native ndarrays
+
backend dispatch through autoray
```

Everything else should be built on top of that boundary.
