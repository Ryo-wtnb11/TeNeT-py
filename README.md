# tenet-py

Symmetry-aware multidimensional tensors in pure Python, with backend-native arrays and explicit non-Abelian categorical structure.

`tenet-py` is a Python library for tensor-network calculations with Abelian, non-Abelian, fermionic, and eventually more general fusion-category symmetries.

The central design choice is:

> **A tensor is presented to the user as an ordered multidimensional array with symmetry-aware edges, not as a permanent domain-to-codomain `TensorMap`.**

Each tensor edge carries its local categorical information, while nonlocal structure such as fusion trees, recoupling, braiding, and temporary matrix partitions is handled separately.

Reduced numerical data are stored directly as ordinary multidimensional arrays provided by NumPy, JAX, PyTorch, or other compatible backends.

> **Status:** early design and implementation. The API is not stable.

---

## Motivation

Tensor-network algorithms are naturally expressed in terms of multidimensional arrays:

```python
C = tenet.tensordot(A, B, axes=((2,), (0,)))
```

or

```python
C = tenet.einsum("abc,cde->abde", A, B)
```

Factorizations similarly act on a temporary bipartition of tensor axes:

```python
U, S, Vh = tenet.linalg.svd(
    A,
    axes=((0, 1), (2, 3)),
)
```

A permanent decomposition

```text
domain -> codomain
```

is therefore often artificial from the point of view of tensor-network manipulation.

At the same time, a symmetric tensor is **not** merely a block-sparse ndarray.

For non-Abelian symmetries and more general fusion categories, tensor operations may involve:

- sector selection rules;
- non-Abelian fusion channels;
- fusion multiplicities;
- fusion-tree basis changes;
- duality;
- evaluation and coevaluation;
- \(F\)-moves;
- braiding and \(R\)-symbols;
- Frobenius--Schur and pivotal data.

`tenet-py` therefore separates three layers:

```text
user tensor
    │
    │ ordered symmetry-aware edges
    │
    ▼
categorical structure
    │
    │ sectors / fusion trees / duality / F / R
    │
    ▼
reduced numerical arrays
    │
    │ NumPy / JAX / PyTorch / ...
    ▼
dense numerical kernels
```

The goal is to retain the mathematical structure required by general symmetric tensor networks without forcing tensor-network code into a matrix-oriented API.

---

## Core design

The public tensor object is conceptually

```python
Tensor(
    edges=(e0, e1, ..., eN),
    blocks=...,
)
```

where each `Edge` describes one logical tensor axis.

For example,

```python
e0 = Edge(V)
e1 = Edge(W, dual=True)
e2 = Edge(X)

A = Tensor(
    edges=(e0, e1, e2),
    blocks=blocks,
)
```

represents a rank-3 symmetric tensor.

The public structure is therefore

```text
                  Tensor

        axis 0     axis 1     axis 2
          │          │          │
          ▼          ▼          ▼
       Edge(V)    Edge(W*)    Edge(X)

                  │
                  ▼

       fusion / symmetry structure

                  │
                  ▼

        BlockKey -> backend ndarray
```

rather than

```text
             codomain
                ↑
                │
             TensorMap
                │
                ↑
              domain
```

A domain/codomain partition may still be introduced when an operation actually requires a linear-map interpretation, but it is not part of the permanent public tensor shape.

---

## Edges

An `Edge` stores information intrinsic to one logical tensor axis.

A minimal interface is expected to contain something like

```python
Edge(
    space=V,
    dual=False,
    name=None,
)
```

where:

- `space` specifies the graded representation space;
- `dual` distinguishes \(V\) from \(V^*\);
- `name` is optional user-facing metadata.

Additional local features may be introduced when required.

The important design rule is:

> **Only information intrinsic to one edge belongs to `Edge`.**

For example:

| Information | Owner |
|---|---|
| graded space | `Edge` |
| dual / orientation | `Edge` |
| optional name or tag | `Edge` |
| axis order | `Tensor` |
| total/root sector | `Tensor` / categorical structure |
| fusion tree | tensor-level categorical structure |
| fusion multiplicity labels | fusion structure |
| \(F\)-symbols | symmetry provider |
| \(R\)-symbols | symmetry provider |
| contraction labels | operation |
| SVD left/right partition | operation |
| matrix input/output partition | operation |
| execution batching | execution plan |

In particular, fusion trees and braid history are **not** properties of an individual edge.

---

## Duality is not input/output

Two different notions must not be conflated:

1. whether an object is \(V\) or its dual \(V^*\);
2. whether a leg is regarded as an input or output of a particular linear map.

For example,

\[
T:
A \otimes B^*
\longrightarrow
C^* \otimes D
\]

contains all four possibilities:

```text
A     : input,  non-dual
B*    : input,  dual
C*    : output, dual
D     : output, non-dual
```

The duality of an edge is therefore intrinsic categorical information.

The input/output role is instead associated with a chosen linear-map presentation.

`tenet-py` does not require every tensor to carry such a presentation permanently.

---

## Relation to tensor maps

In a rigid tensor category,

\[
\operatorname{Hom}(D,C)
\simeq
\operatorname{Hom}
\left(
\mathbf{1},
C \otimes D^*
\right).
\]

Thus a tensor map

\[
T:D\rightarrow C
\]

can be represented as an oriented tensor whose former domain legs appear dualized.

This motivates the array-first representation used by `tenet-py`.

However, the isomorphism above must **not** be interpreted as saying that moving a leg between input and output is merely changing metadata.

For ordinary finite-dimensional vector spaces, bending a leg can often be made invisible at the level of numerical coefficients.

For a general fusion category, bending may change the reduced basis and therefore require nontrivial categorical transformations.

Schematically,

```text
morphism representation

        V
        ↑
        │
        T
        │
        W

        │ bend W
        ▼

oriented tensor representation

        V ── T ── W*
```

may require more than

```python
edge.dual = not edge.dual
```

Internally it may involve:

```text
evaluation / coevaluation
        +
fusion-basis transformation
        +
B-symbols or equivalent rigidity data
        +
quantum-dimension factors
        +
Frobenius-Schur / pivotal factors
```

depending on the category and normalization convention.

The array-first API therefore does **not** trivialize categorical duality.

---

## Temporary linear-map views

Some operations are inherently matrix-like.

For a rank-4 tensor

\[
A_{ijkl},
\]

an SVD is only defined after selecting a bipartition such as

\[
(ij)\mid(kl).
\]

The preferred user API is therefore operation-local:

```python
U, S, Vh = tenet.linalg.svd(
    A,
    axes=((0, 1), (2, 3)),
)
```

rather than requiring the tensor to have been constructed with a permanent domain and codomain.

Internally this may be lowered to a temporary linear-map representation:

```text
Tensor
  edges = (0, 1, 2, 3)
          │
          │ axes=((0,1),(2,3))
          ▼
temporary map view

  codomain = (0, 1)
  domain   = (2, 3)

          │
          ▼
fusion-tree pair / block matrices
          │
          ▼
backend SVD
```

An advanced API may expose this explicitly:

```python
M = A.as_map(
    outputs=(0, 1),
    inputs=(2, 3),
)
```

but such a view is not the canonical tensor representation.

---

## Graded spaces

A symmetry-graded space has the form

\[
V
=
\bigoplus_a
\mathbb{C}^{m_a}\otimes V_a,
\]

where:

- \(a\) is a symmetry sector;
- \(V_a\) is the corresponding irreducible representation;
- \(m_a\) is the degeneracy of that sector.

For example, an SU(2)-graded space

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

The degeneracy \(m_a\) must not be confused with the physical irrep dimension.

For SU(2),

\[
\dim V_j = 2j+1,
\]

but a reduced numerical axis associated with sector \(j\) has size \(m_j\), not \(2j+1\).

---

## Reduced blocks are multidimensional arrays

The fundamental numerical-storage rule is:

> **One logical reduced block is one backend-native multidimensional array.**

For a tensor with five logical edges and fixed external sectors

\[
(a_0,a_1,a_2,a_3,a_4),
\]

a corresponding reduced block may have shape

\[
(m_{a_0},m_{a_1},m_{a_2},m_{a_3},m_{a_4}).
\]

For example,

```python
block.shape == (2, 4, 3, 8, 5)
```

represents the reduced tensor

\[
A_{\alpha_0\alpha_1\alpha_2\alpha_3\alpha_4}.
\]

A block may be backed by

```text
numpy.ndarray
jax.Array
torch.Tensor
...
```

The backend array is responsible for:

- memory allocation;
- dtype;
- device placement;
- physical strides;
- dense numerical kernels.

`tenet-py` is responsible for:

- sectors;
- graded spaces;
- duality;
- allowed block structure;
- fusion trees;
- recoupling;
- categorical permutation;
- braiding;
- mapping logical tensor operations to dense kernels.

---

## Three different kinds of indices

`tenet-py` distinguishes three conceptually different kinds of indices.

### Structural categorical indices

Examples include:

- U(1) charge;
- SU(2) spin;
- intermediate fusion channels;
- fusion multiplicity labels;
- final coupled sectors;
- fusion-tree topology.

These are structural objects.

They are not ndarray axes.

### Reduced numerical indices

These correspond to degeneracy spaces such as

\[
\alpha=1,\ldots,m_a.
\]

They are ordinary ndarray axes.

For example,

\[
A_{\alpha\beta\gamma}
\]

is stored naturally as

```python
array.shape == (m_a, m_b, m_c)
```

inside one reduced block.

### Physical irrep indices

For SU(2), these include magnetic quantum numbers

\[
m=-j,-j+1,\ldots,j.
\]

They belong to symmetry tensors such as Clebsch--Gordan coefficients and are not normally stored explicitly in the reduced numerical block.

Schematically:

```text
physical tensor index
        │
        ├── sector label
        │      structural metadata
        │
        ├── degeneracy index
        │      ndarray axis
        │
        └── irrep basis index
               encoded by symmetry data
```

---

## Fusion trees

For Abelian symmetries, external sector labels may be sufficient to identify a reduced block.

For U(1),

\[
q_1\otimes q_2
\rightarrow
q_1+q_2
\]

has a unique fusion result.

For non-Abelian symmetry this is no longer true.

Consider

\[
j_1\otimes j_2\otimes j_3.
\]

With a left-associated tree,

```text
j1     j2       j3
 \     /
  \   /
   j12
      \
       \
        J
```

the intermediate sector \(j_{12}\) can take several values.

Therefore the external sectors

\[
(j_1,j_2,j_3)
\]

do not uniquely specify a reduced basis element.

`tenet-py` therefore treats fusion structure as explicit immutable metadata.

Conceptually:

```python
tree = FusionTree(
    uncoupled=(j1, j2, j3),
    intermediate=(j12,),
    coupled=J,
)
```

The first implementation should use a canonical tree convention.

Other tree topologies are related by categorical basis transformations such as \(F\)-moves.

---

## Block keys

In the array-first representation, a block key identifies a categorical basis sector of the **ordered tensor boundary**.

Conceptually:

```python
key = BlockKey(
    sectors=(a0, a1, a2, a3),
    fusion=...,
)
```

where `fusion` contains the intermediate channels and multiplicity data required to specify the reduced basis.

The canonical block store is

```python
Mapping[BlockKey, Array]
```

with each value an ordinary backend-native multidimensional array.

This differs deliberately from a permanent pair

```text
codomain fusion tree
+
domain fusion tree
```

because no permanent domain/codomain partition is imposed on the tensor.

When an operation requires a linear-map representation, the relevant block structure can be recoupled into a pair-of-fusion-trees representation temporarily.

---

## Fusion multiplicities

For a general fusion category,

\[
a\otimes b
=
\bigoplus_c
N_{ab}^{c}\,c.
\]

When

\[
N_{ab}^{c}>1,
\]

a fusion vertex carries an additional multiplicity label.

The implementation should not force these multiplicity degrees of freedom into the wrong abstraction merely for storage convenience.

Possible representations include:

1. structural multiplicity labels in the fusion-tree key;
2. explicit dense multiplicity axes in reduced numerical blocks;
3. a hybrid representation selected according to the operation.

The choice should be determined by the requirements of:

- \(F\)-moves;
- contractions;
- basis transformations;
- batching;
- numerical efficiency.

The public `Tensor` and `Edge` abstractions should not depend on this internal choice.

---

## Tensor operations

### Contraction

The primary contraction interface should be axis-oriented.

```python
C = tenet.tensordot(
    A,
    B,
    axes=((2,), (0,)),
)
```

and, where mathematically unambiguous,

```python
C = tenet.einsum(
    "abc,cde->abde",
    A,
    B,
)
```

A contraction proceeds conceptually as

```text
Tensor A + Tensor B
        │
        ▼
identify contracted edges
        │
        ▼
validate spaces and duality
        │
        ▼
match compatible sectors
        │
        ▼
determine fusion-tree transformations
        │
        ▼
construct output categorical structure
        │
        ▼
generate dense numerical operations
        │
        ▼
autoray / backend
        │
        ▼
Tensor C
```

The tensor-network expression remains axis-based even when the implementation internally lowers parts of the contraction to matrix multiplication.

---

## Permutation is not raw ndarray transpose

A logical tensor permutation and a backend-array transpose are different operations.

A backend operation such as

```python
ar.do("transpose", block, axes)
```

only reorders numerical degeneracy axes.

A categorical permutation may additionally require:

- recoupling;
- \(F\)-moves;
- fermionic signs;
- braiding;
- \(R\)-symbols.

Thus

```python
B = A.transpose((2, 0, 1))
```

means

> construct the same categorical tensor with its logical edges reordered,

not simply

> call `transpose` independently on every stored ndarray.

The categorical layer determines which numerical transformations implement the requested permutation.

For genuinely braided anyonic categories, arbitrary permutations may additionally require explicit information about the braid or planar embedding. An Einstein-summation string alone does not in general encode over/under crossings.

---

## Bending is a categorical operation

Changing

```text
V
```

into

```text
V*
```

as part of bending a tensor leg must not be implemented as only

```python
edge.dual = not edge.dual
```

on an existing tensor.

The `dual` flag records the orientation of the logical edge.

Changing that orientation may require a transformation of the reduced coefficients.

Therefore the library should distinguish:

```text
constructing a tensor whose edge is dual
```

from

```text
bending an existing tensor edge
```

The second operation belongs to the categorical transformation layer.

---

## Linear algebra

Matrix factorizations use operation-local axis partitions.

For example:

```python
U, S, Vh = tenet.linalg.svd(
    A,
    axes=((0, 2), (1, 3)),
)
```

Conceptually:

```text
rank-4 Tensor
     │
     │ choose (0,2) | (1,3)
     ▼
categorical recoupling
     │
     ▼
sector-resolved matrix blocks
     │
     ▼
backend SVD
     │
     ▼
reconstruct symmetric tensors
```

The same model applies to:

- QR;
- eigendecomposition;
- polar decomposition;
- truncation;
- isometry construction.

The tensor itself does not need to remember the selected matrix partition after the operation has completed.

---

## Logical representation versus execution representation

The public representation should remain simple even if optimized execution is added later.

For example:

```text
logical blocks

key A -> array(shape=(4, 8, 16))
key B -> array(shape=(4, 8, 16))
key C -> array(shape=(4, 8, 16))
```

may temporarily be lowered to

```text
array(shape=(3, 4, 8, 16))
             ↑
         block batch
```

for a batched kernel.

Likewise, an operation may temporarily convert the array-oriented tensor into a block-matrix representation:

```text
Tensor
  │
  ▼
categorical lowering
  │
  ├── recouple edges
  ├── choose map partition
  ├── group compatible blocks
  ├── transpose / reshape
  └── batch kernels
  │
  ▼
backend arrays
  │
  ▼
reconstruct Tensor
```

The distinction is fundamental:

> **The public tensor representation is not the execution representation.**

Packing, batching, block-matrix lowering, and kernel scheduling should remain implementation details.

---

## Why not packed storage?

A native tensor runtime may represent reduced blocks using

```text
one contiguous buffer
+
offsets
+
shapes
+
strides
```

to control memory allocation, scheduling, and device placement precisely.

That model is appropriate for a low-level HPC implementation.

`tenet-py` deliberately starts from a different point.

Python already has mature multidimensional array systems.

Reimplementing their dense-array functionality would make interoperability harder while duplicating functionality provided by existing numerical frameworks.

The canonical representation is therefore:

```text
Tensor
│
├── categorical metadata
│
└── blocks
    ├── key A -> backend ndarray
    ├── key B -> backend ndarray
    ├── key C -> backend ndarray
    └── ...
```

Optimized packed or batched representations may still be introduced below this layer when profiling demonstrates a benefit.

---

## Backend model

`tenet-py` should use a small backend-dispatch layer based on `autoray`.

Dense numerical operations may be expressed through primitives such as

```python
import autoray as ar

ar.do("reshape", x, shape)
ar.do("transpose", x, axes)
ar.do("conj", x)
ar.do("tensordot", x, y, axes=...)
ar.do("matmul", x, y)
```

The categorical layer answers:

> Which reduced blocks participate, and how must they be transformed?

The backend answers:

> How are the resulting dense multidimensional operations executed?

Schematically:

```text
Tensor operation
       │
       ▼
categorical analysis
       │
       ▼
numerical program
       │
       ▼
autoray
       │
  ┌────┼─────┐
  │    │     │
NumPy JAX PyTorch
```

The goal is not to create another dense array library.

---

## Backend consistency

One tensor should normally use one numerical backend consistently.

For example, all blocks should be NumPy arrays or all should be JAX arrays.

The library should not silently move data between frameworks or devices.

Backend conversion should be explicit:

```python
T_jax = T.to_backend("jax")
T_numpy = T.to_backend("numpy")
```

Such conversion changes numerical storage while preserving the logical categorical tensor.

---

## NumPy as the reference backend

NumPy should provide the simplest correctness-oriented backend.

Core algorithms should remain testable without JAX or GPU support.

This provides:

- simple debugging;
- straightforward numerical tests;
- a reference implementation;
- separation of categorical correctness from compilation behavior.

---

## JAX as a first-class backend

JAX is an important backend, but it does not define the tensor abstraction.

For a JAX-backed tensor,

```python
blocks[key] = jax.Array(...)
```

the categorical structure is static while reduced arrays are dynamic numerical leaves.

Conceptually:

```text
Tensor
│
├── static structure
│   ├── edges
│   ├── spaces
│   ├── sectors
│   ├── fusion trees
│   └── BlockKeys
│
└── dynamic leaves
    ├── jax.Array
    ├── jax.Array
    └── ...
```

A `Tensor` can therefore be represented as a JAX PyTree.

Different categorical structures may naturally lead to different compiled programs.

This is intentional: categorical structure plays a role similar to static shape and type information.

---

## Symmetry providers

Symmetry data should be supplied through capability-based providers instead of global symmetry-specific branches.

Avoid:

```python
if symmetry == "u1":
    ...
elif symmetry == "su2":
    ...
```

Prefer algorithms that request the mathematical operations they require.

A minimal interface may begin with

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

More advanced providers may supply:

```text
fusion multiplicities
F-symbols
R-symbols
evaluation
coevaluation
pivotal structure
Frobenius-Schur data
```

Algorithms should depend only on the capabilities they actually use.

The symmetry provider is independent of the numerical array backend.

---

## Initial symmetry providers

### Trivial symmetry

The trivial provider verifies that the high-level API reduces to ordinary multidimensional tensor algebra when no nontrivial categorical structure is present.

### U(1)

Use integer charges with

\[
0^*=0,
\qquad
q^*=-q,
\]

and

\[
q_1\otimes q_2
=
q_1+q_2.
\]

U(1) provides the first nontrivial block-sparse implementation while retaining unique fusion.

### SU(2)

Use twice-spin integers:

```python
SU2Sector(two_j=1)  # j = 1/2
SU2Sector(two_j=2)  # j = 1
```

with fusion

\[
j_1\otimes j_2
=
\bigoplus_{j=|j_1-j_2|}^{j_1+j_2}j.
\]

SU(2) is included early so that the architecture is forced to handle genuinely non-Abelian fusion channels.

---

## API sketch

```python
import numpy as np

import tenet
from tenet import Edge, GradedSpace, Tensor
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

a = Edge(V)
b = Edge(W, dual=True)
c = Edge(V)

A = Tensor(
    edges=(a, b, c),
    blocks={
        key0: np.zeros((4, 2, 3)),
        key1: np.zeros((3, 2, 4)),
    },
)
```

Tensor operations use logical axes:

```python
B = A.transpose((2, 0, 1))

C = tenet.tensordot(
    A,
    B,
    axes=((2,), (0,)),
)
```

or Einstein notation where supported:

```python
C = tenet.einsum(
    "abc,cde->abde",
    A,
    B,
)
```

Matrix factorizations select their partition locally:

```python
U, S, Vh = tenet.linalg.svd(
    C,
    axes=((0, 1), (2, 3)),
)
```

The same logical tensor can use JAX-backed reduced blocks without changing the categorical API.

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
│       ├── linalg/
│       │   ├── __init__.py
│       │   ├── svd.py
│       │   └── qr.py
│       │
│       ├── edge.py
│       ├── space.py
│       ├── fusion_tree.py
│       ├── block.py
│       ├── tensor.py
│       ├── contraction.py
│       └── lowering.py
│
├── tests/
│   ├── test_space.py
│   ├── test_edge.py
│   ├── test_fusion_tree.py
│   ├── test_tensor.py
│   ├── test_contraction.py
│   ├── test_backends.py
│   └── test_jax.py
│
└── README.md
```

The `array` module should remain intentionally small.

The `lowering` layer is responsible for translating logical tensor operations into categorical transformations and dense backend programs.

---

## Relationship to TeNeT

The native TeNeT implementation provides the categorical foundations relevant to `tenet-py`, including:

- graded spaces and sectors;
- fusion-tree-based non-Abelian structure;
- duality;
- capability-based symmetry providers;
- separation of categorical semantics from numerical execution.

`tenet-py` intentionally does not copy the native runtime architecture.

In particular, the initial Python implementation does not require:

- packed tensor payloads;
- explicit offsets and strides;
- manual scratch allocation;
- native operator queues;
- custom GPU memory management;
- low-level CUDA scheduling.

Those mechanisms belong to a performance-oriented native execution layer.

The Python implementation instead prioritizes interoperability with multidimensional array ecosystems.

---

## Design influences

The public API follows the general array-oriented philosophy used by Python symmetric-tensor libraries such as symmray and YASTN:

```text
tensor
=
ordered logical axes
+
symmetry metadata
+
backend-native reduced arrays
```

The non-Abelian categorical layer follows the fusion-tree viewpoint used in TeNeT and TensorKit:

```text
external sectors
+
fusion channels
+
duality
+
F / R / rigidity data
```

The intended combination is therefore:

```text
             tenet-py

       array-oriented public API
                │
                │
          Tensor + Edge
                │
                ▼
       categorical lowering
                │
                │
     fusion trees / F / R / duality
                │
                ▼
       backend-native arrays
```

---

## Roadmap

### Milestone 1 — Structural foundation

Implement:

- immutable sector types;
- minimal fusion-provider protocol;
- trivial symmetry;
- U(1);
- SU(2) fusion rules;
- `GradedSpace`;
- `Edge`;
- canonical `FusionTree`;
- `BlockKey`;
- `Tensor`;
- multidimensional backend-native blocks;
- minimal backend dispatch;
- NumPy tests;
- JAX PyTree tests.

### Milestone 2 — Basic tensor algebra

Implement operations that do not yet require general recoupling:

- addition;
- scalar multiplication;
- conjugation;
- norm;
- backend conversion;
- structural validation;
- edge inspection and manipulation.

### Milestone 3 — Categorical transformations

Introduce:

- Clebsch--Gordan data where required;
- \(F\)-symbols;
- \(R\)-symbols where applicable;
- recoupling;
- bending;
- categorical permutation;
- adjoint;
- rigidity transformations.

### Milestone 4 — General contraction

Implement:

```text
axis contraction request
        │
        ▼
categorical analysis
        │
        ▼
block matching
        │
        ▼
fusion-tree transformations
        │
        ▼
dense backend program
        │
        ▼
output Tensor
```

Provide `tensordot` first, followed by an `einsum` interface.

### Milestone 5 — Linear algebra

Implement symmetry-aware:

- QR;
- SVD;
- eigendecomposition;
- truncation;
- reconstructed bond spaces.

These operations use temporary axis bipartitions rather than permanent domain/codomain structure.

### Milestone 6 — Performance

Only after profiling, consider:

- operation-plan caching;
- fusion-transform caching;
- shape-based block bucketing;
- stacked block execution;
- batched matrix multiplication;
- JAX compilation specialization;
- custom kernels where justified.

Performance optimizations must not determine the public tensor semantics.

---

## Design principles

### Tensor first, map when needed

The public object is an \(N\)-dimensional symmetry-aware tensor.

A domain/codomain partition is introduced only for operations that require one.

### Edges carry local categorical information

Spaces and duality belong to tensor edges.

Fusion trees and operation-specific roles do not.

### Duality is not a boolean-only operation

A dual flag records orientation.

Changing orientation of an existing tensor may require a genuine categorical basis transformation.

### Structural indices are not ndarray axes

Sector labels and fusion channels are metadata.

Degeneracy indices are numerical array dimensions.

### Reduced blocks remain ordinary arrays

Do not implement a custom dense-storage abstraction unless profiling demonstrates a concrete need.

### Non-Abelian fusion is explicit

Fusion trees are first-class immutable structural data.

### Array syntax does not erase category theory

`transpose`, `tensordot`, and `einsum` are user-facing tensor operations.

Their implementation may require \(F\)-moves, braiding, bending, or other categorical transformations before dense kernels can be executed.

### Backend dispatch stays below tensor semantics

NumPy, JAX, PyTorch, and future backends execute reduced numerical programs.

They do not define the categorical tensor model.

### Optimization stays below the mathematical API

Packing, batching, kernel selection, and execution planning are implementation details.

### Correctness before runtime sophistication

The first implementation should establish a coherent non-Abelian tensor model before introducing native-style scheduling or memory optimization.

---

## Project philosophy

```text
                         tenet-py
                             │
                             │
                    symmetry-aware Tensor
                             │
                  ordered categorical edges
                             │
              ┌──────────────┴──────────────┐
              │                             │
          structure                      values
              │                             │
         GradedSpace                  backend ndarray
         Edge                         multidimensional
         FusionTree                  reduced blocks
         BlockKey                         │
         duality                          │
         fusion                           │
         F / R / rigidity                 │
              │                             │
              └──────── operation ─────────┘
                             │
                             ▼
                     categorical lowering
                             │
                             ▼
                    numerical operations
                             │
                             ▼
                          autoray
                             │
                 ┌───────────┼───────────┐
                 │           │           │
               NumPy        JAX       PyTorch
```

The central rule is:

> **Expose tensors as tensors. Keep categorical structure explicit. Lower to tensor maps or block matrices only when an operation requires them.**
