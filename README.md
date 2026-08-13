# TeNeT-py

**Non-Abelian symmetric tensors with ndarray-style Python APIs and backend-native numerical execution.**

`TeNeT-py` is a pure-Python library for symmetry-aware tensor computation designed to connect non-Abelian tensor algebra with the Python numerical and machine-learning ecosystem.

The central design principle is:

> **TensorMap semantics are the mathematical model; an ndarray-like symmetric tensor is the programming model.**

A tensor is still interpreted categorically as a morphism

```math
T \in \mathrm{Hom}(D,C),
```

with explicit duality, fusion structure, recoupling, and braiding where required.

However, the primary public object is **not** forced into a permanent

```text
codomain × domain
```

axis layout.

Instead, a tensor exposes an ordered sequence of legs, much like the axes of an `ndarray`. Each leg carries enough categorical metadata to recover the corresponding TensorMap exactly.

This allows tensor-network code to use APIs such as

```python
T.shape
T.ndim
T.transpose(...)
tenet.tensordot(A, B, axes=...)
tenet.einsum("abc,cde->abde", A, B)
tenet.linalg.svd(T, axes=((0, 2), (1, 3)))
```

while preserving the categorical semantics required for non-Abelian symmetries and more general fusion categories.

Reduced numerical data are stored in backend-native multidimensional arrays (NumPy, JAX, PyTorch, ...) and dispatched through `autoray`.

The goal is not to port TensorKit or TeNeT to Python.

The goal is to provide a **categorically correct symmetric tensor abstraction that fits naturally into the Python array ecosystem**, making it easier to exploit GPU execution, automatic differentiation, JIT compilation, vectorization, contraction planners, and machine-learning infrastructure in applications such as tensor-network VMC and neural quantum states.

> **Status:** early design and implementation. The API is not stable.

---

# Design thesis

The architecture separates three models:

```text
mathematical model
────────────────────────────────────────
              TensorMap
            T ∈ Hom(D, C)
       duality / fusion / F / R
                    │
                    ▼

programming model
────────────────────────────────────────
            SymmetricTensor
          ordered ndarray-like legs
     shape / ndim / transpose / einsum
                    │
                    ▼

execution model
────────────────────────────────────────
      backend-native reduced arrays
     NumPy / JAX / PyTorch via autoray
                    │
                    ▼
          GPU / AD / JIT / vmap
```

The important distinction is:

```text
TensorMap semantics
        !=
TensorMap-shaped public API
```

`TeNeT-py` keeps the former and deliberately avoids requiring the latter.

---

# Motivation

A non-Abelian symmetric tensor is not simply a block-sparse ndarray.

Its structure may involve:

- graded representation spaces;
- dual objects;
- input and output orientation;
- non-unique fusion channels;
- fusion multiplicities;
- fusion-tree bases;
- recoupling transformations;
- categorical permutations;
- braiding;
- evaluation and coevaluation maps.

At the same time, after these categorical choices are fixed, the remaining variational degrees of freedom are ordinary numerical tensors.

Those numerical tensors should ideally be handled by numerical systems that already provide:

- GPU kernels;
- automatic differentiation;
- JIT compilation;
- vectorization;
- device placement;
- mixed precision;
- optimized matrix multiplication;
- tensor contraction;
- distributed execution.

`TeNeT-py` therefore does not attempt to replace NumPy, JAX, or PyTorch.

Instead, it defines the categorical layer required to tell these backends **which numerical operations must be performed and what those operations mean**.

The basic decomposition is:

```text
categorical structure
        +
backend-native numerical arrays
        +
structural lowering
        =
symmetric tensor computation
```

---

# Design influences

The design is informed by several existing approaches:

- [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl) provides the primary reference for explicit morphism semantics, fusion-tree bases, duality, and non-Abelian block linear algebra.
- [TeNeT](https://github.com/Ryo-wtnb11/TeNeT) provides a reference for capability-based symmetry providers and a strict separation between categorical semantics, planning, and numerical execution.
- [symmray](https://github.com/jcmgray/symmray) demonstrates the value of making symmetric tensors behave like ordinary arrays and using backend-native blocks with `autoray`.
- [YASTN](https://github.com/yastn/yastn) demonstrates an axis-oriented Python tensor API with leg signatures, differentiable backends, and tensor-network operations.
- [froSTspin](https://github.com/ogauthe/frostspin) provides a useful non-Abelian Python reference that combines per-leg signatures with a matrix-oriented row/column partition.

`TeNeT-py` deliberately does not copy any one of these interfaces.

Its target is approximately:

```text
TensorKit / TeNeT categorical semantics
                 +
YASTN / symmray array ergonomics
                 +
Python ML execution ecosystem
```

---

# The central tensor type

The primary owning type is

```python
SymmetricTensor
```

rather than `TensorMap`.

Conceptually:

```python
T = SymmetricTensor(
    legs=(leg0, leg1, leg2, ...),
    blocks=...,
)
```

A tensor has:

```text
ordered public legs
categorical structure
fusion-basis information
reduced numerical blocks
```

The public leg order corresponds to the tensor's ndarray-style axis order.

Therefore:

```python
T.ndim == len(T.legs)
T.shape[i] == T.legs[i].dimension
```

up to the usual distinction between full physical dimension and reduced block dimensions.

The tensor nevertheless has a well-defined domain and codomain.

They are derived from the leg metadata.

---

# Legs

A leg carries local categorical information.

A minimal conceptual representation is:

```python
@dataclass(frozen=True)
class Leg:
    space: GradedSpace
    side: Side
    dual: bool = False
    name: Hashable | None = None
```

where

```python
Side.OUT
Side.IN
```

specify whether the leg belongs to the codomain or domain of the underlying morphism.

The two pieces of information

```text
IN / OUT
```

and

```text
non-dual / dual
```

are independent.

This distinction is essential.

For example,

```math
T:
A \otimes B^*
\longrightarrow
C^* \otimes D
```

contains four different combinations:

```text
A     : IN,  non-dual
B*    : IN,  dual
C*    : OUT, dual
D     : OUT, non-dual
```

A dual object is not the same concept as an input leg.

---

# Recovering TensorMap semantics

Let the effective categorical object associated with leg \(i\) be

```math
X_i =
\begin{cases}
V_i, & \text{if } \mathrm{dual}_i = \mathrm{False},\\
V_i^*, & \text{if } \mathrm{dual}_i = \mathrm{True}.
\end{cases}
```

The codomain is obtained from the output legs:

```math
C(T)
=
\bigotimes_{i:\,\mathrm{side}_i=\mathrm{OUT}}
X_i,
```

and the domain from the input legs:

```math
D(T)
=
\bigotimes_{i:\,\mathrm{side}_i=\mathrm{IN}}
X_i.
```

The relative order of legs on each side is inherited from their public axis order.

Therefore every `SymmetricTensor` defines a morphism

```math
T \in \mathrm{Hom}(D(T),C(T)).
```

For example:

```python
T = SymmetricTensor(
    legs=(
        Leg(C1, OUT),
        Leg(D1, IN),
        Leg(C2, OUT),
        Leg(D2, IN),
    ),
    ...
)
```

has ndarray-facing axis order

```text
(C1, D1, C2, D2)
```

but TensorMap semantics

```math
T:
D_1 \otimes D_2
\longrightarrow
C_1 \otimes C_2.
```

Thus:

```python
T.shape
```

follows

```text
(C1, D1, C2, D2)
```

while

```python
T.codomain
T.domain
```

return

```text
C1 ⊗ C2
D1 ⊗ D2
```

respectively.

This is the key mechanism that permits ndarray-style axis ordering without losing TensorMap semantics.

---

# Why keep both `side` and `dual`?

One could instead convert every morphism

```math
T:D\to C
```

to an all-outgoing tensor using rigidity:

```math
\mathrm{Hom}(D,C)
\simeq
\mathrm{Hom}
\left(
\mathbf 1,
C\otimes D^*
\right).
```

For ordinary vector spaces this often makes input/output information appear redundant.

For a general rigid category, however, this conversion is implemented by evaluation and coevaluation morphisms.

Changing

```text
IN → OUT
```

is therefore a categorical **bend**, not merely a change of a Boolean flag.

At the reduced level it may require:

- duality maps;
- normalization factors;
- Frobenius-Schur factors;
- changes of fusion basis;
- \(B\)-symbols or equivalent bending coefficients.

Therefore `TeNeT-py` keeps both pieces of metadata.

```text
side
    tells us where the object lives in Hom(D,C)

dual
    tells us whether that object is V or V*
```

This allows the user-facing representation to retain full morphism semantics without forcing input and output legs to occupy contiguous axis ranges.

---

# TensorMap views

Although `TensorMap` is not the primary owning type, TensorMap operations remain first-class.

A tensor can expose a map view:

```python
M = T.as_map()
```

with

```python
M.domain
M.codomain
M.compose(...)
M.adjoint()
M.svd()
```

The default map view uses the `side` metadata already stored on the legs.

No categorical transformation is required merely to ask for this view.

Conceptually:

```text
SymmetricTensor

axes = [C1, D1, C2, D2]
side = [OUT, IN, OUT, IN]

              │
              ▼

TensorMapView

codomain = C1 ⊗ C2
domain   = D1 ⊗ D2
```

The map view is therefore a **semantic view**, not necessarily a materialized rearrangement of block storage.

---

# Repartitioning is different from viewing

Changing which legs are inputs and outputs is an actual categorical operation.

For example:

```python
T2 = T.repartition(
    outputs=(0, 1, 2),
    inputs=(3,),
)
```

may move an existing input leg to the output side or vice versa.

This is not equivalent to:

```python
leg.side = OUT
```

because it may require bending a categorical line.

The lowering is conceptually:

```text
old Hom(D, C)
      │
      ▼
evaluation / coevaluation
      │
      ▼
fusion-basis transformation
      │
      ▼
new Hom(D', C')
```

The public API may be compact, but the operation remains categorically exact.

---

# Public axis order

The public axis order exists primarily for ndarray-style programming.

It determines:

- `shape`;
- `ndim`;
- `axis`;
- einsum labels;
- tensordot axes;
- user-facing block dimensions;
- output ordering.

It is intentionally not required to be

```text
all outputs followed by all inputs.
```

Thus the following is valid:

```text
OUT, IN, OUT, IN, OUT
```

This is one of the main differences from a TensorMap-first public API.

---

# `transpose` is an array operation with categorical semantics

A call such as

```python
B = A.transpose(2, 0, 1)
```

means:

> return the same abstract tensor with public axes ordered as `(2, 0, 1)`.

It must not be implemented blindly as

```python
backend.transpose(block, ...)
```

for every symmetry.

The structural layer first determines what changed.

Possible cases include:

1. only reduced numerical axes need to be reordered;
2. the relative order of categorical factors changes;
3. a fusion-tree basis must be transformed;
4. an \(F\)-move is required;
5. a swap carries a fermionic sign;
6. a braid requires an \(R\)-move.

Only after this analysis does the operation lower to backend array primitives.

Thus:

```text
public transpose
       │
       ▼
categorical analysis
       │
       ├── trivial?
       ├── F-transform?
       ├── swap phase?
       └── braid?
       │
       ▼
backend numerical program
```

A raw backend transpose remains a low-level primitive.

It is not the semantic definition of tensor permutation.

---

# Fusion spaces

## `GradedSpace`

A symmetry-graded space has the form

```math
V
=
\bigoplus_a
\mathbb C^{m_a}
\otimes
V_a,
```

where:

- \(a\) is a sector;
- \(V_a\) is the corresponding simple or irreducible object;
- \(m_a\) is its degeneracy.

For example,

```math
V
=
2 V_{1/2}
\oplus
3 V_1
```

can be represented as

```python
V = GradedSpace(
    provider=SU2,
    sectors={
        SU2Sector(two_j=1): 2,
        SU2Sector(two_j=2): 3,
    },
)
```

The degeneracy

```math
m_a
```

is distinct from the irrep dimension or quantum dimension of \(a\).

Reduced ndarray axes contain only these degeneracy dimensions.

---

## `ProductSpace`

`ProductSpace` remains useful, but mainly as a TensorMap-level object.

For example:

```python
T.codomain
```

may return

```python
ProductSpace(C1, C2, C3)
```

and

```python
T.domain
```

may return

```python
ProductSpace(D1, D2)
```

The user does not need to construct a `ProductSpace` merely to create an ordinary tensor with five axes.

This reverses the emphasis of a TensorMap-first interface:

```text
old

ProductSpace
     ↓
TensorMap
     ↓
axes

new

Legs / axes
     ↓
SymmetricTensor
     ↓
derived ProductSpace views
```

---

# Fusion trees are tensor-level structure

Fusion-tree information must **not** be placed independently on individual legs.

For example,

```math
(a\otimes b)\otimes c
```

and

```math
a\otimes(b\otimes c)
```

differ through relations involving multiple legs.

Likewise, an intermediate sector

```math
e
```

in

```math
(a\otimes b)\to e,\qquad
(e\otimes c)\to d
```

does not belong to any single external edge.

Therefore the structure is separated as:

```text
Leg
    local object information

TensorStructure
    ordered collection of legs

FusionBasis
    relationships among several legs

ExecutionPlan
    temporary basis transformations and kernels
```

This prevents the `Leg` type from becoming a container for global categorical state.

---

# Fusion trees

For non-Abelian fusion,

```math
a\otimes b
=
\bigoplus_c N_{ab}^c\,c.
```

External sectors alone may not specify a basis.

For three objects, a left-associated basis can contain

```text
a        b        c
 \      /
    e
     \
      \
       d
```

where the intermediate sector \(e\) is part of the basis.

With fusion multiplicity, additional labels are required:

```math
\mu = 1,\ldots,N_{ab}^{e}.
```

A conceptual representation is:

```python
FusionTree(
    uncoupled=(a, b, c),
    intermediate=(e,),
    multiplicities=(mu0, mu1),
    coupled=d,
)
```

The exact representation may evolve, but the design invariant is:

> Fusion channels and fusion multiplicities are categorical basis labels, not ndarray dimensions.

---

# Tensor-level fusion basis

Since every tensor still has an underlying domain and codomain, its reduced data can naturally use a pair of fusion trees:

```text
output fusion/splitting tree
            │
      coupled sector c
            │
input fusion tree
```

Schematically:

```text
C1        C2        C3
 \         |        /
   output fusion tree
           |
           c
           |
    input fusion tree
        /       \
       D1       D2
```

This retains the main TensorMap advantage:

```math
T
\simeq
\bigoplus_c
B_c \otimes \mathrm{id}_c.
```

The coupled-sector matrices \(B_c\) remain available for:

- composition;
- SVD;
- QR;
- eigendecomposition;
- polar decomposition;
- matrix functions.

The difference is that this matrix representation is no longer forced onto the public axis layout.

---

# Block keys

A logical reduced component can be identified by a pair of compatible trees:

```python
FusionBlockKey(
    output_tree=...,
    input_tree=...,
)
```

with compatible coupled sector.

The block key contains categorical information.

The corresponding value contains only numerical degeneracy data.

Logically, the tensor associates

```python
FusionBlockKey  →  Array
```

Physically, however, the blocks are stored as an ordered tuple

```python
blocks: tuple[Array, ...]
```

whose order is fixed by `structure.block_order`. The keys live in the
structural metadata; the tuple contains only backend arrays. This keeps the
numerical data a clean parameter tree (for `get_params`/`set_params` and
optional JAX PyTree registration): the leaves are exactly the blocks, and no
dict ordering or key hashing enters the dynamic data. A `Mapping`-style view
(`T.block(key)`) is derived, not primary.

---

# Reduced blocks follow public ndarray axes

For a fixed fusion-tree pair and fixed external sectors, the reduced data form an ordinary multidimensional tensor.

Suppose the public leg order is

```text
(C1, D1, C2, D2)
```

with sector degeneracies

```text
(m_c1, m_d1, m_c2, m_d2).
```

Then the corresponding block is stored with

```python
block.shape == (
    m_c1,
    m_d1,
    m_c2,
    m_d2,
)
```

in the same order as the public tensor axes.

This is an important design choice.

The logical block does **not** need to be stored as

```text
all output degeneracies × all input degeneracies
```

merely because the tensor has TensorMap semantics.

When matrix linear algebra is required, the relevant block data are lowered to the appropriate matrix layout.

Thus:

```text
logical block
     multidimensional ndarray
             │
             ▼
temporary map lowering
             │
             ▼
coupled-sector matrix
             │
             ▼
BLAS / SVD / QR / ...
```

This gives ndarray-style storage semantics without discarding the map interpretation.

---

# Three kinds of indices

The implementation must keep three different notions of index separate.

## 1. Public tensor axes

These correspond to tensor legs.

Examples:

```text
axis 0
axis 1
axis 2
```

They are visible to APIs such as:

```python
transpose
tensordot
einsum
svd(..., axes=...)
```

---

## 2. Reduced degeneracy indices

For sector \(a\) with multiplicity \(m_a\),

```math
\alpha = 1,\ldots,m_a
```

is a numerical index.

These are actual ndarray dimensions inside reduced blocks.

For example,

```math
A_{\alpha\beta\gamma}
```

may be represented by

```python
block.shape == (m_a, m_b, m_c)
```

---

## 3. Categorical basis indices

Examples include:

- sector labels;
- intermediate fusion sectors;
- fusion multiplicity;
- fusion-tree labels;
- coupled sectors.

These are static structural metadata.

They are **not** ndarray axes.

---

## 4. Irrep basis indices

An SU(2) sector \(j\) also has magnetic quantum-number indices

```math
m=-j,\ldots,j.
```

These are represented implicitly by symmetry tensors and are absent from the reduced numerical data.

Thus:

```text
physical tensor leg
        │
        ├── public leg identity
        │
        ├── sector
        │       categorical
        │
        ├── reduced degeneracy index
        │       ndarray
        │
        └── irrep basis index
                implicit symmetry structure
```

---

# Symmetry providers

Symmetry-specific mathematics is supplied through capabilities rather than central branching.

Avoid:

```python
if symmetry == "u1":
    ...
elif symmetry == "su2":
    ...
elif symmetry == "fibonacci":
    ...
```

Instead:

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

Additional capabilities can include:

```text
fusion multiplicity
rigidity
evaluation
coevaluation
pivotal structure
F-symbols
R-symbols
Frobenius-Schur data
braiding type
categorical coefficient dtype
```

Tensor algorithms dispatch on the capabilities they require.

For example:

```text
transpose
    requires permutation capability

repartition
    requires rigidity

recouple
    requires F-symbols

braid
    requires R-symbols
```

This keeps symmetry definitions independent from the numerical backend.

---

# Initial providers

The first structural implementation should include at least:

```text
Trivial
U(1)
SU(2)
```

SU(2) should be present early.

A design tested only with Abelian fusion can accidentally encode assumptions that fail once intermediate fusion channels become non-unique.

The initial non-Abelian implementation should therefore validate from the beginning that:

```text
external sectors
        !=
complete fusion basis
```

---

# Array API philosophy

`SymmetricTensor` should look like an array where doing so is mathematically meaningful.

It should not pretend to be an arbitrary dense ndarray.

The intended surface includes properties such as:

```python
T.ndim
T.shape
T.dtype
T.backend
T.device
```

and operations such as:

```python
tenet.transpose
tenet.tensordot
tenet.einsum
tenet.trace
tenet.conj
tenet.linalg.svd
tenet.linalg.qr
```

where their categorical meaning is well-defined.

---

# Do not subclass `numpy.ndarray`

The tensor itself should remain a Python structural object containing backend-native arrays.

It should not subclass:

```python
numpy.ndarray
jax.Array
torch.Tensor
```

The symmetry metadata and block structure do not fit naturally into any one of these storage classes.

Instead:

```text
SymmetricTensor
    │
    ├── static categorical structure
    │
    └── backend arrays
```

The backend arrays themselves remain completely native.

---

# Do not implicitly densify

Operations such as

```python
np.asarray(T)
```

should not silently expand a symmetric tensor to its full dense representation.

Dense expansion may:

- require Clebsch-Gordan tensors;
- allocate dramatically more memory;
- destroy the block-sparse representation;
- obscure categorical structure.

Dense conversion should remain explicit:

```python
T.to_dense()
```

---

# Array protocol integration

The goal is **array-protocol compatibility**, not universal ndarray substitutability.

The integration points, in order of importance:

## 1. `autoray` dispatch (core)

```python
import autoray as ar

C = ar.do(
    "tensordot",
    A,
    B,
    axes=((2,), (0,)),
)
```

uses the symmetric implementation while the reduced blocks are themselves
dispatched through `autoray` to whatever backend they belong to. This is the
symmray model, and it is the primary protocol.

## 2. Parameter extraction (quimb-compatible)

`SymmetricTensor` implements the `get_params` / `set_params` protocol:

```python
params = T.get_params()      # tree of backend arrays (the blocks)
T2 = T.set_params(params)    # same structure, new numerical data
```

This is exactly the interface `quimb`'s `TNOptimizer` uses to run JAX /
PyTorch / TensorFlow optimization over structured tensors without any
framework-specific registration inside the library. Variational workflows
(VMC, PEPS optimization) get AD and JIT this way, at the application level.

## 3. Optional JAX PyTree registration

For users driving raw `jax.jit` / `jax.grad` / `jax.vmap` without quimb, a
small opt-in module registers the tensor as a PyTree:

```python
import tenet.pytree  # requires jax; import-guarded, never imported by core
```

after which

```python
jax.tree.leaves(T) == list(T.blocks)
```

and a gradient with respect to `T` is another `SymmetricTensor` with the same
structure. This works because `blocks` is an ordered tuple of arrays and
`TensorStructure` is immutable, hashable, and array-free — F/R matrices and
other categorical coefficient arrays are never stored in structural metadata,
which would otherwise break treedef hashing and JIT cache keys.

None of these make `jnp.einsum(...)` or `np.einsum(...)` work on
`SymmetricTensor` directly — symmetric operations always go through
`tenet.*` / `ar.do(...)`, which lower to backend array programs on the blocks.

---

# Supported ndarray operations should be explicit

Not every NumPy operation is meaningful for symmetric tensors.

For example, arbitrary:

```text
broadcasting
physical-index slicing
elementwise nonlinear functions
arbitrary reshape
```

can change or destroy symmetry structure.

Therefore the library should define a deliberate supported array surface rather than forwarding every unknown operation to the backend.

The rule is:

> An ndarray-style API is provided when the operation has a clear categorical interpretation.

Unsupported operations should fail clearly rather than silently produce a mathematically different object.

---

# `reshape` and fusion

A generic dense reshape such as

```python
x.reshape(6, 20, 7)
```

does not by itself specify how graded representation spaces should be fused.

Therefore the primitive categorical operation is:

```python
T.fuse(...)
```

with an inverse:

```python
T.unfuse(...)
```

A NumPy-like `reshape` may be provided only when the requested transformation can be resolved unambiguously into leg fusion or splitting.

Conceptually:

```text
public reshape request
        │
        ▼
resolve leg grouping
        │
        ▼
categorical fusion
        │
        ▼
backend reshape / concatenate
```

The distinction between logical and physical fusion should remain available.

For example:

```text
logical fusion
    update axis structure while retaining fine block layout

physical fusion
    materialize larger reduced blocks for efficient kernels
```

These need not initially be separate public APIs, but the architecture should permit the distinction.

---

# Contraction

The main tensor-network API should be axis-oriented.

For example:

```python
C = tenet.tensordot(
    A,
    B,
    axes=((2, 3), (0, 1)),
)
```

or:

```python
C = tenet.einsum(
    "abcd,cdef->abef",
    A,
    B,
)
```

The einsum equation describes the **network-level contraction**, not the reduced block kernel directly.

The implementation performs:

```text
einsum equation
       │
       ▼
match public legs
       │
       ▼
validate categorical compatibility
       │
       ▼
insert required bends / evaluations
       │
       ▼
determine compatible fusion channels
       │
       ▼
recouple / braid if required
       │
       ▼
construct output structure
       │
       ▼
lower to backend array operations
       │
       ▼
SymmetricTensor
```

Thus users can write ordinary tensor-network notation while the categorical layer handles the nontrivial structure.

---

# Composition remains first-class

TensorMap composition is a special and important operation.

For:

```math
A:V\to W,
\qquad
B:U\to V,
```

the operation

```python
C = A @ B
```

can represent

```math
C=A\circ B:U\to W.
```

The implementation can verify exact domain/codomain compatibility and exploit the coupled-sector block-matrix representation directly.

Thus the ndarray API does not replace TensorMap operations.

Both are available:

```text
einsum / tensordot
    tensor-network view

compose / @
    morphism view
```

They operate on the same underlying `SymmetricTensor`.

---

# Bending and contraction

A general einsum may contract legs that are not already arranged as direct TensorMap composition.

The planner may therefore need to transform:

```text
OUT ↔ IN
```

internally.

Such transformations use rigidity and must be represented explicitly in the structural plan.

This is one reason to retain full TensorMap semantics even though the user-facing syntax is ndarray-like.

---

# Conjugation, duality, and adjoint are distinct

The API should not conflate:

```text
complex conjugation
dual object
categorical adjoint
axis reversal
input/output swap
```

For example:

```python
T.conj()
```

and

```python
T.adjoint()
```

are not generally the same operation.

Likewise:

```python
leg.dualized()
```

changes the categorical object associated with a leg, while:

```python
T.repartition(...)
```

changes its location between domain and codomain.

These operations may coincide numerically in simple vector-space cases but should remain semantically distinct.

---

# Linear algebra

TensorMap semantics become particularly valuable for matrix decompositions.

For an arbitrary tensor:

```python
U, S, Vh = tenet.linalg.svd(
    T,
    axes=((0, 2), (1, 3)),
)
```

the user selects the matrix bipartition using ndarray axes.

The lowering is:

```text
SymmetricTensor
       │
       ▼
requested axis partition
       │
       ▼
temporary TensorMap representation
       │
       ▼
fusion-tree basis transformation
       │
       ▼
coupled-sector block matrices B_c
       │
       ▼
backend SVD
       │
       ▼
symmetric U, S, Vh
```

If the current input/output partition already matches the requested split, the temporary map conversion can be trivial.

A map-oriented convenience API may also exist:

```python
U, S, Vh = T.as_map().svd()
```

using the current domain/codomain directly.

Thus both styles are supported:

```text
ndarray style
    svd(T, axes=(left, right))

TensorMap style
    T.as_map().svd()
```

---

# Why retain the coupled-sector matrix representation?

For a symmetric morphism, Schur-type decomposition gives the reduced form

```math
T
\simeq
\bigoplus_c
B_c \otimes \mathrm{id}_c.
```

The matrices \(B_c\) are the natural objects for:

```text
composition
SVD
QR
eigh
polar decomposition
matrix exponential
```

The public ndarray representation should therefore not eliminate this structure.

Instead, the library should make it an internal or explicit map-level view.

```text
array-oriented logical block
           │
           ▼
MapLayout
           │
           ▼
coupled-sector matrix B_c
```

This is one of the main places where TensorMap semantics provide real computational value rather than merely mathematical notation.

---

# Logical storage versus execution layout

The logical representation should not dictate the eventual GPU execution format.

Initially:

```python
blocks: tuple[Array, ...]      # ordered by structure.block_order
```

is attractive because it is:

- simple;
- explicit;
- backend independent;
- a clean parameter tree;
- easy to inspect;
- easy to test.

For example:

```text
key A → jax.Array(shape=(2, 4, 3))
key B → jax.Array(shape=(2, 4, 3))
key C → jax.Array(shape=(5, 1, 8))
```

However, GPU execution may prefer:

```text
stack blocks with identical shape
batch GEMMs
group contractions by kernel shape
```

Therefore later lowering may produce:

```text
logical blocks
      │
      ▼
shape buckets
      │
      ├── stack
      ├── reshape
      ├── gather
      └── coefficient transform
      │
      ▼
batched backend kernels
```

The invariant is:

```text
logical representation
        !=
execution representation
```

Optimization must remain below the mathematical API.

---

# No custom ndarray implementation

`TeNeT-py` should not implement:

```text
its own dense ndarray
its own stride system
its own GPU memory allocator
its own autograd engine
```

Backend arrays already solve these problems.

The categorical layer should instead produce numerical programs expressed using a small collection of backend operations such as:

```text
transpose
reshape
stack
concatenate
tensordot
einsum
matmul
svd
qr
eigh
```

---

# Backend dispatch

`autoray` provides the backend abstraction.

Conceptually:

```python
ar.do("transpose", x, axes)
ar.do("reshape", x, shape)
ar.do("matmul", x, y)
ar.do("einsum", equation, *arrays)
```

The array-dispatch layer should remain intentionally small.

Avoid creating a second numerical framework on top of `autoray`.

The architecture is:

```text
categorical operation
       │
       ▼
structural lowering
       │
       ▼
small numerical IR
       │
       ▼
autoray
       │
   ┌───┼────┐
   │   │    │
 NumPy JAX PyTorch
```

Framework-specific optimization (`jit`, `grad`, `vmap`, `torch.compile`,
device placement) is the responsibility of the application layer, following
the symmray/quimb model. The library's obligation is to stay *traceable*:
structural logic runs in Python on static metadata, so a JAX trace or torch
AD pass sees only clean backend array operations.

---

# Structural planning

Categorical analysis should be separated from numerical execution.

A contraction should conceptually generate a static plan containing information such as:

```text
input block matches
output block keys
fusion-tree transforms
F coefficients
R coefficients
axis permutations
matrix reshapes
accumulation targets
```

The numerical backend then executes only the resulting array program.

```text
SymmetricTensor operation
            │
            ▼
     StructurePlanner
            │
            ▼
       OperationPlan
            │
      static metadata
            │
            ▼
     numerical program
            │
            ▼
      backend arrays
```

This separation is particularly important for JAX.

---

# Plan caching

Structural plans are determined primarily by:

```text
symmetry provider
provider gauge/convention
leg spaces
leg sides
leg dualities
fusion basis
operation
axis pattern
```

They are often reusable across many numerical evaluations.

This is especially important in:

- VMC;
- variational optimization;
- imaginary-time evolution;
- repeated PEPS contractions;
- neural quantum states;
- gradient calculations.

The categorical planning cost should therefore be amortizable.

---

# JAX

JAX is a first-class backend but does not define the tensor model, and core
never imports it. JAX-specific optimization lives at the application level
(e.g. quimb's `TNOptimizer` via `get_params`/`set_params`) or through the
opt-in `tenet.pytree` registration described above.

Under that registration, a JAX-backed tensor is conceptually a PyTree:

```text
SymmetricTensor
│
├── static auxiliary data
│   ├── legs
│   ├── spaces
│   ├── fusion basis
│   ├── block keys
│   └── provider identity
│
└── dynamic leaves
    ├── jax.Array
    ├── jax.Array
    └── ...
```

Only reduced numerical data are differentiable leaves.

Categorical metadata are static.

Since `blocks` is already an ordered tuple, flattening is trivial:

```text
leaves   = T.blocks
treedef  = T.structure  (static, hashable)
```

This aligns JIT caching with the mathematics. Under

```python
@jax.jit
def norm(T):
    return tenet.norm(T)
```

two tensors with the same `TensorStructure` (same provider, legs, fusion
basis, block shapes) share one compiled specialization; only block values
change. If the bond dimension or sector content changes, the treedef changes
and JAX recompiles — which is exactly the desired behavior, because the
structural plan genuinely differs.

Contraction planning happens at trace time regardless of how `jit` is
entered (raw PyTree or quimb-wrapped): the `structure` fields and `axes`
arguments are static Python data, so sector matching, fusion-tree matching,
and F/R coefficient selection run once during tracing, and only the
resulting array program is staged into XLA:

```text
   tracing time              runtime
        │                       │
  TensorStructure            jax.Array blocks
        │                       │
  categorical planner           │
        │                       │
  static array program          │
        └───────────┬───────────┘
                    ▼
                   XLA
                    ▼
              CPU / CUDA / TPU
```

---

# Automatic differentiation

For fixed tensor structure, operations should be differentiable through the backend.

For example:

```text
F-move
    linear combination of blocks

contraction
    einsum / matmul

permutation
    transpose + linear transforms

norm
    backend reduction
```

can all participate naturally in AD.

The categorical coefficients are constants.

The trainable variables are the reduced block entries.

Thus:

```math
\frac{\partial E}
     {\partial A^{(\tau)}_{\alpha\beta\cdots}}
```

is computed by the backend AD system while the categorical labels \(\tau\) remain static.

---

# Structure-changing differentiation

Operations that change block dimensions require additional care.

Examples include:

```text
SVD truncation
charge-sector selection
data-dependent bond-dimension allocation
removing zero sectors
```

These operations may produce data-dependent Python structure or array shapes.

That conflicts with ordinary JAX JIT assumptions.

The architecture should therefore distinguish:

```text
fixed-structure differentiable operations
```

from

```text
structure-changing operations
```

Possible strategies for the latter include:

- perform structural decisions outside JIT;
- use static truncation dimensions;
- use masks or padding;
- explicitly recompile after a structural change.

The library should not hide this distinction.

---

# VMC and neural quantum states

A major goal is to make symmetric tensor structures usable inside ML-style computational workflows.

For example:

```text
symmetric PEPS / MPS parameters
        │
        ▼
JAX PyTree of reduced arrays
        │
        ▼
batched amplitude evaluation
        │
        ▼
jax.vmap
        │
        ▼
Monte Carlo samples
        │
        ▼
energy / fidelity objective
        │
        ▼
jax.grad
```

Likewise, hybrid ansätze can combine:

```text
neural network parameters
+
symmetric tensor parameters
```

inside the same differentiable program.

The library should make this composition straightforward rather than imposing a separate tensor-network runtime abstraction.

This is one of the main reasons to keep reduced data as ordinary backend arrays.

---

# Integration with tensor-network tooling

External contraction planners should not need to understand fusion trees.

For path optimization, the relevant information is often only:

```text
tensor labels
axis labels
effective dimensions
output labels
```

Thus an einsum network can expose a dense-style planning problem to tools such as:

```text
opt_einsum
cotengra
```

while `TeNeT-py` retains responsibility for actual symmetric execution.

```text
SymmetricTensor network
         │
         ├── metadata → contraction planner
         │
         │                 │
         │                 ▼
         │            contraction path
         │
         ▼
categorical execution planner
         │
         ▼
reduced block kernels
```

The external planner chooses **which tensors to contract**.

It does not decide:

```text
fusion channels
recoupling
fermionic signs
braiding
block layout
```

---

# Braiding is the important limit of ndarray syntax

For ordinary symmetric monoidal categories, axis permutation has a canonical symmetric swap.

For fermionic or super vector spaces, permutation carries a canonical graded sign.

These cases can fit naturally behind familiar APIs such as:

```python
transpose
einsum
tensordot
```

For a genuinely braided category, however, a permutation is not uniquely specified by the final axis order.

For example, exchanging two anyon lines can involve:

```math
R
```

or

```math
R^{-1},
```

depending on over/under crossing.

A bare expression such as

```python
T.transpose(1, 0, 2)
```

does not encode that information.

Therefore ndarray compatibility must depend on provider capabilities.

Possible policy:

```text
symmetric category
    ordinary ndarray permutation API

fermionic / graded-symmetric category
    ordinary API with automatic graded signs

generic braided category
    require explicit braid or planar contraction semantics
```

For example:

```python
T.braid(0, 1, over=True)
```

or:

```python
tenet.planar_einsum(...)
```

may be required.

The library must not invent an implicit braid convention merely to imitate NumPy.

Correct categorical semantics take priority over superficial API compatibility.

---

# Array operations versus categorical operations

The following must remain conceptually distinct:

```text
backend ndarray transpose
public tensor transpose
F-move
R-move
bend / repartition
dualization
complex conjugation
adjoint
fusion
```

A high-level operation may lower to several of these.

For example:

```text
T.transpose(...)
       │
       ├── reorder reduced axes
       ├── recouple fusion tree
       └── apply braid coefficients
```

depending on the provider and axis pattern.

This distinction is a core architectural invariant.

---

# Example API

Consider:

```python
from tenet import (
    GradedSpace,
    Leg,
    SymmetricTensor,
    IN,
    OUT,
)

from tenet.symmetry import SU2, SU2Sector
```

Define sectors:

```python
half = SU2Sector(two_j=1)
one = SU2Sector(two_j=2)
```

and graded spaces:

```python
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

A tensor with ndarray-facing axis order

```text
(V, W, V)
```

can be constructed as:

```python
T = SymmetricTensor(
    legs=(
        Leg(V, side=OUT),
        Leg(W, side=IN),
        Leg(V, side=OUT),
    ),
    blocks=blocks,
)
```

Then:

```python
T.ndim == 3
```

and conceptually:

```python
T.codomain == V @ V
T.domain == W
```

even though the input axis occurs between the two output axes in public axis order.

---

# Array-style contraction

Suppose:

```python
A.legs == (a, b, c)
B.legs == (c_dual, d, e)
```

Then:

```python
C = tenet.einsum(
    "abc,cde->abde",
    A,
    B,
)
```

is the preferred tensor-network syntax.

The equation controls the public axes.

The categorical planner controls how the contraction is actually represented and executed.

---

# Map-style composition

For operator-like tensors:

```python
C = A @ B
```

uses the map semantics directly.

This requires the relevant domain and codomain spaces to match.

No einsum equation is necessary.

---

# Factorization

Array style:

```python
U, S, Vh = tenet.linalg.svd(
    T,
    axes=((0, 2), (1, 3)),
)
```

Map style:

```python
U, S, Vh = T.as_map().svd()
```

Both ultimately use the same coupled-sector block-matrix implementation.

---

# Backend conversion

Backend changes should be explicit:

```python
T_jax = T.to_backend("jax")
T_torch = T.to_backend("torch")
T_numpy = T.to_backend("numpy")
```

One tensor should normally use one numerical backend.

The categorical structure is unchanged.

Only the reduced numerical arrays move.

---

# Dense expansion

Dense expansion remains explicit:

```python
dense = T.to_dense()
```

For an SU(2) tensor, schematically,

```math
T_{(\alpha_1,m_1)\cdots(\alpha_N,m_N)}
=
\sum_\tau
A^{(\tau)}_{\alpha_1\cdots\alpha_N}
C^{(\tau)}_{m_1\cdots m_N},
```

where:

- \(\alpha_i\) are reduced degeneracy indices;
- \(m_i\) are irrep basis indices;
- \(\tau\) denotes fusion-basis information;
- \(A^{(\tau)}\) is stored;
- \(C^{(\tau)}\) is determined by the symmetry provider.

Full irrep basis indices appear only in this explicit expansion.

---

# Proposed package structure

```text
TeNeT-py/
│
├── src/
│   └── tenet/
│       ├── __init__.py
│       │
│       ├── symmetry/
│       │   ├── base.py
│       │   ├── trivial.py
│       │   ├── u1.py
│       │   ├── su2.py
│       │   └── ...
│       │
│       ├── space.py
│       ├── leg.py
│       ├── fusion_tree.py
│       ├── structure.py
│       ├── tensor.py
│       ├── map_view.py
│       │
│       ├── ops/
│       │   ├── basic.py
│       │   ├── permutation.py
│       │   ├── contraction.py
│       │   ├── fusion.py
│       │   └── linalg.py
│       │
│       ├── planning/
│       │   ├── transform.py
│       │   ├── contraction.py
│       │   └── cache.py
│       │
│       ├── array/
│       │   ├── dispatch.py
│       │   └── params.py
│       │
│       ├── pytree.py
│       │
│       └── network/
│           ├── einsum.py
│           └── planning.py
│
├── tests/
│   ├── symmetry/
│   ├── tensor/
│   ├── ops/
│   ├── backends/
│   └── integration/
│
└── pyproject.toml
```

There should no longer be a conceptual requirement that

```text
tensor_map.py
```

contain the main owning tensor abstraction.

TensorMap functionality belongs mainly in:

```text
map_view
structural lowering
linear algebra
```

---

# Initial implementation strategy

## Milestone 1 — Semantic foundation

Implement:

- immutable sector types;
- `FusionProvider`;
- trivial symmetry;
- U(1);
- SU(2);
- `GradedSpace`;
- `Leg`;
- independent `side` and `dual`;
- `FusionTree`;
- fusion multiplicity labels;
- `FusionBlockKey`;
- `TensorStructure`;
- `SymmetricTensor` (immutable hashable structure, ordered tuple of blocks);
- derived `domain` and `codomain`;
- NumPy reduced blocks.

The first milestone should already demonstrate a genuinely non-Abelian SU(2) tensor.

---

## Milestone 2 — ndarray-style surface

Implement:

- `ndim`;
- `shape`;
- `dtype`;
- backend detection;
- addition;
- scalar multiplication;
- conjugation;
- norm;
- axis permutation;
- explicit fusion/unfusion;
- `autoray` registration.

At this stage, only operations with clear categorical definitions should be exposed.

---

## Milestone 3 — TensorMap operations

Implement:

- `as_map()`;
- composition;
- identity;
- adjoint;
- repartition;
- bending;
- map compatibility checks.

This milestone validates that the ndarray-facing representation still reproduces full TensorMap semantics.

---

## Milestone 4 — Non-Abelian basis transforms

Implement:

- \(F\)-moves;
- recoupling;
- duality coefficients;
- categorical permutations;
- Frobenius-Schur data where required;
- \(R\)-moves for supported braided providers.

The implementation should separate:

```text
structural plan
```

from:

```text
backend execution
```

from the beginning.

---

## Milestone 5 — `tensordot` and `einsum`

Implement general contraction through public axes.

Pipeline:

```text
einsum
  ↓
axis matching
  ↓
categorical validation
  ↓
map/bending resolution
  ↓
fusion-tree matching
  ↓
basis transforms
  ↓
backend contractions
  ↓
output SymmetricTensor
```

Add path-planner integration only after pairwise contraction is correct.

---

## Milestone 6 — JAX and AD

Implement:

- `get_params` / `set_params` (quimb-compatible parameter extraction);
- optional `tenet.pytree` registration (static structure, dynamic block leaves);
- `jax.grad`;
- `jax.jit` (structural planning at trace time, plan reuse across calls);
- `jax.vmap`;
- fixed-structure contraction tests;
- gradient tests against dense expansion.

This is a primary project milestone rather than a later optional backend feature.

---

## Milestone 7 — Linear algebra

Implement:

- QR;
- SVD;
- eigendecomposition;
- polar decomposition;
- truncation;
- graded bond-space reconstruction.

Both APIs should work:

```python
svd(T, axes=...)
```

and

```python
T.as_map().svd()
```

---

## Milestone 8 — ML and tensor-network integration

Target workflows such as:

- quimb-style tensor networks;
- cotengra contraction planning;
- differentiable MPS/PEPS;
- tensor-network VMC;
- variational PEPS;
- hybrid NQS/TN ansätze;
- batched wavefunction amplitudes.

The goal is not merely backend compatibility in unit tests.

The goal is composition with existing scientific-ML workflows.

---

## Milestone 9 — Performance

Optimize after profiling.

Candidate optimizations include:

- shape bucketing;
- stacked reduced blocks;
- batched GEMM;
- grouped GEMM;
- cached fusion transforms;
- cached contraction plans;
- static JAX specializations;
- backend-specific fast paths;
- custom GPU kernels where justified.

These changes must not alter the public semantic representation.

---

# Explicit non-goals

The initial project does not aim to be:

- a Python port of TensorKit;
- a Python port of Rust TeNeT;
- a complete NumPy drop-in replacement;
- a custom ndarray framework;
- a custom autograd engine;
- a custom GPU runtime;
- a manually packed storage system;
- an implicit dense tensor implementation;
- a system that hides ambiguous anyonic braiding behind NumPy syntax.

The project should reuse the Python numerical ecosystem rather than reproduce it.

---

# Core design invariants

The following should remain true throughout development.

## 1. Every tensor has exact TensorMap semantics

For every `SymmetricTensor`,

```python
T.domain
T.codomain
```

must be well-defined from its leg metadata.

---

## 2. `side` and `dual` are independent

Never identify:

```text
input == dual
```

or:

```text
output == non-dual
```

They represent different categorical information.

---

## 3. Public axis order is independent of domain/codomain grouping

Users may write tensor-network code using arbitrary ndarray-like axis order.

---

## 4. Fusion-tree information is relational

Intermediate sectors, multiplicity channels, and fusion bases belong to tensor-level categorical structure, not individual legs.

---

## 5. Categorical operations are never defined by backend operations

A backend transpose may implement part of a categorical permutation.

It never defines what that permutation means.

---

## 6. Reduced blocks contain only numerical degeneracy indices

Sector labels and fusion channels remain structural metadata.

---

## 7. TensorMap block-matrix structure remains available

Coupled-sector matrices are retained as the canonical lowering for composition and linear algebra.

---

## 8. Numerical leaves are backend-native arrays

NumPy, JAX, and PyTorch should see ordinary arrays at the numerical
boundary. Structural metadata is immutable, hashable, and array-free — F/R
and other coefficient arrays never live in structural fields.

---

## 9. AD acts on reduced numerical data

Categorical metadata remain static. Structure-changing operations
(truncation, sector selection) live outside JIT/compile boundaries, or use
static shapes/masks; the library never hides this distinction.

---

## 10. Execution optimization stays below the public API

Packing, bucketing, batching, caching, and kernel scheduling are execution concerns.

---

## 11. ndarray compatibility is semantic, not cosmetic

An operation should be exposed through an ndarray-like interface only when its categorical meaning is defined.

---

## 12. Braided categories remain explicit where necessary

If an ndarray expression does not uniquely specify a braid, the API must request additional information rather than guess.

---

# Project philosophy

The intended architecture can be summarized as:

```text
                         TeNeT-py
                            │
                            ▼
                    SymmetricTensor
                            │
             ordered ndarray-like legs
                            │
         ┌──────────────────┴──────────────────┐
         │                                     │
         ▼                                     ▼
 TensorMap semantics                    numerical values
         │                                     │
 Hom(D, C)                            backend-native arrays
 side + dual                                  │
 fusion trees                                  │
 multiplicity                                 │
 duality                                      │
 F / R                                        │
         │                                     │
         └──────────────────┬──────────────────┘
                            │
                            ▼
                    structural planner
                            │
                            ▼
                    numerical program
                            │
                            ▼
                         autoray
                            │
              ┌─────────────┼─────────────┐
              │             │             │
            NumPy          JAX         PyTorch
                            │
                  ┌─────────┼─────────┐
                  │         │         │
                 GPU       grad      jit/vmap
                            │
                            ▼
              tensor-network / ML workflows
                            │
               ┌────────────┼────────────┐
               │            │            │
              VMC          PEPS         NQS
```

The project should therefore be described by three sentences:

> **TensorMap is the semantic model.**

> **SymmetricTensor is the ndarray-like programming model.**

> **Backend-native arrays are the execution model.**

This boundary is the main design of `TeNeT-py`.