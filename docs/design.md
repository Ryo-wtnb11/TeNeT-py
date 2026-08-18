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

# Which reference governs which part

Influence is not uniform: each area of the library follows one reference, and the
others are read but not copied.

| Area | Reference | Evidence in this repository |
| --- | --- | --- |
| 2D methods — CTMRG, PEPS | YASTN | `examples/ctmrg.py` was written against YASTN's design and M11b (#114) promoted it into `tenet.network`; the API-naming survey (#120) took `CTMEnv`, `ctmrg`, `Absorb` from that family |
| 1D interfaces and API shape | tenpy, YASTN | the trailing-underscore in-place convention (`dmrg_`, `sweep_`, `canonize_`) and the two-criteria convergence, energy plus Schmidt values, are YASTN's (#112); the one-record-per-sweep schedule shape came from SUNDMRG (#136) |
| Algorithm cores at quantum-chemistry scale | block2 | per-sweep schedule semantics and the guard requiring noise to be zero before convergence (#136); the prepared-operator idea (#141), which MPSKit's `JordanMPO_AC2` shares |
| Coefficients | racah | racah-py is a core dependency and the sole gauge source (M28); TensorKitSectors remains the SU(2) oracle of record and SUNRepresentations.jl the SU(N) one, consulted through racah rather than directly |
| The categorical model | TensorKit | as stated above, with a deliberate divergence: the M24 capability lattice (#158) uses capability protocols and a data/property split instead of TensorKit's space-carried dual flag and type hierarchy |

Where YASTN goes further than `tenet` — two-site unit cells, other CTM move
patterns, fPEPS — that is the direction to take when a caller needs it, not a
reason to invent a different design.

block2 is deliberately not the interface reference. Its `dmrg()` takes roughly
forty arguments with print levels and configuration dictionaries; #136 measured
most of that surface as unused by any caller here and refused it. What `tenet`
does take from block2 is the algorithmic core, and the remaining piece — keeping
the Hamiltonian symbolic through a sweep with complementary operators assembled
per cut — is what M30 designs.

Three things block2 offers are out of reach by construction, each for a stated
reason. Density-matrix and perturbative noise need a reduced density matrix and
an eigendecomposition per variant, while `tenet` splits with `svd_truncated`;
giving that up means giving up the quantum-dimension-weighted sector budget, and
#136 argued this against three references that agreed. MPI five-level
parallelism and disk-backed scratch are production HPC concerns, and there is no
distribution story here. The C++ execution model is not available to a library
that dispatches through `autoray` over backends.

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

A source-level comparison against TensorKit.jl (#142) showed the two libraries hold the same data in two spellings: TensorKit's `GradedSpace` carries its degeneracies **and** a `dual::Bool`, while `TeNeT-py`'s `GradedSpace` is dual-free and the flag lives on the `Leg`. The one operation that correspondence had left missing is `tenet.flip_dual(t, axes, inv=)`: it toggles the named legs' `dual` flags and relabels their spaces through `provider.dual`, keeping the tensor the same morphism by paying the Z-isomorphism's scalar `χ_a · θ_a` per flipped leg per fusion tree (the `FSIndicatorData` and `TwistData` capabilities since M24). Because the relabel and the flag toggle cancel inside `Leg.fused_sector`, the block set is unchanged and `flip_dual` is a per-block scalar multiply; `side` never moves — that remains `repartition`'s bend. The rest of TensorKit's space-level surface was reviewed and tiered in the same issue: `fuse(V₁,V₂)`, a subspace predicate, the orthogonal complement `⊖`, the unit-leg family and the bare twist are named follow-ups with triggering criteria, while the Deligne product `⊠` at the space level, `infimum`/`supremum`, the unit/zero-space family and the `CartesianSpace`/`ComplexSpace` analogues are refused — each either already has a `TeNeT-py` spelling one level up (`ProductProvider`, `TrivialProvider`) or has no caller.

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
class FusionRules(Protocol):
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

## `quimb` in practice

A `SymmetricTensor` is usable as the `data` of a `quimb.tensor.Tensor` today, with
no adapter on either side: the whole integration is the `autoray` registration plus
the `get_params` / `set_params` protocol. `TeNeT-py` never imports `quimb` or
`cotengra`; both are test-only dependencies in the dev group.

```python
import quimb.tensor as qtn

t = qtn.Tensor(data=T, inds=("b0", "p0", "b1"))   # t.backend == "tenet"
tn = qtn.TensorNetwork([t0, t1, t2])
tn.contract(output_inds=("b0", "p0", "p1", "p2", "b3"))
```

Works:

```text
Tensor construction, .shape, .size, .copy(), .transpose(), .conj(), .H
tensor_contract / @ / TensorNetwork.contract with explicit output_inds
get_params / set_params
TNOptimizer with the JAX autodiff backend
```

One caveat on the last line: `quimb`'s `Tensor.set_params` mutates its data object
in place and discards the return value, while `SymmetricTensor` is frozen and
returns a new tensor. Until that is reconciled, parameter injection needs a
one-line forwarding shim:

```python
qtn.Tensor.set_params = lambda self, p: self._set_data(self.data.set_params(p))
```

Without it `TNOptimizer` runs but reports a flat loss, because the optimized
parameters never reach the network.

`quimb` hands the contraction to `cotengra`, which lowers the network itself and
calls back **pairwise only** — `tensordot` for contractions, `einsum` for
label shuffles. No multi-operand `einsum` is ever requested.

Refused, and correctly so:

```text
Tensor.norm(), Tensor.split()   → reshape by shape has no categorical meaning;
                                  use fuse / unfuse, or tenet.linalg.svd_truncated
output_inds dropping an index   → summing an axis away is not equivariant
contraction to a rank-0 scalar  → see below
```

### Scalars are spelled with boundary legs

A `SymmetricTensor` has at least one leg, so `tn ^ all` — a contraction that leaves
no free leg — is refused. This costs nothing in practice: a network is contracted
down to its *boundary* legs, which for an MPS are trivial-sector legs of dimension
one.

```python
ket = qtn.TensorNetwork([...])                       # legs b0 ... bN
bra = qtn.TensorNetwork([...])                       # legs c0 ... cN, adjoint sites
out = (ket | bra).contract(output_inds=("b0", "bN", "c0", "cN"))
# out.data is a rank-4 SymmetricTensor holding a single 1x1x1x1 block
```

The number in that block is `⟨ψ|ψ⟩`. The rank-0 tensor is a separate design
question — it would require every leg's provider to be carried explicitly — and is
deliberately not answered by this pattern's existence.

**Open diagrams are tensors; closed diagrams exit to backend scalars, explicitly and by
name.** `tensordot`, `einsum` and `trace` never return a scalar — a contraction that
closes a network is a `ValueError`, and a `SymmetricTensor` still has no rank 0. Leaving
the tensor world is a separate, named call — `norm`, `full_trace`, `inner` — which
returns the backend's own scalar and is therefore traceable and differentiable.

`full_trace(t)` is `Σ_c qdim(c) · tr(M_c)`, the categorical trace of an endomorphism,
closing the map view (codomain against domain, in order) at any rank and refusing a
non-square map with the same `check_square` message `eigh`/`expm`/`eig`/`eigvals` raise.
It gives the *user* a scalar exit, not the tensor a rank-0 one (#126).

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

## What "PyTorch backend" means, exactly

Install it with the `torch` extra (`pip install tenet-py[torch]`).

Supported and tested (`tests/backends/test_torch.py`, issue #95): every public
op on torch blocks — arithmetic, `norm`, `transpose` (SU(2) braiding and
fermionic Koszul signs), `repartition`, `fuse`/`unfuse`, `adjoint`, `compose`,
`tensordot`, `trace`, `einsum`, `to_dense`/`from_dense`, `embed`/`restrict`,
`direct_sum`, `to_symmetry`, the whole of `tenet.linalg`, and `get_params`/`set_params`
— with results bit-identical to the NumPy ones for everything `tenet` computes
itself. **Eager** autograd works through the parameter protocol and nothing
else:

```python
t = t.set_params(tuple(b.detach().clone().requires_grad_(True) for b in t.get_params()))
tenet.norm(t).backward()
```

Not supported: `torch.compile`, `torch.jit`, `torch.func` transforms
(`grad`/`vmap`/`jacrev`), and `tenet.ad` — the broadened degenerate-SVD VJP is
JAX-only, so a degenerate SVD under torch gives `NaN` gradients. There is no
torch analogue of `tenet.pytree`; eager torch needs none. GPU/MPS placement is
untested; CI is CPU-only.

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

Both directions are backend-generic and traceable: `to_dense` returns an array of
the tensor's own backend and differentiates under `jax.grad`, and

```python
T = SymmetricTensor.from_dense(dense, legs, atol=None)
```

projects a dense carrier-basis array back onto the symmetric subspace. `legs` is
required — a dense array carries no categorical information. Input that is not
symmetric to `atol` (default `sqrt(eps) * ‖dense‖`, relative) is **refused**,
naming the residual and the offending sector tuple, never silently projected;
`atol=math.inf` is the documented "project, don't check" spelling, and the only
one that traces (the comparison is a concrete-value question).

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
│       └── network/          # M11: the driver layer
│           ├── common.py     #   spectrum, ones
│           ├── mps.py
│           ├── env.py
│           ├── dmrg.py
│           └── ctmrg.py      #   M11b: the traced half
│
├── tests/
│   ├── symmetry/
│   ├── tensor/
│   ├── ops/
│   ├── backends/
│   ├── network/
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
- `FusionRules`;
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

## Milestone 11 — `tenet.network`, the driver layer

`examples/dmrg.py` and `examples/ctmrg.py` proved that a complete tensor-network
algorithm can be written over the public API with nothing added to `src/tenet/`.
M11 promotes one algorithm's worth of that machinery into a library layer, under a
migration rule that keeps it honest: the example is rewritten on top of the layer and
its energies compare **exactly equal**, so the promotion cannot quietly reorder a
contraction.

`tenet.network` (M11a) contains:

- `MPS` — a finite open-boundary state on `(left bond OUT, physical OUT, right bond IN)`,
  with `center: int | None` and a `__setitem__` write barrier that repartitions a factor
  from `lq`, `qr` or `svd_truncated` back onto the site partition;
- `MPO` — a separate class, no shape flag, built by `from_w` through
  `SymmetricTensor.from_dense` at its default relative `atol`, so a wrong grading raises;
- `Env` — the `<psi|H|psi>` partial contractions keyed by *directed* bond, with
  `setup_`/`update_`/`clear_`, `heff2` and `measure()`;
- `lanczos`, `sweep_`, `dmrg_()` and `DMRG_out`.

`tenet.network` (M11b) adds the CTMRG half, promoted from `examples/ctmrg.py`:

- `CTMEnv` — a `NamedTuple` `(c, e, bond)`, the *outside* container: `bond` is the frozen
  `GradedSpace` the differentiated region reuses, a jit **cache key** and never a jit
  argument, which is why `ctmrg_unrolled` takes `c`, `e` and `bond` separately (a `NamedTuple`
  is a pytree, and a `GradedSpace` is not a leaf);
- `Absorb` — two closures, `corner(c, e) -> big_c` of rank `2n` whose index groups are
  diagonal mirrors (which licenses `move`'s `ndim // 2`) and `edge(e, p) -> new_e`. Two
  real implementations, no `Protocol`: the closures must be able to capture *traced*
  values, and a `NamedTuple` of functions is hashable for `static_argnums`;
- `single_layer(bulk)` for any rank-4 `(l OUT, u OUT, r IN, d IN)`, `double_layer(ket,
  bra)` and `layers(ket)` for any rank-5 iPEPS ket — model-free names, because neither
  einsum has ever heard of Ising or of a Hamiltonian;
- `init_env`, `single_layer_ctm`, `double_layer_ctm`, `move`, `ctmrg`, `ctmrg_unrolled`,
  `normalized`, `ring`.

The C4v restriction (one corner and one edge, a 1×1 unit cell) is a documented
**precondition** on the tensor the caller hands in, never a symmetrization the library
performs — which is why `c4v` stayed in the example, together with the observables
(`_halves`/`energy` are a measurement API with one geometry) and the bulk tensor.
`network/common.py` holds `spectrum` and `ones`, moved out of `mps.py` and `env.py` with
their bodies unchanged so that `ctmrg.py` need not import a driver it shares no concept
with. The `scalar` and `inner` that moved with them left the driver layer again in #126,
as `tenet.full_trace` and `tenet.inner`.

It is reachable as `tenet.network` and listed in `tenet.__all__`, and it is deliberately
**not** flattened into the top-level namespace: `dmrg_` is not a tensor operation. The
dependency edge is one-way — `network` imports `ops`/`tensor`, never the reverse — in the
shape `ops` already uses for `tensor`. Everything here is built on **public `tenet` API
only**: no `jax`, `torch`, `scipy`, `quimb` or `opt_einsum`, no reach into another
module's private names, and no numerical use of reduced blocks. The one named exception
is reading `t.provider`, `provider.qdim` and `provider.unit`: two reads, one owner each —
`spectrum`'s `sqrt(qdim)` Schmidt weight and `ctmrg.py`'s unit sector.
`tests/network/test_hygiene.py` enforces all of it.

**Which side of a trace a module lives on is a per-module statement, and it is the
complement of invariant 9 rather than an exception to it.** Invariant 9 says
structure-changing operations live outside compile boundaries and the library never hides
the distinction; `tenet.network` is where the data-dependent control flow is then *allowed
to live* — `svd_truncated` re-deciding a bond `GradedSpace` at every bond of every sweep, a
happy breakdown comparing a norm to a tolerance, a loop exiting on a measured energy
change. `mps.py`, `env.py` and `dmrg.py` are **outside** by construction and make no
differentiability claim; `common.py` is trace-neutral and used on both sides; and since
M11b `ctmrg.py` is **both**, stated per function.

The worked example is `ctmrg` against `ctmrg_unrolled`. `ctmrg` reads singular *values* to
decide a bond and a corner spectrum to decide when to stop, so it raises under any trace —
`jax.jit` over it fails at the loop exit, before it ever reaches an SVD. `ctmrg_unrolled` runs
exactly `k` moves at that already-decided bond through `svd(bond=)`, shape-static and
differentiable, and traces once across different block values. `move` is the boundary
itself: `chi=` is the structure-deciding half, `bond=B` the traceable one. The frozen
`GradedSpace` is the only object that crosses, and it crosses as metadata.

**`MPS` and `Env` are mutable containers of immutable tensors**, which leaves
`REPOSITORY_RULES.md`'s "structural/categorical types are immutable" intact: every
`Leg`, `GradedSpace`, `TensorStructure` and `SymmetricTensor` they hold is still frozen.
An MPS is a container plus an orthogonality centre that *moves* — a state machine, not a
categorical object — and in-place methods carry a trailing underscore (`canonize_`,
`setup_`, `update_`, `clear_`) so the invalidation discipline reads as mutation at the
call site.

The split:

- **M11a** — `MPS`, `MPO`, `Env`, `lanczos`, `dmrg_()`; `examples/dmrg.py` rewritten on
  top of it at identical numbers.
- **M11b** — the CTMRG analogue (`CTMEnv`, `Absorb`, both absorbers, `move`, `ctmrg`,
  `ctmrg_unrolled`) in `network/ctmrg.py`, plus `network/common.py`. It was a separate PR because
  its invariants differ: half of it must survive `jax.jit(jax.grad(...))`, so its API
  carries the `svd_truncated`-outside / `svd(bond=)`-inside pairing in its signatures where
  M11a's carries no trace at all. `examples/ctmrg.py` was rewritten on top of it and every
  recorded number — free energies and their gradients, corner spectra, sweep counts, iPEPS
  energies and gradient blocks, SGD traces — is **bit-identical** to the pre-promotion run.
- **M11c** — shipped: `MPS.save`/`load` (a directory of per-tensor `tenet.save` files plus
  an `mps.json` carrying `format`, `n_sites` and `center`, so #94's coefficient-gauge
  verification is inherited per tensor rather than re-implemented), `MPS.compress_`
  (`canonize_(0)` plus one truncating sweep, returning the *total* discarded weight where
  `sweep_` returns the per-bond *maximum*), the two measurements `expectation_1site` /
  `expectation_2site` (both divided by `<psi|psi>`, which is what the name promises and
  what YASTN's undivided `measure_*` does not), and `CTMRG_out` — `ctmrg`'s return, a
  `DMRG_out`-shaped `NamedTuple` replacing the bare `(CTMEnv, history)` tuple two test
  files were re-deriving convergence from, at bit-identical numbers.
  Two items stayed deferred. The CTMRG measurement half (`rdm_1x1`/`rdm_2x1`, froSTspin's
  `ctmrg/rdm.py` as the design) stays in `examples/ctmrg.py` because `_halves` and
  `energy` are C4v- *and* 1×1- *and* 2×1-specific in their indices, so promoting them
  would move physics out of the file whose job is to teach in exchange for a library
  function with one geometry. `MPO.save`/`load` is refused rather than deferred, because
  `MPO.from_w` rebuilds the whole Hamiltonian from one dense `W` in milliseconds and a
  checkpoint of a regenerable object is a second source of truth.

- **M13** — shipped: `MPO.from_terms`, the term-list MPO generator M11a and M11c both
  refused. The refusal was reversed on a direct request, not on new in-repo evidence, so
  the scope carries its own discipline: one new module-level name (`local_op`, a dense
  `(d, d)` operator as rank 3 on `(phys OUT, phys IN, charge OUT)` — the charge has to
  live on a leg, because `S^+` is symmetry-forbidden as a rank-2 tensor), two new methods
  (`MPO.from_terms` and `MPO.to_dense`, the oracle exit two tests were hand-rolling) and
  one private partition normalizer. No new module, no new dependency, no operator
  registry: the matrices stay in `examples/dmrg.py` because *the library takes bond
  spaces; the example computes which spaces are reachable*.
  Construction is assembly by `direct_sum` and compression by `svd_truncated`, not
  tenpy's `MPOGraph` finite-state machine. The FSM is better on bond dimension and is
  exact, and it still loses on the one axis that decides it here: it produces bond
  *labels*, and turning labels into a `GradedSpace` is tenpy's `_calc_legcharges`, over a
  hundred lines of charge bookkeeping that the SVD route gets for free and gets
  **derived** — `svd_truncated` returns a bond space whose degeneracy at `c` is the number
  of kept singular values and which omits `c` entirely when that number is zero. The
  measured result is that `from_terms` recovers `examples/dmrg.py`'s hand-written
  `MPO_BOND` sector for sector, `{0: 3, +2: 1, -2: 1}`, with nothing written down.
  `MPOGraph` (`networks/mpo.py`:2142) is the named upgrade path, and the first thing it
  buys is exponentially-decaying couplings at one virtual state apiece.
  Two refusals, each with a message rather than a silent wrong answer. **Fermionic
  terms**: refused here on the premise that Jordan-Wigner needs a swap gate between an
  odd MPO bond and a physical line. That premise was later *refuted* and the refusal
  lifted (M21/#147, after M23/#160): the Koszul braiding **is** the string — an odd FSM
  bond crossing a spectator's physical lines writes the `Z` with no swap gate and no
  explicit JW operator anywhere in the API — and the actual gap was the cap-direction
  convention, one Koszul sign per odd bond paid by operand order, fixed by M23's
  network-wide composition rule. **Non-Abelian terms**: a *list* of
  operators does not determine a non-Abelian term — three tensor operators fuse through
  several channels and the DSL has no slot for a coupling tree.

- **M13b** — shipped: non-Abelian `MPO.from_terms`, and the first SU(2) DMRG run. M13's
  non-Abelian refusal was not overturned, it was *routed around*: the premise that goes is
  the *list of operators*. A term is now one symmetry-**invariant** *k*-site operator —
  `local_op(dense, phys=...)` with no `charge`, rank `2k` on `(phys OUT)*k, (phys IN)*k`,
  the layout `np.kron` already has — placed on a tuple of sites, and `svd_truncated` peels
  it into `k` MPO tensors (MPSKit's `decompose_localmpo`). **The coupling tree was never
  needed because it lives inside the operator's own blocks**, and the aux bond is not
  fused from declared charges at all: it comes out of the SVD, sector by sector, empty
  sectors omitted. So there is no `mu` slot, at any *k*, for any symmetry — a multiplicity
  arrives as an ordinary `GradedSpace` degeneracy in a derived bond, which an SU(3) two-
  site term confirms (adjoint ⊗ adjoint splits on a bond carrying `8` at degeneracy 2).
  `fused_leg` is recorded as *not* the mechanism. The refusal above still fires for a
  declared chain of charges, because that ambiguity argument is still true; its message
  now names the *k*-site spelling instead of a follow-up. `from_dense`'s default `atol`
  does the rest of the work: an array that is not invariant is refused, so under SU(2) the
  DSL is *incapable* of expressing a symmetry-breaking term. Measured: SU(2) Heisenberg
  against the dense `kron` oracle at N=4 and 6, a bulk MPO bond of `{0: 2, 2: 1}` (3
  blocks, dense 5, against U(1)'s 5 blocks and dense 5), and DMRG at N=6 and N=12 hitting
  the U(1) energies to 1e-10 with 12 multiplets where U(1) needs 32 states. Nothing in
  `Env`, `heff2`, `lanczos` or `sweep_` changed. **`MPO.to_symmetry` is still refused**: `to_symmetry`
  reorders the dense basis per leg, so a per-site comparison fails on a correct
  implementation, and `MPO([to_symmetry(w, U1) for w in h])` is all a test ever wanted.
  `MPO.save`/`load` was promised for "the day a term-list builder lands" and is reversed
  with its reason instead: M11c's argument was regenerability, and a term list makes an
  MPO *more* regenerable, not less.

- **M15** — shipped: symbolic MPO construction. `MPO.from_terms` no longer sums M bond-1
  term strings with `direct_sum` (an intermediate bond exactly as wide as the term
  count); it assembles a finite-state machine over labelled bond states — identity-left,
  identity-right, and one state per distinct open left-partial-string, so terms sharing
  an opening share a state and the closing edge carries the coefficient. Each state
  carries its **own** space (the running fused charge for rank-3 operators, or the graded
  `Leg` a k-site operator's internal SVD derived), the bond at each cut is the direct sum
  of its live states' spaces after pruning unreachable and dead-end states, and each edge
  is placed by `einsum` through 0/1 embedding isometries built at `from_dense`'s default
  `atol` — so an inconsistent state space *raises* instead of being projected away. This
  reverses M13's FSM refusal on evidence: the refusal priced in tenpy's hundred-line
  charge solver, but tenpy's graph edges carry operator *names* while tenet's carry
  tensors that already know their spaces (MPSKit's fourteen-line virtual-space read is
  the precedent), so no solver exists here. The compressing SVD sweeps are demoted from
  the assembly to an optional post-pass: the default `cutoff=1e-13` keeps them (they earn
  their cost exactly on power-law couplings, where they see numerical low rank the graph
  cannot), and `cutoff=None` skips both, making the bond combinatorial and tolerance-free
  — the k-site operator's internal SVD is a different SVD and still runs. Measured:
  R=4 1/r² Ising at N=96 fell from 12 s to under 0.3 s, the finite-range FSM bond equals
  the compressed bond exactly (6 = 6), and the all-pairs pre-compression bond fell from
  496 to 32. Refused with it, each with the criterion that would buy it: block2's
  minimum-vertex-cover bond basis (wins only where the coefficient matrix is dense *and*
  not numerically low rank — the ab-initio integral tensor and essentially only it, plus
  a hand-written max-flow since scipy is optional), the normal/complementary operator
  machinery (a per-Hamiltonian derivation, not an algorithm; no quantum-chemistry
  caller), symbolic simplification rules (they need an operator vocabulary; tenet's
  operators are anonymous by design), lazily shared numeric payloads (a bond of a few
  dozen states is kilobytes), and an exponentially-decaying-coupling front end (now one
  state and one self-loop edge — cheap, but still a model DSL with no caller).

- **M16** — shipped: edge-preserving `heff2` and a structure-keyed compiled matvec.
  `from_terms(cutoff=None)` no longer discards its finite-state machine at
  instantiation: each site keeps an edge-block table (`MPO.edge_blocks`) — the four blocks
  `A`/`B`/`C`/`D` of MPSKit's `(1 C D; · A B; · · 1)` form, identity edges stored as
  `None`, corners implicit — and `Env` folds the two environments into those blocks once
  per bond (MPSKit's `JordanMPO_AC2_Hamiltonian`), amortized over the whole Krylov
  solve and compiled through an injected `compile=` per structure key. The design is
  the block-DMRG correspondence made executable: `IdL`/`IdR` are the not-yet-begun and
  completed channels, each open FSM state is one boundary operator `S`, and the
  prepared matvec is `Ψ·H_env + H_sys·Ψ + Σ S·Ψ·S` in MPO clothing — the identity
  channels (corners plus spectators) ride one composed rank-2 map, so a width-10
  cylinder's 38 live edges against `D_w² = 1024` never touch a dense `W`. Refused with
  it, on measurement: a Python loop over edges inside the matvec — four dense proxy
  implementations measured 0.44×–0.87× at `d = 2`, where a single BLAS GEMM beats every
  bandwidth-bound sparse form — with the criterion that reverses it: **`d ≥ 4`**
  (spinful fermions, `S ≥ 3/2`, grouped sites), where the same experiment measures
  1.93×–2.78×; and recovering a table from a compressed MPO, because the SVD gauge
  mixes the states and leaves zero identity edges on every model measured.

Not planned: TDVP, iDMRG, excited states, fermionic swap gates and PEPS containers.
Fermionic swap gates stay not planned for a stronger reason than before: fermionic
DMRG shipped without them (M21/#147) — the fZ2 braiding is the Jordan-Wigner string,
and the gap M13's refusal guarded against was the cap-direction convention M23/#160
fixed, not a missing gate.

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
and other coefficient arrays never live in structural fields. All three are
enforced, not asserted: `tests/backends/test_torch.py` walks every public op on
torch blocks (see "What 'PyTorch backend' means, exactly").

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