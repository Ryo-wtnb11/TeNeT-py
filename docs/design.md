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

That piece is no longer gated on "when a quantum-chemistry caller appears", and
the gate is retired here because it was circular: #136, #138 and #141 each
deferred the symbolic layer on that rule, but nobody writes a quantum-chemistry
front end against a builder that will not build the Hamiltonian, so the condition
could never fire from outside. **The measured criterion that replaces it is the
phase split inside `from_terms`, not the bond order**, and
`benchmarks/bench_qc_mpo.py` is where it is taken — real ab initio integrals
(FCIDUMP, fetched not vendored) on spin-orbital fZ2 sites, the finite-state
machine's bond per cut beside an independently computed minimum vertex cover of
the same cut and beside what the two compressing SVD sweeps leave. At K=8
(H8 STO-6G, 7 360 terms) the FSM bond is 1 148 against a cover of 122 — 9.4×,
rising to 21× at K=32 and 41× at K=26 on C2 cc-pVDZ, which is #138's own named
minimum-vertex-cover criterion firing. But the compressing sweeps reach that
cover *exactly*, per cut, for 0.1 s of a 19.3 s call, and `_instantiate` is the
other 18.9 s; from K=16 up `from_terms` stops finishing at either cutoff and it
stops in `_instantiate` every time. So what breaks is the cost of materialising a
bond the next step discards — not the bond order, which `svd_truncated` already
recovers — `_instantiate` is where a mechanism has to land, and a second
bond-basis algorithm is not what buys it.

**Stage 1 of that landed as M37, and it moved the instantiation boundary rather
than optimising the old one.** The mechanism has a name and a type: `EdgeTable`,
the pruned finite-state machine with its bond space and dense slot map per cut,
is what `from_terms(cutoff=None)` returns, and `MPO` carries it as the symbolic
one of its two internal representations. Numeric instantiation is what a
*consumer* asks for, per site, through one of the boundary's two doors —
`EdgeTable.site` for a full-width rank-4 `W`, `EdgeTable.edge_blocks` for one
site's group-restricted blocks — so `Env`'s prepared two-site operator is built
without a full-width site tensor existing anywhere on its path. The measurement
that motivated it is the one measured after: at K=16, `from_terms(cutoff=None)`
went from 25.0 s and 21.7 GiB to 0.94 s and 0.17 GiB, and at K=26 it went from
not finishing at all — killed in `_instantiate` at 564 s against the
benchmark's own 8 GiB address-space cap — to 6.4 s at 0.94 GiB. What it
does not buy is bond order, which is why it is a stage of the symbolic layer and
not a substitute for it; the M37 milestone entry below states what remains.

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

## Degenerate singular values

Reverse-mode SVD and eigh VJPs carry `1/(σ_i − σ_j)` and `1/(w_i − w_j)` factors, which
are `NaN` at exact degeneracy (jax-ml/jax#2311, #8732, #2329 — acknowledged upstream, not
a bug awaiting a fix). Under a non-Abelian symmetry that is the generic situation rather
than an edge case: a symmetric fixed point has degenerate singular values *inside* a
coupled sector by construction, and a coupled sector of an SU(2) tensor is exactly where
the members of a multiplet land. The failure is narrower than "JAX has no gradient" — a
function of the singular *values* alone differentiates fine even at degeneracy, and so
does the reconstruction `U S Vh` with a single exactly-zero singular value; two coincident
values are what produces `NaN`.

`tenet.ad` installs the fix the differentiable-tensor-network literature settled on:
Lorentzian broadening of the `F` matrix, `1/x → x/(x² + ε)` (Liao, Liu, Wang, Xiang,
PRX 9, 031041 (2019), Sec. III A; `safe_inverse` in tensorgrad's
`tensornets/adlib/svd.py`). Broadening rather than a hard degeneracy tolerance **because
the hard version cannot be jitted**: MatrixAlgebraKit's `inv_safe(x, atol)` zeroes below a
tolerance and *warns* when the gauge-sensitive part is not small, and the warning — the
part that makes the hard tolerance safe — is a data-dependent branch no traced region can
run (invariant 9). Broadening is a smooth elementwise function with no branch, so it
survives `jit`, `grad` and `vmap` unchanged. The price is the precondition `tenet.ad`
documents at its installation point: the broadened gradient is correct exactly when the
objective is gauge-invariant on each degenerate subspace, which is the same condition
MatrixAlgebraKit enforces by warning and we cannot.

The seam is `autoray.register_function("jax", "linalg.svd", ...)` — autoray's own
extension point, hence process-global for the JAX backend — rather than a hook threaded
through `ops/linalg.py`. That is the honest cost of using the documented seam, and it is
why installation is an explicit function call rather than an import side effect.

Nothing else needs a fix. `qr`/`lq` have no `1/(σ_i − σ_j)` anywhere — their only
inversion is a triangular solve and their instability is rank deficiency of `R`, a
different problem — and `polar` inherits this one for free because it calls
`ar.do("linalg.svd")`. No custom JVP: forward mode has no caller here, since `jax.grad`,
VMC and CTMRG are all reverse mode.

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

Two container decisions, recorded here rather than on the classes. **`MPO` is a separate
class from `MPS`, with no shape flag**: YASTN unifies the two behind `_nr_phys in {1, 2}`
(`_mps_obc.py`:223-225) and pays a runtime branch on that flag at :284, :291, :438, :443
and :90-100, inside the code whose whole job is structural bookkeeping — the pattern
`tenet` is typed to avoid. TenPy agrees for its own reason (`mpo.py`:16-18: "unlike for an
MPS, this doesn't simplify calculations. Thus, an MPO has no `form`"). Two classes, no
branch. **`Env` is one class, not YASTN's factory over eight**: `yastn.tn.mps.Env` is a
function dispatching into `Env2`, `Env_mps_mpo_mps`, `…_precompute`, `Env_mpo_mpo_mpo`,
`Env_mps_mpopbc_mps`, `Env_sum` and `Env_project` (`_env.py`:26-89), and every one of those
serves a feature M11a does not ship — MPO products, PBC, sums of Hamiltonians,
excited-state penalties. The dispatch arrives if and when a second target does.

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
module's private names, and no numerical use of reduced blocks. The one named exception is
reading `t.provider` and a short allow-list of *symmetry-generic* provider metadata, each
entry with a named owner. That list is not restated here, because a second copy goes
stale: `tests/network/test_hygiene.py` both states it (its module docstring, one owner per
attribute) and enforces it (`test_no_provider_branching`'s `allowed` set), and it catches
reads through a local binding such as `sym = space.provider`.

**The composition rule.** Every two-operand `tenet.einsum` in this package is a
**composition**: operand 1 supplies the `IN` end of *every* shared wire and operand 2 the
`OUT` end. For three or more operands the same rule applies pairwise in caller order,
which `_contract_path` guarantees (`ops/contraction.py`:596-603).

The rule exists because the two ends of a wire are not interchangeable for a fermionic
provider — the cap `V*⊗V → 1` is not the cap `V⊗V* → 1` — so the operand order of every
einsum that can contract an odd wire is load-bearing, and a per-call choice is exactly
what produced the `(-1)^(j-i)` discrepancy M21/#147 measured at its gate 1, the
cap-direction Koszul sign. Meeting `IN` against `OUT` is necessary and **not** sufficient:
that condition is symmetric, so it fixes contractibility alone while the cap sign depends
on which operand supplies which end. An earlier revision of `Env` argued exactly the
symmetric reading — "IN against OUT with no leg bend anywhere in this module" — and gate 1
measured it to be insufficient: the operand orders were individually well-formed and still
paid one Koszul sign per odd wire capped in the wrong direction.

A wire that genuinely turns around in the intended planar diagram is therefore **bent
explicitly** with `tenet.repartition` before the einsum (`network/env.py::_composed`),
never left to the einsum's implicit cap. In `env.py` the MPS bond arrow and the MPO bond
arrow cross the two-site cell in opposite directions, so closing either cap turns one rail
around, and each bend is pinned by the dense Jordan-Wigner oracle.

`tests/network/test_hygiene.py::test_every_two_operand_einsum_is_a_composition` pins the
rule at every call site a smoke over the MPS/MPO/DMRG modules reaches, with zero
exemptions, and asserts that coverage rather than assuming it. `ctmrg.py` is deliberately
outside the smoke: a 2D network has loops, where operand order is necessary but not
sufficient (`ops/contraction.py`:575-587), and no fermionic PEPS caller exists.

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
  network-wide composition rule ("The composition rule" above). **Non-Abelian terms**: a *list* of
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
  What `cutoff=None` skips is `from_terms`'s two compressing sweeps, and for finite-range
  models those reduce the bond dimension by exactly nothing — on nearest-neighbour
  Heisenberg, an R=4 chain and width-6/width-10 cylinders, 5 stays 5, 14 stays 14, 20 stays
  20, 32 stays 32 — while the SVD gauge turns 38 sparse edges into 302 dense pairs on the
  width-10 cylinder (#141). Whether keeping the table wins is therefore a backend question:
  with a `compile=` callable injected into `Env` the prepared matvec measured ~20× faster
  than the site-tensor path, while on the plain numpy backend at small bond dimension it pays a
  per-call dispatch premium and a full sweep measured 1.5–2.6× *slower* (#141's tables).
  The default `cutoff` stays `1e-13` for that reason and for power-law couplings, where the
  sweep earns its keep — all-pairs `1/r²` at N=32 takes the bond 33 to 8. The per-site table
  type was spelled `JordanBlocks` until M31/#185 renamed it `EdgeBlocks`: MPSKit's name for
  the partition is the MPO's *Jordan form* and the citation stays, but in a package that
  supports fermions "Jordan" already names the Jordan-Wigner string, and the Jordan normal
  form is one import away from `tenet.linalg` — three meanings for one word, which no
  docstring separates at a call site.

- **M35** — shipped: `MPO.from_arrays`, a second *input layer* onto the same assembler.
  The measurement that opens it is the one above, taken again after the term list became
  the dominant cost: at ab initio scale the wall is the per-term Python work in front of
  `_term_edges`, and the answer is the input shape rather than the assembler. block2
  stores a Hamiltonian as three parallel arrays per operator pattern
  (`integral_general.hpp`:45-57) — `exprs`, a flat index buffer of `nn` sites per term,
  and one coefficient per term — and never builds a term object;
  `MPO.from_arrays(n_sites, ops, blocks)` takes the same three, transposed into an
  iterable of `(expr, indices, data)` triples, with `ops` the caller's name-to-operator
  table so that **no operator vocabulary is introduced**: names mean what the call says
  they mean, and operator identity stays `id(op)`. Sorting each row into site order,
  paying the Koszul sign of each inversion of two sign-braiding operators,
  pre-multiplying coincident sites into one on-site operator, and fusing terms that agree
  on `(operator labels, sites)` all run over whole arrays, once per pattern rather than
  once per term. Two consequences are the point rather than side
  effects: the merge discharges the "two operators of one term sit on site N; multiply
  them first" burden `from_terms` puts on its caller, and it is the only position from
  which an exact cancellation is visible — a cancelled term never allocates a state, so
  the FSM bond can come out *narrower* than the term list's, measured at −0.6% on C2
  cc-pVDZ with the operator agreeing to 7e-15. `screen` (default `1e-12`,
  `ExprBuilder`'s own) is one coefficient-magnitude knob applied after that merge where
  block2 has four; at its default it removes the symmetry-forbidden ~1e-15 entries a real
  integral file carries and nothing else, and it is an accuracy/size trade at `1e-4` and
  above, not a performance lever. `from_terms` is unchanged and neither builder is
  deprecated: a lattice model is a list of terms and an ab initio Hamiltonian is
  `O(K⁴)` terms over a handful of patterns. Refused with it: k-site operators in `ops`
  (a block gives one site index per name, so the invariant `k`-site form has nowhere to
  put its extra indices, and the message points at `from_terms`), and permutational
  symmetry kept implicit as a multiplicity factor — the eight images of `(ij|kl)` are
  eight different operator strings, so the caller expands and the merge removes the
  redundancy afterwards, which is block2's own order.

  **The walk itself became array-driven, and there is exactly one of it.** `_term_edges`
  used to take a Python term — a coefficient and a list of `(operator, sites)` pairs —
  and redo that term's pattern work from scratch: a growing tuple rebuilt and rehashed
  once per operator placed (so the prefix key cost was quadratic in the term length), one
  charge fusion, one `GradedSpace`, one braiding probe and one dense round trip per
  operator, one `multiply` and one `SymmetricTensor.__add__` per closing edge, and a
  spectator span re-walked by every term through it. It now takes two integer rows —
  slots and sites, both in site order — and a coefficient, with a state interned from
  `(parent, slot, site)`, the rank-4 edge cached per `(slot, running charge)`, closing
  coefficients summed before a single `multiply` per surviving edge, and the spectator
  span bounded by a per-state high-water mark. `from_arrays` hands those rows over
  directly; `from_terms` canonicalizes its list into the same rows first, so **both
  builders share one walk** and the term-list route inherits the speedup. Measured on
  40,000 four-operator fZ2 rows over 32 sites, input layer only: `from_arrays` 29.3 →
  **5.9 µs/term**, `from_terms` 30.1 → **16.3 µs/term** — of which the walk is 5.2 and
  the remaining 11.1 is building and sorting the term objects themselves, which is what
  the list API *is* and the reason the array API exists. Both front ends produce the
  identical machine, 19,272 states either way. `_edge_table`, `_place`, `_instantiate`,
  `EdgeBlocks` and `network/env.py` are untouched: a state label is now an `int` where it
  was a tuple or `"IdR"`, and nothing outside `mps.py` ever read a label's value.

- **M37** — shipped: deferred instantiation, built as the *instantiation boundary* of the
  symbolic layer rather than as a shortcut around it (#200, stage 1 of #184's staging).
  `from_terms(cutoff=None)` and `from_arrays(cutoff=None)` materialise nothing: what they
  return is the edge description itself — `EdgeTable`, the pruned finite-state machine
  with its bond space, dense slot map and group subsets per cut, built with no tensor —
  and `MPO` carries it as the symbolic one of its two internal representations, with the
  site tensors as the other. **The boundary is that type and it has exactly two doors.**
  `EdgeTable.site(n)` places one full-width dense-blocked rank-4 `W`;
  `EdgeTable.edge_blocks(n)` places one site's `EdgeBlocks` against the group-restricted
  bonds, the six group embeddings included. They are peers, each cached per site, and
  neither is built out of the other: `_instantiate` is the first door's consumer — the
  streaming compressing sweep, which only runs at a float `cutoff` — and `Env._cores2` is
  the second's. Two consequences are the restructuring rather than side effects.
  `_instantiate` stops being the producer of everything and becomes a function that takes
  the table and returns sites; and the group embedding, which it used to run for every
  cut of the whole MPO before `Env` could touch a block, moved into the per-site core
  builder, which is what #184 named as the real cost of this stage. `Env` reaches the
  description through `MPO.edges` and one site's blocks through `MPO.edge_blocks`, the
  same "the accessor is the only door" rule #141 set for the block table.

  Measured on `benchmarks/bench_qc_mpo.py`, same machine, same term lists, the shipped
  path against the base commit. At K=16 (N2 CAS 6-31G, 32 sites, 34 400 terms),
  `cutoff=None`: **25.0 s / 21.7 GiB peak RSS → 0.94 s / 0.17 GiB**, a 128× memory
  reduction, because assembly now allocates nothing at all. At K=26 (C2 CAS cc-pVDZ, 52
  sites, 208 844 terms), `cutoff=None` went from **not finishing at all** — killed in
  `_instantiate` at 564 s against the benchmark's 8 GiB address-space cap, which is
  where it has stopped since #191 — to **6.4 s / 0.94 GiB**. At K=42 (the synthetic
  generator, 84 sites, 494 092 terms), uncapped so that the base commit finishes at all:
  **375.3 s / 22.5 GiB → 11.5 s / 1.43 GiB**. The default
  `cutoff=1e-13` path is unchanged by construction and measured unchanged: K=16 3.94 →
  4.02 s at 1.26 → 1.25 GiB, K=26 35.7 → 35.0 s at 6.00 → 5.91 GiB, K=42 25.0 → 24.9 s
  at 2.63 → 2.66 GiB, identical bond profiles.

  The honest other half of that measurement: *forcing* every site's block table is where
  the deferred cost lands, and it is a real cost. At K=16 it is 17.6 s and 18.9 GiB peak
  — below the base commit's 21.7 GiB, because the full-width sites no longer exist beside
  the blocks, but the same order — at K=42 it is 266 s and 23.1 GiB, and at K=26 under
  the benchmark's 8 GiB cap it does not fit at all. So the deferral removes the wall from
  *construction* and does not remove it from a consumer that wants the whole operator at
  once. What fixes that is the bond order, which is the next stage; what M37 buys is that
  a Hamiltonian can now be assembled, and its cores prepared one bond at a time, without
  ever paying for the whole thing.

  What M37 does not buy is bond order, exactly as #184 ranked candidate (a): "real and
  insufficient — it removes the memory wall and leaves the cubic". The FSM bond is still
  Θ(K³) against the vertex cover's Θ(K²) (31 441 against 769 at K=26). Two things are
  settled around it. The **minimum vertex cover is closed by measurement**, not deferred:
  #184's reopen criterion — any cut where the post-SVD width exceeds the cover by more
  than 10% — returned 1.0000 at K=16 over 33 cuts and 766-against-769 at K=26, so
  `svd_truncated` reaches the cover on its own and #138's refusal stands at production
  scale. And the caller-supplied operator equivalence relation stays gated on the next
  stage's measurement; `id(op)` is still the only operator identity anywhere.

  **What remains for the symbolic layer (#184 candidate (d)) is named at the boundary.**
  An expression algebra with a per-cut assembler replaces `_edge_table` as the producer
  and `EdgeTable.edge_blocks` as the per-site core builder; `_cores2`'s contract — "give
  me bond `n`'s cores" — and the a/b/c/d partition are the interface it has to keep
  hitting, and `Env._cores`' per-bond cache is the slot it plugs into. What it buys that
  M37 does not is the two things the measurement above says are still open: a bond basis
  that is not a prefix (so the cubic becomes quadratic) and shared numeric payloads, one
  matrix per factor-stripped symbol shared by every occurrence and every scalar multiple
  (block2's `operator_tensor.hpp`:40-47), which buys sweep arithmetic rather than width.
  The two-representation trade #141 accepted stands one level deeper and its deletion
  condition is unchanged: `from_w`'s numeric path and `MPO.to_dense` still need real site
  tensors, so `EdgeTable.site` is not going away and neither is the dense `heff2`.

- **M38** — shipped: the sweep's per-bond caches are held to **one byte budget**, and the
  measurement that chose it also says how much of the wall M37 left is a cache at all
  (#202, the direct follow-up to #200). M37's honest other half was 18.9 GiB to force
  every site's block table; the caches that hold those bytes — `EdgeTable._table` and
  `EdgeTable._embeds`, `Env._cores`, `Env._prepared` and `Env._compiled` — were each
  correct in isolation and each unbounded, and a DMRG sweep visits every bond, so after
  one half sweep the whole prepared operator is resident again. The policy is
  `network/common.CACHE_BUDGET` and the `Recent` dict that reads it: least-recently-*used*
  eviction past the budget, never below two entries (a two-site bond asks for site `n` and
  site `n + 1` in one breath), and every one of the five caches is a bare `Recent()`. One
  number, one place, no flag threaded through `Env` and `EdgeTable` separately.

  **Where the bytes are**, measured over a full run at K=16 (N2 CAS 6-31G, 32 spin-orbital
  sites, `chi=16`, three sweeps, `cutoff=None` operator, unbounded caches, 20.75 GiB peak
  RSS), `benchmarks/bench_qc_mpo.py --dmrg`. "Charged once" walks the caches in data-flow
  order and charges each array buffer to the first cache that reaches it, so the column
  sums to what is resident; "own" re-walks each cache alone.

  | cache | charged once | own |
  |---|---|---|
  | `EdgeTable._table` (block tables) | **13.69 GiB** | 13.69 GiB |
  | `EdgeTable._embeds` (group embeddings) | 0.00 GiB | 2.97 GiB |
  | `Env._cores` | **5.19 GiB** | 15.86 GiB |
  | `Env._prepared` | 0.31 GiB | 11.81 GiB |
  | `Env.F` (environments) | 0.06 GiB | 0.06 GiB |
  | `MPO._sites` | 0.00 GiB | 0.00 GiB |
  | total | 19.19 GiB | — |

  That table answers #202's own question about whether the two caches want the same
  policy, and the answer is that they cannot have different ones: `Env._cores` *owns*
  15.86 GiB but is charged 5.19, because 10.7 GiB of it is `EdgeBlocks` tensors held by
  reference — `ra1`, `ra2`, `open_l`, `open_r`. Evicting the block tables while keeping
  the cores frees almost nothing, and the same holds one level further down for the group
  embeddings, whose 2.97 GiB is entirely inside the block tables that hold them.

  **The trade, over a full run and not a sweep**, same input and same three sweeps; `held`
  is what remains cached at the end, `built` counts every block table and every merged
  core the run had to make. The ground-state energy is `-27.234808138` on every row.

  | budget | peak RSS | held | wall | tables built | cores built |
  |---|---|---|---|---|---|
  | unbounded | 20.75 GiB | 19.19 GiB | 128.4 s | 32 | 31 |
  | 4 GiB | 17.65 GiB | 7.34 GiB | 232.3 s | 146 | 132 |
  | 1 GiB (shipped) | **15.24 GiB** | 1.84 GiB | 317.0 s | 215 | 147 |

  **A fixed count of entries was measured first and rejected, and the number that rejected
  it is a small model, not a large one.** At bounds of 2, 4 and 8 entries the same run
  peaks at 15.08, 17.75 and 20.25 GiB in 346.9, 271.6 and 258.3 s — the same curve. But a
  count evicts on a two-megabyte MPO exactly as eagerly as on a twenty-gibibyte one, and
  measured over `dmrg_` runs on the U(1) Heisenberg-plus-NNN and spinless-fermion chains
  at `n=12, chi=16` and `n=20, chi=32`, a bound of four entries costs **+22 % to +26 %**
  wall time for no memory saved at all. A byte budget is never reached by those models, so
  they evict nothing and pay nothing: the same runs under the shipped budget are **−0.2 %
  to +0.5 %** (median of nine), inside the ±5 % threshold this milestone holds itself to
  and inside the run-to-run noise. That is #202's "the common case must not get slower to
  fix the rare one", met by construction rather than by tuning. A sweep-direction-aware
  window was not built: the build counts above say a sliding sweep already rebuilds one
  table per bond at any budget that evicts at all, and no window changes that count — it
  can only change which bond pays.

  **What the budget cannot buy, measured rather than assumed.** Walking all 32 sites'
  block tables while holding only two of them still peaks at **11.2 GiB**, because the
  widest single site's table is 1.33 GiB and `_place`'s dense buffers to build it cost
  several more. So the floor at K=16 is set by *one bond's working set*, not by how many
  bonds are cached, and no cache policy reaches below it. The budget moves the peak from
  20.75 to 15.24 GiB and the rest is bond width — #184's stage 2, unchanged and still the
  mechanism that changes the order rather than the constant.

  **K=26 does not complete, and the reason is that floor rather than the cache.** At C2
  CAS cc-pVDZ (52 sites, FSM bond peaking at 31 441) a full run at a 1 GiB budget stays
  resident at 19–24 GiB and does not finish `Env.setup_`. Walking the same operator's
  block tables one at a time with only two held says why in one number: site 26's block
  table **alone is 7.90 GiB** — 3.51 GiB of that its two cuts' group embeddings — at a
  bond of 12 124, and the chain's widest bond is 31 441, whose table is `bond_l * bond_r`
  larger again, about 52 GiB for a single site. One entry does not fit, so a policy whose
  floor is two entries has nothing left to give. That is the same verdict at a larger K
  as the K=16 floor above, and it is the measured statement that M38 removes a cache from
  the critical path and stage 2 has to remove the width. `Env.F` keeps its own
  invalidation discipline and is untouched: an eviction here is decided by recency and
  never by correctness, because `edge_blocks` and `_cores2` are pure functions of the edge
  description and a rebuilt entry is bit-identical to the evicted one. The oracles say the
  same thing from the other side — ED, the explicit-JW dense oracle, the MPSKit Heisenberg
  and Hubbard fixtures and `test_dmrg_prepared.py`'s six models are unedited and unchanged,
  and `tests/network/test_deferred.py` asserts a full sweep's caches fall to the two-entry
  floor at a zero budget while the same sweep unbounded holds one entry per bond.

- **M39** — shipped: the two compressing SVD sweeps **pin the `IdL` and `IdR` channels**,
  so a float-`cutoff` MPO is compressed *and* partitioned where #141 measured that it could
  be only one or the other (#204, stage 2 of #184's staging). The mechanism is one
  restriction of gauge freedom, not a new algorithm. At each cut the bond is
  `_merge`'s direct sum over `[_IDL, *open, _IDR]` and both corner states carry the trivial
  `D=1` unit space, so the two corner channels are the first and last degeneracy slot of
  the bond's unit sector; `_instantiate` takes those two rows out, rotates and truncates
  only what is left, and hands site `n-1` the block-diagonal carry `1_IdL ⊕ (u·s) ⊕ 1_IdR`,
  which `_place` folds exactly as it folded the free one. `_compress_forward` is the same
  thing on the open *column* slab. `tenet.direct_sum` puts the three slabs back in
  `_merge`'s own order, which is what lets the compressed description reuse `_merge` to
  describe its own cuts.

  **block2 ships the `IdL` half of this constraint in its own SVD route**, which is why the
  design was taken as a restriction rather than proposed as an invention:
  `general_mpo.hpp`:764-805 removes the delayed identity row from the matrix before the SVD
  and gives it a unit singular value, and the bipartite branch forces vertex 0 into the left
  cover for the same reason. The `IdR` half is tenet's.

  **The carrier is `EdgeTable` again, with a second producer.** `_edge_table` builds the
  finite-state machine; `_compressed_table` builds the compressed description — the
  compressed sites plus, per cut, the three group slabs — with **one open state per cut**
  where the FSM has one per open string, empty per-edge dicts, and
  `EdgeTable.edge_blocks` slicing `W'` where it used to scatter edges. That is legal
  because what `Env.heff2`'s prepared machinery consumes of a bond is the direct-sum
  decomposition and the four blocks placed against it, never the edges: `_cores2`,
  `_fold_last` and `_fold_first` read `a_op`/`b_op`/`c_op`/`d_op`, `idmap`,
  `spec_op`/`a_real_op` and the six embeddings and nothing else. `spec_op` is `None` and
  `a_real_op` is `a_op` on a compressed bond — in a rotated open basis a spectator's
  identity ride no longer separates — and a lattice model that wants the spectator shortcut
  keeps `cutoff=None`, where nothing changed at all. `_cores2`, `_build2`, `_apply2`,
  `_fold_last`, `_fold_first` are untouched.

  **Gate 1 — what pinning costs in bond width**, per cut, `benchmarks/bench_pinned_mpo.py`,
  the same fixtures as `bench_qc_mpo.py`. `all cuts` is `max(pinned/free)` over every cut;
  `inner` excludes the two cuts adjacent to the boundary.

  | fixture | N | max free | max pinned | all cuts | inner |
  |---|---|---|---|---|---|
  | H4 STO-6G | 8 | 30 | 30 | 1.250 | 1.000 |
  | H8 STO-6G | 16 | 122 | 122 | 1.250 | 1.000 |
  | N2 STO-3G | 20 | 96 | 96 | 1.250 | 1.000 |
  | H10 STO-6G | 20 | 192 | 192 | 1.250 | 1.000 |
  | N2 CAS 6-31G (K=16) | 32 | 562 | 562 | 1.250 | 1.000 |
  | C2 CAS cc-pVDZ (K=26) | 52 | 766 | **736** | 1.250 | 1.000 |
  | syn-42 | 84 | 146 | **54** | 1.250 | 0.486 |

  **The 1.10 criterion as written fails, and it fails at exactly two cuts of every
  fixture, for a reason that is structural and bounded.** The two cuts adjacent to the
  boundary go 4 → 5, everywhere, and nowhere else moves: at cut 1 the left block is one
  site, so the free rank saturates at `d² = 4` and the two pinned corner rows are linearly
  dependent on the open block there. A block-diagonal carry cannot absorb that dependency
  — writing the open rows' component along a corner row into the carry makes it block
  *triangular*, which puts non-identity content into the next site's `IdL` column and
  destroys the partition the whole change exists to keep. So the cost is at most one state
  per corner per cut, it bites only where the free bond is already at `d²`, and the maximum
  bond width — the quantity memory and wall depend on — is bit-identical on five fixtures
  and **smaller** on the two largest. The criterion was written expecting "the corners are
  2 states of 766"; that is what the inner column measures and it reads 1.000.

  **The independently computed minimum vertex cover says the same thing from the other
  side, and it is the sharper statement.** `bench_qc_mpo.py` has always printed, per cut,
  the FSM bond beside a combinatorial minimum vertex cover of the same cut (Kuhn matching
  plus Koenig, no `scipy`) beside what the sweeps leave. The pinned sweep's bond is now
  **equal to that cover at every cut** — H4 `1 5 16 25 30 25 16 5 1`, H8
  `1 5 16 33 46 63 84 109 122 109 84 63 46 33 16 5 1`, cover and post-SVD row identical
  entry for entry. The free sweep's 4 at the boundary-adjacent cuts is *below* the cover,
  and the cover is the right optimum here because it is computed for an operator that keeps
  its `IdL`/`IdR` channels: going below it is precisely the act of mixing them away. So the
  1.250 in the table is not the pinned bond being 25 % too wide, it is the free bond being
  one state narrower than any partitioned operator can be.

  The pinned truncation is also not uniformly wider: on syn-42 and C2 it is *narrower*,
  because `rsum2` weighs the discarded singular values against the total weight of the
  matrix it decomposes, and the corner rows — which carry the not-yet-started and
  already-finished channels' whole coefficient mass — are no longer in that total. Accuracy
  is unchanged where it can be checked: `<psi|H|psi>` on random fZ2 states at syn-8 is
  `1.8e-9` and `1.8e-8` relative against the uncompressed operator, against the free sweep's
  `3.1e-8` and `4.3e-9`, and on the fixtures small enough to expand `to_dense` agrees with
  the uncompressed `from_terms` at `6.4e-15` (H4) where the free sweep gives `3.3e-14`.

  **Gate 2 — a full DMRG at `chi=16`, three sweeps, per operator route.** `fsm` is the
  prepared path on the uncompressed FSM bond, which is M38's row and the number to beat;
  `pinned` is this milestone; `free-dense` is gate 2's honest baseline, the freely
  compressed sites handed over in a bare `MPO` so that `heff2` takes the compatibility
  entry —
  the alternative that needs no new code at all. It is a *measurement*, not a shipped
  route: a `from_terms` operator always carries its description and always runs the one
  engine path, and reaching this row means deliberately throwing the description away.

  | K | route | build | 3 sweeps | peak RSS | energy |
  |---|---|---|---|---|---|
  | 16 (N2 CAS 6-31G) | fsm | 1.2 s | 337.0 s | 14.52 GiB | −27.234808137600 |
  | 16 | **pinned** | 4.8 s | **4.5 s** | **1.38 GiB** | −27.258098346200 |
  | 16 | free-dense | 4.2 s | 1.9 s | 1.25 GiB | −27.258098346200 |
  | 26 (C2 CAS cc-pVDZ) | fsm (M38, #203) | — | did not finish `Env.setup_` | 19–24 GiB | — |
  | 26 | **pinned** | 42.0 s | **12.1 s** | **6.04 GiB** | see note |
  | 26 | free-dense | 42.3 s | 6.2 s | 4.72 GiB | see note |

  **One number moved between the first measurement and the re-run, and it moved the wrong
  way.** An early K=26 run reported a 3.21 GiB peak; it is not reproducible. Three clean
  measurements since — this row and the two independent C2 processes of the `chi` grid
  below — put it at **6.04, 6.19 and 6.09 GiB**, so 6.0 GiB is the number and 3.21 was an
  outlier taken while several other jobs were competing for the machine. The peak at K=26
  is the *build* transient (`_place`'s `D_FSM × d² × chi` buffers), not the sweep, which
  is why it is the part that moves with allocator behaviour. What does not move is the
  claim that matters: K=26 completes, at 6 GiB, where M38 had it not finishing `Env.setup_`
  at 19–24 GiB. K=16 is stable across every run at 1.2–1.45 GiB.

  The `fsm` row is M38's own run re-taken on this machine and it reproduces #203's number
  where it matters: the ground-state energy is `-27.2348081376`, which is `cutoff=None`
  behaviour byte-identical, and the peak is 14.52 GiB against the 15.24 GiB recorded there
  (that run also carried the cache-accounting instrumentation this one does not).

  The K=26 energies are a completion-and-resource measurement, not an energy comparison:
  three sweeps at `chi=16` on a 52-site strongly correlated system from a random start is
  nowhere near converged on any route, and the pinned and freely compressed operators are
  two different `chi`-16 truncations, so their sweeps diverge. The energy agreement that
  *is* a correctness statement is the K=16 row, where the pinned and free site-tensor routes
  agree to 5e-13, and `tests/network/test_pinned.py`, which compares `to_dense`.

  **K=26 completes**, which M38 recorded as blocked by one bond's working set: site 26's
  block table alone was 7.90 GiB on the FSM bond of 12 124, and the widest bond of 31 441
  extrapolated to ~52 GiB for a single site. On the compressed bond the widest is 736, and
  the whole 52-site prepared operator fits under the M38 byte budget, so the caches stop
  evicting and M38's recompute tax is gone for exactly the workload that paid it.

  **The engine is one path, and this is where that is settled.** `Env.heff2` runs the
  prepared, symbolic, term-family matvec for every MPO that carries an edge description —
  `from_terms` and `from_arrays`, at either cutoff. **That is block2's engine design
  adopted rather than re-invented**: its `EffectiveHamiltonian` never forms the effective
  Hamiltonian and dispatches the symbolic operator sum term by term against the
  wavefunction (`effective_hamiltonian.hpp`:230-243); the only thing it materialises is
  `diag`, for the preconditioner. A runtime dispatch between two matvec paths was
  measured, built and then **removed**: block2's algorithm choice is a build-time
  argument, and tenet's is `cutoff`. Everything later — parallelism, GPU execution, a
  Davidson preconditioner, one-site DMRG — attaches to this one path and to nothing else,
  which is the reason it had to stop moving before any of that starts.

  **The site-tensor branch is not a second engine; it is a compatibility entry.** It exists for
  an MPO that carries no symbols at all — `from_w` and a bare `MPO(sites)` — and no
  accelerator work targets it. It cannot be closed by recovering symbols from a numeric
  `W`, because in general there are none to recover: #141 measured that a compressed `W`
  retains no edge structure. Closing it would mean *refusing* externally-built MPOs, which
  is a decision about the public surface and not part of #204. block2 has no equivalent
  because block2 is a quantum-chemistry **program** and never receives an operator from
  outside; tenet is a library, and essentially every MPO in the literature is written as a
  `W` matrix, so refusing one would close the library in a way block2 never has to
  consider. This is the same class of judgement that kept disk spilling and the operator
  vocabulary out: adopt block2's engine, not block2's role.

  **The `chi` scaling grid, as information rather than as a decision.** Two DMRG sweeps
  per point, the two routes selected the only way a caller can select them (the operator
  with its description, against `MPO(h.sites)` — the same tensors with no description).
  `ratio` is prepared / dense, so below 1.00 is the prepared path ahead.

  | model | D_w | chi | prepared | dense | ratio | RSS prep | RSS dense |
  |---|---|---|---|---|---|---|---|
  | N=20 U(1) Heisenberg + NNN | 8 | 16 | 10.44 s | 3.25 s | 3.21x | 0.24 G | 0.11 G |
  | | 8 | 64 | 12.65 s | 3.84 s | 3.29x | 0.27 G | 0.12 G |
  | | 8 | 256 | 12.13 s | 3.77 s | 3.22x | 0.36 G | 0.19 G |
  | N2 CAS 6-31G (K=16) | 562 | 16 | 3.50 s | 1.36 s | 2.57x | 1.31 G | 1.29 G |
  | | 562 | 64 | 15.59 s | 8.95 s | 1.74x | 2.99 G | 1.84 G |
  | | 562 | 128 | 65.44 s | 34.48 s | 1.90x | 5.49 G | 3.73 G |
  | C2 CAS cc-pVDZ (K=26) | 736 | 16 | 8.54 s | 3.84 s | 2.22x | 6.19 G | 6.09 G |
  | | 736 | 64 | 46.72 s | **47.98 s** | **0.97x** | 7.83 G | 7.85 G |

  Both routes agree on the energy at every point converged enough to compare — to 1e-13 on
  the lattice model and to 1e-12 on N2 at all three `chi`. (The two C2 rows at `chi=16`
  disagree, 100.6 against 60.7, and that is not a path disagreement: two sweeps at
  `chi=16` on a 52-site strongly correlated system from a random start is nowhere near
  converged, and at `chi=64` the two routes land on the same `39.86930419362358`.)

  **What the grid says.** The lattice ratio is flat in `chi` at ~3.2x: at `D_w = 8` there
  is nothing for a symbolic dispatch to save and the per-bond core construction is pure
  overhead — which is exactly why `cutoff=None` is the lattice-model setting, since the
  finite-state machine keeps the identity channels separable and the same run costs 1.96 s
  against 3.53 s. On ab initio integrals the ratio **narrows in `chi` on the widest bond**,
  reaching 0.97x on C2 at `chi=64`: the per-bond cores do amortise as the two-site tensor
  grows, which is the question the M38-era `chi=16` measurement could not answer. That is
  the honest position of the engine today — a constant factor against the compatibility
  entry's dense contraction at small `chi`, at parity by `chi=64` where the bond is widest
  — and where the optimisation work goes is *inside* this one path, never into a second.

  **The constant factor on a lattice model at a float cutoff is real and is not engineered
  around.** N=20 U(1) Heisenberg, `chi=64`, four sweeps: `cutoff=None` **1.96 s**, default
  `cutoff=1e-13` **3.53 s**, same ground state `-8.682473334398`. The mechanism is named
  rather than hidden: the compressing rotation mixes the open states, so a spectator no
  longer separates onto the rank-2 identity ride — `spec_op` is `None`, `a_real_op` is the
  whole open group — and every open state is operator-carrying. This is not a defect #204
  introduced; it is the uniform mechanism block2 already uses, where the identity is an
  ordinary entry in the `ops` map and an explicit entry in the symbolic matrix, paid like
  any other operator. block2's lattice performance is fine for the same reason tenet's is
  at `cutoff=None`: a spin chain's bond is small in the first place. The knob is `cutoff`,
  it is documented in `MPO.from_terms`' and `Env.heff2`'s Notes, and the caller sets it at
  build time.

  What #204 buys is therefore the *representation*: the operator no longer has to choose
  between being compressed and being partitioned, K=26 completes at 6 GiB where it did
  not complete at 19–24, and `heff2`'s prepared path lost its `cutoff=None` precondition —
  which is what makes it one path rather than two.

  **The pre-placement basis choice is the named successor**, for the K where the build
  transient walls (`D_FSM × d² × chi`; benzene at K=108 extrapolates to ~10² GB). Its shape
  is block2's, now read first-hand rather than inferred: a left-to-right streaming pass
  whose rows are (previous cut's kept basis) × (site operators) — `expr_index_hash` includes
  the kept-basis index (`general_mpo.hpp`:540-541) and Part 6 rebuilds the coefficient
  stream on the kept basis — with both sides interned per cut and a dense per-quantum-number
  SVD of the `szl × szr` block (:763, :828-833), the singular values folded into the next
  site's coefficients rather than into a tensor. It gets its own gate (pre-placement against
  post-placement width) when it is filed, and it slots in behind this milestone's carrier
  without moving the interface again.

- **M54** — shipped: `dmrg_` forwards a caller-supplied `compile=` to its `Env`, and the
  compiled matvec is measured at quantum-chemistry scale for the first time (#220). The code
  change is one keyword-only parameter; the measurement is the content.

  `Env.__init__` has taken `compile=` since M16 and `dmrg_` built its `Env` without a way to
  pass one, so every run through the top-level driver was uncompiled *by construction* and
  the only route to the engine's other regime was to build `Env` by hand and drive `sweep_`
  in a loop. The argument stays a callable the caller supplies — no module-level default, no
  accelerator named at this layer — and the default `None` leaves today's behaviour
  untouched.

  **The instrument.** `benchmarks/bench_dmrg_compile.py`, one process per point so the peak
  RSS is that point's own, two sweeps from a random full-rank seed at `cutoff=1e-10`. The
  lattice model is N=20 U(1) Heisenberg with a next-nearest-neighbour term at `cutoff=None`
  (`D_w = 8`); the two ab initio inputs are `bench_qc_mpo.py`'s FCIDUMPs at the default
  cutoff — M39's pinned operator, which is what #218's grid measured and the only route that
  finishes at K=26. `steady` is the mean per-matvec wall *after* the first call on each
  compiled callable; `trace` is the sum of the first-call latencies with one steady run
  subtracted from each, which is what the XLA trace and compile cost and nothing else. The
  `none (JAX)` row is the same JAX backend with no `compile=` at all, present so that a
  slower sweep is attributed to the right thing.

  | model | chi | compile | steady matvec | first call | trace total | sweep wall | peak RSS |
  |---|---|---|---|---|---|---|---|
  | lattice, N=20, `D_w`=8 | 16 | none (NumPy) | 1.595 ms | 35.0 ms | 2.4 s | 8.35 s | 0.21 G |
  | | 16 | none (JAX) | 13.710 ms | 667.1 ms | 47.7 s | 194.73 s | 7.26 G |
  | | 16 | **`jax.jit`** | **0.151 ms** | 178.2 ms | 13.0 s | 175.86 s | 7.17 G |
  | | 64 | none (NumPy) | 1.881 ms | 44.3 ms | 3.1 s | 9.97 s | 0.23 G |
  | | 64 | **`jax.jit`** | **0.215 ms** | 232.7 ms | 17.0 s | 755.94 s | 8.67 G |
  | | 128 | none (NumPy) | 1.673 ms | 37.9 ms | 2.6 s | 8.65 s | 0.24 G |
  | | 128 | **`jax.jit`** | **0.226 ms** | 229.0 ms | 16.7 s | 702.68 s | 8.38 G |
  | N2 CAS 6-31G, K=16 | 16 | none (NumPy) | 3.361 ms | 5.4 ms | 0.3 s | 3.38 s | 1.50 G |
  | | 16 | **`jax.jit`** | **1.333 ms** | 54.8 ms | 6.5 s | 79.47 s | 5.91 G |
  | | 64 | none (NumPy) | 25.051 ms | 28.1 ms | 0.4 s | 15.40 s | 2.96 G |
  | | 64 | **`jax.jit`** | **8.666 ms** | 78.4 ms | 8.4 s | 92.54 s | 8.01 G |
  | | 128 | none (NumPy) | 111.616 ms | 117.7 ms | 0.7 s | 61.44 s | 5.41 G |
  | | 128 | **`jax.jit`** | **37.632 ms** | 122.0 ms | 10.2 s | 137.91 s | 13.41 G |
  | C2 CAS cc-pVDZ, K=26 | 16 | none (NumPy) | 4.773 ms | 9.9 ms | 1.0 s | 8.30 s | 6.21 G |
  | | 16 | **`jax.jit`** | **2.670 ms** | 181.5 ms | 35.9 s | 676.95 s | 12.08 G |
  | | 64 | none (NumPy) | 46.678 ms | 52.6 ms | 1.2 s | 45.79 s | 7.72 G |
  | | 64 | **`jax.jit`** | **19.858 ms** | 254.0 ms | 47.1 s | 817.59 s | 11.75 G |
  | | 128 | none (NumPy) | 222.853 ms | 235.0 ms | 2.4 s | 200.01 s | 8.59 G |
  | | 128 | **`jax.jit`** | **92.965 ms** | 377.8 ms | 57.3 s | 1076.71 s | 12.93 G |

  Sweep wall excludes the operator build (4.6 s for N2, 40 s for C2, both routes). The NumPy
  rows reproduce #218's grid where they overlap — C2 at `chi=16` and `chi=64` is 8.30 s and
  45.79 s here against 8.54 s and 46.72 s there — and the compiled and uncompiled arms agree
  on the energy to 1e-13 on the lattice at every `chi` and on N2 at every `chi`. They do
  *not* agree on C2, where the two sweeps from a random start on a 52-site strongly
  correlated system are nowhere near converged and a reordering of floating-point sums moves
  the run onto a different bond structure; #218 records the same instability at that point
  for two NumPy routes. Correctness of the compiled path is settled by the lattice and N2
  agreement, not by C2.

  **The compiled matvec is real, and its factor shrinks as the bond widens.** 10.6x on the
  lattice, 2.5–3.0x on N2, 1.8–2.4x on C2. The direction is the finding: at `D_w = 8` nearly
  all of the eager cost is Python-level dispatch, which a trace removes outright, while on a
  bond of 562 or 736 the same matvec already spends its time inside BLAS, where a trace has
  little left to take. **#141's ~20x is a small-`D_w` number**, and reading it as an ab
  initio number would have been wrong.

  **The trace does not explode with K — it is re-paid per bond visit.** `Env.heff2` keys
  `_compiled` per bond and rebuilds the entry when `hit[1] is not p` (`env.py`:662), and
  `_prepare2` returns a *new* `_Prepared` whenever either environment moved, which during a
  sweep is every visit. So the compiled callable is discarded and re-traced at every bond the
  sweep touches: the lattice run makes **73 `compile()` calls against 60 distinct structure
  keys**, N2 **121 against 34**, C2 **201 against 109** — in each case about `2 x sweeps x
  bonds`, and on N2 that is 121 traces for 34 distinct graphs. This is the third outcome the
  question allowed for, and it is the one that happened: not "the trace is cheap once", not
  "the trace explodes with the number of blocks", but *the trace is charged again every
  time*. A single trace costs 55–122 ms at K=16 and 181–378 ms at K=26, so it does grow with
  the block count, mildly and not alarmingly; what makes it dominant is the multiplier.

  **The compiled sweep is slower end to end, and `compile=` is not what makes it slower.**
  On the lattice at `chi=16` the compiled sweep is 175.86 s against 8.35 s, of which 13.0 s
  is tracing and 0.02 s is matvec — 92 % of it is the *rest of the sweep* running eagerly on
  the JAX backend. The `none (JAX)` row settles the attribution: 194.73 s with no `compile=`
  at all, and a matvec 8.6x *slower* than NumPy's. The truncating SVD, the canonicalization
  and the environment updates sit outside `heff2` and outside any trace — they are the
  data-dependent control flow `tenet.network` refuses to trace by construction (M11) — and on
  JAX they pay per-operation dispatch with nothing amortising it. Peak RSS follows at 5.9–13.4
  GiB against 0.2–8.6 GiB, part JAX's own runtime and part the traced executables, which
  accumulate precisely because each `compile()` call produces one that is never reused.

  **Verdict: the compiled path does not fail at K=26, and it does not close the
  constant-factor question either.** It survives — 201 traces at 52 sites complete, the
  per-trace cost grows only mildly from K=16 to K=26, and peak memory stays inside the same
  order as the NumPy run — so the structural failure the issue named (an XLA graph whose
  trace cost scales unlike `ConnectionInfo`'s linear index table) **did not materialise**.
  What replaces it is a lower ceiling and a wasted multiplier: at ab initio scale the matvec
  is only ~2x faster compiled, the trace is paid `2 x sweeps x bonds` times for a handful of
  distinct graphs, and the sweep around it is not traceable at all. Compiling is therefore
  not a route to an order of magnitude on this workload today, and `compile=None` remains the
  right default.

  **What this does not decide.** Whether tenet grows a `ConnectionInfo`-shaped contraction
  plan — a flat index-and-coefficient table built once per structure and reused, the
  backend-neutral analogue of what `jit` does for one backend — is a separate decision with
  this measurement as its input, and it is not taken here. What the measurement contributes
  is specific rather than directional: the reuse `ConnectionInfo` gets for free is the reuse
  the identity test at `env.py`:662 throws away, `n_keys` says how much reuse is actually
  available (34 distinct graphs where 121 traces are taken), and a plan would amortise on the
  NumPy path, where the other 90 % of the sweep already runs.

- **M57** — shipped: one sweep is decomposed phase by phase on the NumPy backend (#225),
  and the compiled callable stops being re-traced at every bond visit. The measurement is
  the deliverable; the code change is four lines in `Env.heff2`.

  **Why decompose.** M54 measured the matvec and, by arithmetic on its own numbers, put it
  at roughly 40 % of an N2 K=16, `chi=128` sweep. The other ~60 % had never been looked at,
  and M54's narrative attributed it to "the truncating SVD, the environment updates, the
  two-site assembly" without measuring which. #206 exists for the SVD on the strength of
  that reading.

  **The instrument.** `benchmarks/bench_sweep_phases.py`, on no CI path. It **wraps call
  sites** rather than sampling: `tenet.linalg.svd_truncated`, `MPS.__setitem__`,
  `Env.heff2`, `Env._prepare2`, `Env.update_`, `dmrg.lanczos`, `dmrg.spectrum`, an identity
  `compile=` stub around `_apply2`, and a proxy over `dmrg.py`'s own `tenet` module global
  so that the three einsums the *sweep* spells are timed while the hundreds inside the
  matvec are not. Nested timers are subtracted, so `lanczos` is reported net of `heff2` and
  `heff2` net of `_prepare2` and `_apply2`. Every point is run in two arms, `plain` (no
  wrapper at all) and `wrapped`, and both walls are reported; **`residual` is wall minus
  the phases and is never distributed**, because a residual is the instrument saying a
  phase is missing.

  One **discarded warm-up sweep** precedes each point. The edge tables, group embeddings
  and merged cores are all built on a bond's *first* visit, so without it whichever arm ran
  first would carry the operator's whole construction — measured at 7x on the lattice, an
  effect larger than anything the table is about. The wall reported is therefore one steady
  sweep, `Env.setup_` and `MPS.canonize_` excluded, which is why it is smaller than M54's
  per-sweep figure for the same point.

  Shares are of the wrapped wall. `apply` is `_apply2`, `ncv=3` per solve; `prepare` is
  `_prepare2`'s environment fold, once per bond visit; `env` is `Env.update_`, i.e.
  `_fold_last`/`_fold_first`; `write` is the `MPS.__setitem__` write barrier plus the two
  gauge einsums.

  | model | chi | assemble | lanczos own | prepare | **apply** | heff2 rest | svd | **env** | write | spectrum | residual | wall (wrapped / plain) |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | lattice, N=20, `D_w`=8 | 64 | 1.83 % | 10.30 % | 15.84 % | **45.81 %** | 0.50 % | 3.27 % | **19.90 %** | 1.77 % | 0.23 % | 0.55 % | 0.504 s / 0.506 s |
  | N2 CAS 6-31G, K=16 | 16 | 0.61 % | 3.87 % | 14.22 % | **59.78 %** | 0.67 % | 1.08 % | **18.76 %** | 0.71 % | 0.07 % | 0.23 % | 1.143 s / 1.148 s |
  | | 64 | 0.13 % | 1.05 % | 4.76 % | **71.73 %** | 0.26 % | 0.55 % | **21.33 %** | 0.13 % | 0.01 % | 0.05 % | 6.780 s / 7.052 s |
  | | 128 | 0.05 % | 0.31 % | 2.67 % | **74.61 %** | 0.19 % | 0.37 % | **21.67 %** | 0.07 % | 0.00 % | 0.06 % | 27.780 s / 27.894 s |
  | C2 CAS cc-pVDZ, K=26 | 16 | 0.51 % | 3.08 % | 13.31 % | **62.37 %** | 0.26 % | 0.89 % | **18.78 %** | 0.55 % | 0.06 % | 0.19 % | 2.315 s / 2.319 s |
  | | 64 | 0.08 % | 0.56 % | 4.34 % | **73.28 %** | 0.18 % | 0.36 % | **21.10 %** | 0.07 % | 0.01 % | 0.03 % | 19.718 s / 20.190 s |
  | | 128 | 0.03 % | 0.20 % | 2.68 % | **74.65 %** | 0.18 % | 0.23 % | **21.99 %** | 0.02 % | 0.00 % | 0.02 % | 91.315 s / 92.130 s |

  **The slack is the residual column: at most 0.55 %, and 0.06 % or less at every `chi=128`
  point.** Nothing is unaccounted for. The wrapping cost is separately bounded: the wrapped
  wall is between 3.9 % *below* and 0.2 % below the plain wall at every point — never above
  it — so the instrumentation sits inside run-to-run noise and no arm needs a correction.

  **What the table says, and it is not what M54's narrative implied.**

  1. **The two-site matvec is the sweep, not 40 % of it.** 60 % at `chi=16`, 72–75 % at
     `chi=64` and `chi=128`, on both quantum-chemistry inputs. M54's ~40 % came from
     multiplying a steady per-matvec time by a matvec count against a wall that also
     contained `Env.setup_`, `canonize_` and every bond's first-visit construction; taken
     against the *steady sweep* the matvec is where the time is. The `chi` trend is the
     confirmation: the matvec's share **rises** with the bond while every Python-side phase
     falls away.
  2. **`svd_truncated` is 0.23–1.08 %, and it falls as `chi` rises.** It is the smallest
     numerical phase in the table at every point except the lattice, where it is 3.27 %.
     **#206 is a rounding error on this workload.** M54's own remainder story named it
     first, and #225 was filed saying "obvious suspect" is exactly the reasoning M54
     falsified — it was falsified again, on the suspect M54 named. A perfect SVD, free,
     buys 0.23 % of a C2 `chi=128` sweep. #206 is therefore demoted on the strength of this
     table; if it is revived it needs a workload where the SVD is not this small.
  3. **The environment update is the second phase, and it is stable at ~19–22 %
     everywhere** — every model, every `chi`, lattice included. It is the only phase besides
     the matvec that does not shrink with `chi`, and nothing was pointing at it. `_fold_last`
     / `_fold_first` are the same shape of contraction as `_apply2` against a rank-3
     environment instead of a rank-4 wavefunction, so a matvec win transfers to it — which
     makes the *engine's contraction*, not the sweep's scaffolding, the whole 95 % that is
     worth attacking.
  4. **Everything Python-side is noise at scale.** Assembly, write-back, `spectrum`, the
     Lanczos recurrence and `heff2`'s own cache lookup sum to **0.61 %** of a C2 `chi=128`
     sweep and 0.71 % of an N2 one. At `chi=16` they reach 5–6 %, and on the lattice 14 % —
     the regime split M54 saw in the compiled matvec is visible here in the same direction.
  5. **`_prepare2`'s fold is a small-bond cost.** 13–16 % at `chi=16` against 2.7 % at
     `chi=128`: the fold is paid once per bond visit while the matvec is paid `ncv` times
     and grows faster, so the amortisation M16 argued for gets better, not worse, with
     `chi`.

  So the next target is the two-site contraction itself on the NumPy path — `_apply2` and
  the environment folds that share its shape — and not the SVD, not the solver's Python
  overhead, and not the sweep's scaffolding. What that work *is* is a separate decision with
  this table as its input; #219's better solver reduces the *number* of matvecs and now has
  a measured ceiling of 75 %, which is a larger prize than M54's arithmetic suggested.

  **The trace-refetch defect, fixed.** M54 found compiled callables discarded and re-traced
  on every bond visit — about `2 x sweeps x bonds` in every case — because `env.py`:662
  rebuilt the cache entry when `hit[1] is not p` and `_prepare2` returns a new `_Prepared`
  on every visit, so the identity test never held. The compiled callable is now kept
  whenever the structure key is unchanged. The key was **checked rather than assumed to be
  complete**: at a fixed bond the live fields of the prepared operator are decided by the
  two sites' edge blocks, which never move, and every leg of every field is fixed by `aa`'s
  two bond legs plus the operator's own, so `(bond, tuple(aa.legs))` does determine the
  traced graph and needed no widening.

  | model, two sweeps at `chi=64` | `n_compile` before | `n_compile` after | distinct keys |
  |---|---|---|---|
  | lattice, N=20 | 73 | **57** | 57 |
  | N2 CAS 6-31G, K=16 | 121 | **35** | 35 |
  | C2 CAS cc-pVDZ, K=26 | 201 | **143** | 115 |

  Lattice and N2 hit the floor exactly. C2's 28 extra compiles are **not** the identity
  defect: they are bonds whose `_compiled` entry the M38 byte budget evicted and which then
  recompiled on the next visit. The entry is weighed by the `_Prepared` it carries, and
  `_prepared` already holds that same object, so the weight is counted twice across the two
  caches — `common.payload`'s docstring records that as the safe direction for a budget, and
  here it is what costs the last 28 traces. Removing `p` from the `_compiled` entry would
  close the gap and would also take `_compiled` out of the byte budget entirely, which is a
  change to M38's cache policy and belongs with whoever revisits it, not here.

  `_prepare2`'s own identity discipline is deliberately untouched: the prepared operator's
  *values* change on every visit, and serving a stale one is the plausible-and-wrong energy
  `Env`'s docstring names as the worst failure mode a DMRG has. Two tests pin the pair —
  one counts a stub `compile`'s invocations against the distinct keys of two sweeps, one
  drives an environment rebuild and checks the operator is rebuilt with it and agrees with a
  freshly built `Env`.

Not planned: TDVP, iDMRG, excited states, fermionic swap gates and PEPS containers.
Fermionic swap gates stay not planned for a stronger reason than before: fermionic
DMRG shipped without them (M21/#147) — the fZ2 braiding is the Jordan-Wigner string,
and the gap M13's refusal guarded against was the cap-direction convention M23/#160
fixed with the composition rule stated in the Milestone 11 section above, not a missing
gate.

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