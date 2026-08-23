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
import tenet                 # requires jax; import-guarded, never imported by core
tenet.enable_jax()
```

(`import tenet.pytree` is what that call runs, and remains a working spelling; see
M46 below for why the `tenet.ad` half is a keyword rather than part of the default.)

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
`atol=tenet.PROJECT` is the "project, don't check" spelling, and the only one that
traces (the comparison is a concrete-value question). `tenet.PROJECT` **is**
`math.inf` — the mode is the limit of the tolerance — so the two spellings are one
call, and `restrict` and `to_symmetry` take it in the same position (#210).

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

- **M53** — **refused at Part 1, on measurement (#219).** The two-site effective
  Hamiltonian's diagonal *cannot* be produced from the same `_Cores`/`EdgeBlocks` the matvec
  uses, and the obstruction is categorical rather than arithmetic. `lanczos` is unchanged, no
  solver was added, and `Env` grew no method. What shipped is the instrument and this entry.

  **What was asked.** block2's engine has two halves and M39 adopted one. The second is the
  diagonal preconditioner: `effective_hamiltonian.hpp`:141-146 allocates `diag` behind a
  `compute_diag` constructor flag, :191-200 fills it, :491-492 asserts it and hands it to
  Davidson, and `iterative_matrix_functions.hpp`:66-72 is the whole preconditioner —
  `q_i /= lambda - aa_i`, Davidson 1975 §III.D. The premise to establish first was that
  `Env` can build `diag(H_eff)` at a bond without forming `H_eff` and without a full-width
  intermediate.

  **The instrument.** `benchmarks/bench_heff2_diagonal.py`, on no CI path. It needs no new
  library code to know the truth: the diagonal of the matrix the solver actually iterates on
  is obtained by probing [Env.heff2][tenet.network.Env.heff2] with the reduced basis' own
  unit vectors and reading the probed entry back — `dim(aa)` matvecs, which is a
  fixture-sized oracle and nothing more. Against that it puts the candidate contraction,
  the one every "take only the diagonal in the two-site index" reading means:

  ```text
  diag[a, p, q, r] = sum_{x, m, y} GL[a, x, a] W1[x, p, p, m] W2[m, q, q, y] GR[r, y, r]
  ```

  evaluated on reduced blocks, over the site-tensor path so that the two `W`s are in hand.

  | provider | worst \|exact − candidate\| | the diagonal's own scale | verdict |
  |---|---|---|---|
  | U(1) Heisenberg (ungraded Abelian) | 1.11e-16 | 0.418 | exact |
  | fZ2 spinless fermions (graded Abelian) | 1.56e+00 | 0.782 | wrong by a sign |
  | SU(2) J1–J2 chain (non-Abelian) | 4.52e-01 | 0.943 | wrong index set |

  1. **Ungraded Abelian is the only case the candidate gets right**, and it gets it right to
     machine precision. That is the whole of the intuition the issue was written on, and it
     is a special case.
  2. **Graded Abelian fails by a parity-dependent sign**, not by a small amount: the ratio
     exact/candidate is exactly ±1 block by block, and it is −1 on three of the six blocks
     with a non-zero diagonal, so the error reaches twice the diagonal's scale. The braiding
     the matvec pays through `_composed` (#147, #160) is paid on the diagonal too, and a
     product of reduced-block diagonals does not carry it.
  3. **Non-Abelian does not fail numerically — it fails structurally.** The two-site
     reduced basis is labelled by a fusion *tree*, and for SU(2) one external sector tuple
     carries several. At the measured bond the tuple (1, ½, ½, 1) carries two inner lines and
     the exact diagonal on them is **−0.16586** and **+0.08648**: not equal, not related, and
     the candidate has no index to tell them apart. A leg-factorized contraction produces a
     value per external sector tuple, so it cannot even have the right *shape*, let alone the
     right values. Invariant 4 states this in advance — fusion-tree information is
     relational, not per-leg — and the diagonal is a per-leg reading of a relational basis.

  **This is what block2 does too, read correctly.** block2 does not reuse its matvec for the
  diagonal, and the reason is not performance: `initialize_diag`
  (`sparse_matrix.hpp`:81-140) is handed the coupling object `cg` and builds each diagonal
  entry with a **Wigner 9j** (:123-125) and an explicit fermionic sign (:130). It is a second
  coefficient path, parallel to the matvec's own (`sparse_matrix.hpp`:225 is the matvec's
  9j), not a cheap projection of the first. The measurement above is the same statement
  arrived at from tenet's side: the missing ingredient is exactly the recoupling coefficient
  and the braiding sign. tenet's job is also strictly harder than block2's here — block2's
  two-site wavefunction is a two-index coupled object, so one 9j closes it, while tenet's
  `aa` is rank-4 with a left-associated tree over `(a, p, q)` and the analogous coefficient
  is a chain of F-moves.

  **The cost was never the problem, and the instrument says so on the record** so that the
  refusal is not misread as a cost refusal. Dense, no symmetry, the candidate contraction
  against one matvec of the same shape:

  | chi | d | D_w | matvec | diagonal | ratio |
  |---|---|---|---|---|---|
  | 64 | 2 | 5 | 0.225 ms | 0.043 ms | 0.193 |
  | 128 | 2 | 5 | 2.238 ms | 0.046 ms | 0.021 |
  | 64 | 4 | 100 | 57.99 ms | 0.233 ms | 0.0040 |
  | 128 | 4 | 100 | 231.5 ms | 0.501 ms | 0.0022 |
  | 128 | 4 | 300 | 1070.6 ms | 2.328 ms | 0.0022 |

  The diagonal is **0.2 % of a matvec at quantum-chemistry scale** and at worst 19 % at the
  smallest lattice point, and it shrinks as `chi` grows because the matvec is `chi^3` where
  the diagonal is `chi^2`. Against M57's 74.6 % matvec share the prize is intact; nothing
  about the arithmetic argues against this milestone.

  **Two API refusals stand between here and a preconditioner, and both are deliberate.**

  * **Building it.** There is no public spelling for a diagonal, and
    `tests/network/test_hygiene.py::test_no_module_uses_reduced_blocks_numerically` forbids
    any `.blocks` read inside `src/tenet/network/`. `Env` therefore cannot write this method
    at all — not "should not", cannot, with a test that fails. A correct diagonal is a **core
    `tenet.ops` operation** that consumes the provider's F/R data, i.e. a new categorical
    contraction mode, and it lands on the invariant-5 boundary ("categorical operations are
    never defined by backend array operations"). That is a milestone of its own with fusion
    trees, duality and braiding in it, not a driver-layer change, and #219 was scoped as the
    latter.
  * **Applying it.** `tenet.multiply`/`divide` refuse a `SymmetricTensor` operand in as many
    words — *"elementwise products of two SymmetricTensors are not a defined categorical
    operation"*. `q / (lambda - diag)` is exactly that product, and block2's own
    preconditioner is a flat loop over storage (`iterative_matrix_functions.hpp`:69-71). This
    half is the smaller one: [apply_blocks][tenet.apply_blocks] already admits *unary*
    coefficient-space elementwise maps and says so ("**Coefficient space, not dense space**"),
    so the missing primitive is its binary sibling over two tensors sharing one structure. It
    is a small, well-defined public addition — but it is a public addition, and it should be
    made by whoever owns the diagonal, not ahead of it.

  **What this closes and what it does not.** It closes "the diagonal is a contraction over
  the blocks `_cores2`/`_build2` already assemble" — it is not, on three of the four
  providers the repository tests. It does **not** close the preconditioned solver: a Jacobi
  preconditioner need not be exact to work, and an approximate diagonal (the candidate above,
  which is exact on U(1) and sign-wrong on fZ2) may still cut the iteration count. That is a
  different claim with a different acceptance test — measured iteration counts, not agreement
  with a formed `H_eff` — and it still needs the binary `apply_blocks` above before it can be
  spelled. A revival takes one of two shapes, and should say which: the **exact** one, which
  is a core categorical operation and carries the SU(2) criterion, or the **approximate**
  one, which is a solver experiment and must drop that criterion explicitly rather than
  quietly.

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

- **M61 Stage A** — shipped: the two core primitives of the sweep step (#232, absorbing
  #231), both in `tenet.ops`, both additions only. [map_diagonal][tenet.map_diagonal] is the
  diagonal of a square map in the reduced basis; [zip_blocks][tenet.zip_blocks] is
  `apply_blocks`' two-argument sibling.

  **What the diagonal turned out to be, against what M53 predicted.** M53 refused the
  leg-factorized candidate and recorded that a correct diagonal "consumes the provider's
  F/R data, i.e. a new categorical contraction mode". Built, it consumes none, and the
  reason is the one already written at the top of `ops/map.py`: composition is one
  `matmul` per coupled sector and nothing recouples, because both sides of the join
  enumerate the same trees in the same order and Clebsch-Gordan orthonormality collapses
  the shared index. So for a square map the matrix a solver iterates on *is*
  `to_matrices(m)`, and its diagonal is the blocks whose two fusion trees coincide, read
  with one `einsum` per block. `Leg.fused_sector` reads `dual` and never `side`, so a
  square map's codomain and domain draw their trees from one set and "the two trees
  coincide" is well defined. No F-symbol, no R-symbol, no twist, no bend: the capability
  set is empty beyond the fusion rules every structure already needs, and `map_diagonal`
  therefore declares no `requires` and no protocol was added to the M24 lattice.

  That is not a retraction of M53's measurement, and the distinction is the whole content
  of this stage. What #230 refuted was *manufacturing* the diagonal from per-leg diagonals
  of the operator's factors — a per-leg reading of a relational basis (invariant 4), which
  loses the inner line on SU(2) and the graded braiding sign on fZ2. Given the map itself,
  both are already in its coefficients, and the F-moves and the sign were paid by whoever
  contracted it. The recoupling did not disappear; it moved to the caller's `einsum`.

  **The four-provider gate, and the constructed SU(2) case.** The oracle is an explicitly
  formed dense `H_eff`: `to_dense` on the assembled map, then `<e|H|e> / <e|e>` on the
  dense image of each reduced-basis unit vector — basis-free, so not a restatement of the
  implementation. A second oracle probes the public `compose(m, e)`. Worst deviation on
  `tests/ops/test_map_diagonal.py`'s fixtures: U(1) `0.0`, fZ2 `0.0`, the spinful-Hubbard
  `d=4` grading `0.0`, SU(2) `1.2e-15` on a diagonal of scale `7.08`. The SU(2) two-inner-line case is *constructed*, not
  encountered: legs `(j=1, ½, ½, 1)` carry exactly two fusion trees over one external
  sector tuple, inner lines `j=½` and `j=3/2`, and both entries are pinned —
  `0.2917652786409168` and `0.8465778161839884` on the assembled `H_eff`,
  `-0.45467078517172255` and `-0.02925182246327349` on a random square map on the same
  legs. `benchmarks/bench_map_diagonal.py` then runs the *real* DMRG object against #230's
  own probing oracle: worst `|exact − map_diagonal|` is `5.6e-17` on U(1), **exactly `0.0`**
  on fZ2 and `7.4e-15` on SU(2), where #230's candidate was off by `1.56` and `0.45` on the
  same fixtures.

  **The cost figure is corrected, not confirmed.** #230 put the diagonal at 0.2 % of a
  matvec; that was the price of the candidate, which is wrong. On the real implementation:

  | | one `Env.heff2` | form `H_eff` | `map_diagonal` |
  |---|---|---|---|
  | U(1) Heisenberg | 1.10 ms | 2.33 ms (2.1x) | 0.021 ms (**1.9 %**) |
  | fZ2 spinless | 0.74 ms | 1.72 ms (2.3x) | 0.017 ms (**2.3 %**) |
  | SU(2) J1–J2 | 0.79 ms | 2.94 ms (3.7x) | 0.013 ms (**1.7 %**) |

  Reading the diagonal is cheap. *Having* the map is not: `H_eff` is `chi^4 d^4`, already
  2 GiB at `chi=64, d=2` and 512 GiB at #230's quantum-chemistry points, where forming it
  is simply unavailable (the dense section of the benchmark reports the ceiling rather than
  extrapolating past it). So Stage B cannot reach the preconditioner by forming the map and
  calling this: it needs the diagonal built from the same `_Cores` the matvec uses, which is
  block2's `initialize_diag` (`sparse_matrix.hpp`:81-140) — a second coefficient path with
  its own 9j (:123-125) and its own fermion sign (:130). `map_diagonal` is that object's
  **oracle**, exact and public, and the agreement section of the benchmark is the harness a
  from-the-pieces implementation gets tested with. Recorded here so the stage is not later
  read as having delivered the cheap route.

  **The three design decisions, with their reasons.**

  * *What it returns.* A `SymmetricTensor` on the map's codomain legs — the structure the
    vectors it acts on carry, so `zip_blocks` pairs the two block for block and
    `q / (lambda - diag)` type-checks. Those legs are all OUT, hence unit-coupled, and that
    is the whole diagonal on the vectors tenet can represent: a `SymmetricTensor` on them is
    invariant by construction (invariant 1), and a targeted charge is carried by an explicit
    charge leg, which is then a leg of the map too.
  * *The refusal.* Not square → `check_square`'s existing message, which names the first
    offending position and both legs. No new checker: the predicate is
    `ProductSpace.matches`, and this is its sixth caller.
  * *The general square map, and the partition the caller has to arrive on.* The two-site
    case is not special-cased; the implementation is rank-generic (one einsum subscript per
    paired axis, 26 of them). The environments hand the operator over on `aa`'s own
    partition — right bond IN on the ket side, OUT on the bra side — so the caller pays one
    `repartition` to reach `(a p q r | a' p' q' r')`. That bend is deliberately *not* done
    inside `map_diagonal`: it needs `BendingCoefficients`, and hiding a capability
    requirement inside an operation that otherwise needs none would make the refusal
    provider-dependent for no gain. It is also harmless to the answer — a bend is one scalar
    per block (`BendPlan.terms`), i.e. a diagonal similarity, and `diag(S M S^-1) = diag(M)`
    — which is why the same numbers come back on `aa`'s basis, and why a solver may
    precondition in either partition. `tests/ops/test_map_diagonal.py` pins that
    one-scalar-per-block claim on U(1) and SU(2) rather than asserting it in prose.

  **`zip_blocks`, and why it does not reopen `multiply`'s refusal.** `multiply` refuses a
  second `SymmetricTensor` because `a * b` asks for a *dense* elementwise product, which has
  no expression in the reduced blocks — `T = Σ_τ A^(τ) ⊗ C^(τ)` — so the plausible blockwise
  answer would be a silently different tensor. `zip_blocks` makes the opposite declaration
  in its name and signature: a map over coefficients, supplied by the caller, with no claim
  that it commutes with dense expansion; it is unreachable from `*` or `/` and requires one
  shared structure, which is what makes "the aligned block pair" mean anything (`block_order`
  is a pure function of the structure). The same argument already licensed the unary
  `apply_blocks`; arity was never what was in question. `multiply`'s refusal test stands
  unedited, and so does
  `tests/network/test_hygiene.py::test_no_module_uses_reduced_blocks_numerically` — the
  placement in `tenet.ops` is what that fence was demanding.

  **No allocation claim is asserted in prose.** It is instrumented on `autoray.do`'s
  outputs: every array `map_diagonal` produces is at most the size of one block of its own
  *result*, with the `to_matrices` route — which concatenates a full-width matrix per
  coupled sector to read its diagonal — as the positive control that trips the instrument.
  A second test breaks `to_dense` and `to_matrices` outright and runs the operation anyway.
- **M61 Stage C** — shipped: the sweep gains block2's **default decimation** and the
  **perturbative noise** that rides on it (#232 Stage C, absorbing #223). `sweep_` and
  `Sweep` gain one keyword-only `noise_type`; `dmrg_`, `lanczos`, `Env.heff2` and the MPO
  builders are unchanged.

  **What block2 actually does, and why the two arrive together.** Its defaults are
  `noise_type = NoiseTypes::DensityMatrix` and `decomp_type = DecompositionTypes::DensityMatrix`
  (`sweep_algorithm.hpp`:104-106). `update_two_dot` (:814-1008) solves, then
  `density_matrix(...)` builds `rho = aa aa^dag` and folds the perturbation in
  (`moving_environment.hpp`:3554, :3636), and `split_density_matrix` (:4250) takes an
  eigendecomposition of `rho` under the cutoff. The SVD split (:4079) is the *other*
  branch, and the wavefunction noise M14 shipped lives on that branch **only**
  (`sweep_algorithm.hpp`:964-978). The perturbation vectors come from
  `h_eff->perturbative_noise(...)` (`effective_hamiltonian.hpp`:263-360): one vector per
  sub-label of the symbolic operator sum — **the noise is the operator's own action on the
  current state, resolved by term family, not randomness.** So the mixer needs a density
  matrix to enter, and the density matrix is only worth building if something is being
  mixed into it; porting either alone re-derives the fragment the other closes. block2 is
  GPL-3.0 and this repository is Apache-2.0: everything here is description in original
  words with `file:line` citations, and no code, comment or docstring crossed.

  **The perturbation vectors, in tenet.** `Env.heff2_families(n, aa)` returns the prepared
  matvec's term families applied to `aa` separately — the identity-through ride, the two
  one-sided `IdL`/`IdR`-anchored sums, and the two open-to-open `AA` remainders, which are
  exactly the branches `_apply2` already dispatched over and this engine's analogue of
  block2's per-sub-label resolution. `_apply2` now sums what `_families2` builds, in its
  accumulator's order, so the matvec is unchanged term for term; the families are a *read*
  of the same `_prepare2` cache, not a second engine, and they are deliberately not
  compiled (`compile=` wraps the summed matvec, which a Krylov solve calls thousands of
  times; this is called once per bond visit). An MPO with no edge description has no
  families to resolve and gets the single vector `H_eff aa` — a weaker mixer, and what an
  operator carrying no symbols can offer. Scaling is block2's
  (`moving_environment.hpp`:3698-3713): each vector normalized, then the collection scaled
  to total squared norm `noise`, so against a unit-norm `aa` the parameter stays
  dimensionless and block2's 1e-4..1e-5 range transfers unchanged.

  **The truncation is the existing one, not a second one.** `rho` is built with
  `tenet.compose` and `tenet.adjoint` on the bent `(l, p | q, r)` partition — a density
  matrix *is* a map composed with its adjoint, so the composition rule (M23/#160) is met
  by construction and no operand order is left to state — and then handed to
  `tenet.linalg.svd_truncated` at `cutoff_mode="rsum1"`. For a Hermitian positive `rho`
  that is an eigendecomposition, its singular values are `aa`'s squared, and `rsum1` on
  that spectrum is *the same keep rule* as the default `rsum2` on `aa`'s: both drop the
  largest set whose `qdim`-weighted `sum sigma^2` stays under `cutoff` times the total.
  `max_bond` walks the same order because squaring is monotone. **So the kept bond space
  is identical by construction rather than by luck**, and `chi`/`cutoff` keep their
  documented meanings without a second selection rule to maintain. The discarded weight
  and the Schmidt spectrum keep theirs too: the factor that carries the weight is the
  truncated state and the other is an isometry, so its norm is the same Pythagoras
  `norm(s)` spells on the SVD branch, and `s` itself is `tenet.block_sqrt` of the returned
  spectrum, which is the matrix square root because that tensor is diagonal.

  **The selection rule is schedule-level, never a runtime dispatch** (#218, reaffirmed):

  | `(noise, noise_type)` | the split |
  |---|---|
  | `noise == 0.0`, any `noise_type` | `svd_truncated` of `aa` |
  | `> 0`, `"wavefunction"` | `svd_truncated` of a perturbed `aa` |
  | `> 0`, `"perturbative"` | `eigh` of a perturbed `rho` |

  No bond width, no `chi` threshold, no probe. A noiseless sweep — including a ramp's
  cooling tail — keeps the SVD split, and that pairing is deliberate rather than a default
  falling out: squaring `aa` into `rho` resolves a singular value `sigma` through
  `sigma^2`, so the split's own accuracy floor moves from machine epsilon to its square
  root, and a converged noiseless sweep is exactly where that costs something. block2
  makes the same pairing from the other side. An unrecognized `noise_type` raises rather
  than falling back, because a typo that silently ran the SVD split would report a mixer
  that never ran.

  The table above is `sweep_`'s docstring table, not a paraphrase of it, and the sentence
  under it there reads: *"A caller reads the rule off the `Sweep` entry: the density-matrix
  split engages exactly when perturbative noise is asked for, and a noiseless sweep --
  including the cooling tail of a ramp, and every sweep of a run that never mentions noise
  -- takes the SVD split."* `Sweep.noise_type` carries the same choice at schedule level,
  so `Sweep(32, noise=1e-4, noise_type="perturbative")` is the whole spelling a caller
  needs and `Sweep(32, noise=1e-4)` is unchanged from M14.

  **The two splits, measured against each other with nothing folded in.** Bond 2 of a
  three-sweep state, both directions, `cutoff=1e-14`
  (`tests/network/test_density_matrix.py` asserts the same thing at looser tolerances):

  | model | chi | bond space | max Schmidt difference | discarded-weight difference | the weight itself |
  |---|---|---|---|---|---|
  | U(1) Heisenberg N=10 | 4 | identical | 5.4e-16 | 0 | 4.3e-04 |
  | U(1) Heisenberg N=10 | 8 | identical | 5.4e-16 | 0 | 0 |
  | U(1) Heisenberg N=10 | 16 | identical | 5.4e-16 | 0 | 0 |
  | fZ2 Hubbard N=6, U/t=4 | 4 | identical | 3.9e-16 | 1.1e-16 | 2.7e-01 |
  | fZ2 Hubbard N=6, U/t=4 | 8 | identical | 3.9e-16 | 1.1e-16 | 1.0e-01 |
  | fZ2 Hubbard N=6, U/t=4 | 16 | identical | 3.9e-16 | 2.2e-16 | 3.0e-03 |

  "Identical bond space" is the `GradedSpace` compared as an object, not its dimension —
  the sectors and their degeneracies agree, which is the claim the `rsum1` argument above
  makes and the one a shape check would miss.

  **`noise=0.0` is the old path, and the measurement says so as loudly as this machine
  allows.** The `noise=0.0` branch is the pre-M61 body statement for statement, so the
  claim is really about the diff; a numerical check was run anyway, over 354 numbers —
  seven DMRG runs' per-sweep energies, energy changes, Schmidt changes and discarded
  weights, plus a full per-bond Schmidt spectrum, on the U(1) Heisenberg chain at two
  sizes and both MPO routes, the fZ2 Hubbard chain at two `U/t`, and the SU(2) chain.
  **Bit-exact reproducibility is not a property this stack has**: the *unmodified*
  baseline, run twice with `VECLIB_MAXIMUM_THREADS=1` and `PYTHONHASHSEED=0`, differs from
  itself by up to 9.4e-14 on those same numbers (threaded LAPACK reductions). Against that
  floor, branch-versus-baseline differs by at most **6.7e-14** — *smaller* than the
  baseline's own run-to-run spread, i.e. the two paths are numerically indistinguishable
  by any instrument this machine has. The suite is the other half of the statement: every
  existing test passes **unedited**, including
  `tests/network/test_hygiene.py::test_every_two_operand_einsum_is_a_composition`, whose
  reachability assertion is why the density matrix is built from `compose`/`adjoint` and
  spells no new `einsum` at all.

  **The convergence gate, and it closes with the table rather than with a claim.** #232's
  Stage C asks for a measured case where the wavefunction noise plateaus and the
  perturbative noise does not, *or the honest finding that it does not* — in which case
  the stage still ships, because block2's default is the design being adopted and not a
  performance patch. It is the second outcome. `benchmarks/bench_noise.py` runs one fixed
  schedule under five noise settings with nothing else varying — same seed, same starting
  MPS, same `chi`, same sweep count, same operator, noise on for the first five sweeps and
  off for the last three so every column ends cooled and the last row is comparable.

  The lattice contrast first, U(1) Heisenberg N=20 from the `D=1` Neel product seed,
  `chi=24` (energies per sweep):

  | sweep | none | wfn 1e-4 | wfn 1e-5 | pert 1e-4 | pert 1e-5 |
  |---|---|---|---|---|---|
  | 1 | -8.605141831 | -8.605140548 | -8.605141724 | **-8.655765651** | **-8.650923250** |
  | 2 | -8.681922862 | -8.681915084 | -8.681922679 | -8.682456851 | -8.682443457 |
  | 3 | -8.682473217 | -8.682473049 | -8.682473215 | -8.682473193 | -8.682473209 |
  | 4 | -8.682473226 | -8.682473070 | -8.682473225 | -8.682473193 | -8.682473209 |
  | 5 | -8.682473226 | -8.682473070 | -8.682473224 | -8.682473193 | -8.682473209 |
  | 6 | -8.682473226 | -8.682473226 | -8.682473226 | -8.682473226 | -8.682473226 |
  | 7 | -8.682473226 | -8.682473226 | -8.682473226 | -8.682473226 | -8.682473226 |
  | 8 | -8.682473226 | -8.682473226 | -8.682473226 | -8.682473226 | -8.682473226 |
  | wall s | 7.3 | 6.4 | 5.0 | 7.6 | 7.3 |

  **Nothing plateaus, so nothing is rescued.** What the perturbative columns do show is a
  real head start: after *one* sweep they are 5.1e-2 and 4.6e-2 below every other column,
  which is the aimed enrichment doing exactly what it is for on a seed whose bonds are
  `D=1` — and by sweep 3 the head start is spent, because at `chi=24` this model is not
  bond-limited and the plain sweep gets there on its own. The wavefunction columns are
  *slower* than no noise at all here, which is the "not very effective" block2's own docs
  say of the cheap end, visible. `wfn 1e-4` is still 1.6e-7 off at sweep 5 and only the
  cooled tail closes it.

  N2 CAS 6-31G at K=16 — 32 spin-orbital fZ2 sites, the `cutoff=None` operator, `chi=24`,
  the same schedule:

  | sweep | none | wfn 1e-4 | wfn 1e-5 | pert 1e-4 | pert 1e-5 |
  |---|---|---|---|---|---|
  | 1 | -24.349864427 | -24.349543127 | -24.349833679 | -24.425951531 | -24.422258663 |
  | 2 | -27.873007896 | -27.872714128 | -27.872972186 | -28.467810250 | -28.649975575 |
  | 3 | -29.496790511 | -29.502600631 | -29.497024609 | -29.744036677 | -29.745973189 |
  | 4 | -29.731075981 | -29.734889316 | -29.734834377 | -29.985652872 | -30.019253895 |
  | 5 | -30.023828869 | -30.031992607 | -30.029534633 | -30.142653652 | -30.290319415 |
  | 6 | -30.373745070 | -30.347110623 | -30.372674421 | -30.220597603 | -30.736297838 |
  | 7 | -30.690928175 | -30.666268136 | -30.715663218 | -30.254880595 | -30.986842395 |
  | 8 | -30.917777919 | -30.855357500 | -30.901566826 | -30.312247486 | -31.042540010 |
  | wall s | 1351 | 924 | 929 | 1468 | 1462 |

  **Reading it.** Eight sweeps at `chi=24` do not converge this Hamiltonian — the noiseless
  column is still falling at sweep 8 (-30.374, -30.691, -30.918 over the last three) — so
  what the table compares is **descent rate on a hard ab initio problem, not two converged
  answers**, and no column has plateaued for another to rescue. Within that, the ordering
  at sweep 8 is:

  | setting | sweep 8 | against no noise |
  |---|---|---|
  | `pert 1e-5` | -31.042540 | **0.125 lower** |
  | `none` | -30.917778 | — |
  | `wfn 1e-5` | -30.901567 | 0.016 higher |
  | `wfn 1e-4` | -30.855358 | 0.062 higher |
  | `pert 1e-4` | -30.312247 | 0.606 higher |

  Three things in that column, and only the first is a point in the mixer's favour.
  **`pert 1e-5` leads at every one of the eight sweeps** and ends 0.125 Ha below the
  noiseless run — the aimed enrichment doing what it is for, on the workload it was
  designed for. **Neither wavefunction setting ever gets ahead of no noise at all**, at
  either strength and at any sweep: 1e-5 tracks the noiseless column to three decimals and
  1e-4 is visibly behind it. That is block2's own "not very effective" for the cheap end,
  measured here rather than quoted. And **`pert 1e-4` is worse than doing nothing** — it
  buys the best first sweep of the whole table and then spends six sweeps climbing back,
  ending 0.606 Ha above the noiseless run.

  So the gate closes the second way it is allowed to: **the wavefunction noise is not
  observed to plateau where the perturbative noise does not, and the claim "improves
  convergence" is not made.** The mechanism ships because it is the decimation block2
  defaults to and this engine now has design parity with it, which was the reason to build
  it. What the table does establish is narrower and worth keeping: at the bottom of
  block2's documented range the aimed perturbation is ahead of every other setting at every
  sweep of a non-converged ab initio descent, and at ten times that strength it is behind
  all of them. A mixer strong enough to reshape a bond is strong enough to reshape it
  wrongly; 1e-4..1e-5 is a *range* because its top is not always the right end, and that is
  now measured here instead of inherited.

  Both tables were taken on the same machine and the same commit; the N2 one is stitched
  from three invocations, because a first attempt was killed after three columns and had
  printed only their timings — which is why `bench_noise.py` grew `--settings` to resume a
  column and `--out` to persist one the moment it finishes. Its wall row is therefore
  **not** a cost comparison: the two `pert` columns ran concurrently with the `none` one
  and the two `wfn` columns ran alone, so the 1468/1462 against 924/929 is contention, not
  the mixer. The cost statement this stage makes is the structural one instead: a mixed sweep pays one
  extra family-resolved application of the operator per bond, against the `ncv` the Krylov
  loop already pays, plus one `eigh` of a `rho` the SVD split never forms.

  **The oracles, with the mixer on** (`tests/integration/test_dmrg_noise.py`, a perturbative
  ramp of three mixed sweeps then a cooled tail): the dense `S^z_tot = 0` restriction of
  the N=12 Heisenberg chain — which is also the MPSKit.jl number — from a random seed and
  from the `D=1` Neel product seed; the MPSKit.jl Hubbard fixture across `U/t`; and
  `test_dmrg_prepared.py`'s six term-family models, mixed against unmixed. All land on the
  same energy. That last set is the load-bearing one: the perturbation is *built from*
  those families, so a family folded in with a wrong coefficient could not leave both runs
  on the same number — the SU(2) and fZ2 rows in particular are what says the fold touches
  no recoupling and no fermionic sign.

  **The one place the mixer does move an energy, and why it is not a counterexample.** The
  Ly=6 cylinder at `chi=32` converges to a *discarded weight of 1e-2* — a percent of the
  state thrown away every sweep — and there the mixed run lands 3.1e-4 above the unmixed
  one, which is 3 % of its own truncation error. Two runs that each discarded a percent of
  themselves differ because they were truncated, not because one was mixed; noise is not
  variational and never claimed to be. At `chi=64` the same model has both runs at a 1e-16
  discarded weight and they agree to **8.9e-16**, and at `chi=128` to 6.2e-15. The test
  therefore asserts the discarded weight is machine-zero *before* comparing energies, so it
  cannot silently decay into a comparison of two truncated states.

  The wavefunction noise does *not* move that same `chi=32` cylinder (6.7e-9 from the
  unmixed run), and the contrast is the mechanism rather than a defect: at equal `noise`
  the family-resolved perturbation is aimed at directions the operator couples to, so on a
  bond where a percent of the state is being discarded every sweep it genuinely reshapes
  which states survive, while a random tensor at the same relative strength is spread over
  every structurally allowed direction and mostly gets truncated away again. Aimed
  enrichment is what the mechanism is *for*; that it therefore has a visible effect exactly
  where truncation is severe is the expected sign, not a warning one.

  **What was deliberately not built.** No `NoiseTypes` lattice — `Reduced`, `Collected`,
  `LowMem` and `MidMem` are memory-layout engineering around block2's `SparseMatrixGroup`,
  not design, and this engine has no such object to lay out. No `Unscaled` variant: it
  exists so a caller can bypass the per-vector normalization, and a knob whose only effect
  is to make `noise` dimension-ful is a knob that makes the documented 1e-4..1e-5 range a
  lie. No White single-site `alpha` mixer: block2's one-site mode is the consumer that
  needs it and one-site sweeps stay out of scope until there is a measured cost to trade.
  The families are not compiled and the density matrix is not cached — the mixer runs once
  per bond visit, against a matvec the Krylov loop runs `ncv` times.

  **Follow-ups this stage leaves.** The `heff2_families` call doubles the bond's matvec
  work on a mixed sweep, and it recomputes what `_apply2` will compute again inside
  `lanczos`; a shared partial-application cache is the obvious saving and is not worth its
  invalidation discipline until a profile asks. The single-vector fallback for an MPO with
  no edge description is honest but weak; closing it needs symbols the compatibility entry
  does not have (#141), which is the same wall `heff2`'s docstring records.

- **M61 Stage D** — shipped: `Env` accepts a **bra that is not the ket**, and the sweep
  can hold its state orthogonal to already-converged ones (#232 Stage D, absorbing #216
  and the engine half of #213). `Env` gains one keyword-only `bra=` and one method
  `project2`; `MPO` gains `identity`; `lanczos`, `sweep_` and `dmrg_` each gain one
  keyword-only argument whose default preserves today's behaviour. block2 is GPL-3.0 and
  this repository is Apache-2.0: everything here is description in original words with
  `file:line` citations, and no code, comment or docstring crossed.

  **What block2's `ext_mes` actually contract: the identity, not `H`.** The question is
  worth asking because `two_dot_eigs_and_perturb` builds each `ortho_bra` through a
  moving environment and calls `multiply` on it (`sweep_algorithm.hpp`:1195-1206), which
  reads like an energy. It is not: the driver constructs those environments with
  `impo = self.get_identity_mpo()` (`pyblock2/driver/core.py`:4817-4830), one per
  converged state, between the sweeping ket and that state. So the per-bond vector is an
  **overlap** — the converged state's two-site reduced form in the sweeping state's
  environment gauge — and that is what is adopted here.

  **Penalty versus hard projection: block2 ships both, and its state-specific default is
  hard.** With `ors` and an empty `projection_weights` its Davidson projects every basis
  vector by `1 - |v><v|`, after Gram-Schmidt-ing the `ors` against each other
  (`iterative_matrix_functions.hpp`:1198-1200, :1219-1237). With a non-empty
  `projection_weights` it instead adds `w_k |v_k><v_k|` to each `sigma` (:1201-1204,
  :1250-1253), i.e. solves `H + sum_k w_k |v_k><v_k|` — the **level-shift** approach its
  own documentation names as such and warns about, since a weight below the gap reports
  unphysical eigenvalues `E_k + w_k` (`docs/source/user/keywords.rst`, `proj_mps_tags`
  and `proj_weights`). Hard projection is what `statespecific` alone runs, it has no
  parameter to get wrong, and its Ritz value is the projected operator's own, so nothing
  has to be subtracted back off. tenet adopts hard projection and ships **no weight
  argument**: a knob whose failure mode is a plausible wrong energy is not worth the
  surface.

  **The gauge the two-state contraction requires: none of the environments, one of the
  reading.** `update_`'s folds are exact for any pair of chains — `MPS._braket`'s
  docstring already recorded the same fact one level down, the transfer tensor holds one
  index from each chain — so `Env(psi, h, bra=phi).measure()` **is** `<phi|H|psi>` for two
  states in any gauge and at any norm, and that is the engine fact #213's public API
  stands on. What does need a gauge is the *use* of `project2`'s output as a projection
  direction, because "project the sweeping state's two-site variational space against the
  converged state" is a statement about an orthonormal basis: it holds exactly when the
  **sweeping** state is mixed-canonical at the bond, which `sweep_` maintains anyway. The
  converged states need nothing and are therefore **held fixed**, their per-bond reduced
  forms recomputed at each bond, rather than canonicalized and propagated the way block2
  moves its `ext_mpss` (:893-917) — a gauge transformation on any of their bonds cancels
  between the two environments and the two-site tensor, so the propagation would buy an
  invariance that is already free.

  **`heff2` refuses on a two-state `Env`, and the refusal is the honest half of the
  design.** The prepared matvec's one-sided `caf`/`abf` terms read the `IdL`/`IdR`
  environment channels as gauge identities, which is true of a canonical chain against
  *itself* and false of a mixed transfer; block2 does not iterate on its `ext_mes`
  either. So a two-state object supports environment building, `measure` and `project2`,
  and `heff2`/`heff2_families` raise with a message naming what the object is for.
  `project2` shares `heff2`'s **compatibility entry** verbatim (factored out as
  `_heff2_full`) — the one path in the class that reads no channel as a gauge identity,
  hence the one that survives `bra is not psi` — so this stage adds no `tenet.einsum`
  call site to `network/` at all and
  `tests/network/test_hygiene.py::test_every_two_operand_einsum_is_a_composition` passes
  unedited, its reachability assertion included.

  **The projection is spelled with `MPO.identity`, not a second environment class.** An
  overlap *is* an environment; giving `Env` the identity operator reuses every cap
  direction the Jordan-Wigner oracle already pinned, at the price of a `D=1` unit leg
  nobody contracts. `MPO.identity` is `tenet.identity` on `(unit, phys)` transposed into
  the MPO's `(wl IN, p OUT, p IN, wr OUT)` axis order — no `einsum`, so no composition
  rule to state — and it carries no `EdgeTable`, so it takes the full-contraction path by
  construction.

  **The finding that cost this stage its afternoon: `tenet.inner` is not the pairing
  `tenet.norm` induces, on a graded provider.** `inner`'s own docstring promises
  `inner(a, a) == norm(a)**2`. For a rank-4 fZ2 tensor on legs
  `(W OUT, V OUT, V OUT, W IN)` with `W` carrying both parities it does not:
  `inner(t, t) = 5.847465` against `norm(t)**2 = 21.149209`, which is also the plain sum
  of squares of `t.to_dense()`. The cause is structural — `inner` contracts every axis
  but the first and closes axis 0 with `full_trace`, which puts a twist on that wire — so
  it bites exactly when the first axis carries an odd sector, i.e. on every two-site
  tensor of a fermionic sweep away from the boundary. It has never been caught because
  the one-state solve is **self-consistent in the twisted form**: `lanczos` builds its
  tridiagonal from `inner`, `Env.heff2` returns its image in the same form, `H_eff`
  commutes with the twist, and the Ritz values come out exact — the fZ2 ground-state
  energies in this suite are exact to 1e-15 and stay so. A **projector** is not
  self-consistent that way: it has to be built in the pairing the state is normalized in.
  `dmrg._dot` therefore spells that pairing directly —
  `full_trace(compose(adjoint(m), m'))` on the bent `(l, p | q, r)` partition, the same
  two primitives `_rho` uses — and it agrees with `inner` on every ungraded provider.
  Built on `inner` instead, the fZ2 excited state converges to a **non-eigenvector**: at
  N=6 it stalls at -2.978 against the exact -3.049, with a residual `||H psi - E psi||`
  of 0.31 and a reported energy equal to its own Rayleigh quotient — a fixed point of the
  sweep that is not a state. With `_dot` it is exact. The core defect is deliberately
  **not** fixed here: it lives in `tenet.ops.contraction.inner`, outside this stage's
  scope, and correcting it would move `lanczos`' numbers on every graded provider. It is
  left as a follow-up with this reproduction attached.

  **The excited-state oracles, every number against exact diagonalization computed in the
  test rather than a recorded literal** (`tests/integration/test_dmrg_excited.py`):

  | model | quantity | exact | DMRG | error |
  |---|---|---|---|---|
  | U(1) Heisenberg N=8, `S^z=0` | ground | -3.3749325986878871 | -3.3749325986878906 | 3.6e-15 |
  | U(1) Heisenberg N=8, `S^z=0` | first excited | -2.9822404877628852 | -2.9822404877628852 | **0.0** |
  | fZ2 free fermions N=12, even | ground | -7.2962298105587546 | -7.2962298105587484 | 6.2e-15 |
  | fZ2 free fermions N=12, even | first excited | -6.8140830895374620 | -6.8140830895374433 | 1.9e-14 |
  | SU(2) Heisenberg N=8, singlets | ground | -3.3749325986878871 | -3.3749325986877823 | 1.0e-13 |
  | SU(2) Heisenberg N=8, singlets | second singlet | -2.3338038644955690 | -2.3338038644955423 | 2.7e-14 |

  The last digits move run to run at the 1e-14 level, for M61 Stage C's reason (threaded
  LAPACK reductions; the unmodified baseline differs from itself by up to 9.4e-14), so the
  asserted tolerances are 1e-10 and 1e-9 rather than these figures.

  `<psi2|psi1>` after convergence, measured with this stage's own machinery
  (`Env(psi2, MPO.identity(...), bra=psi1).measure()`): **1.1e-16** on U(1), **3.6e-16**
  on fZ2 at N=12, **5.6e-17** on SU(2).

  The SU(2) oracle needs the singlet spectrum, and it is obtained without an `S^2`
  matrix: every multiplet of spin `S >= 1` contributes one copy of its energy to both the
  `S^z = 0` and `S^z = 1` blocks, so the multiset difference of the two dense spectra is
  exactly the singlets — which is the sector an SU(2) MPS with a trivial boundary leg
  lives in.

  **The degeneracy check is a cross-sector identity, not a tolerance.** The N=8 open
  chain's first excited state is a triplet, so its energy is *both* the second eigenvalue
  of the `S^z_tot = 0` block — which orthogonality against the ground state reaches — and
  the first of the `S^z_tot = 1` block, which a charged `D=1` boundary leg reaches with no
  projection at all. The ED pair agrees to 3.6e-15 on -2.98224048776288, and the two DMRG
  runs land on it from their two different mechanisms. A converged state whose boundary
  legs put it in *another* sector is dropped from the projection rather than projected
  with — it is orthogonal by the symmetry before the sweep does anything — and the run is
  then byte-identical to the plain ground-state run, which the suite asserts at `abs=0.0`.

  **The standing limitation, asserted rather than described.** Two chains whose bond
  spaces share no sector at some cut have an identically zero transfer there, and
  `tenet.compose` has no block to take its backend reference from (`ops/map.py`:167), so
  the two-state `Env` raises instead of returning the structural zero. It does not reach
  the excited-state workflow, whose states are seeded on one set of bond spaces and swept
  together, and the sector skip above keeps the one case that would reach it out of the
  contraction.

  **The name is TenPy's.** YASTN spells the same argument `project` and TenPy
  `orthogonal_to`; the second is taken because `project` names *a* mechanism and this
  argument is one of two that implement it — block2 ships both — while `orthogonal_to`
  names the result, which is the same under either.

- **M62** — shipped: `tenet.inner` is the Frobenius pairing on every provider, and
  `network/dmrg.py::_dot` is deleted (#236, the defect M61 Stage D found and filed).

  **The pairing is not a diagram.** The old body drew it —
  `full_trace(einsum("L{rest},l{rest}->lL", adjoint(a), b))` — contracting every axis but
  the first and closing axis 0 with the categorical trace. Contracting the rest *first*
  makes the still-open axis-0 lines cross the contracted ones, and on a graded provider
  each crossing of two odd lines pays `-1`. An invariant scalar has
  (axis-0 sector) = (sector of the rest), so exactly the odd-sector blocks entered the
  sum with a flipped sign: `inner(t, t) != norm(t)**2` whenever axis 0 carried an odd
  sector, which is every two-site tensor of a fermionic sweep away from the boundary.

  **The fix is `norm`'s body with the square replaced by a conjugated pair**:
  `<a|b> = Σ_τ qdim(c_τ) · <A_τ, B_τ>`, coefficient space, per fusion-tree block, no
  diagram and therefore no crossing to pay for. That is also TensorKit's spelling —
  `src/tensors/vectorinterface.jl` computes `Σ_c dim(c) · inner(block(t1, c), block(t2, c))`
  in both fusion-style branches — i.e. the pairing MPSKit's Krylov machinery runs on.
  `inner(a, a) == norm(a)**2` now holds *identically* rather than numerically, since the
  two functions iterate the same blocks with the same weight, and the dense
  `Σ conj(a)·b` over `to_dense` is the acceptance oracle. Two surface consequences: the
  structure precondition is now checked here (`_check_same_structure`, the same refusal
  `zip_blocks` and `add` raise) rather than inherited from `einsum`, and `PivotalData` is
  no longer required — `full_trace` is off the path, `QuantumDimensionData` alone carries
  the weight. The rank-26 cap the `string.ascii_lowercase` labelling imposed is gone with
  the einsum.

  **The reproduction, base against fix** — the rank-4 tensor on `(W OUT, V OUT, V OUT, W IN)`
  with `W` carrying both parities, `inner(t, t)` against the dense sum of squares:

  | provider | `inner` before | `inner` after | dense `Σ conj·` |
  |---|---|---|---|
  | fZ2, `W` = {even 2, odd 2} | 19.070505659 | 39.603850021 | 39.603850021 |
  | fZ2 Hubbard `d=4`, `W` = {3, 3} | 41.796480230 | 291.241165763 | 291.241165763 |
  | U(1) | 29.337177840 | 29.337177840 | 29.337177840 |
  | SU(2) | 66.531665748 | 66.531665748 | 66.531665748 |

  The ungraded rows are the control: nothing crosses with a sign there, and the SU(2) row
  also pins that the `qdim` weight survived the rewrite.

  **What moved in the solvers: nothing beyond reproducibility noise, measured rather than
  asserted.** The one-state solve was self-consistent in the twisted form — `lanczos`
  built its tridiagonal from the same pairing its normalization used — so the metric
  entered symmetrically and the Ritz values were already exact. Re-running the graded
  oracles on the base commit and on the fix:

  | run | base | fix | delta |
  |---|---|---|---|
  | U(1) Heisenberg N=8 ground | -3.3749325986878906 | -3.3749325986878920 | 1.3e-15 |
  | U(1) Heisenberg N=8 first excited | -2.9822404877628850 | -2.9822404877628850 | 0.0 |
  | fZ2 free fermions N=12 ground | -7.2962298105587484 | -7.2962298105587528 | 4.4e-15 |
  | fZ2 free fermions N=12 first excited | -6.8140830895374433 | -6.8140830895374575 | 1.4e-14 |
  | SU(2) Heisenberg N=8 ground | -3.3749325986877710 | -3.3749325986877743 | 3.3e-15 |
  | SU(2) Heisenberg N=8 second singlet | -2.3338038644955366 | -2.3338038644955390 | 2.4e-15 |
  | Hubbard N=4 `U/t=0` | -4.4721359549995790 | -4.4721359549995805 | 1.5e-15 |
  | Hubbard N=4 `U/t=4` | -2.6249422715108660 | -2.6249422715108650 | 1.0e-15 |
  | Hubbard N=6 `U/t=4` | -4.4220711477587550 | -4.4220711477587580 | 3.0e-15 |

  Every converged energy above is at or below the 9.4e-14 the unmodified baseline differs
  from *itself* by (M61 Stage C's threaded-LAPACK measurement), and the excited-state
  suite — the one that had to change behaviour — passes through plain `inner`, which is
  the point.

  **What did move is the sweep count on fZ2 Hubbard, and it is a convergence rate rather
  than an answer.** `H_eff` is Hermitian in the fixed pairing and was *not* Hermitian in
  the twisted one — measured directly at a bulk bond of the N=4 Hubbard chain,
  `<b, H a>` against `<H b, a>`: `-18.06754764272969` against `-18.06754764272969` fixed,
  `3.9213134766` against `3.6686612016` twisted. The base was therefore running Lanczos on
  an operator non-symmetric in its own pairing, and its three-term recurrence picked a
  different (here, luckier) direction. With the correct pairing the N=4 chain still reaches
  ED exactly, in more sweeps: `U/t=2` needs 72 sweeps against 13, `U/t=4` under a
  perturbative mixer needs 20 against 12, and the errors at convergence are -1.3e-15 and
  -1.6e-14. Two tests whose sweep budgets were set to the old rate therefore stop short:
  `tests/network/test_hubbard.py::test_hubbard_dmrg_matches_ed_across_the_u_sweep`
  (`max_sweeps=40`, off by 1.3e-9 at `U/t=2`) and
  `tests/network/test_density_matrix.py::test_the_hubbard_ground_state_is_unchanged_with_the_mixer_on`
  (`max_sweeps=6`, off by 2.5e-4 at `U/t=4`). Neither number was pinned — both oracles are
  exact diagonalization computed in the test — and neither test is edited here; the rate
  is the finding.

  **`_dot` is deleted rather than kept as a synonym.** M61 Stage D's workaround spelled
  the norm-inducing pairing directly through `compose`/`full_trace` so the projector
  `1 - |v><v|` was built in the pairing the state is normalized in. With `inner` fixed the
  two are one pairing, and two spellings of one pairing is exactly the split this
  repository removes — `_orthonormal` and `_project_out` now call `tenet.inner`, as
  `lanczos` already did. `examples/toy_codes/dmrg.py`'s hand-written `inner` is rewritten
  the same way: it is U(1)-only, where the twist is invisible, but its docstring claimed to
  be `tenet.inner` verbatim and a reader copying it for a fermionic model would have copied
  the sign error.

  **The other call sites, and why none of them was wrong.** `lanczos`'s `alpha` is the one
  place the twist could reach a graded provider, and it reached it symmetrically —
  self-consistent, as above. `Env.project2`'s docstring pairs its output with a bra-side
  two-site tensor and its doctest is U(1). `tests/ops/test_embed.py` carries its own
  `qdim_inner` helper deliberately: it pairs tensors on *different* structures, which
  `inner`'s precondition forbids.

- **M42** — shipped: `SymmetricTensor.astype(dtype)` and `to_backend(backend, dtype=None)`
  (#207). Blockwise `ar.do("astype", b, dtype)` — the call `ops/embed.py`'s padding
  accumulator already made privately — returning a new tensor, so the frozen-dataclass
  and pytree contracts are untouched. `to_backend`'s single-argument form is byte-identical
  in behaviour; the keyword defaults to today's.

  **`dtype` runs after the move, not as part of it.** Casting first and moving second lets
  the backend re-decide: a NumPy complex128 tensor handed to `jnp.array` is still subject
  to JAX's dtype policy on arrival, so the caller's request would be racing the backend's.
  Moving first and casting second makes the caller's request the last statement, which is
  the whole point of the keyword.

  **What that fixes and what it cannot.** It overrides a backend's *choice* — a real tensor
  moved to JAX and asked for complex arrives complex. It does not override a backend's
  *refusal*, and the issue's third acceptance criterion asked for exactly that:
  `to_backend("jax", dtype=np.float64)` keeping float64 with `jax_enable_x64` unset is not
  achievable, because in that mode JAX has no float64 for `astype` to produce. Measured
  in a fresh interpreter with x64 off:

  | call | result |
  |---|---|
  | `ar.do("array", b, like="jax")` | float32 |
  | `ar.do("astype", jax_block, np.float64)` | float32 (with JAX's truncation warning) |
  | `ar.do("astype", jax_block, np.complex128)` | complex64 |

  `jax_enable_x64` is process-global and, once on, irreversible in practice — which is why
  `tests/conftest.py` sets it session-wide and why the unset state is only reachable from a
  subprocess. `test_to_backend_jax_dtype_cannot_defeat_a_disabled_x64` runs that subprocess
  and pins the table above, so the distinction the `Notes` now draw between a backend's
  choice and its refusal is a test rather than a claim.

  **The dtype spelling is normalized, because otherwise the method meant two things.**
  autoray's torch `astype` routes its argument through `to_backend_dtype`, which wants a
  dtype *name*: `t.astype(np.complex128)` — the spelling this issue's own criteria use,
  and the one that works on NumPy and JAX — raised `TypeError: to() received an invalid
  combination of arguments` on torch blocks. `astype` therefore normalizes anything NumPy
  recognizes to `np.dtype(dtype).name` and passes a backend-native dtype object through
  untouched, so one spelling means one thing on all three backends. The torch suite takes
  both spellings, which is what would have caught this.

  **`astype` refuses a block-free tensor** with `_first_block`'s existing message rather
  than returning an empty tensor of nominal dtype: `dtype` on the result would be as
  undefined as it already is on the input, and inventing a second message for the same
  condition is what the criterion ruled out. `to_backend` without `dtype` still passes a
  block-free tensor through untouched, since the move is well defined where the cast is not.

- **M43** — shipped: `SymmetricTensor.from_blocks(legs, mapping)` and
  `tensor.with_blocks(mapping)`, the keyed counterparts of `items()` (#208). Reading a
  tensor's blocks was keyed and writing them was positional-and-total; both halves now
  take a `FusionBlockKey -> array` mapping, and `from_legs`' sequence form is untouched.

  **The spelling was chosen by writing the workarounds in each candidate first.** The
  issue named three: the SU(3) cup (`tests/symmetry/test_sun.py`), the SU(2) cup
  (`tests/symmetry/test_su2_dual.py`) — both `zeros` → `ones_like` → a hand-written
  `assert len(blocks) == 1 and blocks[0].shape == (1, 1)` — and `dualize_axis`
  (`tests/symmetry/test_su2_dual.py`:155).

  | candidate | the SU(3) cup, written out | verdict |
  |---|---|---|
  | `from_blocks(legs, mapping)` | `structure = TensorStructure(legs)`; `(key,) = structure.block_order`; `from_blocks(legs, {key: np.ones(structure.block_shape(key))})` | taken |
  | `from_legs` also accepting a `Mapping` | the same three lines, spelled `from_legs` | refused |
  | `zeros(...)` + a per-key functional update | `t = zeros(legs)`; `(key,) = t.structure.block_order`; `t.with_blocks({key: np.ones(...)})` | refused as *the* answer, kept as the second half |

  All three delete the assertion, so the assertion is not what separates them. What
  separates them is this. The `Mapping` overload cannot be built at all without changing
  `from_legs`' signature and docstring, which the issue's own criteria freeze — and the
  reason those criteria are right is #120's one-name-one-thing rule: a constructor that
  dispatches on `isinstance(blocks, Mapping)` gives one name two argument grammars with
  two different rules for what a missing block means. The `zeros`-first form keeps
  exactly the step the issue set out to remove — building a zero tensor to learn the
  layout — and it decides dtype and backend from `zeros`' NumPy-float64 default rather
  than from the blocks the caller actually has. `from_blocks` needs neither: the layout
  comes from `TensorStructure(legs)`, which is public and now what the guide points at,
  and the dtype and backend come from the supplied blocks.

  **`with_blocks` is kept anyway**, because replacement is a genuinely separate need —
  changing one block previously meant reproducing all of them in order — and because it
  and `from_blocks` share their key validation, so they are one decision. `dualize_axis`
  is **not** rewritten: it is not the same workaround. It carries a tensor's blocks onto a
  *different* structure, where the correspondence is positional by construction; the keyed
  spelling would be `dict(zip(block_order, t.blocks))`, i.e. the positional assumption
  written out at greater length. It is `from_legs`' sequence form working as intended.

  **An absent key is zero, not an error.** Both readings are defensible on convenience
  grounds, and convenience is not what settles it: strictness would earn its cost if it
  caught the mistake the deleted assertions were guarding against, and it does not. A
  mistyped key is an *unknown* key, not a missing one, so it raises under either rule.
  Requiring every key would only tax the case the constructor exists for.

  **An empty mapping raises** rather than defaulting to NumPy float64, because the zero
  fill has nothing to take a dtype and backend from and guessing would put a silent
  backend choice inside a constructor. The message names `zeros`, which is that tensor.

  **The refusals borrow rather than duplicate.** A foreign key raises `KeyError` — the
  class `index_of` and `block` already raise — naming the first few legal keys and
  pointing at `block_order` for the rest, because a `FusionBlockKey` reprs to roughly 200
  characters and a structure of any size cannot have its whole `block_order` in an
  exception message. A wrong-shaped block is left to `__post_init__`, whose existing
  message already names both the expected shape and the key.
- **M36** — shipped: `tenet.models`, the standard local operator sets, so a Hamiltonian
  stops starting with a hand-written numpy matrix (#198). The layer ships **sites**, not
  models: `spin_half(U1|SU2)`, `spinless_fermion()`, `spinful_fermion()` and
  `hard_core_boson(U1|Trivial)`, each returning a `Site` — a physical `GradedSpace`, an
  `ops` mapping of name to `local_op`-built term operator, and the dense `matrices`
  behind them. Nothing in the package calls `SymmetricTensor.from_dense`; every operator
  goes through `local_op` and therefore arrives with its refusals attached, which a test
  asserts by AST rather than by review.

  **Why this does not reopen the operator-zoo rule.** #112 and #133 put the zoo in the
  caller and `local_op`'s docstring says the matrices are physics and stay there. That is
  a statement about `tenet.network` and the core deciding what a caller's operators
  *mean* — not a prohibition on a separate, optional layer offering the standard ones,
  which is what every reference does. It survives verbatim because the dependency runs
  one way, and `tests/network/test_hygiene.py::test_no_module_imports_tenet_models` is
  what keeps it there. `tenet.models` is also not re-exported from `tenet.__all__`: it is
  imported explicitly, which is the same statement said in the import graph.

  **The naming survey, read rather than recalled.** tenpy's `Site` (`networks/site.py`)
  is a class with `get_op(name)`, `add_op`, `opnames` and `state_labels`, and `get_op`
  **splits its name on whitespace** and contracts the factors — `"Sp Sm"` is a legal
  operator name. Its spinful site names are `Cu, Cdu, Cd, Cdd, Nu, Nd, Ntot, NuNd`,
  where `Cd` is *spin-down annihilation* and creation is `Cdu`/`Cdd`; that collision is
  precisely the name that needs a paragraph, and #120/#185's bar rejects it. YASTN's
  `yastn/operators/` is classes with methods — `Spin12(sym='U1').sp()`, `sm()`, `sz()`,
  `I()`; `SpinlessFermions().c()`, `cp()`, `n()`; `SpinfulFermions().c(spin='u')` — with
  the spin as a *keyword*, and `to_dict()` flattening it back to `'cu'`, `'cpu'`.
  MPSKitModels is free functions with the symmetry as a positional type,
  `S_z(U1Irrep; spin=1//2)`, `e_number_updown(T, particle_symmetry, spin_symmetry)`.
  What is taken: **free functions** returning one record (MPSKitModels' shape, tenet's
  own `local_op`/`from_terms` style), a **mapping** rather than attributes (it is
  literally `from_arrays`'s `ops` argument), and the operator names spelled as the
  field's own symbols — `Sz`, `S+`, `S-`, `S.S`, `c`, `c+`, `n`, `c_up`, `c+_up`,
  `n_up n_dn`. The whitespace product `"n_up n_dn"` is not an invention: it is tenpy's
  `get_op` convention and `mps.py::_expr_names` already parses block expressions the
  same way, so the *same spelling* names the pre-multiplied on-site operator in
  `from_terms` and a two-name coincident block in `from_arrays`.

  **The SU(2) case, answered concretely: the site returns `{S.S}` and no irreducible
  tensor operator.** `S+` is absent because the API cannot hold it, not by preference —
  `local_op`'s charge-leg form reshapes a `(d, d)` array to `(d, d, 1)`, so the emitted
  sector must have dense dimension 1, and the only leg a spin-1 tensor operator could
  emit onto is the `j=1` multiplet, whose dense dimension is 3. `local_op(sz, phys=phys,
  charge=SU2Sector(2))` therefore raises on the shape, and there is nothing to hand back.
  What exists is M13b's invariant *k*-site form, and `S.S` is one whole Heisenberg bond
  term whose coupling lives inside its own blocks. This is also where MPSKitModels lands:
  its single SU(2) method in `spinoperators.jl` is `S_exchange` (alias `SS`), built from
  two three-leg tensors through a spin-1 auxiliary space, while `S_z`/`S_plus` under
  `SU2Irrep` simply have no method. tenpy and YASTN do not reach the question at all —
  both are abelian-only (`yastn/sym/` has no non-abelian module; `site.py` never mentions
  SU(2)). The same `S.S` matrix is invariant under U(1) too, where it sits in `matrices`
  rather than `ops`, because `from_arrays` gives one site index per name and refuses a
  rank-4 entry: `ops` is rank-3 exactly when the grading is abelian, and the SU(2) site's
  `ops` is a `from_terms` table and nothing else.

  **The two end-to-end call shapes** (#197's `ops` argument is what this populates):
  U(1) Heisenberg is `MPO.from_arrays(n, spin_half().ops, [("Sz Sz", bond, ...),
  ("S+ S-", bond, ...), ("S- S+", bond, ...)])`; spinful Hubbard is the same call on
  `spinful_fermion().ops` with `("c+_up c_up", fwd|bwd, -t)` per flavour and
  `("n_up n_dn", [(m, m), ...], U)` — the coincident pair the merge multiplies. Both are
  tested against dense Jordan-Wigner oracles, and the `from_terms` route over the same
  `ops` is tested to agree. There is **no `JW` operator** in any fermionic site and no
  place for one: the string is the `fZ2` braiding an odd MPO bond pays crossing a
  physical line (M21/#147).

  **What is deliberately not shipped, and the argument that keeps the layer finite.** No
  lattice geometry, no model Hamiltonians (`heisenberg(L)` returning an MPO), no
  parameter sweeps. The set of standard *sites* is closed by the physics — a local
  Hilbert space and its grading — and a model zoo is closed by nothing; tenpy's own
  `models/` directory is the demonstration, and it grows with every paper. The
  Hamiltonian stays the caller's term list, which is also the only form in which #197's
  array front end and #191/#193's assembler are reachable. `Site` carries `matrices`
  alongside `ops` for the one honest gap: `expectation_1site` wants rank 2 and
  `expectation_2site` wants rank 4, neither of which is a term form, so the matrix is
  there rather than a second operator mapping.

  Measured as a diff: the three usage-lane examples that hand-wrote these operators
  (`heisenberg.py`, `su2_heisenberg.py`, `bench_dmrg.py`) lose 42 lines and gain 27.
  `examples/toy_codes/` is untouched on purpose — writing the operators out is part of
  what a toy code teaches, which is the same lane rule #183 drew.
- **M58** — shipped: `Env._compiled` leaves the byte budget, and the compile count is the
  distinct-structure-key count at every budget (#227, the defect M57 measured and filed).

  **The decision, of the three #227 offered: `_compiled` does not belong under the byte
  budget.** Its entry was `(structure key, _Prepared, callable)` and it is now
  `(structure key, callable)` — the `_Prepared` was never read from it, only weighed, and
  it is the same object `Env._prepared` holds for that bond. `common.payload` walks each
  cache independently, so those arrays were charged twice, and an eviction on a doubled
  weight threw away the one thing in the entry that is expensive to rebuild. What is left
  weighs nothing: `payload(dict(env._compiled)) == 0` on any run, asserted. So the cache is
  a plain `dict`, bounded by the bond count and by nothing else.

  **The two rejected options, and why.** Teaching `payload` to deduplicate by identity
  across caches is the general fix and it is the wrong one here: it makes a cache's weight
  depend on what the *other* caches hold, and each `Recent` being independent is M38's
  whole appeal — the policy is one number in one place precisely because no cache has to
  ask another one anything. A per-`Env` budget shared by an accumulator is the same
  coupling with a nicer name, and it would also change what M38 measured for the two
  caches that genuinely hold gibibytes, which is a re-measurement and not a defect fix.
  Neither is needed once the entry stops holding a borrowed reference: the double count
  was not incidental, it was systematic and structural, and removing the reference removes
  it at the source. `payload`'s conservative over-estimate for *genuinely* different
  objects is untouched and still correct.

  **The memory the budget bounds is unchanged, and the number is zero.** M38's
  charged-once table at K=16 (N2 CAS 6-31G, 19.19 GiB resident, 7.34 GiB at a 4 GiB budget)
  charges each array buffer to the first cache that reaches it, and `_compiled` was never
  in that column: everything it held, `_prepared` held first. Taking it out of the budget
  therefore removes nothing from the resident total — it removes an eviction pressure that
  bought nothing. The four caches that hold tensors (`EdgeTable._table`,
  `EdgeTable._embeds`, `Env._cores`, `Env._prepared`) are `Recent()` exactly as before.

  **The counting result.** M57's table read 143 compiles against 115 distinct keys on C2
  CAS cc-pVDZ at K=26 — 28 recompiles, at 181–378 ms of `jax.jit` tracing apiece (M54), so
  roughly 10 s of wasted tracing per two sweeps. The 28 were bonds whose `_compiled` entry
  the budget evicted, and the floor is now reached on every model. The regression test does
  not need C2: a `CACHE_BUDGET` of zero reproduces the eviction on a six-site chain, and
  `test_a_squeezed_budget_no_longer_recompiles_a_bond_it_evicted` asserts
  `n_compile == n_distinct_keys` at a budget of 0 and at 1 TiB alike. Before the change it
  failed at both — at zero on the count, at 1 TiB on the payload.

  **One existing test changed, and it is the one #227 named.**
  `test_the_sweep_caches_never_grow_past_the_budget` asserted `set(lengths) == {2}` over
  *five* caches, i.e. that all five evict to the two-entry floor. It now asserts that over
  the four byte-budgeted caches, and asserts separately that `_compiled` is not a `Recent`,
  holds one slot per bond at both ends of the budget, and weighs zero. That is strictly
  more than it pinned before. `test_the_compiled_cache_keeps_one_entry_per_bond_across_two_chis`
  changed one literal, `len(entry) == 3` to `== 2`, which is the entry shape and nothing
  else. `_prepare2`'s identity discipline — the guarantee a stale environment can never be
  served — is untouched, and so is the test M57 added for it.

- **M50** — shipped: the per-bond Schmidt spectrum and the entanglement entropy are
  readable **from the state** (#215). `MPS.schmidt_values`, `MPS.schmidt_sectors` and
  `MPS.entanglement_entropy`, over `network/common.spectrum_sectors` and
  `network/common.entropy`.

  **The datum existed twice and was returned nowhere.** `sweep_` writes
  `schmidt[n] = spectrum(s)` at every bond of every sweep and `dmrg_` uses that dict only
  to compute `max_dSchmidt` — how much the spectrum *moved* — and drops it; `compress_`
  computes the same SVD and returns only the discarded weight. A user asking the most
  ordinary question about a converged state had to canonize, merge every adjacent pair,
  call `svd` and re-derive the `sqrt(qdim)` weight by hand.

  **The readers are on the state, and they canonize a copy.** Both references put these on
  the container (YASTN `get_Schmidt_values`/`get_entropy`, TenPy
  `entanglement_spectrum`/`entanglement_entropy`) rather than on an algorithm's output, and
  the reason is that the answer is a property of the state. `compress_`
  (`mps.py`:417-419) established that a non-canonical gauge's values are not Schmidt values
  and must be canonized first; the difference here is that a *reader* must not re-gauge what
  it reads, so `MPS._bond_svds` runs `compress_`'s body, minus the truncation, on
  `self.copy()`. `center` is therefore never consulted and never a refusal: the cost is one
  `lq` pass, which is what a correct answer costs anyway. Three readers each pay their own
  sweep; a caller wanting two of them keeps the first result.

  **The keys are the bond's left site, `0 .. N-2`.** That is the key `sweep_`'s `schmidt`
  dict already uses, so the vocabulary is the package's own. Both references return `N + 1`
  values including the two boundary cuts of a finite open chain; those are zero by
  construction and a boundary cut has no left site to key on.

  **Nats, stated on the callable, because the references disagree** — YASTN's `get_entropy`
  is base 2, TenPy's `entanglement_entropy` natural. Natural is taken: `S = (c/6) log(x)`
  is what a central-charge fit wants, and every other logarithm here is natural.

  **The multiplet weight is the part that is not bookkeeping.** On a `GradedSpace` bond a
  sector of quantum dimension `d` holds `d` copies of each reduced value in the dense
  Schmidt spectrum, so with `p_i` the `sqrt(qdim)`-weighted value squared,

  ```
  S       = -Σ_i p_i log(p_i / d_i)
  S_alpha = log(Σ_i d_i (p_i / d_i)**alpha) / (1 - alpha)
  ```

  Reading `-Σ p log p` off the flattened spectrum instead reports **0** for an SU(2)
  two-site singlet, whose entropy is `log 2` and whose whole entanglement lives in one
  `j = 1/2` multiplet. `tests/network/test_entanglement.py` pins both halves: the naive sum
  is asserted to be zero (so a regression names itself) and the SU(2) profile is asserted
  equal to the U(1) profile of the same state, at `alpha = 1` and `alpha = 2`, at `N = 2`
  and `N = 6`. Equality against the *other grading of the same state* is what says the
  weight is right rather than merely self-consistent; a dense `numpy.linalg.svd` oracle on
  the `2**6` amplitude vector pins the absolute number at every cut.

  **The `sqrt(qdim)` weight is written once**, in `spectrum_sectors`. `spectrum` keeps its
  signature and its two callers and is now literally the flatten of it — the sorted
  concatenation — so the flat convergence diagnostic (#120, reaffirmed #185) is unchanged
  and there is no second copy of the weight to drift. `entropy` asks the provider for
  `qdim` itself, because a Renyi sum needs the multiplet *count* and that is not recoverable
  from an already-weighted value. A source-reading test pins the split.

  **The sector resolution is the read a graded bond is for**, and is TenPy's
  `entanglement_spectrum(by_charge=True)`. It is a second method rather than a `by_sector=`
  flag on the first: the return types differ, and a boolean that changes a return type is
  the kind of signature a checker cannot narrow and a reader has to run to understand.

  **`DMRG_out` carries no spectrum field, decided rather than omitted.** `out.psi` answers
  for itself, exactly and in any gauge. The sweep's dict is a *truncated* spectrum taken at
  whichever direction visited the bond last, and publishing it would freeze the convergence
  test's internal shape into the public record for a number the state already gives. TenPy
  reports a per-sweep `S` in `sweep_stats` because its `max_S_err` criterion is computed
  from it; tenet converges on the Schmidt *change*, which `max_dSchmidt` already reports.
  The reason is written on `DMRG_out` itself, where a reader looking for the field will be.

  Out of scope and unchanged: `spectrum`'s signature and callers, `svd_truncated` and how a
  bond `GradedSpace` is chosen (#209), `dmrg_`'s convergence test, and the segment/mutual-
  information reads (`mutinf_two_site`, `get_rho_segment`) that have no caller here.

- **M48** — shipped: the public measurement API over the two-state `Env` (#213). `overlap`,
  `measure_mpo`, `correlation_function` and `expectation_profile`.

  **The engine half was M61 Stage D and is not repeated here.** `Env(psi, h, bra=phi)`
  exists, `Env.measure()` on it *is* `<phi|H|psi>`, and `MPO.identity` makes it the plain
  overlap. What was still missing was a name a user could find: `Env`'s first positional
  argument is the *ket*, which is the constructor's shape and not a measurement's, and
  `_braket` — the two-state transfer pass, whose own docstring already said the two chains
  may carry different bond spaces — was private.

  **`overlap(bra, ket)`, undivided.** YASTN's `measure_overlap`/`vdot` and TenPy's
  `MPS.overlap` are both undivided, and so is `Env.measure`; a fidelity is
  `overlap(phi, psi) / (phi.norm() * psi.norm())` and the caller spells the division. The
  divided readings are the ones named `expectation_*`, which is the distinction M11c already
  drew. `MPS.norm` is now `overlap(psi, psi) ** 0.5` — expressed *through* it rather than
  beside it, so the one-state and two-state readings of the same pass cannot drift, and a
  source-reading test pins that `norm` contains no `einsum` of its own.

  **`measure_mpo(bra, h, ket)`**, YASTN's name and argument order, over
  `Env(ket, h, bra=bra).measure()`. With the identity MPO it agrees with `overlap` to 1e-10
  on two different converged states — two genuinely different contractions, one transfer
  pass and one environment sweep, meeting.

  **`correlation_function(psi, a, b, pairs=None)`**, TenPy's name because it is the term of
  art; the house's `_2site` vocabulary stays with `expectation_2site`, whose signature and
  adjacent-pair contract are untouched. The operators are `local_op`'s **rank-3 charged**
  form, which is the form `MPO.from_terms` takes and the only form a fermionic `c` has.

  **Fermions are correct because nothing new decides their sign.** Each pair is measured as
  a one-term MPO through `from_terms` plus `Env.measure`: the Jordan-Wigner string across
  the sites between `i` and `j` is the fZ2 braiding the term builder inserts and #147's
  explicit-JW oracle pins, and the contractions are the ones M23/#160 audited for the
  composition rule. A hand-written transfer walk carrying the charge leg would be faster and
  would re-decide that sign outside the audited machinery, which is exactly how #147
  happened. The test measures `<c+_up,i c_up,j>` on a converged N=4 Hubbard state against
  `_dense_c`, which writes the parity string out site by site, at every separation including
  `j - i >= 2` where a missing sign is a different number rather than a rounding.
  **The cost is stated rather than hidden**: one build and one pass per pair, so the
  all-pairs default is `O(N**2)` builds. `pairs=` is the way around it, and YASTN's cached
  transfer walk (`_measure.py`:130, a ~75-line body) is the named upgrade.

  **`expectation_profile(psi, o)` is the `O(N**2) -> O(N)` half.** `expectation_1site` ends
  in two full-chain transfer passes, so the `<S^z_n>` profile every DMRG user writes — and
  `examples/heisenberg.py` does write — is `O(N)` passes over an `O(N)` chain. The profile
  canonizes a copy, walks the orthogonality centre right by a `qr` per site and reads the
  operator off the centre, which a canonical MPS makes exact because both halves of the
  transfer close to the identity. Both references do exactly this. **Counted, not claimed**:
  at N=24 the test monkeypatches `tenet.einsum` and asserts the per-site loop spends more
  calls than `N**2` while the profile spends fewer than `8 N`, measured at 2280 against 72.

  **Where the four live, and why not one module.** `overlap` and `expectation_profile` sit
  in `mps.py` next to `_braket`, which is their machinery; `measure_mpo` and
  `correlation_function` sit in `env.py`, because they read an `Env` and `env.py` imports
  `mps.py`, not the other way. A `network/measure.py` holding all four cannot exist while
  `MPS.norm` is written through `overlap` — that is a cycle — and it would cost a third
  entry in the hygiene test's module list for no reader's benefit. The comment `mps.py`
  already carried about a future `measure.py` is updated to say so.

  Unchanged: `expectation_1site` and `expectation_2site` signatures and semantics, `Env`'s
  constructor, and the property that a measurement builds its own pass and never writes into
  a sweep's cache — asserted by identity on the caller's `F` entries.

  Out of scope: `H|psi>` and the variance (#214, which builds on `overlap`), excited states
  (#216, shipped as M61 Stage D), the Schmidt spectrum (#215, a canonical-form SVD and not a
  transfer pass), sampling and reduced density matrices, and infinite boundary conditions.

- **M49** — shipped: `MPO.apply` produces `H|psi>` as a new `MPS`, and `MPO.variance` is the
  convergence check that is not a change test (#214, over M48's `overlap`).

  **The variance is why the apply exists.** `dmrg_` stops when the energy stopped moving and
  the Schmidt values stopped moving, and both references say plainly that a change test can
  be satisfied by a run stuck on a wrong bond structure. `<psi|H^2|psi> - E^2` is the check
  that is not one, and it needed either `H @ H` or `H|psi>`. `H|psi>` is the smaller of the
  two — with the product exact, `<psi|H^2|psi>` is `<Hpsi|Hpsi>` and `<psi|H|psi>` is
  `<psi|Hpsi>`, so the whole thing is one apply and three overlaps. **No `MPO @ MPO`
  shipped**, and no `MPS.add`, `MPO.dagger`, `plus_identity` or `is_hermitian` either:
  every one is real in the references, none has a caller here, and a test asserts their
  absence so that "added for symmetry" is a decision rather than a drift.

  **The graded content is one turn-around.** The operator's virtual bond and the state's
  cross a site in *opposite* directions — the fact the Milestone 11 section spends a page on
  — so they cannot be fused until one is turned. Turning the operator's left virtual leg is
  a duality relabel, `tenet.flip_dual`, which charges `chi * theta` per fusion tree: `+1` on
  every bosonic sector, `-1` on an odd fermionic one. Writing the fusion without it gives a
  state that is wrong only under fZ2, and wrong by a *number* rather than by an error —
  measured at 1.1 against a scale of 1.5 on an N=4 Hubbard state before the flip was added.

  **The direction is fixed by the leg, not by the flag**: `inv = not leg.dual`. This is not
  a nicety. `MPO.from_terms`' two representations write that flag differently — a compressed
  table's internal bonds come back `dual=True`, a deferred table's `dual=False` — and
  charging by the flag would make `H|psi>` depend on which representation built `H`,
  silently and only for fermions. A brute force over every per-site flip pattern found the
  compressed rule ("flip exactly one of the two legs meeting at each bond") and found *no*
  pattern at all for the deferred table until `inv` entered; with `inv = not dual` both
  representations agree with the dense oracle and with each other, at N=2, 4 and 6, under
  U(1), SU(2) and fZ2, from `from_terms` at either cutoff and from `from_w`.

  **The operator's D=1 boundary legs are capped, not fused.** Fusing them would give the
  product a boundary leg that is the state's own written in the *other* dual convention, and
  `overlap(psi, h.apply(psi))` then refuses to contract — which is how the doctest found it.
  Capping with a `network.ones` vector, the same move `MPO.to_dense` makes by slicing
  `[0, ..., 0]`, leaves the product's boundary legs identical to `psi`'s, which is the
  property the variance rests on. A test asserts leg identity at both ends.

  **A deferred operator is materialised through `MPO.__getitem__`, and the docstring says
  so.** This is a whole-state product, not a sweep step: there is no bond at which the
  operator could stay symbolic, so one full `W` per site is what it costs and the cost is
  stated at the call. The sweep's own deferred path (#200, #204) is untouched.

  **Truncation is `MPS.compress_`, by name, and `apply` takes no `chi`.** `compress_`
  already takes the `chi`/`cutoff` pair the sweep takes and already returns the **total**
  discarded weight `sqrt(sum_bond dw)` — the convention this question wants, and the one
  `sweep_`'s per-bond *maximum* deliberately is not. Giving `apply` its own `chi=` would put
  a second name on that number and be the one place the two conventions could blur. The
  untruncated product is therefore the only thing `apply` returns, and the two-call form is
  the truncating one. Simplification: the zip-up apply that truncates *during* the sweep
  (YASTN's `zipper`, TenPy's `apply_zipup`) is the named upgrade and is a change with a
  measurement attached; the variance does not need it.

  Verified the two ways #214 names: `<psi|H|psi>` read through the apply plus `overlap`
  agrees with `Env.measure()` to 1e-10 on U(1), SU(2) and fZ2, and the variance of a
  converged N=8 Heisenberg state falls to below 1e-8 as `chi` goes 2 -> 32. The variance is
  additionally checked against a dense `<H^2> - <H>^2` oracle under SU(2) and fZ2.
- **M45** — shipped: the "project, don't check" mode of `from_dense`, `restrict` and
  `to_symmetry` has a name, `tenet.PROJECT` (#210).

  **The choice, and why it was the third option.** #210 offered three: split
  `from_dense` into a second entry point `project_dense`; split all three, adding
  `project_restrict` and `project_to_symmetry`; or name the sentinel. The counter-argument
  in the issue is real — `atol=math.inf` was documented in all three docstrings *and* in
  the two error messages, so it was never an undocumented magic value, and unlike the `-1`
  bond sentinel `svd_truncated` refuses, `inf` is not an arbitrary code: it is the limit of
  the parameter it is passed as, "any residual acceptable". The mode and the tolerance
  value genuinely coincide.

  That is exactly what makes the third option the right one. A split would have created a
  second function whose body is `from_dense(..., atol=math.inf)`, i.e. two names for one
  code path, and doing it consistently means three of them — three names, three docstrings
  and three error-message families for one concept. Naming the value costs one name, one
  line of code and no branch, applies to all three functions at once (so the consistency
  criterion is met without any inconsistency to state), and leaves every existing call site
  and test working *identically* rather than merely compatibly: `tenet.PROJECT is math.inf`
  is an assertion in the suite, not a promise in prose.

  **What it does not fix, honestly.** `atol=tenet.PROJECT` still reads as passing a
  tolerance, because it is one. What changes is that the reader of a call site no longer
  has to know the idiom to see which of the two operations — validating construction, or
  projection — is happening. If a caller ever needs projection to differ from validation in
  more than the check (a different residual convention, say), that is the point at which a
  separate entry point earns its name; it does not today.

  **Surface.** One addition, `tenet.PROJECT: float`. No signature changed anywhere:
  `atol` keeps its `float | None` type and its `None` default in `from_dense`, `restrict`
  and `to_symmetry`. The two error messages now name `atol=tenet.PROJECT (== math.inf)`,
  and `tests/ops/test_embed.py:649` — which asserts on that message — moved with them.

- **M46** — shipped: `tenet.enable_jax()`, one public entry for the JAX-facing features
  (#211).

  **One function, and the invasive half is a keyword.** The three statements it replaces
  were `import tenet.pytree`, `import tenet.ad`, `tenet.ad.install()`. They are not one
  feature: the pytree registration is local — it registers *our* type with JAX and changes
  nothing about anyone else's — while `install()` writes
  `autoray.register_function("jax", "linalg.svd", ...)`, which is process-global and
  reaches quimb and every other autoray user in the process. `tenet.ad`'s module docstring
  records the rule that follows: mutating another library's dispatch table is the user's
  act, not an import's.

  `enable_jax()` complies with that rule — it is an explicit call, which is the user's act
  the rule demands — but complying is not enough on its own, because a *default* is not an
  act. The two documented use cases decide the signature: the pytree alone is the common
  case (`README`, the VMC tutorial), and the broadened VJPs matter only for degenerate
  spectra (the CTMRG example). So `ad` is a keyword defaulting to `False`: the common call
  does the benign half, and the process-global half is opted into by name,
  `tenet.enable_jax(ad=True)`. A single call doing both by default would have made the
  invasive half a consequence of asking for the benign one, which is the thing the rule
  exists to prevent; two separate functions would have been a second name for a one-line
  body.

  **Idempotent, and loud without JAX.** Re-importing `tenet.pytree` is a `sys.modules`
  hit and `install()` documents itself idempotent, so repeat calls change nothing — pinned
  by a test that compares the observable registry state across a second and a third call.
  Without JAX the function raises its own `ImportError` naming `tenet-py[jax]`, so a
  JAX-less user gets a sentence rather than a traceback out of a submodule; the test blocks
  `jax` with a meta-path finder in a subprocess and asserts the raise came from
  `tenet/__init__.py`. It is written as a re-raise around the submodule imports, *not* as a
  `try: import jax` of its own, because three source greps
  (`tests/array/test_dispatch.py`, `tests/test_tensor_properties.py`,
  `tests/backends/test_pytree.py`) hold "core never imports jax" up by walking the files
  for that string — `pytree.py` and `ad.py` are the only two names allowed to contain it,
  and the invariant is worth more than the shape of one guard.

  **Nothing is deprecated.** `import tenet.pytree` and `tenet.ad.install()` are what
  `enable_jax` runs — one implementation of each — and `install()`'s signature and
  docstring are byte-identical. `tenet.ad`'s and `tenet.pytree`'s *module* docstrings gained
  a pointer at `enable_jax`; the docs (README, the VMC and DMRG tutorials, the guide, and
  the CTMRG and VMC examples) now teach the one-call spelling and no longer the
  three-statement one. JAX stays an optional extra: `pyproject.toml` is untouched and core
  still imports nothing from it.
- **M44** — shipped: the truncation *decision* is a returned object. `tenet.linalg.select_bond`
  makes the choice `svd_truncated` used to consume and hands it back as a `BondSelection`
  (#209); `svd(t, axes, bond=selection.bond)` then runs the numerics, jittable as ever.

  **One keep rule, not two.** The private `_decide` now owns the whole selection —
  `_admissible`'s cutoff prefix, the `qdim`-weighted greedy walk under `max_bond`, the
  two ValueErrors and the `renorm` factor — and `svd_truncated` is a caller of it. Its
  signature, docstring and behaviour are unchanged, checked by an `ast`-extracted
  comparison of every public name in `ops/linalg.py` against the base commit: the diff is
  `BondSelection` and `select_bond` and nothing else. `_spectrum` now takes `{c: magnitudes}`
  rather than the SVD tuples and returns the *index* within each sector, so a caller whose
  kept set is not a prefix can gather by index — that is M40's requirement, paid for here
  rather than duplicated there. The refusal message is `_not_traceable(caller)`, one
  sentence pattern for every truncating entry point.

  **The type is a frozen dataclass, and it is deliberately not a pytree.** It sits beside
  `MapLayout`, the other array-free structural record: immutable, no arrays, decided
  outside the trace. `pytree.py` registers `SymmetricTensor` and nothing else, so
  `BondSelection` is neither a registered container nor an intended leaf; a `NamedTuple`
  was rejected precisely because JAX flattens one *automatically*, which would turn the
  record's Python floats into leaves the moment it crossed a `jit` boundary — the accident
  the whole structure/numerics split exists to prevent. Only `.bond`, a hashable
  `GradedSpace`, is meant to cross. `tests/ops/test_select_bond.py` asserts
  `tree_leaves(selection) == [selection]`.

  **The discarded singular values are always retained, on the measured size.** One
  `(sigma, sector, index)` triple costs a measured 116 bytes of Python object; a spectrum
  of `N` values therefore costs `116 N` against the `8 · Σ_c rows_c · cols_c` bytes the
  blocks already occupy, a ratio of `14.5 / max(rows_c, cols_c)`:

  | fixture | spectrum `N` | blocks | discarded list | ratio |
  |---|---|---|---|---|
  | U(1), three sectors × `m=8` | 24 | 1.5 KB | 2.5 KB | 1.64 |
  | U(1), three sectors × `m=64` | 192 | 96 KB | 22 KB | 0.23 |
  | SU(2), `{j=0: 8, j=1/2: 8}` | 16 | 1.0 KB | 1.8 KB | 1.73 |
  | SU(2), `{j=0: 48, j=1/2: 48}` | 96 | 36 KB | 11 KB | 0.31 |

  The ratio exceeds 1 only where the tensor is a few kilobytes, i.e. where nobody is
  counting. The case the proposal worried about — a K=26 quantum-chemistry cut computing
  ~5·10⁴ singular values — is ~6 MB against the 6 GiB that run is measured at in the M39
  table above: 0.1%. An opt-in flag would recover that at the price of a keyword whose only
  job is to make the object's contents conditional, and `discarded_weight` walks the same
  list anyway. Never-retain was refused for the same reason: the weight is the number every
  DMRG caller actually wants.

  **What the record carries, and why `dense_dim` and `reduced_dim` are separate fields.**
  `max_bond` bounds `Σ_c qdim(c)·m_c`; callers routinely mean `Σ_c m_c`. For U(1) and fZ2
  the two coincide and the distinction is invisible, which is exactly why it must be
  spelled out rather than inferred. `undershoot = max_bond - dense_dim` and
  `next_multiplet` / `next_dense_cost` make the non-Abelian boundary case readable for the
  first time. The constructed case, `{j=0: 1, j=1: 3}` at `max_bond=5`
  (`tests/ops/test_select_bond.py::test_su2_max_bond_landing_inside_a_multiplet_is_reported`):
  the walk admits one triplet, reaches `dense_dim = 3`, and stops because the next entry is
  another triplet costing 3 and `3 + 3 > 5`. `undershoot` is 2.0 of a budget of 5 —
  40% — and `next_multiplet` names the `j=1` multiplet it stopped short of.
  `max_bond=6` spends the budget exactly. That number was always what `svd_truncated`
  produced; until now nothing reported it.

  **`renorm` is reported, not applied.** Every magnitude in the record is bare and `scale`
  carries `sqrt(Σ_all qdim σ² / Σ_kept qdim σ²)`. Mixing rescaled kept values with bare
  discarded ones would put two units in one object; `svd_truncated` reads `scale` off the
  same selection, so the rescaling is computed once.

  **The naming.** `select_bond` is verb-then-noun like `map_layout`, `flip_dual` and
  `to_matrices`; `BondSelection` is a noun-record like `MapLayout` and `CTMEnv`. The
  keyword set stays quimb's (`max_bond`, `cutoff`, `cutoff_mode`, `renorm`), unchanged from
  the M8 shim, and M31's naming audit is not reopened — this is an addition to the surface,
  not a rename of it. M40's `eigh_truncated` consumes this object rather than defining its
  own, which is why `_decide` takes a magnitude spectrum rather than SVD output.

- **M40** — shipped in the tensor layer, **measured and refused in the driver layer**:
  `tenet.linalg.eigh_truncated` and `eigh(..., bond=)` exist and are `svd_truncated` /
  `svd(..., bond=)`'s twins (#205); `network/ctmrg.py` is **unchanged**, and the reason is
  a measurement rather than a preference.

  **Part 1, the gate: the double-layer corner is indefinite, and the number is large.**
  `benchmarks/bench_ctm_corner_signs.py` walks six C4v moves per fixture and reports, per
  move, the corner's Hermiticity defect under both candidate basis pairings, its negative
  eigenvalues, how many of those sit above the projector's own truncation threshold, and
  `max_j |u_j - v_j|` between the two isometries `svd_truncated` would produce. That last
  quantity needs no gauge fixing: the SVD's freedom is a *joint* phase on `(u_j, v_j)`, so
  the difference is invariant, and it is `2|v_j|` on exactly the columns whose eigenvalue
  is negative.

  "Eigenvalue" presupposes an endomorphism, and `check_square` refuses every CTM corner —
  the C4v mirror identifies a space with its *dual*, which is a `flip_dual` and not an
  equality — so the instrument computes both candidate pairings and reads the spectrum off
  whichever is Hermitian. `direct` is `move`'s own `ndim // 2` order; `swapped` exchanges
  the domain's last two axes, i.e. a double-layer corner's ket/bra pair. A within-side
  transpose is a unitary on the domain, so it leaves `U` and `Sigma` exactly as `move` sees
  them.

  | fixture | move | herm `direct` | herm `swapped` | negatives | `|w_neg|/w_max` | `sigma_cut/sigma_max` | kept | `max|u-v|` |
  |---|---|---|---|---|---|---|---|---|
  | single-layer Ising `beta=0.4` (control) | 0–5 | ≤1.4e-16 | 0.49–0.84 | 0 | 0 | 0.014–0.049 | 0 | ≤6.7e-16 |
  | c4v iPEPS U(1), `chi=4` | 0 | 8.1e-01 | **0.0** | 1 | 3.87e-01 | 3.87e-01 | **1** | **1.70** |
  | c4v iPEPS U(1), `chi=4` | 1–5 | 0.59–0.62 | 0.48–0.52 | n/a | n/a | 0.59–0.62 | n/a | n/a |
  | c4v iPEPS SU(2) seed 1, `chi=6` | 0 | 1.2e+00 | **7.9e-16** | 1 | 6.14e-02 | 2.54e-01 | 0 | 8.3e-16 |
  | c4v iPEPS SU(2) seed 3, `chi=6` | 0 | 2.0e+00 | **9.2e-16** | 4 | 7.33e-01 | 1.22e-01 | **1** | **2.00** |
  | c4v iPEPS SU(2) seed 3, `chi=6` | 1–5 | 1.65–1.99 | 1.30–1.99 | n/a | n/a | 0.16–0.21 | n/a | n/a |

  **Verdict: the gate fires.** The single-layer Ising corner is Hermitian to 1.4e-16 at
  every move and has no negative eigenvalue anywhere — the control reads zero, so the
  instrument reads zero when there is nothing to read, and #102's Onsager agreement is
  explained rather than assumed. The double-layer corner is Hermitian at move 0, where
  nothing has been approximated yet, and **indefinite there**: a negative eigenvalue at 39 %
  (U(1)) and 73 % (SU(2) seed 3) of the largest, both above the projector's own cut, and
  `max|u-v|` of 1.70 and 2.00 — 2.00 being the textbook full sign flip. #77's trigger has
  fired and the defect is not a tolerance.

  **A second finding the issue did not anticipate: from move 1 on the corner is not
  Hermitian under either pairing** (0.48–1.99 relative), so "negative eigenvalue" stops
  being defined and the table says `n/a` rather than inventing a number. That is the C4v
  single-move environment losing self-consistency, and it is *not* repaired by the
  Hermitian projector — see below.

  **Part 2, the tensor layer, shipped.** `eigh(t, axes, *, bond=B)` is the same keyword
  with the same meaning and the same `_keep_counts` refusal, and `eigh_truncated` is
  `svd_truncated`'s signature, its six quimb `cutoff_mode` strings, its `qdim`-weighted
  cost and weight, and its `StructureChangingError`. Both go through M44's one keep rule:
  `eigh_truncated` hands `_decide` the magnitudes `|w|` and consumes the returned
  `BondSelection`, so there is no second truncation policy. Two places where the mirror is
  deliberately not literal:

  - **the kept set is not a prefix.** `svd` slices `[:k]` because `sigma_c` comes back
    descending; eigenvalues come back ascending, so "the `k` largest by `|w|`" is an
    `argsort` and a gather. A gather is a value-dependent *permutation*, never a
    value-dependent *shape*, so `eigh(..., bond=)` traces exactly as `svd(..., bond=)`
    does — `jit` and `grad` are both pinned, the latter against central differences.
    `eigh_truncated` calls the *same* gather helper rather than indexing by the selection's
    per-sector index list, which is both why the two-call form reproduces the one-call form
    exactly and why the torch and JAX backends work (a Python list is not an index there).
  - **the sign survives.** Only the ordering key is `|w|`; `W`'s retained entries are the
    signed eigenvalues. On a positive-definite input the two routes agree factor for
    factor; on an indefinite one, keeping everything, `U S U†` is wrong by exactly
    `Sum_neg qdim(c) (2 w)^2` — asserted as an equality, not a bound, on U(1), fermionic
    parity and SU(2).

  AD needed no new code, for M44's reason and #77's: `tenet.ad` already broadens
  `linalg.eigh`'s VJP alongside the SVD's, and the bond degeneracy is `min(rows_c, cols_c)`
  rather than the numerical rank.

  **Part 2, the CTMRG consumer, built and then withdrawn.** A `move` that routes the
  corner through `eigh_truncated` / `eigh(..., bond=)` whenever the two index groups pair
  into an endomorphism — a purely structural test on leg metadata, so identical inside and
  outside a trace, and one that the single-layer corner fails and the double-layer corner
  passes after the ket/bra transpose — was implemented and run. It is **not** in this
  branch, for two independent reasons:

  1. **It changes two pinned numbers in existing tests.** The converged iPEPS energy moves
     from -0.310993394006 to -0.199164542117 (U(1)) and from -0.038475159359 to
     -0.032991280629 (SU(2)). Those are `tests/integration/test_ctmrg.py`'s
     `ENERGY_BASELINE`, #107's migration criterion asserting bit-equality with a deleted
     implementation, plus the committed output fence of `docs/examples/toy-ctmrg.md`.
     Everything else in that module passes with the change — the three `double_layer_ctm`
     tests, the gradient against central differences, the SGD descent, the traced-structure
     check, and the single-layer Onsager agreement, which the structural branch leaves on
     the SVD route untouched. A pinned-baseline edit is a decision for whoever owns #107's
     criterion, not a side effect of this issue.
  2. **The measurement says it does not deliver what the issue expected.** Re-running the
     Part 1 instrument with the Hermitian projector in place leaves the corner just as
     non-Hermitian from move 1 on (0.49–0.51 against 0.48–0.52 on the U(1) fixture). So the
     sign diagonal is *not* the only thing wrong with a single C4v move on a double layer,
     and `ctmrg.py`:510-518's precondition and Simplification comment cannot honestly be
     deleted on the strength of `eigh_truncated` alone. The remaining defect is the one
     those comments name second — four directional moves — and it stays #205's out-of-scope
     item with a number behind it now instead of a prose expectation.

  #77's "add both together when a caller needs the Hermitian route" is therefore
  **discharged**: the caller was found, measured, and the pair it needs is shipped. What is
  left open is the driver-layer change, which now has its own two-line argument above.

Not planned: TDVP, iDMRG, fermionic swap gates and PEPS containers. Excited states
left that list with M61 Stage D above.
- **M55** — **measured, not shipped**: the pre-placement basis choice — M39's named successor,
  block2's Part 2 position — was built as a prototype, taken to its gate, and the gate says
  the change does not belong in `src/` (#222). The measurement is the whole deliverable.

  **What was built.** `benchmarks/bench_preplace_mpo.py` carries a self-contained
  pre-placement assembler with block2's shape, read first-hand and written independently
  (`general_mpo.hpp`:540-541, :666-780, :723-730, :763, :764-805, :828-833, :1210-1214,
  :1381+; block2 is GPL-3.0 and no line of it is reproduced): the coefficient stream is
  carried forward on the *kept* basis so every decomposed matrix has
  `(kept basis) × (site fan-out)` rows and never `D_FSM`; both sides are interned per cut,
  left rows by `(kept index, site slot)` and right columns by the remaining operator string;
  the SVD runs dense on the `szl × szr` scalar block per quantum number; the singular values
  are folded into the next cut's coefficients rather than into a tensor; and `IdL`/`IdR` are
  excluded from the choice and given their own slots, which is M39's corner pinning applied
  one stage earlier. Correctness first: the operator built through the chosen basis agrees
  with `from_terms` at `cutoff=None` to 2e-16 – 9e-16 on the spin, spinless-fermion and
  spinful Hubbard fixtures, and with M39's own compressed operator to the same tolerance.

  **Gate 1**, per cut, on M39's fixtures. `post` is M39's shipped width. Three widths of the
  one mechanism, because the basis choice and the truncation policy are separable and only
  the first is what the successor was about: `pre` truncates *inside* the pre-placement SVD;
  `exact` is the same choice at a rank-revealing threshold, which is the width `_place`
  would be handed; `swept` is `exact` after M39's own two pinned truncating sweeps, which on
  the pre-placement bond decompose a `χ × d² × χ` tensor instead of a `D_FSM × d² × χ` one.
  `place` is the widest `_place` buffer each route allocates.

  | fixture | N | D_FSM | post | pre | exact | swept | pre/post | swept/post | place M39 | place pre | pre wall | pre RSS |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | H4 STO-6G | 8 | 108 | 30 | 30 | 30 | 30 | 1.000 | 1.000 | 0.000 G | 0.000 G | 0.0 s | 0.00 G |
  | H8 STO-6G | 16 | 1 148 | 122 | 122 | 122 | 122 | 1.000 | 1.000 | 0.001 G | 0.000 G | 0.1 s | 0.02 G |
  | N2 STO-3G | 20 | 588 | 96 | 96 | 96 | 96 | 1.000 | 1.000 | 0.001 G | 0.000 G | 0.1 s | 0.02 G |
  | H10 STO-6G | 20 | 2 376 | 192 | 192 | 192 | 192 | 1.000 | 1.000 | 0.003 G | 0.001 G | 0.2 s | 0.10 G |
  | N2 CAS 6-31G (K=16) | 32 | 5 111 | 562 | 562 | 562 | 562 | 1.000 | 1.000 | 0.048 G | 0.008 G | 2.4 s | 0.60 G |
  | C2 CAS cc-pVDZ (K=26) | 52 | 31 441 | 736 | 766 | 766 | **736** | 1.333 | **1.000** | 0.277 G | 0.017 G | 38.7 s | **5.14 G** |
  | syn-42 (K=42) | 84 | 10 764 | 54 | 175 | 228 | **54** | 4.154 | **1.000** | 0.013 G | 0.002 G | 56.9 s | **4.33 G** |

  Every ratio is `max` over the cuts; the inner reading (excluding the two boundary-adjacent
  cuts, M39's own split) is identical to the all-cuts reading on every row, so nothing here
  turns on which one is read. **The literal gate — truncate inside the pre-placement SVD —
  fails at 4.154.** **The mechanism's own gate — the pre-placement basis with M39's
  truncation left where it is — passes at exactly 1.000, cut for cut, on every fixture.**

  **The two readings differ for a reason that is about the metric, not the basis.** `rsum2`
  weighs a discarded direction against the norm of the matrix it decomposes. Post-placement
  that matrix is the operator; pre-placement it is a matrix of scalars whose columns are
  operator *strings*, and the string basis is orthogonal only where the strings differ on
  some site by orthogonal operators. The prototype carries each slot's Frobenius norm
  relative to the identity's so the two metrics agree where the basis is orthogonal, and it
  moved C2 not at all and syn-42 from 4.200 to 4.154 — a dense synthetic integral set is
  precisely where `n`-type strings overlap and the diagonal approximation is worst. Where no
  truncation happens at all the two readings are bit-identical, which is the cleanest
  statement available that the *rank* is reproduced: five fixtures agree at every cut, and
  C2's six untruncated middle cuts (631, 598, 585, 576, 571, 570) agree entry for entry.

  **The reason the change does not ship is not the width. It is that the transient it
  removes is smaller than the transient it introduces.** The object #222 set out to delete
  is `_place`'s `D_FSM × d² × χ` buffer; measured, that buffer is **0.277 GiB at K=26** and
  **0.013 GiB at K=42**. The pre-placement basis choice that replaces it needs
  `szl × szr` scalars per quantum number, and measured, that costs **5.14 GiB at K=26** and
  **4.33 GiB at K=42** (4.74 GiB on an independent run of the same point) — one to two
  orders of magnitude more than the thing it removes, at a wall time (38.7 s at K=26) equal
  to the entire shipped build (38.9 s). Pre-placement
  moves the ceiling **backwards** on this input set.

  **Why block2 does not pay this and tenet would.** `szr` is the number of distinct
  *remaining* operator strings at a cut — 79 838 at C2's first cut, ~19 000 at its middle —
  and the SVD is dense on `szl × szr` **per quantum number**. block2 blocks by particle
  number, `Sz` and point group, which at K=26 is of order a hundred blocks, and
  `Σ_q szl_q · szr_q` falls with the block count. tenet's spin-orbital sites are graded
  **fZ2**, which is two blocks, so almost nothing falls out. The mechanism whose cost block2's
  symmetry pays for is the one tenet does not have at this layer. That is a statement about
  the *grading carried on the MPO bond*, not about the order of operations, and it is where
  a successor to this measurement would have to start: an MPO bond graded by `U(1) × U(1)`
  (particle number and spin) rather than by fZ2 is what makes the pre-placement block sizes
  fall, and that is a change to what `local_op`'s charge means, not to the assembler.

  **A second finding, independent of the memory one: the design as filed does not cover
  tenet's non-Abelian route.** block2's operator alphabet is single-site and second-quantized,
  so a left partial string is a sequence of on-site symbols and the coefficient stream is
  scalars. tenet's SU(2) MPOs cannot be written that way — `local_op` refuses a
  symmetry-breaking single-site factor on non-Abelian legs, so an SU(2) Hamiltonian is a
  *k-site invariant* operator that `_split` peels into pieces with a **derived internal
  bond** (`tests/network/test_mpo.py`'s `su2_heisenberg` is the whole model in two lines).
  Where that bond carries degeneracy above one, a bond direction pairs a left factor and a
  right factor through an index that the coefficient stream cannot see: the pre-placement
  matrix stops being `coefficients × atoms` and the rank of the scalar matrix stops bounding
  the operator's. It is fixable — expand the interning key by the piece's own bond index and
  block the SVD by it — but it is a design the successor's issue does not contain, and it is
  the one place where "adopt block2's shape" does not transfer, for the same reason M39
  recorded for the coupling coefficients: block2 knows what its operators *are* and tenet's
  are opaque caller tensors.

  **What stands from the measurement.** The pre-placement basis choice reaches M39's bond
  exactly, so the *basis* question the successor was filed on is answered and closed: there
  is nothing to gain in width and nothing to lose. `_place`'s buffer would shrink 16× at
  K=26 (0.277 → 0.017 GiB) and 9× at K=42, which is real and which is not where the build's
  6–7 GiB peak is. **The K ceiling therefore does not move**, and M35's `K^4.22` build wall
  and the K≈48–53 estimate stand unrevised; no ceiling measurement at K=42 or beyond is
  reported here because no production change shipped to measure one on. The refusal of a
  max-flow (#138, re-measured twice) is untouched and remains a refusal.

- **M52** — shipped: `MPO.from_entries` builds an MPO from the **non-zero entries of each
  site's `W`**, and what it produces is an `EdgeTable` rather than a list of site tensors —
  so a hand-built operator is indistinguishable to `Env` from a `from_terms` one and runs
  on M39's single prepared engine path (#217, over #204's engine and #200's boundary).

  **The fixed constraint decided the shape.** The point was never convenience alone:
  `from_w` produces a numeric MPO with no symbols, and that is *why* `Env.heff2` needs a
  compatibility entry at all — symbols cannot be recovered from a numeric `W` (#141
  measured that a compressed one retains no edge structure). So the builder had to fill
  the same five structures `_edge_table` consumes — `states`, `order`, `moves`, `stops`,
  `spectators` — and a per-site sparse `{(i, j): entry}` mapping matches `moves[n]`'s
  `{(state_l, state_r): W}` one for one. It does more than match it: the builder *drives*
  `_Walk` itself, so the fused running charge, the per-state `GradedSpace`, the `dual`
  convention and the fermionic R-coefficient on the site's own physical line are the same
  code the term list runs, not a second copy of it.

  **The spelling: per-site sparse dicts, against the three references.** MPSKit is the
  closest fit and was already cited twice in `mps.py` — its matrix constructor takes
  entries that are `MPOTensor`, `Missing` or `Number`, which is exactly the vocabulary
  `moves`/`stops`/`spectators` already carry, so tenet's four spellings are `None` (the
  identity, a spectator ride on `(i, i)`), a number (that multiple of it), a rank-3
  `local_op` operator, and the pair `(coefficient, operator)` — the pair added because a
  `W` is written on paper as a coefficient times a named operator. TenPy's `MPOGraph`
  (`{keyL: {keyR: [(opname, strength)]}}`) is the same graph keyed by arbitrary hashables
  and reached the same conclusion from the other side; its `from_grids` dense nested lists
  are what this builder exists not to make anyone write. **YASTN's contribution is a
  negative result**: between a fully formed tensor (`A[n] = t`) and a term list (`Hterm`,
  `generate_mpo`) it offers nothing at all, which is the gap being filled. A flat
  `(site, i, j, op)` iterable was refused — it loses the visual correspondence with the
  printed `W`, which is the whole reason a caller reaches for this entry — and a builder
  object was refused as an interface with one implementation and one call site.

  **`IdL` and `IdR` are by convention: bond index `0` and bond index `-1`.** MPSKit fixes
  the same two by position (`V[1] = V[end] = _rightunit`, the `(1 C D; . A B; . . 1)`
  partition `EdgeBlocks` implements) and tenet's own layout already assumes it — `_merge`
  direct-sums a cut in `[_IDL, *open, _IDR]` order and M39's pinning reads the two corners
  off the first and last slot of the bond's unit sector. TenPy carries `IdL`/`IdR`
  *explicitly* because its graph keys are arbitrary hashables with no order to lean on;
  here an explicit pair would be a second source of truth that `_merge` could contradict,
  which is the same argument `from_terms` makes for having no `phys=`. Python's `-1` is
  what makes the convention free: **no bond width is ever declared or inferred**, because
  the last index needs no width to name, and the caller's channel numbering is the
  textbook's rather than the grading's (`from_w`'s dense rows have to be ordered by charge
  to line up with `GradedSpace`; these do not). The convention is then made
  *self-enforcing* rather than assumed: an entry into `IdL`, an entry out of `IdR`, or a
  non-identity on either corner is refused by name — which is the same four zeros the
  corner-exactness property asserts, so the refusals and the property are one statement.

  **An invariant rank-2k operator is refused**, with `from_arrays`' argument and a pointer
  to `from_terms`: one `W` entry sits on one site, and `local_op`'s invariant form spans
  *k* of them through an SVD, so it has nowhere to put the other *k*−1 indices.

  **The two boundary bonds are `D=1`** and are trimmed by the finite-state machine's own
  reachability pruning rather than by slicing — bond `0` keeps only `IdL` and the last bond
  only `IdR` — which is `from_w`'s `start` row and `end` column and is what lets one bulk
  mapping be handed over for every site including the ends. A dead channel therefore raises
  only when it is dead at **every** bond it is named at: a range-2 coupling's channel is
  legitimately dead at the last bond, and refusing that would refuse every finite-range `W`.

  **`from_w` is kept, unchanged, and is not deprecated.** Its input is a *dense array*,
  which is what you have when the `W` comes out of a paper or another library — there the
  entries are numbers, no charge can be recovered from them, and the caller must supply the
  grading. `from_entries` cannot take one. Deprecating `from_w` would also not close the
  compatibility entry in `Env.heff2`, because `MPO(sites)` is public and equally
  symbol-free; closing that branch means refusing externally-built MPOs, which is a
  public-surface decision of its own and stays out of scope. What changes is the
  *recommendation*: `from_entries` is the way to hand-build, and `docs/guide/models-and-sites.md`
  and `examples/heisenberg_walkthrough.py` say so while showing both — the walkthrough
  still leads with the 5×5 `W` a textbook prints, because that is what teaches what an MPO
  is, and now runs all three routes to the same twelve digits.

  **No `cutoff`.** `from_terms`' two compressing sweeps exist because a term list can be
  numerically low-rank in a way the graph cannot see (a power law, an integral file); a
  `W` somebody sat down and wrote is the bond they chose. An operator that wants the sweeps
  wants `from_terms`, which is where the knob lives.
- **M64** — **measured, not shipped**: tenet against YASTN, head to head, on the U(1)
  Heisenberg chain and the fermionic Hubbard chain (#245). `benchmarks/bench_vs_yastn.py`
  is the whole deliverable; there is no `src/` change and YASTN is **not** a dependency of
  this project — not core, not dev, not test. The script's header carries the one-line
  install and the `uv run --no-sync` that keeps a later sync from removing it again, which
  is the packaging statement `REPOSITORY_RULES.md` asks for: a benchmark's opponent
  library is not something `uv sync` should ever fetch.

  YASTN is the fairest available opponent. Both are pure Python over NumPy, both block by
  symmetry sector, both run two-site DMRG with an `ncv = 3` Lanczos — tenet's default was
  deliberately YASTN's. Against block2 a gap would be a C++/MKL constant factor and would
  say nothing; against YASTN what is left is design.

  **The conditions, and the one that could not be matched.** Every knob below is a knob
  that, left unmatched, makes the numbers a lie, so each is stated rather than assumed.

  | condition | how it is held equal |
  |---|---|
  | Hamiltonian | one term list, `(amplitude, [(op, site), ...])`, translated per arm — tenet `MPO.from_terms`, YASTN `Hterm` + `generate_mpo`. The #213/#244 correspondence, exercised. |
  | truncation | `chi` only, on both sides: tenet `chi=chi, cutoff=0.0`, YASTN `opts_svd={'D_total': chi}`. See the rule mapping below. |
  | Lanczos | `ncv=3` both; YASTN `opts_eigs={'hermitian': True, 'ncv': 3, 'which': 'SR'}`. |
  | method | `'2site'` both, and one sweep means the same thing: left-to-right then right-to-left (tenet `sweep_`, YASTN `_dmrg_sweep_2site_`). |
  | convergence | **disabled**, so the comparison is per sweep and not per "convergence": YASTN `energy_tol=None, Schmidt_tol=None`; tenet `energy_tol=0.0, schmidt_tol=0.0` (`denergy` is an absolute value, so `< 0.0` never fires). |
  | initial state | one `bond_charges` spec — charge to degeneracy, per bond — and both arms seed a random MPS on exactly those spaces, full rank from sweep one so `chi` means what it says. The realized bond dimensions are recorded per point and **match bond for bond** on every point of both grids. |
  | threads | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` all set to `1` **in the process, before NumPy is imported**; one process per point. The #224/#226 discipline. |
  | measurement | first sweep discarded (it carries canonization and environment setup on both sides); steady wall is the mean of the last four; peak RSS is `resource.getrusage(RUSAGE_SELF)`; one JSONL row appended the moment a point finishes, so a kill loses at most one point and a rerun skips what is already recorded. |

  **The one mismatch: the random *entries*.** The bond spaces are identical by
  construction, but the two libraries have different generators and there is no way to make
  one draw the other's. That costs nothing in the wall column — the arithmetic sees block
  shapes, and those match — and shows up in the energy column only as the residual
  convergence difference at a fixed sweep count, which is what the `dE` column measures.

  **The truncation rule, mapped.** YASTN's `tol` keeps `sigma > tol * max|sigma|`
  (`yastn/tensor/linalg.py`, `truncation_mask`: `ff(S.data) > tol * backend.max_abs(S.data)`).
  That is **exactly** tenet's `cutoff_mode="rel"` (`ops/linalg.py::_admissible`:
  `sigma > cutoff * spectrum[0][0]`) — the same rule, not a near neighbour. tenet's sweep
  does not expose `cutoff_mode` and uses the `"rsum2"` default, which is a *different*
  rule (a discarded-weight budget, not a threshold on the largest value). Rather than
  compare two rules, the cutoff is switched **off on both sides** — `cutoff=0.0` admits
  the whole spectrum in `rsum2`, YASTN's `tol` defaults to `-inf` — leaving `chi` /
  `D_total` as the only rule acting. Those two *are* the same rule: both keep the largest
  values up to a total kept count, and on U(1) and Z2 every sector has quantum dimension
  1, so tenet's `qdim`-weighted dense budget is a plain count. A run wanting both knobs at
  once would have to reach past `dmrg_` to `svd_truncated` for `cutoff_mode="rel"`, which
  is a `src/` question and not this measurement's.

  **The fermionic grading, and what it costs the comparison.** tenet grades the Hubbard
  site by **fZ2** — the Jordan-Wigner string *is* the braiding (M21/#147) — so the `d = 4`
  site is two blocks of 2, even `{|0>, |ud>}` and odd `{|u>, |d>}`. YASTN is run through
  `SpinfulFermions(sym='Z2')`, whose site is the same two blocks of 2 in the same basis
  order. **Z2 and not U(1)xU(1) on purpose**: U(1)xU(1) grades by `(n_up, n_dn)` and would
  hand YASTN four blocks of 1 on the site and correspondingly finer virtual blocks, i.e.
  strictly less arithmetic for the same `chi`. That would be a comparison of gradings, not
  of implementations, and the reader should not have to infer it: **YASTN is deliberately
  run at the coarser of the two gradings it offers, the one tenet has.**

  **The anchor: the two term lists are the same Hamiltonian.** Exact diagonalization is out
  of reach at every grid point (the smallest is `N = 16` spinful, i.e. `4**16`), so the ED
  check is a separate small run whose only job is to prove the translation. Both arms, both
  models, against a dense oracle built independently in the benchmark — the fermionic one
  on `tests/network/test_hubbard.py`'s on-site matrices, which that module checks against
  the two-mode `kron` construction before any chain is built:

  | model | N | chi | E tenet | E YASTN | E dense | tenet − ED | YASTN − ED |
  |---|---|---|---|---|---|---|---|
  | Heisenberg | 10 | 32 | -4.258035207282879 | -4.258035207282889 | -4.258035207282894 | 1.5e-14 | 5e-15 |
  | Hubbard `U/t=4` | 4 | 16 | -2.624942271510865 | -2.624942271510864 | -2.624942271510848 | 1.7e-14 | 1.6e-14 |

  **U(1) Heisenberg**, `J = 1`, nearest neighbour, `S^z_tot = 0`, 30 sweeps; steady wall is
  the mean of the last four and is flat to the reported precision on both arms, `run` is
  the whole 30-sweep wall including the transient. Apple M4 Max, macOS 15.5,
  single-threaded Accelerate, NumPy 2.5.2, Python 3.12.11, YASTN 1.6.2.dev401+gb0187c49b.

  | N | chi | sweeps | E tenet | E YASTN | dE | steady tenet | steady YASTN | steady x | run tenet | run YASTN | run x | RSS tenet | RSS YASTN | RSS x |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 32 | 64 | 30 | -13.997315618 | -13.997315618 | 3e-13 | 1.14 s | 0.49 s | **2.32x** | 75 s | 17 s | **4.5x** | 0.68 G | 0.14 G | 5x |
  | 32 | 128 | 30 | -13.997315618 | -13.997315618 | 5e-15 | 1.28 s | 0.58 s | **2.23x** | 102 s | 20 s | **5.2x** | 1.11 G | 0.19 G | 6x |
  | 32 | 256 | 30 | -13.997315618 | -13.997315618 | 1e-13 | 1.63 s | 0.75 s | **2.17x** | 119 s | 25 s | **4.7x** | 1.56 G | 0.34 G | 5x |
  | 64 | 64 | 30 | -28.175424807 | -28.175424807 | 1e-10 | 2.45 s | 1.06 s | **2.31x** | 377 s | 41 s | **9.1x** | 3.40 G | 0.20 G | 17x |
  | 64 | 128 | 30 | -28.175424860 | -28.175424860 | 2e-13 | 3.19 s | 1.46 s | **2.18x** | 573 s | 57 s | **10.0x** | 4.94 G | 0.28 G | 18x |
  | 64 | 256 | 30 | -28.175424860 | -28.175424860 | 3e-13 | 4.40 s | 2.11 s | **2.08x** | 649 s | 77 s | **8.4x** | 6.03 G | 0.48 G | 13x |

  **Spinful Hubbard**, `U/t = 4`, `t = 1`, even total parity, 20 sweeps. tenet on fZ2,
  YASTN on `SpinfulFermions(sym='Z2')` — the *same* grading, chosen as above; a U(1)xU(1)
  YASTN run would have smaller blocks and is not what is compared here.

  | N | chi | sweeps | E tenet | E YASTN | dE | steady tenet | steady YASTN | steady x | run tenet | run YASTN | run x | RSS tenet | RSS YASTN | RSS x |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 16 | 64 | 20 | -12.541897371 | -12.541897073 | 3e-07 | 0.61 s | 0.18 s | **3.42x** | 13 s | 4 s | **3.3x** | 1.23 G | 0.12 G | 11x |
  | 16 | 128 | 20 | -12.541951444 | -12.541951446 | 2e-09 | 1.78 s | 0.50 s | **3.54x** | 36 s | 11 s | **3.3x** | 2.75 G | 0.49 G | 6x |
  | 16 | 256 | 20 | -12.541952157 | -12.541952157 | 3e-11 | 6.88 s | 1.79 s | **3.84x** | 139 s | 37 s | **3.8x** | 4.07 G | 0.76 G | 5x |
  | 32 | 64 | 20 | -25.693800734 | -25.693803905 | 3e-06 | 1.41 s | 0.45 s | **3.14x** | 29 s | 9 s | **3.1x** | 1.89 G | 0.19 G | 10x |
  | 32 | 128 | 20 | -25.695279169 | -25.695279118 | 5e-08 | 4.44 s | 1.49 s | **2.98x** | 90 s | 29 s | **3.1x** | 3.48 G | 0.52 G | 7x |
  | 32 | 256 | 20 | -25.695370428 | -25.695370431 | 2e-09 | 19.43 s | 5.50 s | **3.53x** | 391 s | 111 s | **3.5x** | 4.49 G | 0.88 G | 5x |

  Every point completed inside its budget; nothing is recorded as unfinished and nothing
  was dropped. **The energies agree**: 5e-15 – 1e-10 on Heisenberg, and on Hubbard
  3e-6 – 3e-11 falling monotonically with `chi` at fixed `N`, which is the signature of a
  residual *convergence* difference between two different random seeds rather than of a
  Hamiltonian difference — the ED anchor closes that question at 1.7e-14.

  **The verdict, in three parts.**

  **1. Per sweep, the lattice lane is at parity within a small constant factor: tenet is
  2.1–2.3x slower on U(1) Heisenberg and 3.0–3.8x slower on fZ2 Hubbard.** The factor is
  **flat in `chi` over 64 → 256** on both models — a 4x `chi` is roughly 64x the arithmetic
  per block, and the ratio does not move. That is the informative part: a gap that were
  fixed Python dispatch cost per block would *collapse* toward 1 as the blocks grew, and it
  does not. The gap scales with the arithmetic, so it lives in the contraction path rather
  than in per-call overhead. M57's per-phase instrument, re-run on a U(1) chain at
  `N ∈ {32, 64}` and `chi ∈ {64, 256}` (`benchmarks/bench_sweep_phases.py --model lattice`;
  that fixture carries an extra `J2` term, so `D_w = 8` against the head-to-head's
  `D_w = 5`, and it is cited for *where the time is*, not for the wall itself), says where:

  | N | chi | `heff2_apply` | `env_update` | `heff2_prepare` | `lanczos_own` | `svd` | assemble + writeback + spectrum | residual |
  |---|---|---|---|---|---|---|---|---|
  | 32 | 64 | 51.1 % | 20.3 % | 14.7 % | 5.5 % | 3.6 % | 4.1 % | 0.6 % |
  | 32 | 256 | 50.2 % | 20.2 % | 12.6 % | 5.5 % | 7.2 % | 3.7 % | 0.6 % |
  | 64 | 64 | 55.7 % | 18.8 % | 12.1 % | 5.2 % | 3.5 % | 4.2 % | 0.5 % |
  | 64 | 256 | 55.0 % | 18.9 % | 10.6 % | 5.2 % | 5.7 % | 4.1 % | 0.5 % |

  (`heff2_other` is under 0.05 % on every row and is left out of the table rather than
  rounded into a neighbour.) **83–87 % of a tenet sweep is the two-site matvec plus the two
  environment folds**, and the driver's own bookkeeping — assemble, writeback, spectrum,
  residual — is under 5 %. Any 2–4x therefore has to be found in `_apply2`, `_prepare2` and
  `Env.update_` and nowhere else: the sweep around them has no 2x to give.

  **2. The transient is the bigger number, and it is invisible in the per-sweep column.**
  On the fZ2 Hubbard the run ratio equals the steady ratio (3.1–3.8x against 3.0–3.8x):
  there is no transient. On the U(1) Heisenberg they diverge — the steady ratio is
  2.1–2.3x but the **whole-run ratio is 4.5–10.0x**, and at `N = 64` tenet's first sweep is
  140–147 s against YASTN's 4–5 s and does not reach steady state for roughly ten sweeps.
  Charged properly: 30 sweeps at `N = 64`, `chi = 256` cost tenet 649 s, of which about
  520 s is approach and 130 s is steady state, against YASTN's 77 s of which about 14 s is
  approach. The distinguishing feature is the *sector count on the bond*: fZ2 has two
  sectors and its bond structure is fixed from sweep one, while a U(1) bond at `N = 64`
  carries up to 33 sectors whose degeneracies the truncating SVD redistributes every sweep
  until the state settles — and every new bond structure is a new structural plan. This is
  the same object #227/M58 measured from the inside, as a distinct-structure-key count,
  seen here from the outside and on a lattice model rather than a quantum-chemistry one.

  A benchmark that reported only the steady sweep would have reported parity and been
  wrong about the thing a user waits for. **Both columns are in the tables for that
  reason**, and the run column is the one a fixed-sweep-count production run pays.

  **3. Memory is the material gap: tenet's peak RSS is 5–18x YASTN's**, and unlike the wall
  it does **not** track the state. At `chi = 64` the Heisenberg state is a few tens of MiB,
  yet tenet peaks at 0.68 GiB at `N = 32` and 3.40 GiB at `N = 64` while YASTN moves 0.14 →
  0.20 GiB. RSS growing with `N` at fixed `chi`, on a sweep whose working set does not, is
  per-bond state rather than the two-site tensor: `Env` holds two per-bond caches,
  `_prepared` and `_cores`, each bounded by `common.CACHE_BUDGET = 1 GiB` (#202), so 2 GiB
  of the gap is a designed and deliberate trade — bought, per M58, to stop re-preparing an
  operator the sweep is about to ask for again. What is measured here is that the trade is
  priced in gibibytes on a *lattice* model where the thing it buys is worth about 2x, and
  that the budget is a constant rather than a fraction of what the run can afford.

  **What is not claimed.** No optimisation was attempted in response; that is the
  follow-up's business, with the number attached. Nothing here says the 2–4x is
  irreducible, and nothing here says it is easy: the phase table names three functions and
  says the rest of the sweep has no room in it, which is a starting point and not a
  diagnosis. The comparison is at a *matched grading*, so it does not speak to what YASTN
  would do on U(1)xU(1) Hubbard, and it does not speak to the non-Abelian lane at all,
  where YASTN has no counterpart to compare against.

- **M64b** — **measured, not shipped**: the same head-to-head with a third arm, tenet on the
  **site-tensor path** (#245, follow-up to M64). `benchmarks/bench_vs_yastn.py` gains
  `--arm tenet-sites`, identical to `--arm tenet` in every respect but one: the operator
  handed to `dmrg_` is `MPO(h.sites)` — the same tensors `from_terms` built, with the edge
  description dropped — so `Env.heff2`, which routes on `self.h.edges is not None` and on
  nothing else, takes the **compatibility entry** (YASTN's `Heff2` contraction order)
  instead of the prepared, block-shaped one. Same state from the same `bond_charges` spec
  and the same seed, same `chi`, `ncv = 3`, cutoff off, same sweep counts (30 / 20),
  single-threaded, one process per point, warm-up discarded, steady = last four. The
  realized bond dimensions match the prepared arm's bond for bond at every one of the
  twelve points, and the two arms' energies agree to **3e-13 or better everywhere**, which
  is the operator identity the routing claim needs. The `tenet` and `yastn` columns below
  are **#247's own rows, not re-run** — same machine, same session, same conditions.

  **Why the arm.** Two earlier measurements already priced the prepared path against the
  site-tensor path *internally*: M16/#141 measured a full sweep 1.5–2.6x **slower** on the
  prepared path on plain NumPy at small bond dimension, and M39/#218's `chi` grid put the
  lattice ratio flat in `chi` at **~3.2x** at `D_w = 8`. M64 then measured tenet 2.1–2.3x
  (U(1)) and 3.0–3.8x (fZ2) slower than YASTN — and every M64 tenet point went through
  `from_terms`, i.e. through the prepared path. The arm separates the two factors:
  `tenet-sites / yastn` prices tenet's contraction path against YASTN's, and
  `tenet / tenet-sites` prices the prepared machinery on its own, on the same run.

  **U(1) Heisenberg**, 30 sweeps. `dE` is against the prepared arm at the same point.

  | N | chi | steady sites | steady YASTN | sites/YASTN | tenet/sites | run sites | run YASTN | run x | RSS sites | RSS YASTN | RSS x | E sites | dE |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 32 | 64 | 0.55 s | 0.49 s | **1.13x** | 2.06x | 28 s | 17 s | 1.65x | 0.26 G | 0.14 G | 1.8x | -13.997315618007 | 5e-14 |
  | 32 | 128 | 0.65 s | 0.58 s | **1.12x** | 1.98x | 38 s | 20 s | 1.93x | 0.42 G | 0.19 G | 2.2x | -13.997315618224 | 2e-15 |
  | 32 | 256 | 0.91 s | 0.75 s | **1.21x** | 1.80x | 49 s | 25 s | 1.92x | 0.61 G | 0.34 G | 1.8x | -13.997315618224 | 2e-15 |
  | 64 | 64 | 1.20 s | 1.06 s | **1.13x** | 2.04x | 121 s | 41 s | 2.93x | 1.17 G | 0.20 G | 5.8x | -28.175424807381 | 5e-14 |
  | 64 | 128 | 1.61 s | 1.46 s | **1.10x** | 1.98x | 182 s | 57 s | 3.19x | 1.73 G | 0.28 G | 6.3x | -28.175424859649 | 7e-15 |
  | 64 | 256 | 2.70 s | 2.11 s | **1.28x** | 1.63x | 230 s | 77 s | 2.97x | 2.08 G | 0.48 G | 4.4x | -28.175424859743 | 3e-14 |

  **Spinful Hubbard** `U/t = 4`, fZ2 against YASTN's `SpinfulFermions(sym='Z2')`, 20 sweeps.

  | N | chi | steady sites | steady YASTN | sites/YASTN | tenet/sites | run sites | run YASTN | run x | RSS sites | RSS YASTN | RSS x | E sites | dE |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 16 | 64 | 0.30 s | 0.18 s | **1.68x** | 2.04x | 6 s | 4 s | 1.65x | 0.22 G | 0.12 G | 1.9x | -12.541897370651 | 2e-14 |
  | 16 | 128 | 0.86 s | 0.50 s | **1.71x** | 2.07x | 18 s | 11 s | 1.66x | 0.59 G | 0.49 G | 1.2x | -12.541951443921 | 9e-15 |
  | 16 | 256 | 3.40 s | 1.79 s | **1.90x** | 2.02x | 70 s | 37 s | 1.91x | 1.15 G | 0.76 G | 1.5x | -12.541952156541 | 0 |
  | 32 | 64 | 0.75 s | 0.45 s | **1.66x** | 1.89x | 15 s | 9 s | 1.64x | 0.25 G | 0.19 G | 1.3x | -25.693800734431 | 7e-15 |
  | 32 | 128 | 2.40 s | 1.49 s | **1.61x** | 1.85x | 49 s | 29 s | 1.70x | 0.68 G | 0.52 G | 1.3x | -25.695279169232 | 1e-14 |
  | 32 | 256 | 10.17 s | 5.50 s | **1.85x** | 1.91x | 201 s | 111 s | 1.82x | 1.42 G | 0.88 G | 1.6x | -25.695370428286 | 3e-13 |

  Every point completed inside its budget; nothing was dropped.

  **The first ten sweeps at `N = 64`, `chi = 256` U(1)**, all three arms side by side, in
  seconds — the transient M64's part 2 named, seen on each path:

  | sweep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | tenet (prepared) | 146.95 | 73.61 | 67.87 | 64.48 | 61.29 | 51.36 | 43.73 | 27.10 | 14.93 | 7.55 |
  | tenet-sites | 38.66 | 23.37 | 21.58 | 20.67 | 19.49 | 16.60 | 14.73 | 10.05 | 6.20 | 4.43 |
  | YASTN | 5.00 | 4.03 | 3.89 | 3.80 | 3.66 | 3.55 | 3.30 | 2.85 | 2.54 | 2.29 |

  Against steady state (2.11 s YASTN, 2.70 s `tenet-sites`, 4.40 s prepared) the first
  sweep is 2.4x on YASTN, **14.3x on `tenet-sites`** and 33x on the prepared path.

  **The verdict, in three parts.**

  **1. On U(1) the site-tensor path is at parity with YASTN per steady sweep — 1.10–1.28x
  over the whole `N` x `chi` grid — and the prepared path costs 1.63–2.06x on top of it.**
  So roughly half of M64's 2.1–2.3x on U(1) is the prepared machinery and the other half
  is the shared contraction path. On fZ2 the site-tensor path is **1.61–1.90x**, and the
  prepared surcharge is the same 1.85–2.07x it is on U(1): about half of M64's 3.0–3.8x on
  fZ2 is **path-independent** and stays where it was — it is not the prepared machinery,
  and it is inside the contraction path both arms share, which is what #249 is set up to
  attribute operation by operation. Both readings hold flat in `chi`
  from 64 to 256, as M64's did.

  **2. The transient does not vanish.** At `N = 64`, `chi = 256` the site-tensor path's
  first sweep is 38.66 s against YASTN's 5.00 s, and the whole-run ratio is 2.93–3.19x at
  `N = 64` against a steady ratio of 1.10–1.28x — the same divergence M64 reported, only
  smaller. The prepared path is a 3.8x multiplier on the transient (146.95 s against
  38.66 s), not its cause: something that both paths share, in environment construction
  and `Env.update_`, also pays per bond-structure change. On fZ2, where the bond structure
  is fixed from sweep one, run ratio equals steady ratio on this arm too (1.64–1.91x
  against 1.61–1.90x), exactly as in M64.

  **3. Memory falls but does not reach YASTN's order on U(1).** Dropping the description
  takes peak RSS down by 2.4–2.9x on U(1) and 3.2–7.5x on fZ2 relative to the prepared
  arm, which puts `tenet-sites` at **1.2–2.2x YASTN** on every point except U(1) at
  `N = 64`, where it is **4.4–6.3x** (1.17–2.08 G against 0.20–0.48 G). So M64's 5–18x is
  mostly, but not entirely, the two per-bond caches: at `N = 64`, `chi = 64` the state is a
  few tens of MiB and this arm still peaks at 1.17 G with no `_prepared` and no `_cores`
  in play.

  **What is not claimed, and what is not measured.** Nothing here proposes a routing
  change: which path an MPO with an `EdgeTable` should take is a design decision with
  consequences beyond wall time — `heff2_families`, and with it `sweep_`'s
  family-resolved perturbative noise, exists only on the prepared path — and it is the
  next issue's, with these numbers attached. Nothing here re-opens the ab initio lane,
  where M39/#218 measured the prepared path reaching **0.97x** on C2 at `chi = 64`: this
  grid is two lattice models and says nothing about the case the prepared path was built
  for. No `src/` change was made and no third-arm ED anchor was run — the arm's operator
  is the prepared arm's own tensors, and the two arms' energies agreeing to 3e-13 is the
  stronger statement.

- **M67** — shipped: the lattice lane runs on the **site-tensor path**, and it has a name
  to spell it with (#251). `MPO.materialize()` returns the same operator holding its rank-4
  site tensors and no edge description, so `Env.heff2` — which routes on
  `self.h.edges is not None` and on nothing else — takes the site-tensor contraction.
  `docs/guide/models-and-sites.md` and `tenet.models`' package example teach it on every
  lattice model. **No default changed**, in `src/` or anywhere else.

  **What the decision rests on.** M64b's three-arm grid: on U(1) Heisenberg the site-tensor
  path is 1.10–1.28× YASTN per steady sweep across the whole `N × chi` grid, and the
  prepared, symbolic path adds 1.63–2.06× on top of that; on the fZ2 Hubbard chain the
  surcharge is the same 1.85–2.07×. A `D_w = 5` bond does not repay block2's per-bond
  cores, prepared operator and structure-keyed cache. M39's grid says the opposite at
  quantum-chemistry width: the prepared path is 0.97× the site-tensor one on C2 at
  `chi = 64` and is the only route that fits `K = 26` in memory at all. The reference
  policy above already assigns those two lanes to two references — 1-D interfaces and API
  shape to tenpy/YASTN, algorithm cores *at quantum-chemistry scale* to block2 — and this
  is that table applied to the engine rather than only to the structures.

  **#218's scope, restated.** #218 settled the engine as **one path for symbolic
  operators**: an MPO carrying an edge description gets the prepared term-family matvec at
  either cutoff, and later parallelism and accelerator work attaches there and nowhere
  else. That stands, unqualified. What #218 did not decide — and what it was read as
  deciding — is which *representation* a lattice Hamiltonian should be in before it reaches
  an engine at all. `from_terms` always yields a description, so "one path for symbolic
  operators" plus "the builders always build symbolic" silently became "one path", and a
  Heisenberg chain built the recommended way paid block2's machinery. M67 separates the two
  statements. The site-tensor path is therefore **the lattice lane's engine**, not a
  compatibility entry; it remains what an externally built MPO gets as well, because
  symbols cannot be recovered from a numeric `W` (#141), but that is no longer the only
  reason it exists.

  **The spelling: a method, not a `symbolic=` keyword on the builders.** The keyword was
  the other candidate and it is worse on three counts. *Nothing about the assembly
  changes* — the description is how the site tensors are produced, so `from_terms(...,
  symbolic=False)` would advertise a different build where there is only a different thing
  to keep; a method after the fact says exactly what happens, which is that the operator is
  built once and the caller decides what to hand the engine. *It would have to be answered
  three times*, on `from_terms`, `from_arrays` and `from_entries`, and kept consistent
  across them. And *the two builders cannot split the default between them*, which is the
  shape the issue floated: `from_arrays` is not an ab initio front end only —
  `docs/guide/models-and-sites.md` teaches it for lattice models too, because a
  `models.Site`'s `ops` is exactly the table it takes — so flipping `from_terms` while
  `from_arrays` kept the symbolic default would leave the guide's own recommended route on
  the path this milestone is moving away from, which is the acceptance criterion failing on
  the spelling meant to satisfy it. `materialize` is the word `MPO.sites`' own docstring
  already uses for realising a deferred site tensor, so it is the package's vocabulary and
  not a new coinage (#120/#185).

  **The default is unchanged, deliberately.** Flipping `from_terms`' default was defensible
  — its callers really are overwhelmingly the lattice lane — and it was rejected because it
  buys nothing the guide's one method call does not, while costing a breaking change to
  every example, tutorial, oracle and benchmark that reads `edges` off a `from_terms`
  operator, and while leaving the `from_arrays` half of the guide unfixed anyway. A default
  that is right for one builder's majority and wrong for the other's is a rule a caller has
  to memorise; "your model decides, and you say which at build time" is a rule they can
  predict from, and it is the same rule `cutoff=None`-versus-a-float already follows.

  **The grid, re-run on the recommended spelling** (`benchmarks/bench_vs_yastn.py --arm
  tenet-sites`, which now spells it `h.materialize()`), against M64b's own `tenet-sites`
  column, same machine:

  | model | N | chi | steady M67 | steady M64b | ratio | E | dE against M64b |
  |---|---|---|---|---|---|---|---|
  | Heisenberg U(1) | 32 | 64 | 0.60 s | 0.55 s | 1.09× | -13.997315618007 | 2e-13 |
  | Heisenberg U(1) | 32 | 256 | 0.95 s | 0.91 s | 1.05× | -13.997315618224 | 4e-13 |
  | Hubbard fZ2 | 16 | 64 | 0.31 s | 0.30 s | 1.02× | -12.541897370651 | 3e-13 |
  | Hubbard fZ2 | 16 | 256 | 3.44 s | 3.40 s | 1.01× | -12.541952156541 | 5e-13 |

  Realized bond dimensions match M64b's at every point and the energies agree to the last
  digit M64b printed; the `dE` column is that printing precision, not a disagreement. The
  1–9 % on the walls is run-to-run spread on a laptop, and it is below M64b's own
  arm-to-arm gaps by more than an order of magnitude.

  **What the lattice path gives up, measured rather than asserted.** `edge_blocks` is
  `None` there, so `Env.heff2_families` has no families to resolve and returns the single
  vector `H_eff aa`; M61 Stage C's perturbative noise becomes a one-vector mixer. Stage C's
  own lattice table, re-run on both representations (U(1) Heisenberg N=20 from the `D=1`
  Neel product seed, `chi=24`, noise on for the first five of eight sweeps — the symbolic
  columns reproduce Stage C's published table digit for digit):

  | sweep | none | wfn 1e-5 | pert 1e-5, symbolic | pert 1e-5, site tensors | pert 1e-4, symbolic | pert 1e-4, site tensors |
  |---|---|---|---|---|---|---|
  | 1 | -8.605141831 | -8.605141724 | **-8.650923250** | -8.605607266 | **-8.655765651** | -8.605104134 |
  | 2 | -8.681922862 | -8.681922679 | -8.682443457 | -8.681922983 | -8.682456851 | -8.681880637 |
  | 3 | -8.682473217 | -8.682473215 | -8.682473209 | -8.682473217 | -8.682473193 | -8.682473208 |
  | 4 | -8.682473226 | -8.682473225 | -8.682473209 | -8.682473226 | -8.682473193 | -8.682473226 |
  | 8 | -8.682473226 | -8.682473226 | -8.682473226 | -8.682473226 | -8.682473226 | -8.682473226 |
  | wall s | 7.2 / 2.8 | 4.7 / 2.3 | 6.9 | 2.5 | 7.5 | 3.2 |

  **What is lost is the first-sweep head start, and nothing else.** On the symbolic path
  the aimed perturbation puts sweep 1 at 5.1e-2 and 4.6e-2 below every other column, which
  is exactly what Stage C reported; on the site-tensor path that is gone — the single
  vector is `H_eff aa` itself, which is not a *resolution* of anything and enriches the
  bond no better than the plain sweep. By sweep 3 the two representations agree to 1e-8 and
  by sweep 4 both are at -8.682473226, the same converged energy every column reaches.
  Stage C had already measured the head start spent by sweep 3 on this model, "because at
  `chi=24` this model is not bond-limited and the plain sweep gets there on its own"; the
  re-run says the same thing from the other side. So the weaker mixer costs a sweep of head
  start and no accuracy here, and it is **not** a reason to keep a finite-range lattice
  Hamiltonian symbolic. It stays a real reason for a **bond-limited** run, which is what
  Stage C's N2 K=16 column measures and which lives in the lane that keeps its description
  anyway — and that is the honest form of "keep the symbolic route reachable, rather than
  default to it" the issue asked for.

  **The quantum-chemistry lane is untouched.** `from_arrays`, `from_terms` and
  `from_entries` all still return an operator carrying its description, so N2 at `K=16`
  takes the prepared path exactly as before; `tests/network/test_materialize.py` asserts
  that routing directly, by counting calls to the two module globals `Env.heff2` chooses
  between, rather than inferring it from `h.edges`. `cutoff=None` behaviour is unchanged in
  every respect: it selects which *operator* is built and is orthogonal to which
  representation is handed to the engine.

  **Public surface.** One addition, `MPO.materialize()`. No signature changed, no default
  changed, no keyword added, no existing test edited. `benchmarks/bench_vs_yastn.py`'s
  `tenet-sites` arm now spells `h.materialize()` where it spelled `MPO(h.sites)` — the same
  object, in the name the milestone gives it, so the grid above measures the recommended
  spelling and not a lookalike.
- **M65** — **shipped**: the U(1) bond-growth transient attributed, and the mechanism it
  names fixed at the root (#248, on M64b's site-tensor path). M64b left the transient
  path-independent and unexplained: at `N = 64`, `chi = 256` U(1) the site-tensor path's
  first sweep is 14x its own steady sweep with none of the prepared machinery running.
  `benchmarks/bench_sweep_transient.py` is M57's phase instrument re-aimed at that — the
  same decomposition **per sweep over the first twelve**, on `bench_vs_yastn.py`'s exact
  fixture (same seed, same `bond_charges` spec, `MPO(h.sites)`, `cutoff = 0.0`, `ncv = 3`,
  single-threaded BLAS, one process), plus two columns M57 does not have: the wall spent
  inside the `functools.cache`d **plan layer on a miss**, and the **bond churn**, how many
  of the `N - 1` internal bonds changed their `GradedSpace` since the previous sweep, split
  into "the sector set moved" and "only the degeneracies moved". The phase columns
  partition the wall; the plan-layer column is a *cross-cut* of them (it sits inside
  `heff2`, `update_` and `svd`), timed at the outermost plan frame only so no second is
  counted twice. Every point is run in both a wrapped and a `--plain` arm, and both walls
  are in the table: the instrument costs 12 % on U(1) and 1.6 % on fZ2. The named phases
  sum to the wall to within 0.7 % on every row of every table below.

  **U(1) Heisenberg `N = 64`, `chi = 256`, site-tensor path, before the change.** Seconds.

  | sweep | wall | plain | assemble | lanczos own | `heff2` | `svd` | `update_` | writeback | rest | plan layer | share | bonds moved | RSS |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 35.34 | 31.75 | 1.69 | 0.42 | 22.63 | 0.71 | 9.30 | 0.55 | 0.04 | **31.58** | **89 %** | 47 / 10 | 0.49 G |
  | 2 | 25.99 | 23.32 | 1.21 | 0.35 | 16.62 | 0.66 | 6.70 | 0.41 | 0.04 | **22.44** | **86 %** | 31 / 16 | 0.75 G |
  | 3 | 24.11 | 21.55 | 1.11 | 0.34 | 14.78 | 0.63 | 6.83 | 0.37 | 0.04 | **20.66** | **86 %** | 17 / 30 | 0.96 G |
  | 4 | 23.11 | 20.64 | 1.05 | 0.34 | 14.68 | 0.63 | 6.00 | 0.36 | 0.04 | **19.71** | **85 %** | 17 / 30 | 1.16 G |
  | 5 | 22.00 | 19.49 | 0.99 | 0.33 | 13.30 | 0.62 | 6.38 | 0.34 | 0.04 | **18.67** | **85 %** | 20 / 27 | 1.37 G |
  | 6 | 18.39 | 16.64 | 0.85 | 0.32 | 11.60 | 0.60 | 4.68 | 0.30 | 0.04 | **15.18** | **83 %** | 38 / 9 | 1.55 G |
  | 7 | 16.49 | 14.54 | 1.64 | 0.30 | 9.81 | 0.59 | 3.86 | 0.25 | 0.03 | **13.37** | **81 %** | 34 / 13 | 1.71 G |
  | 8 | 11.03 | 9.94 | 0.48 | 0.28 | 7.10 | 0.55 | 2.42 | 0.17 | 0.03 | **8.13** | **74 %** | 46 / 1 | 1.86 G |
  | 9 | 6.99 | 6.25 | 0.25 | 0.27 | 4.41 | 0.49 | 1.44 | 0.11 | 0.03 | **4.24** | **61 %** | 9 / 27 | 1.94 G |
  | 10 | 5.34 | 4.40 | 0.11 | 0.26 | 3.78 | 0.45 | 0.65 | 0.06 | 0.03 | **2.63** | **49 %** | 0 / 6 | 1.99 G |
  | 11 | 3.42 | 2.95 | 0.07 | 0.26 | 2.16 | 0.44 | 0.41 | 0.06 | 0.03 | **0.77** | **22 %** | 0 / 2 | 2.02 G |
  | 12 | 2.97 | 2.53 | 0.06 | 0.25 | 1.89 | 0.43 | 0.25 | 0.05 | 0.03 | **0.38** | **13 %** | 0 / 0 | 2.03 G |

  **fZ2 Hubbard `N = 32`, `chi = 256`, the control — no transient in M64b.** Seconds.

  | sweep | wall | plain | assemble | lanczos own | `heff2` | `svd` | `update_` | writeback | rest | plan layer | share | bonds moved | RSS |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 9.90 | 9.84 | 0.06 | 0.53 | 6.30 | 2.54 | 0.38 | 0.03 | 0.06 | **0.18** | **2 %** | 0 / 17 | 1.10 G |
  | 2 | 9.96 | 9.87 | 0.06 | 0.50 | 6.47 | 2.48 | 0.36 | 0.04 | 0.05 | **0.14** | **1 %** | 0 / 17 | 1.24 G |
  | 3 | 9.87 | 9.80 | 0.05 | 0.51 | 6.44 | 2.43 | 0.35 | 0.04 | 0.06 | **0.08** | **1 %** | 0 / 5 | 1.28 G |
  | 4 | 10.01 | 9.85 | 0.05 | 0.52 | 6.51 | 2.46 | 0.36 | 0.04 | 0.07 | **0.07** | **1 %** | 0 / 11 | 1.30 G |
  | 5 | 9.85 | 9.67 | 0.05 | 0.52 | 6.38 | 2.46 | 0.34 | 0.04 | 0.06 | **0.06** | **1 %** | 0 / 12 | 1.32 G |
  | 6 | 9.86 | 9.77 | 0.05 | 0.51 | 6.43 | 2.44 | 0.35 | 0.04 | 0.05 | **0.07** | **1 %** | 0 / 7 | 1.35 G |
  | 7 | 10.03 | 9.94 | 0.05 | 0.52 | 6.54 | 2.47 | 0.34 | 0.04 | 0.06 | **0.06** | **1 %** | 0 / 7 | 1.35 G |
  | 8 | 10.00 | 9.91 | 0.05 | 0.54 | 6.53 | 2.43 | 0.35 | 0.04 | 0.06 | **0.07** | **1 %** | 0 / 5 | 1.35 G |
  | 9 | 9.98 | 9.73 | 0.05 | 0.54 | 6.50 | 2.43 | 0.36 | 0.04 | 0.06 | **0.07** | **1 %** | 0 / 10 | 1.36 G |
  | 10 | 10.06 | 9.82 | 0.05 | 0.55 | 6.57 | 2.44 | 0.35 | 0.04 | 0.06 | **0.07** | **1 %** | 0 / 13 | 1.36 G |
  | 11 | 10.06 | 9.88 | 0.05 | 0.55 | 6.53 | 2.46 | 0.36 | 0.04 | 0.07 | **0.09** | **1 %** | 0 / 16 | 1.36 G |
  | 12 | 10.19 | 9.82 | 0.06 | 0.56 | 6.60 | 2.48 | 0.40 | 0.04 | 0.06 | **0.13** | **1 %** | 0 / 13 | 1.39 G |

  **The mechanism, and it is one mechanism.** The plan layer is **89 % of the first sweep**
  and it decays with the churn column and with nothing else: 89 % → 13 % as the bond spaces
  stop moving, while every other phase is flat or falls with it. On fZ2, where the bond
  carries two sectors and the *sector set* never moves, the same column is **0.6–1.8 % and
  flat** — the degeneracies move there too (5–17 bonds a sweep), so what the cost scales
  with is churn **times blocks per structure**, not churn alone. Inside the column, on the
  first sweep: `repartition_plan` 19.7 s, `TensorStructure`'s block enumeration
  (`_block_order`) 5.9 s, `map_layout` 3.3 s, `_block_shape_table` 1.4 s,
  `permutation_plan` 0.7 s, `contraction_plan` 0.02 s — attributed to the *outermost* cache,
  so `repartition_plan`'s figure contains the `permutation_plan`/`bend_plan` chain it calls.
  6 349 `_block_order` misses on sweep 1, 1 on sweep 12.

  **Why they miss.** A `GradedSpace` hashes its degeneracies, a `Leg` hashes its space, a
  `TensorStructure` hashes its legs — and every cache in the plan layer is keyed on a
  `TensorStructure`. So a bond whose degeneracies moved is a new key in all of them, and a
  truncating SVD moves them at every bond of every sweep until the state settles. What the
  plans *contain*, though, is block indices and coefficients: `block_order` is a pure
  function of the legs' **sectors, sides and duals**, and so are the terms of
  `permutation_plan`, `bend_plan` and `repartition_plan`. The keys were finer than the
  results.

  **The change** is that split, in the shared functions and with no runtime dispatch on
  anything. `structure._pattern(s)` is `s` with every degeneracy set to 1 — two structures
  share a pattern exactly when they have the same `block_order` — and `_block_order`,
  `_index_map` and `_axis_sectors_table` delegate to it, so a degeneracy move now costs one
  dict lookup instead of an enumeration over `prod(len(leg.sectors))` assignments.
  `permutation_plan`, `bend_plan` and `repartition_plan` keep their own cache (a repeat call
  still returns the same object) but compute their body one level down, on the pattern; the
  entry then holds nothing but `new_structure`, which is the caller's own legs reordered
  with `side` and `dual` flipped on each leg that crossed. `_block_shape_table` is the one
  table that reads a degeneracy and it stays keyed on the structure itself. `map_layout` and
  `contraction_plan` are untouched — the first genuinely depends on degeneracies (its band
  offsets are extents), the second measured at 0.02 s.

  **After.** Same fixture, same session, same instrument.

  | sweep | wall | plain | assemble | lanczos own | `heff2` | `svd` | `update_` | writeback | rest | plan layer | share | bonds moved | RSS |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 33.95 | 30.50 | 1.61 | 0.41 | 21.98 | 0.64 | 8.80 | 0.47 | 0.04 | **30.32** | **89 %** | 47 / 10 | 0.50 G |
  | 2 | 21.91 | 19.56 | 0.95 | 0.34 | 13.79 | 0.51 | 6.04 | 0.25 | 0.04 | **18.54** | **85 %** | 31 / 16 | 0.72 G |
  | 3 | 12.36 | 10.98 | 0.49 | 0.32 | 7.44 | 0.41 | 3.52 | 0.14 | 0.04 | **9.13** | **74 %** | 17 / 30 | 0.84 G |
  | 4 | 11.94 | 10.40 | 0.44 | 0.32 | 7.35 | 0.42 | 3.23 | 0.14 | 0.04 | **8.73** | **73 %** | 17 / 30 | 0.96 G |
  | 5 | 12.41 | 10.85 | 0.47 | 0.32 | 7.67 | 0.45 | 3.32 | 0.16 | 0.04 | **9.23** | **74 %** | 20 / 27 | 1.07 G |
  | 6 | 14.66 | 13.15 | 0.63 | 0.30 | 9.01 | 0.51 | 3.97 | 0.20 | 0.04 | **11.58** | **79 %** | 38 / 9 | 1.23 G |
  | 7 | 13.86 | 12.15 | 0.57 | 0.29 | 9.03 | 0.52 | 3.25 | 0.17 | 0.03 | **10.88** | **79 %** | 34 / 13 | 1.38 G |
  | 8 | 8.93 | 8.00 | 0.41 | 0.27 | 6.13 | 0.50 | 1.47 | 0.12 | 0.03 | **6.15** | **69 %** | 46 / 1 | 1.50 G |
  | 9 | 3.39 | 2.95 | 0.08 | 0.25 | 2.18 | 0.44 | 0.36 | 0.05 | 0.03 | **0.77** | **23 %** | 9 / 27 | 1.55 G |
  | 10 | 3.07 | 2.60 | 0.06 | 0.25 | 1.97 | 0.43 | 0.28 | 0.05 | 0.03 | **0.44** | **14 %** | 0 / 6 | 1.59 G |
  | 11 | 2.98 | 2.55 | 0.06 | 0.25 | 1.90 | 0.43 | 0.26 | 0.05 | 0.03 | **0.39** | **13 %** | 0 / 2 | 1.60 G |
  | 12 | 2.87 | 2.48 | 0.06 | 0.24 | 1.82 | 0.43 | 0.25 | 0.05 | 0.03 | **0.35** | **12 %** | 0 / 0 | 1.60 G |

  **Whole run**, `benchmarks/bench_vs_yastn.py --arm tenet-sites`, both arms re-run in this
  session against the same code M64b measured (222.2 s / 2.10 G against M64b's 230 s /
  2.08 G, i.e. the session-to-session spread is about 4 %). YASTN's column is M64b's.

  | fixture | run before | run after | YASTN | steady before | steady after | RSS before | RSS after | energy |
  |---|---|---|---|---|---|---|---|---|
  | U(1) Heisenberg `N = 64`, `chi = 256`, 30 sweeps | 222.2 s | **176.8 s** | 77 s | 2.52 s | 2.50 s | 2.10 G | **1.67 G** | -28.17542485974315, both |
  | fZ2 Hubbard `N = 32`, `chi = 256`, 20 sweeps | 191.9 s | 191.0 s | 111 s | 9.55 s | 9.48 s | 1.46 G | 1.42 G | -25.69537042828647, both |

  The U(1) whole-run ratio against YASTN moves **2.89x → 2.30x** while the steady ratio
  stays where M64b put it (1.28x → 1.18x here); the realized bond dimensions and the
  energies are identical before and after, to the last digit printed.

  **What moved and what did not.** The twelve-sweep wall falls **174.0 s → 126.2 s (−27 %)**
  on the plain arm, and the fall is entirely in sweeps 3–9 — sweep 1 is unchanged
  (31.7 → 30.5 s), which is the honest reading of the fix: on the first sweep every pattern
  is genuinely new, and a cache keyed on the right thing still has to build it once. What
  the change removes is the **re-**building, which is why the effect appears exactly where
  the churn is degeneracy-only (sweeps 3–5, 30 of 63 bonds: 24.1 → 12.4 s) and not where the
  sector sets are still moving (sweeps 6–8). Peak RSS at sweep 12 falls **2.03 G → 1.60 G
  (−21 %)**: the per-structure `block_order` tuples, index maps and plan terms are now shared
  across every structure with the same pattern, and what remains per structure is a small
  record. Gap B's U(1) `N = 64` residue is therefore the same mechanism as Gap A seen in
  memory, as M64b guessed, and it is reduced by the same change with no cache-budget touched.
  fZ2 is unmoved on both counts (117.9 → 114.7 s over twelve sweeps, RSS 1.39 → 1.41 G),
  which is what a fix aimed at sector-rich bonds should do to a two-sector one.

  **What is not claimed.** The transient is *reduced, not removed*: the first sweep still
  costs 12x the steady sweep, and the plan layer is still 89 % of it. The remaining cost is
  new patterns rather than re-keyed old ones, so closing it further is a different question
  — whether `repartition_plan`'s transpose-bend-transpose composition has to be O(blocks)
  per new pattern at all — and it needs its own attribution. Nothing here touches
  `cutoff=None`, the prepared path or the family-resolved matvec: the change is one level
  below all of them, in the cache keys, and the energies are bit-identical to the digit
  before and after at both fixtures. No cache budget was changed, no new dependency, no
  public surface moved, and #218's single-path engine is untouched — there is no dispatch
  on churn here, and there is none anywhere else either.

- **M66** — **measured, not shipped**: the site-tensor path's per-sweep factor against
  YASTN, attributed operation by operation inside one steady `_heff2_full` and one steady
  `Env.update_`. No `src/` change. The headline is a correction to the question: **the fZ2
  excess over U(1) is not the fermionic sign mechanism.** The braiding costs 1.17×; the
  rest of it is a provider-independent cost in the shared contraction path that shows on
  the Hubbard block geometry and hides on the Heisenberg one.

  **The two fixtures**, `bench_vs_yastn.py`'s, on the site-tensor path (`h.materialize()`),
  after two sweeps so the bond spaces are the ones `chi` names, mean of five calls, one
  process, single-threaded BLAS. Every number below is one call, in milliseconds.

  | | tenet fZ2 Hubbard N=16 | tenet Z2 control | tenet U(1) Heisenberg N=32 |
  |---|---|---|---|
  | `_heff2_full` wall | 11.95 | 10.18 | 5.12 |
  | `to_matrices` (block → sector matrix) | 5.99 | 6.04 | 1.09 |
  |  · of which strided reshape | 3.04 | 3.14 | 0.45 |
  |  · of which `concatenate` | 2.80 | 2.66 | 0.30 |
  | backend `matmul` | 2.75 | 2.61 | 0.20 |
  | **coefficient pass** (R-coefficient / twist / Frobenius–Schur) | **1.62** | **0.02** | **0.09** |
  | `from_matrices` (sector matrix → blocks) | 0.22 | 0.22 | 1.57 |
  | axis transposes (NumPy views, no copy) | 0.17 | 0.16 | 0.99 |
  | plan lookup + composability check | 0.11 | 0.10 | 0.08 |
  | accounted | 10.85 (91 %) | 9.14 (90 %) | 4.03 (79 %) |
  | residual (`einsum` parsing, object construction, the per-term loops) | 1.10 | 1.04 | 1.09 |

  The coefficient row counts the accumulation loop that applies the coefficient, so the
  two control columns are not zero: they are that loop running with **no** coefficient
  term at all (0 of 316 on Z2, 0 of 2406 on U(1), against 60 of 316 on fZ2), which is
  the loop's own Python and nothing else.

  | | tenet fZ2 | tenet Z2 control | tenet U(1) |
  |---|---|---|---|
  | `Env.update_(to='last')` wall | 1.30 | 1.31 | 1.88 |
  | `to_matrices` | 0.59 | 0.55 | 0.49 |
  | backend `matmul` | 0.28 | 0.32 | 0.09 |
  | **coefficient pass** | **0.075** | **0.005** | **0.030** |
  | `from_matrices` | 0.07 | 0.08 | 0.54 |
  | axis transposes (views) | 0.05 | 0.05 | 0.31 |
  | accounted | 1.10 (85 %) | 1.04 (80 %) | 1.52 (81 %) |

  The same two calls on YASTN, wrapped the same way in a scratchpad harness (its
  `tensordot` at the default `fuse_contracted` policy is: build the merge meta, one
  transpose-and-merge per operand into the flat data buffer, one `backend.dot` over the
  meta list, and — because the outgoing legs are not merged — **no unmerge**):

  | | YASTN fZ2 Hubbard N=16 | YASTN U(1) Heisenberg N=32 |
  |---|---|---|
  | `Heff2` wall | 5.04 | 5.34 |
  | transpose-and-merge (8 merges) | 2.27 | 3.02 |
  | `backend.dot` | 2.01 (56 gemms) | 1.97 (457 gemms) |
  | merge/dot meta | 0.04 | 0.04 |
  | accounted | 4.32 (86 %) | 5.03 (94 %) |
  | | | |
  | `update_env_(to='last')` wall | 1.04 | 2.24 |
  | transpose-and-merge (6 merges) | 0.38 | 1.20 |
  | `backend.dot` | 0.42 (20 gemms) | 0.79 (182 gemms) |
  | accounted | 0.81 (78 %) | 2.02 (90 %) |

  **The ratios, operation against operation.** `_heff2_full` / `Heff2` is **2.37× on fZ2**
  and **0.96× on U(1)** — tenet is *faster* than YASTN on the U(1) fixture's matvec.
  `update_` is 1.24× and 0.84×. Both calls over-predict M64b's 1.71× / 1.12× steady sweep,
  as they must: a sweep also runs the SVD and the Lanczos recurrence, and on N=16 half the
  bonds are far below `chi`.

  **The Z2 control is what settles the attribution.** The middle column above is the same
  Hubbard fixture graded by the *bosonic* `Z2` provider: identical block shapes, identical
  block counts, identical contraction schedule, identical flop count — every coefficient 1.
  It is not the Hubbard model any more, and it is not meant to be; it is the only way to
  price the braiding without also changing the geometry. `_heff2_full` goes 11.95 → 10.18,
  so **the whole fermionic surcharge is 1.77 ms, 1.17×**, and it matches the measured
  coefficient pass (1.62 ms) within the run-to-run spread. On `Env.update_` the surcharge
  is 1.30 → 1.31, i.e. **zero within noise**. Meanwhile the bosonic control is still 2.02×
  YASTN. So of the "1.5× fZ2 excess over U(1)" this issue was rescoped onto, the fermionic
  mechanism accounts for 1.17× and the remaining 2.0× is provider-independent.

  **What the remaining 2.0× is, and why U(1) does not show it.** It is the block-assembly
  round trip. tenet lowers every pairwise contraction to `to_matrices` → `matmul` →
  `from_matrices`; the assembly reshapes each block out of a transposed NumPy *view* (a
  strided copy) and then concatenates the pieces (a second copy). YASTN merges only the
  *contracted* legs and lets the output axis order fall out of `tensordot`, so many of its
  merges are contiguous and it never unmerges at all. On the Hubbard geometry — two coupled
  sectors, sixteen blocks of ~98 k doubles — that is bandwidth, and it is 5.99 ms against
  YASTN's 2.27 ms. On the Heisenberg geometry — 138 blocks of ~2.4 k doubles, forty times
  smaller — both libraries are per-block-Python-bound instead
  (YASTN issues 457 `gemm` calls where tenet issues 35), and tenet comes out ahead. **The
  "fZ2 excess over U(1)" is therefore mostly a block-geometry effect measured on two models
  that differ in more than their grading**, not a fermionic one.

  The bandwidth claim is measured, not inferred: on one block of this fixture's size a
  contiguous NumPy copy is 9.6 µs, a mild transposed copy 14–17 µs and a reversed one
  85 µs, while an identity permutation costs nothing at all because no copy happens.

  **The four candidates, each with its number.**

  - *Contraction order and intermediate sizes* — **refuted**. The two chains are the same
    four pairwise contractions with the same intermediates: `[128, 4, 4, 6, 128]` three
    times then `[128, 4, 4, 128]`, at 16/16/16/8 blocks on fZ2, and
    `[128, 2, 2, 5, 128]` three times then `[128, 2, 2, 128]` at 138/138/138/47 on U(1) —
    identical on both sides, same flop count, no intermediate formed by one and not the
    other. They differ only in the intermediates' *axis order*, which is free for YASTN
    and is what makes its merges cheaper.
  - *The coefficient pass* — **confirmed, and it is an extra pass over memory rather than
    a per-block scalar multiply that rides along.** 1.62 ms of 11.95 (13.6 %) in
    `_heff2_full`, 0.075 ms of 1.30 (5.8 %) in `update_`. Sixty of the call's 316
    permutation and bend terms carry a coefficient, and each one materialises a fresh
    array from a transposed view at ~26 µs per block against ~10 µs for a contiguous copy
    of the same block. The Z2 control's 1.17× is the same number reached from the other
    side.
  - *`from_dense` / `to_dense` / `concatenate` on the hot path* — **`from_dense`/`to_dense`
    refuted: no call site.** The only `from_dense` in `env.py` is `_drop`'s ones-cap and
    it is reached from `_cores2` alone; neither `_heff2_full` nor `update_`'s site-tensor
    branch, nor anything under `tensordot`, densifies at all. `concatenate` **confirmed**, at 2.80 ms of 11.95 (23 %), inside
    `to_matrices`. A NumPy-only `zeros`-plus-slice-assignment variant of exactly the same
    assembly measures 5.18 ms against the concatenating one's 5.99, so the
    concatenate-shaped design costs **0.8 ms, 7 % of the call** — and the other 5.2 ms is
    the strided copy itself, which is not avoidable by changing how the pieces are joined.
    That 0.8 ms is the price of `map_view.py`'s stated rule (no zeros, no scatter, no
    in-place writes, so the module stays JAX-traceable), and it is now priced rather than
    assumed.
  - *`_build2` / the refold* — **not applicable on this path**, as the rescope expected.
    `Env.heff2` routes on `self.h.edges is not None` and `MPO.materialize()` drops the
    description, so `_prepare2`, `_build2` and `_cores2` are never entered: zero calls
    measured in either fixture.

  **Why this stops here.** The issue's own rule was to fix only a single mechanism carrying
  most of the excess and reachable locally in the shared contraction path. The fermionic
  mechanism is neither: at 1.17× it is not most of anything, and it is spread over sixty
  blocks in two operations (`transpose`'s permutation terms and `repartition`'s bend
  terms), each paying a genuine per-block coefficient because that is what a categorical
  engine is — YASTN pays no sign in `Heff2` at all, having folded the Jordan-Wigner string
  into its MPO once at construction through `swap_gate`. **That difference is the design,
  not a defect, and 1.17× per matvec is what it costs.** The larger, provider-independent
  2.0× has a named mechanism and a measured size but is not this issue's target and is not
  local — 5.2 of its 5.99 ms is the strided copy, which only a persisted matrix layout
  between contractions would remove, and `ops/map.py` records that as already prototyped
  and refused for a zero cache-hit rate in real `tensordot` chains.

  **What is not measured.** Every number is NumPy on one machine at `chi=128`; the fZ2
  point is N=16 and the U(1) point N=32, as the rescope fixed them, so nothing here speaks
  to the χ- or N-scaling that M64 already established as flat. The instrumentation replaces
  `transpose`, `repartition`, `compose` and `to_matrices` with probe-carrying clones in a
  scratchpad harness; it is deliberately not committed, because an in-repo copy of four
  function bodies would rot at the first change to any of them, and the Z2 control run
  reproduces the only claim that needs no probe at all. The tree measured is M67's, before
  M65 landed; M65 touches how a structure records its pattern, not how many bytes a block
  assembly moves, so the steady-state split above is not waiting on it.

- **M68** — shipped: the three builders return **site tensors by default**, and the
  symbolic description is the advanced option, asked for by keyword (#255). `from_terms`,
  `from_arrays` and `from_entries` each gained one keyword-only `symbolic: bool = False`;
  at the default they hand back exactly the object `MPO.materialize()` returns, so
  `Env.heff2` — which routes on `self.h.edges is not None` and on nothing else — takes the
  site-tensor contraction. `symbolic=True` keeps the finite-state-machine description and
  the prepared, term-family matvec with it. `MPO.materialize()` is unchanged and is now
  the route *back*, for an operator built symbolic that a caller later wants on the other
  path.

  **Why the default moved, when M67 had just argued it should not.** M67's spelling —
  `builder(...).materialize()` — put the lattice lane on the right path only for a caller
  who writes the method. `from_terms(...)` handed straight to `dmrg_`, which is what every
  first script does, still paid 1.63–2.06× per steady sweep on U(1) Heisenberg, 1.85–2.07×
  on the fZ2 Hubbard chain, the 3.8× bond-growth transient M64b measured, and the per-bond
  cache memory. The architecture this repository states is *simple Hamiltonians on
  MPSKit/YASTN-shaped structures by default, block2 machinery as the advanced option*, and
  a default that only the well-read caller escapes is that statement inverted. M67's own
  objection does not survive the flip being made on **all three** builders: it rejected
  flipping `from_terms` alone because `docs/guide/models-and-sites.md` teaches
  `from_arrays` for lattice models too (a `models.Site`'s `ops` is exactly the table it
  takes), which would have left the guide's recommended route on the prepared path. Three
  flipped defaults have no such hole, and the quantum-chemistry callers — the QC
  benchmarks and the QC tests — now say `symbolic=True` in the open, which is what "the
  advanced option, in the open" means.

  **#218's scope, stated one more time and now exactly.** #218 settled the engine as one
  path **for symbolic operators, which a caller asks for**: an MPO carrying an edge
  description gets the prepared term-family matvec at either cutoff, and later parallelism
  and accelerator work attaches there and nowhere else. Nothing about that changes. What
  changes is that being symbolic is no longer something an operator *is* by accident of
  which builder produced it — it is something the caller requests. "One path for symbolic
  operators" plus "the builders always build symbolic" had silently become "one path"; M67
  separated the statements and M68 makes the second one false by default.

  **`symbolic` and `cutoff` are independent, and both are build-time.** `cutoff` decides
  whether the operator is *compressed*; `symbolic` decides whether the description is
  *kept*. `cutoff=None` with the default therefore yields exact, uncompressed site tensors
  — the plain-NumPy path Milestone 16 first measured, and on a finite-range lattice model
  the minimal bond anyway. Nothing dispatches at run time: no bond-width threshold, no
  `chi` threshold, no probe, no `path=` keyword.

  **The name.** `symbolic=` is the word `Env.heff2`'s own Notes, `MPO.edges`' docstring and
  M39/M64b/M67 already use for the representation ("the prepared, symbolic path", "the
  symbolic layer"), so it is the package's vocabulary rather than a coinage (#120/#185).
  `prepared=` names the engine instead of the operator and would have to be re-explained
  the moment a second consumer of the description appears; `keep_edges=`/`fsm=` name the
  implementation. The docstring explains it in one sentence — *keep the
  finite-state-machine description, so `Env.heff2` runs the term-family matvec* — which is
  the test a keyword has to pass to be worth three signatures.

  **Measured: the bare builder is the fast one.** `benchmarks/bench_vs_yastn.py --arm
  tenet`, unchanged (no `materialize()` in the script), against M67's `tenet-sites` column,
  and against `origin/main`'s own `--arm tenet-sites` run on this machine:

  | model | N | chi | M68 `--arm tenet` | main `--arm tenet-sites` | M67 | E |
  |---|---|---|---|---|---|---|
  | Heisenberg U(1) | 32 | 64 | 0.52 s | — | 0.60 s | -13.997315618007251 |
  | Heisenberg U(1) | 32 | 256 | 0.99 s | — | 0.95 s | -13.997315618224434 |
  | Hubbard fZ2 | 16 | 64 | 0.30 s | 0.30 s | 0.31 s | -12.541840348005870 |
  | Hubbard fZ2 | 16 | 256 | 3.14 s | 3.03 s | 3.44 s | -12.541411654258997 |

  The Heisenberg energies are M67's to every digit it printed. The two Hubbard energies are
  **bit-identical** to `origin/main`'s `tenet-sites` arm run here — the control that matters,
  because M67's printed Hubbard energies (-12.541897370651, -12.541952156541) were produced
  on another machine and are not reproduced by main's own arm on this one either. Walls are
  within run-to-run spread on a laptop.

  **Measured: the keyword is the old path, bit for bit.** N=12 U(1) Heisenberg,
  `cutoff=None`, `chi=64`, five sweeps, seed 0: `origin/main`'s default build and this
  branch's `symbolic=True` build agree on `float.hex` for the converged energy
  (`-0x1.4918034f478c0p+2`) and for every sweep in the history.

  **The quantum-chemistry lane, re-checked under the keyword.** N2 `K=16` through
  `from_arrays(..., symbolic=True)` keeps its description (`edges`, `edge_blocks(0)`) and
  routes every matvec to the prepared branch — instrumented on the two module globals
  `Env.heff2` chooses between: `{prepared: 186, sites: 0}` over one sweep at `chi=16`, first
  sweep energy -27.480379787. C2 `K=26` still completes through
  `bench_pinned_mpo.py --dmrg pinned` with the keyword added to that script: build 41.0 s
  at 6.12 GiB, sweep 2.7 s, first-sweep energy -11.823302501 at `chi=16` (M67: 43.2 s,
  6.05 GiB, 4.9 s). M39's
  corner-exactness property and the deferred-instantiation instrumentation pass unchanged
  under it.

  **What the default gives up, and where that is written down.** A materialized operator has
  no term families, so `Env.heff2_families` returns the single vector `H_eff aa` and
  `sweep_`'s `noise_type="perturbative"` falls back to the one-vector mixer. M67 measured
  that cost on the lattice models: one sweep of head start, no accuracy, the same converged
  -8.682473226. The perturbative-noise suite therefore keeps its subject by passing
  `symbolic=True`, and `sweep_`'s own docstring now says which operators get the
  family-resolved mixer.

  **Public surface.** Three keyword-only additions, nothing else changed in any signature;
  `MPO.materialize` unchanged. Docstrings: the three builders state the rule a caller
  predicts from their own model, `Env.heff2`'s Notes lead with the default path as the
  lattice lane's engine, `Env.heff2_families` states the default fallback, and
  `MPO.materialize`'s Notes drop the argument against a `symbolic=` keyword that this
  milestone overrides. `docs/guide/models-and-sites.md` spells every lattice model as a bare
  builder and keeps a short "going the other way" section for `materialize()`; the decision
  table carries the keyword on the quantum-chemistry row.


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