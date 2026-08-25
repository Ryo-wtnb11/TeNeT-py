# Symmetric tensors from their blocks

A symmetric tensor is not a dense array with zeros in it. It is a list of *blocks*, one
per way its legs can fuse to a common sector, and the entries a symmetry forbids have no
block to live in — they are absent, not stored as zero.

This tutorial builds such tensors by hand. It starts from sectors, puts them into spaces
and legs, asks the structure which blocks are allowed, and only then writes numbers into
them with [from_blocks][tenet.SymmetricTensor.from_blocks]. Both worked examples are
operators a Hamiltonian is made of: the U(1) Heisenberg bond, and the SU(3) exchange.

[Tensors, legs and spaces](../guide/tensors-legs-spaces.md) is the reference for the four
objects used here; this page is the construction, at length.

## Sectors

A sector labels an irreducible representation of the symmetry. For U(1) that label is a
charge $q \in \mathbb{Z}$, and the convention used throughout the spin-1/2 examples is
$q = 2 S^z$, so a spin doublet is $\{-1, +1\}$:

```python
>>> from tenet.symmetry import U1, U1Sector
>>> DOWN, UP = U1Sector(-1), U1Sector(1)
>>> U1.fusion(DOWN, UP)               # charges add
(U1Sector(charge=0),)

```

Fusion is what the symmetry contributes: $q_1 \otimes q_2 = q_1 + q_2$, one outcome, so
U(1) is Abelian and multiplicity-free. `U1.unit` is the unit sector, charge 0.

## Spaces and legs

A [GradedSpace][tenet.GradedSpace] says which sectors appear and with what degeneracy,
$V = \bigoplus_a \mathbb{C}^{m_a} \otimes V_a$. One spin-1/2 site is one copy of each of
the two charges:

```python
>>> from tenet import GradedSpace
>>> PHYS = GradedSpace.new(U1, {DOWN: 1, UP: 1})
>>> tuple(PHYS)
(U1Sector(charge=-1), U1Sector(charge=1))
>>> PHYS.dim, PHYS.reduced_dim
(2, 2)

```

A [Leg][tenet.Leg] attaches that space to one tensor axis and gives it a direction: `OUT`
for the codomain (what the operator emits, a ket index), `IN` for the domain (what it
absorbs, a bra index). An operator on one site is one `OUT` leg and one `IN` leg:

```python
>>> from tenet import IN, OUT, Leg
>>> legs = (Leg(PHYS, OUT, name="p"), Leg(PHYS, IN, name="p*"))

```

## Which blocks exist

[TensorStructure][tenet.TensorStructure] is the tensor's static half — the legs, and
everything derivable from them without any numbers. Its
[block_order][tenet.TensorStructure.block_order] enumerates every allowed block, in the
canonical order the stored blocks follow:

```python
>>> from tenet import TensorStructure
>>> structure = TensorStructure(legs)
>>> structure.num_blocks
2

```

Two blocks, not four. A rank-2 operator on this site has $2 \times 2 = 4$ dense entries,
and the two that change $S^z$ are not part of any block.

## Reading a key

Each entry of `block_order` is a [FusionBlockKey][tenet.FusionBlockKey]: an *output tree*
over the `OUT` legs, an *input tree* over the `IN` legs, and the coupled sector they
share. Invariance is exactly that shared sector — what the tensor emits must carry the
charge it absorbed:

```python
>>> key = structure.block_order[0]
>>> key.output_tree.uncoupled, key.input_tree.uncoupled
((U1Sector(charge=-1),), (U1Sector(charge=-1),))
>>> key.coupled
U1Sector(charge=-1)
>>> structure.block_shape(key)
(1, 1)

```

Each tree lists its uncoupled sectors in public axis order, restricted to its side, so
neither tuple alone says which axis a sector sits on.
[axis_sectors][tenet.TensorStructure.axis_sectors] reassembles them, one sector per
public axis:

```python
>>> structure.axis_sectors(key)
(U1Sector(charge=-1), U1Sector(charge=-1))

```

The sectors in a tree are already *dual-resolved*: a leg with `dual=True` carries $V^{*}$,
and what it contributes to the fusion is the conjugate label — for U(1), the negated
charge:

```python
>>> Leg(PHYS, OUT).fused_sector(UP)
U1Sector(charge=1)
>>> Leg(PHYS, OUT, dual=True).fused_sector(UP)
U1Sector(charge=-1)

```

`axis_sectors` undoes that again, so it always reports the sector as the *space* labels
it, whichever way the leg points.

The block's shape is degeneracies only — one $m_a$ per axis, not the dense dimension.
Here every sector has degeneracy 1, so every block is $1 \times 1$: a single number.

## Writing an operator: `from_blocks`

$S^z$ is diagonal, $-1/2$ on the down sector and $+1/2$ on the up one. That sentence is
the whole construction:

```python
>>> import numpy as np
>>> from tenet import SymmetricTensor
>>> value = {DOWN: -0.5, UP: 0.5}
>>> sz = SymmetricTensor.from_blocks(
...     legs,
...     {k: np.full(structure.block_shape(k), value[k.coupled]) for k in structure.block_order},
... )
>>> sz.to_dense()
array([[-0.5,  0. ],
       [ 0. ,  0.5]])

```

Nothing was said about the off-diagonal zeros. They are not zeros that were written; they
are entries with no block.

### A forbidden key is refused

$S^{+}$ maps down to up. As a key that reads "output up, input down" — two different
coupled sectors, so no such key exists on these legs, and naming one is an error rather
than a tensor that quietly breaks the symmetry:

```python
>>> from tenet import FusionBlockKey, FusionTree
>>> raising = FusionBlockKey(FusionTree((UP,), (), (), UP), FusionTree((DOWN,), (), (), DOWN))
>>> raising in structure.block_order
False
>>> try:
...     SymmetricTensor.from_blocks(legs, {raising: np.ones((1, 1))})
... except KeyError as exc:
...     print(exc.args[0].split(",")[0])
1 key(s) foreign to this structure

```

This refusal is the symmetry check. A wrong grading — a bond space with the wrong charges
on it, an operator that does not conserve what you thought — surfaces here, as a key that
does not exist, instead of as a silent projection onto some other operator.
$S^{+}$ is not U(1)-invariant on these legs; it becomes a legal tensor only with an
auxiliary leg carrying the charge it moves, which is what an MPO bond is.

### Absent keys are zero

`from_blocks` fills every key you leave out with zeros, taking the dtype and backend from
the blocks you did name:

```python
>>> up_only = SymmetricTensor.from_blocks(legs, {structure.block_order[1]: np.full((1, 1), 0.5)})
>>> up_only.blocks
(array([[0.]]), array([[0.5]]))

```

So you name the blocks you have an opinion about. A mistyped key is an *unknown* key, not
a missing one, so it still raises. For an all-zero tensor there is nothing to name at all,
and [zeros][tenet.SymmetricTensor.zeros] is that tensor;
[random][tenet.SymmetricTensor.random] fills every block from a seed.

## A two-site term

The Heisenberg bond $h = S^z \otimes S^z + \tfrac{1}{2}(S^{+} \otimes S^{-} + S^{-}
\otimes S^{+})$ has four legs — two kets out, two bras in — and six blocks:

```python
>>> bond = (Leg(PHYS, OUT), Leg(PHYS, OUT), Leg(PHYS, IN), Leg(PHYS, IN))
>>> pair = TensorStructure(bond)
>>> pair.num_blocks
6
>>> spin = {DOWN: "d", UP: "u"}
>>> for k in pair.block_order:
...     out = "".join(spin[a] for a in k.output_tree.uncoupled)
...     inp = "".join(spin[a] for a in k.input_tree.uncoupled)
...     print(f"{out} <- {inp}   coupled {k.coupled.charge:+d}")
dd <- dd   coupled -2
du <- du   coupled +0
du <- ud   coupled +0
ud <- du   coupled +0
ud <- ud   coupled +0
uu <- uu   coupled +2

```

Six, out of the $2^4 = 16$ entries of the dense $4 \times 4$ matrix. The ten missing ones
are the transitions that change $S^z_{\text{tot}}$; the two off-diagonal keys that survive
are the exchange flipping an antialigned pair. Writing the six numbers is writing the
operator:

```python
>>> value = {
...     "uuuu": 0.25, "dddd": 0.25,       # S^z S^z, aligned
...     "udud": -0.25, "dudu": -0.25,     # S^z S^z, antialigned
...     "uddu": 0.5, "duud": 0.5,         # the exchange
... }
>>> def name(k):
...     return "".join(spin[a] for a in k.output_tree.uncoupled + k.input_tree.uncoupled)
>>> h = SymmetricTensor.from_blocks(
...     bond, {k: np.full(pair.block_shape(k), value[name(k)]) for k in pair.block_order}
... )
>>> h.to_dense().reshape(4, 4)
array([[ 0.25,  0.  ,  0.  ,  0.  ],
       [ 0.  , -0.25,  0.5 ,  0.  ],
       [ 0.  ,  0.5 , -0.25,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.25]])
>>> np.round(np.linalg.eigvalsh(h.to_dense().reshape(4, 4)), 12)
array([-0.75,  0.25,  0.25,  0.25])

```

The singlet at $-3/4$ and the triplet at $+1/4$, from six numbers and a grading.

## Non-Abelian: no Clebsch-Gordan arrays

Under a non-Abelian symmetry a sector carries an internal multiplet, and the block is no
longer the tensor's dense entries — it is the coefficient of a fusion channel,
$T = \sum_\tau A^{(\tau)} \otimes C^{(\tau)}$, where the $C^{(\tau)}$ are Clebsch-Gordan
data the library supplies. You write the $A^{(\tau)}$; you never spell out a $C^{(\tau)}$.

Take SU(3) with one fundamental $\mathbf{3}$ per site, labelled by its Dynkin weights:

```python
>>> from tenet.symmetry import SUNProvider, SUNSector
>>> SU3 = SUNProvider(3)
>>> THREE, THREEBAR, SIX = SUNSector((1, 0)), SUNSector((0, 1)), SUNSector((2, 0))
>>> SITE = GradedSpace.new(SU3, {THREE: 1})
>>> SITE.dim, SITE.reduced_dim       # three states, one multiplet
(3, 1)

```

This is where `dim` and `reduced_dim` part company: three basis states, but one
degeneracy, because the multiplet is indivisible under the symmetry.

The exchange operator $P$ swaps two sites. On $\mathbf{3} \otimes \mathbf{3} =
\mathbf{6} \oplus \bar{\mathbf{3}}$ it is $+1$ on the symmetric $\mathbf{6}$ and $-1$ on
the antisymmetric $\bar{\mathbf{3}}$ — and the coupled sector of a key is exactly that
address:

```python
>>> exch = (Leg(SITE, OUT), Leg(SITE, OUT), Leg(SITE, IN), Leg(SITE, IN))
>>> su3 = TensorStructure(exch)
>>> [k.coupled.dynkin for k in su3.block_order]
[(0, 1), (2, 0)]
>>> su3.block_shape(su3.block_order[0])
(1, 1, 1, 1)

```

Two keys, each a $1 \times 1 \times 1 \times 1$ block, because a single fundamental has
degeneracy 1 on every leg. So the operator is two numbers:

```python
>>> eigenvalue = {THREEBAR: -1.0, SIX: 1.0}
>>> P = SymmetricTensor.from_blocks(
...     exch,
...     {k: np.full(su3.block_shape(k), eigenvalue[k.coupled]) for k in su3.block_order},
... )
>>> P.shape, P.reduced_shape
((3, 3, 3, 3), (1, 1, 1, 1))

```

Those two numbers are the whole operator. Expanded, they reproduce the $81$-entry
permutation matrix $P_{abcd} = \delta_{ad}\delta_{bc}$ to round-off:

```python
>>> swap = np.eye(9).reshape(3, 3, 3, 3).transpose(0, 1, 3, 2)
>>> bool(abs(P.to_dense() - swap).max() < 1e-12)
True

```

The Clebsch-Gordan coefficients of $\mathbf{3} \otimes \mathbf{3}$ are the rest of it, and
they were never written down. This is the payoff of the block form: the part that depends
on the symmetry is supplied, and the part that depends on the physics is two numbers.

## Round trip, and which constructor

[to_dense][tenet.SymmetricTensor.to_dense] expands to the carrier basis;
[from_dense][tenet.SymmetricTensor.from_dense] projects a dense array back. They are
inverses, which makes the round trip the standard self-check:

```python
>>> import tenet
>>> bool(tenet.allclose(SymmetricTensor.from_dense(swap, exch), P))
True

```

`from_dense` refuses an array that is not symmetric to within `atol` rather than
projecting silently, and that refusal is how a wrong grading is caught from the dense
side:

```python
>>> splus = np.array([[0.0, 1.0], [0.0, 0.0]])       # S^+, in the (down, up) basis
>>> try:
...     SymmetricTensor.from_dense(splus, legs)
... except ValueError as exc:
...     print(str(exc).split(";")[0])
from_dense: array is not symmetric — residual 1 exceeds atol 1.49012e-08

```

Pass [PROJECT][tenet.PROJECT] as `atol` when you do mean "project, do not check" — the
symmetric part of $S^{+}$ on these legs is zero, and that is the answer you get.

Which constructor is right:

| you have | use |
|---|---|
| the operator's action on sectors, stated as coefficients | `from_blocks` |
| an existing dense array, from a reference or another library | `from_dense` |
| a starting point for an optimisation | `random` |
| an accumulator to fill in later | `zeros`, then `with_blocks` |

`from_blocks` is the native spelling: it never materialises the forbidden entries, so it
scales to legs where the dense array does not exist, and under a non-Abelian symmetry it
is usually the only way — there is nothing to write dense in the first place, as the SU(3)
exchange showed.

## Reading blocks back

[items][tenet.SymmetricTensor.items] walks `(key, block)` pairs in `block_order`, which is
the constructor read backwards:

```python
>>> for k, block in sz.items():
...     print(k.coupled, float(block[0, 0]))
U1Sector(charge=-1) -0.5
U1Sector(charge=1) 0.5

```

[to_matrices][tenet.to_matrices] takes the other view: it lowers the tensor to one dense
matrix per coupled sector, codomain against domain, which is the form a blockwise
eigensolver or SVD acts on. The Heisenberg bond becomes three matrices, and the $2 \times
2$ at charge 0 is the antialigned subspace the exchange mixes:

```python
>>> mats = tenet.to_matrices(h)
>>> [(c.charge, mats[c].shape) for c in sorted(mats, key=lambda c: c.charge)]
[(-2, (1, 1)), (0, (2, 2)), (2, (1, 1))]
>>> mats[U1Sector(0)]
array([[-0.25,  0.5 ],
       [ 0.5 , -0.25]])

```

Block-diagonal by coupled sector is what every dense linear-algebra routine in the library
sees, and the reason a symmetric problem is cheaper: three small matrices instead of one
$4 \times 4$.

## Where next

- [Tensors, legs and spaces](../guide/tensors-legs-spaces.md) — the reference for
  `GradedSpace`, `Leg`, `SymmetricTensor` and `TensorStructure`.
- [Building a Hamiltonian](../guide/hamiltonians.md) — turning two-site terms like these
  into an MPO.
- [SU(2)](su2.md) — the same block form with degeneracies above 1.
- [Heisenberg, SU(3)](../examples/su3-heisenberg.md) — the exchange above, run through
  DMRG.
