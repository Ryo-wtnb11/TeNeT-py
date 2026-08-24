# Tensors, legs and spaces

Four objects carry the whole model. A [GradedSpace][tenet.GradedSpace] says *which
sectors, with what degeneracy*. A [Leg][tenet.Leg] attaches a space to one tensor axis
with a direction. A [SymmetricTensor][tenet.SymmetricTensor] is a tuple of legs plus one
block per allowed fusion channel. A [TensorStructure][tenet.TensorStructure] is the
tensor's static half: legs and everything derivable from them, no numbers.

## `GradedSpace` — sector to degeneracy

A graded space is a mapping from sectors to positive degeneracies,
$V = \bigoplus_a \mathbb{C}^{m_a} \otimes V_a$. Build one with [new][tenet.GradedSpace.new],
which sorts the sectors canonically:

```python
>>> from tenet import GradedSpace
>>> from tenet.symmetry import U1, U1Sector
>>> V = GradedSpace.new(U1, {U1Sector(1): 1, U1Sector(0): 2})
>>> tuple(V)
(U1Sector(charge=0), U1Sector(charge=1))
>>> V.degeneracy(U1Sector(0))
2

```

A space is immutable, hashable and array-free. It is the only place degeneracies live:
legs, tensors and structures all read them from here.

**Refusals.** `new` rejects a duplicate sector, a degeneracy of zero or less, and a
sector belonging to another symmetry:

```python
>>> GradedSpace.new(U1, {U1Sector(0): 0})
Traceback (most recent call last):
    ...
ValueError: degeneracy of U1Sector(charge=0) must be positive, got 0

```

## `dim` versus `reduced_dim`

Every space answers two size questions. `reduced_dim`, $\sum_a m_a$, counts degeneracies —
the storage-facing size, what the stored blocks are made of. `dim`, $\sum_a m_a d_a$,
weights each degeneracy by the irrep dimension $d_a$ — the physical size, what
`to_dense` produces. For U(1) every irrep is one-dimensional and the two agree; under
SU(2) they part as soon as a sector with $2j > 0$ appears:

```python
>>> from tenet.symmetry import SU2, SU2Sector
>>> W = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2})
>>> W.reduced_dim   # 2 + 2 multiplets
4
>>> W.dim           # 2*1 (singlets) + 2*2 (doublets)
6

```

The same split appears on tensors as
[reduced_shape][tenet.SymmetricTensor.reduced_shape] versus
[shape][tenet.SymmetricTensor.shape]. `reduced_dim` is defined for every provider;
`dim` needs the irrep dimensions ([ClebschGordanData][tenet.symmetry.ClebschGordanData]).

Keep the distinction in mind wherever a dimension is a budget: `svd_truncated`'s
`max_bond` bounds `dim`, so on SU(2) legs it admits fewer, larger multiplets than the
number suggests. [Truncation](truncation.md) has that in detail.

## `Leg` — a space, a side, a dual flag

A leg is one tensor axis: a space, a `side` (`OUT` for the codomain, `IN` for the
domain), a `dual` flag ($V$ versus $V^{*}$), and an optional `name`:

```python
>>> from tenet import IN, OUT, Leg
>>> p = Leg(V, OUT, name="p")
>>> q = Leg(V, IN, dual=True)
>>> (p.side, p.dual, q.side, q.dual)
(<Side.OUT: 'out'>, False, <Side.IN: 'in'>, True)

```

**The `dual` flag lives on the leg.** `side` and `dual` are independent per-leg
metadata, so one `GradedSpace` object is shared by a dual and a non-dual leg — an MPS
bond space is built once and used on both ends of the bond.

What a dual leg changes is which sector it contributes to a fusion tree — for U(1), the
negated charge:

```python
>>> Leg(V, OUT).fused_sector(U1Sector(1))
U1Sector(charge=1)
>>> Leg(V, OUT, dual=True).fused_sector(U1Sector(1))
U1Sector(charge=-1)

```

Legs are immutable. [dualized][tenet.Leg.dualized] and [renamed][tenet.Leg.renamed]
return new legs; moving a leg between domain and codomain is a categorical bend, and it
is [repartition][tenet.SymmetricTensor.repartition] on the tensor.

## `SymmetricTensor` — legs, in public axis order

A tensor is constructed over a flat tuple of legs. The codomain × domain partition is
*derived*: [codomain][tenet.SymmetricTensor.codomain] is the `OUT` legs,
[domain][tenet.SymmetricTensor.domain] the `IN` legs, and the map view they induce is
what `compose`, `svd` and friends act through. Sides may interleave freely in the public
order:

```python
>>> from tenet import SymmetricTensor
>>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN), Leg(V, OUT)), seed=0)
>>> t.ndim
3
>>> (len(t.codomain), len(t.domain))
(2, 1)
>>> t.shape, t.reduced_shape
((3, 3, 3), (3, 3, 3))

```

The stored data is one reduced block per allowed fusion channel; the symmetry-forbidden
entries are never materialized. The constructors:

| call | what it takes |
|---|---|
| [random][tenet.SymmetricTensor.random] | legs and a seed — reproducible standard-normal blocks |
| [zeros][tenet.SymmetricTensor.zeros] | legs and a dtype |
| [from_blocks][tenet.SymmetricTensor.from_blocks] | a mapping from key to block; absent keys are zero |
| [from_dense][tenet.SymmetricTensor.from_dense] | a dense carrier-basis array, projected onto the symmetric subspace |

[to_dense][tenet.SymmetricTensor.to_dense] is `from_dense`'s inverse and the standard
self-check.

`from_dense` refuses an array that is not symmetric to within `atol`, and that refusal
is load-bearing: it is how you learn a grading is wrong. Pass
[PROJECT][tenet.PROJECT] as `atol` when you mean "project, do not check".

### Naming a block instead of counting it

When you hold only *some* of the blocks, name them.
[from_blocks][tenet.SymmetricTensor.from_blocks] takes a mapping from
[FusionBlockKey][tenet.FusionBlockKey] to array and fills the keys you leave out with
zeros; the keys come from `TensorStructure(legs).block_order`, so nothing has to be
built first to find out what the layout is.

This is the natural spelling away from Abelian symmetries, where the reduced block per fusion
tree is the datum and the dense array is derived. The SU(2) evaluation cup
$V_{1/2} \otimes V_{1/2}^{*} \to \mathbf{1}$ has exactly one fusion channel, and its whole
content is that the channel carries coefficient 1:

```python
>>> import numpy as np
>>> from tenet import TensorStructure
>>> S = GradedSpace.new(SU2, {SU2Sector(1): 1})       # one spin-1/2 multiplet
>>> legs = (Leg(S, OUT), Leg(S, OUT, dual=True))
>>> structure = TensorStructure(legs)
>>> key, = structure.block_order                       # the one allowed channel
>>> key.coupled                                        # it couples to the singlet
SU2Sector(two_j=0)
>>> cup = SymmetricTensor.from_blocks(legs, {key: np.ones(structure.block_shape(key))})
>>> np.round(cup.to_dense() * np.sqrt(2), 12)          # the identity, as the cup should be
array([[1., 0.],
       [0., 1.]])

```

To *change* blocks rather than build them,
[with_blocks][tenet.SymmetricTensor.with_blocks] takes the same mapping and carries
every other block over — the immutable spelling of assigning to one block:

```python
>>> t2 = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
>>> k = t2.structure.block_order[0]
>>> u = t2.with_blocks({k: np.zeros(t2.structure.block_shape(k))})
>>> bool(u.block(k).any()), bool(t2.block(k).any())     # t2 is untouched
(False, True)

```

Both refuse a key that is not in `block_order`, with a message naming where the legal
keys live, and a block of the wrong shape is refused naming the shape expected.

### Blockwise maps

[apply_blocks][tenet.apply_blocks] applies a function to every reduced block and
[zip_blocks][tenet.zip_blocks] to the aligned block pairs of two tensors sharing one
structure. Both work in **coefficient space**: the function sees the reduced blocks, not
the dense entries, so a nonlinear function of a tensor is a nonlinear function of its
coefficients.

```python
>>> import tenet
>>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)
>>> b = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=2)
>>> s = tenet.zip_blocks(a, b, lambda x, y: x + y)
>>> bool(tenet.allclose(s, a + b))
True

```

[map_diagonal][tenet.map_diagonal] reads the diagonal of a square map onto its codomain
legs, giving a tensor with the structure the vectors that map acts on carry — so
`zip_blocks` pairs the two block for block.

### dtypes and backends

[astype][tenet.SymmetricTensor.astype] casts every block;
[to_backend][tenet.SymmetricTensor.to_backend] moves the blocks to `"numpy"`, `"jax"` or
`"torch"` and takes an optional `dtype=` applied after the move, so the target backend's
own dtype choice does not get the last word:

```python
>>> t2.astype("complex128").dtype
dtype('complex128')
>>> t2.to_backend("numpy", dtype="complex128").dtype
dtype('complex128')

```

Under JAX's default configuration a request for `float64` lands on float32, exactly as
in `jnp.array`; [JAX and backends](jax-and-backends.md) covers the setting that changes
it.

## `TensorStructure` — the static half

The structure is the tensor minus its numbers: the leg tuple plus the ordered
[block_order][tenet.TensorStructure.block_order] of fusion channels, each channel's
[block_shape][tenet.TensorStructure.block_shape], and the out/in axis split. Two tensors
with equal structures are element-wise compatible. It is hashable, and it is what stays
static under `jax.jit` while the blocks are traced:

```python
>>> t2.structure.num_blocks == len(t2.blocks)
True
>>> t2.structure.block_order[0].coupled
U1Sector(charge=0)

```

You rarely build a `TensorStructure` yourself — every constructor above does it from the
legs — but you read it whenever you ask which blocks a tensor has and why.

## Where next

- [Symmetries and providers](symmetries-and-providers.md) — where `U1Sector` and
  `SU2Sector` come from, and what a provider can do.
- [Contraction](contraction.md) — how legs decide what contracts with what.
- [The `tenet` API page](../api/tenet.md) — the full reference for every name here.
