# Tensors, legs and spaces

Four objects carry the whole model: a [GradedSpace][tenet.GradedSpace] says
*which sectors, with what degeneracy*; a [Leg][tenet.Leg] attaches a space to
one tensor axis with a direction; a [SymmetricTensor][tenet.SymmetricTensor] is
a tuple of legs plus one block per allowed fusion channel; and the
[TensorStructure][tenet.TensorStructure] is the tensor's static half — legs and
derived bookkeeping, no numbers. This page walks them in that order.

## `GradedSpace` — sector → degeneracy

A graded space is a mapping from sectors to positive degeneracies,
`V = ⊕_a C^{m_a} ⊗ V_a`. Build one with [new][tenet.GradedSpace.new], which
sorts the sectors canonically and refuses duplicates, non-positive
degeneracies and sectors of the wrong symmetry:

```python
>>> from tenet import GradedSpace
>>> from tenet.symmetry import U1, U1Sector
>>> V = GradedSpace.new(U1, {U1Sector(1): 1, U1Sector(0): 2})
>>> tuple(V)
(U1Sector(charge=0), U1Sector(charge=1))
>>> V.degeneracy(U1Sector(0))
2
>>> GradedSpace.new(U1, {U1Sector(0): 0})
Traceback (most recent call last):
    ...
ValueError: degeneracy of U1Sector(charge=0) must be positive, got 0

```

A space is immutable, hashable and array-free. It is the *only* place
degeneracies live — legs, tensors and structures all read them from here.

## `dim` versus `reduced_dim`

Every space answers two size questions. `reduced_dim = Σ_a m_a` counts
degeneracies — the storage-facing size, what the stored blocks are made of.
`dim = Σ_a m_a · d_a` weights each degeneracy by the irrep dimension `d_a` —
the physical size, what `to_dense` produces. For U(1) every irrep is
one-dimensional and the two agree; for SU(2) they differ as soon as a sector
with `2j > 0` appears:

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
[shape][tenet.SymmetricTensor.shape]. `reduced_dim` exists for every provider;
`dim` needs the irrep dimensions
([ClebschGordanData][tenet.symmetry.ClebschGordanData]), and a provider with
non-integer quantum dimensions has no physical `dim` at all.

## `Leg` — a space, a side, a dual flag

A leg is one tensor axis: a space, a `side` (`OUT` for the codomain, `IN` for
the domain), a `dual` flag (`V` versus `V*`), and an
optional `name` for bookkeeping:

```python
>>> from tenet import IN, OUT, Leg
>>> p = Leg(V, OUT, name="p")
>>> q = Leg(V, IN, dual=True)
>>> (p.side, p.dual, q.side, q.dual)
(<Side.OUT: 'out'>, False, <Side.IN: 'in'>, True)

```

**The `dual` flag lives on the leg, not on the space.** `side` and `dual` are
independent per-leg metadata, so one `GradedSpace` object can be shared by a
dual and a non-dual leg — a bond space in an MPS is built once and reused on
both ends. There is deliberately no `dual()` method on the space and no way to
flip a leg's `side` in place: moving a leg between domain and codomain is a
categorical bend, and that is [repartition][tenet.SymmetricTensor.repartition]
on the tensor, never a leg-level setter. (Why the flag is placed here rather
than on the space is argued in `space.py`'s docstring and the
[design document](../design.md); this page only states the rule.)

What a dual leg changes is which sector it contributes to a fusion tree — for
U(1), the negated charge:

```python
>>> Leg(V, OUT).fused_sector(U1Sector(1))
U1Sector(charge=1)
>>> Leg(V, OUT, dual=True).fused_sector(U1Sector(1))
U1Sector(charge=-1)

```

## `SymmetricTensor` — legs, not codomain × domain

A tensor is constructed over a flat tuple of legs, in public axis order — you
never declare a codomain × domain partition the way TensorKit-style libraries
require. The partition is *derived*: [codomain][tenet.SymmetricTensor.codomain]
is the `OUT` legs, [domain][tenet.SymmetricTensor.domain] the `IN` legs, and
the map view they induce is what `compose`, `svd` and friends act through.
Sides may interleave freely in the public order:

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

The stored data is one reduced block per allowed fusion channel — the
symmetry-forbidden entries are never materialized. Constructors:
[random][tenet.SymmetricTensor.random] (seeded, reproducible),
[zeros][tenet.SymmetricTensor.zeros],
[from_dense][tenet.SymmetricTensor.from_dense] (which *refuses* a
non-symmetric dense array rather than silently projecting it), and
[from_legs][tenet.SymmetricTensor.from_legs] for blocks you already have.
[to_dense][tenet.SymmetricTensor.to_dense] is the inverse of `from_dense` and
the standard self-check.

Moving between dtypes and backends is blockwise and returns a new tensor:
[astype][tenet.SymmetricTensor.astype] casts every block, and
[to_backend][tenet.SymmetricTensor.to_backend] takes an optional `dtype=`
applied *after* the move, so the target backend's own choice of dtype does not
get the last word:

```python
>>> from tenet import SymmetricTensor
>>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
>>> t.astype("complex128").dtype
dtype('complex128')
>>> t.to_backend("numpy", dtype="complex128").dtype
dtype('complex128')

```

A backend's *refusal* is a different matter from its choice: JAX has no
per-array escape from `jax_enable_x64`, so under the default setting a request
for `float64` truncates to float32 in `astype` exactly as it does in
`jnp.array`. `to_backend`'s Notes say so.

## `TensorStructure` — the static half

The structure is the tensor minus its numbers: the leg tuple plus everything
derivable from it — the ordered [block_order][tenet.TensorStructure.block_order]
of fusion channels, each channel's
[block_shape][tenet.TensorStructure.block_shape], the out/in axis split. Two
tensors with equal structures are element-wise compatible; the structure is
hashable and is exactly what stays static under `jax.jit` while the blocks are
traced (the pytree split, `tenet.pytree`):

```python
>>> t.structure.num_blocks == len(t.blocks)
True
>>> t.structure.block_order[0].coupled
U1Sector(charge=0)

```

You rarely build a `TensorStructure` yourself — every constructor above does it
from the legs — but you read it whenever you ask *which* blocks a tensor has
and why.

## Where next

- [Symmetries and providers](symmetries-and-providers.md) — where sectors like
  `U1Sector` and `SU2Sector` come from, and what a provider can and cannot do.
- [Contraction](contraction.md) — how legs decide what contracts with what.
- The [tenet API page](../api/tenet.md) has the full reference for every class
  named here, with runnable examples on each method.
