# Getting started

## Install

```sh
uv add tenet-py         # or: pip install tenet-py
```

The core install needs only `numpy`, `autoray` and `opt-einsum`. Three optional extras:

```sh
uv add "tenet-py[jax]"      # jax>=0.10 — first release with the wide-matrix qr JVP
uv add "tenet-py[torch]"    # torch>=2.0 — eager only; tenet.ad stays JAX-only
uv add "tenet-py[sun]"      # racah-py — required for SU(N)
```

SU(N) support (`tenet.symmetry.sun`) needs `racah-py` and raises an `ImportError`
naming the extra without it. Everything else — SU(2), U(1), Z2, fermion parity and
their products — runs on the core install.

The [home page](index.md) carries the quickstart: an SU(2) leg, a random invariant
tensor, an `einsum` contraction and an SVD, plus the same tensor under `jax.jit` and
`jax.grad`. This page is the *next* five minutes: what the objects you just built
actually are, how to read them back, and how to check yourself against dense NumPy.

## The smallest complete thing

Every tensor in `tenet` is built from the same three ingredients, in order: a
**graded space** (which sectors, with what degeneracy), a **leg** (that space plus
a direction), and the **tensor** over a tuple of legs. Here with U(1), the plainest
symmetry:

```python
>>> import tenet
>>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
>>> from tenet.symmetry import U1, U1Sector
>>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
>>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
>>> a
SymmetricTensor(ndim=2, shape=(3, 3), dtype=float64, backend='numpy', blocks=2)

```

`V` is a three-dimensional space graded into a two-dimensional charge-0 piece and a
one-dimensional charge-1 piece. `a` is a random tensor with one `OUT` leg and one
`IN` leg on that space — a linear map `V → V` — and, because it must commute with
the symmetry, it stores two blocks (one per charge) rather than nine dense entries.

Contract it with itself and take a norm, exactly as you would in NumPy:

```python
>>> b = tenet.einsum("ab,bc->ac", a, a)
>>> b.legs == (a.legs[0], a.legs[1])
True
>>> bool(tenet.norm(b) > 0)
True

```

Free legs come back exactly as they went in — same space, same side, same `dual`
flag — which is what the `b.legs` check above says. Scalars leave the tensor world
only through named calls ([tenet.norm][], [tenet.inner][], [tenet.full_trace][]);
a contraction that would close every leg is refused.

## Reading a leg back

The `repr` of a leg spells out all four fields — the space (its provider and its
sector → degeneracy pairs), the side, the dual flag and the name:

```python
>>> Leg(V, OUT, name="p")
Leg(space=GradedSpace(provider=U1Provider(name='U1'), sectors=((U1Sector(charge=0), 2), (U1Sector(charge=1), 1))), side=<Side.OUT: 'out'>, dual=False, name='p')

```

Two dimensions are attached to every space, and telling them apart early saves
confusion later: `reduced_dim` counts degeneracies (what the stored blocks are made
of) and `dim` counts physical dimensions (what a dense array would have). For an
abelian symmetry every irrep is one-dimensional and the two agree; for SU(2) they
do not — see [Tensors, legs and spaces](guide/tensors-legs-spaces.md).

```python
>>> V.reduced_dim, V.dim
(3, 3)

```

## `to_dense` as a self-check

[to_dense][tenet.SymmetricTensor.to_dense] materializes the block-sparse tensor as
an ordinary backend array, one dense axis per leg. It is the wrong tool for
computation — it throws away exactly the structure the library exists to keep —
but it is the right tool for convincing yourself the structure is what you think
it is:

```python
>>> import numpy as np
>>> d = a.to_dense()
>>> d.shape
(3, 3)
>>> bool(np.allclose(np.linalg.norm(d), float(tenet.norm(a))))
True

```

The forbidden entries really are zero — the charge-0 block and the charge-1 block
occupy disjoint index ranges, and nothing connects them:

```python
>>> bool(np.allclose(d[2, :2], 0.0)) and bool(np.allclose(d[:2, 2], 0.0))
True

```

That check — contract or factorize symmetrically, `to_dense` both sides, compare
with dense NumPy — works for every operation in the library and is how the test
suite pins most of them.

## Where to go next

- [Tensors, legs and spaces](guide/tensors-legs-spaces.md) — the object model:
  `GradedSpace`, `Leg`, `SymmetricTensor`, `TensorStructure`, and why `dual`
  lives on the leg.
- [Symmetries and providers](guide/symmetries-and-providers.md) — the sector
  conventions (SU(2) labels by **2j**), the capability protocols, and what a
  `CapabilityError` means.
- [Contraction](guide/contraction.md) — `tensordot`, `einsum` and `compose`
  semantics, the label rules, and the composition rule for fermionic wires.
- The [tutorials](tutorials/dmrg.md) walk the algorithms (DMRG, CTMRG, VMC); the
  [examples](examples/index.md) are complete runnable files, each checked against
  a named exact oracle.
- The [API reference](api/tenet.md) documents every public name, with runnable
  examples; the [design document](design.md) is the categorical model underneath.
