# Contraction

Three entry points share one machine. [tenet.tensordot][] pairs explicit axes
of two tensors; [tenet.einsum][] is the label-equation front end for any number
of operands; [tenet.compose][] (spelled `@`) is the categorical composition of
two maps. Everything on this page runs on U(1) examples, but the semantics are
identical for every symmetry — including the fermionic case, which is where
the composition rule at the end earns its keep.

```python
>>> import tenet
>>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
>>> from tenet.symmetry import U1, U1Sector
>>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
>>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
>>> b = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)

```

## `tensordot` — axes, paired in order

`axes=((i, ...), (j, ...))` contracts axis `i` of `a` against axis `j` of `b`,
pairwise; the output is `a`'s free axes followed by `b`'s, exactly as in
`np.tensordot`, and every free leg comes back **unchanged** — same space,
side, `dual` and name:

```python
>>> c = tenet.tensordot(a, b, axes=((1,), (0,)))
>>> c.legs == (a.legs[0], b.legs[1])
True

```

What makes a pair of legs contractible is structural, never numerical: the two
legs must carry the **same space**, and one end of the wire must be the dual
object of the other (opposite `dual xor (side is IN)`). Dimensions are never
compared — two legs of equal dimension on different spaces do not contract,
and the refusal names both legs and the reason:

```python
>>> d = SymmetricTensor.random((Leg(V, IN), Leg(V, OUT)), seed=2)
>>> try:
...     tenet.tensordot(a, d, axes=((1,), (0,)))
... except ValueError as e:
...     print(str(e)[:23])
tensordot: pair 0, axis

```

Two more refusals to know about: a contraction that would leave no free leg is
a `ValueError` — a rank-0 `SymmetricTensor` does not exist, and scalars leave
the tensor world through the named calls [tenet.norm][], [tenet.inner][] and
[tenet.full_trace][] instead — and an axis pattern that moves a leg between
domain and codomain (a line bend) raises
[CapabilityError][tenet.symmetry.CapabilityError] when the provider has no
[BendingCoefficients][tenet.symmetry.BendingCoefficients].

## `einsum` — labels, with rules

The equation language is NumPy's, restricted to what is equivariant. Labels
are single ASCII letters, one per axis; a label occurs **at most twice** in
the whole equation, because a wire has two ends; with `->` omitted, the output
is every once-occurring label, sorted (the NumPy rule):

```python
>>> c2 = tenet.einsum("ab,bc->ac", a, b)
>>> bool(tenet.allclose(c2, tenet.tensordot(a, b, axes=((1,), (0,)))))
True

```

The parser refuses, by name, everything that has no equivariant meaning. A
label repeated *within* one operand is a diagonal or a single-operand trace —
use [tenet.trace][] for the trace; the diagonal is not equivariant:

```python
>>> try:
...     tenet.einsum("aa->", a)
... except ValueError as e:
...     print(str(e).split(", i.e.")[0])
einsum: label 'a' is repeated inside term 'aa'

```

Ellipsis means broadcasting over unlabelled axes, and symmetric tensors do not
broadcast:

```python
>>> try:
...     tenet.einsum("a...,...a->", a, b)
... except ValueError as e:
...     print(str(e).split(";")[0])
einsum: equation 'a...,...a->' contains '...'

```

And an input label missing from the output would sum an axis away — also not
equivariant, also refused:

```python
>>> try:
...     tenet.einsum("ab->a", a)
... except ValueError as e:
...     print(str(e).split(",")[0])
einsum: input label 'b' of 'ab->a' is missing from the output

```

Each message says what to write instead; the full list is in
[tenet.einsum][]'s `Raises`.

## `compose` — `a @ b`, codomain meets domain

Composition is the map-level operation: `b`'s codomain is consumed by `a`'s
domain, which must carry the same `(space, dual)` sequence in the same order —
no axis labels, no reordering, dimensions alone never enough. Its unit is
[tenet.identity][]:

```python
>>> bool(tenet.allclose(tenet.identity(a.codomain) @ a, a))
True

```

Composition never reorders legs within a side; that is [tenet.transpose][],
and moving a leg *between* sides is
[repartition][tenet.SymmetricTensor.repartition].

## Cap direction, and the composition rule

For a fermionic provider the two ends of a wire are not interchangeable: the
cap `V* ⊗ V → 1` is not the cap `V ⊗ V* → 1`, and they differ by a Koszul sign
on every odd sector. So when an einsum contracts an odd wire, *which operand
supplies which end* is load-bearing — a per-call choice is exactly what
produces stray `(-1)` phases.

The library's rule (#160), stated plainly: **every two-operand einsum is a
composition — operand 1 supplies the `IN` end of every shared wire and operand
2 the `OUT` end.** With three or more operands the same rule applies pairwise
in caller order. A wire that genuinely turns around in your diagram is bent
explicitly with [repartition][tenet.SymmetricTensor.repartition] *before* the
einsum, never left to an implicit cap. `tenet.network` follows this rule at
every call site, pinned by a hygiene test with zero exemptions; the formal
statement lives in `tenet.network`'s module docstring and
[tenet.tensordot][]/[tenet.einsum][]'s `Notes`. For a purely bosonic symmetry
every choice gives the same numbers — which is precisely why the discipline
must be kept before the fermions arrive, not after.

## What `optimize` does — and does not

With one or two operands, `optimize` is not consulted (`opt_einsum` is not
even imported). With three or more, `opt_einsum.contract_path` chooses the
*pairwise order* from the operands' physical shapes, and `optimize` is handed
to it unchanged — a strategy name, an explicit path, or any
`opt_einsum.paths.PathOptimizer` (cotengra's optimizers are such objects).
What it never changes is the mathematics: every step of the path is the same
two-operand contraction, shared labels contract in a deterministic order, and
the final transpose is a real categorical permutation, never skipped. The path
depends on static structure only, so under `jax.jit` it is baked in at trace
time like every other structural decision.

## Where next

- [Tensors, legs and spaces](tensors-legs-spaces.md) — the legs whose `side`
  and `dual` this page has been reading.
- [Symmetries and providers](symmetries-and-providers.md) — which providers
  can bend, braid, and take traces.
- [tenet.tensordot][], [tenet.einsum][], [tenet.trace][],
  [tenet.full_trace][], [tenet.compose][] — the full reference, refusal by
  refusal.
