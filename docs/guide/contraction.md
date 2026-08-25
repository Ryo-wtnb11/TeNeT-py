# Contraction

Three entry points share one machine. [tenet.tensordot][] pairs explicit axes of two
tensors. [tenet.einsum][] is the label-equation front end for any number of operands.
[tenet.compose][], spelled `@`, is the categorical composition of two maps. The examples
here use U(1); the semantics are identical for every symmetry, including the fermionic
case, which is where the composition rule at the end earns its keep.

```python
>>> import tenet
>>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
>>> from tenet.symmetry import U1, U1Sector
>>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
>>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
>>> b = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)

```

## `tensordot` — axes, paired in order

`axes=((i, ...), (j, ...))` contracts axis `i` of `a` against axis `j` of `b`, pairwise.
The output is `a`'s free axes followed by `b`'s, as in `np.tensordot`, and every free leg
comes back unchanged — same space, side, `dual` and name:

```python
>>> c = tenet.tensordot(a, b, axes=((1,), (0,)))
>>> c.legs == (a.legs[0], b.legs[1])
True

```

What makes a pair of legs contractible is structural: the two legs must carry the **same
space**, and one end of the wire must be the dual object of the other (opposite
`dual xor (side is IN)`). Dimensions are never compared, so two legs of equal dimension
on different spaces do not contract, and the refusal names both legs and the reason:

```python
>>> d = SymmetricTensor.random((Leg(V, IN), Leg(V, OUT)), seed=2)
>>> try:
...     tenet.tensordot(a, d, axes=((1,), (0,)))
... except ValueError as e:
...     print(str(e)[:23])
tensordot: pair 0, axis

```

`d` carries the same space `V` on both axes, so every dimension matches — and the
contraction is still refused, because `d`'s axis 0 and `a`'s axis 1 present the same end
of the wire rather than dual ends. The message is sliced here only to keep the doctest
short; in full it names both legs and the reason, and the refusals below are printed the
same way.

Two more refusals to know. A contraction that would leave no free leg is a `ValueError`:
a rank-0 `SymmetricTensor` does not exist, and scalars leave the tensor world through
[tenet.norm][], [tenet.inner][] and [tenet.full_trace][]. An axis pattern that moves a
leg between domain and codomain is a line bend, and raises
[CapabilityError][tenet.symmetry.CapabilityError] when the provider has no
[BendingCoefficients][tenet.symmetry.BendingCoefficients].

## `einsum` — labels, with rules

The equation language is NumPy's, restricted to what is equivariant. Labels are single
ASCII letters, one per axis; a label occurs at most twice in the whole equation, because
a wire has two ends; with `->` omitted, the output is every once-occurring label, sorted:

```python
>>> c2 = tenet.einsum("ab,bc->ac", a, b)
>>> bool(tenet.allclose(c2, tenet.tensordot(a, b, axes=((1,), (0,)))))
True

```

The parser refuses, by name, everything with no equivariant meaning. A label repeated
*within* one operand is a diagonal or a single-operand trace — [tenet.trace][] is the
trace, and the diagonal is not equivariant:

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

An input label missing from the output would sum an axis away — also refused:

```python
>>> try:
...     tenet.einsum("ab->a", a)
... except ValueError as e:
...     print(str(e).split(",")[0])
einsum: input label 'b' of 'ab->a' is missing from the output

```

Each message says what to write in place of the refused equation; the full list is in
[tenet.einsum][]'s `Raises`.

## `compose` — `a @ b`, codomain meets domain

Composition is the map-level operation: `b`'s codomain is consumed by `a`'s domain,
which must carry the same `(space, dual)` sequence in the same order. No axis labels, no
reordering. Its unit is [tenet.identity][]:

```python
>>> bool(tenet.allclose(tenet.identity(a.codomain) @ a, a))
True

```

Composition never reorders legs within a side; that is [tenet.transpose][], and moving a
leg between sides is [repartition][tenet.SymmetricTensor.repartition].

## The composition rule

For a fermionic provider the two ends of a wire are not interchangeable: the cap
$V^{*} \otimes V \to \mathbf{1}$ and the cap $V \otimes V^{*} \to \mathbf{1}$ differ by a
Koszul sign on every odd sector. So when an einsum contracts an odd wire, which operand
supplies which end is load-bearing.

**The rule you follow: every two-operand einsum is a composition. Operand 1 supplies the
`IN` end of every shared wire and operand 2 the `OUT` end.** With three or more operands
the same rule applies pairwise in caller order.

A wire that turns around in your diagram is bent explicitly with
[repartition][tenet.SymmetricTensor.repartition] *before* the einsum. The formal
statement is in [tenet.tensordot][]'s and [tenet.einsum][]'s `Notes`, and
`tenet.network` follows the rule at every call site, pinned by a hygiene test with no
exemptions. For a purely bosonic symmetry every choice gives the same numbers, which is
why the discipline is kept before the fermions arrive.

## What `optimize` does

With one or two operands, `optimize` is not consulted and `opt_einsum` is not imported.
With three or more, `opt_einsum.contract_path` chooses the pairwise order from the
operands' physical shapes, and `optimize` is handed to it unchanged — a strategy name,
an explicit path, or any `opt_einsum.paths.PathOptimizer` (cotengra's optimizers are
such objects).

What it never changes is the mathematics: every step of the path is the same two-operand
contraction, shared labels contract in a deterministic order, and the final transpose is
a real categorical permutation. The path depends on static structure only, so under
`jax.jit` it is baked in at trace time like every other structural decision.

For a graded network whose sectors are unevenly filled, a hand-written pairwise order
often beats the path a cost model derived from physical leg sizes picks. Writing the
contraction as a chain of two-operand `einsum` calls is how you take that control.

## Where next

- [Tensors, legs and spaces](tensors-legs-spaces.md) — the legs whose `side` and `dual`
  this page reads.
- [Symmetries and providers](symmetries-and-providers.md) — which providers can bend,
  braid and take traces.
- [tenet.tensordot][], [tenet.einsum][], [tenet.trace][], [tenet.full_trace][],
  [tenet.compose][] — the full reference, refusal by refusal.
