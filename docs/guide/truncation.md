# Truncation

Cutting a bond is the one factorization that is **not** jittable: which sectors survive,
and how many singular values each keeps, is read off the numbers. So the decision and the
numerics are two functions, and you call them together or apart.

## One call

[tenet.ops.linalg.svd_truncated][] decides and factorizes in one go. `max_bond` bounds
the **dense** bond dimension $\sum_c \operatorname{qdim}(c)\, m_c$, `cutoff` applies one of six
`cutoff_mode` rules, and giving both takes the intersection:

```python
>>> import tenet
>>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
>>> from tenet.symmetry import U1, U1Sector
>>> W = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
>>> t = SymmetricTensor.random((Leg(W, OUT), Leg(W, IN)), seed=0)
>>> u, s, vh = tenet.linalg.svd_truncated(t, max_bond=2)
>>> s.shape
(2, 2)

```

Passing neither `max_bond` nor `cutoff` is refused, naming [tenet.ops.linalg.svd][].
`None` means "no truncation" on either; there are no sentinel values.

| `cutoff_mode` | keeps |
|---|---|
| `abs` | $\sigma > \texttt{cutoff}$ |
| `rel` | $\sigma > \texttt{cutoff} \cdot \sigma_{\max}$ |
| `sum2` | drops the largest set with $\sum \operatorname{qdim}(c)\, \sigma^2 < \texttt{cutoff}$ |
| `rsum2` (default) | as `sum2`, threshold $\texttt{cutoff} \cdot \lVert T\rVert^2$ |
| `sum1` | as `sum2` at power 1, weight $\operatorname{qdim}(c)\, \sigma$ |
| `rsum1` | as `rsum2` at power 1 |

`renorm=True` scales the kept singular values by $\sqrt{\lVert T\rVert^2 / \sum_{\mathrm{kept}}
\operatorname{qdim}(c)\, \sigma^2}$, so that `norm(U @ S @ Vh) == norm(t)`. It is a bool.

`S` comes back as a tensor, so absorbing it into either factor is a one-line `compose`, and the
truncation error is exactly $\lVert t\rVert^2 - \lVert USV^{\dagger}\rVert^2$ by Pythagoras.

## Two calls

[tenet.ops.linalg.select_bond][] makes the same decision and *returns* it as a
[tenet.ops.linalg.BondSelection][]. [tenet.ops.linalg.svd][]'s `bond=` keyword takes the
bond space and does the numerics — and that half **is** jittable and differentiable,
because a `GradedSpace` is frozen, array-free metadata decided outside the trace:

```python
>>> selection = tenet.linalg.select_bond(t, max_bond=2)
>>> u2, s2, vh2 = tenet.linalg.svd(t, bond=selection.bond)
>>> bool(tenet.allclose(u @ s @ vh, u2 @ s2 @ vh2))
True

```

The two spellings are the same operator. What the second adds is what the first throws
away — the discarded weight, which is what a DMRG sweep reports and what an
extrapolation to zero truncation error consumes:

```python
>>> round(selection.discarded_weight, 12) == round(
...     float(tenet.norm(t)) ** 2 - float(tenet.norm(u @ s @ vh)) ** 2, 12
... )
True

```

The shape of a differentiable loop is: decide once, outside; project every iteration,
inside.

```python
selection = tenet.linalg.select_bond(t0, axes, max_bond=D)   # outside jit/grad

@jax.jit
def step(t):
    u, s, vh = tenet.linalg.svd(t, axes, bond=selection.bond)
    ...
```

Calling `svd_truncated` under `jax.jit`, `jax.grad` or `jax.vmap` raises
[StructureChangingError][tenet.symmetry.StructureChangingError], naming the pairing to
use.

## How the selection is made

Selection is over **one global spectrum**, in every mode:

- the sort key is the **bare** $\sigma$, descending, ties broken by sector order then
  index — how large a singular value is has nothing to do with multiplicity;
- the **cost** and the **weight** are $\operatorname{qdim}(c)$-weighted, because the reduced
  index $i$ in sector $c$ stands for $\operatorname{qdim}(c)$ dense basis states. That is the same weight
  [tenet.norm][] carries, which makes greedy-descending under a dense budget optimal:
  the result is the best approximation of its achieved dense rank.

The walk **stops** at the first singular value that would overflow the budget, which
keeps the kept set nested as `max_bond` grows. The consequence is that `max_bond` may be
undershot by up to $\max_c \operatorname{qdim}(c) - 1$.

## Why the decision is worth looking at: SU(2)

For U(1) and fermionic parity every sector has $\operatorname{qdim}(c) = 1$, so a `max_bond` of
$D$ keeps exactly $D$ singular values. Under SU(2) a sector is a *multiplet*: $j = 1$ costs
three dense dimensions, $j = 0$ costs one.

Take a bond carrying one singlet and three triplets, and ask for five:

```python
>>> from tenet.symmetry import SU2, SU2Sector
>>> space = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(2): 3})
>>> h = SymmetricTensor.random((Leg(space, OUT), Leg(space, IN)), seed=2)
>>> selection = tenet.linalg.select_bond(h, max_bond=5)
>>> selection.dense_dim, selection.reduced_dim
(3.0, 1)

```

One triplet was admitted, at a dense cost of three. Two units of budget are left and
nothing fits in them:

```python
>>> selection.undershoot
2.0
>>> value, sector, index = selection.next_multiplet
>>> sector, selection.next_dense_cost
(SU2Sector(two_j=2), 3.0)

```

`next_multiplet` is what the cut stopped just short of and `next_dense_cost` is what
admitting it would cost. Raising `max_bond` to 6 spends the budget exactly:

```python
>>> tenet.linalg.select_bond(h, max_bond=6).dense_dim
6.0

```

That is why the decision is a returned object. A U(1) user reads `BondSelection` as a
convergence log; an SU(2) user reads it as the answer to "why is my bond smaller than I
asked for".

The same arithmetic governs a DMRG schedule: `Sweep(chi=64)` on SU(2) legs bounds the
dense dimension, so it keeps roughly a third the multiplets it keeps on U(1) legs — the
same state, held in fewer, larger irreps.

## The Hermitian route

An SVD of a self-adjoint operator returns $\lvert w\rvert$ and drops which eigenvalues were
negative, so `U @ S @ adjoint(U)` reconstructs an indefinite operator with every sign
flipped positive. That is structural, not a tolerance.
[tenet.ops.linalg.eigh_truncated][] is the twin for that case — same keyword set, same
keep rule, same refusal under a trace — and [tenet.ops.linalg.eigh][]'s `bond=` is the
jittable half:

```python
>>> x = SymmetricTensor.random((Leg(W, OUT), Leg(W, IN)), seed=3)
>>> herm = (x + tenet.transpose(tenet.adjoint(x), (1, 0))) / 2  # indefinite
>>> ew, ev = tenet.linalg.eigh_truncated(herm, max_bond=2)
>>> [round(float(z), 6) for z in tenet.to_matrices(ew)[U1Sector(0)].diagonal()]
[2.422876, -0.949726]
>>> ew2, ev2 = tenet.linalg.eigh(herm, bond=ew.structure.legs[0].space)
>>> bool(tenet.allclose(ev @ ew @ tenet.adjoint(ev), ev2 @ ew2 @ tenet.adjoint(ev2)))
True

```

Two places where it is not a literal mirror of the SVD:

- **the kept set is not a prefix.** Singular values come back descending, so `svd`
  slices; eigenvalues come back ascending and signed, so the $k$ largest by
  $\lvert w\rvert$ is an `argsort` and a gather. A gather is a value-dependent permutation, never a
  value-dependent shape, so `eigh(..., bond=)` still traces.
- **the sign survives.** Only the ordering key is $\lvert w\rvert$; `W`'s retained entries
  are the signed eigenvalues.

On a positive-definite input the two routes agree factor for factor.

## The record

| field | meaning |
|---|---|
| `bond` | the truncated `GradedSpace` — what `svd(..., bond=)` consumes |
| `dense_dim` | $\sum_c \operatorname{qdim}(c)\, m_c$, the dimension `max_bond` bounds |
| `reduced_dim` | $\sum_c m_c$, what the reduced blocks are made of |
| `kept` / `discarded` | `(magnitude, sector, index)` triples, descending |
| `discarded_weight` | $\sum_{\mathrm{discarded}} \operatorname{qdim}(c)\, \sigma^2$ |
| `undershoot` | `max_bond - dense_dim`, or `None` |
| `next_multiplet` / `next_dense_cost` | what the cut stopped short of, and its price |
| `scale` | the factor `renorm=True` applies; every magnitude above is bare |

`BondSelection` is frozen and is not a JAX pytree: it is decided outside the traced
region, and its `bond` is what crosses into one.

## Where next

- [DMRG](dmrg.md) — where the sweep spends this budget.
- [JAX and backends](jax-and-backends.md) — the traced half of the pairing.
- [`tenet.linalg`](../api/linalg.md) — the reference.
