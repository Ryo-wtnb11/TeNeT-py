# SU(2) — the same chain, non-Abelian

The Heisenberg chain of the [DMRG tutorial](dmrg.md), graded by SU(2) instead of U(1).
This is [`examples/su2_heisenberg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/su2_heisenberg.py),
which runs both gradings in one process and prints them side by side; its output is
committed on the [Heisenberg, SU(2)](../examples/su2-heisenberg.md) page.

```sh
uv run python examples/su2_heisenberg.py
```

## Sectors are labelled by $2j$

```python
from tenet.symmetry import SU2, SU2Sector

SU2Sector(0)   # the singlet
SU2Sector(1)   # the spin-1/2 doublet, dense dimension 2
SU2Sector(2)   # the spin-1 triplet, dense dimension 3
```

Every label is an exact integer, and no half-integer appears anywhere.

Because an SU(2) sector is a *multiplet*, a space's two dimensions part company.
`reduced_dim` counts multiplets — what the stored blocks are made of. `dim` counts dense
basis states, $\sum_a m_a (2j_a + 1)$. That distinction runs through everything on this page.

## The operator set is `{S.S}`

```python
from tenet.models import spin_half

site = spin_half(SU2)
sorted(site.ops)          # ['S.S']
site.ops["S.S"].ndim      # 4
```

`S^z` and `S+` are not in the table because **there is no such SU(2) operator**. The
rank-3 charge-leg form puts the emitted sector on a `D=1` leg, and the only leg a spin-1
tensor operator could emit onto is the $j = 1$ multiplet, whose dense dimension is 3.

What is invariant is the whole bond term, and `S.S` is one:

```python
from tenet.network import MPO

n = 20
h = MPO.from_terms(n, [(1.0, [(site.ops["S.S"], (i, i + 1))]) for i in range(n - 1)])
```

`from_terms` splits the two-site operator with an SVD, and the graded MPO bond comes out
of that split. Nothing about the recoupling is written down and there is no coupling tree
to name: it is already inside the array's blocks.

Hand it a symmetry-breaking array — `np.kron(sz, sz)` alone — and `from_dense` raises. The
term language cannot express a term that is not invariant, and the refusal is the
statement.

Building one yourself is [`local_op`][tenet.network.local_op] with no `charge`: it takes a
`(d**k, d**k)` or `(d,)*2k` array, the layout `np.kron` produces.

## The seed

`MPS.product` is Abelian-only and refuses: a single sector is not a non-Abelian multiplet.
Seed with [`MPS.random`][tenet.network.MPS.random] over bond spaces with singlet
boundaries:

```python
from tenet import GradedSpace
from tenet.network import MPS

tri = GradedSpace.new(SU2, {SU2Sector(0): 1})
mid = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2, SU2Sector(2): 1})
psi = MPS.random(site.phys, [tri] + [mid] * (n - 1) + [tri], seed=0)
```

`D=1` singlet boundary legs put the run in the total-spin-0 sector, the way a charged
`D=1` boundary leg fixes $S^z_{\mathrm{tot}}$ under U(1).

## What the grading buys

```python
from tenet.network import dmrg_

out = dmrg_(psi, h, chi=64)
mid = out.psi[n // 2].legs[0].space
mid.reduced_dim, mid.dim     # multiplets, dense states
```

At $N = 20$, `chi=64`, the SU(2) run reaches the same energy as the U(1) run —
`-8.682473334` on both, agreeing to `1.6e-12` — from a mid-chain bond of **22 multiplets,
62 dense states**, where U(1) holds 64 states. It gets there in five sweeps instead of
nine.

The MPO bond tells the other half of the story:

| | SU(2) | U(1) |
|---|---|---|
| one $\mathbf{S} \cdot \mathbf{S}$ term's bond | `{2: 1}` — 1 block, dense 3 | `{0: 1, ±2: 1}` — 3 blocks, dense 3 |
| bulk MPO bond | `{0: 2, 2: 1}` — 3 blocks, dense 5 | 5 blocks, dense 5 |

The operator's *dense* bond is 5 either way; SU(2) holds it in three blocks instead of
five. The compression is on the state side.

## `chi` bounds the dense dimension

This is the one convention to internalize before writing an SU(2) schedule. `max_bond`, and
therefore `Sweep(chi=...)`, bounds $\sum_c \operatorname{qdim}(c)\, m_c$ — the **dense**
dimension — not the multiplet count. Under U(1) and fermionic parity the two coincide; under
SU(2) they do not.

So `Sweep(chi=64)` on SU(2) legs keeps roughly a third the multiplets it keeps on U(1)
legs: the same state, held in fewer, larger irreps. An SU(2) ramp written in U(1) habits
is a tighter ramp than it looks.

The greedy walk also stops at the first multiplet that would overflow the budget rather
than scanning on for a cheaper one that still fits, which keeps the kept set nested as
`chi` grows and can undershoot by up to $\max_c \operatorname{qdim}(c) - 1$.
[`select_bond`][tenet.ops.linalg.select_bond] returns the decision — `dense_dim`,
`reduced_dim`, `undershoot`, `next_multiplet` — and is the answer to "why is my bond
smaller than I asked for". [Truncation](../guide/truncation.md) works that through.

## Reading a graded bond

```python
sectors = out.psi.schmidt_sectors()[9]     # bond 9's spectrum, split by sector
entropy = out.psi.entanglement_entropy()   # {bond: S}, in nats
```

[`schmidt_sectors`][tenet.network.MPS.schmidt_sectors] is the read a graded bond is for.
On an SU(2) bond a single $j$ multiplet stands for $2j + 1$ dense Schmidt values, and the
entropy accounts for that — which is why a two-site singlet reports $\log 2$ under SU(2)
and under U(1) alike, while $-\sum_i p_i \log p_i$ over the flattened SU(2) spectrum would
report 0.

## Where next

- [Truncation](../guide/truncation.md) — `max_bond` on a multiplet bond.
- [Symmetries and providers](../guide/symmetries-and-providers.md) — SU(N), products, and
  the capability protocols.
- [DMRG](../guide/dmrg.md) — schedules, noise and measurement.
