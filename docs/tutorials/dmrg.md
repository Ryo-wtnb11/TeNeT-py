# DMRG end to end — a Heisenberg chain

A spin-1/2 Heisenberg chain, $H = \sum_i \mathbf{S}_i \cdot \mathbf{S}_{i+1}$, open boundaries,
U(1)-graded by $2S^z$. This is
[`examples/heisenberg.py`](https://github.com/Ryo-wtnb11/symtenet/blob/main/examples/heisenberg.py)
walked through; the file runs standalone and its output is committed on the [Heisenberg,
U(1)](../examples/heisenberg.md) page.

```sh
uv run python examples/heisenberg.py
```

Nothing outside the core install is involved. Every block below is a doctest: the
chain is $N = 20$ and the whole run takes a couple of seconds.

## The site

```python
>>> from tenet.models import spin_half
>>> SITE = spin_half()
>>> PHYS = SITE.phys
>>> PHYS.sectors                     # U(1), charge 2 S^z, the doublet {-1, +1}
((U1Sector(charge=-1), 1), (U1Sector(charge=1), 1))

```

The physical space is two sectors of degeneracy 1. The charge is $2S^z$ rather than
$S^z$, so every label stays an exact integer: down is $-1$, up is $+1$.

```python
>>> sorted(SITE.ops)
['S+', 'S-', 'Sz']
>>> SITE.ops["S+"].ndim
3

```

`SITE.ops` holds the rank-3 **term** operators: two physical legs plus a `D=1` leg
carrying the charge the operator emits. That third leg is what makes `S+` — which is not
invariant on its own — expressible as a term. `SITE.matrices` holds the dense forms
alongside, plus `S.S`, the invariant two-site operator, which has no rank-3 form.

## The Hamiltonian

```python
>>> from tenet.network import MPO
>>> n_sites = 20
>>> op_sz, op_sp, op_sm = (SITE.ops[name] for name in ("Sz", "S+", "S-"))
>>> terms = []
>>> for i in range(n_sites - 1):
...     terms.append((1.0, [(op_sz, i), (op_sz, i + 1)]))
...     terms.append((0.5, [(op_sp, i), (op_sm, i + 1)]))
...     terms.append((0.5, [(op_sm, i), (op_sp, i + 1)]))

```

Three terms per bond: the $S^zS^z$ diagonal piece, and the two halves of
$\tfrac{1}{2}(S^+_iS^-_{i+1} + S^-_iS^+_{i+1})$ that make up $S^x_iS^x_{i+1} +
S^y_iS^y_{i+1}$. Each entry is `(coefficient, [(operator, site), ...])`, and the identity
is implied on every site the list does not name.

```python
>>> h = MPO.from_terms(n_sites, terms)
>>> len(h)
20

```

No MPO bond space is written down. Each term is a bond-1 MPO, a direct sum stacks them,
and two compressing SVD sweeps collapse the stack to the operator Schmidt rank.

```python
>>> h[10].legs[0].space.sectors      # a bulk MPO bond
((U1Sector(charge=-2), 1), (U1Sector(charge=0), 3), (U1Sector(charge=2), 1))

```

The grading falls out of the operators' own charges: `S+` emits $+2$, `S-` emits $-2$,
`Sz` emits $0$, so the bulk bond comes out $\{-2: 1,\, 0: 3,\, +2: 1\}$ — five channels
in three blocks. You declared none of it.

If you want to see the same operator with its `W` matrix and grading written out by hand,
[`examples/heisenberg_walkthrough.py`](https://github.com/Ryo-wtnb11/symtenet/blob/main/examples/heisenberg_walkthrough.py)
builds it three ways — `from_w`, `from_entries`, `from_terms` — and lands all three on the
same twelve digits. Its page is
[Heisenberg, U(1) walkthrough](../examples/heisenberg-walkthrough.md).

## The seed fixes the sector

```python
>>> from tenet.network import MPS
>>> from tenet.symmetry import U1Sector
>>> psi = MPS.product(PHYS, [U1Sector(1 if n % 2 else -1) for n in range(n_sites)])
>>> psi[0].legs[0].space.sectors      # the left boundary leg
((U1Sector(charge=0), 1),)

```

A Néel product state: one physical sector per site, alternating down and up.
`MPS.product` derives the bonds backwards from those charges, so the running total lands
on bond 0, where it is printable — here charge 0, degeneracy 1.

Both boundary legs then carry the unit sector with degeneracy 1, and invariance of every
site tensor forces $\sum_i 2S^z_i = 0$ — the sector an even chain's ground state lives
in. Nothing in the sweep can leave it: the sector is held structurally, by the seed's
charges and the tensors' invariance, for the whole run.

The general recipe: a `D=1` boundary leg carrying `U1Sector(q)` targets
$S^z_{\mathrm{tot}} = q/2$.

## Sweeping

```python
>>> from tenet.network import Sweep, dmrg_
>>> schedule = [Sweep(16, noise=1e-4)] * 3 + [Sweep(32, noise=1e-5)] * 3 + [Sweep(64)]
>>> out = dmrg_(psi, h, schedule=schedule)
>>> out.sweeps
9
>>> round(out.energy, 12)
-8.682473334398

```

A ramp with noise that cools to a clean `chi=64` tail. The last entry of a schedule
repeats until convergence, so the three written sweeps at `chi=64` are however many the
tail needs — nine sweeps in total here. `-8.682473334398` is the recorded
exact-diagonalization energy of the same finite chain to twelve digits.

On this chain a flat `dmrg_(psi, h, chi=64)` reaches the same energy: a degeneracy-1 U(1)
seed already grows by a factor of $d$ per sweep. The ramp is here because writing one is
what you do on a problem where the seed cannot ramp itself — see
[DMRG](../guide/dmrg.md) for when noise pays.

Inside a sweep, two pieces do the work: `svd_truncated` decides the bond `GradedSpace` at
every bond and reports the discarded weight by Pythagoras; `lanczos` solves the two-site
eigenproblem over `SymmetricTensor` treated as a vector, needing only `add`, `subtract`,
scalar multiply, `norm` and an inner product — no `scipy.sparse.linalg`, and no dense
reshaping of the local problem.

## Reading the state

### Bond energies

The Hamiltonian is a sum of two-site terms, one per bond of the open chain:

$$
H = \sum_{n=1}^{N-1} h_{n,n+1},
\qquad h_{n,n+1} = \mathbf{S}_n \cdot \mathbf{S}_{n+1}.
$$

The **bond energy** is the expectation of one such term on the converged state,

$$
e_n = \langle \psi \vert h_{n,n+1} \vert \psi \rangle,
$$

which is a *measurement* on `out.psi`. It is a different object from the number `dmrg_`
reports: `out.energy` is the effective eigenvalue the two-site eigensolver returned at
the orthogonality centre of the final sweep, $E = \langle \psi \vert H \vert \psi
\rangle$. By linearity the two are tied, $\sum_n e_n = E$, and that identity is the
cross-check below.

The canonical form of the state enters only as the *evaluation method*, not as part of
the definition: with the orthogonality centre placed on bond $n$, every environment
outside the bond is an identity, so $e_n$ is a contraction two sites wide instead of a
pass over the whole chain.

```python
>>> from tenet.network import expectation_2site, local_op
>>> ss = local_op(SITE.matrices["S.S"], phys=PHYS)
>>> profile = [expectation_2site(out.psi, ss, n) for n in range(n_sites - 1)]
>>> round(sum(profile), 12) == round(out.energy, 12)
True

```

`local_op` with no `charge` takes an invariant *k*-site operator — here the $4 \times 4$
$\mathbf{S}\cdot\mathbf{S}$ matrix — so `ss` is rank 4, not rank 3. The 19 bond energies
sum to `out.energy` to twelve digits: `expectation_2site` and `dmrg_` weigh the same
state on the same scale.

The profile itself is not flat. Open boundaries pin the chain, so $e_n$ alternates —
a dimerized pattern, strongest at the two edges and decaying inward towards the uniform
bulk value, which in the thermodynamic limit is $1/4 - \ln 2 \approx -0.4431$ per bond.
On this $N = 20$ chain the outermost bonds read `-0.653` and `-0.294` while the
mid-chain pair has closed to `-0.524` and `-0.364`.

```python
>>> from tenet.network import expectation_1site
>>> op_sz = local_op(SITE.matrices["Sz"], phys=PHYS)
>>> max_sz = max(abs(expectation_1site(out.psi, op_sz, n)) for n in range(n_sites))
>>> max_sz < 1e-11
True

```

$\max_n \lvert\langle S^z_n\rangle\rvert$ comes out at float noise, `~5e-13`. That is not
a converged number that happens to be small: the seed's own charges put the state in the
$S^z_{\mathrm{tot}} = 0$ sector, and a uniform chain in that sector has no site
magnetization to find.

For a whole profile in one pass, use
[`expectation_profile`][tenet.network.expectation_profile] instead of the comprehension:
same numbers, $O(N)$ rather than $O(N^2)$, because it moves the orthogonality centre once
along the chain rather than running two full transfer passes per site.
[DMRG](../guide/dmrg.md) has the full measurement set, the entanglement readers and the
variance check.

## Why there is no `jit` and no `grad` here

DMRG's control flow is data-dependent at every level: the truncation re-decides the bond
space each sweep, `lanczos`'s happy breakdown tests a norm against a tolerance, and the
loop exits on a measured energy change. Each of those is what a traced region cannot do,
and `svd_truncated` says so by raising `StructureChangingError` under a trace. So this
runs on the eager NumPy backend.

One piece inside the sweep is traceable, and `dmrg_` takes a callable for it:
`compile=jax.jit` wraps the two-site matvec. [JAX and backends](../guide/jax-and-backends.md)
and [DMRG](../guide/dmrg.md#compile) describe what that changes.

## Where next

- [Fermions and the Hubbard model](fermions.md) — the same driver with `fZ2` sites.
- [SU(2)](su2.md) — the same chain with the non-Abelian grading.
- [Quantum chemistry](quantum-chemistry.md) — `from_arrays(..., symbolic=True)`.
- [`examples/toy_codes/dmrg.py`](https://github.com/Ryo-wtnb11/symtenet/blob/main/examples/toy_codes/dmrg.py)
  writes the algorithm out by hand on `tenet`'s tensor layer; its page is
  [Toy DMRG](../examples/toy-dmrg.md).
