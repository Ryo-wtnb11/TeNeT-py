# Fermions — a Hubbard chain

The Hubbard model on `fZ2` sites:

$$
H = -t \sum_{i,\sigma} \left( c^{\dagger}_{i\sigma} c_{i+1,\sigma} + \text{h.c.} \right)
  + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

The thing to notice is what you do **not** write: there is no Jordan-Wigner operator
anywhere. The string is the `fZ2` braiding that an odd MPO bond pays when it crosses a
physical line, so the antisymmetry is carried by the symmetry itself. A Hubbard chain is
its terms.

Every block below is a doctest. The chain is four sites at $t = 1$, $U = 4$, small
enough that the whole page runs in under a second and that the converged energy can be
checked against a dense diagonalization.

## The site

```python
>>> from tenet.models import spinful_fermion
>>> site = spinful_fermion()
>>> site.phys.sectors
((FZ2Sector(parity=0), 2), (FZ2Sector(parity=1), 2))

```

The basis is $d = 4$, two states of even fermion parity and two of odd. It is ordered
$(\lvert 0\rangle, \lvert ud\rangle, \lvert u\rangle, \lvert d\rangle)$ — the even sector
first, because a dense array over a `GradedSpace` is laid out sector by sector, in the
order `sectors` prints. Get the ordering wrong and `from_dense` will refuse your matrix,
which is how you find out.

```python
>>> sorted(site.ops)
['c+_dn', 'c+_up', 'c_dn', 'c_up', 'n', 'n_dn', 'n_up', 'n_up n_dn']

```

Eight named term operators: the four creation and annihilation operators, the three
number operators, and the double occupancy `n_up n_dn` — the on-site product, shipped
ready-made because `from_terms` places one operator per site and cannot multiply two
together for you.

[`spinless_fermion`][tenet.models.spinless_fermion] is the $d = 2$ site,
$\{\lvert0\rangle, \lvert1\rangle\}$, with `c`, `c+` and `n`.

## The Hamiltonian

```python
>>> n, t, u = 4, 1.0, 4.0
>>> fwd = [(i, i + 1) for i in range(n - 1)]
>>> bwd = [(i + 1, i) for i in range(n - 1)]

```

`fwd` and `bwd` are the index arrays of the hopping: every nearest-neighbour pair, and
the same pairs with the two sites swapped. A row of such an array is one term's site
list, one entry per operator named in the pattern.

```python
>>> blocks = []
>>> for flavour in ("up", "dn"):
...     expr = f"c+_{flavour} c_{flavour}"
...     blocks += [(expr, fwd, [-t] * (n - 1)), (expr, bwd, [-t] * (n - 1))]
>>> blocks.append(("n_up n_dn", [(i, i) for i in range(n)], [u] * n))
>>> [(expr, len(idx)) for expr, idx, _ in blocks]
[('c+_up c_up', 3), ('c+_up c_up', 3), ('c+_dn c_dn', 3), ('c+_dn c_dn', 3), ('n_up n_dn', 4)]

```

Five `(expr, indices, data)` triples: four hopping blocks — two flavours, each in both
directions — and one interaction block. `expr` names the operators in order, `indices`
gives one site per name, and `data` is the coefficient of each row. The pattern's work is
done once per block in NumPy over the whole index array rather than once per term in
Python.

The interaction block names **two operators on two coincident site indices**, `(i, i)`.
`from_arrays` pre-multiplies them into one on-site operator before the walk, so you do
not have to. `from_terms` would need `site.ops["n_up n_dn"]` handed to it instead.

Writing the backward hop out explicitly is what makes the Hamiltonian Hermitian, and the
sign of the braid is not yours to supply: each row is sorted into site order by a stable
argsort that pays the Koszul sign of every inversion of two sign-braiding operators, so
`("c+_up c_up", (i+1, i))` and its transpose agree as operators by construction.

```python
>>> from tenet.network import MPO
>>> h = MPO.from_arrays(n, site.ops, blocks)
>>> h[1].legs[0].space.sectors
((FZ2Sector(parity=0), 2), (FZ2Sector(parity=1), 4))

```

No MPO bond space was declared. The bulk bond comes out with four odd channels — one per
hopping operator still in flight — and two even ones, and the odd half is exactly where
the Jordan-Wigner string lives: an odd bond crossing a physical line braids, and that
braid *is* the string.

## The seed

`MPS.product` refuses here, and the refusal is exact:

```python
>>> from tenet.network import MPS
>>> from tenet.symmetry import FZ2Sector
>>> MPS.product(site.phys, [FZ2Sector(0)] * n)
Traceback (most recent call last):
    ...
ValueError: sector FZ2Sector(parity=0) has degeneracy 2 in the physical space, and a product state names a basis vector: this constructor has no slot for the degeneracy index

```

A single sector labels two basis states on this site — even parity is $\lvert 0\rangle$
*and* $\lvert ud\rangle$ — so a sector list does not name a product state. Seed with
[`MPS.random`][tenet.network.MPS.random] and a bond profile instead.

```python
>>> from tenet import GradedSpace
>>> from tenet.symmetry import fZ2
>>> vac = GradedSpace.new(fZ2, {FZ2Sector(0): 1})              # D=1, even
>>> mid = GradedSpace.new(fZ2, {FZ2Sector(0): 4, FZ2Sector(1): 4})
>>> psi = MPS.random(site.phys, [vac] + [mid] * (n - 1) + [vac], seed=0)
>>> len(psi)
4

```

The bond list has $N + 1$ entries, one per bond including the two boundaries. The `D=1`
even-parity boundary legs put the run in the even sector and hold it there: every site
tensor is invariant, so the parity of the whole chain cannot change.

**`fZ2` grades by parity, not by particle number.** The sweep conserves the parity you
seeded and nothing finer, so a particular filling is something you read back off `n`
rather than something you declare.

## Sweeping

```python
>>> from tenet.network import Sweep, dmrg_
>>> out = dmrg_(psi, h, schedule=[Sweep(64, noise=1e-3)] * 5 + [Sweep(64)], max_sweeps=30)
>>> out.sweeps
17
>>> round(out.energy, 12)
-2.624942271511

```

Noise matters here in a way it does not on the Heisenberg chain. A random `fZ2` seed can
leave a structurally allowed coupled sector numerically empty, the truncation then drops
that sector from the bond, and a sector that is zero stays zero. Wavefunction noise fills
every allowed sector of the two-site map, which is exactly the local minimum to escape.
The schedule cools to zero noise, because convergence is never declared on a noisy sweep.

```python
>>> import numpy as np
>>> exact = np.linalg.eigvalsh(h.to_dense().reshape(4**n, 4**n))[0]
>>> round(float(exact), 12)
-2.624942271511
>>> round(abs(h.variance(out.psi)), 12)
0.0

```

`to_dense` expands the whole $4^4 \times 4^4$ operator, which is only affordable because
the chain is four sites; `eigvalsh` then gives the exact ground energy of the *same*
operator, and the sweep matches it to twelve digits. The variance
$\langle H^2\rangle - E^2$ is zero to machine precision, which is the check that does not
depend on knowing the answer in advance.

## Reading it back

```python
>>> from tenet.network import expectation_profile, local_op
>>> n_op = local_op(site.matrices["n"], phys=site.phys)
>>> [round(v, 6) for v in expectation_profile(out.psi, n_op)]
[0.394321, 0.605679, 0.605679, 0.394321]

```

`local_op` with no `charge` wraps the dense $4 \times 4$ number matrix as a rank-2
invariant one-site operator — no charge leg, because `n` emits none — and [`expectation_profile`][tenet.network.expectation_profile] reads
it at every site in one pass over the chain. The filling comes out at 2 electrons on
4 sites, pushed towards the middle by the open boundaries — a number that was never
declared, because `fZ2` conserves only parity.

```python
>>> from tenet.network import correlation_function
>>> green = correlation_function(
...     out.psi, site.ops["c+_up"], site.ops["c_up"], pairs=[(0, j) for j in range(1, n)]
... )
>>> {pair: round(v, 6) for pair, v in green.items()}
{(0, 1): 0.234527, (0, 2): 0.197586, (0, 3): 0.157279}

```

[`correlation_function`][tenet.network.correlation_function] takes the rank-3 charged
operators, which is the form a fermionic `c` has, and returns `{(i, j): value}` for
`i < j`. The Jordan-Wigner string across the sites between `i` and `j` is the same `fZ2`
braiding the term builder inserts, so a fermionic correlator at a distance is correct
rather than refused. It costs one MPO build and one pass per pair, so ask for the row or
the distance you want — the `pairs=` above is one row — rather than all $N^2$.

## Where next

- [Building a Hamiltonian](../guide/hamiltonians.md) — the four builders and `symbolic=`.
- [Quantum chemistry](quantum-chemistry.md) — the same `fZ2` sites, $O(K^4)$ terms.
- [DMRG](../guide/dmrg.md) — noise, schedules and measurement in full.
