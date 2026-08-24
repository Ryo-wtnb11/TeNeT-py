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

## The site

```python
from tenet.models import spinful_fermion

site = spinful_fermion()
site.phys.sectors   # ((FZ2Sector(parity=0), 2), (FZ2Sector(parity=1), 2))
sorted(site.ops)    # c+_dn, c+_up, c_dn, c_up, n, n_dn, n_up, 'n_up n_dn'
```

The basis is $d = 4$, ordered
$(\lvert 0\rangle, \lvert ud\rangle, \lvert u\rangle, \lvert d\rangle)$ — the even sector
first, because a dense array over a `GradedSpace` is laid out sector by sector. Get the
ordering wrong and `from_dense` will refuse your matrix, which is how you find out.

[`spinless_fermion`][tenet.models.spinless_fermion] is the $d = 2$ site,
$\{\lvert0\rangle, \lvert1\rangle\}$, with `c`, `c+` and `n`.

## The Hamiltonian

```python
from tenet.network import MPO

n, t, u = 8, 1.0, 4.0
fwd = [(i, i + 1) for i in range(n - 1)]
bwd = [(i + 1, i) for i in range(n - 1)]

blocks = []
for flavour in ("up", "dn"):
    expr = f"c+_{flavour} c_{flavour}"
    blocks += [(expr, fwd, [-t] * (n - 1)), (expr, bwd, [-t] * (n - 1))]
blocks.append(("n_up n_dn", [(i, i) for i in range(n)], [u] * n))

h = MPO.from_arrays(n, site.ops, blocks)
```

Four hopping blocks — two flavours, each in both directions — and one interaction block.
`from_arrays` takes each pattern once and does its work in NumPy over the whole index
array.

The interaction block names **two operators on two coincident site indices**.
`from_arrays` pre-multiplies them into one on-site operator before the walk, so you do not
have to. `from_terms` places one operator per site and needs the product handed to it, and
the site ships it under the same key, `site.ops["n_up n_dn"]`.

The hopping sign: each row is sorted into site order by a stable argsort that pays the
Koszul sign of every inversion of two sign-braiding operators, so `("c+_up c_up", (i+1, i))`
and its transpose agree as operators by construction. Writing the backward hop out
explicitly is what makes the Hamiltonian Hermitian; the sign of the braid is not yours to
supply.

## The seed

`MPS.product` refuses here, and the refusal is exact:

```
sector FZ2Sector(parity=1) has degeneracy 2 in the physical space, and a product
state names a basis vector: this constructor has no slot for the degeneracy index
```

A single sector labels two basis states on this site, so a sector list does not name a
product state. Seed with [`MPS.random`][tenet.network.MPS.random] and a bond profile
instead, with `D=1` even-parity boundary legs:

```python
from tenet import GradedSpace
from tenet.network import MPS
from tenet.symmetry import FZ2Sector, fZ2

vac = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
mid = GradedSpace.new(fZ2, {FZ2Sector(0): 4, FZ2Sector(1): 4})
psi = MPS.random(site.phys, [vac] + [mid] * (n - 1) + [vac], seed=0)
```

The boundary legs put the run in the even-parity sector. **`fZ2` grades by parity, not by
particle number**: the sweep conserves the parity you seeded and nothing finer, so if you
want a particular filling you read it back off `n` rather than declaring it.

## Sweeping

```python
from tenet.network import Sweep, dmrg_

out = dmrg_(psi, h, schedule=[Sweep(64, noise=1e-3)] * 5 + [Sweep(64)], max_sweeps=30)
```

Noise matters here in a way it does not on the Heisenberg chain. A random `fZ2` seed can
leave a structurally allowed coupled sector numerically empty, the truncation then drops
that sector from the bond, and a sector that is zero stays zero. Wavefunction noise fills
every allowed sector of the two-site map, which is exactly the local minimum to escape.
The schedule cools to zero noise, because convergence is never declared on a noisy sweep.

At $n = 4$, $t = 1$, $U = 4$ this lands on `-2.624942271511`, the exact ground energy of
the same operator, and `h.variance(out.psi)` is zero to machine precision.

## Reading it back

```python
from tenet.network import correlation_function, expectation_profile, local_op

n_op = local_op(site.matrices["n"], phys=site.phys)
density = expectation_profile(out.psi, n_op)          # <n_i>, one pass over the chain

green = correlation_function(
    out.psi, site.ops["c+_up"], site.ops["c_up"],
    pairs=[(0, j) for j in range(1, n)],
)
```

[`correlation_function`][tenet.network.correlation_function] takes the rank-3 charged
operators, which is the form a fermionic `c` has, and returns `{(i, j): value}` for
`i < j`. The Jordan-Wigner string across the sites between `i` and `j` is the same `fZ2`
braiding the term builder inserts, so a fermionic correlator at a distance is correct
rather than refused. It costs one MPO build and one pass per pair, so ask for the row or
the distance you want rather than all $N^2$.

## Where next

- [Building a Hamiltonian](../guide/hamiltonians.md) — the four builders and `symbolic=`.
- [Quantum chemistry](quantum-chemistry.md) — the same `fZ2` sites, $O(K^4)$ terms.
- [DMRG](../guide/dmrg.md) — noise, schedules and measurement in full.
