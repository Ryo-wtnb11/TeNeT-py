# Building a Hamiltonian

A Hamiltonian in TeNeT-py is an [MPO][tenet.network.MPO]. You build one from local
operators, and `tenet.models` ships the standard sites so you do not write the same three
matrices again.

## Sites

A [Site][tenet.models.Site] carries three things:

| field | what it is |
|---|---|
| `phys` | the physical `GradedSpace`, ready for `MPS.product` / `MPS.random` |
| `ops` | name to **term** operator, the rank-3 charge-leg form built by `local_op` |
| `matrices` | the dense matrix behind each, for forms the term API does not build |

```python
>>> from tenet.models import spin_half
>>> site = spin_half()
>>> site.phys.sectors            # U(1), the charge is 2 S^z
((U1Sector(charge=-1), 1), (U1Sector(charge=1), 1))
>>> sorted(site.ops)
['S+', 'S-', 'Sz']

```

The shipped sites:

| call | grading | `ops` |
|---|---|---|
| `spin_half()` | `U1`, charge $2S^z$ | `Sz`, `S+`, `S-` (`S.S` as a matrix) |
| `spin_half(SU2)` | `SU2`, one $j = 1/2$ multiplet | `S.S` |
| `spinless_fermion()` | `fZ2` | `c`, `c+`, `n` |
| `spinful_fermion()` | `fZ2`, `d=4` | `c_up`, `c+_up`, `c_dn`, `c+_dn`, `n_up`, `n_dn`, `n`, `n_up n_dn` |
| `hard_core_boson()` | `U1`, the occupation | `b`, `b+`, `n` |
| `hard_core_boson(Trivial)` | ungraded | `b`, `b+`, `n` |

Any other provider is refused by name. `tenet.network` never imports `tenet.models`: the
driver layer decides nothing about what your operators mean, and this layer is a
convenience above it. The library ships *sites*, which are a finite set; the lattice and
the Hamiltonian stay yours.

Each entry of `ops` is rank 3: the two physical legs plus a `D=1` leg carrying the charge
the operator emits. That third leg is what makes `S+` expressible as a term operator.
Write your own with [local_op][tenet.network.local_op], which takes any dense matrix:

```python
>>> import numpy as np
>>> from tenet.network import local_op
>>> from tenet.symmetry import U1Sector
>>> sp = local_op(np.array([[0.0, 0.0], [1.0, 0.0]]), phys=site.phys, charge=U1Sector(-2))
>>> sp.ndim
3

```

With no `charge`, `local_op` takes an invariant *k*-site operator instead — a
`(d**k, d**k)` or `(d,)*2k` array, the layout `np.kron` produces.

## The four builders

| you have | use |
|---|---|
| a list of terms | [`MPO.from_terms`][tenet.network.MPO.from_terms] |
| many terms over a few patterns | [`MPO.from_arrays`][tenet.network.MPO.from_arrays] |
| a `W` you are writing by hand | [`MPO.from_entries`][tenet.network.MPO.from_entries] |
| a `W` that is already a dense array | [`MPO.from_w`][tenet.network.MPO.from_w] |

The first three **derive** the MPO bond spaces from the operators' own charges. You
declare no grading and no bond width.

### `from_terms` — a list of terms

`(coefficient, [(operator, site), ...])` tuples, identity implied on every untouched
site:

```python
>>> from tenet.network import MPO
>>> n = 8
>>> terms = []
>>> for i in range(n - 1):
...     terms.append((1.0, [(site.ops["Sz"], i), (site.ops["Sz"], i + 1)]))
...     terms.append((0.5, [(site.ops["S+"], i), (site.ops["S-"], i + 1)]))
...     terms.append((0.5, [(site.ops["S-"], i), (site.ops["S+"], i + 1)]))
>>> h = MPO.from_terms(n, terms)
>>> len(h)
8

```

Every term is a bond-1 MPO, `direct_sum` stacks them, and two compressing SVD sweeps
collapse the stack to the operator Schmidt rank. `cutoff=` sets those sweeps' threshold;
`cutoff=None` skips the compression and gives the exact finite-state-machine bond.

A term operator may be an invariant *k*-site operator, in which case its sites are given
as a tuple: `(1.0, [(ss, (i, i + 1))])`. `from_terms` splits it with an SVD, and the
graded MPO bond comes out of that split.

**Refusal to know:** two operators of one term on one site. Multiply them into a single
on-site operator first.

### `from_arrays` — patterns and index arrays

One `(expr, indices, data)` triple per operator pattern. `expr` names the operators,
`indices` is a `(T, L)` integer array of sites, `data` is the length-`T` coefficient
array:

```python
>>> bonds = [(i, i + 1) for i in range(n - 1)]
>>> h2 = MPO.from_arrays(n, site.ops, [
...     ("Sz Sz", bonds, [1.0] * (n - 1)),
...     ("S+ S-", bonds, [0.5] * (n - 1)),
...     ("S- S+", bonds, [0.5] * (n - 1)),
... ])
>>> len(h2)
8

```

The pattern's work is done once per block in NumPy instead of once per term in Python,
which is what a Hamiltonian with $O(K^4)$ terms over a handful of patterns needs. Before
the walk, three whole-array steps run: each row is sorted into site order paying the
Koszul sign of every inversion of two sign-braiding operators; operators coinciding on a
site are pre-multiplied into one on-site operator, and a term whose on-site product
vanishes is dropped; terms agreeing on `(operator labels, sites)` are fused and their
coefficients summed. `screen=` then drops merged terms below a magnitude.

Permutational symmetry is expanded by you and merged here: the eight images of
$(ij \vert kl)$ are eight different operator strings.

Every operator here is rank 3 — a block gives one site index per name, so an invariant
*k*-site operator has nowhere to put its extra indices and is refused, naming
`from_terms`.

### `from_entries` — a `W` by hand

The non-zero `(i, j)` entries of each site's `W`, one mapping per site:

```python
>>> sz, sp2, sm = site.ops["Sz"], site.ops["S+"], site.ops["S-"]
>>> w = {                      # the Heisenberg W, its eight non-zero channels
...     (0, 0): None,          # I -- the term has not started
...     (0, 1): (0.5, sm), (1, -1): sp2,
...     (0, 2): (0.5, sp2), (2, -1): sm,
...     (0, 3): sz,        (3, -1): sz,
...     (-1, -1): None,        # I -- the term is finished
... }
>>> h3 = MPO.from_entries([w] * n)
>>> len(h3)
8

```

`0` is a bond's `IdL` channel and `-1` its `IdR` channel, at every bond, the way a
lower-triangular `W` is printed; the open channels are `1, 2, ...`. An entry is `None`
(the identity, a spectator ride on `(i, i)`), a number (that multiple of it), an
operator, or the pair `(coefficient, operator)`. `-1` names the last index, so no bond
width is declared; the charge is already on `local_op`'s third leg, so each channel's
`GradedSpace` is derived and the bond at a cut is the direct sum over its channels. The
two boundary bonds are `D = 1`, so bond 0 keeps only `IdL` and the last bond only `IdR`,
and the same bulk mapping serves the first and last site.

### `from_w` — a dense `W`

[`MPO.from_w`][tenet.network.MPO.from_w] takes one dense bulk `W` array plus the physical
space, the MPO bond space and the boundary space. Use it when the `W` arrives as a dense
array out of a paper or another library: its entries are numbers, no charge can be
recovered from them, so you supply the bond grading yourself. A wrong grading makes the
build *raise*, and that refusal is the proof the grading is right.

## `symbolic=True`

`from_terms`, `from_arrays` and `from_entries` each take `symbolic=`. It decides which
engine [`Env.heff2`][tenet.network.Env.heff2] runs.

At the default, the builder hands back the **site tensors**, and `heff2` contracts them.
That is what a finite-range lattice model wants: its MPO bond is five or eight wide, and
the site-tensor contraction is the cheapest thing that can happen to it.

`symbolic=True` keeps the **finite-state-machine description** the terms were assembled
into, and `heff2` runs the prepared matvec: complementary operators assembled per bond,
the sum dispatched term family by term family. That is what a Hamiltonian with a bond in
the thousands needs — in practice, quantum chemistry, where it is the route that fits in
memory at all.

```python
h = MPO.from_terms(n, terms, symbolic=True)
```

Nothing is decided at run time and no threshold is probed. The representation the
operator is in when you hand it to `dmrg_` *is* the choice, made at build time exactly as
`cutoff` is. The two are independent: `cutoff=None` with no `symbolic` gives exact,
uncompressed site tensors.

[`MPO.materialize()`][tenet.network.MPO.materialize] takes an operator built
`symbolic=True` to the site tensors — the same operator, the description dropped:

```python
h.materialize()
```

## Fermions

The spinful site is the $d = 4$ basis $(\lvert0\rangle, \lvert{\uparrow\downarrow}\rangle,
\lvert{\uparrow}\rangle, \lvert{\downarrow}\rangle)$, the even sector first, because a dense
array over a `GradedSpace` is laid out sector by sector. There is **no Jordan-Wigner operator
to place**: the string is the `fZ2` braiding an odd MPO bond pays when it crosses a physical
line. A Hubbard chain is its terms:

```python
>>> from tenet.models import spinful_fermion
>>> fsite, m, u = spinful_fermion(), 6, 4.0
>>> fwd = [(i, i + 1) for i in range(m - 1)]
>>> bwd = [(i + 1, i) for i in range(m - 1)]
>>> blocks = []
>>> for flavour in ("up", "dn"):
...     expr = "c+_" + flavour + " c_" + flavour
...     blocks += [(expr, fwd, [-1.0] * (m - 1)), (expr, bwd, [-1.0] * (m - 1))]
>>> blocks.append(("n_up n_dn", [(i, i) for i in range(m)], [u] * m))
>>> hub = MPO.from_arrays(m, fsite.ops, blocks)
>>> len(hub)
6

```

The last block names two operators on two *coincident* site indices, and `from_arrays`
multiplies them into one on-site operator before the walk. The site also ships that
product under the same key, `fsite.ops["n_up n_dn"]`, for `from_terms`, which places one
operator per site and so needs it pre-multiplied.

## SU(2)

```python
>>> from tenet.symmetry import SU2
>>> su2_site = spin_half(SU2)
>>> sorted(su2_site.ops)
['S.S']
>>> su2_site.ops["S.S"].ndim
4

```

`S+` is not in the table because **there is no such SU(2) operator**. The rank-3
charge-leg form puts the emitted sector on a `D=1` leg, and the only leg a spin-1 tensor
operator could emit onto is the $j = 1$ multiplet, whose dense dimension is 3. What SU(2)
has is the invariant two-site operator, and `S.S` is one whole Heisenberg bond term:

```python
>>> su2_terms = [(1.0, [(su2_site.ops["S.S"], (i, i + 1))]) for i in range(n - 1)]
>>> hsu2 = MPO.from_terms(n, su2_terms)
>>> len(hsu2)
8

```

`from_terms` splits it with an SVD and the graded MPO bond comes out of that split — the
coupling lives inside the operator's own blocks, so no coupling tree is named.
`from_arrays` cannot express this term, which is why the SU(2) site's `ops` is a
`from_terms` table.

## Applying and checking an operator

```python
>>> from tenet.network import MPS
>>> psi = MPS.product(site.phys, [U1Sector(1 if i % 2 else -1) for i in range(n)])
>>> phi = h.apply(psi)           # H|psi>, exact and untruncated
>>> len(phi)
8

```

[`apply`][tenet.network.MPO.apply] takes no `chi` and no `cutoff`: the product is exact,
and truncation is [`MPS.compress_`][tenet.network.MPS.compress_], by name. The cost of
the exact product is the operator's bond dimension times the state's, so compress
promptly on a wide operator.

[`variance`][tenet.network.MPO.variance] is $\langle\psi\vert H^2
\vert\psi\rangle/\langle\psi\vert\psi\rangle - E^2$, zero for an exact eigenstate — see
[DMRG](dmrg.md) for what to do with it. [`MPO.identity`][tenet.network.MPO.identity] builds the
`D=1` identity operator, and [`to_dense`][tenet.network.MPO.to_dense] expands the whole $d^N
\times d^N$ operator for a small chain.

## Where next

- [DMRG](dmrg.md) — sweeping the operator you just built.
- [Fermions and the Hubbard model](../tutorials/fermions.md) and
  [Quantum chemistry](../tutorials/quantum-chemistry.md) — the two builders at scale.
- [`tenet.models`](../api/models.md) and [`tenet.network`](../api/network.md) — the
  reference.
