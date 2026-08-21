# Building a Hamiltonian from a site

A Hamiltonian in TeNeT-py is a **term list**, and a term is built from local operators.
Writing those operators out by hand is possible — [`local_op`](../api/network.md) takes
any dense matrix — but it is not interesting, and every spin-1/2 chain in the world
starts from the same three matrices. `tenet.models` ships them.

```python
from tenet.models import spin_half

site = spin_half()
site.phys          # the physical GradedSpace: U(1) charge 2 S^z, doublet {-1, +1}
sorted(site.ops)   # ['S+', 'S-', 'Sz']
```

A [`Site`][tenet.models.Site] carries three things and nothing else:

| field | what it is |
|---|---|
| `phys` | the physical `GradedSpace`, ready for `MPS.product` / `MPS.random` |
| `ops` | name → **term** operator, built by `local_op` |
| `matrices` | the dense matrix behind each, for the forms the term API does not build |

`ops` is exactly the shape [`MPO.from_arrays`](../api/network.md) calls `ops`, so a
Hamiltonian is one call away.

## A spin model, end to end

```python
from tenet.models import spin_half
from tenet.network import MPO, MPS, dmrg_
from tenet.symmetry import U1Sector

site, n = spin_half(), 20
bond = [(i, i + 1) for i in range(n - 1)]
h = MPO.from_arrays(
    n,
    site.ops,
    [
        ("Sz Sz", bond, [1.0] * (n - 1)),
        ("S+ S-", bond, [0.5] * (n - 1)),
        ("S- S+", bond, [0.5] * (n - 1)),
    ],
)
psi = MPS.product(site.phys, [U1Sector(1 if i % 2 else -1) for i in range(n)])
out = dmrg_(psi, h, chi=64)
```

A block's first entry is an *expression*: the operator names, whitespace-separated, one
site index per name. That is why the names are the field's own symbols — `"Sz Sz"` is
the term, written the way it is written on paper. The same list goes to
[`MPO.from_terms`](../api/network.md) as `(coefficient, [(operator, site), ...])` pairs
if you would rather build terms than arrays; the two front ends produce the identical
machine.

To measure a bond energy afterwards you want the invariant two-site operator, which has
no rank-3 form and therefore lives in `matrices`:

```python
from tenet.network import expectation_2site, local_op

ss = local_op(site.matrices["S.S"], phys=site.phys)
[expectation_2site(out.psi, ss, i) for i in range(n - 1)]
```

## A fermionic model, end to end

The spinful site is the `d = 4` basis `(|0>, |ud>, |u>, |d>)` — the even sector first,
because a dense array over a `GradedSpace` is laid out sector by sector. There is **no
Jordan-Wigner operator** to place: the string is the `fZ2` braiding an odd MPO bond pays
when it crosses a physical line, so a Hubbard chain is just its terms.

```python
from tenet.models import spinful_fermion
from tenet.network import MPO

site, n, u = spinful_fermion(), 8, 4.0
fwd = [(m, m + 1) for m in range(n - 1)]
bwd = [(m + 1, m) for m in range(n - 1)]
blocks = []
for flavour in ("up", "dn"):
    expr = f"c+_{flavour} c_{flavour}"
    blocks += [(expr, fwd, [-1.0] * (n - 1)), (expr, bwd, [-1.0] * (n - 1))]
blocks.append(("n_up n_dn", [(m, m) for m in range(n)], [u] * n))
h = MPO.from_arrays(n, site.ops, blocks)
```

The last block is worth a look: `"n_up n_dn"` names two operators on two *coincident*
site indices, and `from_arrays` multiplies them into one on-site operator before the
walk. The site also ships that product under the same key, `site.ops["n_up n_dn"]`, for
`from_terms` — which places one operator per site and so needs it pre-multiplied.

## Under SU(2) the set is `{S.S}`

```python
from tenet.models import spin_half
from tenet.symmetry import SU2

sorted(spin_half(SU2).ops)     # ['S.S']
spin_half(SU2).ops["S.S"].ndim  # 4
```

`S+` is not there because **there is no such SU(2) operator**, not because it was left
out. The rank-3 charge-leg form puts the emitted sector on a `D=1` leg; the only leg a
spin-1 tensor operator could emit onto is the `j=1` multiplet, whose dense dimension is
3, so `local_op(sz, phys=phys, charge=SU2Sector(2))` raises on the shape. What SU(2) has
instead is the **invariant two-site operator**, and `S.S` is one whole Heisenberg bond
term:

```python
from tenet.network import MPO

site = spin_half(SU2)
h = MPO.from_terms(n, [(1.0, [(site.ops["S.S"], (i, i + 1))]) for i in range(n - 1)])
```

`MPO.from_terms` splits it with an SVD and the graded MPO bond comes out of that SVD —
the coupling lives inside the operator's own blocks, so no coupling tree is asked for.
`MPO.from_arrays` cannot express this term at all (a block gives one site index per
name), which is why the SU(2) site's `ops` is a `from_terms` table and nothing else.

## What is not here

No lattice geometry, no `heisenberg(L)` that hands back an MPO, no parameter sweeps.
The set of standard *sites* is finite and a model zoo is not; the Hamiltonian stays
yours. `tenet.network` never imports `tenet.models` either — the driver layer decides
nothing about what your operators mean, and this layer is an optional convenience above
it.

## The shipped sites

| call | grading | `ops` |
|---|---|---|
| `spin_half()` | `U1`, charge `2 S^z` | `Sz`, `S+`, `S-` (`S.S` as a matrix) |
| `spin_half(SU2)` | `SU2`, one `j=1/2` multiplet | `S.S` |
| `spinless_fermion()` | `fZ2` | `c`, `c+`, `n` |
| `spinful_fermion()` | `fZ2`, `d=4` | `c_up`, `c+_up`, `c_dn`, `c+_dn`, `n_up`, `n_dn`, `n`, `n_up n_dn` |
| `hard_core_boson()` | `U1`, the occupation | `b`, `b+`, `n` |
| `hard_core_boson(Trivial)` | ungraded | `b`, `b+`, `n` |

Any other provider is refused by name. The full reference is
[`tenet.models`](../api/models.md).
