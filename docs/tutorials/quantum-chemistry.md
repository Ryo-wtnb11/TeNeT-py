# Quantum chemistry — an FCIDUMP through `from_arrays`

An ab initio Hamiltonian is $O(K^4)$ terms over two operator patterns. That shape is what
[`MPO.from_arrays`][tenet.network.MPO.from_arrays] takes, and `symbolic=True` is what
keeps it tractable once the MPO bond runs into the thousands.

$$
H = \sum_{pq,\sigma} h_{pq}\, c^{\dagger}_{p\sigma} c_{q\sigma}
  + \tfrac{1}{2} \sum_{pqrs,\sigma\tau} (pq \vert rs)\,
    c^{\dagger}_{p\sigma} c^{\dagger}_{r\tau} c_{s\tau} c_{q\sigma}
$$

The blocks on this page are illustrative rather than doctests: they need an integral file
you supply, and a real one takes minutes to sweep. The other tutorials
([DMRG](dmrg.md), [fermions](fermions.md), [SU(2)](su2.md)) run as written.

## Reading the file

FCIDUMP is a Fortran namelist header ending in `&END` (or a bare `/`), then `value i j k l`
records with 1-based orbital indices.

```python
import pathlib

def read_fcidump(path):
    text = pathlib.Path(path).read_text()
    head, sep, body = text.partition("&END")
    if not sep:
        head, _, body = text.partition("/\n")
```

Split the file at the header terminator. The two spellings are both in the wild, so try
`&END` first and fall back to a line that is exactly `/`; everything after the separator
is records.

```python
    flat = head.upper().replace(" ", "")
    norb = int(flat.split("NORB=", 1)[1].split(",")[0])
    nelec = int(flat.split("NELEC=", 1)[1].split(",")[0])
```

The header is a namelist, so whitespace is not significant: flatten it, then read the two
fields that matter. `norb` counts *spatial* orbitals — the site count is twice that.
`nelec` is carried along for the filling you expect, not for anything the sweep enforces.

```python
    recs = []
    for line in body.splitlines():
        f = line.split()
        if len(f) == 5:
            recs.append((float(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4])))
    return norb, nelec, recs
```

Each record is a value and four indices. `k == l == 0` marks a one-body element and
`i == j == k == l == 0` the core energy; the loop keeps the raw tuples and sorts that out
next. The core energy is a constant shift, not a term — add it to the converged energy at
the end.

## Expanding to spin orbitals

Each spatial orbital becomes two sites, $2p$ and $2p + 1$. The integral file stores one
representative per permutational orbit, so you expand the orbit yourself: the eight images
of $(pq \vert rs)$ are eight different operator strings, and folding them into one coefficient
would build a different operator. `from_arrays` merges what coincides after the expansion.

```python
import numpy as np

EIGHTFOLD = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
             (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0))
```

The eight index permutations of real-orbital eightfold symmetry: swap within the first
pair, within the second, and the two pairs with each other. Each row is a *reordering* of
$(p, q, r, s)$, applied below by gathering.

```python
def spin_orbital_blocks(recs, screen=1e-12):
    h, v = {}, {}
    for val, i, j, k, m in recs:
        if abs(val) <= screen:
            continue
        if k == 0 and m == 0:
            if i == 0:
                continue                       # the core energy
            h[(i - 1, j - 1)] = h[(j - 1, i - 1)] = val
```

One-body records: drop to 0-based indices and store both $(p,q)$ and $(q,p)$, because a
real one-body matrix is symmetric and the file gives one triangle. The `i == 0` guard
skips the core-energy record, which reached here with all four indices zero.

```python
        else:
            q = (i - 1, j - 1, k - 1, m - 1)
            for perm in EIGHTFOLD:
                v[tuple(q[x] for x in perm)] = val
```

Two-body records: write the value under all eight images. A `dict` is the merge — an orbit
with repeated indices has fewer than eight distinct images, and assigning the same value
twice is a no-op rather than a double count.

```python
    one_idx, one_val, two_idx, two_val = [], [], [], []
    for (p, q), c in h.items():
        for s in (0, 1):
            one_idx.append((2 * p + s, 2 * q + s))
            one_val.append(c)
```

Spin expansion of the one-body part: each spatial pair becomes two rows, spin up and spin
down, at the same coefficient. Site $2p + s$ is spatial orbital $p$ with spin $s$, so
$\sigma$ is conserved by construction — the two operators of a row carry the same `s`.

```python
    for (p, q, r, s), c in v.items():
        for sa in (0, 1):
            for sb in (0, 1):
                two_idx.append((2 * p + sa, 2 * r + sb, 2 * s + sb, 2 * q + sa))
                two_val.append(0.5 * c)
```

Spin expansion of the two-body part: four spin combinations per spatial orbit, and the
$\tfrac{1}{2}$ of the Hamiltonian above folded into the coefficient. **The site order in
the row is the operator order in the pattern**, $c^{\dagger}_p c^{\dagger}_r c_s c_q$, not
the chemists' index order $(pq \vert rs)$ — which is why the tuple reads `p, r, s, q`.

```python
    return [("c+ c", np.array(one_idx), np.array(one_val)),
            ("c+ c+ c c", np.array(two_idx), np.array(two_val))]
```

Two blocks, and each is three parallel arrays: the operator pattern as a string, an
integer `(T, L)` array of sites, and the length-`T` coefficient array whose dtype decides
the MPO's.

## Building the operator

```python
from tenet.models import spinless_fermion
from tenet.network import MPO

norb, nelec, recs = read_fcidump("H4.STO6G.R1.8.FCIDUMP")
site = spinless_fermion()                    # one spin orbital per site: {|0>, |1>} on fZ2
n_sites = 2 * norb
```

The site is *spinless* even though the molecule is not: spin is carried by the chain
layout, one spin orbital per site, so each site is just occupied or empty. That is the
$d = 2$ `fZ2` site, and `n_sites` is twice the spatial orbital count.

```python
h = MPO.from_arrays(n_sites, site.ops, spin_orbital_blocks(recs), symbolic=True)
```

Three things happen inside, all whole-array work:

- each row is sorted into site order by a stable argsort, paying the Koszul sign of every
  inversion of two sign-braiding operators — so the fermionic sign is the builder's job,
  not yours;
- operators coinciding on a site are **pre-multiplied** into one on-site operator, and a
  term whose on-site product vanishes (`c c` on one site) is dropped. Every ab initio term
  with a repeated spin-orbital index needs this, and `from_arrays` is where it happens;
- terms agreeing on `(operator labels, sites)` are fused and their coefficients summed,
  then `screen=` drops what is below its magnitude. At its default, `1e-12`, that removes
  the symmetry-forbidden $\sim 10^{-15}$ entries a real integral file carries and nothing else.
  Raising it to `1e-4` and above is an accuracy-for-size trade you take deliberately.

`cutoff=` controls the compressing SVD sweeps, unchanged from
[`from_terms`][tenet.network.MPO.from_terms]; `cutoff=None` skips them and gives the exact
finite-state-machine bond.

## `symbolic=True` is the point

Without it, the builder hands back the site tensors and
[`Env.heff2`][tenet.network.Env.heff2] contracts them. That is right for a lattice model
whose MPO bond is eight wide, and wrong here.

With it, the finite-state-machine description is kept, and `heff2` runs the prepared
matvec instead: complementary operators assembled per bond, the sum dispatched term family
by term family. For $O(K^4)$ terms over a bond in the thousands that is the route that
fits in memory at all.

Nothing dispatches at run time and no threshold is probed. You state it at build time,
exactly as `cutoff` is stated, and `symbolic` and `cutoff` are independent.
[`materialize()`][tenet.network.MPO.materialize] moves an operator built `symbolic=True`
onto the site-tensor path afterwards.

## Sweeping

`MPS.product` cannot seed this: `spinless_fermion`'s two sectors have degeneracy 1, so it
would work for that site, but a spin-orbital chain wants a bond profile anyway. Seed with
[`MPS.random`][tenet.network.MPS.random] over even-parity `D=1` boundaries.

```python
from tenet import GradedSpace
from tenet.network import MPS, Sweep, dmrg_
from tenet.symmetry import FZ2Sector, fZ2

vac = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
mid = GradedSpace.new(fZ2, {FZ2Sector(0): 32, FZ2Sector(1): 32})
psi = MPS.random(site.phys, [vac] + [mid] * (n_sites - 1) + [vac], seed=0)
```

`vac` is the `D=1` even boundary; `mid` is the starting shape of every internal bond, even
and odd sectors given equal room because a molecular ground state has no reason to prefer
either. The sweep re-decides all of them.

```python
out = dmrg_(psi, h, schedule=[Sweep(500, noise=1e-4)] * 6
                             + [Sweep(1000, noise=1e-5)] * 6
                             + [Sweep(1000)], max_sweeps=40)
energy = out.energy + core_energy
```

A ramp: six sweeps at `chi=500` with noise, six at `chi=1000` with less, then a clean tail
at `chi=1000` that repeats until convergence or `max_sweeps`. The core energy is the
constant the file carried and the operator does not, so it is added once at the end.

**Use noise.** On a four-spin-orbital test case built this way, a plain
`dmrg_(psi, h, chi=64)` from a product seed stops at `-1.83` while the exact ground energy
of the same operator in that parity sector is `-2.3845221144`; a ramp with wavefunction
noise reaches the exact value. A structurally allowed coupled sector the eigensolver left
numerically empty is dropped by the truncation, and a sector that is zero stays zero.
Wavefunction noise fills every allowed sector of the two-site map, which is exactly that
minimum. Taper it to zero: convergence is never declared on a noisy sweep.

`fZ2` grades by parity, so the sweep conserves the parity your boundary legs seeded and
not the electron count. Read the filling back with `expectation_profile` on `n`.

## Checking

```python
h.variance(out.psi)        # <psi|H^2|psi> / <psi|psi> - E^2
```

Run it at two bond dimensions: a state converging on an eigenstate has a variance falling
towards zero as `chi` grows, and one plateaued on the wrong bond structure has one that
does not. This is the check that does not need the answer in advance. On a small enough
case, `h.materialize().to_dense()` and `numpy.linalg.eigvalsh` give the exact answer to
compare against directly.

## Where next

- [Building a Hamiltonian](../guide/hamiltonians.md) — the four builders, side by side.
- [Fermions and the Hubbard model](fermions.md) — the same sites, a lattice Hamiltonian,
  and a sweep small enough to run as a doctest.
- [DMRG](../guide/dmrg.md) — schedules, noise, the extrapolation recipe.
